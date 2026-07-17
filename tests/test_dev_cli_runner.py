import io
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import services.dev_cli_runner as dcr


_ECHO_STDIN_SCRIPT = """\
import sys
data = sys.stdin.buffer.read()
sys.stdout.buffer.write(b"GOT:" + data)
sys.stdout.flush()
"""

_BIG_OUTPUT_SCRIPT = """\
import sys
sys.stdout.buffer.write(b"x" * (6 * 1024 * 1024))
sys.stdout.flush()
"""

_SLEEP_SCRIPT = """\
import sys
import time
time.sleep(float(sys.argv[1]))
print("done")
"""

_NONZERO_EXIT_SCRIPT = """\
import sys
sys.stderr.write("boom")
sys.exit(3)
"""

# Never reads stdin at all - used to prove the stdin writer thread doesn't
# block run_cli()'s own timeout/cancel handling even when nothing drains
# the pipe, and that closing stdin on cancel/timeout unblocks a writer
# stuck on a full, unread pipe.
_IGNORES_STDIN_AND_SLEEPS_SCRIPT = """\
import sys
import time
time.sleep(float(sys.argv[1]))
"""

_ECHO_ENV_VAR_SCRIPT = """\
import os
import sys
sys.stdout.buffer.write(os.environ.get(sys.argv[1], "<MISSING>").encode("utf-8"))
sys.stdout.flush()
"""

# Spawns a grandchild process that keeps appending to a heartbeat file until
# killed (or ~30s pass) - used to prove that killing the tree also reaches
# processes the immediate child itself spawned, not just the immediate child.
_TREE_SCRIPT = """\
import os
import subprocess
import sys
import time

heartbeat_path = sys.argv[1]
child_code = (
    "import sys, time\\n"
    "path = sys.argv[1]\\n"
    "for _ in range(300):\\n"
    "    with open(path, 'a') as f:\\n"
    "        f.write('x')\\n"
    "    time.sleep(0.1)\\n"
)
kwargs = {}
if os.name == "nt":
    kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
else:
    kwargs["start_new_session"] = True
subprocess.Popen([sys.executable, "-c", child_code, heartbeat_path], **kwargs)
time.sleep(30)
"""


class _FakePopen:
    """A stand-in for subprocess.Popen simulating a truly hung process:
    every .wait() call raises TimeoutExpired until .kill() is called (which
    run_cli()'s timeout/cancel handling always does, regardless of whether
    the platform tree-kill mechanism itself reports success), after which
    .wait() succeeds - letting _kill_process_tree()'s behavior be exercised
    deterministically without a real hung process. Unlike
    tests.test_dev_git_runner._FakePopen (which unconditionally succeeds on
    the second .wait() call - correct for dev_git_runner's single
    fixed-timeout .wait()), this run_cli() implementation polls .wait() in
    a loop, so "succeeds on the Nth call" would let it exit before the
    timeout is ever actually observed.
    """

    def __init__(self, *args, **kwargs):
        self.pid = 999999
        self.stdout = io.BytesIO(b"")
        self.stderr = io.BytesIO(b"")
        self.stdin = io.BytesIO()
        self._killed = False

    def wait(self, timeout=None):
        if self._killed:
            return 0
        raise subprocess.TimeoutExpired(cmd=["cli"], timeout=timeout)

    def kill(self):
        self._killed = True


class _ImmediateExitPopen:
    """A stand-in that exits immediately with returncode 0 - used for tests
    that only need to inspect how Popen was invoked (argv/shell/etc.),
    where a real subprocess.run()-based taskkill call must never be
    triggered (unlike _FakePopen, which is deliberately built to hang).
    """

    def __init__(self, *args, **kwargs):
        self.pid = 111
        self.stdout = io.BytesIO(b"")
        self.stderr = io.BytesIO(b"")
        self.stdin = io.BytesIO()

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class _NeverEOFStream:
    def read(self, size=-1):
        time.sleep(5)
        return b""

    def close(self):
        pass


class _StuckReaderPopen:
    def __init__(self, *args, **kwargs):
        self.pid = 4242424
        self.stdout = _NeverEOFStream()
        self.stderr = io.BytesIO(b"")
        self.stdin = io.BytesIO()

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class DevCliRunnerTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="dev_cli_runner_test ")
        self.tmp_path = Path(self._tmp_dir.name)
        self.cwd = self.tmp_path / "cwd"
        self.cwd.mkdir()

        self._echo_script = self._write_script("echo_stdin.py", _ECHO_STDIN_SCRIPT)
        self._big_output_script = self._write_script("big_output.py", _BIG_OUTPUT_SCRIPT)
        self._sleep_script = self._write_script("sleep.py", _SLEEP_SCRIPT)
        self._nonzero_script = self._write_script("nonzero.py", _NONZERO_EXIT_SCRIPT)
        self._tree_script = self._write_script("tree.py", _TREE_SCRIPT)
        self._ignores_stdin_script = self._write_script(
            "ignores_stdin.py", _IGNORES_STDIN_AND_SLEEPS_SCRIPT
        )
        self._echo_env_var_script = self._write_script("echo_env_var.py", _ECHO_ENV_VAR_SCRIPT)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def _write_script(self, name: str, content: str) -> Path:
        path = self.tmp_path / name
        path.write_text(content, encoding="utf-8")
        return path

    # --- basic success + stdin passing (never shell-expanded) ---

    def test_run_cli_success_returns_stdout(self):
        result = dcr.run_cli(
            [sys.executable, str(self._echo_script)],
            cwd=self.cwd,
            timeout=10,
            input_bytes=b"hello",
            max_output_bytes=1_000_000,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"GOT:hello")
        self.assertFalse(result.stdout_truncated)

    def test_run_cli_stdin_is_never_shell_expanded(self):
        dangerous_input = b"$(echo pwned); `echo also_pwned`; & calc.exe & rm -rf /"
        result = dcr.run_cli(
            [sys.executable, str(self._echo_script)],
            cwd=self.cwd,
            timeout=10,
            input_bytes=dangerous_input,
            max_output_bytes=1_000_000,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, b"GOT:" + dangerous_input)

    def test_run_cli_uses_fixed_argv_never_shell(self):
        with mock.patch(
            "services.dev_cli_runner.subprocess.Popen", side_effect=_ImmediateExitPopen
        ) as popen_mock:
            dcr.run_cli(
                [sys.executable, str(self._sleep_script), "0"],
                cwd=self.cwd,
                timeout=10,
                max_output_bytes=1000,
            )
        _, kwargs = popen_mock.call_args
        self.assertFalse(kwargs.get("shell", True))
        self.assertEqual(popen_mock.call_args.args[0], [sys.executable, str(self._sleep_script), "0"])

    # --- non-zero exit ---

    def test_run_cli_nonzero_exit_is_command_failed(self):
        result = dcr.run_cli(
            [sys.executable, str(self._nonzero_script)],
            cwd=self.cwd,
            timeout=10,
            max_output_bytes=1_000_000,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.returncode, 3)
        self.assertEqual(result.error_code, dcr.ERROR_COMMAND_FAILED)

    # --- output capping: never unbounded, no deadlock ---

    def test_run_cli_caps_large_output_without_deadlock(self):
        result = dcr.run_cli(
            [sys.executable, str(self._big_output_script)],
            cwd=self.cwd,
            timeout=30,
            max_output_bytes=1000,
        )
        self.assertTrue(result.success)
        self.assertTrue(result.stdout_truncated)
        self.assertEqual(len(result.stdout), 1000)

    def test_capped_reader_truncates_and_drains(self):
        data = b"y" * 1000
        reader = dcr._CappedReader(io.BytesIO(data), cap=100)
        reader.start()
        reader.join(timeout=5)
        result, truncated = reader.result()
        self.assertEqual(len(result), 100)
        self.assertTrue(truncated)

    def test_capped_reader_no_truncation_for_small_input(self):
        reader = dcr._CappedReader(io.BytesIO(b"hello"), cap=100)
        reader.start()
        reader.join(timeout=5)
        result, truncated = reader.result()
        self.assertEqual(result, b"hello")
        self.assertFalse(truncated)

    # --- timeout ---

    def test_run_cli_timeout_reports_timed_out(self):
        result = dcr.run_cli(
            [sys.executable, str(self._sleep_script), "5"],
            cwd=self.cwd,
            timeout=0.3,
            max_output_bytes=1000,
        )
        self.assertFalse(result.success)
        self.assertTrue(result.timed_out)
        self.assertFalse(result.cancelled)
        self.assertEqual(result.error_code, dcr.ERROR_TIMEOUT)
        self.assertGreaterEqual(result.duration_seconds, 0)

    def test_run_cli_timeout_kills_process_tree_and_confirms_termination_mocked(self):
        with mock.patch("services.dev_cli_runner.subprocess.Popen", side_effect=_FakePopen) as popen_mock, \
                mock.patch("services.dev_cli_runner.subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
            result = dcr.run_cli(
                [sys.executable, str(self._sleep_script), "0"], cwd=self.cwd, timeout=0.01, max_output_bytes=1000
            )

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, dcr.ERROR_TIMEOUT)
        self.assertTrue(popen_mock.called)

        if os.name == "nt":
            self.assertTrue(run_mock.called)
            taskkill_args = run_mock.call_args.args[0]
            self.assertEqual(taskkill_args[0], "taskkill")
            self.assertIn("/T", taskkill_args)
            self.assertIn("/F", taskkill_args)
            self.assertTrue(result.termination_confirmed)
            self.assertFalse(result.child_processes_may_remain)

    def test_run_cli_timeout_reports_unconfirmed_termination_when_taskkill_fails(self):
        if os.name != "nt":
            self.skipTest("taskkill behavior is Windows-specific")
        with mock.patch("services.dev_cli_runner.subprocess.Popen", side_effect=_FakePopen), \
                mock.patch("services.dev_cli_runner.subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout=b"", stderr=b"access denied"
            )
            result = dcr.run_cli(
                [sys.executable, str(self._sleep_script), "0"], cwd=self.cwd, timeout=0.01, max_output_bytes=1000
            )

        self.assertEqual(result.error_code, dcr.ERROR_TIMEOUT)
        self.assertFalse(result.termination_confirmed)
        self.assertTrue(result.child_processes_may_remain)

    # --- cancellation (cooperative, via CancellationToken) ---

    def test_run_cli_cancel_stops_a_real_sleeping_process(self):
        token = dcr.CancellationToken()

        def _cancel_soon():
            time.sleep(0.3)
            token.cancel()

        threading.Thread(target=_cancel_soon, daemon=True).start()
        result = dcr.run_cli(
            [sys.executable, str(self._sleep_script), "10"],
            cwd=self.cwd,
            timeout=30,
            max_output_bytes=1000,
            cancel_token=token,
        )
        self.assertFalse(result.success)
        self.assertTrue(result.cancelled)
        self.assertFalse(result.timed_out)
        self.assertEqual(result.error_code, dcr.ERROR_CANCELLED)
        # Cancellation must be prompt, not "wait out the 30s timeout".
        self.assertLess(result.duration_seconds, 5)

    def test_cancellation_token_is_idempotent(self):
        token = dcr.CancellationToken()
        self.assertFalse(token.is_cancelled())
        token.cancel()
        token.cancel()
        self.assertTrue(token.is_cancelled())

    # --- process tree kill actually reaches a grandchild process ---

    def test_run_cli_timeout_kills_entire_process_tree_including_grandchild(self):
        heartbeat_path = self.tmp_path / "heartbeat.txt"
        result = dcr.run_cli(
            [sys.executable, str(self._tree_script), str(heartbeat_path)],
            cwd=self.cwd,
            timeout=1.0,
            max_output_bytes=1000,
        )
        self.assertFalse(result.success)
        self.assertTrue(result.timed_out)

        # Give the grandchild a moment - if it were still alive it would
        # keep appending; if the tree was actually killed, size stops
        # growing.
        self.assertTrue(heartbeat_path.exists(), "grandchild never got a chance to start")
        size_after_kill = heartbeat_path.stat().st_size
        time.sleep(0.5)
        size_later = heartbeat_path.stat().st_size
        self.assertEqual(size_after_kill, size_later, "grandchild process kept running after tree kill")

    # --- reader thread residual detection ---

    def test_run_cli_reports_failure_when_reader_thread_never_terminates(self):
        with mock.patch("services.dev_cli_runner.subprocess.Popen", side_effect=_StuckReaderPopen), \
                mock.patch("services.dev_cli_runner._READER_JOIN_TIMEOUT_SECONDS", 0.1):
            result = dcr.run_cli(
                [sys.executable, str(self._sleep_script), "0"], cwd=self.cwd, timeout=10, max_output_bytes=1000
            )
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, dcr.ERROR_OUTPUT_READER_STUCK)
        self.assertFalse(result.reader_threads_terminated)

    # --- cwd / cli missing ---

    def test_run_cli_missing_cwd(self):
        missing_cwd = self.tmp_path / "does-not-exist-at-all"
        result = dcr.run_cli([sys.executable, str(self._sleep_script), "0"], cwd=missing_cwd, timeout=10, max_output_bytes=1000)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, dcr.ERROR_CWD_NOT_FOUND)

    def test_run_cli_missing_executable(self):
        missing_exe = self.tmp_path / "does-not-exist.exe"
        result = dcr.run_cli([str(missing_exe)], cwd=self.cwd, timeout=10, max_output_bytes=1000)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, dcr.ERROR_CLI_NOT_FOUND)

    def test_run_cli_distinguishes_cwd_disappearing_after_precheck_from_missing_cli(self):
        vanished_cwd = self.tmp_path / "will-vanish"
        vanished_cwd.mkdir()
        vanished_cwd.rmdir()

        with mock.patch("services.dev_cli_runner._cwd_exists_now", return_value=True):
            result = dcr.run_cli(
                [sys.executable, str(self._sleep_script), "0"], cwd=vanished_cwd, timeout=10, max_output_bytes=1000
            )
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, dcr.ERROR_CWD_NOT_FOUND)

    def test_classify_launch_exception_cwd_winerror(self):
        exc = NotADirectoryError(20, "dir not found", None, 267, None)
        self.assertEqual(dcr._classify_launch_exception(exc, self.cwd), dcr.ERROR_CWD_NOT_FOUND)

    def test_classify_launch_exception_executable_winerror(self):
        exc = FileNotFoundError(2, "file not found", None, 2, None)
        self.assertEqual(dcr._classify_launch_exception(exc, self.cwd), dcr.ERROR_CLI_NOT_FOUND)

    def test_run_cli_launch_failure(self):
        with mock.patch("services.dev_cli_runner.subprocess.Popen", side_effect=OSError("boom")):
            result = dcr.run_cli(
                [sys.executable, str(self._sleep_script), "0"], cwd=self.cwd, timeout=10, max_output_bytes=1000
            )
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, dcr.ERROR_LAUNCH_FAILED)

    # --- stdin write failure ---

    def test_run_cli_stdin_write_failure(self):
        class _BrokenStdinPopen:
            def __init__(self, *args, **kwargs):
                self.pid = 123
                self.stdout = io.BytesIO(b"")
                self.stderr = io.BytesIO(b"")

                class _Stdin:
                    def write(self, data):
                        raise OSError("broken pipe")

                    def close(self):
                        pass

                self.stdin = _Stdin()

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        with mock.patch("services.dev_cli_runner.subprocess.Popen", side_effect=_BrokenStdinPopen):
            result = dcr.run_cli(
                [sys.executable, str(self._sleep_script), "0"],
                cwd=self.cwd,
                timeout=10,
                input_bytes=b"hello",
                max_output_bytes=1000,
            )
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, dcr.ERROR_STDIN_WRITE_FAILED)

    # --- env allowlist: never ambient os.environ unless explicitly asked ---

    def test_run_cli_passes_only_the_given_env(self):
        marker_name = "DEV_CLI_RUNNER_TEST_SECRET_MARKER"
        with mock.patch.dict(os.environ, {marker_name: "should-not-leak"}):
            result = dcr.run_cli(
                [sys.executable, str(self._echo_env_var_script), marker_name],
                cwd=self.cwd,
                timeout=10,
                max_output_bytes=1000,
                env={"PATH": os.environ.get("PATH", "")},
            )
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, b"<MISSING>")

    def test_run_cli_env_none_inherits_ambient_environment(self):
        marker_name = "DEV_CLI_RUNNER_TEST_VISIBLE_MARKER"
        with mock.patch.dict(os.environ, {marker_name: "visible-value"}):
            result = dcr.run_cli(
                [sys.executable, str(self._echo_env_var_script), marker_name],
                cwd=self.cwd,
                timeout=10,
                max_output_bytes=1000,
            )
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, b"visible-value")

    def test_run_cli_custom_env_reaches_the_child(self):
        result = dcr.run_cli(
            [sys.executable, str(self._echo_env_var_script), "DEV_CLI_RUNNER_CUSTOM_VAR"],
            cwd=self.cwd,
            timeout=10,
            max_output_bytes=1000,
            env={"PATH": os.environ.get("PATH", ""), "DEV_CLI_RUNNER_CUSTOM_VAR": "hello-from-allowlist"},
        )
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, b"hello-from-allowlist")

    # --- stdin writer thread: never blocks the wait loop, never leaks ---

    def test_run_cli_large_stdin_to_a_cli_that_never_reads_it_does_not_hang(self):
        large_input = b"y" * (5 * 1024 * 1024)
        result = dcr.run_cli(
            [sys.executable, str(self._ignores_stdin_script), "0.2"],
            cwd=self.cwd,
            timeout=10,
            input_bytes=large_input,
            max_output_bytes=1000,
        )
        # The process exits on its own after ~0.2s without ever reading
        # stdin - since 5 MiB exceeds the OS pipe buffer, the writer
        # thread's write() then fails with a broken pipe once the process
        # is gone (a real, reported failure - ERROR_STDIN_WRITE_FAILED,
        # not success). The property this test actually proves is that
        # run_cli() returns promptly at all (nowhere near the 10s timeout)
        # and the writer thread is cleanly joined, not left running -
        # i.e. the writer thread never blocked the main wait loop.
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, dcr.ERROR_STDIN_WRITE_FAILED)
        self.assertTrue(result.stdin_writer_terminated)
        self.assertLess(result.duration_seconds, 5)

    def test_run_cli_cancel_closes_stdin_to_unblock_writer(self):
        large_input = b"z" * (5 * 1024 * 1024)
        token = dcr.CancellationToken()

        def _cancel_soon():
            time.sleep(0.3)
            token.cancel()

        threading.Thread(target=_cancel_soon, daemon=True).start()
        result = dcr.run_cli(
            [sys.executable, str(self._ignores_stdin_script), "30"],
            cwd=self.cwd,
            timeout=30,
            input_bytes=large_input,
            max_output_bytes=1000,
            cancel_token=token,
        )
        self.assertFalse(result.success)
        self.assertTrue(result.cancelled)
        self.assertTrue(result.stdin_writer_terminated)
        self.assertLess(result.duration_seconds, 5)

    def test_run_cli_reports_stuck_stdin_writer(self):
        class _NeverDrainedStdin:
            def write(self, data):
                time.sleep(5)
                return len(data)

            def close(self):
                pass

        class _StuckStdinPopen:
            def __init__(self, *args, **kwargs):
                self.pid = 555555
                self.stdout = io.BytesIO(b"")
                self.stderr = io.BytesIO(b"")
                self.stdin = _NeverDrainedStdin()

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        with mock.patch("services.dev_cli_runner.subprocess.Popen", side_effect=_StuckStdinPopen), mock.patch(
            "services.dev_cli_runner._WRITER_JOIN_TIMEOUT_SECONDS", 0.1
        ):
            result = dcr.run_cli(
                [sys.executable, str(self._sleep_script), "0"],
                cwd=self.cwd,
                timeout=10,
                input_bytes=b"hello",
                max_output_bytes=1000,
            )
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, dcr.ERROR_STDIN_WRITER_STUCK)
        self.assertFalse(result.stdin_writer_terminated)


if __name__ == "__main__":
    unittest.main()
