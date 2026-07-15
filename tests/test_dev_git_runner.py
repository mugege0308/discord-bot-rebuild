import io
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import services.dev_git_runner as dgr


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True, encoding="utf-8"
    )


class _FakePopen:
    """A stand-in for subprocess.Popen that times out on the first .wait()
    call and succeeds on the second, so _kill_process_tree()'s behavior can
    be exercised without actually running a hung process. stdout/stderr are
    empty real streams so the reader threads have something valid to read
    from (and hit EOF immediately).
    """

    def __init__(self, *args, **kwargs):
        self.pid = 999999
        self._wait_calls = 0
        self.stdout = io.BytesIO(b"")
        self.stderr = io.BytesIO(b"")

    def wait(self, timeout=None):
        self._wait_calls += 1
        if self._wait_calls == 1:
            raise subprocess.TimeoutExpired(cmd=["git"], timeout=timeout)
        return 0

    def kill(self):
        pass


class _NeverEOFStream:
    """A stream whose read() blocks well past any reasonable join timeout
    and never returns EOF (and whose close() does not unblock it either) -
    simulating a reader thread that cannot be made to terminate no matter
    what run_git() tries.
    """

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

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class DevGitRunnerTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="dev_git_runner_test ")
        self.repo_root = Path(self._tmp_dir.name) / "repo"
        self.repo_root.mkdir(parents=True)
        _git(["init", "-q", "-b", "main"], self.repo_root)
        _git(["config", "user.name", "Test User"], self.repo_root)
        _git(["config", "user.email", "test@example.com"], self.repo_root)
        (self.repo_root / "README.md").write_text("hello\n", encoding="utf-8")
        _git(["add", "README.md"], self.repo_root)
        _git(["commit", "-q", "-m", "init"], self.repo_root)

    def tearDown(self):
        self._tmp_dir.cleanup()

    # --- basic success path against a real repo ---

    def test_run_git_success(self):
        result = dgr.run_git(["status"], cwd=self.repo_root, timeout=10)
        self.assertTrue(result.success)
        self.assertEqual(result.returncode, 0)
        self.assertIsInstance(result.stdout, bytes)
        self.assertIsInstance(result.stderr, bytes)

    def test_run_git_nonzero_exit_is_command_failed(self):
        result = dgr.run_git(
            ["show-ref", "--verify", "--quiet", "refs/heads/does-not-exist"], cwd=self.repo_root, timeout=10
        )
        self.assertFalse(result.success)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.error_code, dgr.ERROR_GIT_COMMAND_FAILED)

    # --- cwd-missing vs git-missing must never be conflated ---

    def test_run_git_missing_cwd_is_distinct_from_missing_git(self):
        missing_cwd = self.repo_root / "does-not-exist-at-all"
        result = dgr.run_git(["status"], cwd=missing_cwd, timeout=10)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, dgr.ERROR_CWD_NOT_FOUND)

    def test_run_git_missing_git_executable(self):
        with mock.patch("services.dev_git_runner.subprocess.Popen", side_effect=FileNotFoundError(2, "not found", None, 2, None)):
            result = dgr.run_git(["status"], cwd=self.repo_root, timeout=10)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, dgr.ERROR_GIT_NOT_FOUND)

    def test_run_git_missing_git_executable_without_winerror_hint(self):
        # No winerror/filename hint at all (e.g. a plain POSIX-style
        # FileNotFoundError) - the fallback re-check of cwd (which still
        # exists here) must still classify this as GIT_NOT_FOUND.
        with mock.patch("services.dev_git_runner.subprocess.Popen", side_effect=FileNotFoundError()):
            result = dgr.run_git(["status"], cwd=self.repo_root, timeout=10)
        self.assertEqual(result.error_code, dgr.ERROR_GIT_NOT_FOUND)

    def test_run_git_launch_failure(self):
        with mock.patch("services.dev_git_runner.subprocess.Popen", side_effect=OSError("boom")):
            result = dgr.run_git(["status"], cwd=self.repo_root, timeout=10)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, dgr.ERROR_GIT_LAUNCH_FAILED)

    # --- cwd disappearing between the pre-check and the real Popen() call ---

    def test_run_git_distinguishes_cwd_disappearing_after_precheck_from_missing_git(self):
        vanished_cwd = self.repo_root / "will-vanish"
        vanished_cwd.mkdir()
        vanished_cwd.rmdir()

        # Force the up-front pre-check to report "yes, it's there" (as if
        # it raced and won just before the directory disappeared), letting
        # the real subprocess.Popen() call hit the genuinely-missing
        # directory and raise the platform's real exception.
        with mock.patch("services.dev_git_runner._cwd_exists_now", return_value=True):
            result = dgr.run_git(["status"], cwd=vanished_cwd, timeout=10)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, dgr.ERROR_CWD_NOT_FOUND)

    def test_classify_launch_exception_cwd_winerror(self):
        exc = NotADirectoryError(20, "dir not found", None, 267, None)
        self.assertEqual(dgr._classify_launch_exception(exc, self.repo_root), dgr.ERROR_CWD_NOT_FOUND)

    def test_classify_launch_exception_executable_winerror(self):
        exc = FileNotFoundError(2, "file not found", None, 2, None)
        self.assertEqual(dgr._classify_launch_exception(exc, self.repo_root), dgr.ERROR_GIT_NOT_FOUND)

    def test_classify_launch_exception_falls_back_to_cwd_recheck(self):
        missing_cwd = self.repo_root / "does-not-exist-at-all"
        exc = FileNotFoundError()  # no winerror, no filename
        self.assertEqual(dgr._classify_launch_exception(exc, missing_cwd), dgr.ERROR_CWD_NOT_FOUND)
        self.assertEqual(dgr._classify_launch_exception(exc, self.repo_root), dgr.ERROR_GIT_NOT_FOUND)

    # --- timeout kills the whole process tree, with confirmed termination ---

    def test_run_git_timeout_kills_process_tree_and_confirms_termination(self):
        with mock.patch("services.dev_git_runner.subprocess.Popen", side_effect=_FakePopen) as popen_mock, \
                mock.patch("services.dev_git_runner.subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
            result = dgr.run_git(["status"], cwd=self.repo_root, timeout=0.01)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, dgr.ERROR_GIT_TIMEOUT)
        self.assertTrue(popen_mock.called)

        if os.name == "nt":
            self.assertTrue(run_mock.called)
            taskkill_args = run_mock.call_args.args[0]
            self.assertEqual(taskkill_args[0], "taskkill")
            self.assertIn("/T", taskkill_args)
            self.assertIn("/F", taskkill_args)
            self.assertIn("/PID", taskkill_args)
            self.assertTrue(result.termination_confirmed)
            self.assertFalse(result.child_processes_may_remain)

    def test_run_git_timeout_reports_unconfirmed_termination_when_taskkill_fails(self):
        if os.name != "nt":
            self.skipTest("taskkill behavior is Windows-specific")
        with mock.patch("services.dev_git_runner.subprocess.Popen", side_effect=_FakePopen), \
                mock.patch("services.dev_git_runner.subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout=b"", stderr=b"access denied"
            )
            result = dgr.run_git(["status"], cwd=self.repo_root, timeout=0.01)

        self.assertEqual(result.error_code, dgr.ERROR_GIT_TIMEOUT)
        self.assertFalse(result.termination_confirmed)
        self.assertTrue(result.child_processes_may_remain)

    def test_run_git_timeout_reports_confirmed_termination_when_taskkill_succeeds(self):
        if os.name != "nt":
            self.skipTest("taskkill behavior is Windows-specific")
        with mock.patch("services.dev_git_runner.subprocess.Popen", side_effect=_FakePopen), \
                mock.patch("services.dev_git_runner.subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
            result = dgr.run_git(["status"], cwd=self.repo_root, timeout=0.01)

        self.assertTrue(result.termination_confirmed)
        self.assertFalse(result.child_processes_may_remain)

    def test_non_timeout_results_report_termination_confirmed_by_default(self):
        result = dgr.run_git(["status"], cwd=self.repo_root, timeout=10)
        self.assertTrue(result.termination_confirmed)
        self.assertFalse(result.child_processes_may_remain)

    def test_kill_process_tree_actually_terminates_a_real_process(self):
        # This exercises the REAL platform tree-kill mechanism (real
        # taskkill on Windows), not a mock - so it deliberately does NOT
        # assert a fixed confirmed/may_remain outcome: some environments
        # (restricted permissions, sandboxing, ...) can make taskkill
        # itself fail with access-denied even though the process is a
        # perfectly normal child this test just spawned, which would make
        # a fixed expectation flaky for reasons outside this code's
        # control. What must hold regardless: the two returned flags are
        # each other's logical complement, and the process actually dies
        # one way or another (via taskkill or the unconditional
        # popen.kill() fallback) - if it doesn't, proc.wait() below raises
        # and fails the test loudly. The specific success/failure paths of
        # taskkill itself are covered deterministically by the mocked
        # tests above.
        extra_kwargs = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt"
            else {"start_new_session": True}
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], **extra_kwargs
        )
        try:
            confirmed, may_remain = dgr._kill_process_tree(proc)
            self.assertEqual(confirmed, not may_remain)
            proc.wait(timeout=10)
            self.assertIsNotNone(proc.poll())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    # --- reader thread residual detection: never report success while a
    # reader thread might still be alive/mid-read ---

    def test_run_git_reports_failure_when_reader_thread_never_terminates(self):
        with mock.patch("services.dev_git_runner.subprocess.Popen", side_effect=_StuckReaderPopen), \
                mock.patch("services.dev_git_runner._READER_JOIN_TIMEOUT_SECONDS", 0.1):
            result = dgr.run_git(["status"], cwd=self.repo_root, timeout=10)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, dgr.ERROR_OUTPUT_READER_STUCK)
        self.assertFalse(result.reader_threads_terminated)

    def test_run_git_reader_threads_terminated_true_for_normal_command(self):
        result = dgr.run_git(["status"], cwd=self.repo_root, timeout=10)
        self.assertTrue(result.success)
        self.assertTrue(result.reader_threads_terminated)

    # --- output capping: never unbounded, no deadlock ---

    def test_capped_reader_truncates_and_drains(self):
        data = b"y" * 1000
        reader = dgr._CappedReader(io.BytesIO(data), cap=100)
        reader.start()
        reader.join(timeout=5)
        result, truncated = reader.result()
        self.assertEqual(len(result), 100)
        self.assertTrue(truncated)

    def test_capped_reader_no_truncation_for_small_input(self):
        reader = dgr._CappedReader(io.BytesIO(b"hello"), cap=100)
        reader.start()
        reader.join(timeout=5)
        result, truncated = reader.result()
        self.assertEqual(result, b"hello")
        self.assertFalse(truncated)

    def test_run_git_flags_truncated_for_real_output_over_1mib(self):
        big_content = "x" * (dgr._MAX_CAPTURED_BYTES + 1000)
        (self.repo_root / "big.txt").write_text(big_content, encoding="utf-8")
        _git(["add", "big.txt"], self.repo_root)
        _git(["commit", "-q", "-m", "big file"], self.repo_root)

        result = dgr.run_git(["show", "HEAD:big.txt"], cwd=self.repo_root, timeout=30)
        self.assertTrue(result.success)
        self.assertTrue(result.stdout_truncated)
        self.assertEqual(len(result.stdout), dgr._MAX_CAPTURED_BYTES)

    def test_run_git_does_not_deadlock_on_output_exceeding_pipe_buffer(self):
        # 5x a typical OS pipe buffer size - large enough that, without a
        # concurrently-draining reader thread, the child would block
        # writing to a full pipe while this process blocks in
        # popen.wait(), producing the classic subprocess.PIPE deadlock this
        # design exists to avoid.
        big_content = "z" * (5 * 1024 * 1024)
        (self.repo_root / "huge.txt").write_text(big_content, encoding="utf-8")
        _git(["add", "huge.txt"], self.repo_root)
        _git(["commit", "-q", "-m", "huge file"], self.repo_root)

        result = dgr.run_git(["show", "HEAD:huge.txt"], cwd=self.repo_root, timeout=30)
        self.assertTrue(result.success)
        self.assertTrue(result.stdout_truncated)
        self.assertEqual(len(result.stdout), dgr._MAX_CAPTURED_BYTES)

    # --- check_ref_exists: "not found" (exit 1) vs genuine failure ---

    def test_check_ref_exists_true_for_real_branch(self):
        result = dgr.check_ref_exists("refs/heads/main", cwd=self.repo_root, timeout=10)
        self.assertTrue(result.exists)
        self.assertIsNone(result.error_code)

    def test_check_ref_exists_false_for_missing_ref_is_not_an_error(self):
        result = dgr.check_ref_exists("refs/heads/does-not-exist", cwd=self.repo_root, timeout=10)
        self.assertFalse(result.exists)
        self.assertIsNone(result.error_code)

    def test_check_ref_exists_distinguishes_other_nonzero_from_not_found(self):
        with mock.patch(
            "services.dev_git_runner.run_git",
            return_value=dgr.GitRunResult(
                success=False,
                returncode=128,
                stdout=b"",
                stderr=b"fatal: not a git repository",
                stdout_truncated=False,
                stderr_truncated=False,
                error_code=dgr.ERROR_GIT_COMMAND_FAILED,
            ),
        ):
            result = dgr.check_ref_exists("refs/heads/main", cwd=self.repo_root, timeout=10)
        self.assertIsNone(result.exists)
        self.assertEqual(result.error_code, dgr.ERROR_GIT_COMMAND_FAILED)

    def test_check_ref_exists_propagates_timeout(self):
        with mock.patch(
            "services.dev_git_runner.run_git",
            return_value=dgr.GitRunResult(
                success=False,
                returncode=None,
                stdout=b"",
                stderr=b"",
                stdout_truncated=False,
                stderr_truncated=False,
                error_code=dgr.ERROR_GIT_TIMEOUT,
            ),
        ):
            result = dgr.check_ref_exists("refs/heads/main", cwd=self.repo_root, timeout=10)
        self.assertIsNone(result.exists)
        self.assertEqual(result.error_code, dgr.ERROR_GIT_TIMEOUT)

    def test_map_git_error_preserves_infra_codes(self):
        self.assertEqual(dgr.map_git_error(dgr.ERROR_GIT_TIMEOUT, default="whatever"), dgr.ERROR_GIT_TIMEOUT)
        self.assertEqual(dgr.map_git_error(dgr.ERROR_GIT_COMMAND_FAILED, default="whatever"), "whatever")
        self.assertEqual(dgr.map_git_error(None, default="whatever"), "whatever")


if __name__ == "__main__":
    unittest.main()
