import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import services.claude_headless as ch
import services.dev_copy_workspace as cw
from services.dev_cli_runner import CliRunResult


def _env(**overrides):
    base = {
        "CLAUDE_HEADLESS_ENABLED": "false",
        "CLAUDE_CLI_PATH": "",
        "CLAUDE_HEADLESS_TIMEOUT_SECONDS": "",
        "CLAUDE_HEADLESS_MAX_OUTPUT_BYTES": "",
        "CLAUDE_HEADLESS_ANTHROPIC_API_KEY": "",
        "CLAUDE_HEADLESS_WORKSPACE_MODE": "isolated_copy",
    }
    base.update(overrides)
    return mock.patch.dict(os.environ, base, clear=False)


class ClaudeHeadlessConfigTestCase(unittest.TestCase):
    # --- CLAUDE_HEADLESS_ENABLED: strict bool parsing ---

    def test_enabled_defaults_to_false(self):
        with _env(CLAUDE_HEADLESS_ENABLED=""):
            self.assertFalse(ch.get_claude_headless_enabled())

    def test_enabled_true(self):
        with _env(CLAUDE_HEADLESS_ENABLED="true"):
            self.assertTrue(ch.get_claude_headless_enabled())

    def test_enabled_true_case_insensitive(self):
        with _env(CLAUDE_HEADLESS_ENABLED="TRUE"):
            self.assertTrue(ch.get_claude_headless_enabled())

    def test_enabled_false_explicit(self):
        with _env(CLAUDE_HEADLESS_ENABLED="false"):
            self.assertFalse(ch.get_claude_headless_enabled())

    def test_enabled_invalid_value_defaults_to_false(self):
        with _env(CLAUDE_HEADLESS_ENABLED="yes"):
            self.assertFalse(ch.get_claude_headless_enabled())

    # --- CLAUDE_HEADLESS_TIMEOUT_SECONDS: bounded int ---

    def test_timeout_defaults_when_unset(self):
        with _env(CLAUDE_HEADLESS_TIMEOUT_SECONDS=""):
            self.assertEqual(ch.get_claude_headless_timeout_seconds(), ch.DEFAULT_TIMEOUT_SECONDS)

    def test_timeout_accepts_valid_value(self):
        with _env(CLAUDE_HEADLESS_TIMEOUT_SECONDS="600"):
            self.assertEqual(ch.get_claude_headless_timeout_seconds(), 600)

    def test_timeout_rejects_non_integer(self):
        with _env(CLAUDE_HEADLESS_TIMEOUT_SECONDS="abc"):
            self.assertEqual(ch.get_claude_headless_timeout_seconds(), ch.DEFAULT_TIMEOUT_SECONDS)

    def test_timeout_rejects_below_minimum(self):
        with _env(CLAUDE_HEADLESS_TIMEOUT_SECONDS="1"):
            self.assertEqual(ch.get_claude_headless_timeout_seconds(), ch.DEFAULT_TIMEOUT_SECONDS)

    def test_timeout_rejects_above_maximum(self):
        with _env(CLAUDE_HEADLESS_TIMEOUT_SECONDS="99999999"):
            self.assertEqual(ch.get_claude_headless_timeout_seconds(), ch.DEFAULT_TIMEOUT_SECONDS)

    # --- CLAUDE_HEADLESS_MAX_OUTPUT_BYTES: bounded int ---

    def test_max_output_bytes_defaults_when_unset(self):
        with _env(CLAUDE_HEADLESS_MAX_OUTPUT_BYTES=""):
            self.assertEqual(ch.get_claude_headless_max_output_bytes(), ch.DEFAULT_MAX_OUTPUT_BYTES)

    def test_max_output_bytes_accepts_valid_value(self):
        with _env(CLAUDE_HEADLESS_MAX_OUTPUT_BYTES="2048"):
            self.assertEqual(ch.get_claude_headless_max_output_bytes(), 2048)

    def test_max_output_bytes_rejects_non_integer(self):
        with _env(CLAUDE_HEADLESS_MAX_OUTPUT_BYTES="abc"):
            self.assertEqual(ch.get_claude_headless_max_output_bytes(), ch.DEFAULT_MAX_OUTPUT_BYTES)

    def test_max_output_bytes_rejects_below_minimum(self):
        with _env(CLAUDE_HEADLESS_MAX_OUTPUT_BYTES="1"):
            self.assertEqual(ch.get_claude_headless_max_output_bytes(), ch.DEFAULT_MAX_OUTPUT_BYTES)

    def test_max_output_bytes_rejects_above_maximum(self):
        with _env(CLAUDE_HEADLESS_MAX_OUTPUT_BYTES="99999999999"):
            self.assertEqual(ch.get_claude_headless_max_output_bytes(), ch.DEFAULT_MAX_OUTPUT_BYTES)


class ClaudeCliPathValidationTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="claude_headless_test ")
        self.tmp_path = Path(self._tmp_dir.name)
        self.valid_exe = self.tmp_path / "claude.exe"
        self.valid_exe.write_text("not a real binary, just needs to exist", encoding="utf-8")

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_not_configured(self):
        with _env(CLAUDE_CLI_PATH=""):
            path, error_code = ch.validate_claude_cli_path()
        self.assertIsNone(path)
        self.assertEqual(error_code, ch.ERROR_CLI_PATH_NOT_CONFIGURED)

    def test_relative_path_rejected(self):
        with _env(CLAUDE_CLI_PATH="relative\\claude.exe"):
            path, error_code = ch.validate_claude_cli_path()
        self.assertIsNone(path)
        self.assertEqual(error_code, ch.ERROR_CLI_PATH_NOT_ABSOLUTE)

    def test_nonexistent_path_rejected(self):
        with _env(CLAUDE_CLI_PATH=str(self.tmp_path / "does-not-exist.exe")):
            path, error_code = ch.validate_claude_cli_path()
        self.assertIsNone(path)
        self.assertEqual(error_code, ch.ERROR_CLI_PATH_NOT_FOUND)

    def test_directory_rejected(self):
        directory = self.tmp_path / "a_directory.exe"
        directory.mkdir()
        with _env(CLAUDE_CLI_PATH=str(directory)):
            path, error_code = ch.validate_claude_cli_path()
        self.assertIsNone(path)
        self.assertEqual(error_code, ch.ERROR_CLI_PATH_NOT_A_FILE)

    def test_invalid_extension_rejected_on_windows(self):
        if os.name != "nt":
            self.skipTest("extension allowlist only enforced on Windows")
        bad_ext = self.tmp_path / "claude.txt"
        bad_ext.write_text("x", encoding="utf-8")
        with _env(CLAUDE_CLI_PATH=str(bad_ext)):
            path, error_code = ch.validate_claude_cli_path()
        self.assertIsNone(path)
        self.assertEqual(error_code, ch.ERROR_CLI_PATH_INVALID_EXTENSION)

    def test_valid_path_accepted(self):
        with _env(CLAUDE_CLI_PATH=str(self.valid_exe)):
            path, error_code = ch.validate_claude_cli_path()
        self.assertEqual(path, self.valid_exe)
        self.assertIsNone(error_code)

    def test_reparse_point_rejected(self):
        with _env(CLAUDE_CLI_PATH=str(self.valid_exe)), mock.patch(
            "services.claude_headless._is_reparse_point", return_value=True
        ):
            path, error_code = ch.validate_claude_cli_path()
        self.assertIsNone(path)
        self.assertEqual(error_code, ch.ERROR_CLI_PATH_REPARSE_POINT)


def _cli_result(**overrides) -> CliRunResult:
    defaults = dict(
        success=True,
        returncode=0,
        stdout=b"Claude did the thing.",
        stderr=b"",
        stdout_truncated=False,
        stderr_truncated=False,
        timed_out=False,
        cancelled=False,
        termination_confirmed=True,
        child_processes_may_remain=False,
        duration_seconds=1.5,
        error_code=None,
        stdin_writer_terminated=True,
    )
    defaults.update(overrides)
    return CliRunResult(**defaults)


def _ok_sandbox_verify(sandbox_path: Path) -> cw.SandboxVerifyResult:
    return cw.SandboxVerifyResult(success=True, task_id="t1", sandbox_path=sandbox_path)


class ClaudeHeadlessRunTestCase(unittest.TestCase):
    """services.dev_cli_runner.run_cli() is always mocked here - no real
    CLI (fake or otherwise) is ever launched by this test case; that is
    services.dev_cli_runner's own responsibility (see
    tests/test_dev_cli_runner.py, which uses real fake-script subprocesses)
    and services.dev_copy_workspace's own responsibility (see
    tests/test_dev_copy_workspace.py) - this module's tests only verify
    that claude_headless.run() wires the two together correctly.
    """

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="claude_headless_run_test ")
        self.tmp_path = Path(self._tmp_dir.name)
        self.sandbox_path = self.tmp_path / "sandbox"
        self.sandbox_path.mkdir()
        self.cli_path = self.tmp_path / "claude.exe"
        self.cli_path.write_text("fake", encoding="utf-8")

    def tearDown(self):
        self._tmp_dir.cleanup()

    def _patch_sandbox_ok(self):
        return mock.patch(
            "services.claude_headless.dev_copy_workspace.verify_sandbox_safe",
            return_value=_ok_sandbox_verify(self.sandbox_path),
        )

    def test_disabled_never_launches_cli(self):
        with _env(CLAUDE_HEADLESS_ENABLED="false"), mock.patch(
            "services.claude_headless.dev_cli_runner.run_cli"
        ) as run_cli_mock:
            result = ch.run(task_id="t1", prompt_text="do the thing", timeout_seconds=60)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, ch.ERROR_HEADLESS_DISABLED)
        run_cli_mock.assert_not_called()

    def test_empty_prompt_never_launches_cli(self):
        with _env(CLAUDE_HEADLESS_ENABLED="true", CLAUDE_CLI_PATH=str(self.cli_path)), mock.patch(
            "services.claude_headless.dev_cli_runner.run_cli"
        ) as run_cli_mock:
            result = ch.run(task_id="t1", prompt_text="   ", timeout_seconds=60)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, ch.ERROR_PROMPT_EMPTY)
        run_cli_mock.assert_not_called()

    def test_invalid_cli_path_never_launches_cli(self):
        with _env(CLAUDE_HEADLESS_ENABLED="true", CLAUDE_CLI_PATH=""), mock.patch(
            "services.claude_headless.dev_cli_runner.run_cli"
        ) as run_cli_mock:
            result = ch.run(task_id="t1", prompt_text="do the thing", timeout_seconds=60)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, ch.ERROR_CLI_PATH_NOT_CONFIGURED)
        run_cli_mock.assert_not_called()

    def test_invalid_workspace_mode_never_launches_cli(self):
        with _env(
            CLAUDE_HEADLESS_ENABLED="true",
            CLAUDE_CLI_PATH=str(self.cli_path),
            CLAUDE_HEADLESS_WORKSPACE_MODE="worktree",
        ), mock.patch("services.claude_headless.dev_cli_runner.run_cli") as run_cli_mock:
            result = ch.run(task_id="t1", prompt_text="do the thing", timeout_seconds=60)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, ch.ERROR_WORKSPACE_MODE_INVALID)
        run_cli_mock.assert_not_called()

    def test_sandbox_verification_failure_never_launches_cli(self):
        with _env(CLAUDE_HEADLESS_ENABLED="true", CLAUDE_CLI_PATH=str(self.cli_path)), mock.patch(
            "services.claude_headless.dev_copy_workspace.verify_sandbox_safe",
            return_value=cw.SandboxVerifyResult(
                success=False, task_id="t1", sandbox_path=None, error_code=cw.ERROR_SANDBOX_NOT_FOUND
            ),
        ), mock.patch("services.claude_headless.dev_cli_runner.run_cli") as run_cli_mock:
            result = ch.run(task_id="t1", prompt_text="do the thing", timeout_seconds=60)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, ch.ERROR_SANDBOX_NOT_READY)
        run_cli_mock.assert_not_called()

    def test_success_builds_expected_argv_and_passes_prompt_via_stdin(self):
        with _env(CLAUDE_HEADLESS_ENABLED="true", CLAUDE_CLI_PATH=str(self.cli_path)), self._patch_sandbox_ok(), mock.patch(
            "services.claude_headless.dev_cli_runner.run_cli", return_value=_cli_result()
        ) as run_cli_mock:
            result = ch.run(task_id="t1", prompt_text="fix the bug", timeout_seconds=60)

        self.assertTrue(result.success)
        self.assertIsNone(result.error_code)
        run_cli_mock.assert_called_once()
        args, kwargs = run_cli_mock.call_args
        argv = args[0]
        self.assertEqual(argv[0], str(self.cli_path))
        self.assertIn("-p", argv)
        self.assertIn("--append-system-prompt", argv)
        self.assertIn(ch.SAFETY_SYSTEM_PROMPT, argv)
        # The task body must travel only via stdin, never argv.
        self.assertNotIn("fix the bug", argv)
        self.assertEqual(kwargs["input_bytes"], b"fix the bug")
        self.assertEqual(kwargs["cwd"], self.sandbox_path)

    def test_argv_never_uses_bypass_permissions(self):
        with _env(CLAUDE_HEADLESS_ENABLED="true", CLAUDE_CLI_PATH=str(self.cli_path)), self._patch_sandbox_ok(), mock.patch(
            "services.claude_headless.dev_cli_runner.run_cli", return_value=_cli_result()
        ) as run_cli_mock:
            ch.run(task_id="t1", prompt_text="fix the bug", timeout_seconds=60)
        argv = run_cli_mock.call_args.args[0]
        self.assertNotIn("bypassPermissions", argv)
        self.assertIn("--permission-mode", argv)
        self.assertEqual(argv[argv.index("--permission-mode") + 1], "dontAsk")

    def test_argv_restricts_tools_and_disables_dangerous_integrations(self):
        with _env(CLAUDE_HEADLESS_ENABLED="true", CLAUDE_CLI_PATH=str(self.cli_path)), self._patch_sandbox_ok(), mock.patch(
            "services.claude_headless.dev_cli_runner.run_cli", return_value=_cli_result()
        ) as run_cli_mock:
            ch.run(task_id="t1", prompt_text="fix the bug", timeout_seconds=60)
        argv = run_cli_mock.call_args.args[0]

        self.assertIn("--tools", argv)
        tools_value = argv[argv.index("--tools") + 1]
        allowed_tools = set(tools_value.split(","))
        self.assertEqual(allowed_tools, {"Read", "Edit", "Write", "Glob", "Grep"})
        self.assertNotIn("Bash", allowed_tools)

        for flag in (
            "--bare",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-chrome",
            "--no-session-persistence",
        ):
            self.assertIn(flag, argv)

    def test_timeout_seconds_is_capped_by_config_ceiling(self):
        with _env(
            CLAUDE_HEADLESS_ENABLED="true",
            CLAUDE_CLI_PATH=str(self.cli_path),
            CLAUDE_HEADLESS_TIMEOUT_SECONDS="120",
        ), self._patch_sandbox_ok(), mock.patch(
            "services.claude_headless.dev_cli_runner.run_cli", return_value=_cli_result()
        ) as run_cli_mock:
            ch.run(task_id="t1", prompt_text="fix the bug", timeout_seconds=99999)
        _, kwargs = run_cli_mock.call_args
        self.assertEqual(kwargs["timeout"], 120)

    def test_empty_stdout_despite_returncode_zero_is_failure(self):
        with _env(CLAUDE_HEADLESS_ENABLED="true", CLAUDE_CLI_PATH=str(self.cli_path)), self._patch_sandbox_ok(), mock.patch(
            "services.claude_headless.dev_cli_runner.run_cli",
            return_value=_cli_result(stdout=b"   "),
        ):
            result = ch.run(task_id="t1", prompt_text="fix the bug", timeout_seconds=60)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, ch.ERROR_EMPTY_OUTPUT)

    def test_cli_failure_is_passed_through(self):
        with _env(CLAUDE_HEADLESS_ENABLED="true", CLAUDE_CLI_PATH=str(self.cli_path)), self._patch_sandbox_ok(), mock.patch(
            "services.claude_headless.dev_cli_runner.run_cli",
            return_value=_cli_result(success=False, returncode=1, stdout=b"", error_code="command_failed"),
        ):
            result = ch.run(task_id="t1", prompt_text="fix the bug", timeout_seconds=60)
        self.assertFalse(result.success)
        self.assertEqual(result.returncode, 1)
        self.assertIsNotNone(result.error_message)

    def test_timeout_and_cancelled_flags_are_propagated(self):
        with _env(CLAUDE_HEADLESS_ENABLED="true", CLAUDE_CLI_PATH=str(self.cli_path)), self._patch_sandbox_ok(), mock.patch(
            "services.claude_headless.dev_cli_runner.run_cli",
            return_value=_cli_result(success=False, returncode=None, cancelled=True, error_code="cancelled"),
        ):
            result = ch.run(task_id="t1", prompt_text="fix the bug", timeout_seconds=60)
        self.assertFalse(result.success)
        self.assertTrue(result.cancelled)

    def test_cancel_token_is_forwarded_to_run_cli(self):
        from services.dev_cli_runner import CancellationToken

        token = CancellationToken()
        with _env(CLAUDE_HEADLESS_ENABLED="true", CLAUDE_CLI_PATH=str(self.cli_path)), self._patch_sandbox_ok(), mock.patch(
            "services.claude_headless.dev_cli_runner.run_cli", return_value=_cli_result()
        ) as run_cli_mock:
            ch.run(task_id="t1", prompt_text="fix the bug", timeout_seconds=60, cancel_token=token)
        _, kwargs = run_cli_mock.call_args
        self.assertIs(kwargs["cancel_token"], token)

    # --- env allowlist ---

    def test_env_passed_to_run_cli_is_allowlisted_only(self):
        with mock.patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "super-secret-token"}), _env(
            CLAUDE_HEADLESS_ENABLED="true", CLAUDE_CLI_PATH=str(self.cli_path)
        ), self._patch_sandbox_ok(), mock.patch(
            "services.claude_headless.dev_cli_runner.run_cli", return_value=_cli_result()
        ) as run_cli_mock:
            ch.run(task_id="t1", prompt_text="fix the bug", timeout_seconds=60)
        _, kwargs = run_cli_mock.call_args
        env = kwargs["env"]
        self.assertNotIn("DISCORD_BOT_TOKEN", env)
        for key in env:
            self.assertIn(key, set(ch._ENV_ALLOWED_NAMES) | {"ANTHROPIC_API_KEY"})

    def test_env_includes_anthropic_api_key_when_configured(self):
        with _env(
            CLAUDE_HEADLESS_ENABLED="true",
            CLAUDE_CLI_PATH=str(self.cli_path),
            CLAUDE_HEADLESS_ANTHROPIC_API_KEY="sk-test-123",
        ), self._patch_sandbox_ok(), mock.patch(
            "services.claude_headless.dev_cli_runner.run_cli", return_value=_cli_result()
        ) as run_cli_mock:
            ch.run(task_id="t1", prompt_text="fix the bug", timeout_seconds=60)
        _, kwargs = run_cli_mock.call_args
        self.assertEqual(kwargs["env"]["ANTHROPIC_API_KEY"], "sk-test-123")

    def test_env_omits_anthropic_api_key_when_not_configured(self):
        with _env(CLAUDE_HEADLESS_ENABLED="true", CLAUDE_CLI_PATH=str(self.cli_path)), self._patch_sandbox_ok(), mock.patch(
            "services.claude_headless.dev_cli_runner.run_cli", return_value=_cli_result()
        ) as run_cli_mock:
            ch.run(task_id="t1", prompt_text="fix the bug", timeout_seconds=60)
        _, kwargs = run_cli_mock.call_args
        self.assertNotIn("ANTHROPIC_API_KEY", kwargs["env"])

    # --- discard-guard lifecycle hooks ---

    def test_run_marks_execution_started_and_finished(self):
        with _env(CLAUDE_HEADLESS_ENABLED="true", CLAUDE_CLI_PATH=str(self.cli_path)), self._patch_sandbox_ok(), mock.patch(
            "services.claude_headless.dev_cli_runner.run_cli", return_value=_cli_result()
        ), mock.patch(
            "services.claude_headless.dev_copy_workspace.mark_execution_started"
        ) as started_mock, mock.patch(
            "services.claude_headless.dev_copy_workspace.mark_execution_finished"
        ) as finished_mock:
            ch.run(task_id="t1", prompt_text="fix the bug", timeout_seconds=60)

        started_mock.assert_called_once_with("t1")
        finished_mock.assert_called_once_with("t1", termination_confirmed=True)

    def test_run_marks_finished_with_unconfirmed_termination_on_timeout(self):
        with _env(CLAUDE_HEADLESS_ENABLED="true", CLAUDE_CLI_PATH=str(self.cli_path)), self._patch_sandbox_ok(), mock.patch(
            "services.claude_headless.dev_cli_runner.run_cli",
            return_value=_cli_result(
                success=False, returncode=None, timed_out=True, termination_confirmed=False, error_code="timeout"
            ),
        ), mock.patch("services.claude_headless.dev_copy_workspace.mark_execution_finished") as finished_mock:
            ch.run(task_id="t1", prompt_text="fix the bug", timeout_seconds=60)

        finished_mock.assert_called_once_with("t1", termination_confirmed=False)

    def test_run_cli_raising_still_marks_execution_finished_as_unconfirmed(self):
        with _env(CLAUDE_HEADLESS_ENABLED="true", CLAUDE_CLI_PATH=str(self.cli_path)), self._patch_sandbox_ok(), mock.patch(
            "services.claude_headless.dev_cli_runner.run_cli", side_effect=RuntimeError("boom")
        ), mock.patch("services.claude_headless.dev_copy_workspace.mark_execution_finished") as finished_mock:
            result = ch.run(task_id="t1", prompt_text="fix the bug", timeout_seconds=60)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, ch.ERROR_RUN_CLI_RAISED)
        finished_mock.assert_called_once_with("t1", termination_confirmed=False)


class ClassifyStopOutcomeTestCase(unittest.TestCase):
    def _result(self, **overrides):
        defaults = dict(
            success=True,
            returncode=0,
            stdout=b"ok",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
            timed_out=False,
            cancelled=False,
            termination_confirmed=True,
            child_processes_may_remain=False,
            stdin_writer_terminated=True,
            duration_seconds=1.0,
            error_code=None,
            error_message=None,
        )
        defaults.update(overrides)
        return ch.ClaudeHeadlessResult(**defaults)

    def test_stopped_when_cancelled_and_confirmed(self):
        result = self._result(success=False, cancelled=True, termination_confirmed=True)
        self.assertEqual(ch.classify_stop_outcome(result, was_cancel_requested=True), "stopped")

    def test_termination_unconfirmed_when_cancelled_but_not_confirmed(self):
        result = self._result(success=False, cancelled=True, termination_confirmed=False)
        self.assertEqual(
            ch.classify_stop_outcome(result, was_cancel_requested=True), "termination_unconfirmed"
        )

    def test_already_finished_when_no_cancel_was_requested(self):
        result = self._result(success=True)
        self.assertEqual(
            ch.classify_stop_outcome(result, was_cancel_requested=False), "already_finished"
        )

    def test_already_finished_when_cancel_lost_the_race(self):
        result = self._result(success=True, cancelled=False)
        self.assertEqual(
            ch.classify_stop_outcome(result, was_cancel_requested=True), "already_finished"
        )

    def test_failed_for_any_other_outcome(self):
        result = self._result(success=False, cancelled=False, error_code="command_failed")
        self.assertEqual(ch.classify_stop_outcome(result, was_cancel_requested=True), "failed")


if __name__ == "__main__":
    unittest.main()
