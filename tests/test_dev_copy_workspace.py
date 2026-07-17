import os
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import services.dev_copy_workspace as cw


def _env(**overrides):
    base = {
        "DEV_TASK_SANDBOX_ROOT": "",
        "DEV_TASK_SANDBOX_MAX_FILES": "",
        "DEV_TASK_SANDBOX_MAX_TOTAL_BYTES": "",
        "CLAUDE_HEADLESS_WORKSPACE_MODE": "isolated_copy",
    }
    base.update(overrides)
    return mock.patch.dict(os.environ, base, clear=False)


class DevCopyWorkspaceTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="dev_copy_workspace_test ")
        self.tmp_path = Path(self._tmp_dir.name)

        self.fake_repo_root = self.tmp_path / "repo"
        self.fake_repo_root.mkdir()
        self.sandbox_root = self.tmp_path / "sandboxes"
        self.sandbox_root.mkdir()

        self._original_repo_root = cw.REPO_ROOT
        cw.REPO_ROOT = self.fake_repo_root

        self._env_patch = _env(DEV_TASK_SANDBOX_ROOT=str(self.sandbox_root))
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        cw.REPO_ROOT = self._original_repo_root
        self._tmp_dir.cleanup()
        cw._execution_state.clear()

    def _new_task_id(self) -> str:
        return str(uuid.uuid4())

    def _write(self, relative_path: str, content: str = "x") -> Path:
        path = self.fake_repo_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    # --- allowlist: what gets copied ---

    def test_create_sandbox_copies_allowed_files(self):
        self._write("main.py", "print('hi')\n")
        self._write("README.md", "# hi\n")
        self._write("requirements.txt", "discord.py\n")
        self._write("services/dev_task_executor.py", "x = 1\n")

        task_id = self._new_task_id()
        result = cw.create_sandbox(task_id)

        self.assertTrue(result.success)
        self.assertEqual(result.file_count, 4)
        self.assertTrue((result.sandbox_path / "main.py").is_file())
        self.assertTrue((result.sandbox_path / "README.md").is_file())
        self.assertTrue((result.sandbox_path / "requirements.txt").is_file())
        self.assertTrue((result.sandbox_path / "services" / "dev_task_executor.py").is_file())

    def test_create_sandbox_excludes_git_directory(self):
        self._write(".git/config", "secret git internals")
        self._write("main.py", "print('hi')\n")

        task_id = self._new_task_id()
        result = cw.create_sandbox(task_id)

        self.assertTrue(result.success)
        self.assertFalse((result.sandbox_path / ".git").exists())

    def test_create_sandbox_excludes_env_but_allows_env_example(self):
        self._write(".env", "DISCORD_BOT_TOKEN=super-secret")
        self._write(".env.local", "ALSO_SECRET=1")
        self._write(".env.example", "DISCORD_BOT_TOKEN=placeholder")

        task_id = self._new_task_id()
        result = cw.create_sandbox(task_id)

        self.assertTrue(result.success)
        self.assertFalse((result.sandbox_path / ".env").exists())
        self.assertFalse((result.sandbox_path / ".env.local").exists())
        self.assertTrue((result.sandbox_path / ".env.example").is_file())

    def test_create_sandbox_excludes_data_outputs_logs_work_dirs(self):
        self._write("data/dev_tasks/task.json", "{}")
        self._write("outputs/generated.png.txt", "binary-ish")
        self._write("logs/bot.log.txt", "log line")
        self._write("work/some-task/file.py", "x = 1")
        self._write("main.py", "print('hi')\n")

        task_id = self._new_task_id()
        result = cw.create_sandbox(task_id)

        self.assertTrue(result.success)
        for forbidden in ("data", "outputs", "logs", "work"):
            self.assertFalse((result.sandbox_path / forbidden).exists(), forbidden)
        self.assertTrue((result.sandbox_path / "main.py").is_file())

    def test_create_sandbox_excludes_denied_extensions(self):
        self._write("id_rsa.key", "not a real key")
        self._write("cert.pem", "not a real cert")
        self._write("app.db", "not a real db")
        self._write("bot.log", "not a real log")
        self._write("main.py", "print('hi')\n")

        task_id = self._new_task_id()
        result = cw.create_sandbox(task_id)

        self.assertTrue(result.success)
        self.assertEqual(result.file_count, 1)
        self.assertTrue((result.sandbox_path / "main.py").is_file())

    def test_create_sandbox_excludes_claude_and_codex_state_dirs(self):
        self._write(".claude/session.json", "possibly sensitive session state")
        self._write(".codex/state.json", "possibly sensitive state")
        self._write("main.py", "print('hi')\n")

        task_id = self._new_task_id()
        result = cw.create_sandbox(task_id)

        self.assertTrue(result.success)
        self.assertFalse((result.sandbox_path / ".claude").exists())
        self.assertFalse((result.sandbox_path / ".codex").exists())

    def test_create_sandbox_excludes_bat_and_other_non_allowlisted_extensions(self):
        self._write("start_bot.bat", "@echo off\nstart bot.exe\n")
        self._write("main.py", "print('hi')\n")

        task_id = self._new_task_id()
        result = cw.create_sandbox(task_id)

        self.assertTrue(result.success)
        self.assertFalse((result.sandbox_path / "start_bot.bat").exists())

    # --- symlink / junction: never followed, never copied ---

    def test_create_sandbox_rejects_symlinked_source_file(self):
        real_file = self.tmp_path / "outside_secret.py"
        real_file.write_text("secret = 1\n", encoding="utf-8")
        link_path = self.fake_repo_root / "linked.py"
        try:
            link_path.symlink_to(real_file)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation is not permitted in this environment")

        task_id = self._new_task_id()
        result = cw.create_sandbox(task_id)

        self.assertTrue(result.success)
        self.assertFalse((result.sandbox_path / "linked.py").exists())

    def test_create_sandbox_does_not_follow_junction_directory(self):
        if os.name != "nt":
            self.skipTest("junction test is Windows-specific")

        outside_dir = self.tmp_path / "outside_dir"
        outside_dir.mkdir()
        (outside_dir / "leaked.py").write_text("leaked = 1\n", encoding="utf-8")

        junction_path = self.fake_repo_root / "services_junction"
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(junction_path), str(outside_dir)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            self.skipTest(f"could not create an NTFS junction in this environment: {proc.stderr}")

        self._write("main.py", "print('hi')\n")
        task_id = self._new_task_id()
        result = cw.create_sandbox(task_id)

        self.assertTrue(result.success)
        self.assertFalse((result.sandbox_path / "services_junction").exists())
        self.assertFalse((result.sandbox_path / "services_junction" / "leaked.py").exists())

    # --- limits ---

    def test_create_sandbox_enforces_max_files_limit_and_cleans_up(self):
        self._write("a.py", "1")
        self._write("b.py", "2")
        self._write("c.py", "3")

        task_id = self._new_task_id()
        with _env(DEV_TASK_SANDBOX_ROOT=str(self.sandbox_root), DEV_TASK_SANDBOX_MAX_FILES="2"):
            result = cw.create_sandbox(task_id)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, cw.ERROR_LIMIT_EXCEEDED)
        self.assertFalse(result.sandbox_path.exists())

    def test_create_sandbox_enforces_max_total_bytes_limit_and_cleans_up(self):
        self._write("big.txt", "x" * 5000)

        task_id = self._new_task_id()
        with _env(
            DEV_TASK_SANDBOX_ROOT=str(self.sandbox_root), DEV_TASK_SANDBOX_MAX_TOTAL_BYTES=str(cw.MIN_MAX_TOTAL_BYTES)
        ):
            result = cw.create_sandbox(task_id)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, cw.ERROR_LIMIT_EXCEEDED)
        self.assertFalse(result.sandbox_path.exists())

    # --- config / basic validation ---

    def test_create_sandbox_rejects_invalid_task_id(self):
        result = cw.create_sandbox("not-a-uuid")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, cw.ERROR_INVALID_TASK_ID)

    def test_create_sandbox_rejects_when_sandbox_root_not_configured(self):
        self._write("main.py", "1")
        with _env(DEV_TASK_SANDBOX_ROOT=""):
            result = cw.create_sandbox(self._new_task_id())
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, cw.ERROR_SANDBOX_ROOT_NOT_CONFIGURED)

    def test_create_sandbox_rejects_when_workspace_mode_invalid(self):
        self._write("main.py", "1")
        with _env(CLAUDE_HEADLESS_WORKSPACE_MODE="worktree"):
            result = cw.create_sandbox(self._new_task_id())
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, cw.ERROR_WORKSPACE_MODE_INVALID)

    def test_create_sandbox_rejects_relative_sandbox_root(self):
        with _env(DEV_TASK_SANDBOX_ROOT="relative\\sandboxes"):
            root = cw.get_sandbox_root()
        self.assertIsNone(root)

    def test_create_sandbox_rejects_when_path_already_exists(self):
        self._write("main.py", "1")
        task_id = self._new_task_id()
        (self.sandbox_root / task_id).mkdir()

        result = cw.create_sandbox(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, cw.ERROR_PATH_ALREADY_EXISTS)

    # --- DEV_TASK_SANDBOX_ROOT vs REPO_ROOT containment (must never overlap,
    # in either direction, checked via resolve() rather than string prefix) ---

    def test_create_sandbox_rejects_sandbox_root_equal_to_repo_root(self):
        self._write("main.py", "1")
        task_id = self._new_task_id()
        with _env(DEV_TASK_SANDBOX_ROOT=str(self.fake_repo_root)):
            result = cw.create_sandbox(task_id)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, cw.ERROR_SANDBOX_ROOT_CONFLICTS_WITH_REPO)
        self.assertFalse((self.fake_repo_root / task_id).exists())
        self.assertFalse(cw._baseline_path_for(self.fake_repo_root, task_id).exists())

    def test_create_sandbox_rejects_sandbox_root_directly_under_repo_root(self):
        self._write("main.py", "1")
        nested_root = self.fake_repo_root / "sandboxes"
        task_id = self._new_task_id()
        with _env(DEV_TASK_SANDBOX_ROOT=str(nested_root)):
            result = cw.create_sandbox(task_id)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, cw.ERROR_SANDBOX_ROOT_CONFLICTS_WITH_REPO)
        self.assertFalse(nested_root.exists())
        self.assertFalse(cw._baseline_path_for(nested_root, task_id).exists())

    def test_create_sandbox_rejects_sandbox_root_deeply_nested_under_repo_root(self):
        self._write("main.py", "1")
        nested_root = self.fake_repo_root / "a" / "b" / "c" / "sandboxes"
        task_id = self._new_task_id()
        with _env(DEV_TASK_SANDBOX_ROOT=str(nested_root)):
            result = cw.create_sandbox(task_id)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, cw.ERROR_SANDBOX_ROOT_CONFLICTS_WITH_REPO)
        self.assertFalse(nested_root.exists())
        self.assertFalse((self.fake_repo_root / "a").exists())

    def test_create_sandbox_rejects_repo_root_nested_under_sandbox_root(self):
        # The reverse direction: DEV_TASK_SANDBOX_ROOT configured as an
        # ANCESTOR of REPO_ROOT - equally unsafe (see
        # _sandbox_root_conflicts_with_repo_root()'s own docstring), even
        # though it doesn't trigger the recursive-self-copy failure mode
        # the direct-nesting case does.
        self._write("main.py", "1")
        ancestor_root = self.fake_repo_root.parent
        task_id = self._new_task_id()
        with _env(DEV_TASK_SANDBOX_ROOT=str(ancestor_root)):
            result = cw.create_sandbox(task_id)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, cw.ERROR_SANDBOX_ROOT_CONFLICTS_WITH_REPO)
        self.assertFalse((ancestor_root / task_id).exists())

    def test_create_sandbox_rejects_sandbox_root_matching_repo_root_after_resolve(self):
        # Same real directory as REPO_ROOT, spelled with different casing
        # and forward slashes - must still be caught because the
        # comparison resolves both paths (Path.resolve() + Windows'
        # inherently case-insensitive PureWindowsPath comparison) instead
        # of comparing raw strings.
        if os.name != "nt":
            self.skipTest("case-insensitive path matching is Windows-specific")
        self._write("main.py", "1")
        differently_cased = str(self.fake_repo_root).upper().replace("\\", "/")
        task_id = self._new_task_id()
        with _env(DEV_TASK_SANDBOX_ROOT=differently_cased):
            result = cw.create_sandbox(task_id)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, cw.ERROR_SANDBOX_ROOT_CONFLICTS_WITH_REPO)

    def test_create_sandbox_allows_normal_sandbox_root_outside_repo(self):
        # Sanity check: the new containment guard must never reject a
        # properly configured, wholly separate sandbox root - this is
        # exactly this test class's own default setUp() (self.sandbox_root
        # is a sibling of self.fake_repo_root, not nested either way).
        self._write("main.py", "1")
        task_id = self._new_task_id()
        result = cw.create_sandbox(task_id)

        self.assertTrue(result.success)
        self.assertNotEqual(result.error_code, cw.ERROR_SANDBOX_ROOT_CONFLICTS_WITH_REPO)

    def test_verify_sandbox_safe_rejects_when_sandbox_root_conflicts_with_repo(self):
        # services.claude_headless.run() calls verify_sandbox_safe()
        # immediately before ever launching the Claude CLI - this guard
        # must also apply there, independent of create_sandbox() ever
        # having run for this task_id, so a misconfigured
        # DEV_TASK_SANDBOX_ROOT can never let the CLI launch.
        task_id = self._new_task_id()
        with _env(DEV_TASK_SANDBOX_ROOT=str(self.fake_repo_root)):
            result = cw.verify_sandbox_safe(task_id)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, cw.ERROR_SANDBOX_ROOT_CONFLICTS_WITH_REPO)

    def test_discard_sandbox_rejects_when_sandbox_root_conflicts_with_repo(self):
        task_id = self._new_task_id()
        with _env(DEV_TASK_SANDBOX_ROOT=str(self.fake_repo_root)):
            result = cw.discard_sandbox(task_id)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, cw.ERROR_SANDBOX_ROOT_CONFLICTS_WITH_REPO)

    def test_sandbox_root_conflicts_with_repo_root_helper_direct(self):
        self.assertTrue(cw._sandbox_root_conflicts_with_repo_root(self.fake_repo_root))
        self.assertTrue(cw._sandbox_root_conflicts_with_repo_root(self.fake_repo_root / "nested"))
        self.assertTrue(cw._sandbox_root_conflicts_with_repo_root(self.fake_repo_root.parent))
        self.assertFalse(cw._sandbox_root_conflicts_with_repo_root(self.sandbox_root))

    # --- partial-failure cleanup ---

    def test_create_sandbox_cleans_up_partial_residue_on_copy_failure(self):
        self._write("a.py", "1")
        self._write("b.py", "2")

        task_id = self._new_task_id()
        with mock.patch("services.dev_copy_workspace._copy_file_with_hash", return_value=None):
            result = cw.create_sandbox(task_id)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, cw.ERROR_COPY_FAILED)
        self.assertFalse(result.sandbox_path.exists())

    # --- _copy_file_with_hash hardlink-identity defense ---

    def test_copy_file_with_hash_rejects_hardlinked_destination(self):
        source = self.tmp_path / "hardlink_source.txt"
        source.write_text("original content", encoding="utf-8")
        dest = self.tmp_path / "hardlink_dest.txt"
        try:
            os.link(source, dest)
        except OSError:
            self.skipTest("hardlink creation is not permitted in this environment")

        result = cw._copy_file_with_hash(source, dest)
        self.assertIsNone(result)

    def test_copy_file_with_hash_succeeds_for_a_genuine_independent_copy(self):
        source = self.tmp_path / "src.txt"
        source.write_text("hello world", encoding="utf-8")
        dest = self.tmp_path / "dst.txt"

        file_hash = cw._copy_file_with_hash(source, dest)
        self.assertIsNotNone(file_hash)
        self.assertEqual(dest.read_text(encoding="utf-8"), "hello world")

    # --- verify_sandbox_safe ---

    def test_verify_sandbox_safe_succeeds_after_create(self):
        self._write("main.py", "print(1)\n")
        task_id = self._new_task_id()
        create_result = cw.create_sandbox(task_id)
        self.assertTrue(create_result.success)

        verify_result = cw.verify_sandbox_safe(task_id)
        self.assertTrue(verify_result.success)

    def test_verify_sandbox_safe_rejects_missing_sandbox(self):
        result = cw.verify_sandbox_safe(self._new_task_id())
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, cw.ERROR_SANDBOX_NOT_FOUND)

    def test_verify_sandbox_safe_detects_git_planted_after_creation(self):
        self._write("main.py", "print(1)\n")
        task_id = self._new_task_id()
        create_result = cw.create_sandbox(task_id)
        self.assertTrue(create_result.success)

        (create_result.sandbox_path / ".git").mkdir()

        verify_result = cw.verify_sandbox_safe(task_id)
        self.assertFalse(verify_result.success)
        self.assertEqual(verify_result.error_code, cw.ERROR_FORBIDDEN_CONTENT_FOUND)

    def test_verify_sandbox_safe_detects_env_planted_after_creation(self):
        self._write("main.py", "print(1)\n")
        task_id = self._new_task_id()
        create_result = cw.create_sandbox(task_id)
        self.assertTrue(create_result.success)

        (create_result.sandbox_path / ".env").write_text("SECRET=1", encoding="utf-8")

        verify_result = cw.verify_sandbox_safe(task_id)
        self.assertFalse(verify_result.success)
        self.assertEqual(verify_result.error_code, cw.ERROR_FORBIDDEN_CONTENT_FOUND)

    def test_verify_sandbox_safe_detects_symlink_planted_after_creation(self):
        self._write("main.py", "print(1)\n")
        task_id = self._new_task_id()
        create_result = cw.create_sandbox(task_id)
        self.assertTrue(create_result.success)

        outside = self.tmp_path / "outside.txt"
        outside.write_text("data", encoding="utf-8")
        try:
            (create_result.sandbox_path / "planted_link.txt").symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation is not permitted in this environment")

        verify_result = cw.verify_sandbox_safe(task_id)
        self.assertFalse(verify_result.success)
        self.assertEqual(verify_result.error_code, cw.ERROR_FORBIDDEN_CONTENT_FOUND)

    def test_verify_sandbox_safe_detects_hardlink_shared_with_source(self):
        self._write("main.py", "print(1)\n")
        task_id = self._new_task_id()
        create_result = cw.create_sandbox(task_id)
        self.assertTrue(create_result.success)

        sandbox_file = create_result.sandbox_path / "main.py"
        sandbox_file.unlink()
        try:
            os.link(self.fake_repo_root / "main.py", sandbox_file)
        except OSError:
            self.skipTest("hardlink creation is not permitted in this environment")

        verify_result = cw.verify_sandbox_safe(task_id)
        self.assertFalse(verify_result.success)
        self.assertEqual(verify_result.error_code, cw.ERROR_FORBIDDEN_CONTENT_FOUND)

    def test_verify_sandbox_safe_enforces_limits_live(self):
        self._write("a.py", "1")
        self._write("b.py", "2")
        task_id = self._new_task_id()
        create_result = cw.create_sandbox(task_id)
        self.assertTrue(create_result.success)

        with _env(DEV_TASK_SANDBOX_ROOT=str(self.sandbox_root), DEV_TASK_SANDBOX_MAX_FILES="1"):
            verify_result = cw.verify_sandbox_safe(task_id)
        self.assertFalse(verify_result.success)
        self.assertEqual(verify_result.error_code, cw.ERROR_LIMIT_EXCEEDED)

    # --- path boundary ---

    def test_path_boundary_rejects_junction_between_root_and_target(self):
        if os.name != "nt":
            self.skipTest("junction test is Windows-specific")

        real_dir = self.tmp_path / "real_sandbox_target"
        real_dir.mkdir()
        junction_root = self.tmp_path / "junction_sandbox_root"
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(junction_root), str(real_dir)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            self.skipTest(f"could not create an NTFS junction in this environment: {proc.stderr}")

        task_id = self._new_task_id()
        result = cw._validate_path_boundary(junction_root, junction_root / task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, cw.ERROR_PATH_BOUNDARY_VIOLATION)

    # --- diff (hash-based, never git) ---

    def test_diff_sandbox_reports_no_changes_when_untouched(self):
        self._write("main.py", "print(1)\n")
        task_id = self._new_task_id()
        create_result = cw.create_sandbox(task_id)
        self.assertTrue(create_result.success)

        diff_result = cw.diff_sandbox_against_baseline(task_id)
        self.assertTrue(diff_result.success)
        self.assertEqual(diff_result.changed_file_count, 0)
        self.assertFalse(diff_result.manual_review_required)

    def test_diff_sandbox_detects_added_modified_removed(self):
        self._write("keep.py", "unchanged\n")
        self._write("modify_me.py", "before\n")
        self._write("remove_me.py", "will be removed\n")
        task_id = self._new_task_id()
        create_result = cw.create_sandbox(task_id)
        self.assertTrue(create_result.success)

        (create_result.sandbox_path / "modify_me.py").write_text("after\n", encoding="utf-8")
        (create_result.sandbox_path / "remove_me.py").unlink()
        (create_result.sandbox_path / "new_file.py").write_text("new\n", encoding="utf-8")

        diff_result = cw.diff_sandbox_against_baseline(task_id)
        self.assertTrue(diff_result.success)
        self.assertEqual(diff_result.added_count, 1)
        self.assertEqual(diff_result.modified_count, 1)
        self.assertEqual(diff_result.removed_count, 1)
        self.assertEqual(diff_result.changed_file_count, 3)
        self.assertIn("modify_me.py", diff_result.sample_changed_names)

    # --- independence: neither direction of mutation propagates ---

    def test_source_mutation_after_copy_does_not_affect_sandbox(self):
        source_file = self._write("shared.py", "original\n")
        task_id = self._new_task_id()
        create_result = cw.create_sandbox(task_id)
        self.assertTrue(create_result.success)

        source_file.write_text("mutated in the real repo\n", encoding="utf-8")

        sandbox_content = (create_result.sandbox_path / "shared.py").read_text(encoding="utf-8")
        self.assertEqual(sandbox_content, "original\n")

    def test_sandbox_mutation_does_not_affect_source(self):
        source_file = self._write("shared.py", "original\n")
        task_id = self._new_task_id()
        create_result = cw.create_sandbox(task_id)
        self.assertTrue(create_result.success)

        (create_result.sandbox_path / "shared.py").write_text("mutated by claude\n", encoding="utf-8")

        self.assertEqual(source_file.read_text(encoding="utf-8"), "original\n")

    # --- discard + execution-in-progress guard ---

    def test_discard_sandbox_removes_directory_and_baseline(self):
        self._write("main.py", "1")
        task_id = self._new_task_id()
        create_result = cw.create_sandbox(task_id)
        self.assertTrue(create_result.success)
        baseline_path = cw._baseline_path_for(self.sandbox_root, task_id)
        self.assertTrue(baseline_path.exists())

        discard_result = cw.discard_sandbox(task_id)
        self.assertTrue(discard_result.success)
        self.assertFalse(create_result.sandbox_path.exists())
        self.assertFalse(baseline_path.exists())

    def test_discard_sandbox_idempotent_when_already_absent(self):
        result = cw.discard_sandbox(self._new_task_id())
        self.assertTrue(result.success)

    def test_discard_sandbox_refuses_while_execution_in_progress(self):
        self._write("main.py", "1")
        task_id = self._new_task_id()
        create_result = cw.create_sandbox(task_id)
        self.assertTrue(create_result.success)

        cw.mark_execution_started(task_id)
        try:
            discard_result = cw.discard_sandbox(task_id)
            self.assertFalse(discard_result.success)
            self.assertEqual(discard_result.error_code, cw.ERROR_EXECUTION_IN_PROGRESS)
            self.assertTrue(create_result.sandbox_path.is_dir())
        finally:
            cw.mark_execution_finished(task_id, termination_confirmed=True)

    def test_discard_sandbox_refuses_when_termination_unconfirmed(self):
        self._write("main.py", "1")
        task_id = self._new_task_id()
        create_result = cw.create_sandbox(task_id)
        self.assertTrue(create_result.success)

        cw.mark_execution_started(task_id)
        cw.mark_execution_finished(task_id, termination_confirmed=False)

        discard_result = cw.discard_sandbox(task_id)
        self.assertFalse(discard_result.success)
        self.assertEqual(discard_result.error_code, cw.ERROR_TERMINATION_UNCONFIRMED)
        self.assertTrue(create_result.sandbox_path.is_dir())

    def test_discard_sandbox_allowed_after_execution_finished_confirmed(self):
        self._write("main.py", "1")
        task_id = self._new_task_id()
        create_result = cw.create_sandbox(task_id)
        self.assertTrue(create_result.success)

        cw.mark_execution_started(task_id)
        cw.mark_execution_finished(task_id, termination_confirmed=True)

        discard_result = cw.discard_sandbox(task_id)
        self.assertTrue(discard_result.success)
        self.assertFalse(create_result.sandbox_path.exists())

    def test_discard_sandbox_never_touches_a_different_tasks_sandbox(self):
        self._write("main.py", "1")
        task_id_a = self._new_task_id()
        task_id_b = self._new_task_id()
        result_a = cw.create_sandbox(task_id_a)
        result_b = cw.create_sandbox(task_id_b)
        self.assertTrue(result_a.success)
        self.assertTrue(result_b.success)

        cw.mark_execution_started(task_id_a)
        try:
            discard_b = cw.discard_sandbox(task_id_b)
            self.assertTrue(discard_b.success)
            self.assertTrue(result_a.sandbox_path.is_dir())
            self.assertFalse(result_b.sandbox_path.exists())
        finally:
            cw.mark_execution_finished(task_id_a, termination_confirmed=True)

    # --- filename sanitization ---

    def test_sanitize_filename_strips_ansi_escape(self):
        raw = "before\x1b[31mred\x1b[0mafter.py"
        sanitized = cw.sanitize_filename_for_display(raw)
        self.assertNotIn("\x1b", sanitized)
        self.assertIn("before", sanitized)
        self.assertIn("after.py", sanitized)

    def test_sanitize_filename_strips_control_chars(self):
        raw = "a\x00b\x07c.py"
        sanitized = cw.sanitize_filename_for_display(raw)
        self.assertEqual(sanitized, "abc.py")

    def test_sanitize_filename_strips_bidi_override(self):
        raw = "safe" + chr(0x202E) + "evil.exe" + chr(0x202C) + ".py"
        sanitized = cw.sanitize_filename_for_display(raw)
        self.assertNotIn(chr(0x202E), sanitized)
        self.assertNotIn(chr(0x202C), sanitized)

    def test_sanitize_filename_strips_mentions_and_backticks(self):
        raw = "@everyone `rm -rf /` <@123456> file.py"
        sanitized = cw.sanitize_filename_for_display(raw)
        self.assertNotIn("@", sanitized)
        self.assertNotIn("`", sanitized)
        self.assertNotIn("<", sanitized)
        self.assertNotIn(">", sanitized)

    def test_sanitize_filename_strips_newlines(self):
        raw = "line1\nline2\rline3.py"
        sanitized = cw.sanitize_filename_for_display(raw)
        self.assertNotIn("\n", sanitized)
        self.assertNotIn("\r", sanitized)

    def test_sanitize_filename_truncates_to_max_length(self):
        raw = "a" * 500 + ".py"
        sanitized = cw.sanitize_filename_for_display(raw, max_length=50)
        self.assertEqual(len(sanitized), 50)

    def test_sanitize_filename_handles_non_string_input(self):
        self.assertEqual(cw.sanitize_filename_for_display(None), "")

    # --- workspace mode config ---

    def test_workspace_mode_defaults_to_isolated_copy(self):
        with _env(CLAUDE_HEADLESS_WORKSPACE_MODE=""):
            mode = cw.get_workspace_mode()
        self.assertEqual(mode, cw.WORKSPACE_MODE_ISOLATED_COPY)
        self.assertTrue(cw.is_workspace_mode_valid(mode))

    def test_workspace_mode_worktree_is_invalid(self):
        self.assertFalse(cw.is_workspace_mode_valid("worktree"))


if __name__ == "__main__":
    unittest.main()
