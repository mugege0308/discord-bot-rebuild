import os
import shutil
import subprocess
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from unittest import mock

import services.dev_git_runner as dgr
import services.dev_worktree as wt


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True, encoding="utf-8"
    )


def _branch_exists_real(branch_name: str, cwd: Path) -> bool:
    result = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.returncode == 0


def _clean_info(task_id: str, worktree_path: Path, branch_name: str) -> wt.WorktreeInfoResult:
    """A WorktreeInfoResult reporting "nothing exists yet" for the given
    task, used to fool create_worktree()'s own pre-check exactly once so
    the real `git worktree add` call is reached against a path that (in
    reality) is already occupied - simulating the TOCTOU gap a genuine race
    would produce, without needing an actual second process.
    """
    return wt.WorktreeInfoResult(
        success=True,
        info=wt.WorktreeInfo(
            task_id=task_id,
            worktree_path=worktree_path,
            branch_name=branch_name,
            path_entry_exists=False,
            branch_exists=False,
            registered_in_git_worktree=False,
            registered_branch_ref=None,
            registered_detached=False,
            branch_mismatch=False,
            consistency=wt.CONSISTENCY_NOT_FOUND,
        ),
    )


def _single_shot_bypass(task_id: str, worktree_path: Path, branch_name: str):
    """Return a get_worktree_info side_effect that reports "clean" exactly
    once (for create_worktree()'s own pre-check) and delegates to the real
    function for every subsequent call, so any post-check made after the
    pre-check (e.g. inside _cleanup_failed_create()) sees reality.
    """
    real_get_worktree_info = wt.get_worktree_info
    calls = {"n": 0}

    def fake(tid):
        calls["n"] += 1
        if calls["n"] == 1:
            return _clean_info(task_id, worktree_path, branch_name)
        return real_get_worktree_info(tid)

    return fake


def _git_not_found_result() -> dgr.GitRunResult:
    return dgr.GitRunResult(
        success=False,
        returncode=None,
        stdout=b"",
        stderr=b"",
        stdout_truncated=False,
        stderr_truncated=False,
        error_code=dgr.ERROR_GIT_NOT_FOUND,
    )


class DevWorktreeTestCase(unittest.TestCase):
    def setUp(self):
        # The trailing space in the prefix means every temp repo path used
        # by these tests contains a space, exercising Windows path handling
        # with spaces without needing a dedicated test.
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="dev_worktree_test ")
        self.repo_root = Path(self._tmp_dir.name) / "repo"
        self.repo_root.mkdir(parents=True)
        self._original_repo_root = wt.REPO_ROOT
        wt.REPO_ROOT = self.repo_root

        _git(["init", "-q", "-b", "main"], self.repo_root)
        _git(["config", "user.name", "Test User"], self.repo_root)
        _git(["config", "user.email", "test@example.com"], self.repo_root)
        (self.repo_root / "README.md").write_text("hello\n", encoding="utf-8")
        _git(["add", "README.md"], self.repo_root)
        _git(["commit", "-q", "-m", "init"], self.repo_root)

    def tearDown(self):
        wt.REPO_ROOT = self._original_repo_root
        self._tmp_dir.cleanup()

    def _new_task_id(self) -> str:
        return str(uuid.uuid4())

    # --- 1/2/3: happy path create / remove / no trace left behind ---

    def test_create_worktree_succeeds(self):
        task_id = self._new_task_id()
        result = wt.create_worktree(task_id, "main")

        self.assertTrue(result.success)
        self.assertEqual(result.task_id, task_id)
        self.assertEqual(result.branch_name, f"dev-task/{task_id}")
        self.assertEqual(result.worktree_path, self.repo_root / "work" / task_id)
        self.assertTrue(result.worktree_path.is_dir())
        self.assertTrue(_branch_exists_real(result.branch_name, self.repo_root))
        self.assertIsNone(result.cleanup)

    def test_remove_worktree_succeeds(self):
        task_id = self._new_task_id()
        create_result = wt.create_worktree(task_id, "main")
        self.assertTrue(create_result.success)

        result = wt.remove_worktree(task_id)
        self.assertTrue(result.success)
        self.assertTrue(result.worktree_removed)
        self.assertTrue(result.branch_removed)

    def test_remove_worktree_leaves_no_trace(self):
        task_id = self._new_task_id()
        create_result = wt.create_worktree(task_id, "main")
        wt.remove_worktree(task_id)

        info = wt.get_worktree_info(task_id).info
        self.assertFalse(info.path_entry_exists)
        self.assertFalse(info.branch_exists)
        self.assertFalse(info.registered_in_git_worktree)
        self.assertEqual(info.consistency, wt.CONSISTENCY_NOT_FOUND)
        self.assertFalse(create_result.worktree_path.exists())

    # --- 4/5: task_id validation ---

    def test_create_worktree_rejects_invalid_task_id(self):
        result = wt.create_worktree("not-a-uuid", "main")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_INVALID_TASK_ID)

    def test_create_worktree_rejects_uuid1(self):
        result = wt.create_worktree(str(uuid.uuid1()), "main")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_INVALID_TASK_ID)

    def test_remove_worktree_rejects_invalid_task_id(self):
        result = wt.remove_worktree("not-a-uuid")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_INVALID_TASK_ID)

    def test_get_worktree_info_rejects_invalid_task_id(self):
        result = wt.get_worktree_info("not-a-uuid")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_INVALID_TASK_ID)

    def test_get_worktree_info_rejects_uuid1(self):
        result = wt.get_worktree_info(str(uuid.uuid1()))
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_INVALID_TASK_ID)

    def test_get_worktree_info_for_never_created_task_is_not_found(self):
        task_id = self._new_task_id()
        info = wt.get_worktree_info(task_id).info
        self.assertEqual(info.consistency, wt.CONSISTENCY_NOT_FOUND)
        self.assertFalse(info.path_entry_exists)
        self.assertFalse(info.branch_exists)
        self.assertFalse(info.registered_in_git_worktree)

    # --- branch name derivation: full UUID, not a truncated prefix ---

    def test_branch_name_uses_full_uuid(self):
        task_id = self._new_task_id()
        self.assertEqual(wt._branch_name_for(task_id), f"dev-task/{task_id}")

    def test_two_uuid4_with_same_prefix_derive_different_branches(self):
        first = uuid.uuid4()
        second_int = (first.int & (0xFFFFFFFF << 96)) | (uuid.uuid4().int & ((1 << 96) - 1))
        second = uuid.UUID(int=second_int, version=4)

        first_str, second_str = str(first), str(second)
        self.assertEqual(first_str[:8], second_str[:8])
        self.assertNotEqual(first_str, second_str)
        self.assertNotEqual(wt._branch_name_for(first_str), wt._branch_name_for(second_str))

    # --- 6/7: base_branch validation ---

    def test_validate_base_branch_accepts_existing_branch(self):
        result = wt.validate_base_branch("main")
        self.assertTrue(result.success)

    def test_validate_base_branch_rejects_bool(self):
        result = wt.validate_base_branch(True)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_INVALID_BASE_BRANCH_TYPE)

    def test_validate_base_branch_rejects_non_str(self):
        result = wt.validate_base_branch(123)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_INVALID_BASE_BRANCH_TYPE)

    def test_validate_base_branch_rejects_empty_string(self):
        result = wt.validate_base_branch("")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_INVALID_BASE_BRANCH_FORMAT)

    def test_validate_base_branch_rejects_whitespace_only(self):
        result = wt.validate_base_branch("   \t  ")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_INVALID_BASE_BRANCH_FORMAT)

    def test_validate_base_branch_rejects_control_character(self):
        result = wt.validate_base_branch("main\x01evil")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_INVALID_BASE_BRANCH_FORMAT)

    def test_validate_base_branch_rejects_newline(self):
        result = wt.validate_base_branch("main\nrm -rf /")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_INVALID_BASE_BRANCH_FORMAT)

    def test_validate_base_branch_rejects_leading_hyphen(self):
        result = wt.validate_base_branch("-main")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_INVALID_BASE_BRANCH_FORMAT)

    def test_validate_base_branch_rejects_double_dot(self):
        result = wt.validate_base_branch("main..evil")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_INVALID_BASE_BRANCH_FORMAT)

    def test_validate_base_branch_rejects_at_brace(self):
        result = wt.validate_base_branch("main@{1}")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_INVALID_BASE_BRANCH_FORMAT)

    def test_validate_base_branch_rejects_backslash(self):
        result = wt.validate_base_branch("main\\evil")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_INVALID_BASE_BRANCH_FORMAT)

    def test_validate_base_branch_rejects_invalid_ref_format(self):
        result = wt.validate_base_branch("main branch")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_INVALID_BASE_BRANCH_FORMAT)

    def test_validate_base_branch_rejects_nonexistent_branch(self):
        result = wt.validate_base_branch("does-not-exist")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_BASE_BRANCH_NOT_FOUND)

    def test_create_worktree_rejects_invalid_base_branch(self):
        result = wt.create_worktree(self._new_task_id(), "-evil")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_INVALID_BASE_BRANCH_FORMAT)

    def test_create_worktree_rejects_nonexistent_base_branch(self):
        result = wt.create_worktree(self._new_task_id(), "does-not-exist")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_BASE_BRANCH_NOT_FOUND)

    # --- 8/9/10: duplicate task_id / existing branch / existing path ---

    def test_create_worktree_rejects_duplicate_task_id(self):
        task_id = self._new_task_id()
        first = wt.create_worktree(task_id, "main")
        self.assertTrue(first.success)

        second = wt.create_worktree(task_id, "main")
        self.assertFalse(second.success)
        self.assertEqual(second.error_code, wt.ERROR_PATH_ALREADY_EXISTS)

    def test_create_worktree_rejects_existing_branch_name(self):
        task_id = self._new_task_id()
        branch_name = wt._branch_name_for(task_id)
        _git(["branch", branch_name, "main"], self.repo_root)

        result = wt.create_worktree(task_id, "main")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_BRANCH_ALREADY_EXISTS)

    def test_create_worktree_rejects_existing_path(self):
        task_id = self._new_task_id()
        worktree_path = wt._worktree_path_for(task_id)
        worktree_path.mkdir(parents=True)
        (worktree_path / "junk.txt").write_text("junk", encoding="utf-8")

        result = wt.create_worktree(task_id, "main")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_PATH_ALREADY_EXISTS)

    # --- 11/12: partial-failure semantics on removal ---

    def test_remove_worktree_failure_does_not_delete_branch(self):
        task_id = self._new_task_id()
        create_result = wt.create_worktree(task_id, "main")
        self.assertTrue(create_result.success)

        real_run_git = wt.run_git

        def flaky_run_git(args, **kwargs):
            if args[:2] == ["worktree", "remove"]:
                return dgr.GitRunResult(
                    success=False,
                    returncode=None,
                    stdout=b"",
                    stderr=b"",
                    stdout_truncated=False,
                    stderr_truncated=False,
                    error_code=dgr.ERROR_GIT_TIMEOUT,
                )
            return real_run_git(args, **kwargs)

        with mock.patch("services.dev_worktree.run_git", side_effect=flaky_run_git):
            result = wt.remove_worktree(task_id)

        self.assertFalse(result.success)
        self.assertFalse(result.worktree_removed)
        self.assertFalse(result.branch_removed)
        self.assertEqual(result.error_code, wt.ERROR_GIT_TIMEOUT)

        info = wt.get_worktree_info(task_id).info
        self.assertTrue(info.path_entry_exists)
        self.assertTrue(info.branch_exists)
        self.assertTrue(info.registered_in_git_worktree)

    def test_remove_worktree_partial_failure_when_branch_delete_fails(self):
        task_id = self._new_task_id()
        create_result = wt.create_worktree(task_id, "main")
        self.assertTrue(create_result.success)
        branch_name = create_result.branch_name
        worktree_path = create_result.worktree_path

        real_run_git = wt.run_git

        def flaky_run_git(args, **kwargs):
            if args[:2] == ["branch", "-D"]:
                return dgr.GitRunResult(
                    success=False,
                    returncode=1,
                    stdout=b"",
                    stderr=b"simulated failure",
                    stdout_truncated=False,
                    stderr_truncated=False,
                    error_code=dgr.ERROR_GIT_COMMAND_FAILED,
                )
            return real_run_git(args, **kwargs)

        with mock.patch("services.dev_worktree.run_git", side_effect=flaky_run_git):
            result = wt.remove_worktree(task_id)

        self.assertFalse(result.success)
        self.assertTrue(result.worktree_removed)
        self.assertFalse(result.branch_removed)
        self.assertEqual(result.error_code, wt.ERROR_BRANCH_REMOVE_FAILED)

        self.assertFalse(worktree_path.exists())
        self.assertTrue(_branch_exists_real(branch_name, self.repo_root))

    def test_remove_worktree_not_found_when_nothing_exists(self):
        result = wt.remove_worktree(self._new_task_id())
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_WORKTREE_NOT_FOUND)
        self.assertFalse(result.worktree_removed)
        self.assertFalse(result.branch_removed)

    # --- 13/14: git-not-found / timeout / cwd-missing safety ---

    def test_create_worktree_handles_git_not_found(self):
        with mock.patch("services.dev_worktree.run_git", return_value=_git_not_found_result()):
            result = wt.create_worktree(self._new_task_id(), "main")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_GIT_NOT_FOUND)

    def test_remove_worktree_handles_git_not_found(self):
        with mock.patch("services.dev_worktree.run_git", return_value=_git_not_found_result()):
            result = wt.remove_worktree(self._new_task_id())
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_GIT_NOT_FOUND)

    def test_validate_base_branch_handles_git_not_found(self):
        with mock.patch("services.dev_worktree.run_git", return_value=_git_not_found_result()):
            result = wt.validate_base_branch("main")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_GIT_NOT_FOUND)

    def test_create_worktree_handles_timeout_on_worktree_add(self):
        task_id = self._new_task_id()
        worktree_path = wt._worktree_path_for(task_id)
        branch_name = wt._branch_name_for(task_id)
        real_run_git = wt.run_git

        def flaky_run_git(args, **kwargs):
            if args[:2] == ["worktree", "add"]:
                return dgr.GitRunResult(
                    success=False,
                    returncode=None,
                    stdout=b"",
                    stderr=b"",
                    stdout_truncated=False,
                    stderr_truncated=False,
                    error_code=dgr.ERROR_GIT_TIMEOUT,
                )
            return real_run_git(args, **kwargs)

        with mock.patch("services.dev_worktree.run_git", side_effect=flaky_run_git):
            result = wt.create_worktree(task_id, "main")

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_GIT_TIMEOUT)
        self.assertFalse(worktree_path.exists())
        self.assertFalse(_branch_exists_real(branch_name, self.repo_root))
        self.assertIsNotNone(result.cleanup)
        self.assertFalse(result.cleanup.manual_cleanup_required)
        self.assertFalse(result.cleanup.automatic_cleanup_performed)

    def test_create_worktree_distinguishes_missing_cwd_from_missing_git(self):
        task_id = self._new_task_id()
        # Rename rather than shutil.rmtree(): git marks some object files
        # read-only, which shutil.rmtree() can't delete on Windows without
        # extra onerror handling; renaming away is enough to make REPO_ROOT
        # (still pointing at the old path) stop existing.
        self.repo_root.rename(self.repo_root.parent / "moved-away")

        result = wt.create_worktree(task_id, "main")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_CWD_NOT_FOUND)

    def test_get_worktree_info_propagates_infra_failure_instead_of_reporting_not_found(self):
        task_id = self._new_task_id()
        with mock.patch(
            "services.dev_worktree.check_ref_exists",
            return_value=dgr.RefCheckResult(exists=None, error_code=dgr.ERROR_GIT_TIMEOUT),
        ):
            result = wt.get_worktree_info(task_id)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_GIT_TIMEOUT)

    # --- truncated porcelain output must never be treated as valid data ---

    def test_get_worktree_info_rejects_truncated_porcelain_output(self):
        task_id = self._new_task_id()
        with mock.patch(
            "services.dev_worktree.run_git",
            return_value=dgr.GitRunResult(
                success=True,
                returncode=0,
                stdout=b"worktree /repo\0\0",
                stderr=b"",
                stdout_truncated=True,
                stderr_truncated=False,
                error_code=None,
            ),
        ):
            result = wt.get_worktree_info(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_GIT_OUTPUT_TRUNCATED)

    def test_get_worktree_info_does_not_misreport_unregistered_when_output_truncated(self):
        # A realistic truncated stream where the target task's own record
        # would have appeared right after the cut - this must never be read
        # as "not registered" (which a naive "just try to parse what we
        # have" approach would risk doing for a foreign worktree too).
        task_id = self._new_task_id()
        truncated_stdout = b"worktree /repo\0HEAD abc\0branch refs/heads/main\0\0worktree /repo/work/"
        with mock.patch(
            "services.dev_worktree.run_git",
            return_value=dgr.GitRunResult(
                success=True,
                returncode=0,
                stdout=truncated_stdout,
                stderr=b"",
                stdout_truncated=True,
                stderr_truncated=False,
                error_code=None,
            ),
        ):
            result = wt.get_worktree_info(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_GIT_OUTPUT_TRUNCATED)

    def test_create_worktree_pre_check_fails_safe_on_truncated_porcelain(self):
        task_id = self._new_task_id()
        with mock.patch(
            "services.dev_worktree.run_git",
            return_value=dgr.GitRunResult(
                success=True,
                returncode=0,
                stdout=b"worktree /repo\0\0",
                stderr=b"",
                stdout_truncated=True,
                stderr_truncated=False,
                error_code=None,
            ),
        ):
            result = wt.create_worktree(task_id, "main")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_GIT_OUTPUT_TRUNCATED)

    def test_remove_worktree_pre_check_fails_safe_on_truncated_porcelain(self):
        task_id = self._new_task_id()
        with mock.patch(
            "services.dev_worktree.run_git",
            return_value=dgr.GitRunResult(
                success=True,
                returncode=0,
                stdout=b"worktree /repo\0\0",
                stderr=b"",
                stdout_truncated=True,
                stderr_truncated=False,
                error_code=None,
            ),
        ):
            result = wt.remove_worktree(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_GIT_OUTPUT_TRUNCATED)

    # --- get_worktree_info / consistency classification ---

    def test_get_worktree_info_uses_real_porcelain_output(self):
        task_id = self._new_task_id()
        create_result = wt.create_worktree(task_id, "main")
        self.assertTrue(create_result.success)

        info = wt.get_worktree_info(task_id).info
        self.assertTrue(info.registered_in_git_worktree)
        self.assertTrue(info.path_entry_exists)
        self.assertTrue(info.branch_exists)
        self.assertFalse(info.branch_mismatch)
        self.assertEqual(info.consistency, wt.CONSISTENCY_CONSISTENT)

    def test_classify_consistency_all_combinations(self):
        cases = [
            (False, False, False, False, wt.CONSISTENCY_NOT_FOUND),
            (True, True, True, False, wt.CONSISTENCY_CONSISTENT),
            (False, True, True, False, wt.CONSISTENCY_MISSING_PATH),
            (False, True, False, False, wt.CONSISTENCY_MISSING_PATH),
            (True, False, True, False, wt.CONSISTENCY_MISSING_REGISTRATION),
            (False, False, True, False, wt.CONSISTENCY_ORPHAN_BRANCH),
            (True, True, False, False, wt.CONSISTENCY_PARTIAL_STATE),
            (True, False, False, False, wt.CONSISTENCY_PARTIAL_STATE),
            (True, True, True, True, wt.CONSISTENCY_FOREIGN_WORKTREE),
            (False, True, False, True, wt.CONSISTENCY_FOREIGN_WORKTREE),
        ]
        for path_entry_exists, registered, branch_exists, branch_mismatch, expected in cases:
            with self.subTest(
                path_entry_exists=path_entry_exists,
                registered=registered,
                branch_exists=branch_exists,
                branch_mismatch=branch_mismatch,
            ):
                self.assertEqual(
                    wt._classify_consistency(
                        path_entry_exists=path_entry_exists,
                        registered=registered,
                        branch_exists=branch_exists,
                        branch_mismatch=branch_mismatch,
                    ),
                    expected,
                )

    def test_get_worktree_info_detects_orphan_branch(self):
        task_id = self._new_task_id()
        branch_name = wt._branch_name_for(task_id)
        _git(["branch", branch_name, "main"], self.repo_root)

        info = wt.get_worktree_info(task_id).info
        self.assertTrue(info.branch_exists)
        self.assertFalse(info.path_entry_exists)
        self.assertFalse(info.registered_in_git_worktree)
        self.assertEqual(info.consistency, wt.CONSISTENCY_ORPHAN_BRANCH)

    def test_get_worktree_info_detects_partial_state_path_only(self):
        task_id = self._new_task_id()
        worktree_path = wt._worktree_path_for(task_id)
        worktree_path.mkdir(parents=True)
        (worktree_path / "stray.txt").write_text("stray", encoding="utf-8")

        info = wt.get_worktree_info(task_id).info
        self.assertTrue(info.path_entry_exists)
        self.assertFalse(info.branch_exists)
        self.assertFalse(info.registered_in_git_worktree)
        self.assertEqual(info.consistency, wt.CONSISTENCY_PARTIAL_STATE)

    def test_get_worktree_info_detects_missing_path(self):
        task_id = self._new_task_id()
        create_result = wt.create_worktree(task_id, "main")
        self.assertTrue(create_result.success)

        shutil.rmtree(create_result.worktree_path)

        info = wt.get_worktree_info(task_id).info
        self.assertFalse(info.path_entry_exists)
        self.assertTrue(info.registered_in_git_worktree)
        self.assertEqual(info.consistency, wt.CONSISTENCY_MISSING_PATH)

    # --- foreign worktree protection: never touch someone else's worktree ---

    def test_remove_worktree_refuses_when_foreign_branch_registered_at_path(self):
        task_id = self._new_task_id()
        worktree_path = wt._worktree_path_for(task_id)
        foreign_branch = "foreign/other-branch"

        _git(["worktree", "add", "-b", foreign_branch, str(worktree_path), "main"], self.repo_root)

        info = wt.get_worktree_info(task_id).info
        self.assertTrue(info.registered_in_git_worktree)
        self.assertTrue(info.branch_mismatch)
        self.assertEqual(info.consistency, wt.CONSISTENCY_FOREIGN_WORKTREE)

        result = wt.remove_worktree(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_FOREIGN_WORKTREE)
        self.assertFalse(result.worktree_removed)
        self.assertFalse(result.branch_removed)

        self.assertTrue(worktree_path.exists())
        self.assertTrue(_branch_exists_real(foreign_branch, self.repo_root))
        post_info = wt.get_worktree_info(task_id).info
        self.assertTrue(post_info.registered_in_git_worktree)
        self.assertTrue(post_info.branch_mismatch)

    def test_remove_worktree_refuses_when_detached_worktree_registered_at_path(self):
        task_id = self._new_task_id()
        worktree_path = wt._worktree_path_for(task_id)

        _git(["worktree", "add", "--detach", str(worktree_path), "main"], self.repo_root)

        info = wt.get_worktree_info(task_id).info
        self.assertTrue(info.registered_detached)
        self.assertTrue(info.branch_mismatch)

        result = wt.remove_worktree(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_FOREIGN_WORKTREE)
        self.assertTrue(worktree_path.exists())

    def test_remove_worktree_does_not_touch_unrelated_worktree(self):
        control_id = self._new_task_id()
        control_result = wt.create_worktree(control_id, "main")
        self.assertTrue(control_result.success)

        target_id = self._new_task_id()
        target_result = wt.create_worktree(target_id, "main")
        self.assertTrue(target_result.success)

        removal = wt.remove_worktree(target_id)
        self.assertTrue(removal.success)

        control_info = wt.get_worktree_info(control_id).info
        self.assertTrue(control_info.path_entry_exists)
        self.assertTrue(control_info.branch_exists)
        self.assertTrue(control_info.registered_in_git_worktree)
        self.assertEqual(control_info.consistency, wt.CONSISTENCY_CONSISTENT)

    # --- boundary re-check immediately before `git worktree remove` ---

    def test_remove_worktree_boundary_recheck_blocks_removal_and_never_calls_git(self):
        task_id = self._new_task_id()
        create_result = wt.create_worktree(task_id, "main")
        self.assertTrue(create_result.success)
        worktree_path = create_result.worktree_path

        real_run_git = wt.run_git

        def forbidden_remove(args, **kwargs):
            if args[:2] == ["worktree", "remove"]:
                raise AssertionError("git worktree remove must not be called once the boundary re-check fails")
            return real_run_git(args, **kwargs)

        call_count = {"n": 0}
        real_boundary_check = wt._validate_path_boundary

        def flaky_boundary_check(target_path):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return real_boundary_check(target_path)
            return wt._validation_error(wt.ERROR_PATH_BOUNDARY_VIOLATION)

        with mock.patch("services.dev_worktree._validate_path_boundary", side_effect=flaky_boundary_check), \
                mock.patch("services.dev_worktree.run_git", side_effect=forbidden_remove):
            result = wt.remove_worktree(task_id)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_PATH_BOUNDARY_VIOLATION)
        self.assertEqual(call_count["n"], 2)
        self.assertTrue(worktree_path.exists())

    def test_remove_worktree_rejects_when_work_root_is_a_junction(self):
        if os.name != "nt":
            self.skipTest("junction test is Windows-specific")

        task_id = self._new_task_id()
        create_result = wt.create_worktree(task_id, "main")
        self.assertTrue(create_result.success)

        # Directly exercise the boundary check remove_worktree() runs,
        # against a real junction, without needing to swap out a live
        # worktree directory (git holds that directory open as a real
        # directory; turning it into a junction mid-test would require
        # removing it first, defeating the point of the scenario).
        real_dir = self.repo_root.parent / "real_target_remove"
        real_dir.mkdir()
        work_root = self.repo_root / wt.WORK_DIR_NAME
        # Use a fresh, not-yet-existing WORK_ROOT-like path for the junction
        # check itself (the unit under test is _validate_path_boundary()).
        junction_path = self.repo_root.parent / "junction_work_root"
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(junction_path), str(real_dir)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            self.skipTest(f"could not create an NTFS junction in this environment: {proc.stderr}")

        with mock.patch("services.dev_worktree._work_root", return_value=junction_path):
            result = wt._validate_path_boundary(junction_path / task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_PATH_BOUNDARY_VIOLATION)

    # --- per-task_id in-process lock serializes create/remove, and its
    # dict entries do not accumulate forever ---

    def test_create_and_remove_same_task_id_are_serialized_by_in_process_lock(self):
        task_id = self._new_task_id()
        entered_event = threading.Event()
        release_event = threading.Event()

        def hold_lock():
            with wt._task_lock(task_id):
                entered_event.set()
                release_event.wait(timeout=10)

        holder_thread = threading.Thread(target=hold_lock)
        holder_thread.start()
        self.assertTrue(entered_event.wait(timeout=5))

        result_holder = {}

        def run_create():
            result_holder["result"] = wt.create_worktree(task_id, "main")

        create_thread = threading.Thread(target=run_create)
        create_thread.start()
        create_thread.join(timeout=0.5)
        # Still blocked: create_worktree() must not proceed while the
        # task's lock is held elsewhere.
        self.assertTrue(create_thread.is_alive())

        release_event.set()
        holder_thread.join(timeout=5)
        create_thread.join(timeout=10)
        self.assertFalse(create_thread.is_alive())
        self.assertTrue(result_holder["result"].success)

    def test_different_task_ids_do_not_block_each_other(self):
        task_id_a = self._new_task_id()
        task_id_b = self._new_task_id()

        entered_a = threading.Event()
        release_a = threading.Event()

        def hold_a():
            with wt._task_lock(task_id_a):
                entered_a.set()
                release_a.wait(timeout=10)

        thread_a = threading.Thread(target=hold_a)
        thread_a.start()
        self.assertTrue(entered_a.wait(timeout=5))

        entered_b = threading.Event()

        def hold_b():
            with wt._task_lock(task_id_b):
                entered_b.set()

        thread_b = threading.Thread(target=hold_b)
        thread_b.start()
        # A different task_id must proceed immediately, without waiting on
        # task_id_a's lock.
        self.assertTrue(entered_b.wait(timeout=2))
        thread_b.join(timeout=5)

        release_a.set()
        thread_a.join(timeout=5)

    def test_task_lock_dict_does_not_grow_unbounded(self):
        for _ in range(50):
            task_id = self._new_task_id()
            with wt._task_lock(task_id):
                pass
        self.assertEqual(len(wt._task_locks), 0)

    def test_task_lock_cleans_up_after_exception(self):
        task_id = self._new_task_id()
        with self.assertRaises(RuntimeError):
            with wt._task_lock(task_id):
                raise RuntimeError("boom")
        self.assertEqual(len(wt._task_locks), 0)

    def test_task_lock_reused_while_held_does_not_get_deleted_early(self):
        task_id = self._new_task_id()
        entered_event = threading.Event()
        release_event = threading.Event()

        def hold_lock():
            with wt._task_lock(task_id):
                entered_event.set()
                release_event.wait(timeout=10)

        holder_thread = threading.Thread(target=hold_lock)
        holder_thread.start()
        self.assertTrue(entered_event.wait(timeout=5))

        # While the lock is held, the dict entry must still be present (a
        # second waiter incrementing its ref count must see it, not create
        # a second, unrelated Lock for the same task_id).
        self.assertIn(task_id, wt._task_locks)

        release_event.set()
        holder_thread.join(timeout=5)
        self.assertNotIn(task_id, wt._task_locks)

    # --- 修正必須1: same-task_id race - the loser must never touch the winner ---

    def test_concurrent_create_same_task_id_loser_never_touches_winner(self):
        task_id = self._new_task_id()
        worktree_path = wt._worktree_path_for(task_id)
        branch_name = wt._branch_name_for(task_id)

        winner_result = wt.create_worktree(task_id, "main")
        self.assertTrue(winner_result.success)
        winner_file = worktree_path / "winner.txt"
        winner_file.write_text("winner", encoding="utf-8")

        # Simulate the loser of a same-task_id race: its own pre-check ran
        # before the winner committed anything (a TOCTOU gap), so it
        # proceeds to attempt `git worktree add` for the exact same
        # path/branch, which now genuinely fails because the winner already
        # claimed both.
        fake = _single_shot_bypass(task_id, worktree_path, branch_name)
        with mock.patch("services.dev_worktree.get_worktree_info", side_effect=fake):
            loser_result = wt.create_worktree(task_id, "main")

        self.assertFalse(loser_result.success)
        self.assertIsNotNone(loser_result.cleanup)
        self.assertFalse(loser_result.cleanup.automatic_cleanup_performed)
        self.assertFalse(loser_result.cleanup.ownership_verified)
        self.assertTrue(loser_result.cleanup.manual_cleanup_required)

        # The winner's path, file, branch, and registration are all
        # untouched by the loser's failure handling.
        self.assertTrue(worktree_path.exists())
        self.assertTrue(winner_file.exists())
        self.assertEqual(winner_file.read_text(encoding="utf-8"), "winner")
        self.assertTrue(_branch_exists_real(branch_name, self.repo_root))
        winner_info = wt.get_worktree_info(task_id).info
        self.assertTrue(winner_info.registered_in_git_worktree)
        self.assertEqual(winner_info.consistency, wt.CONSISTENCY_CONSISTENT)

    # --- ownership-safe create-failure cleanup: zero automatic deletion ---

    def test_create_worktree_cleanup_never_deletes_preexisting_content(self):
        control_id = self._new_task_id()
        control_result = wt.create_worktree(control_id, "main")
        self.assertTrue(control_result.success)

        task_id = self._new_task_id()
        worktree_path = wt._worktree_path_for(task_id)
        branch_name = wt._branch_name_for(task_id)
        worktree_path.mkdir(parents=True)
        occupied_file = worktree_path / "occupied.txt"
        occupied_file.write_text("occupied", encoding="utf-8")

        fake = _single_shot_bypass(task_id, worktree_path, branch_name)
        with mock.patch("services.dev_worktree.get_worktree_info", side_effect=fake):
            result = wt.create_worktree(task_id, "main")

        self.assertFalse(result.success)

        # The pre-existing directory and its content must survive: Phase 2
        # never auto-deletes on create failure, period.
        self.assertTrue(worktree_path.exists())
        self.assertTrue(occupied_file.exists())
        self.assertEqual(occupied_file.read_text(encoding="utf-8"), "occupied")

        self.assertIsNotNone(result.cleanup)
        self.assertFalse(result.cleanup.automatic_cleanup_performed)
        self.assertFalse(result.cleanup.ownership_verified)
        self.assertTrue(result.cleanup.manual_cleanup_required)
        # The branch git itself created before failing on the occupied path
        # is left in place too - no automatic `git branch -D` in Phase 2.
        self.assertTrue(result.cleanup.branch_exists_after)
        self.assertTrue(_branch_exists_real(branch_name, self.repo_root))

        control_info = wt.get_worktree_info(control_id).info
        self.assertTrue(control_info.path_entry_exists)
        self.assertTrue(control_info.branch_exists)
        self.assertTrue(control_info.registered_in_git_worktree)
        self.assertEqual(control_info.consistency, wt.CONSISTENCY_CONSISTENT)

    def test_create_worktree_never_auto_deletes_even_when_registration_matches_exactly(self):
        # The scenario the previous (now-removed) "ownership provable"
        # design got wrong: post-failure state that looks exactly like this
        # attempt's own successful registration is NOT proof this attempt
        # made it - Phase 2 must never delete it regardless.
        task_id = self._new_task_id()
        worktree_path = wt._worktree_path_for(task_id)
        branch_name = wt._branch_name_for(task_id)

        _git(["worktree", "add", "-b", branch_name, str(worktree_path), "main"], self.repo_root)

        cleanup = wt._cleanup_failed_create(task_id, worktree_path, branch_name)

        self.assertFalse(cleanup.automatic_cleanup_performed)
        self.assertFalse(cleanup.ownership_verified)
        self.assertTrue(cleanup.path_entry_exists_after)
        self.assertTrue(cleanup.registration_exists_after)
        self.assertTrue(cleanup.branch_exists_after)
        # Residue present -> manual cleanup required, per the invariant.
        self.assertTrue(cleanup.manual_cleanup_required)

        self.assertTrue(worktree_path.exists())
        self.assertTrue(_branch_exists_real(branch_name, self.repo_root))

    def test_create_worktree_failure_cleanup_never_calls_prune(self):
        task_id = self._new_task_id()
        worktree_path = wt._worktree_path_for(task_id)
        branch_name = wt._branch_name_for(task_id)
        worktree_path.mkdir(parents=True)
        (worktree_path / "occupied.txt").write_text("occupied", encoding="utf-8")

        fake = _single_shot_bypass(task_id, worktree_path, branch_name)
        real_run_git = wt.run_git
        prune_calls = []

        def spy_run_git(args, **kwargs):
            if args[:2] == ["worktree", "prune"]:
                prune_calls.append(args)
            return real_run_git(args, **kwargs)

        with mock.patch("services.dev_worktree.get_worktree_info", side_effect=fake), \
                mock.patch("services.dev_worktree.run_git", side_effect=spy_run_git):
            wt.create_worktree(task_id, "main")

        self.assertEqual(prune_calls, [])

    def test_create_worktree_cleanup_after_failed_add_does_not_touch_unrelated_worktree(self):
        control_id = self._new_task_id()
        control_result = wt.create_worktree(control_id, "main")
        self.assertTrue(control_result.success)

        task_id = self._new_task_id()
        worktree_path = wt._worktree_path_for(task_id)
        branch_name = wt._branch_name_for(task_id)
        worktree_path.mkdir(parents=True)
        (worktree_path / "occupied.txt").write_text("occupied", encoding="utf-8")

        fake = _single_shot_bypass(task_id, worktree_path, branch_name)
        with mock.patch("services.dev_worktree.get_worktree_info", side_effect=fake):
            result = wt.create_worktree(task_id, "main")

        self.assertFalse(result.success)

        control_info = wt.get_worktree_info(control_id).info
        self.assertTrue(control_info.path_entry_exists)
        self.assertTrue(control_info.branch_exists)
        self.assertTrue(control_info.registered_in_git_worktree)
        self.assertEqual(control_info.consistency, wt.CONSISTENCY_CONSISTENT)

    # --- CreateCleanupResult invariants ---

    def test_cleanup_result_invariant_no_manual_required_implies_confirmed_clean(self):
        task_id = self._new_task_id()
        worktree_path = wt._worktree_path_for(task_id)
        branch_name = wt._branch_name_for(task_id)

        cleanup = wt._cleanup_failed_create(task_id, worktree_path, branch_name)
        self.assertFalse(cleanup.manual_cleanup_required)
        self.assertFalse(cleanup.path_entry_exists_after)
        self.assertFalse(cleanup.registration_exists_after)
        self.assertFalse(cleanup.branch_exists_after)
        self.assertFalse(cleanup.automatic_cleanup_performed)
        self.assertFalse(cleanup.ownership_verified)
        self.assertIsNone(cleanup.cleanup_error_code)

    def test_cleanup_result_manual_required_when_post_check_state_unknown(self):
        task_id = self._new_task_id()
        worktree_path = wt._worktree_path_for(task_id)
        branch_name = wt._branch_name_for(task_id)

        with mock.patch(
            "services.dev_worktree.get_worktree_info",
            return_value=wt.WorktreeInfoResult(
                success=False, info=None, error_message="x", error_code=wt.ERROR_GIT_TIMEOUT
            ),
        ):
            cleanup = wt._cleanup_failed_create(task_id, worktree_path, branch_name)

        self.assertTrue(cleanup.manual_cleanup_required)
        self.assertIsNone(cleanup.path_entry_exists_after)
        self.assertIsNone(cleanup.registration_exists_after)
        self.assertIsNone(cleanup.branch_exists_after)
        self.assertFalse(cleanup.automatic_cleanup_performed)
        self.assertFalse(cleanup.ownership_verified)
        self.assertEqual(cleanup.cleanup_error_code, wt.ERROR_GIT_TIMEOUT)

    def test_cleanup_result_manual_required_when_any_residue_present(self):
        task_id = self._new_task_id()
        worktree_path = wt._worktree_path_for(task_id)
        branch_name = wt._branch_name_for(task_id)
        worktree_path.mkdir(parents=True)

        cleanup = wt._cleanup_failed_create(task_id, worktree_path, branch_name)
        self.assertTrue(cleanup.path_entry_exists_after)
        self.assertTrue(cleanup.manual_cleanup_required)
        self.assertFalse(cleanup.automatic_cleanup_performed)

    # --- symlink/junction/reparse-point boundary protection ---

    def test_create_worktree_rejects_junction_between_work_root_and_target(self):
        if os.name != "nt":
            self.skipTest("junction test is Windows-specific")

        real_dir = self.repo_root.parent / "real_target"
        real_dir.mkdir()
        work_root = self.repo_root / wt.WORK_DIR_NAME

        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(work_root), str(real_dir)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            self.skipTest(f"could not create an NTFS junction in this environment: {proc.stderr}")

        result = wt.create_worktree(self._new_task_id(), "main")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_PATH_BOUNDARY_VIOLATION)

    def test_create_worktree_rejects_symlinked_work_root_when_symlinks_supported(self):
        work_root = self.repo_root / wt.WORK_DIR_NAME
        real_dir = self.repo_root.parent / "real_target_symlink"
        real_dir.mkdir()

        try:
            work_root.symlink_to(real_dir, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation is not permitted in this environment")

        result = wt.create_worktree(self._new_task_id(), "main")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_PATH_BOUNDARY_VIOLATION)

    def test_is_reparse_point_false_for_plain_directory(self):
        plain_dir = self.repo_root / "plain"
        plain_dir.mkdir()
        self.assertFalse(wt._is_reparse_point(plain_dir))

    def test_validate_path_boundary_allows_normal_fresh_target(self):
        task_id = self._new_task_id()
        target = wt._worktree_path_for(task_id)
        result = wt._validate_path_boundary(target)
        self.assertTrue(result.success)

    # --- 修正必須1: the TARGET path itself (not just WORK_ROOT/ancestors)
    # must be rejected when it is itself a reparse point ---

    def test_validate_path_boundary_rejects_when_target_itself_is_a_junction(self):
        if os.name != "nt":
            self.skipTest("junction test is Windows-specific")

        task_id = self._new_task_id()
        worktree_path = wt._worktree_path_for(task_id)
        worktree_path.parent.mkdir(parents=True, exist_ok=True)

        real_dir = self.repo_root.parent / "foreign_target_for_uuid_junction"
        real_dir.mkdir()
        foreign_file = real_dir / "foreign.txt"
        foreign_file.write_text("foreign data", encoding="utf-8")

        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(worktree_path), str(real_dir)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            self.skipTest(f"could not create an NTFS junction in this environment: {proc.stderr}")

        result = wt._validate_path_boundary(worktree_path)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_PATH_BOUNDARY_VIOLATION)

        create_result = wt.create_worktree(task_id, "main")
        self.assertFalse(create_result.success)
        self.assertTrue(foreign_file.exists())
        self.assertEqual(foreign_file.read_text(encoding="utf-8"), "foreign data")

    def test_remove_worktree_rejects_and_never_calls_git_remove_when_target_itself_is_a_junction(self):
        if os.name != "nt":
            self.skipTest("junction test is Windows-specific")

        task_id = self._new_task_id()
        worktree_path = wt._worktree_path_for(task_id)
        worktree_path.parent.mkdir(parents=True, exist_ok=True)

        real_dir = self.repo_root.parent / "foreign_target_for_remove_junction"
        real_dir.mkdir()
        foreign_file = real_dir / "foreign.txt"
        foreign_file.write_text("foreign data", encoding="utf-8")

        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(worktree_path), str(real_dir)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            self.skipTest(f"could not create an NTFS junction in this environment: {proc.stderr}")

        real_run_git = wt.run_git

        def forbidden_remove(args, **kwargs):
            if args[:2] == ["worktree", "remove"]:
                raise AssertionError(
                    "git worktree remove must not be called when the target itself is a reparse point"
                )
            return real_run_git(args, **kwargs)

        with mock.patch("services.dev_worktree.run_git", side_effect=forbidden_remove):
            result = wt.remove_worktree(task_id)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_PATH_BOUNDARY_VIOLATION)

        # The junction and the foreign directory/file it points at survive.
        self.assertTrue(worktree_path.exists())
        self.assertTrue(foreign_file.exists())
        self.assertEqual(foreign_file.read_text(encoding="utf-8"), "foreign data")

    # --- 修正必須2: lexists-based residue detection (broken links/junctions
    # are still "something is there", even though Path.exists() is False) ---

    def test_path_entry_exists_true_for_broken_junction_while_exists_is_false(self):
        if os.name != "nt":
            self.skipTest("junction test is Windows-specific")

        task_id = self._new_task_id()
        worktree_path = wt._worktree_path_for(task_id)
        worktree_path.parent.mkdir(parents=True, exist_ok=True)

        real_dir = self.repo_root.parent / "target_to_be_deleted"
        real_dir.mkdir()
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(worktree_path), str(real_dir)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            self.skipTest(f"could not create an NTFS junction in this environment: {proc.stderr}")

        real_dir.rmdir()  # break the junction: its target no longer exists

        self.assertFalse(worktree_path.exists())
        self.assertTrue(wt._path_entry_exists(worktree_path))

    def test_path_entry_exists_true_for_broken_symlink_when_supported(self):
        task_id = self._new_task_id()
        worktree_path = wt._worktree_path_for(task_id)
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        real_dir = self.repo_root.parent / "symlink_target_to_delete"
        real_dir.mkdir()
        try:
            worktree_path.symlink_to(real_dir, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation is not permitted in this environment")

        real_dir.rmdir()

        self.assertFalse(worktree_path.exists())
        self.assertTrue(wt._path_entry_exists(worktree_path))

    def test_cleanup_result_manual_required_for_broken_junction_residue(self):
        if os.name != "nt":
            self.skipTest("junction test is Windows-specific")

        task_id = self._new_task_id()
        worktree_path = wt._worktree_path_for(task_id)
        branch_name = wt._branch_name_for(task_id)
        worktree_path.parent.mkdir(parents=True, exist_ok=True)

        real_dir = self.repo_root.parent / "target_to_be_deleted_for_cleanup"
        real_dir.mkdir()
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(worktree_path), str(real_dir)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            self.skipTest(f"could not create an NTFS junction in this environment: {proc.stderr}")
        real_dir.rmdir()

        cleanup = wt._cleanup_failed_create(task_id, worktree_path, branch_name)
        # A broken junction is still residue - Path.exists() alone would
        # have missed it, reporting a false "all clean".
        self.assertTrue(cleanup.path_entry_exists_after)
        self.assertTrue(cleanup.manual_cleanup_required)

    # --- 修正必須3: `git worktree remove`/`git branch -D` reporting success
    # is not trusted on its own - always re-measured afterward ---

    def test_remove_worktree_verifies_after_apparent_success_and_catches_lingering_residue(self):
        task_id = self._new_task_id()
        create_result = wt.create_worktree(task_id, "main")
        self.assertTrue(create_result.success)
        worktree_path = create_result.worktree_path
        branch_name = create_result.branch_name
        marker_file = worktree_path / "external_marker.txt"
        marker_file.write_text("still here", encoding="utf-8")

        real_run_git = wt.run_git

        def lying_run_git(args, **kwargs):
            if args[:2] == ["worktree", "remove"]:
                # Simulate `git worktree remove --force` reporting success
                # (returncode 0) without the path/registration having
                # actually been removed - e.g. a junction or other reparse
                # point left a dangling entry behind despite git believing
                # it cleaned up.
                return dgr.GitRunResult(
                    success=True,
                    returncode=0,
                    stdout=b"",
                    stderr=b"",
                    stdout_truncated=False,
                    stderr_truncated=False,
                    error_code=None,
                )
            return real_run_git(args, **kwargs)

        with mock.patch("services.dev_worktree.run_git", side_effect=lying_run_git):
            result = wt.remove_worktree(task_id)

        self.assertFalse(result.success)
        self.assertFalse(result.worktree_removed)
        self.assertEqual(result.error_code, wt.ERROR_WORKTREE_REMOVE_INCOMPLETE)
        self.assertTrue(result.manual_cleanup_required)

        # Nothing was actually touched: the worktree, its external marker
        # file, and the branch all remain.
        self.assertTrue(worktree_path.exists())
        self.assertTrue(marker_file.exists())
        self.assertTrue(_branch_exists_real(branch_name, self.repo_root))

    def test_remove_worktree_manual_required_when_post_verification_state_unknown(self):
        task_id = self._new_task_id()
        create_result = wt.create_worktree(task_id, "main")
        self.assertTrue(create_result.success)

        real_get_worktree_info = wt.get_worktree_info
        calls = {"n": 0}

        def fake_get_worktree_info(tid):
            calls["n"] += 1
            # First call is remove_worktree()'s own pre-check (must behave
            # normally so the boundary/foreign-branch checks pass); the
            # second call is the post-removal verification - force it to
            # fail, simulating an infra hiccup right after the removal.
            if calls["n"] <= 1:
                return real_get_worktree_info(tid)
            return wt.WorktreeInfoResult(
                success=False, info=None, error_message="x", error_code=wt.ERROR_GIT_TIMEOUT
            )

        with mock.patch("services.dev_worktree.get_worktree_info", side_effect=fake_get_worktree_info):
            result = wt.remove_worktree(task_id)

        self.assertFalse(result.success)
        self.assertTrue(result.manual_cleanup_required)
        self.assertEqual(result.error_code, wt.ERROR_GIT_TIMEOUT)

    def test_remove_worktree_final_branch_check_catches_branch_still_existing(self):
        task_id = self._new_task_id()
        create_result = wt.create_worktree(task_id, "main")
        self.assertTrue(create_result.success)
        branch_name = create_result.branch_name

        real_run_git = wt.run_git

        def lying_branch_delete(args, **kwargs):
            if args[:2] == ["branch", "-D"]:
                # `git branch -D` claims success without the branch
                # actually being gone.
                return dgr.GitRunResult(
                    success=True,
                    returncode=0,
                    stdout=b"",
                    stderr=b"",
                    stdout_truncated=False,
                    stderr_truncated=False,
                    error_code=None,
                )
            return real_run_git(args, **kwargs)

        with mock.patch("services.dev_worktree.run_git", side_effect=lying_branch_delete):
            result = wt.remove_worktree(task_id)

        self.assertFalse(result.success)
        self.assertTrue(result.worktree_removed)
        self.assertFalse(result.branch_removed)
        self.assertTrue(result.manual_cleanup_required)
        self.assertEqual(result.error_code, wt.ERROR_BRANCH_REMOVE_FAILED)
        self.assertTrue(_branch_exists_real(branch_name, self.repo_root))

    # --- 20: prune does not disturb valid worktrees ---

    def test_cleanup_stale_worktree_metadata_preserves_valid_worktrees(self):
        valid_id = self._new_task_id()
        valid_result = wt.create_worktree(valid_id, "main")
        self.assertTrue(valid_result.success)

        stale_id = self._new_task_id()
        stale_result = wt.create_worktree(stale_id, "main")
        self.assertTrue(stale_result.success)
        shutil.rmtree(stale_result.worktree_path)

        prune_result = wt.cleanup_stale_worktree_metadata()
        self.assertTrue(prune_result.success)

        stale_info = wt.get_worktree_info(stale_id).info
        self.assertFalse(stale_info.registered_in_git_worktree)

        valid_info = wt.get_worktree_info(valid_id).info
        self.assertTrue(valid_info.path_entry_exists)
        self.assertTrue(valid_info.registered_in_git_worktree)
        self.assertTrue(valid_info.branch_exists)
        self.assertEqual(valid_info.consistency, wt.CONSISTENCY_CONSISTENT)

    def test_cleanup_stale_worktree_metadata_handles_git_not_found(self):
        with mock.patch("services.dev_worktree.run_git", return_value=_git_not_found_result()):
            result = wt.cleanup_stale_worktree_metadata()
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, wt.ERROR_GIT_NOT_FOUND)


if __name__ == "__main__":
    unittest.main()
