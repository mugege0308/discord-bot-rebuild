import asyncio
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import services.dev_task_executor as executor


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True, encoding="utf-8"
    )


class DevTaskExecutorTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="dev_task_executor_test ")
        self.worktree_path = Path(self._tmp_dir.name) / "repo"
        self.worktree_path.mkdir(parents=True)
        _git(["init", "-q", "-b", "main"], self.worktree_path)
        _git(["config", "user.name", "Test User"], self.worktree_path)
        _git(["config", "user.email", "test@example.com"], self.worktree_path)
        (self.worktree_path / "README.md").write_text("hello\n", encoding="utf-8")
        _git(["add", "README.md"], self.worktree_path)
        _git(["commit", "-q", "-m", "init"], self.worktree_path)

        # Keep the dummy sleep short so tests run fast.
        self._sleep_patch = mock.patch("services.dev_task_executor._DUMMY_EXECUTION_SECONDS", 0.05)
        self._sleep_patch.start()

    def tearDown(self):
        self._sleep_patch.stop()
        self._tmp_dir.cleanup()
        # Guard against any single test leaving the module-level execution
        # slot reserved (e.g. a test that cancels/replaces the running task
        # without going through run_dummy_execution's own release_slot()
        # finally-block) - that would otherwise leak into every later test
        # in this process as a spurious "is_busy()" true.
        executor._current_task_id = None
        executor._current_asyncio_task = None
        if executor._execution_lock.locked():
            executor._execution_lock.release()

    def _new_task_id(self) -> str:
        return str(uuid.uuid4())

    async def test_run_dummy_execution_succeeds_on_clean_repo(self):
        task_id = self._new_task_id()
        self.assertTrue(await executor.reserve_slot(task_id))
        result = await executor.run_dummy_execution(task_id, self.worktree_path, 600)

        self.assertTrue(result.success)
        self.assertEqual(result.changed_file_count, 0)
        self.assertIn("ダミー実行", result.diff_summary)
        self.assertIsNone(result.error_message)
        self.assertFalse(result.cancelled)
        self.assertGreaterEqual(result.duration_seconds, 0)

    async def test_run_dummy_execution_never_modifies_the_worktree(self):
        # Read-only guarantee: no files added, removed, or changed by the
        # dummy execution itself.
        before = sorted(p.relative_to(self.worktree_path) for p in self.worktree_path.rglob("*") if ".git" not in p.parts)
        task_id = self._new_task_id()
        self.assertTrue(await executor.reserve_slot(task_id))
        await executor.run_dummy_execution(task_id, self.worktree_path, 600)
        after = sorted(p.relative_to(self.worktree_path) for p in self.worktree_path.rglob("*") if ".git" not in p.parts)
        self.assertEqual(before, after)

        status = _git(["status", "--short"], self.worktree_path)
        self.assertEqual(status.stdout, "")

    async def test_run_dummy_execution_reports_uncommitted_changes(self):
        (self.worktree_path / "untracked.txt").write_text("someone else's change", encoding="utf-8")
        try:
            task_id = self._new_task_id()
            self.assertTrue(await executor.reserve_slot(task_id))
            result = await executor.run_dummy_execution(task_id, self.worktree_path, 600)
            self.assertTrue(result.success)
            self.assertEqual(result.changed_file_count, 1)
        finally:
            (self.worktree_path / "untracked.txt").unlink()

    async def test_run_dummy_execution_fails_safely_when_worktree_path_is_not_a_git_repo(self):
        not_a_repo = Path(self._tmp_dir.name) / "not-a-repo"
        not_a_repo.mkdir()
        task_id = self._new_task_id()
        self.assertTrue(await executor.reserve_slot(task_id))
        result = await executor.run_dummy_execution(task_id, not_a_repo, 600)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertFalse(result.cancelled)

    async def test_is_busy_and_current_task_id_reflect_in_progress_execution(self):
        task_id = self._new_task_id()
        self.assertFalse(executor.is_busy())
        self.assertIsNone(executor.get_current_task_id())

        reserved = await executor.reserve_slot(task_id)
        self.assertTrue(reserved)
        with mock.patch("services.dev_task_executor._DUMMY_EXECUTION_SECONDS", 1.0):
            task = asyncio.create_task(executor.run_dummy_execution(task_id, self.worktree_path, 600))
            await asyncio.sleep(0.05)
            try:
                self.assertTrue(executor.is_busy())
                self.assertEqual(executor.get_current_task_id(), task_id)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        self.assertFalse(executor.is_busy())
        self.assertIsNone(executor.get_current_task_id())

    async def test_concurrent_reserve_slot_calls_only_one_wins(self):
        # reserve_slot()'s check-then-acquire is documented as race-free
        # within a single event loop specifically because asyncio.Lock.
        # acquire() never yields when the lock is uncontended - so even
        # scheduling both reservations via asyncio.gather() (as opposed to
        # sequential awaits) must still deterministically let only one of
        # them claim the slot, never both.
        first_task_id = self._new_task_id()
        second_task_id = self._new_task_id()

        first_reserved, second_reserved = await asyncio.gather(
            executor.reserve_slot(first_task_id), executor.reserve_slot(second_task_id)
        )

        self.assertNotEqual(first_reserved, second_reserved)
        winner_task_id = first_task_id if first_reserved else second_task_id
        self.assertTrue(executor.is_busy())
        self.assertEqual(executor.get_current_task_id(), winner_task_id)

    async def test_second_execution_is_rejected_not_queued(self):
        with mock.patch("services.dev_task_executor._DUMMY_EXECUTION_SECONDS", 1.0):
            first_task_id = self._new_task_id()
            reserved_first = await executor.reserve_slot(first_task_id)
            self.assertTrue(reserved_first)
            first = asyncio.create_task(executor.run_dummy_execution(first_task_id, self.worktree_path, 600))
            await asyncio.sleep(0.05)

            second_task_id = self._new_task_id()
            reserved_second = await executor.reserve_slot(second_task_id)
            self.assertFalse(reserved_second)
            # A rejected reservation never touches _current_task_id/the lock -
            # the first task is still the sole holder.
            self.assertEqual(executor.get_current_task_id(), first_task_id)

            first.cancel()
            await asyncio.gather(first, return_exceptions=True)

    async def test_lock_released_after_completion_allows_next_execution(self):
        first_task_id = self._new_task_id()
        self.assertTrue(await executor.reserve_slot(first_task_id))
        first_result = await executor.run_dummy_execution(first_task_id, self.worktree_path, 600)
        self.assertTrue(first_result.success)
        self.assertFalse(executor.is_busy())

        second_task_id = self._new_task_id()
        self.assertTrue(await executor.reserve_slot(second_task_id))
        second_result = await executor.run_dummy_execution(second_task_id, self.worktree_path, 600)
        self.assertTrue(second_result.success)

    async def test_lock_released_after_failure_allows_next_execution(self):
        not_a_repo = Path(self._tmp_dir.name) / "not-a-repo-2"
        not_a_repo.mkdir()
        failed_task_id = self._new_task_id()
        self.assertTrue(await executor.reserve_slot(failed_task_id))
        failed_result = await executor.run_dummy_execution(failed_task_id, not_a_repo, 600)
        self.assertFalse(failed_result.success)
        self.assertFalse(executor.is_busy())

        ok_task_id = self._new_task_id()
        self.assertTrue(await executor.reserve_slot(ok_task_id))
        ok_result = await executor.run_dummy_execution(ok_task_id, self.worktree_path, 600)
        self.assertTrue(ok_result.success)

    async def test_request_cancel_stops_the_running_execution(self):
        with mock.patch("services.dev_task_executor._DUMMY_EXECUTION_SECONDS", 5.0):
            task_id = self._new_task_id()
            self.assertTrue(await executor.reserve_slot(task_id))
            task = asyncio.create_task(executor.run_dummy_execution(task_id, self.worktree_path, 600))
            await asyncio.sleep(0.05)

            cancelled = await executor.cancel_dummy_execution_and_wait(task_id)
            self.assertTrue(cancelled)

            result = await task
            self.assertFalse(result.success)
            self.assertTrue(result.cancelled)
            self.assertIsNotNone(result.error_message)

        self.assertFalse(executor.is_busy())

    async def test_request_cancel_returns_false_for_wrong_task_id(self):
        with mock.patch("services.dev_task_executor._DUMMY_EXECUTION_SECONDS", 1.0):
            task_id = self._new_task_id()
            self.assertTrue(await executor.reserve_slot(task_id))
            task = asyncio.create_task(executor.run_dummy_execution(task_id, self.worktree_path, 600))
            await asyncio.sleep(0.05)

            try:
                self.assertFalse(await executor.cancel_dummy_execution_and_wait(self._new_task_id()))
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_request_cancel_returns_false_when_nothing_running(self):
        self.assertFalse(await executor.cancel_dummy_execution_and_wait(self._new_task_id()))


if __name__ == "__main__":
    unittest.main()
