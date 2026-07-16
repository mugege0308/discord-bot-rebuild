import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import services.dev_task_decision as decision
import services.dev_task_store as store
import services.dev_worktree as wt


REQUESTER_ID = 111


class DevTaskDecisionTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="dev_task_decision_test ")
        self._original_data_dir = store.DATA_DIR
        store.DATA_DIR = Path(self._tmp_dir.name) / "dev_tasks"

    def tearDown(self):
        store.DATA_DIR = self._original_data_dir
        self._tmp_dir.cleanup()

    def _create_completed_task(self, *, failed: bool = False) -> store.DevTask:
        result = store.create_task(
            requested_by_user_id=REQUESTER_ID,
            guild_id=333,
            channel_id=444,
            prompt_text="バグを修正して",
            routed_engine="codex",
            routing_reason="test",
            engine_overridden=False,
            timeout_seconds=600,
            base_branch="main",
        )
        task = result.task
        claim_result = store.claim_task_for_execution(
            task.task_id, branch_name=f"dev-task/{task.task_id}"
        )
        self.assertTrue(claim_result.success)
        if failed:
            finish_result = store.finish_task(task.task_id, success=False, error_message="失敗しました。")
        else:
            finish_result = store.finish_task(task.task_id, success=True, result_summary="0 files changed")
        self.assertTrue(finish_result.success)
        return store.get_task(task.task_id).task

    def _fake_remove_result(self, task: store.DevTask, **overrides) -> wt.WorktreeRemoveResult:
        params = dict(
            success=True,
            task_id=task.task_id,
            worktree_path=Path("C:/fake/work") / task.task_id,
            branch_name=task.branch_name,
            worktree_removed=True,
            branch_removed=True,
            manual_cleanup_required=False,
        )
        params.update(overrides)
        return wt.WorktreeRemoveResult(**params)

    # --- keep_worktree ---

    async def test_keep_worktree_success(self):
        task = self._create_completed_task()

        outcome = await decision.keep_worktree(task.task_id)

        self.assertTrue(outcome.success)
        self.assertEqual(outcome.decision, "kept")
        self.assertFalse(outcome.already_decided)
        self.assertFalse(outcome.refused)
        updated = store.get_task(task.task_id).task
        self.assertEqual(updated.decision, "kept")

    async def test_keep_worktree_already_decided(self):
        task = self._create_completed_task()
        first = await decision.keep_worktree(task.task_id)
        self.assertTrue(first.success)

        second = await decision.keep_worktree(task.task_id)

        self.assertFalse(second.success)
        self.assertTrue(second.already_decided)
        self.assertEqual(second.decision, "kept")

    async def test_keep_worktree_task_not_found(self):
        outcome = await decision.keep_worktree("00000000-0000-4000-8000-000000000000")

        self.assertFalse(outcome.success)
        self.assertFalse(outcome.already_decided)
        self.assertIsNotNone(outcome.error_message)

    async def test_keep_worktree_rejects_task_not_yet_terminal(self):
        # pending_approval/running - keep/discard only make sense once the
        # task has actually reached a terminal status.
        result = store.create_task(
            requested_by_user_id=REQUESTER_ID,
            guild_id=333,
            channel_id=444,
            prompt_text="バグを修正して",
            routed_engine="codex",
            routing_reason="test",
            engine_overridden=False,
            timeout_seconds=600,
            base_branch="main",
        )
        task = result.task

        outcome = await decision.keep_worktree(task.task_id)

        self.assertFalse(outcome.success)
        self.assertFalse(outcome.already_decided)
        updated = store.get_task(task.task_id).task
        self.assertIsNone(updated.decision)

    # --- discard_worktree: success / already-decided / not-found ---

    async def test_discard_worktree_success(self):
        task = self._create_completed_task()
        fake_result = self._fake_remove_result(task)

        with mock.patch("services.dev_task_decision.remove_worktree", return_value=fake_result) as mocked:
            outcome = await decision.discard_worktree(task.task_id)

        mocked.assert_called_once_with(task.task_id)
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.decision, "discarded")
        self.assertFalse(outcome.manual_cleanup_required)
        updated = store.get_task(task.task_id).task
        self.assertEqual(updated.decision, "discarded")

    async def test_discard_worktree_already_decided(self):
        task = self._create_completed_task()
        fake_result = self._fake_remove_result(task)
        with mock.patch("services.dev_task_decision.remove_worktree", return_value=fake_result):
            first = await decision.discard_worktree(task.task_id)
        self.assertTrue(first.success)

        second = await decision.discard_worktree(task.task_id)

        self.assertFalse(second.success)
        self.assertTrue(second.already_decided)

    async def test_discard_worktree_task_not_found(self):
        outcome = await decision.discard_worktree("00000000-0000-4000-8000-000000000000")

        self.assertFalse(outcome.success)
        self.assertFalse(outcome.already_decided)
        self.assertIsNotNone(outcome.error_message)

    async def test_discard_worktree_treats_worktree_already_gone_as_a_valid_discard(self):
        task = self._create_completed_task()
        fake_result = self._fake_remove_result(
            task,
            success=False,
            worktree_removed=False,
            branch_removed=False,
            error_message=wt._ERROR_MESSAGES[wt.ERROR_WORKTREE_NOT_FOUND],
            error_code=wt.ERROR_WORKTREE_NOT_FOUND,
        )

        with mock.patch("services.dev_task_decision.remove_worktree", return_value=fake_result):
            outcome = await decision.discard_worktree(task.task_id)

        self.assertTrue(outcome.success)
        self.assertEqual(outcome.decision, "discarded")
        updated = store.get_task(task.task_id).task
        self.assertEqual(updated.decision, "discarded")

    # --- discard_worktree: safety refusals leave the decision unrecorded ---

    async def test_discard_worktree_refuses_on_foreign_worktree(self):
        task = self._create_completed_task()
        fake_result = self._fake_remove_result(
            task,
            success=False,
            worktree_removed=False,
            branch_removed=False,
            error_message=wt._ERROR_MESSAGES[wt.ERROR_FOREIGN_WORKTREE],
            error_code=wt.ERROR_FOREIGN_WORKTREE,
        )

        with mock.patch("services.dev_task_decision.remove_worktree", return_value=fake_result):
            outcome = await decision.discard_worktree(task.task_id)

        self.assertFalse(outcome.success)
        self.assertTrue(outcome.refused)
        self.assertIsNone(outcome.decision)
        updated = store.get_task(task.task_id).task
        self.assertIsNone(updated.decision)

    async def test_discard_worktree_refuses_on_path_boundary_violation(self):
        task = self._create_completed_task()
        fake_result = self._fake_remove_result(
            task,
            success=False,
            worktree_removed=False,
            branch_removed=False,
            error_message=wt._ERROR_MESSAGES[wt.ERROR_PATH_BOUNDARY_VIOLATION],
            error_code=wt.ERROR_PATH_BOUNDARY_VIOLATION,
        )

        with mock.patch("services.dev_task_decision.remove_worktree", return_value=fake_result):
            outcome = await decision.discard_worktree(task.task_id)

        self.assertFalse(outcome.success)
        self.assertTrue(outcome.refused)
        self.assertIsNone(outcome.decision)
        updated = store.get_task(task.task_id).task
        self.assertIsNone(updated.decision)

    async def test_discard_worktree_refusal_allows_a_later_retry(self):
        task = self._create_completed_task()
        refused_result = self._fake_remove_result(
            task,
            success=False,
            worktree_removed=False,
            branch_removed=False,
            error_message=wt._ERROR_MESSAGES[wt.ERROR_FOREIGN_WORKTREE],
            error_code=wt.ERROR_FOREIGN_WORKTREE,
        )
        with mock.patch("services.dev_task_decision.remove_worktree", return_value=refused_result):
            first = await decision.discard_worktree(task.task_id)
        self.assertTrue(first.refused)

        ok_result = self._fake_remove_result(task)
        with mock.patch("services.dev_task_decision.remove_worktree", return_value=ok_result):
            second = await decision.discard_worktree(task.task_id)

        self.assertTrue(second.success)
        self.assertEqual(second.decision, "discarded")

    # --- discard_worktree: partial failure still records the decision ---

    async def test_discard_worktree_partial_failure_still_records_discarded_with_manual_cleanup_flag(self):
        task = self._create_completed_task()
        fake_result = self._fake_remove_result(
            task,
            success=False,
            worktree_removed=False,
            branch_removed=True,
            manual_cleanup_required=True,
            error_message=wt._ERROR_MESSAGES[wt.ERROR_WORKTREE_REMOVE_INCOMPLETE],
            error_code=wt.ERROR_WORKTREE_REMOVE_INCOMPLETE,
        )

        with mock.patch("services.dev_task_decision.remove_worktree", return_value=fake_result):
            outcome = await decision.discard_worktree(task.task_id)

        self.assertTrue(outcome.success)
        self.assertEqual(outcome.decision, "discarded")
        self.assertTrue(outcome.manual_cleanup_required)
        updated = store.get_task(task.task_id).task
        self.assertEqual(updated.decision, "discarded")

    # --- keep/discard mutual-exclusion CAS races ---

    async def test_concurrent_keep_calls_only_one_wins(self):
        task = self._create_completed_task()

        results = await asyncio.gather(
            decision.keep_worktree(task.task_id), decision.keep_worktree(task.task_id)
        )

        successes = [r for r in results if r.success]
        already_decided = [r for r in results if r.already_decided]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(already_decided), 1)
        updated = store.get_task(task.task_id).task
        self.assertEqual(updated.decision, "kept")

    async def test_concurrent_keep_and_discard_only_one_wins(self):
        task = self._create_completed_task()
        fake_result = self._fake_remove_result(task)

        with mock.patch("services.dev_task_decision.remove_worktree", return_value=fake_result):
            keep_outcome, discard_outcome = await asyncio.gather(
                decision.keep_worktree(task.task_id), decision.discard_worktree(task.task_id)
            )

        outcomes = [keep_outcome, discard_outcome]
        successes = [o for o in outcomes if o.success]
        already_decided = [o for o in outcomes if o.already_decided]
        # Exactly one of the two calls durably records a decision; the loser
        # sees the winner's decision already set and changes nothing - never
        # both "kept" and "discarded" recorded for the same task.
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(already_decided), 1)
        updated = store.get_task(task.task_id).task
        self.assertIn(updated.decision, ("kept", "discarded"))
        self.assertEqual(updated.decision, successes[0].decision)

    async def test_decision_lock_dict_does_not_grow_unbounded(self):
        task = self._create_completed_task()
        await decision.keep_worktree(task.task_id)
        self.assertNotIn(task.task_id, decision._decision_locks)


if __name__ == "__main__":
    unittest.main()
