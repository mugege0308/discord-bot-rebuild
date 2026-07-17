import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import discord

import services.dev_task_executor as dev_task_executor
import services.dev_task_store as store
import services.dev_worktree as wt
import views.dev_task as dev_task_view


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True, encoding="utf-8"
    )


class FakeMessage:
    def __init__(self, content: str = ""):
        self.content = content
        self.edits: list[tuple] = []

    async def edit(self, content=None, view=None):
        if content is not None:
            self.content = content
        self.edits.append((content, view))


class FakeResponse:
    def __init__(self):
        self.sent: list[dict] = []
        self.edited: list[dict] = []
        self.deferred = False

    async def send_message(self, content=None, *, ephemeral=False, view=None):
        self.sent.append(dict(content=content, ephemeral=ephemeral, view=view))

    async def edit_message(self, content=None, view=None):
        self.edited.append(dict(content=content, view=view))

    async def defer(self, *, ephemeral=False):
        self.deferred = True


class FakeFollowup:
    def __init__(self):
        self.sent: list[dict] = []

    async def send(self, content=None, *, ephemeral=False, view=None):
        self.sent.append(dict(content=content, ephemeral=ephemeral, view=view))


class FakeInteraction:
    def __init__(self, user_id: int, message: FakeMessage | None = None, client=None):
        self.user = SimpleNamespace(id=user_id)
        self.response = FakeResponse()
        self.followup = FakeFollowup()
        self.message = message
        self.client = client


class FakeUser:
    def __init__(self, user_id: int, raise_exc: Exception | None = None):
        self.id = user_id
        self._raise_exc = raise_exc
        self.sent_messages: list[tuple] = []

    async def send(self, content=None, *, view=None):
        if self._raise_exc is not None:
            raise self._raise_exc
        message = FakeMessage(content)
        self.sent_messages.append((content, view, message))
        return message


class FakeChannel(discord.abc.Messageable):
    def __init__(self, channel_id: int, raise_exc: Exception | None = None):
        self.id = channel_id
        self._raise_exc = raise_exc
        self.sent_messages: list[tuple] = []

    async def _get_channel(self):
        return self

    async def send(self, content=None, *, view=None, **kwargs):
        if self._raise_exc is not None:
            raise self._raise_exc
        message = FakeMessage(content)
        self.sent_messages.append((content, view, message))
        return message


class FakeClient:
    def __init__(self, *, user=None, channel=None, fetch_user_result=None, fetch_channel_result=None):
        self._user = user
        self._channel = channel
        self._fetch_user_result = fetch_user_result
        self._fetch_channel_result = fetch_channel_result

    def get_user(self, user_id):
        return self._user

    async def fetch_user(self, user_id):
        if isinstance(self._fetch_user_result, Exception):
            raise self._fetch_user_result
        return self._fetch_user_result

    def get_channel(self, channel_id):
        return self._channel

    async def fetch_channel(self, channel_id):
        if isinstance(self._fetch_channel_result, Exception):
            raise self._fetch_channel_result
        return self._fetch_channel_result


REQUESTER_ID = 111
OTHER_USER_ID = 222


def _allow(*user_ids: int):
    return mock.patch.dict(
        "os.environ", {"DEV_TASK_ALLOWED_USER_IDS": ",".join(str(u) for u in user_ids)}
    )


class DevTaskApprovalViewTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="dev_task_views_test ")
        self._original_data_dir = store.DATA_DIR
        store.DATA_DIR = Path(self._tmp_dir.name) / "dev_tasks"

    def tearDown(self):
        store.DATA_DIR = self._original_data_dir
        self._tmp_dir.cleanup()

    def _create_task(self, **overrides) -> store.DevTask:
        params = dict(
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
        params.update(overrides)
        result = store.create_task(**params)
        self.assertTrue(result.success)
        return result.task

    # --- interaction_check: requester-only + allow-list re-check ---

    async def test_interaction_check_rejects_non_requester(self):
        task = self._create_task()
        view = dev_task_view.DevTaskApprovalView(task.task_id, REQUESTER_ID)
        interaction = FakeInteraction(OTHER_USER_ID)
        with _allow(REQUESTER_ID, OTHER_USER_ID):
            allowed = await view.interaction_check(interaction)
        self.assertFalse(allowed)
        self.assertIn(dev_task_view.NOT_REQUESTER_MESSAGE, interaction.response.sent[0]["content"])

    async def test_interaction_check_rejects_requester_not_on_allow_list(self):
        task = self._create_task()
        view = dev_task_view.DevTaskApprovalView(task.task_id, REQUESTER_ID)
        interaction = FakeInteraction(REQUESTER_ID)
        with _allow(OTHER_USER_ID):  # requester no longer allowed
            allowed = await view.interaction_check(interaction)
        self.assertFalse(allowed)
        self.assertIn(dev_task_view.DENIED_MESSAGE, interaction.response.sent[0]["content"])

    async def test_interaction_check_allows_requester_on_allow_list(self):
        task = self._create_task()
        view = dev_task_view.DevTaskApprovalView(task.task_id, REQUESTER_ID)
        interaction = FakeInteraction(REQUESTER_ID)
        with _allow(REQUESTER_ID):
            allowed = await view.interaction_check(interaction)
        self.assertTrue(allowed)

    # --- engine override select ---

    async def test_engine_override_select_updates_task(self):
        task = self._create_task(routed_engine="codex")
        view = dev_task_view.DevTaskApprovalView(task.task_id, REQUESTER_ID)
        interaction = FakeInteraction(REQUESTER_ID)
        # Select.values is a read-only property backed by _values (falling
        # back to it when no request-scoped selection context is active,
        # which is exactly the case here since we invoke the callback
        # directly rather than through discord.py's real dispatch).
        view.engine_override_select._values = ["claude"]

        await view.engine_override_select.callback(interaction)

        # Deliberately NOT persisted to the store yet (see
        # DevTaskApprovalView's docstring) - only staged on the view, to be
        # applied atomically together with the approval claim itself.
        self.assertEqual(view.pending_engine_override, "claude")
        unchanged = store.get_task(task.task_id).task
        self.assertEqual(unchanged.routed_engine, "codex")
        self.assertFalse(unchanged.engine_overridden)
        self.assertEqual(len(interaction.response.sent), 1)

    # --- approve button: happy path, disables + delegates to execution ---

    async def test_approve_button_disables_and_starts_execution(self):
        task = self._create_task()
        message = FakeMessage("original content")
        view = dev_task_view.DevTaskApprovalView(task.task_id, REQUESTER_ID)
        interaction = FakeInteraction(REQUESTER_ID, message=message, client=object())

        with mock.patch("views.dev_task.start_dev_task_execution", new=mock.AsyncMock()) as start_mock:
            await view.approve_button.callback(interaction)

        self.assertTrue(all(item.disabled for item in view.children))
        self.assertEqual(len(interaction.response.edited), 1)
        start_mock.assert_awaited_once_with(interaction.client, task.task_id, engine_override=None)

    async def test_approve_button_multi_click_guard(self):
        task = self._create_task()
        message = FakeMessage("original content")
        view = dev_task_view.DevTaskApprovalView(task.task_id, REQUESTER_ID)
        interaction1 = FakeInteraction(REQUESTER_ID, message=message, client=object())
        interaction2 = FakeInteraction(REQUESTER_ID, message=message, client=object())

        with mock.patch("views.dev_task.start_dev_task_execution", new=mock.AsyncMock()) as start_mock:
            await view.approve_button.callback(interaction1)
            await view.approve_button.callback(interaction2)

        start_mock.assert_awaited_once()
        self.assertIn(dev_task_view.ALREADY_PROCESSED_MESSAGE, interaction2.response.sent[0]["content"])

    # --- cancel button ---

    async def test_cancel_button_marks_task_failed(self):
        task = self._create_task()
        message = FakeMessage("original content")
        view = dev_task_view.DevTaskApprovalView(task.task_id, REQUESTER_ID)
        interaction = FakeInteraction(REQUESTER_ID, message=message)

        await view.cancel_button.callback(interaction)

        updated = store.get_task(task.task_id).task
        self.assertEqual(updated.status, store.DevTaskStatus.FAILED.value)
        self.assertIsNotNone(updated.error_message)
        self.assertTrue(all(item.disabled for item in view.children))

    # --- timeout ---

    async def test_on_timeout_marks_pending_task_failed(self):
        task = self._create_task()
        view = dev_task_view.DevTaskApprovalView(task.task_id, REQUESTER_ID)
        view.message = FakeMessage("original content")

        await view.on_timeout()

        updated = store.get_task(task.task_id).task
        self.assertEqual(updated.status, store.DevTaskStatus.FAILED.value)
        self.assertIn("期限切れ", view.message.content)

    async def test_on_timeout_does_not_touch_already_decided_task(self):
        task = self._create_task()
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)
        view = dev_task_view.DevTaskApprovalView(task.task_id, REQUESTER_ID)
        view.message = FakeMessage("original content")

        await view.on_timeout()

        updated = store.get_task(task.task_id).task
        self.assertEqual(updated.status, store.DevTaskStatus.RUNNING.value)


class DevTaskResultViewTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="dev_task_views_test ")
        self._original_data_dir = store.DATA_DIR
        store.DATA_DIR = Path(self._tmp_dir.name) / "dev_tasks"

    def tearDown(self):
        store.DATA_DIR = self._original_data_dir
        self._tmp_dir.cleanup()

    def _create_completed_task(self) -> store.DevTask:
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
        # approved_at/started_at/branch_name, and finished_at/result_summary,
        # are each only ever settable through their own dedicated atomic
        # store function (claim_task_for_execution / finish_task) - plain
        # update_task() rejects them outright (ERROR_UNKNOWN_FIELD). See
        # services.dev_task_store's module notes.
        claim_result = store.claim_task_for_execution(
            task.task_id, branch_name=f"dev-task/{task.task_id}"
        )
        self.assertTrue(claim_result.success)
        finish_result = store.finish_task(task.task_id, success=True, result_summary="0 files changed")
        self.assertTrue(finish_result.success)
        notify_result = store.set_notification_method_once(task.task_id, method="dm")
        self.assertTrue(notify_result.success)
        return store.get_task(task.task_id).task

    async def test_keep_button_records_decision_and_disables(self):
        task = self._create_completed_task()
        message = FakeMessage("result content")
        view = dev_task_view.DevTaskResultView(task.task_id, REQUESTER_ID)
        view.message = message
        interaction = FakeInteraction(REQUESTER_ID, message=message)

        await view.keep_button.callback(interaction)

        updated = store.get_task(task.task_id).task
        self.assertEqual(updated.decision, "kept")
        self.assertTrue(all(item.disabled for item in view.children))

    async def test_keep_button_rejects_second_click(self):
        task = self._create_completed_task()
        message = FakeMessage("result content")
        view = dev_task_view.DevTaskResultView(task.task_id, REQUESTER_ID)
        view.message = message

        await view.keep_button.callback(FakeInteraction(REQUESTER_ID, message=message))
        interaction2 = FakeInteraction(REQUESTER_ID, message=message)
        await view.keep_button.callback(interaction2)

        self.assertIn(dev_task_view.ALREADY_PROCESSED_MESSAGE, interaction2.response.sent[0]["content"])

    async def test_discard_button_success_removes_worktree_and_records_decision(self):
        task = self._create_completed_task()
        message = FakeMessage("result content")
        view = dev_task_view.DevTaskResultView(task.task_id, REQUESTER_ID)
        view.message = message
        interaction = FakeInteraction(REQUESTER_ID, message=message)

        fake_remove_result = wt.WorktreeRemoveResult(
            success=True,
            task_id=task.task_id,
            worktree_path=Path("C:/fake/work") / task.task_id,
            branch_name=task.branch_name,
            worktree_removed=True,
            branch_removed=True,
            manual_cleanup_required=False,
        )
        with mock.patch("services.dev_task_decision.remove_worktree", return_value=fake_remove_result):
            await view.discard_button.callback(interaction)

        updated = store.get_task(task.task_id).task
        self.assertEqual(updated.decision, "discarded")
        self.assertTrue(all(item.disabled for item in view.children))
        self.assertIn("破棄しました", message.content)

    async def test_discard_button_refuses_on_foreign_worktree_and_leaves_decision_unset(self):
        task = self._create_completed_task()
        message = FakeMessage("result content")
        view = dev_task_view.DevTaskResultView(task.task_id, REQUESTER_ID)
        view.message = message
        interaction = FakeInteraction(REQUESTER_ID, message=message)

        fake_remove_result = wt.WorktreeRemoveResult(
            success=False,
            task_id=task.task_id,
            worktree_path=Path("C:/fake/work") / task.task_id,
            branch_name=task.branch_name,
            worktree_removed=False,
            branch_removed=False,
            manual_cleanup_required=False,
            error_message=wt._ERROR_MESSAGES[wt.ERROR_FOREIGN_WORKTREE],
            error_code=wt.ERROR_FOREIGN_WORKTREE,
        )
        with mock.patch("services.dev_task_decision.remove_worktree", return_value=fake_remove_result):
            await view.discard_button.callback(interaction)

        updated = store.get_task(task.task_id).task
        self.assertIsNone(updated.decision)
        self.assertFalse(any(item.disabled for item in view.children))
        self.assertEqual(message.content, "result content")  # never edited
        self.assertIn("安全確認", interaction.followup.sent[0]["content"])

    async def test_discard_button_refuses_on_boundary_violation_and_leaves_decision_unset(self):
        task = self._create_completed_task()
        message = FakeMessage("result content")
        view = dev_task_view.DevTaskResultView(task.task_id, REQUESTER_ID)
        view.message = message
        interaction = FakeInteraction(REQUESTER_ID, message=message)

        fake_remove_result = wt.WorktreeRemoveResult(
            success=False,
            task_id=task.task_id,
            worktree_path=Path("C:/fake/work") / task.task_id,
            branch_name=task.branch_name,
            worktree_removed=False,
            branch_removed=False,
            manual_cleanup_required=False,
            error_message=wt._ERROR_MESSAGES[wt.ERROR_PATH_BOUNDARY_VIOLATION],
            error_code=wt.ERROR_PATH_BOUNDARY_VIOLATION,
        )
        with mock.patch("services.dev_task_decision.remove_worktree", return_value=fake_remove_result):
            await view.discard_button.callback(interaction)

        updated = store.get_task(task.task_id).task
        self.assertIsNone(updated.decision)
        self.assertFalse(any(item.disabled for item in view.children))

    async def test_discard_button_partial_failure_shows_manual_cleanup_required(self):
        task = self._create_completed_task()
        message = FakeMessage("result content")
        view = dev_task_view.DevTaskResultView(task.task_id, REQUESTER_ID)
        view.message = message
        interaction = FakeInteraction(REQUESTER_ID, message=message)

        fake_remove_result = wt.WorktreeRemoveResult(
            success=False,
            task_id=task.task_id,
            worktree_path=Path("C:/fake/work") / task.task_id,
            branch_name=task.branch_name,
            worktree_removed=False,
            branch_removed=False,
            manual_cleanup_required=True,
            error_message=wt._ERROR_MESSAGES[wt.ERROR_WORKTREE_REMOVE_INCOMPLETE],
            error_code=wt.ERROR_WORKTREE_REMOVE_INCOMPLETE,
        )
        with mock.patch("services.dev_task_decision.remove_worktree", return_value=fake_remove_result):
            await view.discard_button.callback(interaction)

        updated = store.get_task(task.task_id).task
        self.assertEqual(updated.decision, "discarded")
        self.assertIn("手動での確認が必要", message.content)

    async def test_interaction_check_rejects_non_requester(self):
        task = self._create_completed_task()
        view = dev_task_view.DevTaskResultView(task.task_id, REQUESTER_ID)
        interaction = FakeInteraction(OTHER_USER_ID)
        with _allow(REQUESTER_ID, OTHER_USER_ID):
            allowed = await view.interaction_check(interaction)
        self.assertFalse(allowed)


class SendResultNotificationTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_dm_success_returns_dm_and_sets_view_message(self):
        user = FakeUser(REQUESTER_ID)
        client = FakeClient(user=user)
        view = SimpleNamespace(message=None)

        method = await dev_task_view._send_result_notification(client, REQUESTER_ID, "hello", view=view)

        self.assertEqual(method, "dm")
        self.assertEqual(len(user.sent_messages), 1)
        self.assertIsNotNone(view.message)

    async def test_dm_failure_falls_back_to_channel(self):
        # FakeChannel is a minimal discord.abc.Messageable stub, not a real
        # discord.TextChannel/Thread/etc., so it can never itself satisfy
        # _fallback_channel_is_private_enough()'s isinstance/permission
        # checks - that fail-closed privacy gate has its own coverage
        # elsewhere; here we're only exercising the DM-fails -> channel-
        # fallback control flow, so the gate is stubbed open.
        user = FakeUser(REQUESTER_ID, raise_exc=discord.Forbidden.__new__(discord.Forbidden))
        channel = FakeChannel(999)
        client = FakeClient(user=user, channel=channel)

        with (
            mock.patch("views.dev_task.get_dev_task_result_channel_id", return_value=999),
            mock.patch("views.dev_task._fallback_channel_is_private_enough", return_value=True),
        ):
            method = await dev_task_view._send_result_notification(client, REQUESTER_ID, "hello", view=None)

        self.assertEqual(method, "channel")
        self.assertEqual(len(channel.sent_messages), 1)

    async def test_dm_and_channel_both_failing_returns_none_without_raising(self):
        user = FakeUser(REQUESTER_ID, raise_exc=discord.Forbidden.__new__(discord.Forbidden))
        channel = FakeChannel(999, raise_exc=discord.Forbidden.__new__(discord.Forbidden))
        client = FakeClient(user=user, channel=channel)

        with (
            mock.patch("views.dev_task.get_dev_task_result_channel_id", return_value=999),
            mock.patch("views.dev_task._fallback_channel_is_private_enough", return_value=True),
        ):
            method = await dev_task_view._send_result_notification(client, REQUESTER_ID, "hello", view=None)

        self.assertEqual(method, "none")

    async def test_dm_failure_with_no_channel_configured_returns_none(self):
        user = FakeUser(REQUESTER_ID, raise_exc=discord.Forbidden.__new__(discord.Forbidden))
        client = FakeClient(user=user)

        with mock.patch("views.dev_task.get_dev_task_result_channel_id", return_value=None):
            method = await dev_task_view._send_result_notification(client, REQUESTER_ID, "hello", view=None)

        self.assertEqual(method, "none")


class StartDevTaskExecutionTestCase(unittest.IsolatedAsyncioTestCase):
    """Integration-style tests exercising the full approve -> (safely
    rejected) -> notify -> persist pipeline against a real temp git repo,
    with only the Discord-facing client faked out.

    Phase 4A-v2: execute_approved_task() safely rejects every engine right
    now, for every value of CLAUDE_HEADLESS_ENABLED - see
    services.dev_task_executor's module-level note on why real Claude
    execution (the isolated-copy sandbox mechanism) is not wired into this
    live pipeline yet. These tests confirm that rejection happens
    consistently, before any worktree is ever created, and that the
    Discord-facing notification/Store bookkeeping around that rejection
    still works correctly.
    """

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="dev_task_views_flow_test ")
        self._original_data_dir = store.DATA_DIR
        store.DATA_DIR = Path(self._tmp_dir.name) / "dev_tasks"

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
        store.DATA_DIR = self._original_data_dir
        wt.REPO_ROOT = self._original_repo_root
        self._tmp_dir.cleanup()
        dev_task_executor._current_task_id = None
        dev_task_executor._current_asyncio_task = None
        if dev_task_executor._execution_lock.locked():
            dev_task_executor._execution_lock.release()

    def _create_task(self, **overrides) -> store.DevTask:
        params = dict(
            requested_by_user_id=REQUESTER_ID,
            guild_id=333,
            channel_id=444,
            prompt_text="バグを修正して",
            routed_engine="claude",
            routing_reason="test",
            engine_overridden=False,
            timeout_seconds=600,
            base_branch="main",
        )
        params.update(overrides)
        return store.create_task(**params).task

    async def test_full_flow_claude_disabled_is_rejected_and_notified(self):
        task = self._create_task()
        user = FakeUser(REQUESTER_ID)
        client = FakeClient(user=user)

        with mock.patch.dict("os.environ", {"CLAUDE_HEADLESS_ENABLED": "false"}):
            await dev_task_view.start_dev_task_execution(client, task.task_id)

        final = store.get_task(task.task_id).task
        self.assertEqual(final.status, store.DevTaskStatus.FAILED.value)
        self.assertIsNotNone(final.error_message)
        self.assertEqual(final.notification_method, "dm")
        self.assertEqual(len(user.sent_messages), 1)
        content = user.sent_messages[0][0]
        self.assertNotIn(str(self.repo_root), content)  # no raw absolute path leaked

        info = wt.get_worktree_info(task.task_id).info
        self.assertEqual(info.consistency, wt.CONSISTENCY_NOT_FOUND)

        # No Result View offered - no worktree/sandbox was ever created.
        sent_view = user.sent_messages[0][1]
        self.assertIsNone(sent_view)

    async def test_full_flow_claude_enabled_is_still_rejected_before_worktree_creation(self):
        # Enabling CLAUDE_HEADLESS_ENABLED must not change this live
        # pipeline's behavior - the isolated-copy sandbox mechanism is
        # intentionally not wired into it yet (see the module-level note
        # in services.dev_task_executor).
        task = self._create_task()
        user = FakeUser(REQUESTER_ID)
        client = FakeClient(user=user)

        with mock.patch.dict("os.environ", {"CLAUDE_HEADLESS_ENABLED": "true"}), mock.patch(
            "services.dev_task_executor.reserve_slot", new=mock.AsyncMock()
        ) as reserve_slot_mock:
            await dev_task_view.start_dev_task_execution(client, task.task_id)

        reserve_slot_mock.assert_not_called()

        final = store.get_task(task.task_id).task
        self.assertEqual(final.status, store.DevTaskStatus.FAILED.value)
        self.assertIsNotNone(final.error_message)
        self.assertEqual(final.notification_method, "dm")

        info = wt.get_worktree_info(task.task_id).info
        self.assertEqual(info.consistency, wt.CONSISTENCY_NOT_FOUND)

        sent_view = user.sent_messages[0][1]
        self.assertIsNone(sent_view)

    async def test_full_flow_codex_engine_is_rejected_before_worktree_creation(self):
        task = self._create_task(routed_engine="codex")
        user = FakeUser(REQUESTER_ID)
        client = FakeClient(user=user)

        with mock.patch(
            "services.dev_task_executor.reserve_slot", new=mock.AsyncMock()
        ) as reserve_slot_mock:
            await dev_task_view.start_dev_task_execution(client, task.task_id)

        # Rejected before ever reserving the execution slot or touching a
        # worktree - never falls back to Claude, never falls back to dummy
        # execution.
        reserve_slot_mock.assert_not_called()

        final = store.get_task(task.task_id).task
        self.assertEqual(final.status, store.DevTaskStatus.FAILED.value)
        self.assertIsNotNone(final.error_message)
        self.assertNotEqual(final.error_message, "")

        info = wt.get_worktree_info(task.task_id).info
        self.assertEqual(info.consistency, wt.CONSISTENCY_NOT_FOUND)

    async def test_full_flow_engine_override_to_codex_is_rejected(self):
        # Even though the task itself was routed to claude, an approval-time
        # override to codex must still be safely rejected - never executed.
        task = self._create_task(routed_engine="claude")
        user = FakeUser(REQUESTER_ID)
        client = FakeClient(user=user)

        with mock.patch.dict("os.environ", {"CLAUDE_HEADLESS_ENABLED": "true"}), mock.patch(
            "services.dev_task_executor.reserve_slot", new=mock.AsyncMock()
        ) as reserve_slot_mock:
            await dev_task_view.start_dev_task_execution(client, task.task_id, engine_override="codex")

        reserve_slot_mock.assert_not_called()
        final = store.get_task(task.task_id).task
        self.assertEqual(final.status, store.DevTaskStatus.FAILED.value)
        info = wt.get_worktree_info(task.task_id).info
        self.assertEqual(info.consistency, wt.CONSISTENCY_NOT_FOUND)


if __name__ == "__main__":
    unittest.main()
