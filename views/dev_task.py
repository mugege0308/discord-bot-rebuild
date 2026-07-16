from __future__ import annotations

import asyncio
import logging

import discord

from services import dev_task_decision
from services import dev_task_executor
from services import dev_task_store as store
from services.dev_task_config import get_dev_task_result_channel_id, is_dev_task_allowed


logger = logging.getLogger("secretary-bot.dev_task_view")

NOT_REQUESTER_MESSAGE = "この操作は、起票した本人だけができます。"
DENIED_MESSAGE = "この操作は、許可されたユーザーだけが実行できます。"
ALREADY_PROCESSED_MESSAGE = "このタスクは既に処理されています。"

_DUMMY_EXECUTION_NOTICE = "（ダミー実行です。実際のCodex/Claude Codeは実行していません）"

_APPROVAL_VIEW_TIMEOUT_SECONDS = 900  # 15 minutes

# Guild channel types this bot knows how to check privacy for. Anything
# else (DM channels, group DMs, forum channels, ...) is refused outright as
# a fallback destination - see _fallback_channel_is_private_enough().
_SUPPORTED_FALLBACK_CHANNEL_TYPES = (
    discord.TextChannel,
    discord.Thread,
    discord.VoiceChannel,
    discord.StageChannel,
)


def _short_task_id(task_id: str) -> str:
    # Full UUID, not a truncated prefix: /dev_task_status only ever does an
    # exact task_id lookup (services.dev_task_store requires a syntactically
    # valid UUID4), so showing a shortened id anywhere would make it
    # impossible for the user to actually look the task up again.
    return task_id


def _fallback_channel_is_private_enough(channel: discord.abc.Messageable, requester_id: int) -> bool:
    """Fail-closed: True only when this is a supported guild channel type,
    @everyone does NOT have view_channel here, and both the bot and the
    requester can view (and the bot can send) here. Any exception or
    unsupported case returns False - never send real task content to a
    channel we couldn't positively confirm is private.
    """
    if not isinstance(channel, _SUPPORTED_FALLBACK_CHANNEL_TYPES):
        logger.error(
            "DEV_TASK_RESULT_CHANNEL_ID is not a supported channel type for private fallback: %s",
            type(channel),
        )
        return False

    guild = getattr(channel, "guild", None)
    if guild is None:
        logger.error("DEV_TASK_RESULT_CHANNEL_ID channel has no guild; refusing as unsafe.")
        return False

    try:
        everyone_perms = channel.permissions_for(guild.default_role)
        if everyone_perms.view_channel:
            logger.error("DEV_TASK_RESULT_CHANNEL_ID is viewable by @everyone; refusing as unsafe.")
            return False

        bot_member = guild.me
        if bot_member is None:
            logger.error("Could not resolve bot member in guild for fallback channel privacy check.")
            return False
        bot_perms = channel.permissions_for(bot_member)
        if not (bot_perms.view_channel and bot_perms.send_messages):
            logger.error("Bot lacks view/send permission in DEV_TASK_RESULT_CHANNEL_ID; refusing.")
            return False

        requester_member = guild.get_member(requester_id)
        if requester_member is None:
            logger.error("Requester is not a member of the fallback channel's guild; refusing.")
            return False
        requester_perms = channel.permissions_for(requester_member)
        if not requester_perms.view_channel:
            logger.error("Requester cannot view DEV_TASK_RESULT_CHANNEL_ID; refusing.")
            return False
    except Exception:
        logger.exception("Failed to verify DEV_TASK_RESULT_CHANNEL_ID privacy; refusing as unsafe.")
        return False

    return True


async def _send_result_notification(
    client: discord.Client,
    user_id: int,
    content: str,
    *,
    view: discord.ui.View | None,
) -> str:
    """Attempt to DM `user_id` first, then fall back to
    DEV_TASK_RESULT_CHANNEL_ID only if it passes
    _fallback_channel_is_private_enough(). Returns "dm", "channel", or
    "none" - this never raises, so a total delivery failure is logged but
    can never crash the caller or take down the bot. A fallback channel
    that fails the privacy check is treated exactly like total delivery
    failure ("none") - the real content (branch name, diff, failure
    reason) is never sent to a channel this function couldn't positively
    confirm is private.
    """
    try:
        user = client.get_user(user_id)
        if user is None:
            user = await client.fetch_user(user_id)
        message = await (user.send(content, view=view) if view is not None else user.send(content))
        if view is not None:
            view.message = message
        return "dm"
    except (discord.Forbidden, discord.HTTPException, discord.NotFound):
        logger.warning("Failed to DM dev task result to user_id=%s", user_id, exc_info=True)
    except Exception:
        logger.exception("Unexpected error while DMing dev task result to user_id=%s", user_id)

    channel_id = get_dev_task_result_channel_id()
    if channel_id is not None:
        try:
            channel = client.get_channel(channel_id)
            if channel is None:
                channel = await client.fetch_channel(channel_id)
            if not _fallback_channel_is_private_enough(channel, user_id):
                logger.error(
                    "Refusing to use DEV_TASK_RESULT_CHANNEL_ID as fallback - not confirmed private: "
                    "channel_id=%s",
                    channel_id,
                )
            else:
                message = await (
                    channel.send(content, view=view) if view is not None else channel.send(content)
                )
                if view is not None:
                    view.message = message
                return "channel"
        except (discord.Forbidden, discord.HTTPException, discord.NotFound):
            logger.warning(
                "Failed to send dev task result to fallback channel_id=%s", channel_id, exc_info=True
            )
        except Exception:
            logger.exception(
                "Unexpected error while sending dev task result to fallback channel_id=%s", channel_id
            )

    logger.error("Failed to deliver dev task result notification via DM or channel: user_id=%s", user_id)
    return "none"


async def _send_and_record_notification(
    client: discord.Client,
    task_id: str,
    user_id: int,
    content: str,
    *,
    view: discord.ui.View | None,
) -> None:
    """Send the result notification and record how it went, exactly once,
    via the dedicated set_notification_method_once() CAS - deliberately
    never bundled into the same store write as the terminal state itself,
    since the delivery method isn't known until after that state is
    already durably saved (see services.dev_task_store's module notes).
    """
    notification_method = await _send_result_notification(client, user_id, content, view=view)
    record_result = store.set_notification_method_once(task_id, method=notification_method)
    if not record_result.success:
        logger.error(
            "Failed to record dev task notification_method (delivery itself already happened): "
            "task_id=%s method=%s error_code=%s",
            task_id,
            notification_method,
            record_result.error_code,
        )


def _build_success_message(task_id: str, engine: str, branch_name: str | None, execution_result) -> str:
    return "\n".join(
        [
            f"【開発タスク結果】{_DUMMY_EXECUTION_NOTICE}",
            f"タスクID: {_short_task_id(task_id)}",
            f"エンジン: {engine}",
            "状態: 完了",
            f"所要時間: {execution_result.duration_seconds:.1f}秒",
            f"ブランチ: {branch_name or '不明'}",
            f"変更ファイル数: {execution_result.changed_file_count}",
            "git diff概要:",
            f"```\n{execution_result.diff_summary}\n```",
        ]
    )


def _build_failure_message(
    task_id: str, engine: str, reason: str, *, branch_name: str | None = None, duration_seconds: float | None = None
) -> str:
    lines = [
        f"【開発タスク結果】{_DUMMY_EXECUTION_NOTICE}",
        f"タスクID: {_short_task_id(task_id)}",
        f"エンジン: {engine}",
        "状態: 失敗",
    ]
    if duration_seconds is not None:
        lines.append(f"所要時間: {duration_seconds:.1f}秒")
    if branch_name is not None:
        lines.append(f"ブランチ: {branch_name}")
    lines.append(f"失敗理由: {reason}")
    return "\n".join(lines)


class DevTaskResultView(discord.ui.View):
    """Offered only when a worktree actually exists for this task (create_
    worktree() succeeded, regardless of whether the subsequent dummy
    execution then completed or failed) - there is nothing to keep/discard
    otherwise.

    keep/discard themselves are delegated entirely to
    services.dev_task_decision - a discord.py-independent module - which
    is what actually makes the decision atomic/CAS-safe against concurrent
    clicks; this class only builds the Discord-facing message/response
    around that outcome.

    Not a persistent view: after a bot restart this in-memory view object
    (and its message reference) is gone, so old keep/discard buttons on
    messages sent before a restart simply stop responding. /dev_task_recover
    can re-send a fresh Result View for a terminal, still-undecided task.
    """

    def __init__(self, task_id: str, requester_id: int) -> None:
        super().__init__(timeout=None)
        self.task_id = task_id
        self.requester_id = requester_id
        self.message: discord.Message | None = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(NOT_REQUESTER_MESSAGE, ephemeral=True)
            return False
        if not is_dev_task_allowed(interaction.user.id):
            logger.warning("Denied dev_task result interaction by user_id=%s", interaction.user.id)
            await interaction.response.send_message(DENIED_MESSAGE, ephemeral=True)
            return False
        return True

    def _disable_children(self) -> None:
        for item in self.children:
            item.disabled = True

    async def _reflect_note(self, note: str) -> None:
        if self.message is None:
            return
        try:
            await self.message.edit(content=f"{self.message.content}\n\n→ {note}", view=self)
        except discord.HTTPException:
            logger.exception("Failed to edit dev task result message: task_id=%s", self.task_id)

    @discord.ui.button(label="worktreeを保持", style=discord.ButtonStyle.secondary)
    async def keep_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        outcome = await dev_task_decision.keep_worktree(self.task_id)
        if outcome.already_decided:
            await interaction.response.send_message(ALREADY_PROCESSED_MESSAGE, ephemeral=True)
            return
        if not outcome.success:
            await interaction.response.send_message(outcome.message, ephemeral=True)
            return

        self._disable_children()
        self.stop()
        base_content = interaction.message.content if interaction.message else ""
        await interaction.response.edit_message(content=f"{base_content}\n\n→ {outcome.message}", view=self)

    @discord.ui.button(label="worktreeを破棄", style=discord.ButtonStyle.danger)
    async def discard_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.defer(ephemeral=True)
        outcome = await dev_task_decision.discard_worktree(self.task_id)

        if outcome.already_decided:
            await interaction.followup.send(ALREADY_PROCESSED_MESSAGE, ephemeral=True)
            return
        if outcome.refused:
            # Never force past this - leave the view active so the
            # requester can investigate and try again later.
            await interaction.followup.send(outcome.message, ephemeral=True)
            return
        if not outcome.success:
            await interaction.followup.send(outcome.message, ephemeral=True)
            return

        self._disable_children()
        self.stop()
        await interaction.followup.send(f"→ {outcome.message}", ephemeral=True)
        await self._reflect_note(outcome.message)


class DevTaskApprovalView(discord.ui.View):
    """Not a persistent view (see DevTaskResultView's docstring): a bot
    restart makes any pending approval view unresponsive.
    mark_running_tasks_failed_on_startup() (Phase 1) already handles a task
    that was mid-execution during the restart; a task still sitting in
    pending_approval with a now-dead button can be failed via
    /dev_task_recover.

    Engine override is intentionally NOT persisted the moment the Select
    changes - it is only recorded together with the approval claim itself
    (services.dev_task_executor.execute_approved_task() ->
    services.dev_task_store.claim_task_for_execution()), so the override
    and the approval are always one atomic write, never two separate
    update_task() calls that a concurrent approval could interleave with.
    """

    def __init__(self, task_id: str, requester_id: int) -> None:
        super().__init__(timeout=_APPROVAL_VIEW_TIMEOUT_SECONDS)
        self.task_id = task_id
        self.requester_id = requester_id
        self.message: discord.Message | None = None
        self.pending_engine_override: str | None = None
        self._decided = False

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(NOT_REQUESTER_MESSAGE, ephemeral=True)
            return False
        if not is_dev_task_allowed(interaction.user.id):
            logger.warning("Denied dev_task approval interaction by user_id=%s", interaction.user.id)
            await interaction.response.send_message(DENIED_MESSAGE, ephemeral=True)
            return False
        return True

    def _disable_children(self) -> None:
        for item in self.children:
            item.disabled = True

    async def on_timeout(self) -> None:
        self._disable_children()
        if self.message is not None:
            try:
                await self.message.edit(content=f"{self.message.content}\n\n（承認期限切れ）", view=self)
            except discord.HTTPException:
                logger.exception("Failed to edit dev_task approval message on timeout: task_id=%s", self.task_id)

        result = store.fail_pending_approval(
            self.task_id, reason="承認が時間内に行われなかったため、自動的に失敗としました。"
        )
        if result.success:
            logger.info("Dev task approval timed out: task_id=%s", self.task_id)
        elif result.error_code == store.ERROR_INVALID_TRANSITION:
            # Already claimed (approved) or otherwise moved on - exactly the
            # race fail_pending_approval() exists to make safe. Nothing to do.
            logger.info(
                "Dev task approval timeout no-op: task was no longer pending_approval: task_id=%s",
                self.task_id,
            )
        else:
            logger.error(
                "Failed to fail dev task on approval timeout: task_id=%s error_code=%s",
                self.task_id,
                result.error_code,
            )

    @discord.ui.select(
        placeholder="エンジンを上書き（任意）",
        min_values=0,
        max_values=1,
        options=[
            discord.SelectOption(label="Codex", value="codex"),
            discord.SelectOption(label="Claude", value="claude"),
        ],
    )
    async def engine_override_select(self, interaction: discord.Interaction, select: discord.ui.Select) -> None:
        if self._decided:
            await interaction.response.send_message(ALREADY_PROCESSED_MESSAGE, ephemeral=True)
            return
        if not select.values:
            await interaction.response.defer(ephemeral=True)
            return

        # Deliberately NOT persisted here - see this class's docstring.
        self.pending_engine_override = select.values[0]
        await interaction.response.send_message(
            f"エンジンを{self.pending_engine_override}に変更する設定にしました（承認時に反映されます）。",
            ephemeral=True,
        )

    @discord.ui.button(label="承認して実行", style=discord.ButtonStyle.success)
    async def approve_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if self._decided:
            await interaction.response.send_message(ALREADY_PROCESSED_MESSAGE, ephemeral=True)
            return

        self._decided = True
        self._disable_children()
        self.stop()

        base_content = interaction.message.content if interaction.message else ""
        await interaction.response.edit_message(content=f"{base_content}\n\n承認しました。実行を開始します…", view=self)

        await start_dev_task_execution(
            interaction.client, self.task_id, engine_override=self.pending_engine_override
        )

    @discord.ui.button(label="キャンセル", style=discord.ButtonStyle.danger)
    async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if self._decided:
            await interaction.response.send_message(ALREADY_PROCESSED_MESSAGE, ephemeral=True)
            return

        self._decided = True
        self._disable_children()
        self.stop()

        result = store.fail_pending_approval(self.task_id, reason="起票者によりキャンセルされました。")
        if not result.success:
            logger.error(
                "Failed to cancel dev task: task_id=%s error_code=%s", self.task_id, result.error_code
            )

        base_content = interaction.message.content if interaction.message else ""
        await interaction.response.edit_message(content=f"{base_content}\n\nタスクをキャンセルしました。", view=self)


async def start_dev_task_execution(
    client: discord.Client, task_id: str, *, engine_override: str | None = None
) -> None:
    """Approval -> (slot reservation -> store claim -> worktree creation ->
    dummy execution -> terminal-state save), all via
    services.dev_task_executor.execute_approved_task() - then, ONLY once
    that terminal state is confirmed durably saved, builds and sends the
    Discord notification (DM, falling back to a verified-private channel),
    decoupled from the approval interaction's token lifetime.
    """
    task_result = store.get_task(task_id)
    if not task_result.success:
        logger.error("Dev task disappeared before execution could start: task_id=%s", task_id)
        return
    dev_task = task_result.task

    branch_name = f"dev-task/{task_id}"
    outcome = await dev_task_executor.execute_approved_task(
        task_id,
        base_branch=dev_task.base_branch,
        timeout_seconds=dev_task.timeout_seconds,
        branch_name=branch_name,
        engine_override=engine_override,
        routing_reason_override="承認時に手動で上書きされたため" if engine_override else None,
    )

    if not outcome.claimed:
        # No worktree was ever created - nothing to keep/discard, and the
        # failure reason is already a safe, curated message.
        content = _build_failure_message(
            task_id, engine_override or dev_task.routed_engine, outcome.failure_reason or "不明なエラー"
        )
        await _send_and_record_notification(client, task_id, dev_task.requested_by_user_id, content, view=None)
        return

    if outcome.final_task is None:
        # The terminal state itself could not be durably saved (e.g. a disk
        # write failure) - never claim "completed" or "failed" to the user
        # when that was never actually confirmed in storage.
        logger.error("Dev task finished but final state could not be saved: task_id=%s", task_id)
        content = "\n".join(
            [
                f"【開発タスク結果】{_DUMMY_EXECUTION_NOTICE}",
                f"タスクID: {_short_task_id(task_id)}",
                "状態を保存できなかったため、結果を確定できませんでした。",
                "/dev_task_status で最新の状態を確認してください。",
            ]
        )
        await _send_and_record_notification(client, task_id, dev_task.requested_by_user_id, content, view=None)
        return

    final_task = outcome.final_task
    if outcome.execution_result is not None and outcome.execution_result.success:
        content = _build_success_message(
            task_id, final_task.routed_engine, outcome.branch_name, outcome.execution_result
        )
    else:
        content = _build_failure_message(
            task_id,
            final_task.routed_engine,
            final_task.error_message or outcome.failure_reason or "不明なエラー",
            branch_name=outcome.branch_name,
            duration_seconds=(
                outcome.execution_result.duration_seconds if outcome.execution_result is not None else None
            ),
        )

    result_view = DevTaskResultView(task_id, dev_task.requested_by_user_id) if outcome.worktree_created else None
    await _send_and_record_notification(
        client, task_id, dev_task.requested_by_user_id, content, view=result_view
    )
