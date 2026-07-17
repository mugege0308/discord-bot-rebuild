from __future__ import annotations

import io
import logging

import discord
from discord import app_commands

from services import dev_task_decision
from services import dev_task_executor
from services import dev_task_store as store
from services.dev_task_config import (
    get_dev_task_default_base_branch,
    get_dev_task_timeout_seconds,
    is_dev_task_allowed,
)
from services.dev_task_router import DevTaskRoutingError, route_dev_task
from services.dev_worktree import (
    CONSISTENCY_CONSISTENT,
    CONSISTENCY_FOREIGN_WORKTREE,
    CONSISTENCY_MISSING_PATH,
    CONSISTENCY_MISSING_REGISTRATION,
    CONSISTENCY_NOT_FOUND,
    CONSISTENCY_ORPHAN_BRANCH,
    CONSISTENCY_PARTIAL_STATE,
    get_worktree_info,
    validate_base_branch,
)
from views.dev_task import DevTaskApprovalView, DevTaskResultView


logger = logging.getLogger("secretary-bot.dev_task")

DENIED_MESSAGE = "この操作は、許可されたユーザーだけが実行できます。"
MAX_TASK_LENGTH = 4000

# Discord message `content` is capped at 2000 characters. Leave headroom for
# the header lines and a short notice, and attach the full prompt text as a
# file instead of risking a rejected (or silently truncated) send whenever
# the header + task text would exceed this.
_DISCORD_CONTENT_SAFE_LIMIT = 1900

_CONSISTENCY_LABELS = {
    CONSISTENCY_CONSISTENT: "正常",
    CONSISTENCY_NOT_FOUND: "未作成",
    CONSISTENCY_MISSING_PATH: "登録はあるがフォルダなし",
    CONSISTENCY_MISSING_REGISTRATION: "フォルダはあるが未登録",
    CONSISTENCY_ORPHAN_BRANCH: "ブランチのみ残存",
    CONSISTENCY_PARTIAL_STATE: "不整合な状態",
    CONSISTENCY_FOREIGN_WORKTREE: "別ブランチが登録済み",
}

_ENGINE_CHOICES = [
    app_commands.Choice(name="自動判定", value="auto"),
    app_commands.Choice(name="Codex", value="codex"),
    app_commands.Choice(name="Claude", value="claude"),
]


def _describe_worktree_state(task_id: str) -> str:
    """Never surfaces an absolute filesystem path, raw git stderr, or any
    other internal detail - only a friendly Japanese label for the
    consistency classification services.dev_worktree already computes.
    """
    info_result = get_worktree_info(task_id)
    if not info_result.success:
        return f"確認できませんでした（{info_result.error_code}）"
    return _CONSISTENCY_LABELS.get(info_result.info.consistency, "不明")


def _build_approval_content_and_file(dev_task: store.DevTask) -> tuple[str, discord.File | None]:
    """Keep `content` safely under Discord's 2000-character limit
    regardless of how long prompt_text is (up to MAX_TASK_LENGTH): if the
    header plus the full task text would exceed the safe limit, the full
    text is attached as a UTF-8 .txt file instead of being inlined.
    """
    header_lines = [
        f"タスクID: {dev_task.task_id}",
        f"判定エンジン: {dev_task.routed_engine}",
        f"判定理由: {dev_task.routing_reason}",
        f"ベースブランチ: {dev_task.base_branch}",
        "",
        "タスク内容:",
    ]
    header = "\n".join(header_lines)
    combined = f"{header}\n{dev_task.prompt_text}"
    if len(combined) <= _DISCORD_CONTENT_SAFE_LIMIT:
        return combined, None

    file_bytes = dev_task.prompt_text.encode("utf-8")
    file = discord.File(io.BytesIO(file_bytes), filename=f"dev_task_{dev_task.task_id}.txt")
    notice = f"{header}\n（本文が長いため添付ファイルを参照してください）"
    return notice, file


def _build_status_message(dev_task: store.DevTask) -> str:
    is_current = dev_task_executor.get_current_task_id() == dev_task.task_id
    lines = [
        f"タスクID: {dev_task.task_id}",
        f"状態: {dev_task.status}",
        f"エンジン: {dev_task.routed_engine}",
        f"起票日時: {dev_task.requested_at}",
        f"ベースブランチ: {dev_task.base_branch}",
        f"worktree状態: {_describe_worktree_state(dev_task.task_id)}",
        f"現在実行中: {'はい' if is_current else 'いいえ'}",
    ]
    if dev_task.error_message:
        lines.append(f"エラー内容: {dev_task.error_message}")
    if dev_task.decision:
        lines.append(f"worktreeの判断: {dev_task.decision}")
    return "\n".join(lines)


def _resolve_target_task(task_id: str, requester_id: int) -> store.DevTaskResult:
    stripped_task_id = task_id.strip()
    if stripped_task_id:
        result = store.get_task(stripped_task_id)
    else:
        result = store.get_latest_task_for_user(requester_id)

    if not result.success:
        return result
    if result.task.requested_by_user_id != requester_id:
        # Least-privilege: never let one allowed user look up (or recover)
        # another allowed user's task, even though both may be present in
        # DEV_TASK_ALLOWED_USER_IDS.
        return store.DevTaskResult(
            success=False, task=None, error_message=DENIED_MESSAGE, error_code="not_owner"
        )
    return result


def setup(client: discord.Client) -> None:
    @client.tree.command(name="dev_task", description="Codex/Claude Codeへの開発タスクを起票します。")
    @app_commands.describe(
        task="依頼したい内容",
        engine="実行エンジン（省略時は自動判定）",
        base_branch="ベースブランチ（省略時は既定値を使用）",
    )
    @app_commands.choices(engine=_ENGINE_CHOICES)
    async def slash_dev_task(
        interaction: discord.Interaction, task: str, engine: str = "auto", base_branch: str = ""
    ) -> None:
        if not is_dev_task_allowed(interaction.user.id):
            logger.warning("Denied /dev_task by user_id=%s", interaction.user.id)
            await interaction.response.send_message(DENIED_MESSAGE, ephemeral=True)
            return

        stripped_task = task.strip()
        if not stripped_task:
            await interaction.response.send_message("task を入力してください。", ephemeral=True)
            return
        if len(stripped_task) > MAX_TASK_LENGTH:
            await interaction.response.send_message(
                f"task は{MAX_TASK_LENGTH}文字以内にしてください。", ephemeral=True
            )
            return

        try:
            decision = route_dev_task(stripped_task, engine)
        except DevTaskRoutingError as exc:
            await interaction.response.send_message(f"engineの指定が不正です: {exc}", ephemeral=True)
            return

        if interaction.channel_id is None:
            await interaction.response.send_message(
                "このコマンドはチャンネルまたはDM内で使ってください。", ephemeral=True
            )
            return

        resolved_base_branch = base_branch.strip() or get_dev_task_default_base_branch()

        # Validate the base branch BEFORE ever writing a task JSON: an
        # invalid/nonexistent base branch would otherwise still create a
        # pending_approval task that is guaranteed to fail later at
        # approval time, wasting a worktree-creation attempt for nothing.
        branch_validation = validate_base_branch(resolved_base_branch)
        if not branch_validation.success:
            await interaction.response.send_message(
                f"ベースブランチが不正です: {branch_validation.error_message}", ephemeral=True
            )
            return

        create_result = store.create_task(
            requested_by_user_id=interaction.user.id,
            guild_id=interaction.guild.id if interaction.guild else None,
            channel_id=interaction.channel_id,
            prompt_text=stripped_task,
            routed_engine=decision.engine.value,
            routing_reason=decision.reason,
            engine_overridden=decision.overridden,
            timeout_seconds=get_dev_task_timeout_seconds(),
            base_branch=resolved_base_branch,
        )
        if not create_result.success:
            await interaction.response.send_message(
                f"タスクの保存に失敗しました: {create_result.error_message}", ephemeral=True
            )
            return

        dev_task = create_result.task
        content, file = _build_approval_content_and_file(dev_task)
        view = DevTaskApprovalView(dev_task.task_id, interaction.user.id)

        try:
            if file is not None:
                await interaction.response.send_message(content, view=view, file=file, ephemeral=True)
            else:
                await interaction.response.send_message(content, view=view, ephemeral=True)
        except discord.HTTPException:
            logger.exception("Failed to send dev_task approval UI: task_id=%s", dev_task.task_id)
            # The task JSON was already created - never leave it stuck as an
            # orphaned pending_approval with no way to interact with it.
            store.fail_pending_approval(dev_task.task_id, reason="承認画面の送信に失敗しました。")
            return

        try:
            view.message = await interaction.original_response()
        except discord.HTTPException:
            logger.exception(
                "Failed to fetch original dev_task approval response: task_id=%s", dev_task.task_id
            )

    @client.tree.command(name="dev_task_status", description="開発タスクの状態を確認します。")
    @app_commands.describe(task_id="確認したいタスクIDの完全な値。省略時は自分の最新タスク")
    async def slash_dev_task_status(interaction: discord.Interaction, task_id: str = "") -> None:
        if not is_dev_task_allowed(interaction.user.id):
            logger.warning("Denied /dev_task_status by user_id=%s", interaction.user.id)
            await interaction.response.send_message(DENIED_MESSAGE, ephemeral=True)
            return

        result = _resolve_target_task(task_id, interaction.user.id)
        if not result.success:
            await interaction.response.send_message(
                f"タスクが見つかりません: {result.error_message}", ephemeral=True
            )
            return

        await interaction.response.send_message(_build_status_message(result.task), ephemeral=True)

    @client.tree.command(
        name="dev_task_stop", description="実行中の開発タスクを停止します。"
    )
    async def slash_dev_task_stop(interaction: discord.Interaction) -> None:
        if not is_dev_task_allowed(interaction.user.id):
            logger.warning("Denied /dev_task_stop by user_id=%s", interaction.user.id)
            await interaction.response.send_message(DENIED_MESSAGE, ephemeral=True)
            return

        current_task_id = dev_task_executor.get_current_task_id()
        if current_task_id is None:
            await interaction.response.send_message("現在実行中のタスクはありません。", ephemeral=True)
            return

        task_result = store.get_task(current_task_id)
        if not task_result.success or task_result.task.requested_by_user_id != interaction.user.id:
            await interaction.response.send_message(
                "実行中のタスクは他のユーザーのものです。停止できません。", ephemeral=True
            )
            return

        await interaction.response.defer(ephemeral=True)
        cancelled = await dev_task_executor.cancel_current_execution_and_wait(current_task_id)
        if cancelled:
            await interaction.followup.send("停止しました。結果は通知先を確認してください。", ephemeral=True)
        else:
            await interaction.followup.send(
                "停止に失敗しました（既に終了した可能性があります）。", ephemeral=True
            )

    @client.tree.command(
        name="dev_task_recover",
        description="Bot再起動などで操作できなくなった開発タスクを復旧します。",
    )
    @app_commands.describe(task_id="対象タスクIDの完全な値。省略時は自分の最新タスク")
    async def slash_dev_task_recover(interaction: discord.Interaction, task_id: str = "") -> None:
        if not is_dev_task_allowed(interaction.user.id):
            logger.warning("Denied /dev_task_recover by user_id=%s", interaction.user.id)
            await interaction.response.send_message(DENIED_MESSAGE, ephemeral=True)
            return

        result = _resolve_target_task(task_id, interaction.user.id)
        if not result.success:
            await interaction.response.send_message(
                f"タスクが見つかりません: {result.error_message}", ephemeral=True
            )
            return

        dev_task = result.task

        if dev_task.status == store.DevTaskStatus.PENDING_APPROVAL.value:
            fail_result = store.fail_pending_approval(
                dev_task.task_id,
                reason="Bot再起動後、承認画面が失われたため /dev_task_recover により失敗としました。",
            )
            if fail_result.success:
                await interaction.response.send_message(
                    "承認待ちのまま操作できなくなっていたタスクを失敗として記録しました。", ephemeral=True
                )
            else:
                await interaction.response.send_message(
                    f"復旧に失敗しました: {fail_result.error_message}", ephemeral=True
                )
            return

        if dev_task.status in (store.DevTaskStatus.COMPLETED.value, store.DevTaskStatus.FAILED.value):
            if dev_task.decision is not None:
                await interaction.response.send_message(
                    f"このタスクは既に「{dev_task.decision}」と判断済みです。", ephemeral=True
                )
                return
            if dev_task.branch_name is None:
                await interaction.response.send_message(
                    "このタスクにはworktreeが作成されなかったため、復旧の必要はありません。", ephemeral=True
                )
                return

            view = DevTaskResultView(dev_task.task_id, interaction.user.id)
            try:
                message = await interaction.user.send(
                    f"（再送）タスクID: {dev_task.task_id} の結果操作を復元しました。\n"
                    f"worktreeを保持または破棄してください。",
                    view=view,
                )
                view.message = message
            except discord.HTTPException:
                logger.exception("Failed to resend dev_task result view: task_id=%s", dev_task.task_id)
                await interaction.response.send_message(
                    "DMの送信に失敗しました。DMを許可しているか確認してください。", ephemeral=True
                )
                return

            await interaction.response.send_message("DMへ結果操作を再送しました。", ephemeral=True)
            return

        await interaction.response.send_message("このタスクに復旧が必要な状態はありません。", ephemeral=True)

    logger.info(
        "Registered dev task slash commands: dev_task, dev_task_status, dev_task_stop, dev_task_recover"
    )
