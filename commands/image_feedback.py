"""Self-improvement Phase 1: recording user feedback on /generate_ai images.

Deliberately a separate UI surface from ImageRatingView (commands/prompt.py)'s
favorite/like/meh/off_target buttons. Those buttons already fill every
available row on the result message (4 rating rows + 1 regenerate row =
Discord's 5-action-row cap - see commands/prompt.py's MAX_RATEABLE_IMAGES/
REGENERATE_ROW comments), so this feature cannot add components to that
message. Instead it is a message context menu command ("画像にフィードバック
を送る"): right-click (or long-press) the result message itself. This has two
further advantages over trying to reuse the existing view: it keeps working
after ImageRatingView's 1-hour timeout or a bot restart (it looks the
generation up fresh from generation_history.db every time, not from any
live View state), and it does not touch the existing rating feature at all.
"""

import asyncio
import logging

import discord
from discord import app_commands

from services.generation_history import (
    ALLOWED_ISSUE_CATEGORIES,
    FeedbackImageTarget,
    ImageFeedbackRecord,
    get_generation_images_for_message,
    get_generation_request,
    list_feedback_for_user,
)
from services.image_generation_feedback import (
    get_character_feedback_summary,
    get_common_failure_categories,
    save_image_feedback,
)
from services.image_generation_profiles import get_image_generation_profile
from services.pc_control import is_pc_control_allowed


logger = logging.getLogger("secretary-bot.image_feedback")

FEEDBACK_VIEW_TIMEOUT_SECONDS = 600
FEEDBACK_COMMENT_MAX_LENGTH = 1000

FEEDBACK_ISSUE_LABELS = {
    "face_mismatch": "顔が違う",
    "hair_mismatch": "髪型が違う",
    "outfit_mismatch": "衣装が違う",
    "composition_mismatch": "構図が違う",
    "low_quality": "画質が悪い",
    "other": "その他",
}
# Single source of truth for allowed values stays generation_history's
# ALLOWED_ISSUE_CATEGORIES; this just orders/labels them for display.
FEEDBACK_ISSUE_ORDER = tuple(ALLOWED_ISSUE_CATEGORIES)

FEEDBACK_NOT_FOUND_MESSAGE = "このメッセージに紐づく生成画像の履歴が見つかりませんでした。"
FEEDBACK_ONLY_OWNER_MESSAGE = "この画像のフィードバックは、生成を依頼した本人だけが記録できます。"
FEEDBACK_REQUEST_MISSING_MESSAGE = "元の生成情報が見つかりませんでした。履歴が削除されたか、DBが利用できない可能性があります。"
FEEDBACK_SAVED_MESSAGE = "フィードバックを保存しました。"
FEEDBACK_SAVE_FAILED_MESSAGE = "フィードバックの保存に失敗しました。時間をおいてもう一度試してね。"
FEEDBACK_LOOKUP_FAILED_MESSAGE = "生成画像の履歴取得に失敗しました。時間をおいてもう一度試してね。"
FEEDBACK_IMAGE_PICKER_PLACEHOLDER = "評価する画像を選んでね"
FEEDBACK_ISSUE_SELECT_PLACEHOLDER = "問題カテゴリを選択（複数可）"
FEEDBACK_ACCEPT_BUTTON_LABEL = "✅ 採用として保存"
FEEDBACK_COMMENT_MODAL_TITLE = "コメントを追加（任意）"
FEEDBACK_COMMENT_INPUT_LABEL = "コメント"

FEEDBACK_HISTORY_DEFAULT_LIMIT = 10
FEEDBACK_HISTORY_MAX_LIMIT = 25
FEEDBACK_HISTORY_EMPTY_MESSAGE = "まだ記録したフィードバックがありません。生成結果のメッセージを右クリック（またはロングタップ）→アプリ→「画像にフィードバックを送る」から記録できるよ。"
FEEDBACK_HISTORY_FAILED_MESSAGE = "フィードバック履歴の取得に失敗しました。時間をおいてもう一度試してね。"
FEEDBACK_STATS_FAILED_MESSAGE = "フィードバック統計の集計に失敗しました。時間をおいてもう一度試してね。"
FEEDBACK_STATS_EMPTY_MESSAGE = "まだ記録されたフィードバックがありません。"
FEEDBACK_STATS_LOW_SAMPLE_NOTE = "件数が少ないキャラクターは参考程度に見てね。"
FEEDBACK_STATS_TOP_CHARACTER_LIMIT = 10

# /generate_ai (and therefore every image this feature evaluates) is already
# gated to is_pc_control_allowed users only - see commands/prompt.py's
# SD_DENIED_MESSAGE. The history/stats commands below are read paths over
# that same data and are gated the same way /generate_ai_stats already is,
# so a non-permitted user cannot see other people's aggregate feedback
# either.
FEEDBACK_DENIED_MESSAGE = "この操作は、許可されたユーザーだけが実行できます。"


def _display_model_name(model_key: str | None) -> str:
    try:
        return get_image_generation_profile(model_key).display_name
    except ValueError:
        return model_key or "不明"


def _character_names_from_snapshot(snapshot: dict) -> str:
    try:
        resolutions = snapshot.get("character_tags", {}).get("resolutions", [])
    except AttributeError:
        resolutions = []
    names = [entry.get("display_name") for entry in resolutions if entry.get("display_name")]
    return "、".join(names) if names else "-"


def _judgement_text(record: ImageFeedbackRecord) -> str:
    if record.accepted:
        return "✅ 採用"
    labels = [FEEDBACK_ISSUE_LABELS.get(category, category) for category in record.issue_categories]
    return "⚠️ " + "、".join(labels) if labels else "-"


class FeedbackImagePicker(discord.ui.Select):
    def __init__(self, targets: list[FeedbackImageTarget], *, default_image_id: int, row: int) -> None:
        options = [
            discord.SelectOption(
                label=f"画像 {position}",
                value=str(target.image_id),
                default=target.image_id == default_image_id,
            )
            for position, target in enumerate(targets, start=1)
        ]
        super().__init__(
            placeholder=FEEDBACK_IMAGE_PICKER_PLACEHOLDER,
            min_values=1,
            max_values=1,
            options=options,
            row=row,
            custom_id="genai_fb_pick",
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        view = self.view
        if isinstance(view, ImageFeedbackView):
            view.selected_image_id = int(self.values[0])
        try:
            await interaction.response.defer()
        except Exception:
            logger.exception("Failed to acknowledge image feedback picker selection.")


class FeedbackCommentModal(discord.ui.Modal):
    """Opened after FeedbackIssueSelect - collects an optional free-text
    comment, then saves accepted=False with the categories chosen just
    before this modal was opened (view.pending_categories)."""

    def __init__(self, *, view: "ImageFeedbackView") -> None:
        super().__init__(title=FEEDBACK_COMMENT_MODAL_TITLE)
        self._view = view
        self.comment = discord.ui.TextInput(
            label=FEEDBACK_COMMENT_INPUT_LABEL,
            style=discord.TextStyle.paragraph,
            max_length=FEEDBACK_COMMENT_MAX_LENGTH,
            required=False,
        )
        self.add_item(self.comment)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        # Modal submission is a separate interaction from the Select that
        # opened it, and is not covered by ImageFeedbackView.interaction_check
        # (that only guards the View's own components) - re-checked here the
        # same way commands/prompt.py's CustomInstructionModal re-checks
        # owner_id inside on_submit.
        if not self._view.is_allowed(interaction.user.id):
            try:
                await interaction.response.send_message(FEEDBACK_ONLY_OWNER_MESSAGE, ephemeral=True)
            except Exception:
                logger.exception("Failed to reply to feedback comment modal permission check.")
            return

        comment_text = str(self.comment.value).strip() or None
        saved = await self._view.save(
            interaction,
            accepted=False,
            issue_categories=self._view.pending_categories,
            comment=comment_text,
        )
        text = FEEDBACK_SAVED_MESSAGE if saved else FEEDBACK_SAVE_FAILED_MESSAGE
        try:
            await interaction.response.send_message(text, ephemeral=True)
        except Exception:
            logger.exception("Failed to reply to feedback comment modal.")


class FeedbackIssueSelect(discord.ui.Select):
    def __init__(self, *, row: int) -> None:
        options = [
            discord.SelectOption(label=FEEDBACK_ISSUE_LABELS[category], value=category)
            for category in FEEDBACK_ISSUE_ORDER
        ]
        super().__init__(
            placeholder=FEEDBACK_ISSUE_SELECT_PLACEHOLDER,
            min_values=1,
            max_values=len(options),
            options=options,
            row=row,
            custom_id="genai_fb_issues",
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        view = self.view
        if not isinstance(view, ImageFeedbackView):
            return
        view.pending_categories = list(self.values)
        try:
            await interaction.response.send_modal(FeedbackCommentModal(view=view))
        except Exception:
            logger.exception("Failed to open feedback comment modal.")


class FeedbackAcceptButton(discord.ui.Button):
    def __init__(self, *, row: int) -> None:
        super().__init__(
            style=discord.ButtonStyle.success,
            label=FEEDBACK_ACCEPT_BUTTON_LABEL,
            custom_id="genai_fb_accept",
            row=row,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        view = self.view
        if not isinstance(view, ImageFeedbackView):
            return
        saved = await view.save(interaction, accepted=True, issue_categories=(), comment=None)
        text = FEEDBACK_SAVED_MESSAGE if saved else FEEDBACK_SAVE_FAILED_MESSAGE
        try:
            await interaction.response.send_message(text, ephemeral=True)
        except Exception:
            logger.exception("Failed to reply to feedback accept button.")


class ImageFeedbackView(discord.ui.View):
    """Ephemeral, per-invocation view built fresh by the context menu
    handler below - never persisted, never restored across a bot restart.
    That is fine (unlike ImageRatingView, which is deliberately reachable
    for an hour after generation): the context menu itself is always
    available on the original message regardless of this view's lifetime,
    so a fresh one is just re-opened.
    """

    def __init__(
        self,
        targets: list[FeedbackImageTarget],
        *,
        owner_id: int,
        guild_id: int | None,
        channel_id: int | None,
        message_id: int | None,
    ) -> None:
        super().__init__(timeout=FEEDBACK_VIEW_TIMEOUT_SECONDS)
        self.targets = {target.image_id: target for target in targets}
        self.owner_id = owner_id
        self.guild_id = guild_id
        self.channel_id = channel_id
        self.message_id = message_id
        self.selected_image_id = targets[0].image_id
        self.pending_categories: list[str] = []

        row = 0
        if len(targets) > 1:
            self.add_item(FeedbackImagePicker(targets, default_image_id=self.selected_image_id, row=row))
            row += 1
        self.add_item(FeedbackAcceptButton(row=row))
        row += 1
        self.add_item(FeedbackIssueSelect(row=row))

    def is_allowed(self, user_id: int) -> bool:
        return user_id == self.owner_id or is_pc_control_allowed(user_id)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if not self.is_allowed(interaction.user.id):
            try:
                await interaction.response.send_message(FEEDBACK_ONLY_OWNER_MESSAGE, ephemeral=True)
            except Exception:
                logger.exception("Failed to reply to image feedback permission check.")
            return False
        return True

    async def save(
        self,
        interaction: discord.Interaction,
        *,
        accepted: bool,
        issue_categories,
        comment: str | None,
    ) -> bool:
        target = self.targets[self.selected_image_id]
        try:
            request = await asyncio.to_thread(get_generation_request, target.request_id)
        except Exception:
            logger.exception("Failed to load generation request for feedback: request_id=%s", target.request_id)
            request = None
        if request is None:
            return False
        try:
            return await asyncio.to_thread(
                save_image_feedback,
                request=request,
                image=target,
                user_id=interaction.user.id,
                guild_id=self.guild_id,
                channel_id=self.channel_id,
                message_id=self.message_id,
                accepted=accepted,
                issue_categories=issue_categories,
                comment=comment,
            )
        except Exception:
            logger.exception("Unexpected error while saving image feedback: image_id=%s", target.image_id)
            return False


async def _handle_image_feedback_context_menu(interaction: discord.Interaction, message: discord.Message) -> None:
    try:
        targets = await asyncio.to_thread(
            get_generation_images_for_message, message.channel.id, message.id
        )
    except Exception:
        logger.exception("Failed to look up generation images for message: message_id=%s", message.id)
        await interaction.response.send_message(FEEDBACK_LOOKUP_FAILED_MESSAGE, ephemeral=True)
        return

    if not targets:
        await interaction.response.send_message(FEEDBACK_NOT_FOUND_MESSAGE, ephemeral=True)
        return

    owner_id = targets[0].request_user_id
    if interaction.user.id != owner_id and not is_pc_control_allowed(interaction.user.id):
        await interaction.response.send_message(FEEDBACK_ONLY_OWNER_MESSAGE, ephemeral=True)
        return

    view = ImageFeedbackView(
        targets,
        owner_id=owner_id,
        guild_id=interaction.guild_id,
        channel_id=message.channel.id,
        message_id=message.id,
    )
    try:
        await interaction.response.send_message(
            "評価する内容を選んでね。「✅ 採用として保存」は問題カテゴリと同時には選べないよ。",
            view=view,
            ephemeral=True,
        )
    except Exception:
        logger.exception("Failed to open image feedback view: message_id=%s", message.id)


def _format_history_line(record: ImageFeedbackRecord) -> str:
    snapshot = record.generation_snapshot or {}
    request = snapshot.get("request", {}) if isinstance(snapshot, dict) else {}
    character_names = _character_names_from_snapshot(snapshot if isinstance(snapshot, dict) else {})
    model_name = _display_model_name(request.get("model_key"))
    image_section = snapshot.get("image", {}) if isinstance(snapshot, dict) else {}
    seed = image_section.get("seed")
    request_id = request.get("request_id")
    comment = f"\n  コメント: {record.comment}" if record.comment else ""
    return (
        f"- {record.updated_at} / {_judgement_text(record)}\n"
        f"  キャラクター: {character_names} / モデル: {model_name} / seed: {seed}\n"
        f"  generation_id: {record.image_id} / request_id: {request_id}"
        f"{comment}"
    )


async def _slash_image_feedback_history(interaction: discord.Interaction, limit: int = FEEDBACK_HISTORY_DEFAULT_LIMIT) -> None:
    if not is_pc_control_allowed(interaction.user.id):
        logger.warning("Denied /image_feedback_history by user_id=%s", interaction.user.id)
        await interaction.response.send_message(FEEDBACK_DENIED_MESSAGE, ephemeral=True)
        return

    capped_limit = max(1, min(limit, FEEDBACK_HISTORY_MAX_LIMIT))
    await interaction.response.defer(ephemeral=True)
    try:
        records = await asyncio.to_thread(list_feedback_for_user, interaction.user.id, capped_limit)
    except Exception:
        logger.exception("Failed to load feedback history: user_id=%s", interaction.user.id)
        await interaction.followup.send(FEEDBACK_HISTORY_FAILED_MESSAGE, ephemeral=True)
        return

    if not records:
        await interaction.followup.send(FEEDBACK_HISTORY_EMPTY_MESSAGE, ephemeral=True)
        return

    lines = [f"あなたが記録したフィードバック（新しい順、最大{capped_limit}件）:"]
    lines.extend(_format_history_line(record) for record in records)
    text = "\n".join(lines)
    if len(text) > 1900:
        text = text[:1900].rstrip() + "\n...(以下省略)"
    await interaction.followup.send(text, ephemeral=True)


def _build_feedback_stats_embed() -> discord.Embed:
    summaries = get_character_feedback_summary()
    categories = get_common_failure_categories()

    embed = discord.Embed(title="画像フィードバック統計", color=discord.Color.blurple())
    total = sum(summary.total for summary in summaries)
    accepted_total = sum(summary.accepted for summary in summaries)
    embed.add_field(
        name="全体",
        value=f"件数: {total} / 採用: {accepted_total}",
        inline=False,
    )
    category_lines = [f"{FEEDBACK_ISSUE_LABELS.get(entry.category, entry.category)}: {entry.count}" for entry in categories]
    embed.add_field(name="問題カテゴリ別（全体）", value="\n".join(category_lines) or "-", inline=False)

    for summary in summaries[:FEEDBACK_STATS_TOP_CHARACTER_LIMIT]:
        rate = "-" if summary.acceptance_rate is None else f"{summary.acceptance_rate:.0%}"
        value = (
            f"件数: {summary.total} / 採用: {summary.accepted} ({rate})\n"
            f"顔:{summary.face_mismatch} 髪型:{summary.hair_mismatch} 衣装:{summary.outfit_mismatch} "
            f"構図:{summary.composition_mismatch} 画質:{summary.low_quality} その他:{summary.other}"
        )
        embed.add_field(name=summary.display_name, value=value[:1024], inline=False)

    embed.set_footer(text=FEEDBACK_STATS_LOW_SAMPLE_NOTE)
    return embed


async def _slash_image_feedback_stats(interaction: discord.Interaction) -> None:
    if not is_pc_control_allowed(interaction.user.id):
        logger.warning("Denied /image_feedback_stats by user_id=%s", interaction.user.id)
        await interaction.response.send_message(FEEDBACK_DENIED_MESSAGE, ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)
    try:
        embed = await asyncio.to_thread(_build_feedback_stats_embed)
    except Exception:
        logger.exception("Failed to build image feedback stats.")
        await interaction.followup.send(FEEDBACK_STATS_FAILED_MESSAGE, ephemeral=True)
        return
    await interaction.followup.send(embed=embed, ephemeral=True)


def setup(client) -> None:
    feedback_context_menu = app_commands.ContextMenu(
        name="画像にフィードバックを送る",
        callback=_handle_image_feedback_context_menu,
    )
    client.tree.add_command(feedback_context_menu)

    @client.tree.command(name="image_feedback_history", description="自分が記録した画像フィードバックの履歴を表示します。")
    @app_commands.describe(limit=f"表示件数。最大{FEEDBACK_HISTORY_MAX_LIMIT}件")
    async def slash_image_feedback_history(interaction: discord.Interaction, limit: int = FEEDBACK_HISTORY_DEFAULT_LIMIT) -> None:
        await _slash_image_feedback_history(interaction, limit)

    @client.tree.command(name="image_feedback_stats", description="画像フィードバックをキャラクター別・問題カテゴリ別に集計します。")
    async def slash_image_feedback_stats(interaction: discord.Interaction) -> None:
        await _slash_image_feedback_stats(interaction)

    logger.info("Registered image feedback commands: image_feedback_history, image_feedback_stats, context menu")
