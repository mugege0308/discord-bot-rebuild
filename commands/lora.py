"""Discord preparation commands for LoRA datasets.

The external LoRA manager remains responsible for actually running training.
These commands make the input contract explicit and require a preview
confirmation before changing captions or saving the sidecar metadata.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import tomllib
import uuid
from pathlib import Path
from typing import Any

import discord
from discord import app_commands

from services.lora_training import (
    LORA_TYPE_ADDITIONAL,
    LORA_TYPE_CHARACTER,
    LORA_TYPE_LABELS,
    LORA_TYPE_STYLE,
    IMAGE_EXTENSIONS,
    CaptionPreview,
    LoraMetadata,
    LoraTrainingError,
    apply_lora_captions,
    format_caption_preview,
    load_lora_registration_defaults,
    preview_lora_captions,
    resolve_lora_file,
    save_lora_metadata,
    validate_training_metadata,
)
from services.pc_control import is_pc_control_allowed
from services.lora_catalog import refresh_comfy_styles
from services.lora_paths import get_lora_manager_outputs_dir
from services import lora_job, lora_upload


logger = logging.getLogger("secretary-bot.lora")


def _new_diagnostic_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


LORA_TYPE_CHOICES = [
    app_commands.Choice(name=LORA_TYPE_LABELS[LORA_TYPE_STYLE], value=LORA_TYPE_STYLE),
    app_commands.Choice(name=LORA_TYPE_LABELS[LORA_TYPE_CHARACTER], value=LORA_TYPE_CHARACTER),
    app_commands.Choice(name=LORA_TYPE_LABELS[LORA_TYPE_ADDITIONAL], value=LORA_TYPE_ADDITIONAL),
]

# Temporary diagnostic switch.  Keep the in-process LoRA candidate cache
# current, but do not mutate the live /generate, /generate_raw, or
# /generate_ai command definitions until the post-registration issue is
# isolated.  Re-enable by changing this one value back to True.
ENABLE_LORA_SLASH_COMMAND_CHOICE_REFRESH = False


def _preview_message(preview: CaptionPreview) -> str:
    return (
        "LoRA学習用captionのプレビューです。\n\n"
        + format_caption_preview(preview)
        + "\n\nこの内容を適用するには /lora_prepare を実行して確認ボタンを押してください。"
    )[:1900]


def _refresh_lora_choice_cache(client: discord.Client) -> None:
    if not ENABLE_LORA_SLASH_COMMAND_CHOICE_REFRESH:
        logger.info("LoRA metadata registration: live Slash Command choice refresh is disabled for this service-only commit.")
        return
    del client
    logger.warning("LoRA live generation choice refresh is not connected in Commit 3A-3.")


def _resolve_dataset_dir_for_prepare(value: str) -> Path:
    """Resolve a managed dataset name to its configured image directory."""
    raw = value.strip()
    requested = Path(raw)
    if not requested.is_absolute() and len(requested.parts) == 1:
        configured_root = os.getenv("LORA_MANAGER_ROOT", "").strip().strip('"')
        manager_root = Path(configured_root).resolve() if configured_root else get_lora_manager_outputs_dir().resolve().parent
        managed = (manager_root / "lora_datasets" / raw).resolve()
        if managed.is_dir():
            searched = [managed]
            if _has_direct_training_image(managed):
                # Preserve the older layout where the supplied directory
                # itself contains the images and captions.
                return managed

            for candidate in _dataset_toml_image_dirs(managed):
                searched.append(candidate)
                if _has_direct_training_image(candidate):
                    return candidate

            train_dir = managed / "train"
            searched.append(train_dir)
            if _has_direct_training_image(train_dir):
                return train_dir

            displayed = ", ".join(str(path) for path in searched)
            raise LoraTrainingError(
                f"dataset_dir '{raw}' を解決しましたが対応画像がありません。検索先: {displayed}"
            )
    return requested


def _has_direct_training_image(directory: Path) -> bool:
    try:
        return any(
            path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
            for path in directory.iterdir()
        )
    except OSError:
        return False


def _dataset_toml_image_dirs(dataset_root: Path) -> list[Path]:
    """Read safe image_dir values from plans belonging to this dataset."""

    plans_root = dataset_root / "training" / "plans"
    candidates: list[Path] = []
    if not plans_root.is_dir():
        return candidates
    for dataset_toml in sorted(plans_root.glob("*/dataset.toml"), key=lambda path: str(path).casefold()):
        try:
            with dataset_toml.open("rb") as handle:
                document = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError):
            continue
        for dataset in document.get("datasets", []):
            if not isinstance(dataset, dict):
                continue
            for subset in dataset.get("subsets", []):
                if not isinstance(subset, dict) or not isinstance(subset.get("image_dir"), str):
                    continue
                raw_image_dir = Path(subset["image_dir"])
                candidate = raw_image_dir if raw_image_dir.is_absolute() else dataset_toml.parent / raw_image_dir
                try:
                    resolved = candidate.resolve()
                    resolved.relative_to(dataset_root.resolve())
                except (OSError, ValueError):
                    # Do not allow a dataset name to escape the managed root
                    # through an arbitrary image_dir in a TOML file.
                    continue
                if resolved not in candidates:
                    candidates.append(resolved)
    return candidates


def resolve_lora_prepare_metadata(
    *,
    lora_path: Path,
    lora_name: str,
    lora_type: str | None = None,
    trigger_tag: str | None = None,
    recommended_strength: float | None = None,
) -> tuple[LoraMetadata, str]:
    """Resolve optional command values over saved training metadata."""

    saved_metadata, defaults_source = load_lora_registration_defaults(lora_path)
    selected_type = lora_type if lora_type is not None else saved_metadata.lora_type
    selected_trigger = saved_metadata.trigger_tag if trigger_tag is None else trigger_tag
    selected_strength = (
        saved_metadata.recommended_strength
        if recommended_strength is None
        else recommended_strength
    )
    return (
        validate_training_metadata(
            name=lora_name,
            lora_type=selected_type,
            trigger_tag=selected_trigger or "",
            checkpoint=saved_metadata.checkpoint,
            model_family=saved_metadata.model_family,
            recommended_strength=selected_strength,
        ),
        defaults_source,
    )


def _upload_view(session: dict[str, Any], owner_id: int, *, page: int = 0) -> "LoraUploadView":
    return LoraUploadView(session_id=str(session["session_id"]), owner_id=owner_id, page=page)


class LoraUploadView(discord.ui.View):
    """添付受付を明示的な確認・確定・キャンセルに分けるビュー。"""

    PAGE_SIZE = 15
    def __init__(self, *, session_id: str, owner_id: int, page: int = 0) -> None:
        super().__init__(timeout=1800)
        self.session_id = session_id
        self.owner_id = owner_id
        self.page = page
        session = lora_upload.get_session(session_id)
        items = session.get("items", [])
        start = page * self.PAGE_SIZE
        for item in items[start : start + self.PAGE_SIZE]:
            item_id = int(item["id"])
            included = lora_upload.is_item_included(item)
            button = discord.ui.Button(
                label=f"{'除外' if included else '復帰'} {item_id}",
                style=discord.ButtonStyle.secondary,
                row=(item_id - start - 1) // 5,
            )

            async def toggle(interaction: discord.Interaction, button: discord.ui.Button, item_id: int = item_id) -> None:
                del button
                if not await self._check_owner(interaction):
                    return
                try:
                    current = lora_upload.get_session(self.session_id)
                    selected = next(item for item in current.get("items", []) if int(item["id"]) == item_id)
                    lora_upload.toggle_item(self.session_id, item_id, not bool(selected.get("included", True)))
                    updated = lora_upload.get_session(self.session_id)
                    await interaction.response.edit_message(
                        content=lora_upload.format_summary(updated, page=self.page),
                        view=_upload_view(updated, self.owner_id, page=self.page),
                    )
                except lora_upload.LoraUploadError as exc:
                    await interaction.response.send_message(str(exc), ephemeral=True)

            button.callback = toggle
            self.add_item(button)
        if page > 0:
            previous = discord.ui.Button(label="前ページ", style=discord.ButtonStyle.secondary, row=4)

            async def previous_page(interaction: discord.Interaction, button: discord.ui.Button) -> None:
                del button
                if await self._check_owner(interaction):
                    session = lora_upload.get_session(self.session_id)
                    await interaction.response.edit_message(
                        content=lora_upload.format_summary(session, page=self.page - 1),
                        view=_upload_view(session, self.owner_id, page=self.page - 1),
                    )

            previous.callback = previous_page
            self.add_item(previous)
        if len(items) > start + self.PAGE_SIZE:
            next_page = discord.ui.Button(label="次ページ", style=discord.ButtonStyle.secondary, row=4)

            async def next_page_callback(interaction: discord.Interaction, button: discord.ui.Button) -> None:
                del button
                if await self._check_owner(interaction):
                    session = lora_upload.get_session(self.session_id)
                    await interaction.response.edit_message(
                        content=lora_upload.format_summary(session, page=self.page + 1),
                        view=_upload_view(session, self.owner_id, page=self.page + 1),
                    )

            next_page.callback = next_page_callback
            self.add_item(next_page)

    async def _check_owner(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("この確認ボタンは実行者本人だけが押せます。", ephemeral=True)
            return False
        try:
            session = lora_upload.get_session(self.session_id)
        except lora_upload.LoraUploadError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return False
        if session.get("status") == "cancelled" and session.get("cancel_reason") == "replaced_by_new_session":
            await interaction.response.send_message("この受付は新しいセッションに置き換えられました", ephemeral=True)
            return False
        if session.get("status") != "open":
            await interaction.response.send_message("この受付はすでに確定済みか終了しています。", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="受付内容を確認", style=discord.ButtonStyle.primary, row=4)
    async def review(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        del button
        if await self._check_owner(interaction):
            session = lora_upload.get_session(self.session_id)
            await interaction.response.edit_message(
                content=lora_upload.format_summary(session, page=self.page),
                view=_upload_view(session, self.owner_id, page=self.page),
            )

    @discord.ui.button(label="画像を追加", style=discord.ButtonStyle.secondary, row=4)
    async def add_more(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        del button
        if await self._check_owner(interaction):
            await interaction.response.send_message("このチャンネルへ画像またはZIPを添付してください。", ephemeral=True)

    @discord.ui.button(label="確定してplan作成", style=discord.ButtonStyle.success, row=4)
    async def finalize(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        del button
        if not await self._check_owner(interaction):
            return
        await interaction.response.defer(ephemeral=True)
        try:
            result = await lora_job.job_manager.create_plan(self.session_id, self.owner_id)
        except (lora_job.LoraJobError, lora_upload.LoraUploadError) as exc:
            await interaction.followup.send(f"planを作成できませんでした。正式データは上書きしていません。\n{exc}", ephemeral=True)
            return
        await interaction.followup.send(
            "学習データセットとplanを作成しました。まだ学習は開始していません。\n"
            + _format_plan_preview(result)
            + f"\n学習開始は /lora_train の選択後、確認ボタンを押してください。\nplan選択ID: `{self.session_id}`",
            ephemeral=True,
        )

    @discord.ui.button(label="キャンセル", style=discord.ButtonStyle.danger, row=4)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        del button
        if not await self._check_owner(interaction):
            return
        try:
            lora_upload.cancel_session(self.session_id)
        except lora_upload.LoraUploadError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        self.stop()
        await interaction.response.edit_message(content="アップロード受付をキャンセルしました。一時ファイルだけを削除しました。", view=None)


def _format_plan_preview(plan: dict[str, Any]) -> str:
    settings = plan.get("settings", {})
    subsets = settings.get("subsets") if isinstance(settings, dict) else None
    if isinstance(subsets, list) and subsets:
        subset_lines = [
            f"{item.get('path', 'subset')}: {item.get('image_count', '?')}枚 × {item.get('repeats', '?')}"
            for item in subsets
            if isinstance(item, dict)
        ]
        repeat_text = "subset別:\n" + "\n".join(subset_lines)
    else:
        repeat_text = f"repeats: {settings.get('repeats', '?')}"
    return (
        f"データセット: {plan.get('dataset_name', '不明')} / LoRA: {plan.get('output_name', '不明')}\n"
        f"種別: {plan.get('lora_type', '不明')} / trigger_tag: {plan.get('trigger_tag') or 'なし'} / "
        f"推奨強度: {plan.get('recommended_strength', '?')}\n"
        f"checkpoint: {Path(str(plan.get('checkpoint', '不明'))).name} / model family: {plan.get('model_family', '不明')}\n"
        f"画像: {plan.get('image_count', '?')}枚 / 総step: {plan.get('total_steps', settings.get('total_steps', '?'))}\n"
        f"{repeat_text} / epochs: {settings.get('epochs', '?')} / "
        f"resolution: {settings.get('resolution', '?')} / rank: {settings.get('network_dim', '?')} / alpha: {settings.get('network_alpha', '?')}\n"
        f"learning rate: {settings.get('learning_rate', '?')} / optimizer: {settings.get('optimizer', '?')} / "
        f"scheduler: {settings.get('scheduler', '?')} / batch size: {settings.get('batch_size', '?')}\n"
        "安全設定: U-Netのみ／Text Encoder学習なし"
    )


class LoraTrainConfirmView(discord.ui.View):
    def __init__(self, *, owner_id: int, session_id: str) -> None:
        super().__init__(timeout=300)
        self.owner_id = owner_id
        self.session_id = session_id
        self.resolved = False

    async def _check_owner(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.owner_id:
            return True
        await interaction.response.send_message("この確認ボタンは実行者本人だけが押せます。", ephemeral=True)
        return False

    @discord.ui.button(label="確認して学習開始", style=discord.ButtonStyle.success)
    async def start(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        del button
        if not await self._check_owner(interaction):
            return
        if self.resolved:
            await interaction.response.send_message("この確認はすでに完了しています。", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        try:
            job = await lora_job.job_manager.start(
                self.session_id, self.owner_id,
                channel_id=interaction.channel_id,
                message_id=interaction.message.id if interaction.message else None,
            )
        except lora_job.LoraJobError as exc:
            await interaction.followup.send(f"学習を開始できませんでした。既存LoRAやデータは変更していません。\n{exc}", ephemeral=True)
            return
        self.resolved = True
        self.stop()
        progress_message = await interaction.followup.send(
            f"学習を開始しました。\n{_format_job_status(job)}\n/lora_status で確認できます。",
            wait=True,
        )
        if progress_message is not None and interaction.channel_id is not None:
            try:
                lora_job.job_manager.update_notification_target(
                    str(job["job_id"]), channel_id=interaction.channel_id, message_id=progress_message.id
                )
            except lora_job.LoraJobError:
                logger.warning("Could not persist LoRA progress message target.", exc_info=True)

    @discord.ui.button(label="キャンセル", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        del button
        if await self._check_owner(interaction):
            self.resolved = True
            self.stop()
            await interaction.response.edit_message(content="学習開始をキャンセルしました。planと生成済みデータは削除していません。", view=None)


class LoraCancelConfirmView(discord.ui.View):
    def __init__(self, *, owner_id: int, job_id: str) -> None:
        super().__init__(timeout=180)
        self.owner_id = owner_id
        self.job_id = job_id
        self.resolved = False

    @discord.ui.button(label="確認して安全停止", style=discord.ButtonStyle.danger)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        del button
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("この確認ボタンは実行者本人だけが押せます。", ephemeral=True)
            return
        if self.resolved:
            await interaction.response.send_message("この確認はすでに完了しています。", ephemeral=True)
            return
        self.resolved = True
        self.stop()
        try:
            await lora_job.job_manager.cancel(self.job_id, self.owner_id)
        except lora_job.LoraJobError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        await interaction.response.edit_message(content="Botが開始した対象ジョブへ停止要求を送りました。完了状態は /lora_status で確認してください。", view=None)

    @discord.ui.button(label="キャンセル", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        del button
        if interaction.user.id == self.owner_id:
            self.stop()
            await interaction.response.edit_message(content="学習停止をキャンセルしました。", view=None)


def _format_job_status(job: dict[str, Any]) -> str:
    progress = lora_job.progress_snapshot(job)
    started = bool(progress.get("training_started")) or (
        isinstance(progress.get("step"), int) and progress.get("step", 0) > 0
    )
    step = progress.get("step", "?") if started else "準備中"
    total = progress.get("total_steps", "?")
    percent = progress.get("percent") if started else None
    percent_text = f" / 進捗: {percent:.1f}%" if isinstance(percent, (int, float)) else ""
    speed = progress.get("speed_text") or "速度取得前"
    if not started:
        eta = "準備中"
    elif progress.get("eta"):
        eta = progress["eta"]
    elif isinstance(progress.get("eta_seconds"), (int, float)):
        eta = f"約{int(progress['eta_seconds'])}秒"
    else:
        eta = "取得不可"
    loss = progress.get("loss")
    return (
        f"ジョブ: `{job.get('job_id', '不明')}` / データセット: {job.get('dataset_name', '不明')}\n"
        f"LoRA: {job.get('output_name', '不明')} / 種別: {job.get('lora_type', '不明')}\n"
        f"状態: {job.get('status', '不明')} / step: {step} / 全step: {total}{percent_text}\n"
        f"速度: {speed} / 推定残り: {eta} / loss: {loss if loss is not None else '取得不可'}"
    )


def _loaded_plan_payload(plan: dict[str, Any]) -> dict[str, Any]:
    metadata = plan.get("lora_metadata", {})
    return {
        "dataset_name": plan.get("dataset_name"),
        "output_name": plan.get("output_name"),
        "lora_type": metadata.get("lora_type", plan.get("settings", {}).get("lora_type")),
        "trigger_tag": metadata.get("trigger_tag", plan.get("settings", {}).get("trigger_tag", "")),
        "recommended_strength": metadata.get("recommended_strength", plan.get("settings", {}).get("recommended_strength")),
        "checkpoint": plan.get("paths", {}).get("checkpoint"),
        "model_family": metadata.get("model_family", plan.get("model_family", "unknown")),
        "image_count": plan.get("image_count"),
        "total_steps": plan.get("settings", {}).get("total_steps"),
        "settings": plan.get("settings", {}),
    }


def _find_owner_job(owner_id: int, job_id: str | None) -> dict[str, Any]:
    if job_id:
        return lora_job.job_manager.get(job_id)
    jobs = lora_job.job_manager.list_for_owner(owner_id)
    if not jobs:
        raise lora_job.LoraJobError("学習ジョブがありません。")
    return next((job for job in jobs if job.get("status") in {"starting", "running"}), jobs[0])


async def handle_attachment_message(message: discord.Message) -> bool:
    """通常メッセージに添付された受付ファイルを処理したか返す。"""
    if not message.attachments:
        return False
    sessions = lora_upload.list_sessions(message.author.id)
    if not sessions:
        return False
    session = sessions[0]
    if session.get("channel_id") not in {None, message.channel.id}:
        await message.channel.send("このアップロード受付は開始したチャンネルで添付してください。")
        return True
    results: list[str] = []
    for attachment in message.attachments:
        try:
            data = await attachment.read()
            result = lora_upload.ingest_attachment(
                str(session["session_id"]), filename=attachment.filename, data=data
            )
            if result.get("duplicate"):
                results.append(f"{attachment.filename}: 重複のため除外")
            elif "images" in result:
                results.append(f"{attachment.filename}: {result.get('image_count', 0)}枚")
            else:
                results.append(f"{attachment.filename}: 追加しました")
        except lora_upload.LoraUploadError as exc:
            results.append(f"{attachment.filename}: {exc}")
        except Exception:
            logger.exception("Failed to receive LoRA attachment")
            results.append(f"{attachment.filename}: 読み込みに失敗しました")
    updated = lora_upload.get_session(str(session["session_id"]))
    await message.channel.send("\n".join(results) + "\n\n" + lora_upload.format_summary(updated))
    return True


async def recover_jobs() -> list[dict[str, Any]]:
    return await lora_job.job_manager.recover()


async def shutdown() -> None:
    await lora_job.job_manager.shutdown()


def validate_startup_configuration() -> bool:
    """起動時にLoRA経路を検証し、秘密値をログへ出さない。"""
    try:
        configuration = lora_job.validate_configuration()
    except lora_job.LoraJobError as exc:
        logger.error("LoRA連携設定の起動時検証に失敗しました: %s", exc)
        return False
    logger.info(
        "LoRA連携設定の起動時検証に成功しました (固定実行経路、%s)",
        Path(configuration["train_script"]).name,
    )
    return True


async def _notify_job(client: discord.Client, job: dict[str, Any], status: str) -> None:
    channel_id = job.get("channel_id")
    if not isinstance(channel_id, int):
        return
    channel = client.get_channel(channel_id)
    if channel is None:
        try:
            channel = await client.fetch_channel(channel_id)
        except Exception:
            return
    if not isinstance(channel, discord.abc.Messageable):
        return
    if status == "progress":
        message_id = job.get("message_id")
        if not isinstance(message_id, int):
            return
        try:
            progress_message = await channel.fetch_message(message_id)
            await progress_message.edit(content=_format_job_status(job))
        except Exception:
            logger.debug("Could not update LoRA progress message.", exc_info=True)
        return
    message = f"LoRA学習の状態が変わりました。\n{_format_job_status(job)}"
    if status == "success":
        output_name = Path(str(job.get("expected_output", ""))).name
        dataset_dir = Path(str(job.get("dataset_dir", ""))).name or str(job.get("dataset_name", ""))
        message += (
            f"\n学習済みLoRA: `{output_name}`"
            "\nmetadata登録は自動実行していません。内容を確認して、次を実行してください。"
            f"\n`/lora_prepare lora_name:{job.get('output_name')} dataset_dir:{dataset_dir} lora_file:{output_name}`"
        )
    elif status in {"failed", "stopped"}:
        message += f"\n理由: {job.get('error_summary') or '詳細はPC側の学習ログを確認してください。'}"
    await channel.send(message[:1900])


class LoraPreparationView(discord.ui.View):
    def __init__(
        self,
        *,
        owner_id: int,
        preview: CaptionPreview,
        lora_path: Path,
        metadata: LoraMetadata,
        diagnostic_id: str | None = None,
    ) -> None:
        super().__init__(timeout=300)
        self.owner_id = owner_id
        self.preview = preview
        self.lora_path = lora_path
        self.metadata = metadata
        self.diagnostic_id = diagnostic_id or _new_diagnostic_id("lora-register")
        self.resolved = False

    async def _check_owner(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.owner_id:
            return True
        await interaction.response.send_message("この確認ボタンは実行者本人だけが押せます。", ephemeral=True)
        return False

    @discord.ui.button(label="captionとメタデータを適用", style=discord.ButtonStyle.success)
    async def apply(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        del button
        diagnostic_id = self.diagnostic_id
        started_at = time.perf_counter()
        logger.info(
            "LoRA metadata registration start diagnostic_id=%s user_id=%s lora_file=%s dataset_dir=%s items=%s",
            diagnostic_id,
            interaction.user.id,
            self.lora_path,
            self.preview.dataset_dir,
            len(self.preview.items),
        )
        if not await self._check_owner(interaction):
            logger.info(
                "LoRA metadata registration rejected diagnostic_id=%s reason=owner_check",
                diagnostic_id,
            )
            return
        if self.resolved:
            logger.info(
                "LoRA metadata registration rejected diagnostic_id=%s reason=already_resolved",
                diagnostic_id,
            )
            await interaction.response.send_message("この確認はすでに完了しています。", ephemeral=True)
            return
        self.resolved = True
        self.stop()
        logger.info(
            "LoRA metadata registration confirmation accepted diagnostic_id=%s view_stopped=true",
            diagnostic_id,
        )
        logger.info(
            "LoRA metadata registration step=interaction_defer start diagnostic_id=%s",
            diagnostic_id,
        )
        try:
            await interaction.response.defer(ephemeral=True)
        except Exception:
            logger.exception(
                "LoRA metadata registration step=interaction_defer failed diagnostic_id=%s elapsed_ms=%s",
                diagnostic_id,
                _elapsed_ms(started_at),
            )
            raise
        logger.info(
            "LoRA metadata registration step=interaction_defer complete diagnostic_id=%s",
            diagnostic_id,
        )
        try:
            step_started_at = time.perf_counter()
            logger.info(
                "LoRA metadata registration step=apply_captions start diagnostic_id=%s executor=thread",
                diagnostic_id,
            )
            changed = await asyncio.to_thread(
                apply_lora_captions, self.preview, operation_id=diagnostic_id
            )
            logger.info(
                "LoRA metadata registration step=apply_captions complete diagnostic_id=%s elapsed_ms=%s changed_count=%s",
                diagnostic_id,
                _elapsed_ms(step_started_at),
                changed,
            )

            step_started_at = time.perf_counter()
            logger.info(
                "LoRA metadata registration step=save_metadata start diagnostic_id=%s executor=thread",
                diagnostic_id,
            )
            metadata_path = await asyncio.to_thread(
                save_lora_metadata,
                self.lora_path,
                self.metadata,
                operation_id=diagnostic_id,
            )
            logger.info(
                "LoRA metadata registration step=save_metadata complete diagnostic_id=%s elapsed_ms=%s metadata_path=%s",
                diagnostic_id,
                _elapsed_ms(step_started_at),
                metadata_path,
            )
        except LoraTrainingError as exc:
            logger.error(
                "LoRA metadata registration failed diagnostic_id=%s stage=caption_or_metadata elapsed_ms=%s error=%s",
                diagnostic_id,
                _elapsed_ms(started_at),
                exc,
                exc_info=True,
            )
            try:
                await interaction.followup.send(
                    f"LoRAメタデータ登録に失敗しました: {exc}", ephemeral=True
                )
            except Exception:
                logger.exception(
                    "LoRA metadata registration error response failed diagnostic_id=%s",
                    diagnostic_id,
                )
                raise
            return
        except asyncio.CancelledError:
            logger.error(
                "LoRA metadata registration cancelled diagnostic_id=%s stage=caption_or_metadata elapsed_ms=%s",
                diagnostic_id,
                _elapsed_ms(started_at),
                exc_info=True,
            )
            raise
        except Exception:
            logger.exception(
                "LoRA metadata registration unexpected failure diagnostic_id=%s stage=caption_or_metadata elapsed_ms=%s",
                diagnostic_id,
                _elapsed_ms(started_at),
            )
            raise

        cache_updated = True
        try:
            step_started_at = time.perf_counter()
            logger.info(
                "LoRA metadata registration step=refresh_comfy_styles start diagnostic_id=%s executor=thread",
                diagnostic_id,
            )
            refreshed_styles = await asyncio.to_thread(refresh_comfy_styles)
            logger.info(
                "LoRA metadata registration step=refresh_comfy_styles complete diagnostic_id=%s elapsed_ms=%s style_count=%s",
                diagnostic_id,
                _elapsed_ms(step_started_at),
                len(refreshed_styles),
            )

            step_started_at = time.perf_counter()
            logger.info(
                "LoRA metadata registration step=choice_cache_refresh start diagnostic_id=%s executor=event_loop_sync",
                diagnostic_id,
            )
            _refresh_lora_choice_cache(interaction.client)
            logger.info(
                "LoRA metadata registration step=choice_cache_refresh complete diagnostic_id=%s elapsed_ms=%s",
                diagnostic_id,
                _elapsed_ms(step_started_at),
            )
        except Exception:
            cache_updated = False
            logger.error(
                "LoRA metadata registration step=choice_cache_refresh failed diagnostic_id=%s elapsed_ms=%s",
                diagnostic_id,
                _elapsed_ms(started_at),
                exc_info=True,
            )
        cache_message = (
            "Bot内のLoRA候補catalogを更新しました。"
            if cache_updated else
            "LoRA候補catalogの更新に失敗しました。次回再起動時に再読込されます。"
        )
        logger.info(
            "LoRA metadata registration step=final_response start diagnostic_id=%s",
            diagnostic_id,
        )
        try:
            await interaction.followup.send(
                f"完成済みLoRAのメタデータを登録しました。caption変更: {changed}件\n"
                f"メタデータ: {metadata_path}\n{cache_message}",
                ephemeral=True,
            )
        except Exception:
            logger.exception(
                "LoRA metadata registration final response failed diagnostic_id=%s elapsed_ms=%s",
                diagnostic_id,
                _elapsed_ms(started_at),
            )
            raise
        logger.info(
            "LoRA metadata registration step=final_response complete diagnostic_id=%s elapsed_ms=%s",
            diagnostic_id,
            _elapsed_ms(started_at),
        )
        logger.info(
            "LoRA metadata registration complete diagnostic_id=%s elapsed_ms=%s changed_count=%s cache_updated=%s",
            diagnostic_id,
            _elapsed_ms(started_at),
            changed,
            cache_updated,
        )

    @discord.ui.button(label="キャンセル", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        del button
        if not await self._check_owner(interaction):
            return
        self.resolved = True
        self.stop()
        await interaction.response.edit_message(content="LoRA学習準備をキャンセルしました。", view=None)


def setup(client: discord.Client) -> None:
    async def notify_job(job: dict[str, Any], status: str) -> None:
        await _notify_job(client, job, status)

    lora_job.job_manager.set_notifier(notify_job)

    @client.tree.command(
        name="lora_upload",
        description="LoRA学習用画像のアップロード受付を開始します。",
    )
    @app_commands.choices(lora_type=LORA_TYPE_CHOICES)
    @app_commands.describe(
        dataset_name="新しいデータセット名。既存名は上書きしません。",
        lora_type="character / style / additional",
        trigger_tag="characterでは必須。style / additionalでは既存仕様どおり任意です。",
        recommended_strength="省略時0.5。0.0～1.0",
    )
    async def slash_lora_upload(
        interaction: discord.Interaction,
        dataset_name: str,
        lora_type: app_commands.Choice[str],
        trigger_tag: str | None = None,
        recommended_strength: float | None = None,
    ) -> None:
        if not is_pc_control_allowed(interaction.user.id):
            await interaction.response.send_message("この操作は、許可されたユーザーだけが実行できます。", ephemeral=True)
            return
        try:
            session = lora_upload.create_session(
                owner_id=interaction.user.id,
                guild_id=interaction.guild_id,
                channel_id=interaction.channel_id or 0,
                message_id=None,
                dataset_name=dataset_name,
                lora_type=lora_type.value,
                trigger_tag=trigger_tag or "",
                recommended_strength=0.5 if recommended_strength is None else recommended_strength,
            )
        except lora_upload.LoraUploadError as exc:
            await interaction.response.send_message(f"アップロード受付を開始できませんでした: {exc}", ephemeral=True)
            return
        await interaction.response.send_message(
            "アップロード受付を開始しました。このチャンネルへPNG/JPG/JPEG/WebPまたはZIPを添付してください。\n"
            "添付後に受付内容を確認し、除外対象があればボタンで切り替えてください。\n\n"
            + lora_upload.format_summary(session),
            view=_upload_view(session, interaction.user.id),
            ephemeral=True,
        )

    @client.tree.command(
        name="lora_train",
        description="作成済みLoRA planをプレビューして学習を開始します。",
    )
    @app_commands.describe(session_id="/lora_uploadで表示されたplan選択ID")
    async def slash_lora_train(interaction: discord.Interaction, session_id: str) -> None:
        await interaction.response.defer(ephemeral=True)
        if not is_pc_control_allowed(interaction.user.id):
            await interaction.followup.send("この操作は、許可されたユーザーだけが実行できます。", ephemeral=True)
            return
        try:
            plan = await asyncio.to_thread(lora_job.job_manager.plan_preview, session_id, interaction.user.id)
        except lora_job.LoraJobError as exc:
            await interaction.followup.send(f"学習planを確認できませんでした: {exc}", ephemeral=True)
            return
        await interaction.followup.send(
            "学習開始前の確認です。確認ボタンを押すまで学習は開始しません。\n"
            + _format_plan_preview(_loaded_plan_payload(plan)),
            view=LoraTrainConfirmView(owner_id=interaction.user.id, session_id=session_id),
            ephemeral=True,
        )

    @slash_lora_train.autocomplete("session_id")
    async def autocomplete_lora_train(
        interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        if not is_pc_control_allowed(interaction.user.id):
            return []
        choices: list[app_commands.Choice[str]] = []
        for session in lora_job.job_manager.plans_for_owner(interaction.user.id):
            session_id = str(session.get("session_id", ""))
            label = f"{session.get('dataset_name', '不明')} / {session.get('plan', {}).get('output_name', 'plan')}"
            if current.casefold() in label.casefold() or current.casefold() in session_id.casefold():
                choices.append(app_commands.Choice(name=label[:100], value=session_id))
        return choices[:25]

    @client.tree.command(
        name="lora_status",
        description="Botが管理するLoRA学習の状態を確認します。",
    )
    @app_commands.describe(job_id="省略時は自分の実行中、または最新ジョブ")
    async def slash_lora_status(interaction: discord.Interaction, job_id: str | None = None) -> None:
        if not is_pc_control_allowed(interaction.user.id):
            await interaction.response.send_message("この操作は、許可されたユーザーだけが実行できます。", ephemeral=True)
            return
        try:
            job = await asyncio.to_thread(_find_owner_job, interaction.user.id, job_id)
        except lora_job.LoraJobError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        if job.get("owner_id") != interaction.user.id:
            await interaction.response.send_message("この学習ジョブを確認する権限がありません。", ephemeral=True)
            return
        await interaction.response.send_message(_format_job_status(job), ephemeral=True)

    @slash_lora_status.autocomplete("job_id")
    async def autocomplete_lora_status(
        interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        if not is_pc_control_allowed(interaction.user.id):
            return []
        choices = []
        for job in lora_job.job_manager.list_for_owner(interaction.user.id):
            job_id = str(job.get("job_id", ""))
            label = f"{job.get('dataset_name', '不明')} / {job.get('status', '不明')}"
            if not current or current.casefold() in job_id.casefold() or current.casefold() in label.casefold():
                choices.append(app_commands.Choice(name=f"{label} / {job_id[:8]}"[:100], value=job_id))
        return choices[:25]

    @client.tree.command(
        name="lora_cancel",
        description="Botが開始したLoRA学習を確認後に安全停止します。",
    )
    @app_commands.describe(job_id="省略時は自分の実行中ジョブ")
    async def slash_lora_cancel(interaction: discord.Interaction, job_id: str | None = None) -> None:
        if not is_pc_control_allowed(interaction.user.id):
            await interaction.response.send_message("この操作は、許可されたユーザーだけが実行できます。", ephemeral=True)
            return
        try:
            job = await asyncio.to_thread(_find_owner_job, interaction.user.id, job_id)
        except lora_job.LoraJobError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        if job.get("owner_id") != interaction.user.id:
            await interaction.response.send_message("この学習ジョブを停止する権限がありません。", ephemeral=True)
            return
        if job.get("status") not in {"starting", "running"}:
            await interaction.response.send_message(f"このジョブは停止対象ではありません。\n{_format_job_status(job)}", ephemeral=True)
            return
        await interaction.response.send_message(
            "次の学習だけを停止します。Bot外で起動したプロセスは停止しません。\n\n" + _format_job_status(job),
            view=LoraCancelConfirmView(owner_id=interaction.user.id, job_id=str(job["job_id"])),
            ephemeral=True,
        )

    @client.tree.command(
        name="lora_caption_preview",
        description="LoRA学習前の最終captionをプレビューします。",
    )
    @app_commands.choices(lora_type=LORA_TYPE_CHOICES)
    @app_commands.describe(
        dataset_dir="画像とcaption（同名.txt）があるデータセット名またはフォルダのパス。",
        lora_type="LoRA種別。キャラクターだけトリガータグ必須です。",
        trigger_tag="キャラクターLoRAのトリガータグ。絵柄・追加では任意です。",
    )
    async def slash_lora_caption_preview(
        interaction: discord.Interaction,
        dataset_dir: str,
        lora_type: app_commands.Choice[str],
        trigger_tag: str | None = None,
    ) -> None:
        diagnostic_id = _new_diagnostic_id("lora-caption-preview")
        started_at = time.perf_counter()
        logger.info(
            "LoRA caption preview start diagnostic_id=%s user_id=%s dataset_input=%s",
            diagnostic_id,
            interaction.user.id,
            dataset_dir,
        )
        logger.info(
            "LoRA caption preview step=interaction_defer start diagnostic_id=%s",
            diagnostic_id,
        )
        try:
            await interaction.response.defer(ephemeral=True)
        except Exception:
            logger.exception(
                "LoRA caption preview step=interaction_defer failed diagnostic_id=%s elapsed_ms=%s",
                diagnostic_id,
                _elapsed_ms(started_at),
            )
            raise
        logger.info(
            "LoRA caption preview step=interaction_defer complete diagnostic_id=%s",
            diagnostic_id,
        )
        if not is_pc_control_allowed(interaction.user.id):
            logger.info(
                "LoRA caption preview rejected diagnostic_id=%s reason=permission",
                diagnostic_id,
            )
            await interaction.followup.send("この操作は、許可されたユーザーだけが実行できます。", ephemeral=True)
            return
        try:
            step_started_at = time.perf_counter()
            logger.info(
                "LoRA caption preview step=resolve_dataset_dir start diagnostic_id=%s executor=thread",
                diagnostic_id,
            )
            resolved_dataset_dir = await asyncio.to_thread(
                _resolve_dataset_dir_for_prepare, dataset_dir
            )
            logger.info(
                "LoRA caption preview step=resolve_dataset_dir complete diagnostic_id=%s elapsed_ms=%s resolved_dataset_dir=%s",
                diagnostic_id,
                _elapsed_ms(step_started_at),
                resolved_dataset_dir,
            )

            step_started_at = time.perf_counter()
            logger.info(
                "LoRA caption preview step=preview_captions start diagnostic_id=%s executor=thread",
                diagnostic_id,
            )
            preview = await asyncio.to_thread(
                preview_lora_captions,
                resolved_dataset_dir,
                lora_type=lora_type.value,
                trigger_tag=trigger_tag or "",
            )
            logger.info(
                "LoRA caption preview step=preview_captions complete diagnostic_id=%s elapsed_ms=%s item_count=%s",
                diagnostic_id,
                _elapsed_ms(step_started_at),
                len(preview.items),
            )
        except LoraTrainingError as exc:
            logger.error(
                "LoRA caption preview failed diagnostic_id=%s elapsed_ms=%s error=%s",
                diagnostic_id,
                _elapsed_ms(started_at),
                exc,
                exc_info=True,
            )
            await interaction.followup.send(f"captionプレビューに失敗しました: {exc}", ephemeral=True)
            return
        except asyncio.CancelledError:
            logger.error(
                "LoRA caption preview cancelled diagnostic_id=%s elapsed_ms=%s",
                diagnostic_id,
                _elapsed_ms(started_at),
                exc_info=True,
            )
            raise
        except Exception:
            logger.exception(
                "LoRA caption preview unexpected failure diagnostic_id=%s elapsed_ms=%s",
                diagnostic_id,
                _elapsed_ms(started_at),
            )
            raise
        try:
            await interaction.followup.send(_preview_message(preview), ephemeral=True)
        except Exception:
            logger.exception(
                "LoRA caption preview response failed diagnostic_id=%s elapsed_ms=%s",
                diagnostic_id,
                _elapsed_ms(started_at),
            )
            raise
        logger.info(
            "LoRA caption preview complete diagnostic_id=%s elapsed_ms=%s",
            diagnostic_id,
            _elapsed_ms(started_at),
        )

    @client.tree.command(
        name="lora_prepare",
        description="完成済みLoRAのcaptionとメタデータをプレビュー確認後に登録します。",
    )
    @app_commands.choices(lora_type=LORA_TYPE_CHOICES)
    @app_commands.describe(
        lora_name="学習後のLoRA名。",
        dataset_dir="画像とcaption（同名.txt）があるデータセット名またはフォルダのパス。",
        lora_file="既存LoRA本体のsafetensorsパス。ファイル名だけならLoRA管理出力から解決します。",
        lora_type="LoRA種別。省略時は学習時の保存値を使います。キャラクターだけトリガータグ必須です。",
        trigger_tag="キャラクターLoRAのトリガータグ。絵柄・追加では任意です。",
        recommended_strength="生成時の推奨強度（0.0～1.0）。省略時は学習時の保存値を使います。",
    )
    async def slash_lora_prepare(
        interaction: discord.Interaction,
        lora_name: str,
        dataset_dir: str,
        lora_file: str,
        lora_type: app_commands.Choice[str] | None = None,
        trigger_tag: str | None = None,
        recommended_strength: float | None = None,
    ) -> None:
        diagnostic_id = _new_diagnostic_id("lora-prepare")
        started_at = time.perf_counter()
        logger.info(
            "LoRA metadata prepare start diagnostic_id=%s user_id=%s lora_name=%s lora_file_input=%s dataset_input=%s",
            diagnostic_id,
            interaction.user.id,
            lora_name,
            lora_file,
            dataset_dir,
        )
        logger.info(
            "LoRA metadata prepare step=interaction_defer start diagnostic_id=%s",
            diagnostic_id,
        )
        try:
            await interaction.response.defer(ephemeral=True)
        except Exception:
            logger.exception(
                "LoRA metadata prepare step=interaction_defer failed diagnostic_id=%s elapsed_ms=%s",
                diagnostic_id,
                _elapsed_ms(started_at),
            )
            raise
        logger.info(
            "LoRA metadata prepare step=interaction_defer complete diagnostic_id=%s",
            diagnostic_id,
        )
        if not is_pc_control_allowed(interaction.user.id):
            logger.info(
                "LoRA metadata prepare rejected diagnostic_id=%s reason=permission",
                diagnostic_id,
            )
            await interaction.followup.send("この操作は、許可されたユーザーだけが実行できます。", ephemeral=True)
            return
        try:
            step_started_at = time.perf_counter()
            logger.info(
                "LoRA metadata prepare step=resolve_lora_file start diagnostic_id=%s executor=thread",
                diagnostic_id,
            )
            lora_path = await asyncio.to_thread(
                resolve_lora_file,
                lora_file,
                search_root=get_lora_manager_outputs_dir(),
            )
            logger.info(
                "LoRA metadata prepare step=resolve_lora_file complete diagnostic_id=%s elapsed_ms=%s resolved_lora_path=%s",
                diagnostic_id,
                _elapsed_ms(step_started_at),
                lora_path,
            )

            step_started_at = time.perf_counter()
            logger.info(
                "LoRA metadata prepare step=resolve_metadata start diagnostic_id=%s executor=thread",
                diagnostic_id,
            )
            metadata, defaults_source = await asyncio.to_thread(
                resolve_lora_prepare_metadata,
                lora_path=lora_path,
                lora_name=lora_name,
                lora_type=None if lora_type is None else lora_type.value,
                trigger_tag=trigger_tag,
                recommended_strength=recommended_strength,
            )
            logger.info(
                "LoRA metadata prepare step=resolve_metadata complete diagnostic_id=%s elapsed_ms=%s defaults_source=%s lora_type=%s",
                diagnostic_id,
                _elapsed_ms(step_started_at),
                defaults_source,
                metadata.lora_type,
            )
            selected_type = metadata.lora_type
            selected_trigger = metadata.trigger_tag

            step_started_at = time.perf_counter()
            logger.info(
                "LoRA metadata prepare step=resolve_dataset_dir start diagnostic_id=%s executor=thread",
                diagnostic_id,
            )
            resolved_dataset_dir = await asyncio.to_thread(
                _resolve_dataset_dir_for_prepare, dataset_dir
            )
            logger.info(
                "LoRA metadata prepare step=resolve_dataset_dir complete diagnostic_id=%s elapsed_ms=%s resolved_dataset_dir=%s",
                diagnostic_id,
                _elapsed_ms(step_started_at),
                resolved_dataset_dir,
            )

            step_started_at = time.perf_counter()
            logger.info(
                "LoRA metadata prepare step=preview_captions start diagnostic_id=%s executor=thread",
                diagnostic_id,
            )
            preview = await asyncio.to_thread(
                preview_lora_captions,
                resolved_dataset_dir,
                lora_type=selected_type,
                trigger_tag=selected_trigger or "",
            )
            logger.info(
                "LoRA metadata prepare step=preview_captions complete diagnostic_id=%s elapsed_ms=%s item_count=%s",
                diagnostic_id,
                _elapsed_ms(step_started_at),
                len(preview.items),
            )
        except LoraTrainingError as exc:
            logger.error(
                "LoRA metadata prepare failed diagnostic_id=%s elapsed_ms=%s error=%s",
                diagnostic_id,
                _elapsed_ms(started_at),
                exc,
                exc_info=True,
            )
            await interaction.followup.send(f"LoRA登録準備を作成できませんでした: {exc}", ephemeral=True)
            return
        except asyncio.CancelledError:
            logger.error(
                "LoRA metadata prepare cancelled diagnostic_id=%s elapsed_ms=%s",
                diagnostic_id,
                _elapsed_ms(started_at),
                exc_info=True,
            )
            raise
        except Exception:
            logger.exception(
                "LoRA metadata prepare unexpected failure diagnostic_id=%s elapsed_ms=%s",
                diagnostic_id,
                _elapsed_ms(started_at),
            )
            raise

        message = (
            _preview_message(preview)
            + f"\n\nLoRA名: {metadata.name}\n"
            + f"checkpoint: {metadata.checkpoint or '未指定'} / 系統: {metadata.model_family or '不明'}\n"
            + f"推奨強度: {metadata.recommended_strength:g}\n"
            + f"初期値: {defaults_source}（入力された項目は上書き）\n"
            + "内容を確認して適用してください。"
        )[:1900]
        try:
            await interaction.followup.send(
                message,
                view=LoraPreparationView(
                    owner_id=interaction.user.id,
                    preview=preview,
                    lora_path=lora_path,
                    metadata=metadata,
                    diagnostic_id=diagnostic_id,
                ),
                ephemeral=True,
            )
        except Exception:
            logger.exception(
                "LoRA metadata prepare response failed diagnostic_id=%s elapsed_ms=%s",
                diagnostic_id,
                _elapsed_ms(started_at),
            )
            raise
        logger.info(
            "LoRA metadata prepare complete diagnostic_id=%s elapsed_ms=%s awaiting_confirmation=true",
            diagnostic_id,
            _elapsed_ms(started_at),
        )
    PAGE_SIZE = 15
