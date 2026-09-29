"""Discord slash commands for the local image gallery."""

from __future__ import annotations

import asyncio
import logging

import discord
from discord import app_commands

from services.gallery_control import (
    GalleryControlError,
    format_gallery_links,
    get_gallery_ai_status,
    get_gallery_status,
    start_gallery,
    start_gallery_review,
    stop_gallery,
)
from services.pc_control import is_pc_control_allowed


logger = logging.getLogger("secretary-bot.gallery_commands")

DENIED_MESSAGE = "この操作は、PC_CONTROL_ALLOWED_USER_IDS で許可されたユーザーだけが実行できます。"


def _link_view(status) -> discord.ui.View:
    """Prefer the Tailscale URL while keeping a local fallback."""

    url = status.urls.tailscale[0] if status.urls.tailscale else status.urls.local
    view = discord.ui.View(timeout=None)
    view.add_item(
        discord.ui.Button(
            label="画像ギャラリーを開く",
            style=discord.ButtonStyle.link,
            url=url,
        )
    )
    return view


def _format_review_job(job: dict[str, object]) -> str:
    status = str(job.get("status") or "unknown")
    stage = str(job.get("stage") or "")
    processed = job.get("processed", 0)
    total = job.get("total", 0)
    if status in {"running", "starting"}:
        return f"AI自動選別を開始しました。\n状態: {stage} ({processed}/{total})\n既存の手動レビュー状態は変更されません。"
    if status == "completed":
        return f"AI自動選別が完了しました。\n評価件数: {total}\nAI候補とスコアをギャラリーで確認できます。"
    if status == "failed":
        return f"AI自動選別に失敗しました。\n{job.get('error') or '詳細はギャラリーの状態表示を確認してください。'}"
    return f"AI自動選別の状態: {status}"


async def _watch_review(interaction: discord.Interaction, job_id: str) -> None:
    """Best-effort completion notification while the interaction webhook is valid."""

    try:
        for _ in range(240):
            await asyncio.sleep(5)
            job = await asyncio.to_thread(get_gallery_ai_status)
            if str(job.get("job_id") or "") != job_id:
                continue
            if str(job.get("status")) not in {"completed", "failed"}:
                continue
            status = await asyncio.to_thread(get_gallery_status)
            await interaction.followup.send(
                _format_review_job(job),
                view=_link_view(status) if status.running else None,
                ephemeral=True,
            )
            return
    except Exception:
        logger.exception("AI review completion notification failed.")


def setup(client: discord.Client) -> None:
    """Register the user-facing Gallery command group."""

    gallery_group = app_commands.Group(
        name="gallery",
        description="ローカル画像ギャラリーを操作します。",
    )

    async def reject_if_unavailable(interaction: discord.Interaction) -> bool:
        if is_pc_control_allowed(interaction.user.id):
            return False
        logger.warning("Denied gallery command by user_id=%s", interaction.user.id)
        await interaction.response.send_message(DENIED_MESSAGE, ephemeral=True)
        return True

    @gallery_group.command(name="open", description="ギャラリーを起動してアクセスリンクを表示します。")
    async def gallery_open(interaction: discord.Interaction) -> None:
        if await reject_if_unavailable(interaction):
            return

        await interaction.response.defer(ephemeral=True)
        try:
            result = await asyncio.to_thread(start_gallery)
        except GalleryControlError as exc:
            logger.warning("Gallery start failed: %s", exc)
            await interaction.followup.send(f"ギャラリーを起動できませんでした。\n{exc}", ephemeral=True)
            return
        except Exception:
            logger.exception("Unexpected /gallery open failure.")
            await interaction.followup.send(
                "ギャラリー起動中に予期しないエラーが発生しました。Botのログを確認してください。",
                ephemeral=True,
            )
            return

        await interaction.followup.send(
            format_gallery_links(result.status, started=not result.already_running),
            view=_link_view(result.status),
            ephemeral=True,
        )

    @gallery_group.command(name="status", description="ギャラリーの起動状態とアクセスリンクを表示します。")
    async def gallery_status(interaction: discord.Interaction) -> None:
        if await reject_if_unavailable(interaction):
            return

        await interaction.response.defer(ephemeral=True)
        try:
            status = await asyncio.to_thread(get_gallery_status)
        except Exception:
            logger.exception("Unexpected /gallery status failure.")
            await interaction.followup.send(
                "ギャラリーの状態確認に失敗しました。Botのログを確認してください。",
                ephemeral=True,
            )
            return

        await interaction.followup.send(format_gallery_links(status), view=_link_view(status), ephemeral=True)

    @gallery_group.command(name="stop", description="ギャラリーを安全に停止します。")
    async def gallery_stop(interaction: discord.Interaction) -> None:
        if await reject_if_unavailable(interaction):
            return

        await interaction.response.defer(ephemeral=True)
        try:
            result = await asyncio.to_thread(stop_gallery)
        except GalleryControlError as exc:
            logger.warning("Gallery stop failed: %s", exc)
            await interaction.followup.send(f"ギャラリーを停止できませんでした。\n{exc}", ephemeral=True)
            return
        except Exception:
            logger.exception("Unexpected /gallery stop failure.")
            await interaction.followup.send(
                "ギャラリー停止中に予期しないエラーが発生しました。Botのログを確認してください。",
                ephemeral=True,
            )
            return

        message = "ギャラリーはすでに停止中です。" if result.already_stopped else "ギャラリーを停止しました。"
        await interaction.followup.send(message, ephemeral=True)

    @gallery_group.command(name="review", description="AIで画像を評価し、おすすめ候補を更新します。")
    @app_commands.describe(reference_name="登録済みのキャラクター参照名（任意）")
    async def gallery_review(interaction: discord.Interaction, reference_name: str = "") -> None:
        if await reject_if_unavailable(interaction):
            return

        await interaction.response.defer(ephemeral=True)
        try:
            result = await asyncio.to_thread(start_gallery_review, reference_name=reference_name.strip())
        except GalleryControlError as exc:
            logger.warning("Gallery AI review failed: %s", exc)
            await interaction.followup.send(f"AI自動選別を開始できませんでした。\n{exc}", ephemeral=True)
            return
        except Exception:
            logger.exception("Unexpected /gallery review failure.")
            await interaction.followup.send(
                "AI自動選別の開始中に予期しないエラーが発生しました。Botのログを確認してください。",
                ephemeral=True,
            )
            return

        await interaction.followup.send(
            _format_review_job(result.job),
            view=_link_view(result.status),
            ephemeral=True,
        )
        job_id = str(result.job.get("job_id") or "")
        if job_id and str(result.job.get("status")) in {"running", "starting"}:
            asyncio.create_task(_watch_review(interaction, job_id))

    client.tree.add_command(gallery_group)
    logger.info("Registered gallery commands: gallery open/status/stop/review")
