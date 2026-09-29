"""Discord commands for persistent user-owned reference images."""

from __future__ import annotations

import asyncio
import io
import json
import logging
from pathlib import Path

import discord
from discord import app_commands

from services.image_analyzer import MAX_IMAGE_BYTES, detect_image_content_type, validate_image_bytes
from services.pc_control import is_pc_control_allowed
from services.saved_references import (
    REFERENCE_KINDS, delete_reference, list_references, random_reference,
    read_reference_bytes, save_reference,
)
from services.sd_control import get_latest_image_paths


logger = logging.getLogger("secretary-bot.references")
REF_KIND_CHOICES = [app_commands.Choice(name=kind, value=kind) for kind in REFERENCE_KINDS]
DENIED_MESSAGE = "保存済み参考画像は、許可されたユーザーだけが利用できます。"


def _load_metadata(path: Path) -> dict[str, object]:
    candidates = (path.with_name(path.name + ".metadata.json"), path.with_suffix(".json"))
    for candidate in candidates:
        try:
            if candidate.is_file():
                value = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    return value
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read image metadata sidecar: %s", candidate, exc_info=True)
    return {}


def _format_reference(item) -> str:
    tags = ", ".join(item.tags) if item.tags else "なし"
    seed = "-" if item.seed is None else str(item.seed)
    return f"`{item.id}` [{item.kind}] {item.name} / tags: {tags} / seed: {seed}"


def setup(client: discord.Client) -> None:
    @client.tree.command(name="ref_save", description="直前の生成画像または添付画像を参考画像として保存します。")
    @app_commands.choices(kind=REF_KIND_CHOICES)
    @app_commands.describe(name="保存名", kind="pose / reference / style / derive", tags="カンマ区切りのタグ。例: standing, beach", note="任意のメモ", image="省略時は直前の生成画像を保存します。")
    async def slash_ref_save(interaction: discord.Interaction, name: str, kind: app_commands.Choice[str] | None = None, tags: str | None = None, note: str | None = None, image: discord.Attachment | None = None) -> None:
        await interaction.response.defer(ephemeral=True)
        if not is_pc_control_allowed(interaction.user.id):
            await interaction.followup.send(DENIED_MESSAGE, ephemeral=True)
            return
        metadata: dict[str, object] = {}
        try:
            if image is not None:
                if image.size > MAX_IMAGE_BYTES:
                    raise ValueError("添付画像は8MB以内で指定してください。")
                image_bytes = await image.read()
                content_type = detect_image_content_type(image_bytes, image.content_type)
                validate_image_bytes(image_bytes, content_type)
                saved = await asyncio.to_thread(save_reference, user_id=interaction.user.id, name=name, kind=(kind.value if kind else "reference"), tags=tags, note=note or "", image_bytes=image_bytes, extension={"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(content_type or "", ".png"), metadata=metadata)
            else:
                latest = get_latest_image_paths()
                if not latest:
                    raise ValueError("直前の生成画像がありません。画像を添付して試してください。")
                source = latest[0]
                saved = await asyncio.to_thread(save_reference, user_id=interaction.user.id, name=name, kind=(kind.value if kind else "reference"), tags=tags, note=note or "", source_path=source, metadata=_load_metadata(source))
        except Exception as exc:
            logger.exception("Failed to save reference: user_id=%s", interaction.user.id)
            await interaction.followup.send(f"参考画像を保存できませんでした: {str(exc)[:400]}", ephemeral=True)
            return
        await interaction.followup.send("参考画像を保存しました。\n" + _format_reference(saved), ephemeral=True)

    @client.tree.command(name="ref_list", description="保存済み参考画像の一覧を表示します。")
    @app_commands.choices(kind=REF_KIND_CHOICES)
    @app_commands.describe(kind="種類で絞り込み", tag="完全一致するタグで絞り込み")
    async def slash_ref_list(interaction: discord.Interaction, kind: app_commands.Choice[str] | None = None, tag: str | None = None) -> None:
        await interaction.response.defer(ephemeral=True)
        if not is_pc_control_allowed(interaction.user.id):
            await interaction.followup.send(DENIED_MESSAGE, ephemeral=True)
            return
        try:
            items = await asyncio.to_thread(list_references, interaction.user.id, kind=None if kind is None else kind.value, tag=tag, limit=25)
        except Exception:
            logger.exception("Failed to list saved references: user_id=%s", interaction.user.id)
            await interaction.followup.send("保存済み参考画像の一覧を取得できませんでした。", ephemeral=True)
            return
        if not items:
            await interaction.followup.send("保存済み参考画像はありません。", ephemeral=True)
            return
        text = "保存済み参考画像（自分専用）\n\n" + "\n".join(_format_reference(item) for item in items)
        await interaction.followup.send(text[:1900], ephemeral=True)

    @client.tree.command(name="ref_delete", description="自分の保存済み参考画像をIDで削除します。")
    @app_commands.describe(reference_id="/ref_listで表示された保存ID")
    async def slash_ref_delete(interaction: discord.Interaction, reference_id: int) -> None:
        await interaction.response.defer(ephemeral=True)
        if not is_pc_control_allowed(interaction.user.id):
            await interaction.followup.send(DENIED_MESSAGE, ephemeral=True)
            return
        try:
            deleted = await asyncio.to_thread(delete_reference, interaction.user.id, reference_id)
        except Exception:
            logger.exception("Failed to delete saved reference: user_id=%s id=%s", interaction.user.id, reference_id)
            await interaction.followup.send("保存済み参考画像を削除できませんでした。", ephemeral=True)
            return
        await interaction.followup.send("保存済み参考画像を削除しました。" if deleted else "指定した保存IDが見つかりません。", ephemeral=True)

    @client.tree.command(name="ref_random", description="条件に合う保存済み参考画像をランダムに1枚返します。")
    @app_commands.choices(kind=REF_KIND_CHOICES)
    @app_commands.describe(kind="種類で絞り込み", tag="完全一致するタグで絞り込み")
    async def slash_ref_random(interaction: discord.Interaction, kind: app_commands.Choice[str] | None = None, tag: str | None = None) -> None:
        await interaction.response.defer(ephemeral=True)
        if not is_pc_control_allowed(interaction.user.id):
            await interaction.followup.send(DENIED_MESSAGE, ephemeral=True)
            return
        try:
            item = await asyncio.to_thread(random_reference, interaction.user.id, kind=None if kind is None else kind.value, tag=tag)
            if item is None:
                await interaction.followup.send("条件に合う保存済み参考画像がありません。", ephemeral=True)
                return
            image_bytes = await asyncio.to_thread(read_reference_bytes, item)
        except Exception:
            logger.exception("Failed to select random saved reference: user_id=%s", interaction.user.id)
            await interaction.followup.send("保存済み参考画像を読み込めませんでした。", ephemeral=True)
            return
        file = discord.File(io.BytesIO(image_bytes), filename=f"reference_{item.id}{item.image_path.suffix or '.png'}")
        try:
            await interaction.followup.send(_format_reference(item), file=file, ephemeral=True, allowed_mentions=discord.AllowedMentions.none())
        finally:
            file.close()
