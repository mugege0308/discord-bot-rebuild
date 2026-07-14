import asyncio
import io
import logging

import discord
from discord import app_commands

from services.image_analyzer import (
    ImageAnalysisResult,
    analyze_image_bytes,
    build_image_memory_text,
    detect_image_content_type,
    validate_image_bytes,
)
from services.image_memory import (
    build_image_memory_detail_message,
    build_image_memory_list_message,
    delete_image_memory,
    format_image_memory_rating,
    get_image_memory_name,
    get_image_memory_rating,
    get_image_memory_summary,
    get_image_memory_use_for_generation,
    load_image_memory,
    list_image_memories,
    rename_image_memory,
    resolve_image_memory_image_path,
    rate_image_memory,
    sanitize_image_memory_name,
    save_image_memory,
    set_image_memory_generation_use,
    suggest_image_memory_name,
)
from services.memory import add_memory, delete_memory, rename_linked_image_memory
from services.prompt_generator import PromptGeneratorError, PromptGeneratorTimeout


logger = logging.getLogger("secretary-bot.image_analyze")

ONLY_OWNER_MESSAGE = "この保存ボタンは、/image_analyze を実行した本人だけが押せるよ。"
EXTENSION_CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _attachment_content_type(image: discord.Attachment) -> str | None:
    if image.content_type:
        return image.content_type
    lower_name = image.filename.lower()
    for extension, content_type in EXTENSION_CONTENT_TYPES.items():
        if lower_name.endswith(extension):
            return content_type
    return None


def _parse_memory_id(response: str) -> int | None:
    marker = "ID:"
    if marker not in response:
        return None
    value = response.rsplit(marker, 1)[-1].strip()
    try:
        return int(value)
    except ValueError:
        return None


def _truncate_for_discord(text: str, limit: int = 1800) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 20].rstrip() + "\n...(省略)"


def _build_analysis_message(analysis: ImageAnalysisResult) -> str:
    tags = analysis.tags.get("sd_tags") or [tag for values in analysis.tags.values() for tag in values]
    tag_text = ", ".join(dict.fromkeys(tags)) or "抽出できませんでした。"
    return _truncate_for_discord(f"【日本語の要約】\n{analysis.analysis_text}\n\n【SDタグ候補】\n{tag_text}")


def _analysis_file(analysis: str) -> discord.File:
    data = analysis.encode("utf-8")
    buffer = io.BytesIO(data)
    return discord.File(buffer, filename="image_analysis.txt")


async def _send_long_text(
    interaction: discord.Interaction,
    message: str,
    *,
    view: discord.ui.View | None = None,
    ephemeral: bool,
) -> None:
    if len(message) <= 1900:
        await interaction.followup.send(message, view=view, ephemeral=ephemeral)
        return

    summary = message[:1500].rstrip() + "\n\n全文は添付txtを確認してください。"
    file = _analysis_file(message)
    try:
        await interaction.followup.send(summary, file=file, view=view, ephemeral=ephemeral)
    finally:
        file.close()


async def _send_image_memory_list(interaction: discord.Interaction, page: int, page_size: int = 3) -> None:
    items = list_image_memories(interaction.user.id)
    if not items:
        await interaction.response.send_message("画像参考メモはまだありません。", ephemeral=True)
        return
    total_pages = max((len(items) + page_size - 1) // page_size, 1)
    page = min(max(page, 1), total_pages)
    selected = items[(page - 1) * page_size : page * page_size]
    embeds: list[discord.Embed] = []
    files: list[discord.File] = []
    try:
        for item in selected:
            memory_id = int(item.get("id", 0))
            name = get_image_memory_name(item, memory_id)
            rating = format_image_memory_rating(get_image_memory_rating(item))
            generation_use = "ON" if get_image_memory_use_for_generation(item) else "OFF"
            embed = discord.Embed(
                title=f"ID {memory_id}「{name}」",
                description=get_image_memory_summary(item),
                color=discord.Color.blurple(),
            )
            embed.add_field(name="評価", value=rating, inline=True)
            embed.add_field(name="生成利用", value=generation_use, inline=True)
            image_path = resolve_image_memory_image_path(item, memory_id)
            if image_path is not None:
                filename = image_path.name
                files.append(discord.File(str(image_path), filename=filename))
                embed.set_thumbnail(url=f"attachment://{filename}")
            else:
                embed.set_footer(text="元画像なし")
            embeds.append(embed)
        await interaction.response.send_message(
            content=f"画像参考メモ一覧 {page}/{total_pages}（別ページは /image_memory_list page:番号）",
            embeds=embeds,
            files=files,
            ephemeral=True,
        )
    finally:
        for file in files:
            file.close()


async def _send_image_memory_detail(interaction: discord.Interaction, memory_id: int) -> None:
    item = load_image_memory(memory_id)
    message = build_image_memory_detail_message(interaction.user.id, memory_id)
    image_path = None
    if item:
        try:
            is_owner = int(item.get("user_id", 0)) == int(interaction.user.id)
        except (TypeError, ValueError):
            is_owner = False
        if is_owner:
            image_path = resolve_image_memory_image_path(item, memory_id)

    files: list[discord.File] = []
    try:
        if image_path is not None:
            files.append(discord.File(str(image_path), filename=image_path.name))
        if len(message) > 1900:
            summary = message[:1500].rstrip() + "\n\n全文は添付txtを確認してください。"
            files.append(_analysis_file(message))
        else:
            summary = message
        await interaction.followup.send(content=summary, files=files, ephemeral=True)
    finally:
        for file in files:
            file.close()


class ImageMemorySaveModal(discord.ui.Modal, title="画像参考Memoryを保存"):
    def __init__(self, save_view: "SaveImageAnalysisMemoryView", default_name: str) -> None:
        super().__init__(timeout=300)
        self.save_view = save_view
        self.name_input = discord.ui.TextInput(
            label="名前",
            default=default_name,
            min_length=1,
            max_length=50,
            required=True,
        )
        self.add_item(self.name_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.save_view.owner_id:
            await interaction.response.send_message(ONLY_OWNER_MESSAGE, ephemeral=True)
            return
        try:
            name = sanitize_image_memory_name(str(self.name_input.value))
        except ValueError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        await self.save_view.save_memory(interaction, name)


class SaveImageAnalysisMemoryView(discord.ui.View):
    def __init__(self, owner_id: int, analysis: ImageAnalysisResult, note: str, name: str, image_bytes: bytes, image_content_type: str) -> None:
        super().__init__(timeout=900)
        self.owner_id = owner_id
        self.analysis = analysis
        self.note = note
        self.requested_name = name
        self.suggested_name = suggest_image_memory_name(analysis.tags)
        self.image_bytes = image_bytes
        self.image_content_type = image_content_type
        self.is_saving = False
        self.is_saved = False
        self.save_lock = asyncio.Lock()
        self.source_message = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(ONLY_OWNER_MESSAGE, ephemeral=True)
            return False
        return True

    @discord.ui.button(label="Memoryに保存", style=discord.ButtonStyle.secondary)
    async def save_memory_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        self.source_message = interaction.message
        if self.requested_name:
            await interaction.response.defer(ephemeral=True)
            await self.save_memory(interaction, self.requested_name)
            return
        await interaction.response.send_modal(ImageMemorySaveModal(self, self.suggested_name))

    async def save_memory(self, interaction: discord.Interaction, name: str) -> None:
        async with self.save_lock:
            if self.is_saved:
                state = "saved"
            elif self.is_saving:
                state = "saving"
            else:
                self.is_saving = True
                state = "start"
        if state == "saved":
            await interaction.followup.send("この解析結果はすでに保存済みです。", ephemeral=True)
            return
        if state == "saving":
            await interaction.followup.send("保存処理中です。少し待ってね。", ephemeral=True)
            return

        linked_memory_id: int | None = None
        try:
            safe_name = sanitize_image_memory_name(name)
            memory_text = build_image_memory_text(self.analysis, safe_name)
            response = await asyncio.to_thread(add_memory, interaction.user.id, memory_text)
            linked_memory_id = _parse_memory_id(response)
            if linked_memory_id is None:
                raise RuntimeError(response)
            image_memory = await asyncio.to_thread(
                save_image_memory,
                interaction.user.id,
                self.analysis.analysis_text,
                self.note,
                "",
                linked_memory_id,
                self.analysis.tags,
                safe_name,
                self.image_bytes,
                self.image_content_type,
            )
        except Exception:
            logger.exception("Failed to save image analysis memory.")
            if linked_memory_id is not None:
                await asyncio.to_thread(delete_memory, interaction.user.id, linked_memory_id)
            async with self.save_lock:
                self.is_saving = False
            await interaction.followup.send("保存に失敗したみたい。ログを確認してね。", ephemeral=True)
            return

        async with self.save_lock:
            self.is_saving = False
            self.is_saved = True
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True
                child.label = "保存済み"
        if self.source_message is not None:
            try:
                await self.source_message.edit(view=self)
            except Exception:
                logger.exception("Failed to disable image memory save button after save.")
        self.stop()
        await interaction.followup.send(
            f"ID {image_memory['id']}「{image_memory['name']}」として保存しました。",
            ephemeral=True,
        )


def setup(client: discord.Client) -> None:
    @client.tree.command(name="image_analyze", description="参考画像を画像生成向けに分解します。")
    @app_commands.describe(
        image="解析したい参考画像",
        name="画像参考Memoryの名前。未指定なら解析結果から自動生成します。",
        note="任意の補足。例: 衣装を中心に見て",
        private="trueなら自分だけに表示します。既定はチャンネルに残します。",
    )
    async def slash_image_analyze(
        interaction: discord.Interaction,
        image: discord.Attachment,
        name: str = "",
        note: str = "",
        private: bool = False,
    ) -> None:
        try:
            safe_name = sanitize_image_memory_name(name, allow_empty=True)
        except ValueError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        await interaction.response.defer(ephemeral=private)

        try:
            image_bytes = await image.read()
            content_type = detect_image_content_type(image_bytes, _attachment_content_type(image))
            validate_image_bytes(image_bytes, content_type)
            analysis = await asyncio.to_thread(
                analyze_image_bytes,
                image_bytes,
                content_type or "image/png",
                note,
            )
        except PromptGeneratorTimeout:
            logger.exception("/image_analyze timed out.")
            await interaction.followup.send("LM Studioから応答がありませんでした。", ephemeral=True)
            return
        except PromptGeneratorError as exc:
            logger.exception("/image_analyze failed.")
            await interaction.followup.send(str(exc), ephemeral=True)
            return
        except Exception:
            logger.exception("Unexpected /image_analyze error.")
            await interaction.followup.send("画像解析に失敗しました。LM Studioの起動状態を確認してね。", ephemeral=True)
            return

        await _send_long_text(
            interaction,
            _build_analysis_message(analysis),
            view=SaveImageAnalysisMemoryView(
                interaction.user.id,
                analysis,
                note,
                safe_name,
                image_bytes,
                content_type or "image/png",
            ),
            ephemeral=private,
        )

    @client.tree.command(name="image_memory_list", description="保存済み画像参考メモを一覧表示します。")
    @app_commands.describe(page="ページ番号。未指定なら1ページ目です。")
    async def slash_image_memory_list(interaction: discord.Interaction, page: int = 1) -> None:
        await _send_image_memory_list(interaction, page)

    @client.tree.command(name="image_memory_show", description="画像参考メモの詳細を表示します。")
    @app_commands.describe(id="画像参考メモID")
    async def slash_image_memory_show(interaction: discord.Interaction, id: int) -> None:
        await interaction.response.defer(ephemeral=True)
        await _send_image_memory_detail(interaction, id)

    @client.tree.command(name="image_memory_rename", description="画像参考メモの名前を変更します。")
    @app_commands.describe(id="画像参考メモID", name="新しい名前（1〜50文字）")
    async def slash_image_memory_rename(interaction: discord.Interaction, id: int, name: str) -> None:
        try:
            safe_name = sanitize_image_memory_name(name)
        except ValueError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        item = await asyncio.to_thread(rename_image_memory, interaction.user.id, id, safe_name)
        if not item:
            await interaction.response.send_message("そのIDの画像参考メモは見つからないか、名前を変更できません。", ephemeral=True)
            return
        linked_memory_id = item.get("memory_id")
        if isinstance(linked_memory_id, int):
            await asyncio.to_thread(rename_linked_image_memory, interaction.user.id, linked_memory_id, safe_name)
        await interaction.response.send_message(f"画像参考メモ ID {id} の名前を「{safe_name}」に変更しました。", ephemeral=True)

    @client.tree.command(name="image_memory_rate", description="画像参考メモを1〜5で評価します。")
    @app_commands.describe(id="画像参考メモID", rating="評価（1〜5）")
    async def slash_image_memory_rate(interaction: discord.Interaction, id: int, rating: int) -> None:
        try:
            item = await asyncio.to_thread(rate_image_memory, interaction.user.id, id, rating)
        except ValueError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        if not item:
            await interaction.response.send_message("そのIDの画像参考メモは見つからないか、評価できません。", ephemeral=True)
            return
        name = get_image_memory_name(item, id)
        stars = format_image_memory_rating(rating)
        await interaction.response.send_message(f"ID {id}「{name}」\n評価: {stars}", ephemeral=True)

    @client.tree.command(name="image_memory_use", description="画像参考メモを通常生成の自動候補に含めるか設定します。")
    @app_commands.describe(id="画像参考メモID", enabled="trueなら通常生成の参考候補に使用します")
    async def slash_image_memory_use(interaction: discord.Interaction, id: int, enabled: bool) -> None:
        item = await asyncio.to_thread(set_image_memory_generation_use, interaction.user.id, id, enabled)
        if not item:
            await interaction.response.send_message("そのIDの画像参考メモは見つからないか、設定を変更できません。", ephemeral=True)
            return
        name = get_image_memory_name(item, id)
        state = "ON" if enabled else "OFF"
        await interaction.response.send_message(f"ID {id}「{name}」\n生成への使用: {state}", ephemeral=True)

    @client.tree.command(name="image_memory_delete", description="画像参考メモを削除します。")
    @app_commands.describe(id="画像参考メモID")
    async def slash_image_memory_delete(interaction: discord.Interaction, id: int) -> None:
        item = await asyncio.to_thread(load_image_memory, id)
        deleted = await asyncio.to_thread(delete_image_memory, interaction.user.id, id)
        if deleted:
            memory_id = item.get("memory_id") if item else None
            if isinstance(memory_id, int):
                await asyncio.to_thread(delete_memory, interaction.user.id, memory_id)
            await interaction.response.send_message(f"画像参考メモ #{id} を削除しました。", ephemeral=True)
        else:
            await interaction.response.send_message("そのIDの画像参考メモは見つからないか、削除できません。", ephemeral=True)

    logger.info(
        "Registered image analysis slash commands: image_analyze, image_memory_list, image_memory_show, image_memory_rename, image_memory_rate, image_memory_use, image_memory_delete"
    )
