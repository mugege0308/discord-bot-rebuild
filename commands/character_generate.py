import logging
from typing import Any

import discord
from discord import app_commands

from commands.prompt import (
    GENERATING_MESSAGE,
    SD_DENIED_MESSAGE,
    _build_prompt_result,
    _generate_images_from_prompts,
    _safe_followup_send,
)
from services.character_template import (
    CATEGORY_ORDER,
    SPECIAL_NONE,
    build_character_instruction,
    build_selection_summary,
    load_character_template_options,
    resolve_character_choices,
    select_options,
)
from services.pc_control import is_pc_control_allowed
from services.image_memory import get_owned_image_memory_name


logger = logging.getLogger("secretary-bot.character_generate")
ONLY_OWNER_MESSAGE = "このキャラクターテンプレートは、コマンドを実行した本人だけが操作できます。"


class CharacterCategorySelect(discord.ui.Select):
    def __init__(self, key: str, category: dict[str, Any], selected: str) -> None:
        options = [
            discord.SelectOption(label=label, value=value, default=value == selected)
            for label, value in select_options(category)
        ]
        super().__init__(placeholder=f"{category['label']}を選択", options=options, min_values=1, max_values=1)
        self.category_key = key

    async def callback(self, interaction: discord.Interaction) -> None:
        view = self.view
        if not isinstance(view, CharacterSelectionView):
            await interaction.response.defer()
            return
        view.selected[self.category_key] = self.values[0]
        await interaction.response.defer()


class CharacterSelectionView(discord.ui.View):
    def __init__(
        self,
        owner_id: int,
        reference_id: int | None,
        options: dict[str, dict[str, Any]],
        reference_name: str = "",
        selected: dict[str, str] | None = None,
        page: int = 1,
    ) -> None:
        super().__init__(timeout=900)
        self.owner_id = owner_id
        self.reference_id = reference_id
        self.options_data = options
        self.reference_name = reference_name
        self.selected = selected or {key: SPECIAL_NONE for key in CATEGORY_ORDER}
        self.page = page
        keys = CATEGORY_ORDER[:4] if page == 1 else CATEGORY_ORDER[4:]
        for row, key in enumerate(keys):
            select = CharacterCategorySelect(key, options[key], self.selected.get(key, SPECIAL_NONE))
            select.row = row
            self.add_item(select)

        if page == 1:
            next_button = discord.ui.Button(label="次へ", style=discord.ButtonStyle.primary, row=4)
            next_button.callback = self._next_page
            self.add_item(next_button)
        else:
            back_button = discord.ui.Button(label="戻る", style=discord.ButtonStyle.secondary, row=3)
            back_button.callback = self._previous_page
            self.add_item(back_button)
            confirm_button = discord.ui.Button(label="確認する", style=discord.ButtonStyle.primary, row=3)
            confirm_button.callback = self._confirm
            self.add_item(confirm_button)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(ONLY_OWNER_MESSAGE, ephemeral=True)
            return False
        return True

    async def _next_page(self, interaction: discord.Interaction) -> None:
        view = CharacterSelectionView(self.owner_id, self.reference_id, self.options_data, self.reference_name, self.selected, page=2)
        await interaction.response.edit_message(content=_selection_page_message(2, self.reference_id, self.reference_name), view=view)

    async def _previous_page(self, interaction: discord.Interaction) -> None:
        view = CharacterSelectionView(self.owner_id, self.reference_id, self.options_data, self.reference_name, self.selected, page=1)
        await interaction.response.edit_message(content=_selection_page_message(1, self.reference_id, self.reference_name), view=view)

    async def _confirm(self, interaction: discord.Interaction) -> None:
        resolved, display = resolve_character_choices(self.selected, self.options_data)
        instruction = build_character_instruction(resolved, self.options_data)
        summary = build_selection_summary(display, self.options_data)
        content = _confirmation_message(summary, instruction, self.reference_id, self.reference_name)
        view = CharacterConfirmationView(
            self.owner_id,
            self.reference_id,
            self.options_data,
            self.reference_name,
            self.selected,
            summary,
            instruction,
        )
        await interaction.response.edit_message(content=content, view=view)


class CharacterConfirmationView(discord.ui.View):
    def __init__(
        self,
        owner_id: int,
        reference_id: int | None,
        options: dict[str, dict[str, Any]],
        reference_name: str,
        selected: dict[str, str],
        summary: str,
        instruction: str,
    ) -> None:
        super().__init__(timeout=900)
        self.owner_id = owner_id
        self.reference_id = reference_id
        self.options_data = options
        self.reference_name = reference_name
        self.selected = selected
        self.summary = summary
        self.instruction = instruction

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(ONLY_OWNER_MESSAGE, ephemeral=True)
            return False
        return True

    @discord.ui.button(label="生成する", style=discord.ButtonStyle.success)
    async def generate_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not is_pc_control_allowed(interaction.user.id):
            await interaction.response.send_message(SD_DENIED_MESSAGE, ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        result = await _build_prompt_result(
            interaction,
            self.instruction,
            label_prefix="character_generate",
            reference_id=self.reference_id,
        )
        if result is None:
            return
        if result.blocked or not result.positive_prompt or not result.negative_prompt:
            await _safe_followup_send(
                interaction,
                label="character_generate.prompt_blocked",
                content="キャラクタープロンプトを生成できませんでした。選択内容を見直してください。",
                ephemeral=True,
            )
            return
        await _safe_followup_send(
            interaction,
            label="character_generate.started",
            content=GENERATING_MESSAGE,
            ephemeral=True,
        )
        await _generate_images_from_prompts(
            interaction,
            positive_prompt=result.positive_prompt,
            negative_prompt=result.negative_prompt,
            label_prefix="character_generate",
        )

    @discord.ui.button(label="作り直す", style=discord.ButtonStyle.secondary)
    async def rebuild_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        view = CharacterSelectionView(self.owner_id, self.reference_id, self.options_data, self.reference_name, page=1)
        await interaction.response.edit_message(content=_selection_page_message(1, self.reference_id, self.reference_name), view=view)


def _selection_page_message(page: int, reference_id: int | None, reference_name: str = "") -> str:
    fields = "髪色・髪型・目の色・表情" if page == 1 else "服装・背景・雰囲気"
    reference_text = f"\n参考画像: ID {reference_id}「{reference_name}」" if reference_id is not None else ""
    return f"キャラクター生成テンプレート {page}/2\n{fields}を大雑把に選んでください。{reference_text}"


def _confirmation_message(summary: str, instruction: str, reference_id: int | None, reference_name: str = "") -> str:
    reference_text = f"\n- 参考画像: ID {reference_id}「{reference_name}」" if reference_id is not None else ""
    return (
        "キャラクター生成 最終確認\n\n"
        f"【選択内容】\n{summary}{reference_text}\n\n"
        f"【組み立てた日本語】\n{instruction}\n\n"
        "この内容で生成するか、選択を作り直してください。"
    )


def setup(client: discord.Client) -> None:
    @client.tree.command(name="character_generate", description="選択式テンプレートからキャラクター画像を生成します。")
    @app_commands.describe(reference_id="全タグを使う画像参考メモID。未指定でも使えます。")
    async def slash_character_generate(interaction: discord.Interaction, reference_id: int | None = None) -> None:
        try:
            options = load_character_template_options()
            reference_name = get_owned_image_memory_name(interaction.user.id, reference_id) if reference_id is not None else ""
        except ValueError as exc:
            logger.exception("Character template options load failed.")
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        view = CharacterSelectionView(interaction.user.id, reference_id, options, reference_name)
        await interaction.response.send_message(_selection_page_message(1, reference_id, reference_name), view=view, ephemeral=True)

    logger.info("Registered character generation slash command: character_generate")
