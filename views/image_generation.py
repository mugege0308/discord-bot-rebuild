"""Touch-friendly entry point for the existing /generate_ai pipeline."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass

import discord

from commands import prompt as prompt_commands
from services.image_composition import COMPOSITION_LABELS, get_image_size_spec
from services.image_generation_profiles import DEFAULT_IMAGE_GENERATION_MODEL_KEY, get_image_generation_profile
from services.saved_references import list_references
from views.panel import OwnerOnlyView


logger = logging.getLogger("secretary-bot.image_generation_panel")

PANEL_TIMEOUT_SECONDS = 900
PANEL_PROMPT_PREVIEW_LIMIT = 500
PANEL_TEXT_ERROR = "パネルの設定を読み込めませんでした。いったん閉じて、もう一度開いてね。"
PANEL_GENERATION_ERROR = "画像生成を開始できませんでした。Botのログを確認してね。"


def _choice_name(choices, value: str | None, fallback: str = "未選択") -> str:
    if value is None:
        return fallback
    for choice in choices:
        if choice.value == value:
            return choice.name
    return value


def _selection_for_type(settings, lora_type: str):
    return next((item for item in settings.loras if item.lora_type == lora_type), None)


@dataclass
class ImageGenerationPanelState:
    """Ephemeral, per-user state for one touch-panel session."""

    user_id: int
    request: str = ""
    model_key: str = DEFAULT_IMAGE_GENERATION_MODEL_KEY
    character: str | None = None
    input_mode: str = prompt_commands.JAPANESE_AI_INPUT_MODE
    rating: str = prompt_commands.PROMPT_RATING_AUTO
    style_key: str | None = None
    style_strength: float | None = None
    character_lora_key: str | None = None
    character_lora_strength: float | None = None
    additional_lora_key: str | None = None
    additional_lora_strength: float | None = None
    seed: int | None = None
    reference_mode: str = prompt_commands.REFERENCE_MODE_NONE
    use_last_reference: bool = False
    reference_strength: float | None = None
    saved_reference_id: int | None = None
    selection_ai: bool = prompt_commands.GENERATE_AI_DEFAULT_SELECTION_AI
    candidate_count: int = 5
    size_key: str = prompt_commands.DEFAULT_SIZE_KEY
    composition_preset: str = prompt_commands.DEFAULT_COMPOSITION_PRESET

    @classmethod
    def from_user(
        cls,
        user_id: int,
        *,
        inherit: bool = True,
    ) -> "ImageGenerationPanelState":
        settings = prompt_commands.get_generate_ai_panel_defaults(user_id, inherit=inherit)
        character_lora = _selection_for_type(settings, "character")
        style_lora = _selection_for_type(settings, "style")
        additional_lora = _selection_for_type(settings, "additional")
        reference_mode = getattr(settings, "reference_mode", prompt_commands.REFERENCE_MODE_NONE)
        return cls(
            user_id=user_id,
            model_key=settings.model_key,
            input_mode=getattr(settings, "input_mode", prompt_commands.JAPANESE_AI_INPUT_MODE),
            rating=settings.rating,
            style_key=settings.style_key,
            style_strength=(
                settings.lora_strength
                if style_lora is None
                else style_lora.strength
            ),
            character_lora_key=None if character_lora is None else character_lora.key,
            character_lora_strength=(
                None if character_lora is None else character_lora.strength
            ),
            additional_lora_key=None if additional_lora is None else additional_lora.key,
            additional_lora_strength=(
                None if additional_lora is None else additional_lora.strength
            ),
            reference_mode=reference_mode,
            reference_strength=(
                settings.reference_strength
                if reference_mode != prompt_commands.REFERENCE_MODE_NONE
                else None
            ),
            selection_ai=settings.selection_ai,
            candidate_count=settings.candidate_count,
            size_key=settings.size_key,
            composition_preset=settings.composition_preset,
        )


class _ImageGenerationView(OwnerOnlyView):
    def __init__(self, owner_id: int, bot_mention: str, *, timeout: float = PANEL_TIMEOUT_SECONDS) -> None:
        super().__init__(owner_id, bot_mention, timeout=timeout)
        self.message: discord.Message | None = None

    async def on_timeout(self) -> None:
        if self.message is None:
            return
        try:
            await self.message.edit(view=None)
        except (discord.HTTPException, discord.NotFound):
            logger.debug("Image generation panel message already expired.", exc_info=True)


def _new_main_view(owner_id: int, bot_mention: str, state: ImageGenerationPanelState) -> "ImageGenerationPanel":
    return ImageGenerationPanel(owner_id, bot_mention, state=state)


async def _edit_main_panel(
    interaction: discord.Interaction,
    owner_id: int,
    bot_mention: str,
    state: ImageGenerationPanelState,
) -> None:
    view = _new_main_view(owner_id, bot_mention, state)
    await interaction.response.edit_message(content=view.render(), view=view)
    view.message = interaction.message


async def _edit_picker_panel(
    interaction: discord.Interaction,
    view: _ImageGenerationView,
    content: str,
) -> None:
    await interaction.response.edit_message(content=content, view=view)
    view.message = interaction.message


async def _send_component_error(interaction: discord.Interaction, message: str) -> None:
    if interaction.response.is_done():
        await interaction.followup.send(message, ephemeral=True)
    else:
        await interaction.response.send_message(message, ephemeral=True)


class ImageGenerationPanel(_ImageGenerationView):
    def __init__(
        self,
        owner_id: int,
        bot_mention: str,
        *,
        state: ImageGenerationPanelState | None = None,
    ) -> None:
        super().__init__(owner_id, bot_mention)
        self.state = state or ImageGenerationPanelState.from_user(owner_id)
        try:
            prompt_commands.sanitize_panel_state_for_model(self.state)
        except ValueError:
            self.state.model_key = prompt_commands.DEFAULT_IMAGE_GENERATION_MODEL_KEY
            prompt_commands.sanitize_panel_state_for_model(self.state)
        profile = get_image_generation_profile(self.state.model_key)
        for child in self.children:
            if not isinstance(child, discord.ui.Button):
                continue
            if child.label and "LoRA" in child.label:
                child.disabled = not profile.supports_lora
            elif child.label and "参照画像" in child.label:
                child.disabled = not profile.supports_reference

    def render(self) -> str:
        state = self.state
        try:
            profile = get_image_generation_profile(state.model_key)
            size_label = get_image_size_spec(state.size_key).label
            composition_label = COMPOSITION_LABELS[state.composition_preset]
            loras = prompt_commands.resolve_lora_selections(
                character_key=state.character_lora_key,
                character_strength=state.character_lora_strength,
                style_key=state.style_key,
                style_strength=state.style_strength,
                additional_key=state.additional_lora_key,
                additional_strength=state.additional_lora_strength,
                model_family=profile.model_family,
            )
            lora_text = prompt_commands.get_lora_status_text(loras).replace("\n", " / ")
        except (KeyError, ValueError):
            profile = get_image_generation_profile(DEFAULT_IMAGE_GENERATION_MODEL_KEY)
            size_label = state.size_key
            composition_label = state.composition_preset
            lora_text = "LoRA: 設定値を確認中"

        prompt_text = state.request.strip() or "未入力"
        if len(prompt_text) > PANEL_PROMPT_PREVIEW_LIMIT:
            prompt_text = prompt_text[:PANEL_PROMPT_PREVIEW_LIMIT] + "…"
        prompt_text = prompt_text.replace("\n", " ")
        character_text = state.character or "未選択"
        reference_text = (
            f"{_choice_name(prompt_commands.GENERATE_AI_REFERENCE_MODE_CHOICES, state.reference_mode)} / "
            + (
                _reference_info_text(self.owner_id)
                if state.use_last_reference
                else "画像は /generate_ai で添付"
            )
        )
        seed_text = "ランダム" if state.seed is None else str(state.seed)
        input_mode_text = _choice_name(
            prompt_commands.GENERATE_AI_INPUT_MODE_CHOICES,
            state.input_mode,
        )
        rating_text = _choice_name(
            prompt_commands.GENERATE_AI_PROMPT_RATING_CHOICES,
            state.rating,
        )
        return "\n".join(
            [
                "🎨 画像生成",
                f"Prompt: {prompt_text}",
                f"Model: {profile.display_name}",
                f"Character: {character_text}",
                f"Size: {size_label}",
                f"Composition: {composition_label}",
                f"Rating: {rating_text}",
                lora_text,
                f"Reference: {reference_text}",
                f"Input: {input_mode_text} / Seed: {seed_text}",
                (
                    "Qwenは自然文・512×512・1枚生成（LoRA/参照/ポーズ/Variationなし）"
                    if not profile.supports_lora
                    else ""
                ),
            ]
        )

    async def _refresh_from_modal(self, interaction: discord.Interaction) -> None:
        view = _new_main_view(self.owner_id, self.bot_mention, self.state)
        if interaction.message is not None:
            await interaction.response.edit_message(content=view.render(), view=view)
            view.message = interaction.message
            return
        if self.message is not None:
            await interaction.response.defer(ephemeral=True)
            await self.message.edit(content=view.render(), view=view)
            view.message = self.message
            return
        await interaction.response.send_message(view.render(), view=view, ephemeral=True)
        try:
            view.message = await interaction.original_response()
        except discord.HTTPException:
            logger.debug("Failed to retain image generation panel message.", exc_info=True)

    @discord.ui.button(label="✏️ プロンプト", style=discord.ButtonStyle.primary)
    async def prompt_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_modal(PromptModal(self))

    @discord.ui.button(label="👤 キャラ", style=discord.ButtonStyle.secondary)
    async def character_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_modal(CharacterModal(self))

    @discord.ui.button(label="🧠 Model", style=discord.ButtonStyle.secondary)
    async def model_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        view = SettingPickerPanel(
            self.owner_id,
            self.bot_mention,
            self.state,
            title="Modelを選択",
            setting_name="model_key",
            options=prompt_commands.GENERATE_AI_MODEL_CHOICES,
        )
        await _edit_picker_panel(interaction, view, "Modelを選んでね。")

    @discord.ui.button(label="📐 サイズ", style=discord.ButtonStyle.secondary)
    async def size_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        view = SettingPickerPanel(
            self.owner_id,
            self.bot_mention,
            self.state,
            title="サイズを選択",
            setting_name="size_key",
            options=prompt_commands.get_model_size_choices(self.state.model_key),
        )
        await _edit_picker_panel(interaction, view, "画像サイズを選んでね。")

    @discord.ui.button(label="🧍 構図", style=discord.ButtonStyle.secondary)
    async def composition_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        view = SettingPickerPanel(
            self.owner_id,
            self.bot_mention,
            self.state,
            title="構図を選択",
            setting_name="composition_preset",
            options=prompt_commands.GENERATE_AI_COMPOSITION_CHOICES,
        )
        await _edit_picker_panel(interaction, view, "構図を選んでね。")

    @discord.ui.button(label="🔞 Rating", style=discord.ButtonStyle.secondary)
    async def rating_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        view = SettingPickerPanel(
            self.owner_id,
            self.bot_mention,
            self.state,
            title="Ratingを選択",
            setting_name="rating",
            options=prompt_commands.GENERATE_AI_PROMPT_RATING_CHOICES,
        )
        await _edit_picker_panel(interaction, view, "Ratingを選んでね。")

    @discord.ui.button(label="🧩 LoRA", style=discord.ButtonStyle.secondary)
    async def lora_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        view = LoraPanel(self.owner_id, self.bot_mention, self.state)
        await _edit_picker_panel(interaction, view, "使いたいLoRAを枠ごとに選んでね。")

    @discord.ui.button(label="🖼️ 参照画像", style=discord.ButtonStyle.secondary)
    async def reference_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        view = ReferencePanel(self.owner_id, self.bot_mention, self.state)
        await _edit_picker_panel(interaction, view, view.render())

    @discord.ui.button(label="⚙️ 詳細", style=discord.ButtonStyle.secondary)
    async def advanced_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        view = AdvancedPanel(self.owner_id, self.bot_mention, self.state)
        await _edit_picker_panel(interaction, view, view.render())

    @discord.ui.button(label="✨ 生成する", style=discord.ButtonStyle.success)
    async def generate_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self.state.request.strip():
            await interaction.response.send_message("先に「✏️ プロンプト」から要望を入力してね。", ephemeral=True)
            return
        try:
            await prompt_commands.run_generate_ai_from_panel(
                interaction,
                request=self.state.request,
                character=self.state.character,
                model_key=self.state.model_key,
                input_mode=self.state.input_mode,
                rating=self.state.rating,
                style_key=self.state.style_key,
                style_strength=self.state.style_strength,
                character_lora_key=self.state.character_lora_key,
                character_lora_strength=self.state.character_lora_strength,
                additional_lora_key=self.state.additional_lora_key,
                additional_lora_strength=self.state.additional_lora_strength,
                seed=self.state.seed,
                reference_mode=self.state.reference_mode,
                use_last_reference=self.state.use_last_reference,
                reference_strength=self.state.reference_strength,
                saved_reference_id=self.state.saved_reference_id,
                selection_ai=self.state.selection_ai,
                candidate_count=self.state.candidate_count,
                size_key=self.state.size_key,
                composition_preset=self.state.composition_preset,
            )
        except ValueError as exc:
            await _send_component_error(interaction, str(exc))
        except Exception:
            logger.exception("Failed to run /generate_ai from image generation panel.")
            await _send_component_error(interaction, PANEL_GENERATION_ERROR)

    @discord.ui.button(label="🔄 リセット", style=discord.ButtonStyle.secondary)
    async def reset_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        try:
            self.state = ImageGenerationPanelState.from_user(self.owner_id, inherit=False)
            await interaction.response.edit_message(content=self.render(), view=self)
            self.message = interaction.message
        except Exception:
            logger.exception("Failed to reset image generation panel.")
            await _send_component_error(interaction, PANEL_TEXT_ERROR)

    @discord.ui.button(label="🔙 戻る", style=discord.ButtonStyle.secondary)
    async def back_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        from views.panel import PANEL_MESSAGE, OperationPanel

        view = OperationPanel(self.owner_id, self.bot_mention)
        await interaction.response.edit_message(content=PANEL_MESSAGE, view=view)


class PromptModal(discord.ui.Modal):
    def __init__(self, panel: ImageGenerationPanel) -> None:
        super().__init__(title="画像生成プロンプト")
        self.panel = panel
        self.request_input = discord.ui.TextInput(
            label="作りたい画像の内容",
            style=discord.TextStyle.paragraph,
            placeholder="日本語入力、英語直接入力、外部Prompt整理に対応。外部PromptはPositive:/Negative:も利用できます。",
            default=panel.state.request,
            max_length=prompt_commands.RAW_PROMPT_MAX_LENGTH,
            required=True,
        )
        self.add_item(self.request_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.panel.owner_id:
            await interaction.response.send_message("このパネルは呼び出した本人用だよ。", ephemeral=True)
            return
        value = str(self.request_input.value).strip()
        if not value:
            await interaction.response.send_message("プロンプトを入力してね。", ephemeral=True)
            return
        self.panel.state.request = value
        await self.panel._refresh_from_modal(interaction)


class CharacterModal(discord.ui.Modal):
    def __init__(self, panel: ImageGenerationPanel) -> None:
        super().__init__(title="キャラクターPreset")
        self.panel = panel
        self.character_input = discord.ui.TextInput(
            label="登録済みキャラクター名",
            placeholder="空欄で未選択に戻します。完全一致で入力してね。",
            default=panel.state.character or "",
            max_length=100,
            required=False,
        )
        self.add_item(self.character_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.panel.owner_id:
            await interaction.response.send_message("このパネルは呼び出した本人用だよ。", ephemeral=True)
            return
        self.panel.state.character = str(self.character_input.value).strip() or None
        await self.panel._refresh_from_modal(interaction)


class _PanelChoiceSelect(discord.ui.Select):
    def __init__(
        self,
        owner_id: int,
        bot_mention: str,
        state: ImageGenerationPanelState,
        *,
        setting_name: str,
        options,
        placeholder: str,
    ) -> None:
        self.owner_id = owner_id
        self.bot_mention = bot_mention
        self.state = state
        self.setting_name = setting_name
        select_options = [
            discord.SelectOption(label=str(choice.name)[:100], value=str(choice.value))
            for choice in options
        ]
        super().__init__(
            placeholder=placeholder,
            min_values=1,
            max_values=1,
            options=select_options,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("このパネルは呼び出した本人用だよ。", ephemeral=True)
            return
        setattr(self.state, self.setting_name, self.values[0])
        if self.setting_name == "model_key":
            try:
                prompt_commands.sanitize_panel_state_for_model(self.state)
            except ValueError as exc:
                await interaction.response.send_message(str(exc), ephemeral=True)
                return
        if self.setting_name == "reference_mode" and self.values[0] == prompt_commands.REFERENCE_MODE_NONE:
            self.state.use_last_reference = False
            self.state.reference_strength = None
            self.state.saved_reference_id = None
        await _edit_main_panel(interaction, self.owner_id, self.bot_mention, self.state)


class SettingPickerPanel(_ImageGenerationView):
    def __init__(
        self,
        owner_id: int,
        bot_mention: str,
        state: ImageGenerationPanelState,
        *,
        title: str,
        setting_name: str,
        options,
    ) -> None:
        super().__init__(owner_id, bot_mention, timeout=PANEL_TIMEOUT_SECONDS)
        self.state = state
        self.add_item(
            _PanelChoiceSelect(
                owner_id,
                bot_mention,
                state,
                setting_name=setting_name,
                options=options,
                placeholder=title,
            )
        )

    @discord.ui.button(label="🔙 画像生成パネルへ", style=discord.ButtonStyle.secondary)
    async def back_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await _edit_main_panel(interaction, self.owner_id, self.bot_mention, self.state)


class LoraSlotSelect(discord.ui.Select):
    def __init__(
        self,
        owner_id: int,
        bot_mention: str,
        state: ImageGenerationPanelState,
        *,
        lora_type: str,
        label: str,
    ) -> None:
        self.owner_id = owner_id
        self.bot_mention = bot_mention
        self.state = state
        self.lora_type = lora_type
        options = [discord.SelectOption(label="なし（無効）", value=prompt_commands.GENERATE_AI_NO_LORA_VALUE)]
        for display_name, key in prompt_commands.get_comfy_lora_choice_specs(lora_type):
            options.append(discord.SelectOption(label=str(display_name)[:100], value=str(key)))
        super().__init__(placeholder=label, min_values=1, max_values=1, options=options)
        if not prompt_commands.model_supports(state.model_key, "lora"):
            self.disabled = True

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("このパネルは呼び出した本人用だよ。", ephemeral=True)
            return
        key = self.values[0]
        key = None if key == prompt_commands.GENERATE_AI_NO_LORA_VALUE else key
        setting_name = {
            "character": "character_lora_key",
            "style": "style_key",
            "additional": "additional_lora_key",
        }[self.lora_type]
        strength_name = {
            "character": "character_lora_strength",
            "style": "style_strength",
            "additional": "additional_lora_strength",
        }[self.lora_type]
        setattr(self.state, setting_name, key)
        if key is None:
            setattr(self.state, strength_name, None)
        await _edit_main_panel(interaction, self.owner_id, self.bot_mention, self.state)


class LoraPanel(_ImageGenerationView):
    def __init__(self, owner_id: int, bot_mention: str, state: ImageGenerationPanelState) -> None:
        super().__init__(owner_id, bot_mention, timeout=PANEL_TIMEOUT_SECONDS)
        self.state = state
        for lora_type, label in (
            ("character", "キャラクターLoRA"),
            ("style", "絵柄LoRA"),
            ("additional", "追加LoRA"),
        ):
            self.add_item(
                LoraSlotSelect(
                    owner_id,
                    bot_mention,
                    state,
                    lora_type=lora_type,
                    label=label,
                )
            )

        if not prompt_commands.model_supports(state.model_key, "lora"):
            for child in self.children:
                child.disabled = True

    @discord.ui.button(label="強度を設定", style=discord.ButtonStyle.primary)
    async def strength_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_modal(LoraStrengthModal(self))

    @discord.ui.button(label="🔙 画像生成パネルへ", style=discord.ButtonStyle.secondary)
    async def back_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await _edit_main_panel(interaction, self.owner_id, self.bot_mention, self.state)


class LoraStrengthModal(discord.ui.Modal):
    def __init__(self, panel: LoraPanel) -> None:
        super().__init__(title="LoRA強度")
        self.panel = panel
        values = (
            panel.state.character_lora_strength,
            panel.state.style_strength,
            panel.state.additional_lora_strength,
        )
        self.inputs = []
        for label, value in zip(("キャラLoRA", "絵柄LoRA", "追加LoRA"), values):
            field = discord.ui.TextInput(
                label=f"{label} strength",
                default="" if value is None else f"{value:g}",
                placeholder="空欄ならメタデータの推奨値",
                max_length=8,
                required=False,
            )
            self.inputs.append(field)
            self.add_item(field)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.panel.owner_id:
            await interaction.response.send_message("このパネルは呼び出した本人用だよ。", ephemeral=True)
            return
        parsed = []
        try:
            for field in self.inputs:
                raw = str(field.value).strip()
                if not raw:
                    parsed.append(None)
                    continue
                value = float(raw)
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError
                parsed.append(value)
        except ValueError:
            await interaction.response.send_message("LoRA strengthは0.0～1.0で入力してね。", ephemeral=True)
            return
        self.panel.state.character_lora_strength, self.panel.state.style_strength, self.panel.state.additional_lora_strength = parsed
        await _refresh_main_from_aux_modal(self.panel, self.panel.state, interaction)


async def _refresh_main_from_aux_modal(
    panel: _ImageGenerationView,
    state: ImageGenerationPanelState,
    interaction: discord.Interaction,
) -> None:
    view = _new_main_view(panel.owner_id, panel.bot_mention, state)
    if panel.message is not None and interaction.message is None:
        await interaction.response.defer(ephemeral=True)
        await panel.message.edit(content=view.render(), view=view)
        view.message = panel.message
        return
    if interaction.message is not None:
        await interaction.response.edit_message(content=view.render(), view=view)
        view.message = interaction.message
        return
    await interaction.response.send_message(view.render(), view=view, ephemeral=True)
    try:
        view.message = await interaction.original_response()
    except discord.HTTPException:
        logger.debug("Failed to retain image generation panel after modal.", exc_info=True)


def _reference_strength_text(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


def _reference_source_label(source_type: object) -> str:
    return {
        "new_attachment": "今回の添付",
        "last_reference": "直前の画像",
        "saved_reference": "保存済み参照",
    }.get(str(source_type), "参照画像")


def _reference_info_text(owner_id: int) -> str:
    info = prompt_commands.get_last_reference_image_info(owner_id)
    if not info:
        return "画像未指定"
    width = info.get("width") or "?"
    height = info.get("height") or "?"
    source = _reference_source_label(info.get("source_type"))
    filename = str(info.get("original_filename") or "-")
    if len(filename) > 32:
        filename = f"…{filename[-31:]}"
    return f"{width}×{height} / {source} / {filename}"


class ReferencePanel(_ImageGenerationView):
    def __init__(self, owner_id: int, bot_mention: str, state: ImageGenerationPanelState) -> None:
        super().__init__(owner_id, bot_mention, timeout=PANEL_TIMEOUT_SECONDS)
        self.state = state
        self._reference_supported = prompt_commands.model_supports(state.model_key, "reference")
        if not self._reference_supported:
            for child in self.children:
                child.disabled = True

    def render(self) -> str:
        if not self._reference_supported:
            return "🖼️ 参考画像\nこのモデルでは参考画像を利用できません。"
        available = prompt_commands.has_last_reference_image(self.owner_id)
        availability = "利用可能" if available else "未保存"
        return (
            "🖼️ 参考画像\n"
            f"直前の /generate_ai 参考画像: {availability}\n"
            f"参照画像: {_reference_info_text(self.owner_id)}\n"
            f"現在: {_choice_name(prompt_commands.GENERATE_AI_REFERENCE_MODE_CHOICES, self.state.reference_mode)} / "
            f"{'保存ID ' + str(self.state.saved_reference_id) if self.state.saved_reference_id else ('画像あり' if self.state.use_last_reference else '画像未指定')}\n"
            f"strength: {_reference_strength_text(self.state.reference_strength) or '既定値'}\n\n"
            "画像本体は /generate_ai の参考画像添付、または直前の画像を使います。"
        )

    async def _set_reference_mode(self, interaction: discord.Interaction, mode: str) -> None:
        if not self._reference_supported:
            await interaction.response.send_message("このモデルでは参考画像を利用できません。", ephemeral=True)
            return
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("このパネルは呼び出した本人用だよ。", ephemeral=True)
            return
        self.state.reference_mode = mode
        if mode == prompt_commands.REFERENCE_MODE_NONE:
            self.state.use_last_reference = False
            self.state.reference_strength = None
            self.state.saved_reference_id = None
        await _edit_main_panel(interaction, self.owner_id, self.bot_mention, self.state)

    @discord.ui.button(label="参照なし", style=discord.ButtonStyle.secondary)
    async def mode_none_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._set_reference_mode(interaction, prompt_commands.REFERENCE_MODE_NONE)

    @discord.ui.button(label="構図参照", style=discord.ButtonStyle.primary)
    async def mode_composition_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._set_reference_mode(interaction, prompt_commands.REFERENCE_MODE_COMPOSITION)

    @discord.ui.button(label="ポーズ参照", style=discord.ButtonStyle.primary)
    async def mode_pose_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await self._set_reference_mode(interaction, prompt_commands.REFERENCE_MODE_POSE)

    @discord.ui.button(label="直前の画像を使う", style=discord.ButtonStyle.primary)
    async def use_last_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not prompt_commands.has_last_reference_image(self.owner_id):
            await interaction.response.send_message(
                "再利用できる参考画像がありません。先に /generate_ai の reference_image で1枚送ってね。",
                ephemeral=True,
            )
            return
        if self.state.reference_mode == prompt_commands.REFERENCE_MODE_NONE:
            self.state.reference_mode = prompt_commands.REFERENCE_MODE_COMPOSITION
        self.state.use_last_reference = True
        self.state.saved_reference_id = None
        await _edit_main_panel(interaction, self.owner_id, self.bot_mention, self.state)

    @discord.ui.button(label="保存済みを選ぶ", style=discord.ButtonStyle.primary)
    async def saved_reference_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("このパネルは呼び出した本人用だよ。", ephemeral=True)
            return
        try:
            items = await asyncio.to_thread(list_references, self.owner_id, limit=25)
        except Exception:
            logger.exception("Failed to list saved references for panel: user_id=%s", self.owner_id)
            await interaction.response.send_message("保存済み参考画像の一覧を取得できませんでした。", ephemeral=True)
            return
        if not items:
            await interaction.response.send_message("保存済み参考画像はありません。先に /ref_save を使ってね。", ephemeral=True)
            return
        view = SavedReferencePickerPanel(self.owner_id, self.bot_mention, self.state, items)
        await _edit_picker_panel(interaction, view, "使いたい保存済み参考画像を選んでね。")

    @discord.ui.button(label="参考画像なし", style=discord.ButtonStyle.secondary)
    async def disable_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        self.state.use_last_reference = False
        self.state.saved_reference_id = None
        self.state.reference_mode = prompt_commands.REFERENCE_MODE_NONE
        self.state.reference_strength = None
        await _edit_main_panel(interaction, self.owner_id, self.bot_mention, self.state)

    @discord.ui.button(label="強度", style=discord.ButtonStyle.secondary)
    async def strength_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_modal(ReferenceStrengthModal(self))

    @discord.ui.button(label="🔙 戻る", style=discord.ButtonStyle.secondary)
    async def back_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await _edit_main_panel(interaction, self.owner_id, self.bot_mention, self.state)


class SavedReferenceSelect(discord.ui.Select):
    def __init__(self, owner_id: int, bot_mention: str, state: ImageGenerationPanelState, items) -> None:
        self.owner_id = owner_id
        self.bot_mention = bot_mention
        self.state = state
        self.items_by_id = {str(item.id): item for item in items}
        options = [
            discord.SelectOption(
                label=f"{item.name[:70]} ({item.kind})",
                value=str(item.id),
                description=(", ".join(item.tags) or "タグなし")[:100],
            )
            for item in items[:25]
        ]
        super().__init__(placeholder="保存済み参考画像", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("このパネルは呼び出した本人用だよ。", ephemeral=True)
            return
        item = self.items_by_id.get(self.values[0])
        if item is None:
            await interaction.response.send_message("保存済み参考画像が見つかりません。", ephemeral=True)
            return
        self.state.saved_reference_id = item.id
        self.state.use_last_reference = False
        self.state.reference_mode = (
            prompt_commands.REFERENCE_MODE_POSE
            if item.kind == "pose"
            else prompt_commands.REFERENCE_MODE_COMPOSITION
        )
        await _edit_main_panel(interaction, self.owner_id, self.bot_mention, self.state)


class SavedReferencePickerPanel(_ImageGenerationView):
    def __init__(self, owner_id: int, bot_mention: str, state: ImageGenerationPanelState, items) -> None:
        super().__init__(owner_id, bot_mention, timeout=PANEL_TIMEOUT_SECONDS)
        self.state = state
        self.add_item(SavedReferenceSelect(owner_id, bot_mention, state, items))

    @discord.ui.button(label="保存済み参照を解除", style=discord.ButtonStyle.secondary)
    async def clear_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message("このパネルは呼び出した本人用だよ。", ephemeral=True)
            return
        self.state.saved_reference_id = None
        self.state.reference_mode = prompt_commands.REFERENCE_MODE_NONE
        self.state.use_last_reference = False
        self.state.reference_strength = None
        await _edit_main_panel(interaction, self.owner_id, self.bot_mention, self.state)

    @discord.ui.button(label="🔙 参考画像設定へ", style=discord.ButtonStyle.secondary)
    async def back_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        view = ReferencePanel(self.owner_id, self.bot_mention, self.state)
        await _edit_picker_panel(interaction, view, view.render())


class ReferenceStrengthModal(discord.ui.Modal):
    def __init__(self, panel: ReferencePanel) -> None:
        super().__init__(title="参考画像strength")
        self.panel = panel
        self.value_input = discord.ui.TextInput(
            label="reference strength",
            default=_reference_strength_text(panel.state.reference_strength),
            placeholder="0.0～1.0",
            max_length=8,
            required=False,
        )
        self.add_item(self.value_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.panel.owner_id:
            await interaction.response.send_message("このパネルは呼び出した本人用だよ。", ephemeral=True)
            return
        raw = str(self.value_input.value).strip()
        try:
            value = None if not raw else float(raw)
            if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError
        except ValueError:
            await interaction.response.send_message("reference strengthは0.0～1.0で入力してね。", ephemeral=True)
            return
        self.panel.state.reference_strength = value
        await _refresh_main_from_aux_modal(self.panel, self.panel.state, interaction)


class AdvancedPanel(_ImageGenerationView):
    def __init__(self, owner_id: int, bot_mention: str, state: ImageGenerationPanelState) -> None:
        super().__init__(owner_id, bot_mention, timeout=PANEL_TIMEOUT_SECONDS)
        self.state = state
        self.add_item(
            _PanelChoiceSelect(
                owner_id,
                bot_mention,
                state,
                setting_name="input_mode",
                options=prompt_commands.GENERATE_AI_INPUT_MODE_CHOICES,
                placeholder="入力方式",
            )
        )
        if not prompt_commands.model_supports(state.model_key, "bulk"):
            for child in self.children:
                if isinstance(child, discord.ui.Button) and child.label in {"Selection AI切替", "候補枚数"}:
                    child.disabled = True

    def render(self) -> str:
        input_mode = _choice_name(prompt_commands.GENERATE_AI_INPUT_MODE_CHOICES, self.state.input_mode)
        return (
            "⚙️ 詳細設定\n"
            f"入力方式: {input_mode}\n"
            f"Selection AI: {'有効' if self.state.selection_ai else '無効'}\n"
            f"候補枚数: {self.state.candidate_count}\n"
            f"Seed: {'ランダム' if self.state.seed is None else self.state.seed}"
        )

    @discord.ui.button(label="Seed", style=discord.ButtonStyle.secondary)
    async def seed_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_modal(SeedModal(self))

    @discord.ui.button(label="Selection AI切替", style=discord.ButtonStyle.secondary)
    async def selection_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        self.state.selection_ai = not self.state.selection_ai
        view = AdvancedPanel(self.owner_id, self.bot_mention, self.state)
        await _edit_picker_panel(interaction, view, view.render())

    @discord.ui.button(label="候補枚数", style=discord.ButtonStyle.secondary)
    async def candidate_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.send_modal(CandidateCountModal(self))

    @discord.ui.button(label="🔙 戻る", style=discord.ButtonStyle.secondary)
    async def back_button(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await _edit_main_panel(interaction, self.owner_id, self.bot_mention, self.state)


class SeedModal(discord.ui.Modal):
    def __init__(self, panel: AdvancedPanel) -> None:
        super().__init__(title="Seed設定")
        self.panel = panel
        self.seed_input = discord.ui.TextInput(
            label="Seed",
            default="" if panel.state.seed is None else str(panel.state.seed),
            placeholder="空欄ならランダム（0～4294967295）",
            max_length=10,
            required=False,
        )
        self.add_item(self.seed_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.panel.owner_id:
            await interaction.response.send_message("このパネルは呼び出した本人用だよ。", ephemeral=True)
            return
        raw = str(self.seed_input.value).strip()
        try:
            value = None if not raw else int(raw)
            if value is not None and not 0 <= value <= 2**32 - 1:
                raise ValueError
        except ValueError:
            await interaction.response.send_message("Seedは0～4294967295の整数で入力してね。", ephemeral=True)
            return
        self.panel.state.seed = value
        await _refresh_main_from_aux_modal(self.panel, self.panel.state, interaction)


class CandidateCountModal(discord.ui.Modal):
    def __init__(self, panel: AdvancedPanel) -> None:
        super().__init__(title="Selection AI候補枚数")
        self.panel = panel
        self.count_input = discord.ui.TextInput(
            label="候補枚数",
            default=str(panel.state.candidate_count),
            placeholder="1～20",
            max_length=2,
            required=True,
        )
        self.add_item(self.count_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.panel.owner_id:
            await interaction.response.send_message("このパネルは呼び出した本人用だよ。", ephemeral=True)
            return
        try:
            value = int(str(self.count_input.value).strip())
            if not 1 <= value <= 20:
                raise ValueError
        except ValueError:
            await interaction.response.send_message("候補枚数は1～20で入力してね。", ephemeral=True)
            return
        self.panel.state.candidate_count = value
        await _refresh_main_from_aux_modal(self.panel, self.panel.state, interaction)
