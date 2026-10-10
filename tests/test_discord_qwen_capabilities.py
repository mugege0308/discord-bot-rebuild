import asyncio
import os
import unittest
from unittest import mock

from discord import app_commands

from commands import prompt as prompt_commands
from services.image_generation_profiles import QWEN_IMAGE_MODEL_KEY
from views.image_generation import ImageGenerationPanel, ImageGenerationPanelState


class DiscordQwenCapabilityTests(unittest.TestCase):
    def _qwen_choices(self):
        return [
            *prompt_commands.GENERATE_AI_MODEL_CHOICES,
            app_commands.Choice(name="Qwen-Image 2.1 (W4A8)", value=QWEN_IMAGE_MODEL_KEY),
        ]

    def test_feature_flag_controls_discord_model_choices(self):
        with mock.patch.dict(os.environ, {"ENABLE_QWEN_IMAGE_2_1": "0"}, clear=False):
            self.assertNotIn(
                QWEN_IMAGE_MODEL_KEY,
                {choice.value for choice in prompt_commands._build_generate_ai_model_choices()},
            )
        with mock.patch.dict(os.environ, {"ENABLE_QWEN_IMAGE_2_1": "1"}, clear=False):
            self.assertIn(
                QWEN_IMAGE_MODEL_KEY,
                {choice.value for choice in prompt_commands._build_generate_ai_model_choices()},
            )

    def test_qwen_size_and_capabilities_are_exposed_to_panel(self):
        with mock.patch.dict(os.environ, {"ENABLE_QWEN_IMAGE_2_1": "1"}, clear=False):
            self.assertEqual(["square"], [choice.value for choice in prompt_commands.get_model_size_choices(QWEN_IMAGE_MODEL_KEY)])
            self.assertFalse(prompt_commands.model_supports(QWEN_IMAGE_MODEL_KEY, "lora"))
            self.assertFalse(prompt_commands.model_supports(QWEN_IMAGE_MODEL_KEY, "reference"))
            self.assertFalse(prompt_commands.model_supports(QWEN_IMAGE_MODEL_KEY, "pose"))
            self.assertFalse(prompt_commands.model_supports(QWEN_IMAGE_MODEL_KEY, "bulk"))

    def test_panel_model_switch_clears_stale_sdxl_state(self):
        state = ImageGenerationPanelState(
            user_id=1,
            model_key=QWEN_IMAGE_MODEL_KEY,
            style_key="natsume_soft_style_v1.safetensors",
            character_lora_key="character.safetensors",
            additional_lora_key="extra.safetensors",
            style_strength=0.7,
            character_lora_strength=0.6,
            additional_lora_strength=0.5,
            reference_mode=prompt_commands.REFERENCE_MODE_POSE,
            use_last_reference=True,
            saved_reference_id=42,
            reference_strength=0.8,
            selection_ai=True,
            candidate_count=4,
            size_key="portrait",
        )
        with mock.patch.dict(os.environ, {"ENABLE_QWEN_IMAGE_2_1": "1"}, clear=False):
            prompt_commands.sanitize_panel_state_for_model(state)

        self.assertIsNone(state.style_key)
        self.assertIsNone(state.character_lora_key)
        self.assertIsNone(state.additional_lora_key)
        self.assertEqual(prompt_commands.REFERENCE_MODE_NONE, state.reference_mode)
        self.assertFalse(state.use_last_reference)
        self.assertIsNone(state.saved_reference_id)
        self.assertFalse(state.selection_ai)
        self.assertEqual(1, state.candidate_count)
        self.assertEqual("square", state.size_key)

    def test_qwen_panel_disables_lora_and_reference_controls(self):
        state = ImageGenerationPanelState(user_id=1, model_key=QWEN_IMAGE_MODEL_KEY)
        with mock.patch.dict(os.environ, {"ENABLE_QWEN_IMAGE_2_1": "1"}, clear=False):
            panel = ImageGenerationPanel(1, "@bot", state=state)

        disabled_labels = {
            child.label
            for child in panel.children
            if getattr(child, "disabled", False)
        }
        self.assertIn("🧩 LoRA", disabled_labels)
        self.assertIn("🖼️ 参照画像", disabled_labels)
        self.assertIn("Square", panel.render())
        self.assertIn("自然文", panel.render())

    def test_qwen_bridge_sanitizes_stale_values_before_callback(self):
        captured = []

        async def callback(interaction, request, **kwargs):
            captured.append((request, kwargs))

        previous_callback = prompt_commands._generate_ai_panel_callback
        prompt_commands._generate_ai_panel_callback = callback
        try:
            with mock.patch.dict(os.environ, {"ENABLE_QWEN_IMAGE_2_1": "1"}, clear=False), mock.patch.object(
                prompt_commands, "GENERATE_AI_MODEL_CHOICES", self._qwen_choices()
            ):
                asyncio.run(
                    prompt_commands.run_generate_ai_from_panel(
                        "interaction",
                        request="a quiet red fox in snow",
                        model_key=QWEN_IMAGE_MODEL_KEY,
                        style_key="natsume_soft_style_v1.safetensors",
                        character_lora_key="character.safetensors",
                        additional_lora_key="extra.safetensors",
                        reference_mode=prompt_commands.REFERENCE_MODE_POSE,
                        use_last_reference=True,
                        saved_reference_id=99,
                        selection_ai=True,
                        candidate_count=5,
                        size_key="portrait",
                    )
                )
        finally:
            prompt_commands._generate_ai_panel_callback = previous_callback

        self.assertEqual(1, len(captured))
        _, payload = captured[0]
        self.assertEqual(QWEN_IMAGE_MODEL_KEY, payload["model"].value)
        self.assertEqual(prompt_commands.GENERATE_AI_NO_LORA_VALUE, payload["style"].value)
        self.assertEqual(prompt_commands.GENERATE_AI_NO_LORA_VALUE, payload["character_lora"].value)
        self.assertEqual(prompt_commands.GENERATE_AI_NO_LORA_VALUE, payload["additional_lora"].value)
        self.assertEqual(prompt_commands.REFERENCE_MODE_NONE, payload["reference_mode"].value)
        self.assertFalse(payload["use_last_reference"])
        self.assertIsNone(payload["saved_reference_id"])
        self.assertFalse(payload["selection_ai"])
        self.assertEqual(1, payload["candidate_count"])
        self.assertEqual("square", payload["size"].value)

    def test_qwen_generation_options_use_natural_language_defaults(self):
        with mock.patch.dict(os.environ, {"ENABLE_QWEN_IMAGE_2_1": "1"}, clear=False):
            options = prompt_commands._build_generation_options(
                "a calm red fox in snow",
                "masterpiece, low quality",
                model_key=QWEN_IMAGE_MODEL_KEY,
                model_family="qwen_image",
                loras=(),
                selection_enabled=True,
                selection_generate_count=4,
            )

        self.assertEqual("qwen_image", options.engine)
        self.assertEqual("qwen_image_2_1_w4a8", options.workflow_key)
        self.assertEqual((512, 512), (options.width, options.height))
        self.assertEqual((8, 1.0, "euler", "simple"), (options.steps, options.cfg, options.sampler, options.scheduler))
        self.assertEqual(1, options.count)
        self.assertEqual("", options.negative_prompt)
        self.assertFalse(options.loras)
        self.assertNotIn("masterpiece", options.prompt)

    def test_direct_qwen_resolver_discards_explicit_sdxl_controls(self):
        qwen = app_commands.Choice(name="Qwen-Image 2.1", value=QWEN_IMAGE_MODEL_KEY)
        stale_lora = app_commands.Choice(name="old SDXL LoRA", value="natsume")
        with mock.patch.dict(os.environ, {"ENABLE_QWEN_IMAGE_2_1": "1"}, clear=False), mock.patch.object(
            prompt_commands, "GENERATE_AI_MODEL_CHOICES", self._qwen_choices()
        ):
            settings = prompt_commands._resolve_generate_ai_settings(
                1,
                model=qwen,
                style=stale_lora,
                reference_mode=app_commands.Choice(name="ポーズ参照", value=prompt_commands.REFERENCE_MODE_POSE),
                candidate_count=4,
                size=app_commands.Choice(name="Portrait", value="portrait"),
                inherit_previous=False,
            )

        self.assertFalse(settings.loras)
        self.assertEqual(prompt_commands.REFERENCE_MODE_NONE, settings.reference_mode)
        self.assertFalse(settings.selection_ai)
        self.assertEqual(1, settings.candidate_count)
        self.assertEqual("square", settings.size_key)


if __name__ == "__main__":
    unittest.main()
