import asyncio
import unittest
from unittest import mock

from commands import prompt as prompt_commands
from services.generate_ai_last_settings import GenerateAiLastSettings
from views.image_generation import ImageGenerationPanelState


class PanelSettingsBridgeTests(unittest.TestCase):
    def test_reset_defaults_are_safe_and_complete(self):
        settings = prompt_commands.get_generate_ai_panel_defaults(1001, inherit=False)

        self.assertEqual(settings.model_key, "animagine")
        self.assertEqual(settings.input_mode, "ja_ai")
        self.assertEqual(settings.rating, "auto")
        self.assertIsNone(settings.seed)
        self.assertTrue(settings.selection_ai)
        self.assertGreaterEqual(settings.candidate_count, 1)
        self.assertEqual(settings.size_key, "square")
        self.assertEqual(settings.composition_preset, "none")
        self.assertEqual(settings.reference_mode, "none")
        self.assertFalse(settings.use_last_reference)
        self.assertFalse(settings.loras)

    def test_last_settings_are_inherited_with_lora_resolution(self):
        previous = GenerateAiLastSettings(
            style_key="natsume",
            lora_strength=0.65,
            rating="safe",
            reference_strength=0.9,
            selection_ai=False,
            candidate_count=3,
            size_key="portrait",
            composition_preset="full_body",
            loras=(
                {"key": "natsume", "lora_type": "style", "strength": 0.65},
            ),
            reference_mode="pose",
            input_mode="prompt_import",
        )
        with (
            mock.patch.object(prompt_commands, "get_last_model_key", return_value="janku"),
            mock.patch.object(prompt_commands, "get_last_settings", return_value=previous),
        ):
            settings = prompt_commands.get_generate_ai_panel_defaults(1001, inherit=True)

        self.assertEqual(settings.model_key, "janku")
        self.assertEqual(settings.input_mode, "prompt_import")
        self.assertEqual(settings.rating, "safe")
        self.assertFalse(settings.selection_ai)
        self.assertEqual(settings.candidate_count, 3)
        self.assertEqual(settings.size_key, "portrait")
        self.assertEqual(settings.composition_preset, "full_body")
        self.assertEqual(settings.reference_mode, "none")
        self.assertFalse(settings.use_last_reference)
        self.assertIsNone(settings.reference_strength)
        self.assertEqual(settings.loras[0].key, "natsume")
        self.assertEqual(settings.loras[0].lora_type, "style")

    def test_panel_state_can_be_constructed_from_shared_defaults(self):
        with mock.patch.object(
            prompt_commands,
            "get_generate_ai_panel_defaults",
            return_value=prompt_commands.get_generate_ai_panel_defaults(1001, inherit=False),
        ):
            state = ImageGenerationPanelState.from_user(1001)

        self.assertEqual(state.model_key, "animagine")
        self.assertEqual(state.input_mode, "ja_ai")
        self.assertEqual(state.reference_mode, "none")

    def test_bridge_calls_one_existing_callback_and_forwards_shared_fields(self):
        captured = []

        async def callback(interaction, request, **kwargs):
            captured.append((interaction, request, kwargs))

        previous_callback = prompt_commands._generate_ai_panel_callback
        prompt_commands._generate_ai_panel_callback = callback
        try:
            asyncio.run(
                prompt_commands.run_generate_ai_from_panel(
                    "interaction",
                    request="上半身を正面から",
                    character="キャラ",
                    model_key="janku",
                    input_mode="prompt_import",
                    rating="safe",
                    style_key="natsume",
                    style_strength=0.55,
                    seed=123,
                    selection_ai=False,
                    candidate_count=2,
                    size_key="portrait",
                    composition_preset="upper_body",
                )
            )
        finally:
            prompt_commands._generate_ai_panel_callback = previous_callback

        self.assertEqual(len(captured), 1)
        _, request, payload = captured[0]
        self.assertEqual(request, "上半身を正面から")
        self.assertEqual(payload["model"].value, "janku")
        self.assertEqual(payload["input_mode"].value, "prompt_import")
        self.assertEqual(payload["rating"].value, "safe")
        self.assertEqual(payload["style"].value, "natsume")
        self.assertEqual(payload["strength"], 0.55)
        self.assertEqual(payload["seed"], 123)
        self.assertFalse(payload["selection_ai"])
        self.assertEqual(payload["candidate_count"], 2)
        self.assertEqual(payload["size"].value, "portrait")
        self.assertEqual(payload["composition_preset"].value, "upper_body")
        self.assertEqual(payload["reference_mode"].value, "none")

    def test_bridge_rejects_invalid_shared_values(self):
        async def callback(*args, **kwargs):
            raise AssertionError("callback must not run")

        previous_callback = prompt_commands._generate_ai_panel_callback
        prompt_commands._generate_ai_panel_callback = callback
        try:
            with self.assertRaises(ValueError):
                asyncio.run(
                    prompt_commands.run_generate_ai_from_panel(
                        "interaction",
                        request="test",
                        size_key="unknown",
                    )
                )
        finally:
            prompt_commands._generate_ai_panel_callback = previous_callback


if __name__ == "__main__":
    unittest.main()
