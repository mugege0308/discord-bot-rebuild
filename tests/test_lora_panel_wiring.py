import tempfile
import unittest
from pathlib import Path
from unittest import mock

from discord import app_commands

from commands import prompt as prompt_commands
from services.generate_ai_last_settings import GenerateAiLastSettings
from services.lora_catalog import COMFY_STYLES, ComfyStyle
from services.sd_control import LoraSelection


class LoraPanelWiringTests(unittest.TestCase):
    def setUp(self):
        self.original_catalog = dict(COMFY_STYLES)
        COMFY_STYLES.clear()
        COMFY_STYLES.update(
            {
                "character": ComfyStyle(
                    display_name="Character",
                    lora_name="character.safetensors",
                    trigger="character_trigger",
                    default_strength=0.8,
                    lora_type="character",
                    model_family="sdxl",
                ),
                "style": ComfyStyle(
                    display_name="Style",
                    lora_name="style.safetensors",
                    trigger="style_trigger",
                    default_strength=0.4,
                    lora_type="style",
                    model_family="sdxl",
                ),
                "additional": ComfyStyle(
                    display_name="Additional",
                    lora_name="additional.safetensors",
                    trigger="",
                    default_strength=0.2,
                    lora_type="additional",
                    model_family="sdxl",
                ),
            }
        )

    def tearDown(self):
        COMFY_STYLES.clear()
        COMFY_STYLES.update(self.original_catalog)

    def test_resolver_returns_three_typed_slots_in_stable_order(self):
        settings = prompt_commands._resolve_generate_ai_settings(
            1,
            model=app_commands.Choice(name="Animagine", value="animagine"),
            style=app_commands.Choice(name="Style", value="style"),
            strength=0.0,
            character_lora=app_commands.Choice(name="Character", value="character"),
            character_lora_strength=0.7,
            additional_lora=app_commands.Choice(name="Additional", value="additional"),
            additional_lora_strength=1.0,
            inherit_previous=False,
        )

        self.assertEqual([item.lora_type for item in settings.loras], ["character", "style", "additional"])
        self.assertEqual([item.strength for item in settings.loras], [0.7, 0.0, 1.0])
        self.assertEqual(settings.style_key, "style")
        self.assertEqual(settings.lora_strength, 0.0)

    def test_explicit_no_lora_clears_inherited_slot(self):
        previous = GenerateAiLastSettings(
            style_key="style",
            lora_strength=0.4,
            rating="auto",
            reference_strength=0.5,
            selection_ai=True,
            candidate_count=5,
            loras=(
                {"key": "style", "lora_type": "style", "strength": 0.4},
            ),
        )
        with (
            mock.patch.object(prompt_commands, "get_last_model_key", return_value="animagine"),
            mock.patch.object(prompt_commands, "get_last_settings", return_value=previous),
        ):
            settings = prompt_commands._resolve_generate_ai_settings(
                1,
                style=app_commands.Choice(name="なし", value=prompt_commands.GENERATE_AI_NO_LORA_VALUE),
            )

        self.assertEqual(settings.loras, ())
        self.assertIsNone(settings.style_key)
        self.assertIsNone(settings.lora_strength)

    def test_incompatible_model_family_is_rejected_by_shared_resolver(self):
        COMFY_STYLES["sd15"] = ComfyStyle(
            display_name="SD15",
            lora_name="sd15.safetensors",
            trigger="sd15",
            default_strength=0.5,
            lora_type="style",
            model_family="sd15",
        )
        with self.assertRaisesRegex(ValueError, "モデル系統"):
            prompt_commands._resolve_generate_ai_settings(
                1,
                style=app_commands.Choice(name="SD15", value="sd15"),
                inherit_previous=False,
            )

    def test_panel_defaults_use_resolved_lora_selection_not_catalog_copy(self):
        previous = GenerateAiLastSettings(
            style_key="style",
            lora_strength=0.55,
            rating="safe",
            reference_strength=0.5,
            selection_ai=True,
            candidate_count=4,
            loras=(
                {"key": "style", "lora_type": "style", "strength": 0.55},
            ),
        )
        with (
            mock.patch.object(prompt_commands, "get_last_model_key", return_value="animagine"),
            mock.patch.object(prompt_commands, "get_last_settings", return_value=previous),
        ):
            settings = prompt_commands.get_generate_ai_panel_defaults(1, inherit=True)

        self.assertIsInstance(settings.loras[0], LoraSelection)
        self.assertEqual(settings.loras[0].key, "style")
        self.assertEqual(settings.loras[0].strength, 0.55)

    def test_last_settings_round_trip_keeps_lora_order_and_strength(self):
        from services.generate_ai_last_settings import get_last_settings, save_last_settings

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "settings.json"
            settings = GenerateAiLastSettings(
                style_key="style",
                lora_strength=0.4,
                rating="auto",
                reference_strength=0.5,
                selection_ai=True,
                candidate_count=5,
                loras=(
                    {"key": "character", "lora_type": "character", "strength": 0.8},
                    {"key": "style", "lora_type": "style", "strength": 0.4},
                ),
            )
            save_last_settings(1, "animagine", settings, path=path)
            loaded = get_last_settings(1, "animagine", path=path)

        self.assertIsNotNone(loaded)
        self.assertEqual(
            [(item["key"], item["strength"]) for item in loaded.loras],
            [("character", 0.8), ("style", 0.4)],
        )

    def test_generate_ai_schema_exposes_lora_controls_without_use_last_reference(self):
        import main

        command = next(item for item in main.client.tree.get_commands() if item.name == "generate_ai")
        option_names = {option["name"] for option in command.to_dict(main.client.tree).get("options", [])}

        self.assertLessEqual(len(option_names), 25)
        self.assertTrue(
            {
                "style",
                "strength",
                "character_lora",
                "character_lora_strength",
                "additional_lora",
                "additional_lora_strength",
            }.issubset(option_names)
        )
        self.assertNotIn("use_last_reference", option_names)


if __name__ == "__main__":
    unittest.main()
