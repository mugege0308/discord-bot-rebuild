import tempfile
import unittest
from pathlib import Path
from unittest import mock

from discord import app_commands

from commands import prompt as prompt_commands
from services.generate_ai_last_settings import GenerateAiLastSettings, get_last_settings, save_last_settings
from services.image_generation_profiles import get_image_generation_profile
from services.sd_control import COMFY_CHECKPOINTS, COMFY_MODEL_FAMILIES, _resolve_comfy_checkpoint
from services.prompt_generator import PromptGeneratorError
from services.shogou_prompt_designer import load_shogou_instruction


class AdultModelProfileTests(unittest.TestCase):
    def test_adult_profiles_are_separate_profiles_with_verified_shared_checkpoints(self):
        animagine = get_image_generation_profile("animagine")
        animagine_adult = get_image_generation_profile("animagine_adult")
        janku = get_image_generation_profile("janku")
        janku_adult = get_image_generation_profile("janku_adult")

        self.assertTrue(animagine_adult.is_adult)
        self.assertTrue(janku_adult.is_adult)
        self.assertFalse(animagine.is_adult)
        self.assertFalse(janku.is_adult)
        self.assertEqual(animagine.checkpoint, animagine_adult.checkpoint)
        self.assertEqual(janku.checkpoint, janku_adult.checkpoint)
        self.assertEqual("sdxl", COMFY_MODEL_FAMILIES["animagine_adult"])
        self.assertEqual("sdxl", COMFY_MODEL_FAMILIES["janku_adult"])
        self.assertEqual(COMFY_CHECKPOINTS["animagine_adult"], _resolve_comfy_checkpoint("animagine_adult"))
        self.assertEqual(COMFY_CHECKPOINTS["janku_adult"], _resolve_comfy_checkpoint("janku_adult"))

    def test_model_choices_expose_normal_and_adult_profiles(self):
        self.assertEqual(
            [choice.value for choice in prompt_commands.GENERATE_AI_MODEL_CHOICES],
            ["animagine", "animagine_adult", "janku", "janku_adult"],
        )

    def test_adult_prompt_instructions_are_profile_specific(self):
        animagine_instruction = load_shogou_instruction("animagine_adult")
        janku_instruction = load_shogou_instruction("janku_adult")
        self.assertIn("Adult", animagine_instruction)
        self.assertIn("Adult", janku_instruction)
        self.assertNotEqual(
            get_image_generation_profile("janku").prompt_designer_instruction,
            get_image_generation_profile("janku_adult").prompt_designer_instruction,
        )

    def test_adult_model_family_is_passed_to_lora_resolution(self):
        with mock.patch(
            "commands.prompt.resolve_lora_selections",
            return_value=(),
        ) as resolve_lora:
            prompt_commands._resolve_generate_ai_settings(
                1,
                model=app_commands.Choice(name="Adult", value="animagine_adult"),
                inherit_previous=False,
            )
        self.assertTrue(resolve_lora.called)
        self.assertEqual("sdxl", resolve_lora.call_args.kwargs["model_family"])

    def test_last_settings_are_scoped_by_adult_model_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            settings = GenerateAiLastSettings(
                style_key=None,
                lora_strength=None,
                rating="explicit",
                reference_strength=0.5,
                selection_ai=True,
                candidate_count=5,
            )
            save_last_settings(7, "animagine_adult", settings, path)
            self.assertEqual(settings, get_last_settings(7, "animagine_adult", path))
            self.assertIsNone(get_last_settings(7, "animagine", path))

    def test_normal_profile_rejects_adult_only_rating_but_adult_accepts_it(self):
        with self.assertRaisesRegex(PromptGeneratorError, "Adultモデル"):
            prompt_commands._resolve_generate_ai_settings(
                1,
                model=app_commands.Choice(name="Normal", value="animagine"),
                rating=app_commands.Choice(name="Explicit", value="explicit"),
                inherit_previous=False,
            )
        resolved = prompt_commands._resolve_generate_ai_settings(
            1,
            model=app_commands.Choice(name="Adult", value="animagine_adult"),
            rating=app_commands.Choice(name="Explicit", value="explicit"),
            inherit_previous=False,
        )
        self.assertEqual("animagine_adult", resolved.model_key)
        self.assertEqual("explicit", resolved.rating)


if __name__ == "__main__":
    unittest.main()
