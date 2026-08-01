import unittest
from unittest import mock

from services.image_generation_profiles import (
    DEFAULT_IMAGE_GENERATION_MODEL_KEY,
    get_image_generation_profile,
)
from services.prompt_generator import PromptGeneratorError
from services.shogou_prompt_designer import (
    generate_shogou_composition_variation,
    generate_shogou_custom_instruction_variation,
    generate_shogou_prompt_result,
    load_shogou_instruction,
)


VALID_JSON = (
    '{"positive":"1girl, anime style","negative":"lowres",'
    '"summary":"要約","assumptions":[]}'
)


class ImageGenerationProfileTestCase(unittest.TestCase):
    def test_default_profile_is_animagine(self):
        profile = get_image_generation_profile()
        self.assertEqual(DEFAULT_IMAGE_GENERATION_MODEL_KEY, "animagine")
        self.assertEqual(profile.model_key, "animagine")
        self.assertEqual(profile.checkpoint, "animagineXL40Opt_clear.safetensors")
        self.assertEqual(profile.filename_prefix, "Shogou_AnimagineXL")

    def test_janku_profile_uses_exact_checkpoint_instruction_and_prefix(self):
        profile = get_image_generation_profile("janku")
        self.assertEqual(profile.display_name, "JANKU v7.77")
        self.assertEqual(profile.checkpoint, "JANKUTrainedChenkinNoobai_v777.safetensors")
        self.assertEqual(profile.prompt_designer_instruction.name, "shogou_chan_prompt_designer_instruction_janku.md")
        self.assertEqual(profile.filename_prefix, "Shogou_JANKU")

    def test_unknown_model_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "対応していません"):
            get_image_generation_profile("../../unknown")

    def test_animagine_instruction_is_existing_file(self):
        instruction = load_shogou_instruction("animagine")
        self.assertIn("Animagine XL 4.0", instruction)
        self.assertNotIn("JANKU v7.77 向け", instruction)

    def test_animagine_instruction_makes_summary_and_assumptions_model_neutral(self):
        instruction = load_shogou_instruction("animagine")
        for prohibited_subject in ("生成モデル名", "checkpoint名", "Prompt Designer名"):
            self.assertIn(prohibited_subject, instruction)
        self.assertIn("使用する生成モデルを推測・説明しない", instruction)
        self.assertIn("画像内容だけ", instruction)
        self.assertIn("実際に補完した条件や解決した矛盾だけ", instruction)

    def test_janku_instruction_contains_required_positive_and_negative_policy(self):
        instruction = load_shogou_instruction("janku")
        for tag in ("anime style", "2d", "illustration", "clean lineart", "detailed shading"):
            self.assertIn(tag, instruction)
        for tag in ("photorealistic", "realistic", "3d", "doll", "render"):
            self.assertIn(tag, instruction)
        self.assertIn("タグ順序", instruction)
        self.assertIn("重複", instruction)

    def test_janku_instruction_makes_summary_and_assumptions_model_neutral(self):
        instruction = load_shogou_instruction("janku")
        for prohibited_subject in ("生成モデル名", "checkpoint名", "Prompt Designer名"):
            self.assertIn(prohibited_subject, instruction)
        self.assertIn("使用する生成モデルを推測・説明しない", instruction)
        self.assertIn("画像内容だけ", instruction)
        self.assertIn("実際に補完した条件や解決した矛盾だけ", instruction)
        self.assertNotIn("Animagine XL 4.0", instruction)

    def test_prompt_designer_rejects_unknown_model(self):
        with self.assertRaises(PromptGeneratorError):
            load_shogou_instruction("unknown")


class PromptDesignerModelWiringTestCase(unittest.TestCase):
    def _common_patches(self):
        return (
            mock.patch("services.shogou_prompt_designer.append_memory_to_system_prompt", side_effect=lambda body, *_args, **_kwargs: body),
            mock.patch("services.shogou_prompt_designer.build_image_memory_prompt_context", return_value=""),
            mock.patch("services.shogou_prompt_designer._post_chat_completion", return_value=VALID_JSON),
        )

    def test_normal_generation_selects_janku_instruction_and_keeps_context_services(self):
        memory_patch, image_memory_patch, post_patch = self._common_patches()
        with mock.patch(
            "services.shogou_prompt_designer.load_shogou_instruction", return_value="JANKU_INSTRUCTION"
        ) as instruction_mock, memory_patch as memory_mock, image_memory_patch as image_memory_mock, post_patch:
            generate_shogou_prompt_result("猫耳の女の子", user_id=42, model_key="janku")

        instruction_mock.assert_called_once_with("janku")
        memory_mock.assert_called_once()
        image_memory_mock.assert_called_once_with(42, "猫耳の女の子", None)

    def test_composition_regeneration_selects_parent_model_instruction(self):
        memory_patch, image_memory_patch, post_patch = self._common_patches()
        with mock.patch(
            "services.shogou_prompt_designer.load_shogou_instruction", return_value="JANKU_INSTRUCTION"
        ) as instruction_mock, memory_patch, image_memory_patch, post_patch:
            generate_shogou_composition_variation(
                "元の依頼", "1girl", "lowres", "要約", (), user_id=42, model_key="janku"
            )
        instruction_mock.assert_called_once_with("janku")

    def test_custom_regeneration_selects_parent_model_instruction(self):
        memory_patch, image_memory_patch, post_patch = self._common_patches()
        with mock.patch(
            "services.shogou_prompt_designer.load_shogou_instruction", return_value="JANKU_INSTRUCTION"
        ) as instruction_mock, memory_patch, image_memory_patch, post_patch:
            generate_shogou_custom_instruction_variation(
                "元の依頼",
                "1girl",
                "lowres",
                "要約",
                (),
                "笑顔にする",
                user_id=42,
                model_key="janku",
            )
        instruction_mock.assert_called_once_with("janku")


if __name__ == "__main__":
    unittest.main()
