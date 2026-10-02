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
        self.assertIn("`summary` は採用した生成内容の日本語一文", instruction)
        self.assertIn("`assumptions` は採用した補完・矛盾解決だけの配列", instruction)
        self.assertIn("summary・assumptionsへ戻さない", instruction)
        self.assertIn("requested rating、resolved tags、tag IDs", instruction)

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

    def test_janku_instruction_checks_six_completion_categories_and_respects_explicit_input(self):
        instruction = load_shogou_instruction("janku")
        for category in (
            "A. framing / composition",
            "B. gaze",
            "C. pose",
            "D. character features",
            "E. outfit details",
            "F. background",
        ):
            self.assertIn(category, instruction)
        self.assertIn("足りないカテゴリだけを補う", instruction)
        self.assertIn("明示指定と競合する補完をしない", instruction)
        self.assertIn("物語設定や小物を勝手に大量追加しない", instruction)

    def test_janku_instruction_avoids_default_close_up_and_completes_background(self):
        instruction = load_shogou_instruction("janku")
        self.assertIn("`close-up` / `face close-up` を既定にしない", instruction)
        self.assertIn("`upper body` または `medium shot` を基本候補", instruction)
        self.assertIn("`medium shot` または `cowboy shot` を優先", instruction)
        self.assertIn("`portrait` という語だけを理由に", instruction)
        for explicit_background in ("plain background", "white background", "transparent background"):
            self.assertIn(explicit_background, instruction)
        self.assertIn("背景要素を最低1つ補完", instruction)
        self.assertIn("`detailed background` を機械的に毎回追加しない", instruction)

    def test_janku_instruction_prioritizes_known_character_details_without_strong_invention(self):
        instruction = load_shogou_instruction("janku")
        self.assertIn("キャラクター辞書から渡された正式タグ", instruction)
        self.assertIn("正式な視覚特徴を `positive` へ反映", instruction)
        self.assertIn("強く捏造しない", instruction)
        self.assertIn("控えめな補完", instruction)

    def test_janku_instruction_concretizes_expression_without_overexpansion(self):
        instruction = load_shogou_instruction("janku")
        for vague_expression in ("sexy expression", "cute expression", "sad expression"):
            self.assertIn(vague_expression, instruction)
        for concrete_expression in (
            "slightly blushed cheeks",
            "half-lidded eyes",
            "soft smile",
            "slightly parted lips",
            "teary eyes",
        ):
            self.assertIn(concrete_expression, instruction)
        self.assertIn("毎回すべての表情タグを追加せず", instruction)
        self.assertIn("ユーザーの意図より強い表現へ勝手に拡張せず", instruction)
        self.assertIn("表情だけで構図全体を決めない", instruction)

    def test_janku_instruction_avoids_duplicate_and_conflicting_visual_tags(self):
        instruction = load_shogou_instruction("janku")
        self.assertIn("genericなタグより具体的な視覚タグを優先", instruction)
        self.assertIn("`cool colors` と `blue tones`", instruction)
        self.assertIn("`soft lighting` と `neutral lighting`", instruction)
        self.assertIn("矛盾・競合する照明タグを同時に出さない", instruction)
        self.assertIn("`positive` を極端に短くしすぎず", instruction)
        self.assertIn("長大で散漫なタグ列", instruction)

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
