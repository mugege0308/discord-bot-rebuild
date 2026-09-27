import json
import unittest
from unittest import mock

from services.prompt_semantic_grounding import ground_prompt_semantics
from services.shogou_prompt_designer import generate_shogou_prompt_result


class PromptSemanticGroundingTestCase(unittest.TestCase):
    def test_unrequested_scene_and_pose_completion_is_removed(self):
        result = ground_prompt_semantics(
            "1girl, asta (honkai: star rail), safe, cute, standing, facing viewer, "
            "hands behind back, slouched posture, school uniform, classroom background, "
            "natural lighting, soft lighting, balanced color palette, masterpiece, "
            "high score, great score, absurdres",
            user_input="崩壊スターレイルのアスター 後ろで手を組んで前かがみ",
        )

        self.assertIn("hands behind back", result.positive)
        self.assertIn("leaning forward", result.positive)
        for unwanted in (
            "cute",
            "standing",
            "facing viewer",
            "slouched posture",
            "school uniform",
            "classroom background",
            "natural lighting",
            "soft lighting",
            "balanced color palette",
        ):
            self.assertNotIn(unwanted, result.positive)
        self.assertEqual(
            result.pose_normalizations,
            ("前かがみ -> leaning forward", "後ろで手を組む -> hands behind back"),
        )

    def test_explicit_clothing_background_and_lighting_are_preserved(self):
        result = ground_prompt_semantics(
            "1girl, asta (honkai: star rail), safe, school uniform, classroom background, "
            "soft lighting, smile, masterpiece",
            user_input="崩壊スターレイルのアスターが制服で教室にいる",
        )

        self.assertIn("school uniform", result.positive)
        self.assertIn("classroom background", result.positive)
        self.assertNotIn("soft lighting", result.positive)
        self.assertNotIn("smile", result.positive)

    def test_explicit_soft_lighting_and_smile_are_preserved(self):
        result = ground_prompt_semantics(
            "1girl, asta (honkai: star rail), safe, classroom, soft lighting, smile, masterpiece",
            user_input="崩壊スターレイルのアスターが柔らかい光の教室で笑っている",
        )

        self.assertIn("classroom", result.positive)
        self.assertIn("soft lighting", result.positive)
        self.assertIn("smile", result.positive)

    def test_front_bending_is_not_slouched_posture(self):
        result = ground_prompt_semantics(
            "1girl, kochiya sanae, safe, slouched posture, masterpiece",
            user_input="東風谷早苗が前かがみ",
        )

        self.assertIn("leaning forward", result.positive)
        self.assertNotIn("slouched posture", result.positive)

    def test_explicit_white_hair_does_not_authorize_unrequested_length(self):
        result = ground_prompt_semantics(
            "1girl, kochiya sanae, safe, white hair, long hair, hair ornament, masterpiece",
            user_input="東風谷早苗が白髪になっている",
        )

        self.assertIn("white hair", result.positive)
        self.assertNotIn("long hair", result.positive)
        self.assertNotIn("hair ornament", result.positive)

    def test_unrequested_anime_style_and_summary_assumptions_are_grounded(self):
        result = ground_prompt_semantics(
            "1girl, asta (honkai: star rail), safe, cute, anime style, hands behind back, "
            "leaning forward, masterpiece",
            user_input="崩壊スターレイルのアスター 後ろで手を組んで前かがみ",
            assumptions=("anime style", "blue tones", "stylish outfit"),
        )

        self.assertNotIn("cute", result.positive)
        self.assertNotIn("anime style", result.positive)
        self.assertEqual(
            result.removed_assumptions,
            ("anime style", "blue tones", "stylish outfit"),
        )

    def test_explicit_anime_style_is_preserved(self):
        result = ground_prompt_semantics(
            "1girl, safe, anime style, masterpiece",
            user_input="アニメ風の女の子",
        )

        self.assertIn("anime style", result.positive)

    def test_prompt_designer_result_exposes_grounding_trace(self):
        text = "崩壊スターレイルのアスター 後ろで手を組んで前かがみ"
        raw = json.dumps(
            {
                "positive": (
                    "1girl, asta (honkai: star rail), safe, cute, standing, "
                    "hands behind back, slouched posture, school uniform, "
                    "classroom background, natural lighting, masterpiece"
                ),
                "negative": "lowres",
                "summary": "test",
                "assumptions": ["school uniform", "classroom setting"],
            },
            ensure_ascii=False,
        )
        with mock.patch(
            "services.shogou_prompt_designer._post_chat_completion",
            return_value=raw,
        ), mock.patch(
            "services.shogou_prompt_designer.load_shogou_instruction",
            return_value="SYSTEM",
        ), mock.patch(
            "services.shogou_prompt_designer.append_memory_to_system_prompt",
            side_effect=lambda body, *_args, **_kwargs: body,
        ), mock.patch(
            "services.shogou_prompt_designer.build_image_memory_prompt_context",
            return_value="",
        ):
            result = generate_shogou_prompt_result(
                text,
                requested_rating="safe",
            )

        self.assertIn("school uniform", result.raw_llm_positive)
        self.assertIn("hands behind back", result.user_grounded_tags)
        self.assertIn("school uniform", result.removed_unrequested_semantic_tags)
        self.assertIn("leaning forward", result.pose_normalizations[0])
        self.assertNotIn("school uniform", result.positive)
        self.assertEqual(result.assumptions, ())

    def test_prompt_designer_rechecks_late_additions_and_rebuilds_summary(self):
        text = "崩壊スターレイルのアスター 後ろで手を組んで前かがみ"
        raw = json.dumps(
            {
                "positive": (
                    "1 girl, asta (honkai: star rail), safe, cute, anime style, "
                    "hands behind back, leaning forward, masterpiece, high score, "
                    "great score, absurdres"
                ),
                "negative": "lowres",
                "summary": (
                    "A cute girl in a stylish outfit with blue tones is seen from the back, "
                    "hands behind her back and slouched posture."
                ),
                "assumptions": ["anime style", "blue tones", "stylish outfit"],
            },
            ensure_ascii=False,
        )
        with (
            mock.patch(
                "services.shogou_prompt_designer._post_chat_completion",
                return_value=raw,
            ),
            mock.patch(
                "services.shogou_prompt_designer.load_shogou_instruction",
                return_value="SYSTEM",
            ),
            mock.patch(
                "services.shogou_prompt_designer.append_memory_to_system_prompt",
                side_effect=lambda body, *_args, **_kwargs: body,
            ),
            mock.patch(
                "services.shogou_prompt_designer.build_image_memory_prompt_context",
                return_value="",
            ),
            mock.patch(
                "services.shogou_prompt_designer.apply_prompt_tag_resolution",
                side_effect=lambda positive, _resolution: positive + ", cute, anime style",
            ),
        ):
            result = generate_shogou_prompt_result(
                text,
                requested_rating="safe",
            )

        self.assertIn("asta (honkai: star rail)", result.positive)
        self.assertIn("1girl", result.positive)
        self.assertIn("hands behind back", result.positive)
        self.assertIn("leaning forward", result.positive)
        self.assertNotIn("1 girl", result.positive)
        self.assertNotIn("cute", result.positive)
        self.assertNotIn("anime style", result.positive)
        self.assertEqual(result.assumptions, ())
        self.assertIn("cute", result.raw_llm_positive)
        self.assertIn("cute", result.raw_summary)
        self.assertEqual(
            result.raw_assumptions,
            ("anime style", "blue tones", "stylish outfit"),
        )
        self.assertEqual(result.final_summary, result.summary)
        self.assertNotIn("cute", result.summary)
        self.assertNotIn("stylish outfit", result.summary)
        self.assertNotIn("blue tones", result.summary)
        self.assertNotIn("slouched posture", result.summary)
        self.assertTrue(result.semantic_grounding_result)


if __name__ == "__main__":
    unittest.main()
