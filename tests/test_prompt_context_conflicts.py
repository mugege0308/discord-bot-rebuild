import unittest
from unittest import mock

from services.image_memory import build_image_memory_prompt_context
from services.memory import (
    _select_prompt_memory_tags,
    build_memory_summary,
    memory_text_to_tags,
)
from services.prompt_context_conflicts import (
    analyze_explicit_visual_constraints,
    build_explicit_lighting_control_instruction,
    filter_conflicting_weak_visual_hints,
    validate_explicit_prompt_output,
    weak_visual_hint_conflicts_with_request,
)


TRACE_EQUIVALENT_REQUEST = (
    "顔のアップは禁止。上半身だけの構図は禁止。"
    "顔〜太ももまで必ず入れる。少し引いた中距離構図。"
)


class PromptContextConflictDetectionTests(unittest.TestCase):
    def test_trace_request_detects_wide_and_prohibited_close_framing(self):
        constraints = analyze_explicit_visual_constraints(TRACE_EQUIVALENT_REQUEST)
        self.assertTrue(constraints.wide_framing)
        self.assertTrue(constraints.prohibit_close_framing)
        self.assertTrue(constraints.controls_composition)

    def test_conflicting_close_framing_tags_are_filtered_individually(self):
        kept, excluded = filter_conflicting_weak_visual_hints(
            [
                "close-up",
                "close-up of torso",
                "upper body",
                "upper body visible",
                "portrait",
                "bust shot",
                "warm lighting",
            ],
            TRACE_EQUIVALENT_REQUEST,
        )
        self.assertEqual(kept, ["warm lighting"])
        self.assertEqual(
            excluded,
            [
                "close-up",
                "close-up of torso",
                "upper body",
                "upper body visible",
                "portrait",
                "bust shot",
            ],
        )

    def test_active_pose_filters_calm_pose_hints_but_keeps_unrelated_hints(self):
        kept, excluded = filter_conflicting_weak_visual_hints(
            ["standing", "sitting", "soft lighting"],
            "街を走っている動きのあるポーズ",
        )
        self.assertEqual(kept, ["soft lighting"])
        self.assertEqual(excluded, ["standing", "sitting"])

    def test_matching_wide_memory_hint_is_not_filtered(self):
        self.assertFalse(
            weak_visual_hint_conflicts_with_request("全身構図が好き", "全身が入る構図")
        )

    def test_trace_request_requires_medium_thigh_and_lower_body_semantics(self):
        validation = validate_explicit_prompt_output(
            TRACE_EQUIVALENT_REQUEST + "腰〜太ももを画面中心。",
            "1girl, safe, side view, cowboy shot, masterpiece",
        )

        self.assertIn(
            "medium framing (medium shot equivalent)",
            validation.missing_requirements,
        )
        self.assertNotIn(
            "thigh framing (thighs visible/cowboy shot equivalent)",
            validation.missing_requirements,
        )
        self.assertIn(
            "lower-body visibility (lower body visible equivalent)",
            validation.missing_requirements,
        )
        self.assertEqual(
            validation.missing_tags,
            ("medium shot", "lower body visible"),
        )
        self.assertEqual(validation.preserved_tags, ("cowboy shot",))

    def test_prohibited_close_framing_detects_positive_conflicts(self):
        validation = validate_explicit_prompt_output(
            TRACE_EQUIVALENT_REQUEST,
            (
                "1girl, safe, medium shot, thighs visible, lower body visible, "
                "close-up of torso, upper body visible, portrait, masterpiece"
            ),
        )

        self.assertEqual(
            validation.conflicting_tags,
            ("close-up of torso", "upper body visible", "portrait"),
        )

    def test_complete_explicit_framing_is_valid(self):
        validation = validate_explicit_prompt_output(
            TRACE_EQUIVALENT_REQUEST + "腰〜太ももを画面中心。",
            (
                "1girl, safe, medium shot, thighs visible, lower body visible, "
                "side view, masterpiece"
            ),
        )
        self.assertTrue(validation.valid)

    def test_required_face_visibility_rejects_hidden_face_tags(self):
        request = (
            "顔のアップは禁止。夜凪ルナの顔、胸、腹部、腰、太ももまで"
            "必ず画面内に入れる。少し引いた中距離構図。"
        )
        for hidden_tag in (
            "face not visible",
            "faceless",
            "head out of frame",
            "cropped head",
        ):
            with self.subTest(hidden_tag=hidden_tag):
                validation = validate_explicit_prompt_output(
                    request,
                    (
                        "1girl, safe, medium shot, cowboy shot, lower body visible, "
                        f"{hidden_tag}, masterpiece"
                    ),
                )
                self.assertTrue(validation.face_visibility_required)
                self.assertEqual(validation.framing_missing_tags, ())
                self.assertEqual(validation.framing_conflicting_tags, (hidden_tag,))
                self.assertFalse(validation.valid)

    def test_close_up_prohibition_alone_does_not_require_face_visibility(self):
        validation = validate_explicit_prompt_output(
            "顔のアップは禁止。少し引いた中距離構図。",
            "1girl, safe, medium shot, face not visible, masterpiece",
        )
        self.assertFalse(validation.face_visibility_required)
        self.assertNotIn("face not visible", validation.framing_conflicting_tags)

    def test_explicit_warm_indirect_lighting_rejects_other_source_and_assumption(self):
        request = "柔らかな暖色の間接照明で照らす。"
        validation = validate_explicit_prompt_output(
            request,
            "1girl, safe, soft lighting, warm indirect lighting, firefly lighting",
            ("暖色の照明にした",),
        )

        self.assertEqual(validation.conflicting_tags, ("firefly lighting",))
        self.assertEqual(validation.assumption_violations, ("暖色の照明にした",))
        self.assertEqual(validation.requested_lighting, ("soft", "warm", "indirect"))
        self.assertIn("soft lighting", validation.preserved_tags)
        self.assertIn("warm indirect lighting", validation.preserved_tags)
        instruction = build_explicit_lighting_control_instruction(request)
        self.assertIn("soft, warm, indirect", instruction)
        self.assertIn("Do not invent a different light source", instruction)
        self.assertIn("Do not repeat lighting", instruction)

    def test_soft_and_warm_tags_are_an_indirect_lighting_proxy(self):
        validation = validate_explicit_prompt_output(
            "柔らかな暖色の間接照明。",
            "1girl, safe, soft lighting, warm tones, masterpiece",
        )
        self.assertEqual(validation.lighting_missing_tags, ())
        self.assertEqual(validation.lighting_conflicting_tags, ())
        self.assertTrue(validation.valid)

    def test_indirect_lighting_accepts_animagine_equivalent_tag(self):
        validation = validate_explicit_prompt_output(
            "間接照明の室内。",
            "1girl, safe, ambient lighting, masterpiece",
        )
        self.assertEqual(validation.lighting_missing_tags, ())
        self.assertTrue(validation.valid)

    def test_unrequested_light_source_still_conflicts_with_proxy(self):
        validation = validate_explicit_prompt_output(
            "柔らかな暖色の間接照明。",
            "1girl, safe, soft lighting, warm tones, firefly lighting, masterpiece",
        )
        self.assertEqual(validation.lighting_missing_tags, ())
        self.assertEqual(validation.lighting_conflicting_tags, ("firefly lighting",))
        self.assertFalse(validation.valid)

    def test_explicit_warm_lighting_rejects_default_natural_and_neutral_tags(self):
        validation = validate_explicit_prompt_output(
            "暖色の照明。",
            "1girl, safe, warm lighting, natural lighting, neutral lighting, masterpiece",
        )

        self.assertEqual(
            validation.lighting_conflicting_tags,
            ("natural lighting", "neutral lighting"),
        )
        self.assertFalse(validation.valid)

    def test_explicit_natural_light_is_recognized_as_a_requested_concept(self):
        validation = validate_explicit_prompt_output(
            "自然光。",
            "1girl, safe, natural lighting, masterpiece",
        )

        self.assertEqual(validation.requested_lighting, ("natural",))
        self.assertTrue(validation.valid)

    def test_explicit_background_absence_rejects_concrete_scene_tags(self):
        validation = validate_explicit_prompt_output(
            "背景は指定しない。",
            "1girl, safe, cute, classroom background, outdoor, natural lighting, masterpiece",
        )

        self.assertEqual(
            validation.background_conflicting_tags,
            ("classroom background", "outdoor"),
        )
        self.assertFalse(validation.valid)

        weak_background = validate_explicit_prompt_output(
            "背景は指定しない。",
            "1girl, safe, simple background, cute, masterpiece",
        )
        self.assertTrue(weak_background.valid)

    def test_retry_exhausted_trace_separates_face_and_lighting_results(self):
        request = (
            "構図は少し引いた中距離構図。顔のアップは禁止。上半身だけは禁止。"
            "夜凪ルナの顔、胸、腹部、腰、太ももまで必ず画面内に入れる。"
            "柔らかな暖色の間接照明。"
        )
        validation = validate_explicit_prompt_output(
            request,
            (
                "1girl, 1boy, explicit, medium shot, cowboy shot, "
                "face not visible, lower body visible, soft lighting, warm tones, "
                "masterpiece"
            ),
            ("柔らかな暖色の間接照明",),
        )

        self.assertEqual(validation.framing_missing_tags, ())
        self.assertEqual(validation.framing_conflicting_tags, ("face not visible",))
        self.assertEqual(validation.lighting_missing_tags, ())
        self.assertEqual(validation.lighting_conflicting_tags, ())
        self.assertEqual(validation.assumption_violations, ("柔らかな暖色の間接照明",))
        self.assertFalse(validation.valid)


class PromptMemoryConflictFilteringTests(unittest.TestCase):
    def test_memory_mood_and_color_rules_do_not_invent_extra_lighting_or_expression(self):
        self.assertEqual(memory_text_to_tags("かわいい"), ["cute"])
        self.assertEqual(memory_text_to_tags("寒色"), ["cool colors", "blue tones"])

    def test_memory_summary_excludes_conflicting_framing_item_only(self):
        memories = [
            {"id": 1, "text": "バストアップやポートレート寄りの構図が好き"},
            {"id": 2, "text": "暖色の柔らかい光が好き"},
        ]
        with mock.patch("services.memory.search_memories", return_value=memories):
            summary = build_memory_summary(
                1,
                TRACE_EQUIVALENT_REQUEST,
                mode="prompt",
                compact=True,
            )

        self.assertNotIn("バストアップ", summary)
        self.assertNotIn("ポートレート", summary)
        self.assertNotIn("暖色", summary)
        self.assertNotIn("warm lighting", summary)

    def test_converted_memory_tags_use_same_conflict_filter(self):
        selected = _select_prompt_memory_tags(
            TRACE_EQUIVALENT_REQUEST,
            ["upper body", "portrait", "warm lighting"],
        )
        self.assertEqual(selected, [])

    def test_non_prompt_memory_mode_is_unchanged(self):
        memories = [{"id": 1, "text": "バストアップやポートレート寄りの構図が好き"}]
        with mock.patch("services.memory.search_memories", return_value=memories):
            summary = build_memory_summary(1, TRACE_EQUIVALENT_REQUEST, mode="chat")
        self.assertIn("バストアップ", summary)

    def test_image_memory_context_filters_conflicting_tags_too(self):
        with mock.patch(
            "services.image_memory.select_image_memory_tags",
            return_value=([7], ["black hair", "upper body", "portrait", "warm lighting"]),
        ):
            context = build_image_memory_prompt_context(1, TRACE_EQUIVALENT_REQUEST)

        self.assertIn("black hair", context)
        self.assertIn("warm lighting", context)
        self.assertNotIn("upper body", context)
        self.assertNotIn("portrait", context)


if __name__ == "__main__":
    unittest.main()
