import json
import re
import unittest
from unittest import mock

from services.shogou_prompt_designer import (
    ShogouPromptResult,
    _strip_think_blocks,
    generate_shogou_composition_variation,
    generate_shogou_prompt_result,
    load_shogou_instruction,
    parse_shogou_prompt_response,
)


VALID_JSON = (
    '{"positive": "1girl, solo, safe, masterpiece, high score, great score, absurdres", '
    '"negative": "lowres, bad anatomy", '
    '"summary": "テスト用の要約", '
    '"assumptions": ["補完1", "補完2"]}'
)


class ParseShogouPromptResponseTestCase(unittest.TestCase):
    def test_valid_four_key_json_parses_all_fields(self):
        result = parse_shogou_prompt_response(VALID_JSON)

        self.assertFalse(result.blocked)
        self.assertEqual(result.positive, "1girl, solo, safe, masterpiece, high score, great score, absurdres")
        self.assertEqual(result.negative, "lowres, bad anatomy")
        self.assertEqual(result.summary, "テスト用の要約")
        self.assertEqual(result.assumptions, ("補完1", "補完2"))

    def test_json_wrapped_in_closed_think_block_is_removed_before_parsing(self):
        raw = f"<think>ここで色々検討する。矛盾はない。</think>{VALID_JSON}"
        result = parse_shogou_prompt_response(raw)

        self.assertFalse(result.blocked)
        self.assertEqual(result.summary, "テスト用の要約")

    def test_unclosed_think_like_tag_does_not_erase_the_whole_response(self):
        # No closing </think> anywhere; a naive "delete everything after <think>"
        # implementation would destroy the JSON below. The real JSON must still
        # be recoverable via brace-scanning instead of being wiped out.
        raw = f'<think>閉じタグを出力し忘れた例 {{不完全な括弧\n{VALID_JSON}'
        result = parse_shogou_prompt_response(raw)

        self.assertFalse(result.blocked)
        self.assertEqual(result.summary, "テスト用の要約")

    def test_strip_think_blocks_leaves_text_unchanged_when_unclosed(self):
        raw = "<think>never closes " + VALID_JSON
        self.assertEqual(_strip_think_blocks(raw), raw)

    def test_markdown_code_fence_wrapped_json_is_parsed(self):
        raw = f"```json\n{VALID_JSON}\n```"
        result = parse_shogou_prompt_response(raw)

        self.assertFalse(result.blocked)
        self.assertEqual(result.positive, "1girl, solo, safe, masterpiece, high score, great score, absurdres")

    def test_explanatory_text_around_json_is_parsed(self):
        raw = f"以下が結果です。\n{VALID_JSON}\nご確認ください。"
        result = parse_shogou_prompt_response(raw)

        self.assertFalse(result.blocked)
        self.assertEqual(result.negative, "lowres, bad anatomy")

    def test_missing_required_key_is_blocked(self):
        raw = '{"positive": "1girl", "negative": "lowres", "summary": "テスト"}'
        result = parse_shogou_prompt_response(raw)

        self.assertTrue(result.blocked)
        self.assertIn("assumptions", result.blocked_reason)

    def test_assumptions_not_a_list_is_blocked(self):
        raw = '{"positive": "1girl", "negative": "lowres", "summary": "テスト", "assumptions": "not a list"}'
        result = parse_shogou_prompt_response(raw)

        self.assertTrue(result.blocked)
        self.assertIn("assumptions", result.blocked_reason)

    def test_summary_not_a_string_is_blocked(self):
        raw = '{"positive": "1girl", "negative": "lowres", "summary": 123, "assumptions": []}'
        result = parse_shogou_prompt_response(raw)

        self.assertTrue(result.blocked)
        self.assertIn("summary", result.blocked_reason)

    def test_empty_positive_is_blocked(self):
        raw = '{"positive": "   ", "negative": "lowres", "summary": "テスト", "assumptions": []}'
        result = parse_shogou_prompt_response(raw)

        self.assertTrue(result.blocked)
        self.assertIn("positive", result.blocked_reason)

    def test_empty_negative_is_blocked(self):
        raw = '{"positive": "1girl", "negative": "", "summary": "テスト", "assumptions": []}'
        result = parse_shogou_prompt_response(raw)

        self.assertTrue(result.blocked)
        self.assertIn("negative", result.blocked_reason)

    def test_unparseable_response_is_blocked(self):
        result = parse_shogou_prompt_response("this is not json at all")

        self.assertTrue(result.blocked)


class GenerateShogouPromptResultMemoryWiringTestCase(unittest.TestCase):
    def test_normal_memory_is_appended_after_instruction_body(self):
        with mock.patch(
            "services.shogou_prompt_designer.load_shogou_instruction",
            return_value="INSTRUCTION_BODY",
        ), mock.patch(
            "services.shogou_prompt_designer.append_memory_to_system_prompt",
            return_value="INSTRUCTION_BODY\n\nMEMORY_SUMMARY",
        ) as append_mock, mock.patch(
            "services.shogou_prompt_designer.build_image_memory_prompt_context",
            return_value="",
        ) as image_memory_mock, mock.patch(
            "services.shogou_prompt_designer._post_chat_completion",
            return_value=VALID_JSON,
        ) as post_mock:
            result = generate_shogou_prompt_result("猫耳の女の子", user_id=42, reference_id=None)

        append_mock.assert_called_once_with("INSTRUCTION_BODY", 42, "猫耳の女の子", mode="prompt")
        image_memory_mock.assert_called_once_with(42, "猫耳の女の子", None)
        sent_system_prompt = post_mock.call_args.args[0]
        self.assertEqual(sent_system_prompt, "INSTRUCTION_BODY\n\nMEMORY_SUMMARY")
        self.assertFalse(result.blocked)

    def test_reference_id_specified_is_passed_through_to_image_memory(self):
        with mock.patch(
            "services.shogou_prompt_designer.load_shogou_instruction",
            return_value="INSTRUCTION_BODY",
        ), mock.patch(
            "services.shogou_prompt_designer.append_memory_to_system_prompt",
            return_value="INSTRUCTION_BODY",
        ), mock.patch(
            "services.shogou_prompt_designer.build_image_memory_prompt_context",
            return_value="画像参考Memory #7 (明示指定のため全タグ): silver hair",
        ) as image_memory_mock, mock.patch(
            "services.shogou_prompt_designer._post_chat_completion",
            return_value=VALID_JSON,
        ) as post_mock:
            generate_shogou_prompt_result("この子で", user_id=1, reference_id=7)

        image_memory_mock.assert_called_once_with(1, "この子で", 7)
        sent_system_prompt = post_mock.call_args.args[0]
        self.assertIn("画像参考Memory #7", sent_system_prompt)
        # The auxiliary memory context must not appear before the instruction contract.
        self.assertTrue(sent_system_prompt.startswith("INSTRUCTION_BODY"))

    def test_reference_id_omitted_reuses_existing_automatic_selection(self):
        with mock.patch(
            "services.shogou_prompt_designer.load_shogou_instruction",
            return_value="INSTRUCTION_BODY",
        ), mock.patch(
            "services.shogou_prompt_designer.append_memory_to_system_prompt",
            return_value="INSTRUCTION_BODY",
        ), mock.patch(
            "services.shogou_prompt_designer.build_image_memory_prompt_context",
            return_value="",
        ) as image_memory_mock, mock.patch(
            "services.shogou_prompt_designer._post_chat_completion",
            return_value=VALID_JSON,
        ):
            generate_shogou_prompt_result("犬耳の子", user_id=1, reference_id=None)

        # reference_id=None must be forwarded unchanged so the existing
        # automatic relevance selection in build_image_memory_prompt_context
        # (unmodified by this integration) decides which tags to use.
        image_memory_mock.assert_called_once_with(1, "犬耳の子", None)


class GenerateShogouCompositionVariationTestCase(unittest.TestCase):
    """Phase 1C "change_composition" regeneration."""

    def test_parent_fields_and_instruction_are_sent_to_lm_studio(self):
        with mock.patch(
            "services.shogou_prompt_designer.load_shogou_instruction",
            return_value="INSTRUCTION_BODY",
        ), mock.patch(
            "services.shogou_prompt_designer.append_memory_to_system_prompt",
            return_value="INSTRUCTION_BODY",
        ), mock.patch(
            "services.shogou_prompt_designer.build_image_memory_prompt_context",
            return_value="",
        ), mock.patch(
            "services.shogou_prompt_designer._post_chat_completion",
            return_value=VALID_JSON,
        ) as post_mock:
            result = generate_shogou_composition_variation(
                "雨の日の喫茶店で読書する女の子",
                "1girl, cafe, reading, rain",
                "lowres, bad anatomy",
                "喫茶店で読書する猫耳の女の子",
                ("背景を補完",),
                user_id=42,
                reference_id=None,
            )

        self.assertFalse(result.blocked)
        sent_system_prompt = post_mock.call_args.args[0]
        sent_user_input = post_mock.call_args.args[1]
        # The parent's request/positive/negative/summary/assumptions must
        # all reach LM Studio, so it has the same generation to vary from.
        self.assertIn("雨の日の喫茶店で読書する女の子", sent_system_prompt)
        self.assertIn("1girl, cafe, reading, rain", sent_system_prompt)
        self.assertIn("lowres, bad anatomy", sent_system_prompt)
        self.assertIn("喫茶店で読書する猫耳の女の子", sent_system_prompt)
        self.assertIn("背景を補完", sent_system_prompt)
        # Explicit instruction: keep character/clothing/mood, vary only
        # composition/camera/pose/layout.
        self.assertIn("キャラクター", sent_system_prompt)
        self.assertIn("衣装", sent_system_prompt)
        self.assertIn("雰囲気", sent_system_prompt)
        self.assertIn("構図", sent_system_prompt)
        self.assertIn("カメラ角度", sent_system_prompt)
        self.assertIn("ポーズ", sent_system_prompt)
        # Phase 1E: composition-only regeneration must not silently reset the
        # parent's color/brightness/light direction back to some default.
        self.assertIn("色彩", sent_system_prompt)
        self.assertIn("明るさ", sent_system_prompt)
        self.assertEqual(sent_user_input, "雨の日の喫茶店で読書する女の子")

    def test_instruction_is_appended_after_reference_context(self):
        with mock.patch(
            "services.shogou_prompt_designer.load_shogou_instruction",
            return_value="INSTRUCTION_BODY",
        ), mock.patch(
            "services.shogou_prompt_designer.append_memory_to_system_prompt",
            return_value="INSTRUCTION_BODY",
        ), mock.patch(
            "services.shogou_prompt_designer.build_image_memory_prompt_context",
            return_value="画像参考Memory #7 (明示指定のため全タグ): silver hair",
        ) as image_memory_mock, mock.patch(
            "services.shogou_prompt_designer._post_chat_completion",
            return_value=VALID_JSON,
        ) as post_mock:
            generate_shogou_composition_variation(
                "この子で", "1girl", "lowres", "要約", (), user_id=1, reference_id=7
            )

        image_memory_mock.assert_called_once_with(1, "この子で", 7)
        sent_system_prompt = post_mock.call_args.args[0]
        self.assertLess(
            sent_system_prompt.index("画像参考Memory #7"),
            sent_system_prompt.index("構図"),
        )

    def test_does_not_call_the_plain_prompt_result_path(self):
        with mock.patch(
            "services.shogou_prompt_designer.load_shogou_instruction",
            return_value="INSTRUCTION_BODY",
        ), mock.patch(
            "services.shogou_prompt_designer.append_memory_to_system_prompt",
            return_value="INSTRUCTION_BODY",
        ), mock.patch(
            "services.shogou_prompt_designer.build_image_memory_prompt_context",
            return_value="",
        ), mock.patch(
            "services.shogou_prompt_designer._post_chat_completion",
            return_value=VALID_JSON,
        ) as post_mock, mock.patch(
            "services.shogou_prompt_designer.generate_shogou_prompt_result"
        ) as plain_mock:
            generate_shogou_composition_variation("元の依頼", "1girl", "lowres", "要約", (), user_id=1)

        plain_mock.assert_not_called()
        post_mock.assert_called_once()

    def test_blocked_json_from_lm_studio_is_surfaced_as_blocked(self):
        with mock.patch(
            "services.shogou_prompt_designer.load_shogou_instruction",
            return_value="INSTRUCTION_BODY",
        ), mock.patch(
            "services.shogou_prompt_designer.append_memory_to_system_prompt",
            return_value="INSTRUCTION_BODY",
        ), mock.patch(
            "services.shogou_prompt_designer.build_image_memory_prompt_context",
            return_value="",
        ), mock.patch(
            "services.shogou_prompt_designer._post_chat_completion",
            return_value="not json at all",
        ):
            result = generate_shogou_composition_variation("元の依頼", "1girl", "lowres", "要約", (), user_id=1)

        self.assertTrue(result.blocked)


class InstructionContentTestCase(unittest.TestCase):
    """Phase 1E: shogou_chan_prompt_designer_instruction.md gained a
    dedicated color/lighting decision procedure to counteract a cool/dark
    bias, without touching the JSON output contract or tag ordering rules
    it shares with every other Phase. These tests read the real file
    (not a mocked instruction body) so a future edit that silently drops
    one of these rules, or breaks the existing contract, fails loudly.
    """

    @classmethod
    def setUpClass(cls):
        cls.instruction = load_shogou_instruction()

    def test_json_output_contract_is_unchanged(self):
        # Regression guard (test case 11): the color/lighting section must
        # not have disturbed the pre-existing output contract.
        self.assertIn('`positive` `negative` `summary` `assumptions`', self.instruction)
        self.assertIn("この順で並べる", self.instruction)
        self.assertIn("JSON の前後に、前置き・挨拶・説明・見出し・Markdown", self.instruction)

    def test_tag_ordering_rule_is_unchanged(self):
        self.assertIn(
            "人数(1girl/1boy/1other) → キャラクター名 → 作品名 → rating(safe/sensitive 等) → "
            "年齢感/顔/髪/衣装/表情/ポーズ/構図/背景/光 → 品質タグ",
            self.instruction,
        )
        self.assertIn("品質タグは必ず末尾に置く", self.instruction)

    def test_rating_tag_rules_are_unchanged(self):
        self.assertIn("既定は `safe`", self.instruction)
        self.assertIn("既知キャラは**キャラクター名と作品名を優先して出力**する", self.instruction)

    def test_priority_order_for_color_and_light_is_present(self):
        for phrase in (
            "ユーザーの明示指定（時間帯・天候・光源・色・雰囲気）を最優先で採用する",
            "既知キャラクターがいる場合、公式のイメージカラー・配色を尊重する",
            "髪色・瞳色・衣装の色から自然に調和する方向性を選ぶ",
            "シーンの種類",
            "ニュートラルで自然な明るさを既定にする",
        ):
            self.assertIn(phrase, self.instruction)

    def test_brightness_and_color_temperature_are_judged_separately(self):
        self.assertIn("明るさと色温度を**別々に**判断する", self.instruction)

    def test_default_is_neutral_not_cool_or_dark(self):
        self.assertIn("neutral lighting", self.instruction)
        self.assertIn("natural lighting", self.instruction)
        self.assertIn("balanced color palette", self.instruction)
        self.assertIn("寒色・暗色をデフォルトにしない", self.instruction)
        self.assertIn("暗くしすぎない", self.instruction)

    def test_backlight_is_bounded_to_subtle_by_default(self):
        self.assertIn("subtle backlight", self.instruction)
        self.assertIn("soft rim light", self.instruction)
        self.assertIn("strong backlight", self.instruction)
        self.assertIn("dramatic backlighting", self.instruction)
        self.assertIn("intense rim light", self.instruction)
        self.assertIn("機械的に追加しない", self.instruction)

    def test_screen_wide_cool_tint_is_not_a_default(self):
        self.assertIn("blue ambient light", self.instruction)
        self.assertIn("cyan lighting", self.instruction)
        self.assertIn("cool color grading", self.instruction)
        self.assertIn("デフォルトで追加しない", self.instruction)
        self.assertIn("肌が青白く見えるほど寒色照明を強くしない", self.instruction)

    def test_hair_color_guidance_covers_warm_and_cool_and_black(self):
        self.assertIn("金髪・赤髪・橙髪・茶髪・暖色衣装", self.instruction)
        self.assertIn("warm〜neutralを優先する", self.instruction)
        self.assertIn("青髪・紫髪・銀髪・白髪・寒色衣装", self.instruction)
        # Black hair must not be an automatic trigger for night/cool defaults.
        self.assertIn("黒髪だけを理由に自動で夜・寒色にしない", self.instruction)

    def test_mixed_hair_and_eye_color_guidance_is_present(self):
        self.assertIn("金髪+青目のような混色", self.instruction)
        self.assertIn("主光はneutral〜warmにし", self.instruction)

    def test_explicit_user_color_or_light_request_overrides_defaults(self):
        self.assertIn("青い照明", self.instruction)
        self.assertIn("月夜", self.instruction)
        self.assertIn("雪景色", self.instruction)
        self.assertIn("その指定を最優先する", self.instruction)

    def test_scene_examples_cover_school_cafe_beach_sunset_night_fantasy(self):
        for phrase in (
            "学園・日常",
            "カフェ・部屋",
            "海・青空",
            "夕暮れ",
            "夜景",
            "ファンタジー",
        ):
            self.assertIn(phrase, self.instruction)
        self.assertIn("色エフェクトでキャラクターを埋没させない", self.instruction)

    def test_negative_color_tags_are_conditional_not_fixed_in_the_standard_template(self):
        standard_template_match = re.search(r"標準テンプレ: `([^`]+)`", self.instruction)
        self.assertIsNotNone(standard_template_match, "expected the fixed negative template line")
        standard_template = standard_template_match.group(1)
        for conditional_tag in (
            "excessive blue tint",
            "overly cool color palette",
            "blue skin tone",
            "underexposed",
            "overly dark",
            "harsh backlight",
            "blown highlights",
            "excessive bloom",
            "overly dramatic lighting",
        ):
            self.assertIn(conditional_tag, self.instruction)
            self.assertNotIn(conditional_tag, standard_template)
        self.assertIn("該当する場合のみ", self.instruction)

    def test_few_shot_examples_are_still_valid_single_json_objects(self):
        outputs = re.findall(r"^出力:\n(\{.*\})$", self.instruction, flags=re.MULTILINE)
        self.assertGreaterEqual(len(outputs), 3, "expected at least the three documented few-shot examples")
        for raw in outputs:
            payload = json.loads(raw)
            self.assertEqual(list(payload.keys()), ["positive", "negative", "summary", "assumptions"])

    def test_neutral_daylight_few_shot_example_does_not_default_to_cool_or_dark(self):
        # Anchors the "weak/no color cue -> neutral, not cool/dark" default
        # (test case 1: blonde girl, daytime school) as an example the model
        # can pattern-match against, not just prose.
        outputs = re.findall(r"^出力:\n(\{.*\})$", self.instruction, flags=re.MULTILINE)
        daylight_examples = [json.loads(raw) for raw in outputs if "blonde hair" in raw]
        self.assertTrue(daylight_examples, "expected a blonde-hair daytime example")
        positive = daylight_examples[0]["positive"]
        for absent_cool_or_dark_tag in ("cool lighting", "blue tone", "night", "moody lighting", "dark"):
            self.assertNotIn(absent_cool_or_dark_tag, positive)
        self.assertIn("bright", positive)


if __name__ == "__main__":
    unittest.main()
