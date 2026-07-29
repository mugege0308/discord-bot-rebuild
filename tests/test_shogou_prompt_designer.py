import json
import re
import unittest
from unittest import mock

from services.prompt_generator import PromptGeneratorError
from services.shogou_prompt_designer import (
    ShogouPromptResult,
    _strip_think_blocks,
    generate_shogou_composition_variation,
    generate_shogou_custom_instruction_variation,
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


class GenerateShogouCustomInstructionVariationTestCase(unittest.TestCase):
    """「指示を追加して再生成」: parent fields plus the user's free-text
    instruction must reach LM Studio, with an explicit "preserve everything
    else" instruction, and the function must not fall back to the plain or
    composition-only prompt paths."""

    def test_parent_fields_and_custom_instruction_are_sent_to_lm_studio(self):
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
            result = generate_shogou_custom_instruction_variation(
                "雨の日の喫茶店で読書する女の子",
                "1girl, cafe, reading, rain",
                "lowres, bad anatomy",
                "喫茶店で読書する猫耳の女の子",
                ("背景を補完",),
                "髪型をポニーテールに",
                user_id=42,
                reference_id=None,
            )

        self.assertFalse(result.blocked)
        sent_system_prompt = post_mock.call_args.args[0]
        sent_user_input = post_mock.call_args.args[1]
        self.assertIn("雨の日の喫茶店で読書する女の子", sent_system_prompt)
        self.assertIn("1girl, cafe, reading, rain", sent_system_prompt)
        self.assertIn("lowres, bad anatomy", sent_system_prompt)
        self.assertIn("喫茶店で読書する猫耳の女の子", sent_system_prompt)
        self.assertIn("背景を補完", sent_system_prompt)
        self.assertIn("髪型をポニーテールに", sent_system_prompt)
        # Explicit instruction: keep the original policy, vary only what the
        # additional instruction targets.
        self.assertIn("元の生成方針をできるだけ維持", sent_system_prompt)
        self.assertIn("色彩", sent_system_prompt)
        self.assertIn("構図", sent_system_prompt)
        self.assertIn("雰囲気", sent_system_prompt)
        self.assertIn("衣装", sent_system_prompt)
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
            generate_shogou_custom_instruction_variation(
                "この子で", "1girl", "lowres", "要約", (), "笑顔にして", user_id=1, reference_id=7
            )

        image_memory_mock.assert_called_once_with(1, "この子で", 7)
        sent_system_prompt = post_mock.call_args.args[0]
        self.assertLess(
            sent_system_prompt.index("画像参考Memory #7"),
            sent_system_prompt.index("今回の追加指示"),
        )

    def test_does_not_call_the_plain_or_composition_prompt_paths(self):
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
        ) as plain_mock, mock.patch(
            "services.shogou_prompt_designer.generate_shogou_composition_variation"
        ) as composition_mock:
            generate_shogou_custom_instruction_variation(
                "元の依頼", "1girl", "lowres", "要約", (), "笑顔にして", user_id=1
            )

        plain_mock.assert_not_called()
        composition_mock.assert_not_called()
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
            result = generate_shogou_custom_instruction_variation(
                "元の依頼", "1girl", "lowres", "要約", (), "笑顔にして", user_id=1
            )

        self.assertTrue(result.blocked)

    def test_empty_custom_instruction_raises_without_calling_lm_studio(self):
        with mock.patch(
            "services.shogou_prompt_designer._post_chat_completion"
        ) as post_mock:
            with self.assertRaises(PromptGeneratorError):
                generate_shogou_custom_instruction_variation(
                    "元の依頼", "1girl", "lowres", "要約", (), "   ", user_id=1
                )

        post_mock.assert_not_called()

    def test_custom_instruction_over_max_length_raises_without_calling_lm_studio(self):
        with mock.patch(
            "services.shogou_prompt_designer._post_chat_completion"
        ) as post_mock:
            with self.assertRaises(PromptGeneratorError):
                generate_shogou_custom_instruction_variation(
                    "元の依頼", "1girl", "lowres", "要約", (), "あ" * 1001, user_id=1
                )

        post_mock.assert_not_called()

    def test_custom_instruction_at_max_length_is_accepted(self):
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
            result = generate_shogou_custom_instruction_variation(
                "元の依頼", "1girl", "lowres", "要約", (), "あ" * 1000, user_id=1
            )

        self.assertFalse(result.blocked)
        post_mock.assert_called_once()


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


class AssumptionsScopeInstructionTestCase(unittest.TestCase):
    """Prompt Designer refinement: `assumptions` must record only
    missing-*information* completion (clothing / place / minimal time of day /
    count / age band / general scene), never color/lighting/staging (cool
    tones, blue tones, rim light, ...). Staging is decided by the Prompt
    Designer body (priority step 4) and lands in positive/negative tags, not
    in `assumptions`. These read the real instruction file so a future edit
    that drops a rule - or a few-shot example that regresses to writing
    staging into `assumptions` - fails loudly. The JSON output contract is
    unchanged (asserted by InstructionContentTestCase).
    """

    # Color/lighting/staging terms that must never appear inside an
    # `assumptions` entry. Deliberately excludes plain scene/time words that
    # ARE allowed as minimal completion (indoor, daylight, night, sunset).
    FORBIDDEN_STAGING_IN_ASSUMPTIONS = (
        "cool tones",
        "warm tones",
        "blue tones",
        "orange lighting",
        "cinematic lighting",
        "rim light",
        "backlight",
        "bloom",
        "dramatic lighting",
        "color grading",
        "moonlight",
    )

    @classmethod
    def setUpClass(cls):
        cls.instruction = load_shogou_instruction()
        cls.examples = [
            json.loads(raw)
            for raw in re.findall(r"^出力:\n(\{.*\})$", cls.instruction, flags=re.MULTILINE)
        ]

    def _example_with(self, needle: str) -> dict:
        matches = [example for example in self.examples if needle in example["positive"]]
        self.assertTrue(matches, f"expected a few-shot example whose positive contains {needle!r}")
        return matches[0]

    def test_explicit_priority_order_is_documented(self):
        # 1 user -> 2 character -> 3 missing-info completion -> 4 staging.
        priority_index = self.instruction.index("判断の優先順位（明文化・厳守）")
        for phrase in (
            "ユーザー指定",
            "キャラクター情報",
            "不足情報の補完",
            "色彩・ライティング・演出の決定",
        ):
            self.assertIn(phrase, self.instruction[priority_index:])
        self.assertIn("「演出」は「不足情報」ではない", self.instruction)

    def test_allowed_assumption_completions_are_listed(self):
        for phrase in ("服装", "場所", "時間帯（必要最低限）", "人数", "年齢層", "一般的なシーン情報"):
            self.assertIn(phrase, self.instruction)

    def test_forbidden_staging_terms_are_called_out_as_not_for_assumptions(self):
        for phrase in self.FORBIDDEN_STAGING_IN_ASSUMPTIONS:
            self.assertIn(phrase, self.instruction)
        self.assertIn("`assumptions` には**書かない**", self.instruction)

    def test_background_unspecified_allows_indoor_but_forbids_color_jump(self):
        self.assertIn("背景未指定 → `indoor`", self.instruction)
        self.assertIn("背景未指定 → `blue tones`", self.instruction)
        self.assertIn("飛躍は禁止", self.instruction)

    def test_time_of_day_minimal_completion_allowed_but_not_its_lighting(self):
        # Test case 3 (no daytime specified -> `daylight` minimal completion
        # allowed) and test case 4 (`moonlight` staging must not go into
        # `assumptions`) live in the same rule sentence.
        self.assertIn("daylight", self.instruction)
        self.assertIn("night", self.instruction)
        self.assertIn("moonlight", self.instruction)
        self.assertIn("最低限の補完は `assumptions` に書いてよい", self.instruction)

    def test_internal_steps_keep_staging_out_of_assumptions(self):
        self.assertIn(
            "色彩・ライティング・演出は「不足情報」ではないので、ここでは補完せず `assumptions` にも書かない",
            self.instruction,
        )
        self.assertIn(
            "ここで決めた色彩・ライティング・演出は positive/negative のタグにのみ反映し、`assumptions` には書かない",
            self.instruction,
        )

    def test_no_few_shot_example_puts_staging_into_assumptions(self):
        # The strong regression guard: whatever the prose says, none of the
        # shipped few-shot examples may demonstrate staging inside assumptions.
        self.assertGreaterEqual(len(self.examples), 5, "expected the documented few-shot examples")
        for example in self.examples:
            assumptions_text = " ".join(example["assumptions"]).lower()
            for forbidden in self.FORBIDDEN_STAGING_IN_ASSUMPTIONS:
                self.assertNotIn(
                    forbidden,
                    assumptions_text,
                    f"staging term {forbidden!r} must not appear in assumptions: {example['assumptions']}",
                )

    def test_gym_ponytail_example_keeps_assumptions_free_of_cool_or_blue_tones(self):
        # Reported case: 黒髪ポニーテール・体操服. No color was requested, so
        # assumptions must not add cool/blue tones, and the picture stays
        # neutral rather than defaulting to a cool tint.
        example = self._example_with("gym uniform")
        assumptions_text = " ".join(example["assumptions"]).lower()
        for banned in ("cool tones", "blue tones", "cool", "blue"):
            self.assertNotIn(banned, assumptions_text)
        # Background was unspecified -> indoor completion IS allowed here.
        self.assertIn("indoor", assumptions_text)
        # Neutral default in the tags, not a cool/blue default.
        positive = example["positive"].lower()
        self.assertIn("neutral lighting", positive)
        for banned in ("cool lighting", "blue tones", "cool tones"):
            self.assertNotIn(banned, positive)

    def test_night_example_keeps_night_but_moves_moonlight_out_of_assumptions(self):
        # Night was explicitly requested: `night` may be recorded as a
        # time-of-day completion, but `moonlight` (staging) belongs in the
        # tags only, not in assumptions.
        example = self._example_with("night")
        assumptions_text = " ".join(example["assumptions"]).lower()
        self.assertIn("night", assumptions_text)
        self.assertNotIn("moonlight", assumptions_text)
        # Staging still lands in the positive tags (body decision).
        self.assertIn("moonlight", example["positive"].lower())


class ExistingCharacterPriorityInstructionTestCase(unittest.TestCase):
    """Reported failure: 「東方の古明地こいしが神社にいる」 was summarized correctly
    (character recognized) but the Positive tags replaced the character with
    generic descriptors (`ancient spirit`, `long black hair`, `red eyes`,
    `traditional clothing`) and added unrequested `cool colors` / `blue tones`.
    The instruction file must state, and demonstrate via few-shot, that a
    recognized existing character's name/series must never be dropped or
    generalized away, that unknown appearance details must not be fabricated,
    and that this is prioritized over background/color staging. These read
    the real instruction file so a future edit that silently weakens the
    rule fails loudly.
    """

    @classmethod
    def setUpClass(cls):
        cls.instruction = load_shogou_instruction()

    def test_existing_character_section_is_present(self):
        self.assertIn("既存キャラクターの扱い（最優先で厳守）", self.instruction)
        self.assertIn("キャラクター再現を背景・色彩演出・雰囲気より優先する", self.instruction)

    def test_character_name_must_not_be_generalized_away(self):
        self.assertIn("一般的な特徴語", self.instruction)
        self.assertIn("固有名を消して雰囲気語へ変換することは禁止", self.instruction)
        self.assertIn("そのキャラクター名を出力から絶対に落とさない", self.instruction)

    def test_unknown_appearance_details_must_not_be_fabricated(self):
        self.assertIn("分からない外見情報は無理に埋めず、捏造しない", self.instruction)
        self.assertIn("キャラ辞書や外部検索は使わない", self.instruction)

    def test_scene_completion_still_allowed_for_existing_characters(self):
        self.assertIn(
            "ポーズ・構図・背景・時間帯・人数・カメラ距離・視線など「場面情報」は通常どおり補完してよい",
            self.instruction,
        )

    def test_existing_character_positive_order_is_documented(self):
        section = self.instruction[self.instruction.index("既存キャラクター時の `positive` 順序") :]
        for phrase in (
            "1. character name",
            "2. series name",
            "3. 人数・rating",
            "4. キャラクター固有情報",
            "9. 品質タグ",
        ):
            self.assertIn(phrase, section)

    def test_original_character_case_is_explicitly_exempted(self):
        self.assertIn("オリジナルキャラクター", self.instruction)
        self.assertIn("この既存キャラクタールールを適用せず", self.instruction)

    def test_pose_breakdown_guidance_is_present(self):
        for phrase in (
            "facing viewer",
            "looking at viewer",
            "one hand raised",
            "hands at sides",
            "weight on one leg",
            "upper body turned",
        ):
            self.assertIn(phrase, self.instruction)
        self.assertIn("ControlNetやOpenPose用の座標・骨格指定は行わない", self.instruction)


class ExistingCharacterAssumptionsRestrictionTestCase(unittest.TestCase):
    """`assumptions` must not carry a hair/eye/clothing/species/color guess for
    an existing character - only genuinely missing scene information
    (background/time/camera distance/pose/count).
    """

    @classmethod
    def setUpClass(cls):
        cls.instruction = load_shogou_instruction()

    def test_assumptions_restriction_for_existing_characters_is_documented(self):
        self.assertIn(
            "キャラクターの髪色・瞳色・衣装・種族・設定を推定した内容は `assumptions` に書かない",
            self.instruction,
        )

    def test_color_and_wardrobe_alteration_is_bounded_for_existing_characters(self):
        self.assertIn("勝手な和風化・洋風化、勝手な衣装変更", self.instruction)
        self.assertIn("キャラクター固有色を壊す全体色調", self.instruction)


class KomeijiKoishiFewShotRegressionTestCase(unittest.TestCase):
    """Anchors the reported failure case as a concrete few-shot example so a
    future edit that regresses to replacing a recognized character with
    generic descriptors, or adds unrequested cool/blue staging, fails loudly.
    Deliberately checks for substrings/keywords rather than full-string
    equality so minor tag-order or wording drift doesn't make this test
    unnecessarily brittle.
    """

    @classmethod
    def setUpClass(cls):
        cls.instruction = load_shogou_instruction()
        outputs = re.findall(r"^出力:\n(\{.*\})$", cls.instruction, flags=re.MULTILINE)
        examples = [json.loads(raw) for raw in outputs]
        matches = [example for example in examples if "koishi" in example["positive"].lower()]
        assert matches, "expected a Komeiji Koishi few-shot example"
        cls.example = matches[0]

    def test_character_name_and_series_are_present_in_positive(self):
        positive = self.example["positive"].lower()
        self.assertIn("koishi", positive)
        self.assertIn("touhou", positive)

    def test_character_is_not_replaced_with_generic_descriptors(self):
        positive = self.example["positive"].lower()
        for forbidden in (
            "ancient spirit",
            "long black hair",
            "red eyes",
            "traditional clothing",
            "mysterious girl",
            "shrine maiden",
        ):
            self.assertNotIn(forbidden, positive)

    def test_no_unrequested_cool_or_blue_staging_is_added(self):
        positive = self.example["positive"].lower()
        for forbidden in ("cool colors", "cool tones", "blue tones", "blue ambient", "cyan"):
            self.assertNotIn(forbidden, positive)

    def test_assumptions_contain_no_appearance_or_color_guesses(self):
        assumptions_text = " ".join(self.example["assumptions"]).lower()
        for forbidden in (
            "髪",
            "瞳",
            "衣装",
            "和服",
            "black hair",
            "red eyes",
            "traditional clothing",
            "cool",
            "blue",
        ):
            self.assertNotIn(forbidden, assumptions_text)

    def test_summary_still_names_the_character(self):
        self.assertIn("こいし", self.example["summary"])

    def test_background_completion_is_still_allowed(self):
        # Scene information (shrine background) may still be completed and
        # recorded in assumptions - only appearance/color guesses are banned.
        assumptions_text = " ".join(self.example["assumptions"])
        self.assertIn("神社", assumptions_text)


class KnownCharacterContextInstructionTestCase(unittest.TestCase):
    """The instruction file must explain how to use a `Known character:`
    context block (injected by the character tag dictionary resolver, see
    services/character_tag_dictionary.py) when one is present in the system
    prompt: use the given tags verbatim, near the front of positive, without
    re-translating/splitting/generalizing them.
    """

    @classmethod
    def setUpClass(cls):
        cls.instruction = load_shogou_instruction()

    def test_known_character_block_format_is_documented(self):
        self.assertIn("Known character:", self.instruction)
        self.assertIn("- character tag:", self.instruction)
        self.assertIn("- series tag:", self.instruction)

    def test_verbatim_usage_instruction_is_present(self):
        self.assertIn("一字一句そのまま", self.instruction)
        self.assertIn("do not translate, split, abbreviate, replace, or generalize these tags", self.instruction)


class SparkleFewShotRegressionTestCase(unittest.TestCase):
    """Reported failure: 「崩壊：スターレイルの火花」was mistranslated to
    `spark, eclipse star rail` by the Prompt Designer, which an image model
    cannot recognize as the character. The character tag dictionary resolves
    this to the canonical `sparkle` / `honkai: star rail` tags; this few-shot
    example anchors the correct behavior so a future edit that regresses back
    to a generic/mistranslated rendering fails loudly.
    """

    @classmethod
    def setUpClass(cls):
        cls.instruction = load_shogou_instruction()
        outputs = re.findall(r"^出力:\n(\{.*\})$", cls.instruction, flags=re.MULTILINE)
        examples = [json.loads(raw) for raw in outputs]
        matches = [example for example in examples if "sparkle" in example["positive"].lower()]
        assert matches, "expected a Sparkle (Honkai: Star Rail) few-shot example"
        cls.example = matches[0]

    def test_character_and_series_tags_are_present_verbatim(self):
        positive = self.example["positive"].lower()
        self.assertIn("sparkle", positive)
        self.assertIn("honkai: star rail", positive)

    def test_character_is_not_mistranslated_or_generalized(self):
        positive = self.example["positive"].lower()
        for forbidden in ("eclipse star rail", "fireworks", "traditional clothing"):
            self.assertNotIn(forbidden, positive)
        # "spark" alone (not "sparkle") must not appear as a separate tag.
        tags = [tag.strip() for tag in positive.split(",")]
        self.assertNotIn("spark", tags)

    def test_assumptions_contain_no_appearance_or_color_guesses(self):
        assumptions_text = " ".join(self.example["assumptions"]).lower()
        for forbidden in ("髪", "瞳", "衣装", "cool", "blue"):
            self.assertNotIn(forbidden, assumptions_text)


class KinakoNanobanaFewShotRegressionTestCase(unittest.TestCase):
    """Reported failure: 「イナズマイレブンGOの菜花きなこ」was mistranslated to
    `kina ko` by the Prompt Designer. The character tag dictionary resolves
    this to `nanobana kinako` / `inazuma eleven go` (surname-first, matching
    Danbooru's real `nanobana_kinako` tag - the dictionary's original
    `kinako nanobana` order was corrected after external verification).
    """

    @classmethod
    def setUpClass(cls):
        cls.instruction = load_shogou_instruction()
        outputs = re.findall(r"^出力:\n(\{.*\})$", cls.instruction, flags=re.MULTILINE)
        examples = [json.loads(raw) for raw in outputs]
        matches = [example for example in examples if "nanobana kinako" in example["positive"].lower()]
        assert matches, "expected a Nanobana Kinako (Inazuma Eleven GO) few-shot example"
        cls.example = matches[0]

    def test_character_and_series_tags_are_present_verbatim(self):
        positive = self.example["positive"].lower()
        self.assertIn("nanobana kinako", positive)
        self.assertIn("inazuma eleven go", positive)

    def test_character_is_not_split_or_mistranslated(self):
        positive = self.example["positive"].lower()
        self.assertNotIn("kina ko", positive)


class KnownCharacterContextWiringTestCase(unittest.TestCase):
    """Character tag dictionary integration: when the request text resolves
    to a known character, the `Known character:` context block must reach LM
    Studio via the system prompt for all three generation entry points
    (plain / composition variation / custom instruction variation), and must
    NOT appear when nothing resolves (so it doesn't pollute unrelated
    requests). Custom instruction regeneration must resolve from
    parent_request_text so the character tags survive the "add an
    instruction" flow.
    """

    def test_plain_generation_includes_known_character_context_when_resolved(self):
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
            generate_shogou_prompt_result("崩壊:スターレイルの火花、笑顔", user_id=1, reference_id=None)

        sent_system_prompt = post_mock.call_args.args[0]
        self.assertIn("Known character:", sent_system_prompt)
        self.assertIn("character tag: sparkle", sent_system_prompt)
        self.assertIn("series tag: honkai: star rail", sent_system_prompt)

    def test_plain_generation_omits_known_character_context_when_nothing_resolves(self):
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
            generate_shogou_prompt_result("猫耳の女の子が公園にいる", user_id=1, reference_id=None)

        sent_system_prompt = post_mock.call_args.args[0]
        self.assertNotIn("Known character:", sent_system_prompt)

    def test_composition_variation_includes_known_character_context_when_resolved(self):
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
            generate_shogou_composition_variation(
                "イナズマイレブンGOの菜花きなこ",
                "1girl, nanobana kinako, inazuma eleven go",
                "lowres",
                "菜花きなこ",
                (),
                user_id=1,
            )

        sent_system_prompt = post_mock.call_args.args[0]
        self.assertIn("Known character:", sent_system_prompt)
        self.assertIn("character tag: nanobana kinako", sent_system_prompt)
        self.assertIn("series tag: inazuma eleven go", sent_system_prompt)

    def test_custom_instruction_variation_preserves_character_tags_from_parent_request_text(self):
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
            generate_shogou_custom_instruction_variation(
                "崩壊:スターレイルの火花",
                "1girl, sparkle, honkai: star rail",
                "lowres",
                "火花",
                (),
                "笑顔にして",
                user_id=1,
            )

        sent_system_prompt = post_mock.call_args.args[0]
        # The character tag context must survive alongside the custom
        # instruction, not be dropped or overwritten by it.
        self.assertIn("Known character:", sent_system_prompt)
        self.assertIn("character tag: sparkle", sent_system_prompt)
        self.assertIn("series tag: honkai: star rail", sent_system_prompt)
        self.assertIn("笑顔にして", sent_system_prompt)


if __name__ == "__main__":
    unittest.main()
