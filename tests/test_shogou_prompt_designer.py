import unittest
from unittest import mock

from services.shogou_prompt_designer import (
    ShogouPromptResult,
    _strip_think_blocks,
    generate_shogou_prompt_result,
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


if __name__ == "__main__":
    unittest.main()
