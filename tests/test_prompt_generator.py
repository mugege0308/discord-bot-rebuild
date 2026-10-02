import json
import unittest
from unittest import mock

import requests

from services import prompt_generator


class FixedTemplateSafetyInstructionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instruction = prompt_generator.load_system_prompt()

    def test_allows_consensual_adult_fictional_content(self):
        self.assertIn(
            "Consensual erotic, nude, or NSFW content involving fictional adults is allowed.",
            self.instruction,
        )
        self.assertIn(
            "Do not refuse solely because the request is sexual, erotic, nude, or NSFW.",
            self.instruction,
        )
        self.assertIn(
            "provide a non-empty positive_prompt",
            self.instruction,
        )

    def test_keeps_required_refusal_conditions(self):
        for text in (
            "minors or childlike bodies",
            "non-consensual sexual content",
            "sexual content involving real identifiable people",
            "violent sexual exploitation",
        ):
            with self.subTest(text=text):
                self.assertIn(text, self.instruction)

    def test_safe_check_json_schema_is_unchanged(self):
        schema_text = self.instruction.split("Return only JSON with this exact shape:", 1)[1]
        schema_text = schema_text.split("Prompt style:", 1)[0].strip()
        schema = json.loads(schema_text)

        self.assertEqual(
            set(schema),
            {"safe_check", "positive_prompt", "negative_prompt", "notes"},
        )
        self.assertEqual(
            set(schema["safe_check"]),
            {"adult", "consent_ok", "blocked_elements"},
        )


class PromptDesignerNativeClientTestCase(unittest.TestCase):
    def test_native_request_uses_reasoning_off_and_prompt_settings(self):
        response = mock.Mock(status_code=200)
        response.raise_for_status.return_value = None
        response.json.return_value = {"output": [{"type": "message", "content": "{}"}]}
        with mock.patch.object(prompt_generator, "_get_lmstudio_api_url", return_value="http://127.0.0.1:1234/v1"), mock.patch.object(
            prompt_generator.requests, "post", return_value=response
        ) as post:
            result = prompt_generator.post_prompt_designer_completion(
                "system", "user", temperature=0.35, max_tokens=512
            )

        self.assertEqual(result, "{}")
        request_url, = post.call_args.args
        payload = post.call_args.kwargs["json"]
        self.assertEqual(request_url, "http://127.0.0.1:1234/api/v1/chat")
        self.assertEqual(payload["model"], prompt_generator._get_lmstudio_text_model())
        self.assertEqual(payload["reasoning"], "off")
        self.assertEqual(payload["temperature"], 0.35)
        self.assertEqual(payload["max_output_tokens"], 512)
        self.assertFalse(payload["store"])
        self.assertEqual(post.call_args.kwargs["timeout"], 30)

    def test_native_response_extracts_message_and_ignores_reasoning(self):
        response = mock.Mock(status_code=200)
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "output": [
                {"type": "reasoning", "content": "internal"},
                {"type": "message", "content": '{"positive":"portrait"}'},
            ]
        }
        with mock.patch.object(prompt_generator, "_get_lmstudio_api_url", return_value="http://127.0.0.1:1234/v1"), mock.patch.object(
            prompt_generator.requests, "post", return_value=response
        ):
            result = prompt_generator.post_prompt_designer_completion("system", "user")

        self.assertEqual(result, '{"positive":"portrait"}')
        self.assertNotIn("internal", result)

    def test_valid_native_text_is_not_retried_when_json_is_invalid(self):
        response = mock.Mock(status_code=200)
        response.raise_for_status.return_value = None
        response.json.return_value = {"output": [{"type": "message", "content": "not json"}]}
        with mock.patch.object(prompt_generator, "_get_lmstudio_api_url", return_value="http://127.0.0.1:1234/v1"), mock.patch.object(
            prompt_generator.requests, "post", return_value=response
        ) as post, mock.patch.object(prompt_generator, "_post_chat_completion") as fallback:
            result = prompt_generator.post_prompt_designer_completion("system", "user")

        self.assertEqual(result, "not json")
        post.assert_called_once()
        fallback.assert_not_called()

    def test_native_failure_falls_back_once_to_openai_compatible(self):
        native = mock.Mock(side_effect=requests.ConnectionError("unavailable"))
        with mock.patch.object(prompt_generator, "_get_lmstudio_api_url", return_value="http://127.0.0.1:1234/v1"), mock.patch.object(
            prompt_generator.requests, "post", native
        ), mock.patch.object(prompt_generator, "_post_chat_completion", return_value="fallback text") as fallback:
            result = prompt_generator.post_prompt_designer_completion(
                "system", "user", temperature=0.35, max_tokens=512
            )

        self.assertEqual(result, "fallback text")
        native.assert_called_once()
        fallback.assert_called_once_with("system", "user", temperature=0.35, max_tokens=512)
