import json
import unittest

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
