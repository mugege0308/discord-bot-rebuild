import unittest

from services.prompt_generator import PromptGeneratorError
from services.shogou_prompt_designer import (
    extract_prompt_rating,
    validate_requested_prompt_rating,
)


class PromptRatingHelperTests(unittest.TestCase):
    def test_extracts_each_supported_rating(self):
        for rating in ("safe", "sensitive", "nsfw", "explicit"):
            with self.subTest(rating=rating):
                self.assertEqual(extract_prompt_rating(f"1girl, {rating}, smile"), rating)

    def test_normalizes_case_and_legacy_questionable_tag(self):
        self.assertEqual(extract_prompt_rating("1girl,  EXPLICIT  "), "explicit")
        self.assertEqual(extract_prompt_rating("1girl, questionable"), "nsfw")

    def test_uses_highest_rating_when_multiple_are_present(self):
        self.assertEqual(extract_prompt_rating("safe, sensitive, nsfw, explicit"), "explicit")

    def test_ignores_unknown_or_malformed_rating_text(self):
        self.assertIsNone(extract_prompt_rating("1girl, rating: explicit, explicitness"))
        self.assertIsNone(extract_prompt_rating(""))

    def test_missing_rating_is_none(self):
        self.assertIsNone(extract_prompt_rating("1girl, smile, masterpiece"))

    def test_requested_rating_validation_preserves_adult_gate(self):
        self.assertEqual(validate_requested_prompt_rating(None, is_adult=False), "auto")
        self.assertEqual(validate_requested_prompt_rating("SAFE", is_adult=False), "safe")
        self.assertEqual(validate_requested_prompt_rating("explicit", is_adult=True), "explicit")
        with self.assertRaises(PromptGeneratorError):
            validate_requested_prompt_rating("explicit", is_adult=False)
        with self.assertRaises(PromptGeneratorError):
            validate_requested_prompt_rating("unknown", is_adult=True)


if __name__ == "__main__":
    unittest.main()
