import unittest
from unittest import mock

from services.sd_control import DEFAULT_NEGATIVE_PROMPT, SdGenerationOptions, validate_options


class SdGenerationOptionsTestCase(unittest.TestCase):
    def test_existing_mode_strips_prompts_and_uses_default_negative(self):
        with mock.patch.dict("os.environ", {"SD_DEFAULT_NEGATIVE_PROMPT": " configured default "}):
            result = validate_options(SdGenerationOptions(prompt="  positive  "))

        self.assertEqual(result.prompt, "positive")
        self.assertEqual(result.negative_prompt, "configured default")
        self.assertFalse(result.preserve_prompt_text)

    def test_raw_mode_preserves_positive_and_negative_exactly(self):
        result = validate_options(
            SdGenerationOptions(
                prompt="  positive, detailed  ",
                negative_prompt="  low quality, blurry  ",
                preserve_prompt_text=True,
            )
        )

        self.assertEqual(result.prompt, "  positive, detailed  ")
        self.assertEqual(result.negative_prompt, "  low quality, blurry  ")
        self.assertTrue(result.preserve_prompt_text)

    def test_raw_mode_allows_explicit_empty_negative(self):
        result = validate_options(
            SdGenerationOptions(
                prompt="positive",
                negative_prompt="",
                preserve_prompt_text=True,
            )
        )

        self.assertEqual(result.negative_prompt, "")

    def test_raw_mode_still_rejects_whitespace_only_positive(self):
        with self.assertRaisesRegex(ValueError, "prompt is empty"):
            validate_options(
                SdGenerationOptions(
                    prompt="   ",
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    preserve_prompt_text=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
