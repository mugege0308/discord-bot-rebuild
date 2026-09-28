from __future__ import annotations

import unittest
from unittest.mock import patch

from services.direct_prompt_parser import (
    ENGLISH_DIRECT_INPUT_MODE,
    JAPANESE_AI_INPUT_MODE,
    PROMPT_IMPORT_INPUT_MODE,
    parse_direct_prompt,
)
from services.generation_service import build_generation_metadata_payload, build_one_image_generation_options
from services.prompt_import import QUALITY_SUFFIX, resolve_prompt_import, tokenize_prompt


STANDARD_NEGATIVE = "lowres, bad anatomy, bad hands, text"


class PromptImportTestCase(unittest.TestCase):
    def resolve(self, text: str, *, rating: str = "auto", is_adult: bool = False):
        with patch("services.prompt_import.get_negative_prompt", return_value=STANDARD_NEGATIVE):
            return resolve_prompt_import(text, requested_rating=rating, is_adult=is_adult)

    def test_example_canonicalizes_character_and_adds_safe_quality_suffix(self):
        result = self.resolve(
            "1girl, hu tao, genshin impact, arms crossed, from below, masterpiece, best quality"
        )
        self.assertEqual(
            result.positive,
            "1girl, hu tao (genshin impact), safe, arms crossed, from below, "
            "masterpiece, high score, great score, absurdres",
        )
        self.assertIn("arms crossed -> imported_prompt", result.tag_provenance)

    def test_unknown_imported_tags_are_preserved(self):
        result = self.resolve("1girl, blue hair, school uniform, classroom, looking at viewer")
        for tag in ("blue hair", "school uniform", "classroom", "looking at viewer"):
            self.assertIn(tag, result.positive)
        self.assertTrue(all(tag in result.imported_prompt_tags for tag in ("blue hair", "classroom")))

    def test_positive_negative_sections_are_separated(self):
        result = self.resolve("Positive:\n1girl, upper body\n\nNegative:\nlowres, bad hands")
        self.assertIn("1girl", result.positive)
        self.assertIn("upper body", result.positive)
        self.assertTrue(result.raw_imported_negative.startswith("lowres"))
        self.assertEqual(result.negative, "lowres, bad hands, bad anatomy, text")

    def test_tokenizer_preserves_nested_comma(self):
        self.assertEqual(
            tokenize_prompt("1girl, (tag:1.2), [detail, preserved], classroom"),
            ("1girl", "(tag:1.2)", "[detail, preserved]", "classroom"),
        )
        result = self.resolve("1girl, (tag:1.2), blue_hair")
        self.assertIn("(tag:1.2)", result.positive)
        weighted = self.resolve("1girl, (blue_hair:1.2)")
        self.assertIn("(blue hair:1.2)", weighted.positive)

    def test_character_canonicalization_does_not_add_signature_tags(self):
        result = self.resolve("1girl, hu tao, genshin impact")
        self.assertIn("hu tao (genshin impact)", result.positive)
        self.assertNotIn("red hair", result.positive)
        self.assertEqual(len(result.character_resolution), 1)

    def test_external_clothing_is_not_removed(self):
        result = self.resolve("1girl, hu tao, genshin impact, school uniform, blue jacket")
        self.assertIn("school uniform", result.positive)
        self.assertIn("blue jacket", result.positive)

    def test_verified_upper_body_framing_is_kept(self):
        result = self.resolve("1girl, upper body")
        self.assertIn("upper body", result.positive)
        self.assertFalse(result.conflict_removed_tags)

    def test_verified_framing_conflict_keeps_first_tag(self):
        result = self.resolve("1girl, upper body, full body")
        self.assertIn("upper body", result.positive)
        self.assertNotIn("full body", result.positive)
        self.assertIn("full body", result.conflict_removed_tags)

    def test_duplicate_removal_is_not_reported_as_framing_conflict(self):
        result = self.resolve("1girl, blue hair, blue_hair, upper body")
        self.assertNotIn("blue hair", result.conflict_removed_tags)
        self.assertTrue(any("duplicate removed" in item for item in result.canonicalized_tags))

    def test_dictionary_external_framing_is_preserved(self):
        result = self.resolve("1girl, close-up, thigh up, body focus, cropped")
        for tag in ("close-up", "thigh up", "body focus", "cropped"):
            self.assertIn(tag, result.positive)

    def test_standard_negative_is_used_when_external_negative_is_missing(self):
        result = self.resolve("1girl")
        self.assertEqual(result.negative, STANDARD_NEGATIVE)

    def test_external_negative_precedes_standard_and_is_deduplicated(self):
        result = self.resolve("Positive:\n1girl\nNegative:\nbad hands, custom negative")
        self.assertEqual(result.negative, "bad hands, custom negative, lowres, bad anatomy, text")

    def test_explicit_panel_rating_wins(self):
        result = self.resolve("1girl, safe", rating="explicit", is_adult=True)
        self.assertIn("explicit", result.positive)
        self.assertNotIn(", safe,", f", {result.positive},")

    def test_auto_uses_imported_rating_for_adult_profile(self):
        result = self.resolve("1girl, explicit", rating="auto", is_adult=True)
        self.assertIn("explicit", result.positive)

    def test_auto_without_imported_rating_uses_safe(self):
        result = self.resolve("1girl", rating="auto", is_adult=True)
        self.assertIn("safe", result.positive)

    def test_quality_suffix_is_exactly_once_at_end(self):
        result = self.resolve(
            "1girl, masterpiece, high score, masterpiece, best quality, absurdres"
        )
        self.assertTrue(result.positive.endswith(", ".join(QUALITY_SUFFIX)))
        for tag in QUALITY_SUFFIX:
            self.assertEqual(result.positive.split(", ").count(tag), 1)
        self.assertNotIn("best quality", result.positive)

    def test_existing_input_mode_parser_values_remain_distinct(self):
        self.assertEqual(JAPANESE_AI_INPUT_MODE, "ja_ai")
        self.assertEqual(ENGLISH_DIRECT_INPUT_MODE, "en_direct")
        self.assertEqual(PROMPT_IMPORT_INPUT_MODE, "prompt_import")
        direct = parse_direct_prompt("Positive:\n1girl\nNegative:\nlowres", "default")
        self.assertEqual(direct.positive, "1girl")
        self.assertEqual(direct.negative, "lowres")

    def test_generation_metadata_keeps_prompt_import_input_mode(self):
        payload = build_generation_metadata_payload(
            positive_prompt="1girl, safe",
            negative_prompt=STANDARD_NEGATIVE,
            model_key="animagine",
            model_family="sdxl",
            loras=(),
            extras={"input_mode": PROMPT_IMPORT_INPUT_MODE},
        )
        self.assertEqual(payload["input_mode"], PROMPT_IMPORT_INPUT_MODE)

    def test_final_generation_options_keep_prompt_without_lora_changes(self):
        result = self.resolve("1girl, upper body")
        options = build_one_image_generation_options(
            positive_prompt=result.positive,
            negative_prompt=result.negative,
            model_key="animagine",
            model_family="sdxl",
            loras=(),
            preserve_prompt_text=True,
        )
        self.assertEqual(options.prompt, result.positive)

    def test_reference_and_pose_options_are_passed_through(self):
        result = self.resolve("1girl, upper body")
        reference = b"reference"
        pose = b"pose"
        options = build_one_image_generation_options(
            positive_prompt=result.positive,
            negative_prompt=result.negative,
            model_key="animagine",
            model_family="sdxl",
            reference_image_bytes=reference,
            reference_mode="composition",
            pose_reference_bytes=pose,
            pose_strength=0.8,
            pose_end_percent=0.9,
            loras=(),
            preserve_prompt_text=True,
        )
        self.assertIs(options.reference_image_bytes, reference)
        self.assertIs(options.pose_reference_bytes, pose)
        self.assertEqual(options.pose_strength, 0.8)
        self.assertEqual(options.pose_end_percent, 0.9)


if __name__ == "__main__":
    unittest.main()
