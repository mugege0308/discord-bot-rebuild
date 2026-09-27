import unittest

from services.prompt_positive_cleanup import cleanup_positive_prompt


class PromptPositiveCleanupTestCase(unittest.TestCase):
    def test_removes_numbers_and_prefers_canonical_expression_and_gesture(self):
        result = cleanup_positive_prompt(
            "1girl, kochiya sanae, 2, safe, smile, long_hair, "
            "masterpiece, smiley face, v",
            canonical_prompt_tags=("smile",),
            user_input="東風谷早苗が白髪になって笑顔でピース",
        )

        self.assertIn("smile", result.positive)
        self.assertIn("peace sign", result.positive)
        self.assertNotIn(", 2,", ", " + result.positive + ", ")
        self.assertNotIn("smiley face", result.positive)
        self.assertNotIn(", v", result.positive)
        self.assertIn("long_hair", result.positive)
        self.assertEqual(
            result.removed_tags,
            ("2", "smiley face", "v"),
        )

    def test_underscore_and_space_duplicates_are_collapsed(self):
        result = cleanup_positive_prompt(
            "safe, long_hair, long hair, smile",
            canonical_prompt_tags=("smile",),
        )

        self.assertEqual(result.positive, "safe, long_hair, smile")
        self.assertEqual(result.removed_tags, ("long hair",))
        self.assertEqual(
            result.removed_reasons,
            (("long hair", "duplicate_normalized_tag"),),
        )

    def test_without_authoritative_canonical_tag_semantic_aliases_are_kept(self):
        result = cleanup_positive_prompt("safe, smiley face, v", user_input="雰囲気")

        self.assertEqual(result.positive, "safe, smiley face, v")
        self.assertEqual(result.removed_tags, ())

    def test_composition_tags_are_not_treated_as_cleanup_aliases(self):
        result = cleanup_positive_prompt(
            "kochiya sanae, safe, from below, looking back, smile",
            canonical_prompt_tags=("from below", "looking back", "smile"),
            user_input="東風谷早苗が少し下から振り返って笑う",
        )

        self.assertEqual(
            result.positive,
            "kochiya sanae, safe, from below, looking back, smile",
        )
        self.assertEqual(result.removed_tags, ())


if __name__ == "__main__":
    unittest.main()
