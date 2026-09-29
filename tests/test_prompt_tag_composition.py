import unittest

from services.prompt_tag_resolver import apply_composition_preset_to_positive


class PromptTagCompositionTests(unittest.TestCase):
    def test_inserts_before_quality_suffix(self):
        result = apply_composition_preset_to_positive(
            "1girl, arms crossed, masterpiece, high score, great score, absurdres",
            "full_body",
        )
        self.assertEqual(
            "1girl, arms crossed, full body, masterpiece, high score, great score, absurdres",
            result.positive,
        )
        self.assertEqual("full body", result.composition_tag_applied)

    def test_removes_only_known_conflicting_framing(self):
        result = apply_composition_preset_to_positive(
            "1girl, upper body, close-up, masterpiece",
            "full_body",
        )
        self.assertNotIn("upper body", result.positive)
        self.assertIn("close-up", result.positive)
        self.assertEqual(("upper body",), result.conflict_removed_tags)

    def test_weighted_framing_is_recognized_without_breaking_unknown_tags(self):
        result = apply_composition_preset_to_positive(
            "1girl, (upper body:1.1), (unknown, tag:1.2), masterpiece",
            "full_body",
        )
        self.assertNotIn("upper body", result.positive)
        self.assertIn("(unknown, tag:1.2)", result.positive)
        self.assertEqual(("(upper body:1.1)",), result.conflict_removed_tags)

    def test_none_is_a_noop(self):
        positive = "1girl, close-up, masterpiece"
        result = apply_composition_preset_to_positive(positive, None)
        self.assertEqual(positive, result.positive)
        self.assertIsNone(result.composition_tag_applied)


if __name__ == "__main__":
    unittest.main()
