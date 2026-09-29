import unittest

from services.character_appearance_filter import enforce_confirmed_character_appearance


class ConfirmedCharacterAppearanceTests(unittest.TestCase):
    def test_missing_confirmed_tags_are_inserted_after_identity_tags(self):
        result = enforce_confirmed_character_appearance(
            "1girl, castorice, honkai: star rail, smile",
            ("purple hair", "purple eyes"),
            identity_tags=("castorice", "honkai: star rail"),
        )

        self.assertEqual(
            result.positive,
            "1girl, castorice, honkai: star rail, purple hair, purple eyes, smile",
        )
        self.assertEqual(result.removed_tags, ())

    def test_conflicting_llm_appearance_is_removed_for_confirmed_category(self):
        result = enforce_confirmed_character_appearance(
            "1girl, castorice, blue hair, green eyes, smile",
            ("purple hair", "purple eyes"),
            identity_tags=("castorice",),
        )

        self.assertIn("purple hair", result.positive)
        self.assertIn("purple eyes", result.positive)
        self.assertNotIn("blue hair", result.positive)
        self.assertNotIn("green eyes", result.positive)
        self.assertEqual(
            result.removed_tags,
            ("blue hair", "green eyes"),
        )

    def test_explicit_user_appearance_has_priority_over_confirmed_tag(self):
        result = enforce_confirmed_character_appearance(
            "1girl, castorice, blue hair, smile",
            ("purple hair",),
            user_input="キャストリス、青髪",
            explicit_prompt_tags=("blue hair",),
            identity_tags=("castorice",),
        )

        self.assertIn("blue hair", result.positive)
        self.assertNotIn("purple hair", result.positive)
        self.assertEqual(result.removed_tags, ())

    def test_unregistered_character_has_no_confirmed_appearance_injection(self):
        result = enforce_confirmed_character_appearance(
            "1girl, unknown character, blue hair",
            (),
            identity_tags=(),
        )

        self.assertEqual(result.positive, "1girl, unknown character, blue hair")
        self.assertEqual(result.removed_tags, ())


if __name__ == "__main__":
    unittest.main()
