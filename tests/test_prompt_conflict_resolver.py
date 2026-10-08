from __future__ import annotations

import unittest

from services.prompt_conflict_resolver import (
    PromptTagRecord,
    classify_text_constraints,
    detect_people_semantics,
    resolve_prompt_tag_conflicts,
)


class PromptConflictResolverTests(unittest.TestCase):
    def test_auto_solo_is_suppressed_by_explicit_multi_person_interaction(self):
        result = resolve_prompt_tag_conflicts(
            "2girls, hugging, solo",
            explicit_tags=("2girls", "hugging"),
        )
        self.assertIn("2girls", result.positive)
        self.assertIn("hugging", result.positive)
        self.assertNotIn("solo", result.positive)
        self.assertEqual("solo", result.suppressed_tags[0].tag)

    def test_explicit_solo_and_multi_person_are_retained_with_warning(self):
        result = resolve_prompt_tag_conflicts(
            "solo, 2girls",
            explicit_tags=("solo", "2girls"),
        )
        self.assertEqual("solo, 2girls", result.positive)
        self.assertTrue(any(conflict.category == "people_count" for conflict in result.prompt_conflicts))

    def test_complementary_singular_people_are_not_a_count_conflict(self):
        result = resolve_prompt_tag_conflicts(
            "1girl, 1boy, holding hands",
            explicit_tags=("1girl", "1boy", "holding hands"),
        )
        self.assertEqual("1girl, 1boy, holding hands", result.positive)
        self.assertFalse(any(conflict.category == "people_count" for conflict in result.prompt_conflicts))

    def test_expression_pose_framing_and_camera_locks(self):
        cases = (
            ("smile, serious", ("smile",), "serious"),
            ("running, sitting", ("running",), "sitting"),
            ("full body, upper body", ("full body",), "upper body"),
            ("from side, three-quarter view", ("from side",), "three-quarter view"),
        )
        for positive, explicit, suppressed in cases:
            with self.subTest(suppressed=suppressed):
                result = resolve_prompt_tag_conflicts(positive, explicit_tags=explicit)
                self.assertNotIn(suppressed, result.positive)
                self.assertTrue(any(item.tag == suppressed for item in result.suppressed_tags))

    def test_request_text_is_a_user_constraint_without_llm_call(self):
        result = resolve_prompt_tag_conflicts(
            "smile, serious, solo",
            request_text="2girls hugging, smile",
        )
        self.assertIn("smile", result.positive)
        self.assertNotIn("serious", result.positive)
        self.assertNotIn("solo", result.positive)
        self.assertIn("expression", result.locked_categories)

    def test_base_provenance_and_variation_source_are_preserved(self):
        result = resolve_prompt_tag_conflicts(
            "smile, serious, upper body",
            explicit_tags=("smile",),
            variation_tags=("serious", "upper body"),
            base_provenance=(PromptTagRecord("smile", "user_explicit", "expression"),),
        )
        self.assertEqual("smile, upper body", result.positive)
        sources = {item.tag: item.source for item in result.provenance}
        self.assertEqual("user_explicit", sources["smile"])
        self.assertEqual("variation_auto", sources["upper body"])

    def test_non_conflicting_automatic_tags_remain(self):
        result = resolve_prompt_tag_conflicts(
            "smile, portrait, masterpiece",
            explicit_tags=("smile",),
        )
        self.assertEqual("smile, portrait, masterpiece", result.positive)
        self.assertFalse(result.suppressed_tags)

    def test_text_classifier_uses_existing_taxonomy(self):
        constraints = classify_text_constraints("full body, from side, running, looking at viewer")
        self.assertIn("framing", constraints)
        self.assertIn("camera_horizontal", constraints)
        self.assertIn("basic_pose", constraints)
        self.assertIn("gaze", constraints)

    def test_people_semantics_detects_multi_person_and_mixed_sex_signals(self):
        clothed_sex = detect_people_semantics(
            ("1girl", "clothed_sex"),
            explicit_values=("1girl", "clothed_sex"),
            context_text="clothed_sex",
        )
        self.assertTrue(clothed_sex.requires_multiple_people)
        self.assertTrue(clothed_sex.visible_partner_preferred)
        self.assertEqual(("1girl",), clothed_sex.explicit_people_count)
        self.assertEqual(("clothed_sex",), clothed_sex.source_interactions)
        self.assertFalse(clothed_sex.implies_mixed_sex)

        hetero = detect_people_semantics(("hetero",), explicit_values=("hetero",))
        self.assertTrue(hetero.requires_multiple_people)
        self.assertTrue(hetero.implies_mixed_sex)
        self.assertEqual(("hetero",), hetero.source_interactions)

    def test_late_clothed_sex_output_suppresses_auto_solo(self):
        result = resolve_prompt_tag_conflicts(
            "1girl, solo, clothed_sex, upper body",
            request_text="clothed_sex",
        )
        self.assertNotIn("solo", result.positive)
        self.assertTrue(result.people_semantics.requires_multiple_people)
        suppressed = next(item for item in result.suppressed_tags if item.tag == "solo")
        self.assertEqual("conflicts_with_multi_person_interaction", suppressed.reason)
        self.assertEqual("clothed_sex", suppressed.source_interaction)
        self.assertEqual("suppressed", suppressed.status)

    def test_gender_unknown_carrying_suppresses_solo_without_inventing_counts(self):
        result = resolve_prompt_tag_conflicts(
            "solo, carrying",
            request_text="carrying",
        )
        self.assertNotIn("solo", result.positive)
        self.assertTrue(result.people_semantics.requires_multiple_people)
        self.assertFalse(result.people_semantics.implies_mixed_sex)
        self.assertEqual((), result.people_semantics.explicit_people_count)

    def test_explicit_solo_and_clothed_sex_are_kept_without_suppression(self):
        result = resolve_prompt_tag_conflicts(
            "solo, clothed_sex",
            explicit_tags=("solo", "clothed_sex"),
        )
        self.assertEqual("solo, clothed_sex", result.positive)
        self.assertFalse(any(item.tag == "solo" for item in result.suppressed_tags))


if __name__ == "__main__":
    unittest.main()
