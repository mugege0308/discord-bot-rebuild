from __future__ import annotations

import unittest

from services.generation_api import _build_gallery_metadata
from services.generation_service import GenerationRequest, GenerationService
from services.shogou_prompt_designer import ShogouPromptResult


def _service(positive: str) -> GenerationService:
    return GenerationService(
        prompt_resolver=lambda *_args, **_kwargs: ShogouPromptResult(
            positive=positive,
            raw_llm_positive=positive,
            negative="bad quality",
        )
    )


class PromptConstraintSanitizerTests(unittest.TestCase):
    def test_holding_hands_safe_removes_conflicting_designer_tags(self):
        service = _service("1girl, 1boy, sex, missionary, lying on back, man on top, explicit")
        resolved = service.resolve_request(GenerationRequest(
            freeform_character="1girl, 1boy", interaction_id="holding_hands",
            rating="safe", multi_person_framing="full_body",
        ))
        positive = {token.strip() for token in resolved.prompt_result.positive.split(",")}
        self.assertEqual({"1girl", "1boy", "holding hands", "standing together", "full body"}, positive)
        self.assertFalse({"sex", "missionary", "lying on back", "man on top", "explicit"} & positive)
        removed = {item.tag: item.reason for item in resolved.suppressed_tags}
        self.assertEqual("rating_safe_conflict", removed["sex"])
        self.assertEqual("rating_safe_conflict", removed["missionary"])
        self.assertEqual("composition_conflict", removed["lying on back"])

    def test_safe_nonsexual_designer_tags_are_preserved(self):
        resolved = _service("1girl, 1boy, smiling, outdoors").resolve_request(GenerationRequest(
            freeform_character="1girl, 1boy", interaction_id="holding_hands", rating="safe"
        ))
        self.assertIn("smiling", resolved.prompt_result.positive)
        self.assertIn("outdoors", resolved.prompt_result.positive)

    def test_released_interaction_removes_competing_designer_interaction(self):
        resolved = _service("1girl, 1boy, hugging, smiling").resolve_request(GenerationRequest(
            freeform_character="1girl, 1boy", interaction_id="holding_hands", rating="safe"
        ))
        self.assertNotIn("hugging", resolved.prompt_result.positive)
        self.assertIn("holding hands", resolved.prompt_result.positive)
        suppressed = next(item for item in resolved.suppressed_tags if item.tag == "hugging")
        self.assertEqual("interaction_conflict", suppressed.reason)

    def test_no_interaction_keeps_existing_safe_behavior(self):
        resolved = _service("1girl, sex, explicit").resolve_request(
            GenerationRequest(freeform_character="1girl", rating="safe")
        )
        self.assertIn("sex", resolved.prompt_result.positive)
        self.assertIn("explicit", resolved.prompt_result.positive)

    def test_explicit_request_is_not_subject_to_safe_filter(self):
        resolved = _service("1girl, sex, explicit, missionary").resolve_request(
            GenerationRequest(freeform_character="1girl", model_key="animagine_adult", rating="explicit")
        )
        self.assertIn("sex", resolved.prompt_result.positive)
        self.assertIn("explicit", resolved.prompt_result.positive)
        self.assertIn("missionary", resolved.prompt_result.positive)

    def test_other_phase1_profiles_keep_their_profile_tags(self):
        for interaction_id, addition in (
            ("dancing_together", "dancing together"),
            ("hand_on_shoulder", "hand on partner's shoulder"),
            ("arm_around_shoulder", "arm around partner's shoulder"),
            ("leaning_on_partner", "leaning on partner"),
        ):
            with self.subTest(interaction_id=interaction_id):
                resolved = _service("1girl, 1boy, smiling").resolve_request(GenerationRequest(
                    freeform_character="1girl, 1boy", interaction_id=interaction_id, rating="safe"
                ))
                self.assertIn(addition, resolved.prompt_result.positive)
                self.assertIn("smiling", resolved.prompt_result.positive)

    def test_metadata_records_raw_prompt_and_constraint_removals(self):
        service = _service("1girl, 1boy, sex, missionary, lying on back")
        request = GenerationRequest(
            freeform_character="1girl, 1boy", interaction_id="holding_hands",
            rating="safe", multi_person_framing="full_body",
        )
        prepared = service.prepare_generation(request)
        metadata = _build_gallery_metadata("constraint-job", request, prepared)
        self.assertIn("sex", metadata["prompt_designer_raw_positive"])
        removed = {item["tag"] for item in metadata["prompt_constraint_removed_tags"]}
        self.assertTrue({"sex", "missionary", "lying on back"}.issubset(removed))
        self.assertIn("holding hands", metadata["positive"])
        self.assertNotIn("sex", metadata["positive"])


if __name__ == "__main__":
    unittest.main()
