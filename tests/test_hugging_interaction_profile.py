from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from services.generation_api import GenerationApiError, _build_request, _build_gallery_metadata
from services.generation_service import GenerationRequest, GenerationService
from services.interaction_ipadapter import (
    preflight_ipadapter_profile,
    validate_hugging_compatibility,
)
from services.interaction_profiles import (
    InteractionChecksumMismatch,
    InteractionProfileError,
    InteractionProfileCompatibilityError,
    InteractionReferenceMissing,
    resolve_interaction_profile,
    resolve_production_interaction,
)


REFERENCE_ROOT = Path(r"C:\Users\PC_User\Documents\Codex\rp-validation-20261012\ComfyUI-Next")
EXPECTED_SHA = "285bf5ed20a9785ecbd7b3fd2f81d76d8e4f4a8692cbde5864ab98d89d2300c9"


def _request(**overrides: object) -> GenerationRequest:
    values: dict[str, object] = {
        "freeform_character": "1girl, 1boy",
        "interaction_id": "hugging",
        "width": 768,
        "height": 1024,
        "count": 1,
        "rating": "safe",
        "reference": None,
        "pose": None,
        "loras": (),
    }
    values.update(overrides)
    return GenerationRequest(**values)


class HuggingInteractionProfileTests(unittest.TestCase):
    def test_production_resolver_exposes_hugging_profile(self) -> None:
        profile = resolve_production_interaction("hugging")
        self.assertEqual("ipadapter_profile", profile.mode)
        self.assertEqual("production_candidate", profile.classification)
        self.assertTrue(profile.ready)
        self.assertTrue(profile.dedicated_positive)
        self.assertTrue(profile.dedicated_negative)

    def test_valid_request_is_accepted(self) -> None:
        request = _build_request(
            {
                "freeform_character": "1girl, 1boy",
                "interaction_id": "hugging",
                "size": "portrait",
                "count": 1,
                "rating": "safe",
            }
        )
        self.assertEqual((768, 1024, 1), (request.width, request.height, request.count))

    def test_batch_is_rejected(self) -> None:
        with self.assertRaises(GenerationApiError) as caught:
            _build_request({"freeform_character": "1girl, 1boy", "interaction_id": "hugging", "size": "portrait", "count": 2})
        self.assertEqual("interaction_batch_unsupported", caught.exception.code)

    def test_size_is_rejected(self) -> None:
        with self.assertRaises(GenerationApiError) as caught:
            _build_request({"freeform_character": "1girl, 1boy", "interaction_id": "hugging", "size": "square"})
        self.assertEqual("interaction_size_unsupported", caught.exception.code)

    def test_reference_pose_and_lora_conflicts_are_explicit(self) -> None:
        with self.assertRaises(InteractionProfileCompatibilityError) as caught:
            validate_hugging_compatibility(_request(reference={"mode": "composition"}))
        self.assertEqual("interaction_reference_conflict", caught.exception.code)
        with self.assertRaises(InteractionProfileCompatibilityError) as caught:
            validate_hugging_compatibility(_request(reference={"mode": "pose"}))
        self.assertEqual("interaction_pose_reference_conflict", caught.exception.code)
        with self.assertRaises(InteractionProfileCompatibilityError) as caught:
            validate_hugging_compatibility(_request(loras=("style",)))
        self.assertEqual("interaction_lora_unsupported", caught.exception.code)

    def test_reference_preflight_checksum_and_ipadapter_fields(self) -> None:
        profile = resolve_interaction_profile("hugging")
        with patch.dict(os.environ, {"INTERACTION_REFERENCE_ROOT": str(REFERENCE_ROOT)}, clear=False):
            preflight = preflight_ipadapter_profile(profile)
        self.assertEqual(EXPECTED_SHA, preflight.reference_sha256)
        self.assertEqual(64, len(preflight.reference_sha256))
        self.assertEqual("composition", preflight.ipadapter["weight_type"])
        self.assertEqual(0.5, preflight.ipadapter["weight"])
        self.assertGreater(len(preflight.image_bytes), 0)

    def test_missing_reference_is_explicit(self) -> None:
        profile = resolve_interaction_profile("hugging")
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(os.environ, {"INTERACTION_REFERENCE_ROOT": root}, clear=False):
                with self.assertRaises(InteractionReferenceMissing):
                    preflight_ipadapter_profile(profile)

    def test_checksum_mismatch_is_explicit(self) -> None:
        profile = resolve_interaction_profile("hugging")
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "input"
            target.mkdir()
            (target / "ipadapter_hugging_reference.png").write_bytes(b"wrong")
            with patch.dict(os.environ, {"INTERACTION_REFERENCE_ROOT": root}, clear=False):
                with self.assertRaises(InteractionChecksumMismatch):
                    preflight_ipadapter_profile(profile)

    def test_missing_ipadapter_fields_are_explicit(self) -> None:
        profile = resolve_interaction_profile("hugging")
        with self.assertRaises(InteractionProfileError) as caught:
            preflight_ipadapter_profile(replace(profile, ipadapter={}))
        self.assertIn("interaction_profile_schema_invalid", str(caught.exception))

    def test_prepare_uses_dedicated_prompt_and_ipadapter_config(self) -> None:
        service = GenerationService()
        with patch.dict(os.environ, {"INTERACTION_REFERENCE_ROOT": str(REFERENCE_ROOT)}, clear=False):
            prepared = service.prepare_generation(_request())
        self.assertIn("arms around each other", prepared.options.prompt)
        self.assertEqual("bad quality, worst quality, worst detail, sketch, censor", prepared.options.negative_prompt)
        self.assertEqual("composition", prepared.options.reference_mode)
        self.assertEqual(0.5, prepared.options.reference_strength)
        self.assertEqual("hugging", prepared.options.interaction_execution["profile_id"])
        self.assertEqual(EXPECTED_SHA, prepared.options.interaction_execution["reference_sha256"])
        self.assertEqual("composition", prepared.options.interaction_ipadapter["weight_type"])

    def test_metadata_contains_execution_without_absolute_path(self) -> None:
        service = GenerationService()
        request = _request()
        with patch.dict(os.environ, {"INTERACTION_REFERENCE_ROOT": str(REFERENCE_ROOT)}, clear=False):
            prepared = service.prepare_generation(request)
        metadata = _build_gallery_metadata("job-hug", request, prepared)
        self.assertEqual("hugging", metadata["interaction"]["profile_id"])
        self.assertEqual(EXPECTED_SHA, metadata["interaction_execution"]["reference_sha256"])
        self.assertNotIn("reference_path", str(metadata))
        self.assertNotIn("image_bytes", metadata)

    def test_prompt_only_and_no_interaction_are_unchanged(self) -> None:
        prompt_only = _build_request({"freeform_character": "couple", "interaction_id": "holding_hands"})
        self.assertEqual("holding_hands", prompt_only.interaction_id)
        ordinary = _build_request({"freeform_character": "girl", "interaction_id": None})
        self.assertIsNone(ordinary.interaction_id)


if __name__ == "__main__":
    unittest.main()
