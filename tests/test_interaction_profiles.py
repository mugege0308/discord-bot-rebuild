from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from services.generation_api import (
    GenerationApiError,
    _build_gallery_metadata,
    _build_request,
    create_generation_app,
)
from services.generation_service import GenerationRequest, GenerationService
from services.interaction_profiles import (
    InteractionProfileNotFound,
    apply_prompt_only_profile,
    list_phase1_profiles,
    list_production_profiles,
    resolve_interaction_profile,
    resolve_phase1_interaction,
)
from services.shogou_prompt_designer import ShogouPromptResult


class InteractionProfileTests(unittest.TestCase):
    def test_phase1_listing_contains_only_allowlist(self) -> None:
        profiles = list_phase1_profiles()
        self.assertEqual(
            {
                "holding_hands",
                "dancing_together",
                "hand_on_shoulder",
                "arm_around_shoulder",
                "leaning_on_partner",
            },
            {item["id"] for item in profiles},
        )
        self.assertNotIn("hugging", {item["id"] for item in profiles})

    def test_prompt_merge_preserves_order_and_deduplicates(self) -> None:
        profile = resolve_phase1_interaction("holding_hands")
        positive, negative = apply_prompt_only_profile(
            profile,
            "1girl, holding hands, full body",
            "bad quality",
        )
        self.assertEqual("1girl, holding hands, full body, standing together", positive)
        self.assertEqual("bad quality", negative)

    def test_production_listing_includes_released_hugging(self) -> None:
        profiles = list_production_profiles()
        self.assertEqual(
            {
                "holding_hands",
                "dancing_together",
                "hand_on_shoulder",
                "arm_around_shoulder",
                "leaning_on_partner",
                "hugging",
            },
            {item["id"] for item in profiles},
        )

    def test_all_phase1_profiles_are_accepted_as_prompt_only(self) -> None:
        expected = {
            "holding_hands",
            "dancing_together",
            "hand_on_shoulder",
            "arm_around_shoulder",
            "leaning_on_partner",
        }
        for interaction_id in expected:
            with self.subTest(interaction_id=interaction_id):
                request = _build_request(
                    {"freeform_character": "couple", "interaction_id": interaction_id}
                )
                self.assertEqual(interaction_id, request.interaction_id)
                profile = resolve_phase1_interaction(interaction_id)
                self.assertTrue(profile.ready)
                self.assertEqual("production_candidate", profile.classification)
                self.assertEqual("prompt_only", profile.mode)

    def test_out_of_scope_profiles_are_rejected_without_fallback(self) -> None:
        cases = {
            "sitting_together": "interaction_not_released",
            "embracing": "interaction_not_released",
            "carrying": "interaction_not_ready",
            "does_not_exist": "interaction_not_found",
        }
        for interaction_id, expected_code in cases.items():
            with self.subTest(interaction_id=interaction_id):
                with self.assertRaises(GenerationApiError) as caught:
                    _build_request(
                        {"freeform_character": "couple", "interaction_id": interaction_id}
                    )
                self.assertEqual(expected_code, caught.exception.code)

    def test_unknown_and_unreleased_are_explicit(self) -> None:
        with self.assertRaises(InteractionProfileNotFound):
            resolve_interaction_profile("does_not_exist")
        with self.assertRaises(GenerationApiError) as caught:
            _build_request({"freeform_character": "couple", "interaction_id": "sitting_together"})
        self.assertEqual("interaction_not_released", caught.exception.code)

    def test_request_without_interaction_keeps_default(self) -> None:
        request = _build_request({"freeform_character": "girl"})
        self.assertIsNone(request.interaction_id)
        batch_request = _build_request({"freeform_character": "girl", "count": 2})
        self.assertIsNone(batch_request.interaction_id)
        self.assertEqual(2, batch_request.count)

    def test_generation_service_applies_profile_without_prompt_requery(self) -> None:
        calls: list[str] = []

        def prompt_resolver(*args, **kwargs):
            calls.append(str(args[0]))
            return ShogouPromptResult(
                positive="1girl, 1boy, full body",
                negative="bad quality",
            )

        service = GenerationService(prompt_resolver=prompt_resolver)
        resolved = service.resolve_request(
            GenerationRequest(freeform_character="couple", interaction_id="holding_hands")
        )
        self.assertEqual(1, len(calls))
        self.assertIn("holding hands", resolved.prompt_result.positive)
        self.assertIn("standing together", resolved.prompt_result.positive)

    def test_gallery_metadata_has_safe_interaction_metadata_only(self) -> None:
        service = GenerationService(
            prompt_resolver=lambda *_args, **_kwargs: ShogouPromptResult(
                positive="1girl, 1boy, full body",
                negative="bad quality",
            )
        )
        request = GenerationRequest(
            freeform_character="couple",
            interaction_id="holding_hands",
            width=768,
            height=1024,
        )
        prepared = service.prepare_generation(request)
        metadata = _build_gallery_metadata("job-1", request, prepared)
        self.assertEqual(
            {
                "requested_id": "holding_hands",
                "profile_id": "holding_hands",
                "classification": "production_candidate",
                "mode": "prompt_only",
                "ledger_version": 1,
            },
            metadata["interaction"],
        )
        self.assertNotIn("source_path", str(metadata))
        self.assertNotIn("C:\\", str(metadata))

    def test_gallery_metadata_omits_interaction_without_request_field(self) -> None:
        service = GenerationService(
            prompt_resolver=lambda *_args, **_kwargs: ShogouPromptResult(
                positive="1girl, full body",
                negative="bad quality",
            )
        )
        request = GenerationRequest(freeform_character="girl")
        prepared = service.prepare_generation(request)
        metadata = _build_gallery_metadata("job-no-interaction", request, prepared)
        self.assertNotIn("interaction", metadata)


class InteractionProfileApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_profile_listing_api_exposes_released_profiles(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tempdir:
            app = create_generation_app(overnight_db_path=Path(tempdir) / "overnight.db")
            server = TestServer(app)
            client = TestClient(server)
            await client.start_server()
            try:
                response = await client.get("/api/generation/interaction-profiles?released=production_candidate")
                self.assertEqual(200, response.status)
                payload = await response.json()
                self.assertEqual(6, len(payload))
                self.assertIn("hugging", {item["id"] for item in payload})
                self.assertNotIn("prompt_additions", str(payload))
                response = await client.get("/api/generation/interaction-profiles?released=experimental")
                self.assertEqual([], await response.json())
            finally:
                await client.close()


if __name__ == "__main__":
    unittest.main()
