from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer

from services.generation_api import (
    GenerationApiConfig,
    _build_overnight_child_request,
    _build_request,
    _build_gallery_metadata,
    create_generation_app,
)
from services.generation_service import GenerationRequest, GenerationService
from services.image_selection import ImageSelectionResult
from services.sd_control import GeneratedImage
from services.shogou_prompt_designer import ShogouPromptResult


class _FakeGenerationService:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.requests = []

    def prepare_generation(self, request):
        self.requests.append(request)
        return SimpleNamespace(request=request)

    def submit_generation(self, request, *, prepared):
        if self.failure is not None:
            raise self.failure
        return SimpleNamespace(
            requested_seed=request.seed,
            images=(
                GeneratedImage(
                    path=Path(r"C:\private\outputs\image.png"),
                    batch_index=0,
                    seed=12345,
                ),
            ),
        )


class _MultiImageService(GenerationService):
    def __init__(self, *, failure_at: int | None = None) -> None:
        self.prompt_call_count = 0
        self.failure_at = failure_at

        def prompt_resolver(*_args, **_kwargs):
            self.prompt_call_count += 1
            return ShogouPromptResult(
                positive="1girl, portrait, safe",
                negative="low quality",
            )

        super().__init__(
            prompt_resolver=prompt_resolver
        )
        self.submit_calls = []

    def submit_generation(self, request, *, prepared):
        self.submit_calls.append((request, prepared))
        if self.failure_at == len(self.submit_calls):
            raise RuntimeError("slot failed")
        return SimpleNamespace(
            requested_seed=request.seed,
            images=(
                GeneratedImage(
                    path=Path(f"C:/private/outputs/image-{len(self.submit_calls)}.png"),
                    batch_index=0,
                    seed=100 + len(self.submit_calls),
                ),
            ),
        )


class GenerationApiTests(unittest.IsolatedAsyncioTestCase):
    def test_normal_count_one_auto_keeps_existing_off_semantics(self) -> None:
        request = _build_request({"freeform_character": "girl", "count": 1, "variation": "auto"})
        self.assertEqual(1, request.count)
        self.assertEqual("off", request.variation)

    def test_overnight_count_one_child_preserves_parent_auto(self) -> None:
        request = _build_overnight_child_request(
            {"freeform_character": "girl", "count": 1, "variation": "auto"}
        )
        self.assertEqual(1, request.count)
        self.assertEqual("auto", request.variation)

    async def asyncSetUp(self):
        self.service = _FakeGenerationService()
        self.app = create_generation_app(service=self.service)
        self.server = TestServer(self.app)
        self.client = TestClient(self.server)
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def _wait_for_terminal(self, job_id: str) -> dict:
        for _ in range(100):
            response = await self.client.get(f"/api/generation/jobs/{job_id}")
            payload = await response.json()
            if payload["state"] in {"completed", "failed"}:
                return payload
            await asyncio.sleep(0.01)
        self.fail("job did not reach a terminal state")

    async def test_health_does_not_probe_or_expose_urls(self):
        with patch.dict(
            os.environ,
            {"LMSTUDIO_API_URL": "http://127.0.0.1:1234/v1", "SD_API_URL": "http://127.0.0.1:8188"},
            clear=False,
        ):
            response = await self.client.get("/api/generation/health")
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertTrue(payload["service_ready"])
        self.assertTrue(payload["lm_studio_configured"])
        self.assertTrue(payload["comfyui_url_configured"])
        self.assertNotIn("127.0.0.1:8188", str(payload))

    async def test_options_expose_safe_catalog_values_only(self):
        response = await self.client.get("/api/generation/options")
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertIn("animagine", {item["key"] for item in payload["models"]})
        self.assertEqual({"square", "portrait", "landscape"}, {item["key"] for item in payload["sizes"]})
        self.assertIn("style", payload["loras"])
        self.assertIn("character_presets", payload)
        self.assertNotIn("source_path", str(payload))
        self.assertNotIn("metadata_path", str(payload))

    async def test_valid_request_is_async_and_preserves_input_layers(self):
        response = await self.client.post(
            "/api/generation/jobs",
            json={
                "freeform_character": "かわいい女の子",
                "explicit_tags": ["from side", "upper body", "closed mouth"],
                "appearance_text": "黒髪ロング、赤目",
                "request_text": "白背景、少し微笑む",
                "model_key": "animagine",
                "rating": "safe",
                "size": "portrait",
                "count": 1,
                "seed": 42,
            },
        )
        self.assertEqual(response.status, 202)
        created = await response.json()
        self.assertEqual("queued", created["state"])
        result = await self._wait_for_terminal(created["job_id"])
        self.assertEqual("completed", result["state"])
        request = self.service.requests[0]
        self.assertEqual(("from side", "upper body", "closed mouth"), request.explicit_tags)
        self.assertEqual("黒髪ロング、赤目", request.appearance_text)
        self.assertEqual("白背景、少し微笑む", request.request_text)
        self.assertEqual(768, request.width)
        self.assertEqual(1024, request.height)
        self.assertEqual(42, request.seed)
        self.assertEqual("image.png", result["output"]["images"][0]["filename"])
        self.assertEqual(["image.png"], result["output"]["image_identifiers"])
        self.assertEqual(created["job_id"], result["output"]["gallery_job_id"])
        self.assertNotIn("C:\\private", str(result))

    def test_gallery_metadata_keeps_request_layers_and_seed_contract(self):
        service = GenerationService(
            prompt_resolver=lambda *_args, **_kwargs: ShogouPromptResult(
                positive="1girl, portrait, black hair, blue eyes",
                negative="low quality",
            )
        )
        request = GenerationRequest(
            freeform_character="かわいい女の子",
            explicit_tags=("portrait",),
            appearance_text="黒髪、青い目",
            request_text="simple safe portrait",
            rating="safe",
            size_preset="square",
            width=1024,
            height=1024,
            seed=-1,
            source="generation_api",
        )
        prepared = service.prepare_generation(request)

        metadata = _build_gallery_metadata("api-job-1", request, prepared)

        self.assertEqual("generation_api", metadata["source"])
        self.assertEqual("api-job-1", metadata["job_id"])
        self.assertEqual(-1, metadata["requested_seed"])
        self.assertEqual(
            {
                "1girl",
                "solo",
                "safe",
                "portrait",
                "masterpiece",
                "high score",
                "great score",
                "absurdres",
                "black hair",
                "blue eyes",
            },
            {token.strip() for token in metadata["positive"].split(",")},
        )
        self.assertEqual("黒髪、青い目", metadata["appearance_text"])
        self.assertEqual({"preset": "square", "width": 1024, "height": 1024}, metadata["size"])
        self.assertNotIn("C:\\", str(metadata))
        self.assertIn("tag_provenance", metadata)
        self.assertIn("suppressed_tags", metadata)
        self.assertIn("prompt_conflicts", metadata)

        variation_metadata = _build_gallery_metadata(
            "api-job-1",
            request,
            prepared,
            variation_index=2,
            variation_tags=("upper body", "three-quarter view"),
        )
        self.assertEqual(2, variation_metadata["variation_index"])
        self.assertEqual(["upper body", "three-quarter view"], variation_metadata["variation_tags"])

    async def test_invalid_model_rating_lora_and_count_are_rejected(self):
        cases = [
            ({"freeform_character": "x", "model_key": "not-a-model"}, "model"),
            ({"freeform_character": "x", "rating": "not-a-rating"}, "rating"),
            ({"freeform_character": "x", "loras": [{"key": "not-a-lora", "lora_type": "style", "strength": 0.5}]}, "lora"),
            ({"freeform_character": "x", "count": 9}, "count"),
        ]
        for payload, _label in cases:
            with self.subTest(payload=payload):
                response = await self.client.post("/api/generation/jobs", json=payload)
                self.assertEqual(response.status, 400)

    async def test_raw_workflow_and_path_fields_are_not_an_api_input(self):
        response = await self.client.post(
            "/api/generation/jobs",
            json={
                "freeform_character": "x",
                "workflow": {"20": {"inputs": {"lora_name": "C:\\secret\\a.safetensors"}}},
            },
        )
        self.assertEqual(response.status, 400)

    async def test_oversized_payload_is_rejected(self):
        body = b"{" + (b"x" * 70000) + b"}"
        response = await self.client.post(
            "/api/generation/jobs",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status, 413)

    async def test_generation_failure_is_sanitized(self):
        await self.client.close()
        self.service = _FakeGenerationService(failure=RuntimeError(r"failed at C:\\secret\\workflow.json"))
        self.app = create_generation_app(service=self.service)
        self.server = TestServer(self.app)
        self.client = TestClient(self.server)
        await self.client.start_server()
        response = await self.client.post("/api/generation/jobs", json={"freeform_character": "x"})
        created = await response.json()
        result = await self._wait_for_terminal(created["job_id"])
        self.assertEqual("failed", result["state"])
        self.assertNotIn("C:\\secret", result["error_summary"])
        self.assertNotIn("Traceback", result["error_summary"])

    async def test_count_four_auto_resolves_once_and_tracks_each_variation(self):
        await self.client.close()
        service = _MultiImageService()
        self.app = create_generation_app(service=service)
        self.server = TestServer(self.app)
        self.client = TestClient(self.server)
        await self.client.start_server()

        response = await self.client.post(
            "/api/generation/jobs",
            json={
                "freeform_character": "かわいい女の子",
                "explicit_tags": ["closed mouth"],
                "request_text": "portrait",
                "model_key": "animagine",
                "rating": "safe",
                "count": 4,
                "variation": "auto",
                "seed": 100,
            },
        )
        self.assertEqual(202, response.status)
        created = await response.json()
        result = await self._wait_for_terminal(created["job_id"])

        self.assertEqual("completed", result["state"])
        self.assertEqual(4, result["total_count"])
        self.assertEqual(4, result["completed_count"])
        self.assertEqual(0, result["failed_count"])
        self.assertEqual(4, len(result["output"]["images"]))
        self.assertEqual(1, service.prompt_call_count)
        self.assertEqual(4, len(service.submit_calls))
        self.assertEqual([100, 101, 102, 103], [item[0].seed for item in service.submit_calls])
        self.assertFalse(
            {"smile", "serious", "surprised", "blush"}
            & {tag for item in result["output"]["images"] for tag in item["variation_tags"]}
        )

    async def test_count_four_keeps_partial_failure_counts(self):
        await self.client.close()
        service = _MultiImageService(failure_at=2)
        self.app = create_generation_app(service=service)
        self.server = TestServer(self.app)
        self.client = TestClient(self.server)
        await self.client.start_server()

        response = await self.client.post(
            "/api/generation/jobs",
            json={"freeform_character": "girl", "count": 4, "variation": "off"},
        )
        created = await response.json()
        result = await self._wait_for_terminal(created["job_id"])
        self.assertEqual("completed", result["state"])
        self.assertEqual(3, result["completed_count"])
        self.assertEqual(1, result["failed_count"])
        self.assertEqual(3, result["output"]["image_count"])

    async def test_count_eight_runs_sequentially_with_one_prompt_resolution(self):
        await self.client.close()
        service = _MultiImageService()
        self.app = create_generation_app(service=service)
        self.server = TestServer(self.app)
        self.client = TestClient(self.server)
        await self.client.start_server()

        response = await self.client.post(
            "/api/generation/jobs",
            json={"freeform_character": "girl", "count": 8, "variation": "auto", "seed": 100},
        )
        self.assertEqual(202, response.status)
        created = await response.json()
        result = await self._wait_for_terminal(created["job_id"])

        self.assertEqual("completed", result["state"])
        self.assertEqual(8, result["total_count"])
        self.assertEqual(8, result["completed_count"])
        self.assertEqual(0, result["failed_count"])
        self.assertEqual(1, service.prompt_call_count)
        self.assertEqual(8, len(service.submit_calls))
        self.assertEqual(list(range(100, 108)), [item[0].seed for item in service.submit_calls])
        self.assertEqual(list(range(1, 9)), [item["variation_index"] for item in result["output"]["images"]])

    async def test_variation_offset_four_continues_fixed_seed_slots_five_to_eight(self):
        await self.client.close()
        service = _MultiImageService()
        self.app = create_generation_app(service=service)
        self.server = TestServer(self.app)
        self.client = TestClient(self.server)
        await self.client.start_server()

        response = await self.client.post(
            "/api/generation/jobs",
            json={
                "freeform_character": "girl",
                "count": 4,
                "variation": "auto",
                "variation_offset": 4,
                "seed": 100,
            },
        )
        self.assertEqual(202, response.status)
        created = await response.json()
        result = await self._wait_for_terminal(created["job_id"])

        self.assertEqual(4, result["total_count"])
        self.assertEqual(4, result["variation_offset"])
        self.assertEqual(list(range(104, 108)), [item[0].seed for item in service.submit_calls])
        self.assertEqual([5, 6, 7, 8], [item["variation_index"] for item in result["output"]["images"]])

    async def test_count_nine_and_out_of_range_variation_offset_are_rejected(self):
        response = await self.client.post(
            "/api/generation/jobs",
            json={"freeform_character": "girl", "count": 9},
        )
        self.assertEqual(400, response.status)
        response = await self.client.post(
            "/api/generation/jobs",
            json={"freeform_character": "girl", "count": 4, "variation_offset": 5},
        )
        self.assertEqual(400, response.status)

    async def test_bulk_job_runs_sequentially_once_resolves_and_keeps_snapshot(self):
        await self.client.close()
        service = _MultiImageService()
        self.app = create_generation_app(service=service)
        self.server = TestServer(self.app)
        self.client = TestClient(self.server)
        await self.client.start_server()

        with patch(
            "services.generation_api.select_generated_images",
            return_value=ImageSelectionResult(
                [Path("C:/selector/score_0p900_01_image-1.png")],
                True,
                message="selected",
            ),
        ) as selection_mock:
            response = await self.client.post(
                "/api/generation/bulk-jobs",
                json={
                    "total_count": 16,
                    "selection_mode": "best1",
                    "freeform_character": "snapshot girl",
                    "request_text": "snapshot request",
                    "variation": "auto",
                    "seed": 100,
                },
            )
            self.assertEqual(202, response.status)
            created = await response.json()
            self.assertEqual(created["bulk_id"], created["job_id"])
            result = await self._wait_for_bulk_terminal(created["bulk_id"])

        self.assertEqual("completed", result["state"])
        self.assertEqual(16, result["requested_total"])
        self.assertEqual("auto", result["variation"])
        self.assertEqual(16, result["completed_count"])
        self.assertEqual(0, result["failed_count"])
        self.assertEqual(1, service.prompt_call_count)
        self.assertEqual(16, len(service.submit_calls))
        self.assertTrue(all(item[0].variation == "auto" for item in service.submit_calls))
        self.assertEqual(list(range(100, 116)), [item[0].seed for item in service.submit_calls])
        self.assertEqual(list(range(1, 17)), [item["variation_index"] for item in result["output"]["images"]])
        self.assertTrue(all(item["variation_tags"] for item in result["output"]["images"]))
        self.assertEqual("completed", result["selection_state"])
        selection_mock.assert_called_once()
        self.assertEqual(1, len(result["output"]["selected_images"]))
        self.assertEqual("image-1.png", result["output"]["selected_images"][0]["filename"])
        self.assertEqual(1, result["output"]["selected_images"][0]["variation_index"])
        self.assertTrue(result["output"]["images"][0]["selected"])

    async def test_bulk_rejects_invalid_total_without_changing_normal_limit(self):
        response = await self.client.post(
            "/api/generation/bulk-jobs",
            json={"total_count": 33, "freeform_character": "girl"},
        )
        self.assertEqual(400, response.status)
        response = await self.client.post(
            "/api/generation/bulk-jobs",
            json={"total_count": 16, "selection_mode": "unknown", "freeform_character": "girl"},
        )
        self.assertEqual(400, response.status)

    async def test_bulk_partial_failures_continue_and_selection_failure_does_not_fail_job(self):
        await self.client.close()
        service = _MultiImageService(failure_at=2)
        self.app = create_generation_app(service=service)
        self.server = TestServer(self.app)
        self.client = TestClient(self.server)
        await self.client.start_server()

        with patch(
            "services.generation_api.select_generated_images",
            return_value=ImageSelectionResult([], False, error="selector unavailable"),
        ):
            response = await self.client.post(
                "/api/generation/bulk-jobs",
                json={"total_count": 16, "selection_mode": "best4", "freeform_character": "girl"},
            )
            created = await response.json()
            result = await self._wait_for_bulk_terminal(created["bulk_id"])

        self.assertEqual("completed", result["state"])
        self.assertEqual(15, result["completed_count"])
        self.assertEqual(1, result["failed_count"])
        self.assertEqual("failed", result["selection_state"])
        self.assertEqual(16, len(service.submit_calls))

    async def _wait_for_bulk_terminal(self, bulk_id: str) -> dict:
        for _ in range(300):
            response = await self.client.get(f"/api/generation/bulk-jobs/{bulk_id}")
            payload = await response.json()
            if payload["state"] in {"completed", "failed", "stopped"}:
                return payload
            await asyncio.sleep(0.01)
        self.fail("bulk job did not reach a terminal state")

    def test_only_loopback_binding_is_allowed(self):
        with self.assertRaises(ValueError):
            GenerationApiConfig(host="0.0.0.0")
        self.assertEqual("127.0.0.1", GenerationApiConfig(host="localhost").host)


if __name__ == "__main__":
    unittest.main()
