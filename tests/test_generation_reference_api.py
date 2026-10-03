from __future__ import annotations

import asyncio
import os
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from services.generation_api import create_generation_app
from services.generation_service import GenerationRequest, GenerationService
from services.sd_control import GeneratedImage
from services.shogou_prompt_designer import ShogouPromptResult


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 24), (255, 0, 0)).save(output, format="PNG")
    return output.getvalue()


class _ReferenceService:
    def __init__(self) -> None:
        self.requests = []

    def prepare_generation(self, request):
        self.requests.append(request)
        return SimpleNamespace(request=request)

    def submit_generation(self, request, *, prepared):
        return SimpleNamespace(
            requested_seed=request.seed,
            images=(GeneratedImage(path=Path("image.png"), batch_index=0, seed=7),),
        )


class GenerationReferenceApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.env = patch.dict(os.environ, {"GENERATION_REFERENCE_ROOT": self.temporary.name})
        self.env.start()
        self.service = _ReferenceService()
        self.client = TestClient(TestServer(create_generation_app(service=self.service)))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()
        self.env.stop()
        self.temporary.cleanup()

    async def _upload(self) -> str:
        boundary = "----codex-reference"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="reference.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode("utf-8") + _png_bytes() + f"\r\n--{boundary}--\r\n".encode("utf-8")
        response = await self.client.post(
            "/api/generation/references",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        self.assertEqual(201, response.status, await response.text())
        payload = await response.json()
        self.assertEqual("image/png", payload["content_type"])
        self.assertEqual(32, payload["width"])
        self.assertEqual(24, payload["height"])
        return payload["reference_image_id"]

    async def _wait(self, job_id: str) -> dict:
        for _ in range(100):
            response = await self.client.get(f"/api/generation/jobs/{job_id}")
            payload = await response.json()
            if payload["state"] in {"completed", "failed"}:
                return payload
            await asyncio.sleep(0.01)
        self.fail("job did not reach a terminal state")

    async def test_upload_and_composition_job_use_opaque_asset(self) -> None:
        image_id = await self._upload()
        response = await self.client.post(
            "/api/generation/jobs",
            json={
                "freeform_character": "girl",
                "reference_mode": "composition",
                "reference_image_id": image_id,
                "reference_strength": 0.5,
                "reference_fit": "contain",
            },
        )
        self.assertEqual(202, response.status)
        result = await self._wait((await response.json())["job_id"])
        self.assertEqual("completed", result["state"])
        reference = self.service.requests[0].reference
        self.assertEqual("composition", reference["mode"])
        self.assertEqual(image_id, reference["image_id"])
        self.assertEqual(b"\x89PNG", reference["image_bytes"][:4])
        self.assertNotIn(self.temporary.name, str(result))

    async def test_pose_requires_asset_and_unknown_mode_is_rejected(self) -> None:
        for payload in (
            {"freeform_character": "girl", "reference_mode": "pose"},
            {"freeform_character": "girl", "reference_mode": "unknown"},
            {"freeform_character": "girl", "reference_mode": "composition", "reference_image_id": "../secret"},
        ):
            response = await self.client.post("/api/generation/jobs", json=payload)
            self.assertEqual(400, response.status)

    async def test_invalid_image_bytes_are_rejected(self) -> None:
        boundary = "----codex-reference"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="reference.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
            "not-an-image\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        response = await self.client.post(
            "/api/generation/references",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        self.assertEqual(400, response.status)


class GenerationReferenceServiceTests(unittest.TestCase):
    def test_prepare_maps_composition_to_existing_sd_options(self) -> None:
        service = GenerationService(
            prompt_resolver=lambda *_args, **_kwargs: ShogouPromptResult(positive="girl", negative="bad")
        )
        request = GenerationRequest(
            freeform_character="girl",
            reference={
                "mode": "composition",
                "image_id": "asset-1",
                "image_bytes": _png_bytes(),
                "content_type": "image/png",
                "width": 32,
                "height": 24,
                "strength": 0.4,
                "fit": "contain",
            },
        )
        prepared = service.prepare_generation(request)
        self.assertEqual("composition", prepared.options.reference_mode)
        self.assertEqual(0.4, prepared.options.reference_strength)
        self.assertIsNotNone(prepared.options.reference_image_bytes)
        self.assertIsNone(prepared.options.pose_reference_bytes)

    def test_prepare_maps_pose_to_existing_pose_options(self) -> None:
        service = GenerationService(
            prompt_resolver=lambda *_args, **_kwargs: ShogouPromptResult(positive="girl", negative="bad")
        )
        request = GenerationRequest(
            freeform_character="girl",
            reference={
                "mode": "pose",
                "image_id": "asset-1",
                "image_bytes": _png_bytes(),
                "content_type": "image/png",
                "width": 32,
                "height": 24,
                "strength": 0.6,
                "fit": "cover",
            },
        )
        prepared = service.prepare_generation(request)
        self.assertEqual("pose", prepared.options.reference_mode)
        self.assertEqual(0.6, prepared.options.pose_strength)
        self.assertEqual("cover", prepared.options.pose_fit_mode)
        self.assertIsNotNone(prepared.options.pose_reference_bytes)
        self.assertIsNone(prepared.options.reference_image_bytes)
