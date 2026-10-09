from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer

from services.generation_api import (
    GenerationApiError,
    _build_gallery_metadata,
    _build_request,
    create_generation_app,
)
from services.generation_service import GenerationRequest, GenerationService
from services.prompt_generator import PromptGeneratorError
from services.shogou_prompt_designer import ShogouPromptResult
from services.image_generation_profiles import (
    QWEN_IMAGE_MODEL_KEY,
    enabled_image_generation_profiles,
)
from services.qwen_image_prompt import build_qwen_prompt
from services.qwen_workflow_adapter import (
    QWEN_REQUIRED_RUNTIME_NODES,
    load_qwen_workflow,
    patch_qwen_workflow,
    validate_qwen_runtime_nodes,
)


class QwenAdapterTests(unittest.TestCase):
    def _request(self, **overrides):
        payload = {
            "model_key": QWEN_IMAGE_MODEL_KEY,
            "freeform_character": "a red fox in snow",
            "size": "square",
        }
        payload.update(overrides)
        return _build_request(payload)

    def test_feature_flag_hides_profile_by_default(self):
        with patch.dict(os.environ, {"ENABLE_QWEN_IMAGE_2_1": "0"}, clear=False):
            self.assertNotIn(QWEN_IMAGE_MODEL_KEY, {p.model_key for p in enabled_image_generation_profiles()})
            with self.assertRaisesRegex(GenerationApiError, "現在無効"):
                self._request()

    def test_feature_flag_exposes_profile_and_qwen_defaults(self):
        with patch.dict(os.environ, {"ENABLE_QWEN_IMAGE_2_1": "1"}, clear=False):
            request = self._request()
            prepared = GenerationService().prepare_generation(request)
        options = prepared.options
        self.assertEqual("qwen_image", options.engine)
        self.assertEqual((512, 512), (request.width, request.height,))
        self.assertEqual((8, 1.0, "euler", "simple"), (options.steps, options.cfg, options.sampler, options.scheduler))
        self.assertEqual("", options.negative_prompt)
        self.assertIn("a red fox in snow", options.prompt)
        self.assertNotIn("masterpiece", options.prompt.casefold())
        self.assertNotIn("high score", options.prompt.casefold())

    def test_qwen_prompt_keeps_natural_language_layers(self):
        with patch.dict(os.environ, {"ENABLE_QWEN_IMAGE_2_1": "1"}, clear=False):
            request = self._request()
        prompt = build_qwen_prompt(request)
        self.assertEqual("a red fox in snow", prompt)
        self.assertNotIn("danbooru", prompt.casefold())
        self.assertNotIn("absurdres", prompt.casefold())

    def test_qwen_capabilities_reject_unsupported_inputs(self):
        with patch.dict(os.environ, {"ENABLE_QWEN_IMAGE_2_1": "1"}, clear=False):
            with self.assertRaisesRegex(GenerationApiError, "複数枚"):
                self._request(count=2)
            with self.assertRaisesRegex(GenerationApiError, "LoRA"):
                self._request(loras=[{"key": "style", "strength": 0.5}])
            with self.assertRaisesRegex(GenerationApiError, "Reference"):
                self._request(reference_mode="composition")
            with self.assertRaisesRegex(GenerationApiError, "size"):
                self._request(size="portrait")
            request = self._request()
            with self.assertRaisesRegex(PromptGeneratorError, "variation"):
                GenerationService().prepare_generation(
                    GenerationRequest(
                        freeform_character="fox",
                        model_key=QWEN_IMAGE_MODEL_KEY,
                        variation="auto",
                    )
                )

    def test_animagine_profile_keeps_existing_sdxl_contract(self):
        request = _build_request({"model_key": "animagine", "freeform_character": "girl", "size": "square"})
        prepared = GenerationService(
            prompt_resolver=lambda *_args, **_kwargs: ShogouPromptResult(
                positive="1girl, safe",
                negative="low quality",
            )
        ).prepare_generation(request)
        self.assertEqual("sdxl_tag", prepared.options.engine)
        self.assertEqual((1024, 1024), (request.width, request.height))
        self.assertEqual("Euler a", prepared.options.sampler)

    def test_qwen_metadata_identifies_engine_workflow_and_components(self):
        with patch.dict(os.environ, {"ENABLE_QWEN_IMAGE_2_1": "1"}, clear=False):
            request = self._request(seed=17)
            prepared = GenerationService().prepare_generation(request)
            metadata = _build_gallery_metadata("job-qwen", request, prepared)
        self.assertEqual("qwen_image", metadata["engine"])
        self.assertEqual("qwen_image_2_1_w4a8", metadata["workflow"])
        self.assertEqual({"diffusion_model", "text_encoder", "vae"}, set(metadata["model_components"]))
        self.assertEqual((512, 512), (metadata["width"], metadata["height"]))
        self.assertEqual((8, "euler", "simple"), (metadata["steps"], metadata["sampler"], metadata["scheduler"]))


class QwenWorkflowAdapterTests(unittest.TestCase):
    def test_workflow_loads_and_patches_all_qwen_inputs(self):
        workflow, _ = load_qwen_workflow()
        patched = patch_qwen_workflow(
            workflow,
            prompt="a calm landscape",
            width=512,
            height=512,
            seed=42,
        )
        classes = {node["class_type"]: node["inputs"] for node in patched.values()}
        self.assertEqual("a calm landscape", classes["TextEncodeQwenImage21"]["prompt"])
        self.assertEqual(512, classes["TextEncodeQwenImage21"]["resolution"])
        self.assertEqual((512, 512), (classes["EmptyLatentImage"]["width"], classes["EmptyLatentImage"]["height"]))
        self.assertEqual((42, 8, 1.0, "euler", "simple"), (
            classes["KSampler"]["seed"],
            classes["KSampler"]["steps"],
            classes["KSampler"]["cfg"],
            classes["KSampler"]["sampler_name"],
            classes["KSampler"]["scheduler"],
        ))

    def test_runtime_preflight_requires_qwen_nodes(self):
        validate_qwen_runtime_nodes({name: {} for name in QWEN_REQUIRED_RUNTIME_NODES})
        with self.assertRaisesRegex(RuntimeError, "Qwen用node"):
            validate_qwen_runtime_nodes({"TextEncodeQwenImage21": {}})


class QwenOptionsApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tempdir = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.client = TestClient(TestServer(create_generation_app(
            overnight_db_path=Path(self.tempdir.name) / "overnight.db",
        )))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.tempdir.cleanup()

    async def test_options_hide_qwen_when_feature_flag_is_off(self):
        with patch.dict(os.environ, {"ENABLE_QWEN_IMAGE_2_1": "0"}, clear=False):
            response = await self.client.get("/api/generation/options")
            payload = await response.json()
        self.assertEqual(200, response.status)
        self.assertNotIn(QWEN_IMAGE_MODEL_KEY, {item["key"] for item in payload["models"]})

    async def test_options_expose_qwen_capabilities_when_enabled(self):
        with patch.dict(os.environ, {"ENABLE_QWEN_IMAGE_2_1": "1"}, clear=False):
            response = await self.client.get("/api/generation/options")
            payload = await response.json()
        self.assertEqual(200, response.status)
        qwen = next(item for item in payload["models"] if item["key"] == QWEN_IMAGE_MODEL_KEY)
        self.assertEqual(["square"], qwen["supported_sizes"])
        self.assertEqual({"lora": False, "reference": False, "pose": False, "bulk": False, "overnight": False}, qwen["capabilities"])
        self.assertEqual(512, payload["model_sizes"][QWEN_IMAGE_MODEL_KEY][0]["width"])
        self.assertNotIn(QWEN_IMAGE_MODEL_KEY, str(payload["loras"]))

    async def test_bulk_and_overnight_reject_qwen_before_queueing(self):
        with patch.dict(os.environ, {"ENABLE_QWEN_IMAGE_2_1": "1"}, clear=False):
            bulk_response = await self.client.post(
                "/api/generation/bulk-jobs",
                json={"model_key": QWEN_IMAGE_MODEL_KEY, "total_count": 16, "freeform_character": "fox"},
            )
            overnight_response = await self.client.post(
                "/api/generation/overnight-jobs",
                json={"model_key": QWEN_IMAGE_MODEL_KEY, "total_count": 128, "freeform_character": "fox"},
            )
        self.assertEqual(400, bulk_response.status)
        self.assertIn("Bulk", (await bulk_response.json())["error"])
        self.assertEqual(400, overnight_response.status)
        self.assertIn("Overnight", (await overnight_response.json())["error"])


if __name__ == "__main__":
    unittest.main()
