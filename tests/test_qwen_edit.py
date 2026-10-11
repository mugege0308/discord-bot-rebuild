from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services.generation_api import GenerationApiError, _build_gallery_metadata, _build_request
from services.generation_service import GenerationRequest, GenerationService
from services.image_generation_profiles import QWEN_IMAGE_MODEL_KEY
from services.image_generation_profiles import get_image_generation_profile
from services.qwen_image_edit_adapter import (
    QWEN_EDIT_REQUIRED_NODE_TYPES,
    load_qwen_edit_workflow,
    patch_qwen_edit_workflow,
    validate_qwen_edit_runtime_nodes,
)
from services.qwen_workflow_adapter import QWEN_MODEL_COMPONENTS
from services.sd_control import SdGenerationOptions, validate_options


_PNG = b"\x89PNG\r\n\x1a\nreference"


def _asset() -> SimpleNamespace:
    return SimpleNamespace(
        asset_id="asset-edit-1",
        image_bytes=_PNG,
        content_type="image/png",
        width=32,
        height=32,
    )


class QwenEditAdapterTests(unittest.TestCase):
    def test_workflow_patches_single_reference_and_native_encoder(self) -> None:
        workflow, _ = load_qwen_edit_workflow()
        patched = patch_qwen_edit_workflow(
            workflow,
            reference_image_name="upload-edit.png",
            prompt="Turn the sky into a warm sunset",
            width=512,
            height=512,
            seed=42,
        )
        classes = {node["class_type"]: node["inputs"] for node in patched.values()}
        self.assertEqual(set(QWEN_EDIT_REQUIRED_NODE_TYPES), set(classes))
        self.assertEqual("upload-edit.png", classes["LoadImage"]["image"])
        self.assertEqual("Turn the sky into a warm sunset", classes["TextEncodeQwenImage21"]["prompt"])
        self.assertEqual("", classes["TextEncodeQwenImage21"]["negative_prompt"])
        self.assertEqual({"image_1": ["5", 0]}, classes["TextEncodeQwenImage21"]["images"])
        self.assertEqual((42, 8, 1.0, "euler", "simple"), (
            classes["KSampler"]["seed"],
            classes["KSampler"]["steps"],
            classes["KSampler"]["cfg"],
            classes["KSampler"]["sampler_name"],
            classes["KSampler"]["scheduler"],
        ))
        self.assertEqual(QWEN_MODEL_COMPONENTS["diffusion_model"], classes["UNETLoader"]["unet_name"])

    def test_runtime_requires_native_qwen_21_encoder_and_cache(self) -> None:
        validate_qwen_edit_runtime_nodes({"TextEncodeQwenImage21": {}, "QwenImage21Cache": {}})
        with self.assertRaisesRegex(RuntimeError, "Qwen"):
            validate_qwen_edit_runtime_nodes({"TextEncodeQwenImageEdit": {}})


class QwenEditRequestTests(unittest.TestCase):
    def _payload(self, **overrides):
        payload = {
            "model_key": QWEN_IMAGE_MODEL_KEY,
            "mode": "edit",
            "freeform_character": "a calm lake at sunset",
            "size": "square",
            "reference_image_id": "asset-edit-1",
        }
        payload.update(overrides)
        return payload

    def test_edit_capability_is_internal_until_options_ui_is_enabled(self) -> None:
        profile = get_image_generation_profile(QWEN_IMAGE_MODEL_KEY)
        self.assertTrue(profile.supports_edit)
        self.assertTrue(profile.supports_image_reference)
        self.assertEqual(1, profile.max_reference_images)

    def test_edit_accepts_one_reference_without_quality_tags(self) -> None:
        with patch.dict(os.environ, {"ENABLE_QWEN_IMAGE_2_1": "1"}, clear=False), patch(
            "services.generation_api.load_reference_asset", return_value=_asset()
        ):
            request = _build_request(self._payload())
            prepared = GenerationService().prepare_generation(request)
            metadata = _build_gallery_metadata("edit-job", request, prepared)
        self.assertEqual("edit", request.mode)
        self.assertEqual("edit", request.reference["mode"])
        self.assertEqual("edit", prepared.options.mode)
        self.assertEqual("", prepared.options.negative_prompt)
        self.assertEqual(1, metadata["reference_image_count"])
        self.assertEqual("edit", metadata["mode"])
        self.assertNotIn("masterpiece", prepared.options.prompt.casefold())
        self.assertNotIn("danbooru", prepared.options.prompt.casefold())

    def test_edit_requires_exactly_one_reference(self) -> None:
        with patch.dict(os.environ, {"ENABLE_QWEN_IMAGE_2_1": "1"}, clear=False), patch(
            "services.generation_api.load_reference_asset", return_value=_asset()
        ):
            with self.assertRaisesRegex(GenerationApiError, "1枚"):
                _build_request(self._payload(reference_image_ids=["asset-edit-1", "asset-edit-2"]))
            with self.assertRaisesRegex(GenerationApiError, "1枚"):
                _build_request(self._payload(reference_image_id=None))

    def test_qwen_t2i_default_remains_unchanged(self) -> None:
        with patch.dict(os.environ, {"ENABLE_QWEN_IMAGE_2_1": "1"}, clear=False):
            request = _build_request({
                "model_key": QWEN_IMAGE_MODEL_KEY,
                "freeform_character": "a red fox in snow",
                "size": "square",
            })
            prepared = GenerationService().prepare_generation(request)
        self.assertEqual("t2i", request.mode)
        self.assertEqual("t2i", prepared.options.mode)
        self.assertIsNone(prepared.options.reference_image_bytes)

    def test_sd_validation_preserves_edit_mode(self) -> None:
        options = SdGenerationOptions(
            prompt="edit this image",
            model_key=QWEN_IMAGE_MODEL_KEY,
            model_family="qwen_image",
            engine="qwen_image",
            mode="edit",
            width=512,
            height=512,
            count=1,
            steps=8,
            cfg=1.0,
            sampler="euler",
            scheduler="simple",
            preserve_prompt_text=True,
            reference_image_bytes=b"png",
            reference_image_content_type="image/png",
            reference_mode="edit",
        )
        checked = validate_options(options)
        self.assertEqual("edit", checked.mode)
        self.assertEqual("edit", checked.reference_mode)


if __name__ == "__main__":
    unittest.main()
