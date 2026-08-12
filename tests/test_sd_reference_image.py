import json
import math
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from services.sd_control import (
    DEFAULT_REFERENCE_STRENGTH,
    SdGenerationOptions,
    _apply_comfy_options,
    _get_comfy_workflow_file,
    _upload_comfy_reference_image,
    get_reference_status_text,
    validate_options,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_WORKFLOW = ROOT / "workflows" / "comfy_workflow_ipadapter.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class ReferenceValidationTestCase(unittest.TestCase):
    def test_reference_uses_safe_default_strength(self):
        result = validate_options(
            SdGenerationOptions(
                prompt="1girl",
                reference_image_bytes=b"png",
                reference_image_content_type="image/png",
            )
        )

        self.assertEqual(result.reference_strength, DEFAULT_REFERENCE_STRENGTH)
        self.assertEqual(get_reference_status_text(True, None), "Reference image: Used\nReference strength: 0.5")

    def test_reference_omitted_keeps_previous_defaults(self):
        result = validate_options(SdGenerationOptions(prompt="1girl"))

        self.assertIsNone(result.reference_image_bytes)
        self.assertIsNone(result.reference_strength)
        self.assertEqual(get_reference_status_text(False, None), "Reference image: Not used")

    def test_strength_without_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference_image"):
            validate_options(SdGenerationOptions(prompt="1girl", reference_strength=0.5))

    def test_invalid_reference_strength_is_rejected(self):
        for strength in (-0.01, 1.01, math.nan, math.inf):
            with self.subTest(strength=strength):
                with self.assertRaisesRegex(ValueError, "0.0～1.0"):
                    validate_options(
                        SdGenerationOptions(
                            prompt="1girl",
                            reference_image_bytes=b"png",
                            reference_image_content_type="image/png",
                            reference_strength=strength,
                        )
                    )


class ReferenceWorkflowTestCase(unittest.TestCase):
    def test_reference_workflows_are_api_format(self):
        workflow = _load(REFERENCE_WORKFLOW)
        self.assertTrue(workflow)
        self.assertTrue(all("class_type" in node and "inputs" in node for node in workflow.values()))
        self.assertEqual(sum(node["class_type"] == "LoadImage" for node in workflow.values()), 1)
        self.assertEqual(sum(node["class_type"] == "IPAdapterAdvanced" for node in workflow.values()), 1)

    def test_reference_selects_dedicated_workflow(self):
        self.assertEqual(_get_comfy_workflow_file(True), Path("workflows/comfy_workflow_ipadapter.json"))

    def test_reference_name_weight_seed_and_prompt_are_applied(self):
        options = validate_options(
            SdGenerationOptions(
                prompt="1girl",
                negative_prompt="low quality",
                reference_image_bytes=b"png",
                reference_image_content_type="image/png",
                reference_strength=0.65,
            )
        )
        result, seed_node, seed_key = _apply_comfy_options(
            deepcopy(_load(REFERENCE_WORKFLOW)), options, "discord_reference_test.png"
        )

        self.assertEqual(result["21"]["inputs"]["image"], "discord_reference_test.png")
        self.assertEqual(result["24"]["inputs"]["weight"], 0.65)
        self.assertIsInstance(result[seed_node]["inputs"][seed_key], int)
        self.assertEqual(result["6"]["inputs"]["text"], "1girl")
        self.assertEqual(result["24"]["inputs"]["end_at"], 0.8)


class ReferenceUploadTestCase(unittest.TestCase):
    def test_reference_is_uploaded_as_hashed_comfy_input(self):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "name": "discord_reference_abc.png",
            "subfolder": "",
            "type": "input",
        }
        with mock.patch("services.sd_control.requests.post", return_value=response) as post_mock:
            name = _upload_comfy_reference_image(b"image-bytes", "image/png")

        self.assertEqual(name, "discord_reference_abc.png")
        _, kwargs = post_mock.call_args
        self.assertEqual(kwargs["data"], {"type": "input", "overwrite": "true"})
        self.assertEqual(kwargs["files"]["image"][2], "image/png")
        self.assertTrue(kwargs["files"]["image"][0].startswith("discord_reference_"))


if __name__ == "__main__":
    unittest.main()
