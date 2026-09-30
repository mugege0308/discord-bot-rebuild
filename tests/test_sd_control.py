import ast
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from services.sd_control import (
    DEFAULT_NEGATIVE_PROMPT,
    SdGenerationOptions,
    _apply_comfy_options,
    generate_images_with_metadata,
    validate_options,
)


def _workflow(checkpoints: int = 1) -> dict:
    workflow = {
        "positive": {
            "inputs": {"text": "old positive"},
            "class_type": "CLIPTextEncode",
        },
        "negative": {
            "inputs": {"text": "old negative"},
            "class_type": "CLIPTextEncode",
        },
        "sampler": {
            "inputs": {
                "positive": ["positive", 0],
                "negative": ["negative", 0],
                "seed": 123,
                "steps": 20,
                "cfg": 5.0,
                "sampler_name": "euler_ancestral",
            },
            "class_type": "KSampler",
        },
        "save": {
            "inputs": {"images": ["decode", 0], "filename_prefix": "unchanged"},
            "class_type": "SaveImage",
        },
        "unrelated": {
            "inputs": {"value": "must remain unchanged"},
            "class_type": "UnrelatedNode",
        },
    }
    for index in range(checkpoints):
        workflow[f"checkpoint_{index}"] = {
            "inputs": {"ckpt_name": "existing.safetensors", "other": "unchanged"},
            "class_type": "CheckpointLoaderSimple",
        }
    return workflow


class SdGenerationOptionsTestCase(unittest.TestCase):
    def test_existing_mode_strips_prompts_and_uses_default_negative(self):
        with mock.patch.dict("os.environ", {"SD_DEFAULT_NEGATIVE_PROMPT": " configured default "}):
            result = validate_options(SdGenerationOptions(prompt="  positive  "))

        self.assertEqual(result.prompt, "positive")
        self.assertEqual(result.negative_prompt, "configured default")
        self.assertFalse(result.preserve_prompt_text)

    def test_raw_mode_preserves_positive_and_negative_exactly(self):
        result = validate_options(
            SdGenerationOptions(
                prompt="  positive, detailed  ",
                negative_prompt="  low quality, blurry  ",
                preserve_prompt_text=True,
            )
        )

        self.assertEqual(result.prompt, "  positive, detailed  ")
        self.assertEqual(result.negative_prompt, "  low quality, blurry  ")
        self.assertTrue(result.preserve_prompt_text)

    def test_raw_mode_allows_explicit_empty_negative(self):
        result = validate_options(
            SdGenerationOptions(
                prompt="positive",
                negative_prompt="",
                preserve_prompt_text=True,
            )
        )

        self.assertEqual(result.negative_prompt, "")

    def test_raw_mode_still_rejects_whitespace_only_positive(self):
        with self.assertRaisesRegex(ValueError, "prompt is empty"):
            validate_options(
                SdGenerationOptions(
                    prompt="   ",
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    preserve_prompt_text=True,
                )
            )


class ComfyCheckpointSelectionTestCase(unittest.TestCase):
    def _apply(self, workflow: dict, model_key: str | None) -> dict:
        result, _, _ = _apply_comfy_options(
            workflow,
            SdGenerationOptions(
                prompt="new positive",
                negative_prompt="new negative",
                model_key=model_key,
            ),
        )
        return result

    def test_model_omitted_keeps_existing_checkpoint(self):
        workflow = _workflow()

        result = self._apply(workflow, None)

        self.assertEqual(result["checkpoint_0"]["inputs"]["ckpt_name"], "existing.safetensors")

    def test_animagine_selects_allowed_checkpoint(self):
        result = self._apply(_workflow(), "animagine")

        self.assertEqual(
            result["checkpoint_0"]["inputs"]["ckpt_name"],
            "animagineXL40Opt_clear.safetensors",
        )

    def test_janku_selects_allowed_checkpoint(self):
        result = self._apply(_workflow(), "janku")

        self.assertEqual(
            result["checkpoint_0"]["inputs"]["ckpt_name"],
            "JANKUTrainedChenkinNoobai_v777.safetensors",
        )

    def test_unknown_model_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "対応していません"):
            self._apply(_workflow(), "../../arbitrary.safetensors")

    def test_missing_checkpoint_loader_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "CheckpointLoaderSimple ノードが見つかりません"):
            self._apply(_workflow(checkpoints=0), "animagine")

    def test_multiple_checkpoint_loaders_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "CheckpointLoaderSimple ノードが複数あります"):
            self._apply(_workflow(checkpoints=2), "animagine")

    def test_unrelated_node_values_are_not_changed(self):
        workflow = _workflow()
        unrelated_before = deepcopy(workflow["unrelated"])
        checkpoint_other_before = workflow["checkpoint_0"]["inputs"]["other"]

        result = self._apply(workflow, "janku")

        self.assertEqual(result["unrelated"], unrelated_before)
        self.assertEqual(result["checkpoint_0"]["inputs"]["other"], checkpoint_other_before)


class ComfyGenerationSettingsTestCase(unittest.TestCase):
    def _workflow_with_generation_nodes(self) -> dict:
        workflow = _workflow()
        workflow["latent"] = {
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
            "class_type": "EmptyLatentImage",
        }
        workflow["sampler"]["inputs"]["latent_image"] = ["latent", 0]
        return workflow

    def test_explicit_generation_settings_are_applied_to_matching_nodes(self):
        workflow = self._workflow_with_generation_nodes()
        result, _, _ = _apply_comfy_options(
            workflow,
            SdGenerationOptions(
                prompt="positive",
                negative_prompt="negative",
                width=512,
                height=512,
                steps=12,
                cfg=5.0,
                sampler="euler_ancestral",
                explicit_generation_fields=frozenset({"width", "height", "steps", "cfg", "sampler"}),
            ),
        )
        self.assertEqual(result["latent"]["inputs"]["width"], 512)
        self.assertEqual(result["latent"]["inputs"]["height"], 512)
        self.assertEqual(result["sampler"]["inputs"]["steps"], 12)
        self.assertEqual(result["sampler"]["inputs"]["cfg"], 5.0)
        self.assertEqual(result["sampler"]["inputs"]["sampler_name"], "euler_ancestral")

    def test_unspecified_generation_settings_keep_workflow_defaults(self):
        workflow = self._workflow_with_generation_nodes()
        result, _, _ = _apply_comfy_options(
            workflow,
            SdGenerationOptions(prompt="positive", negative_prompt="negative"),
        )
        self.assertEqual(result["latent"]["inputs"]["width"], 1024)
        self.assertEqual(result["latent"]["inputs"]["height"], 1024)
        self.assertEqual(result["sampler"]["inputs"]["steps"], 20)
        self.assertEqual(result["sampler"]["inputs"]["cfg"], 5.0)
        self.assertEqual(result["sampler"]["inputs"]["sampler_name"], "euler_ancestral")

    def test_explicit_settings_fail_when_matching_node_is_missing(self):
        with self.assertRaisesRegex(RuntimeError, "EmptyLatentImage"):
            _apply_comfy_options(
                _workflow(),
                SdGenerationOptions(
                    prompt="positive",
                    negative_prompt="negative",
                    width=512,
                    explicit_generation_fields=frozenset({"width"}),
                ),
            )

    def test_unknown_explicit_generation_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "explicit_generation_fields"):
            validate_options(
                SdGenerationOptions(
                    prompt="positive",
                    explicit_generation_fields=frozenset({"scheduler"}),
                )
            )

    def test_animagine_filename_prefix_is_applied_to_single_save_node(self):
        workflow = _workflow()
        result, _, _ = _apply_comfy_options(
            workflow,
            SdGenerationOptions(
                prompt="positive",
                negative_prompt="negative",
                model_key="animagine",
                filename_prefix="Shogou_AnimagineXL",
            ),
        )
        self.assertEqual(result["save"]["inputs"]["filename_prefix"], "Shogou_AnimagineXL")

    def test_janku_filename_prefix_is_applied_to_single_save_node(self):
        workflow = _workflow()
        result, _, _ = _apply_comfy_options(
            workflow,
            SdGenerationOptions(
                prompt="positive",
                negative_prompt="negative",
                model_key="janku",
                filename_prefix="Shogou_JANKU",
            ),
        )
        self.assertEqual(result["save"]["inputs"]["filename_prefix"], "Shogou_JANKU")

    def test_filename_prefix_rejects_multiple_save_nodes(self):
        workflow = _workflow()
        workflow["save_2"] = deepcopy(workflow["save"])
        with self.assertRaisesRegex(RuntimeError, "SaveImage ノードが複数"):
            _apply_comfy_options(
                workflow,
                SdGenerationOptions(
                    prompt="positive",
                    negative_prompt="negative",
                    model_key="janku",
                    filename_prefix="Shogou_JANKU",
                ),
            )

    def test_existing_generate_builder_does_not_select_a_model(self):
        prompt_path = Path(__file__).resolve().parents[1] / "commands" / "prompt.py"
        tree = ast.parse(prompt_path.read_text(encoding="utf-8"), filename=str(prompt_path))
        builder = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_build_generation_options"
        )
        options_call = next(
            node
            for node in ast.walk(builder)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SdGenerationOptions"
        )

        model_keyword = next(keyword for keyword in options_call.keywords if keyword.arg == "model_key")
        self.assertIsInstance(model_keyword.value, ast.Name)
        self.assertEqual(model_keyword.value.id, "model_key")
        defaults = dict(zip((arg.arg for arg in builder.args.kwonlyargs), builder.args.kw_defaults))
        self.assertIsNone(defaults["model_key"].value)
        self.assertIsNone(SdGenerationOptions(prompt="existing generate").model_key)


class ModelBackendValidationTestCase(unittest.TestCase):
    def test_explicit_model_selection_is_rejected_outside_comfy(self):
        options = SdGenerationOptions(
            prompt="positive",
            model_key="janku",
            model_selection_explicit=True,
        )
        with mock.patch("services.sd_control._backend", return_value="webui"), mock.patch(
            "services.sd_control._generate_webui_images_with_metadata"
        ) as webui_mock:
            with self.assertRaisesRegex(RuntimeError, "ComfyUIバックエンド"):
                generate_images_with_metadata(options)
        webui_mock.assert_not_called()

    def test_default_animagine_marker_does_not_break_existing_webui_path(self):
        options = SdGenerationOptions(
            prompt="positive",
            model_key="animagine",
            model_selection_explicit=False,
        )
        with mock.patch("services.sd_control._backend", return_value="webui"), mock.patch(
            "services.sd_control._generate_webui_images_with_metadata", return_value=[]
        ) as webui_mock, mock.patch("services.sd_control._save_latest_paths"):
            result = generate_images_with_metadata(options)
        self.assertEqual(result, [])
        webui_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
