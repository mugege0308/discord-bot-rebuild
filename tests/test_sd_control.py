import ast
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from services.sd_control import DEFAULT_NEGATIVE_PROMPT, SdGenerationOptions, _apply_comfy_options, validate_options


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
            "animaginexl40opt_clear.safetensors",
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

        self.assertNotIn("model_key", {keyword.arg for keyword in options_call.keywords})
        self.assertIsNone(SdGenerationOptions(prompt="existing generate").model_key)


if __name__ == "__main__":
    unittest.main()
