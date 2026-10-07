import math
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from services.lora_catalog import COMFY_STYLES, ComfyStyle
from services.sd_control import (
    LoraSelection,
    SdGenerationOptions,
    _apply_comfy_options,
    _get_comfy_workflow_file,
    _load_comfy_workflow,
    apply_lora_triggers,
    effective_lora_selections,
    resolve_lora_selections,
    resolve_saved_lora_selections,
    validate_options,
)


ROOT = Path(__file__).resolve().parents[1]


def _workflow() -> dict:
    return {
        "4": {
            "inputs": {"ckpt_name": "existing.safetensors"},
            "class_type": "CheckpointLoaderSimple",
        },
        "5": {
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
            "class_type": "EmptyLatentImage",
        },
        "6": {
            "inputs": {"text": "old positive", "clip": ["4", 1]},
            "class_type": "CLIPTextEncode",
        },
        "7": {
            "inputs": {"text": "old negative", "clip": ["4", 1]},
            "class_type": "CLIPTextEncode",
        },
        "10": {
            "inputs": {
                "seed": 123,
                "steps": 20,
                "cfg": 5.0,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
            "class_type": "KSampler",
        },
        "17": {
            "inputs": {"samples": ["10", 0], "vae": ["4", 2]},
            "class_type": "VAEDecode",
        },
        "19": {
            "inputs": {"filename_prefix": "test", "images": ["17", 0]},
            "class_type": "SaveImage",
        },
    }


def _catalog() -> dict[str, ComfyStyle]:
    return {
        "character": ComfyStyle(
            display_name="Character",
            lora_name="character.safetensors",
            trigger="character_trigger",
            default_strength=0.8,
            workflow_file=Path("workflows/comfy_workflow_lora.json"),
            lora_type="character",
            model_family="sdxl",
        ),
        "style": ComfyStyle(
            display_name="Style",
            lora_name="style.safetensors",
            trigger="style_trigger",
            default_strength=0.4,
            workflow_file=Path("workflows/comfy_workflow_lora.json"),
            lora_type="style",
            model_family="sdxl",
        ),
        "additional": ComfyStyle(
            display_name="Additional",
            lora_name="additional.safetensors",
            trigger="",
            default_strength=0.2,
            workflow_file=Path("workflows/comfy_workflow_lora.json"),
            lora_type="additional",
            model_family="sdxl",
        ),
        "sd15": ComfyStyle(
            display_name="SD15",
            lora_name="sd15.safetensors",
            trigger="sd15_trigger",
            default_strength=0.5,
            workflow_file=Path("workflows/comfy_workflow_lora.json"),
            lora_type="style",
            model_family="sd15",
        ),
    }


class LoraGenerationTestCase(unittest.TestCase):
    def setUp(self):
        self.catalog = _catalog()
        self.catalog_backup = dict(COMFY_STYLES)
        COMFY_STYLES.clear()
        COMFY_STYLES.update(self.catalog)
        # Workflow unit tests use synthetic catalog names; the resolver itself
        # is covered by tests/test_comfy_lora_resolver.py.
        self.resolver_patch = mock.patch(
            "services.sd_control.resolve_comfy_lora_names",
            side_effect=lambda names: tuple(names),
        )
        self.resolver_patch.start()

    def tearDown(self):
        self.resolver_patch.stop()
        COMFY_STYLES.clear()
        COMFY_STYLES.update(self.catalog_backup)

    def test_no_lora_keeps_normal_workflow_and_prompt_unchanged(self):
        workflow = _workflow()
        result, seed_node, seed_key = _apply_comfy_options(
            workflow,
            SdGenerationOptions(
                prompt="1girl",
                negative_prompt="low quality",
                seed=1879852802,
            ),
        )

        self.assertFalse(any(node.get("class_type") == "LoraLoader" for node in result.values()))
        self.assertEqual(result["6"]["inputs"]["text"], "1girl")
        self.assertEqual(result[seed_node]["inputs"][seed_key], 1879852802)
        self.assertEqual(result["10"]["inputs"]["model"], ["4", 0])
        self.assertEqual(result["6"]["inputs"]["clip"], ["4", 1])

    def test_single_lora_applies_loader_strength_and_trigger(self):
        selections = resolve_lora_selections(character_key="character", model_family="sdxl")
        result, _, _ = _apply_comfy_options(
            _workflow(),
            SdGenerationOptions(
                prompt="1girl",
                negative_prompt="low quality",
                loras=selections,
                seed=42,
            ),
        )

        loader_ids = [key for key, node in result.items() if node.get("class_type") == "LoraLoader"]
        self.assertEqual(loader_ids, ["20"])
        self.assertEqual(result["20"]["inputs"]["lora_name"], "character.safetensors")
        self.assertEqual(result["20"]["inputs"]["strength_model"], 0.8)
        self.assertEqual(result["20"]["inputs"]["strength_clip"], 0.8)
        self.assertEqual(result["20"]["inputs"]["model"], ["4", 0])
        self.assertEqual(result["10"]["inputs"]["model"], ["20", 0])
        self.assertEqual(result["6"]["inputs"]["clip"], ["20", 1])
        self.assertEqual(result["7"]["inputs"]["clip"], ["20", 1])
        self.assertEqual(result["6"]["inputs"]["text"], "character_trigger, 1girl")

    def test_mixed_case_catalog_key_is_accepted_by_lora_selection(self):
        COMFY_STYLES["matuTake_style_v1"] = ComfyStyle(
            display_name="matuTake_style_v1",
            lora_name="matuTake/matuTake_style_v1.safetensors",
            trigger="",
            default_strength=0.5,
            lora_type="style",
            model_family="sdxl",
        )
        selections = resolve_lora_selections(
            style_key="matuTake_style_v1",
            model_family="sdxl",
        )
        self.assertEqual(1, len(selections))
        self.assertEqual("matutake_style_v1", selections[0].key)
        self.assertEqual(0.5, selections[0].strength)

    def test_multiple_loras_are_serially_connected_and_ordered(self):
        selections = resolve_lora_selections(
            character_key="character",
            style_key="style",
            additional_key="additional",
            model_family="sdxl",
        )
        result, _, _ = _apply_comfy_options(
            _workflow(),
            SdGenerationOptions(prompt="1girl", negative_prompt="low quality", loras=selections),
        )

        self.assertEqual(
            [result[node]["inputs"]["lora_name"] for node in ("20", "21", "22")],
            ["character.safetensors", "style.safetensors", "additional.safetensors"],
        )
        self.assertEqual(result["21"]["inputs"]["model"], ["20", 0])
        self.assertEqual(result["21"]["inputs"]["clip"], ["20", 1])
        self.assertEqual(result["22"]["inputs"]["model"], ["21", 0])
        self.assertEqual(result["22"]["inputs"]["clip"], ["21", 1])
        self.assertEqual(result["10"]["inputs"]["model"], ["22", 0])
        self.assertEqual(result["6"]["inputs"]["clip"], ["22", 1])
        self.assertEqual(
            result["6"]["inputs"]["text"],
            "character_trigger, style_trigger, 1girl",
        )

    def test_workflow_uses_exact_comfy_candidate_without_changing_strength_or_trigger(self):
        selection = resolve_lora_selections(character_key="character", model_family="sdxl")[0]
        with mock.patch(
            "services.sd_control.resolve_comfy_lora_names",
            return_value=(r"character\character.safetensors",),
        ):
            result, _, _ = _apply_comfy_options(
                _workflow(),
                SdGenerationOptions(prompt="1girl", loras=(selection,)),
            )

        self.assertEqual(result["20"]["inputs"]["lora_name"], r"character\character.safetensors")
        self.assertEqual(result["20"]["inputs"]["strength_model"], selection.strength)
        self.assertEqual(result["20"]["inputs"]["strength_clip"], selection.strength)
        self.assertEqual(result["6"]["inputs"]["text"], "character_trigger, 1girl")

    def test_existing_trigger_and_atomic_tag_are_not_duplicated(self):
        selection = resolve_lora_selections(character_key="character", model_family="sdxl")[0]
        self.assertEqual(
            apply_lora_triggers("hu tao (genshin impact), character_trigger", (selection,)),
            "hu tao (genshin impact), character_trigger",
        )

    def test_types_and_default_strengths_are_preserved(self):
        selections = resolve_lora_selections(
            character_key="character", style_key="style", additional_key="additional", model_family="sdxl"
        )
        self.assertEqual([selection.lora_type for selection in selections], ["character", "style", "additional"])
        self.assertEqual([selection.strength for selection in selections], [0.8, 0.4, 0.2])

    def test_invalid_strengths_are_rejected(self):
        for strength in (-0.01, 1.01, math.nan, math.inf, "0.5"):
            with self.subTest(strength=strength):
                with self.assertRaisesRegex(ValueError, "0.0～1.0"):
                    resolve_lora_selections(
                        character_key="character",
                        character_strength=strength,
                        model_family="sdxl",
                    )

    def test_missing_unknown_and_malformed_lora_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "未登録"):
            resolve_lora_selections(character_key="missing", model_family="sdxl")
        with self.assertRaisesRegex(ValueError, "保存済みLoRA構成"):
            resolve_saved_lora_selections({"key": "character"})
        with self.assertRaisesRegex(ValueError, "未登録"):
            resolve_saved_lora_selections([{"key": "missing", "strength": 0.5}])

    def test_managed_lora_missing_source_file_is_rejected(self):
        self.catalog["missing_file"] = ComfyStyle(
            display_name="Missing file",
            lora_name="missing_file.safetensors",
            trigger="missing_file",
            default_strength=0.5,
            workflow_file=Path("workflows/comfy_workflow_lora.json"),
            source_path=Path("does-not-exist/missing_file.safetensors"),
            lora_type="style",
            model_family="sdxl",
        )
        COMFY_STYLES.clear()
        COMFY_STYLES.update(self.catalog)
        with self.assertRaisesRegex(ValueError, "見つかりません"):
            resolve_lora_selections(style_key="missing_file", model_family="sdxl")

    def test_incompatible_model_family_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "モデル系統"):
            resolve_lora_selections(style_key="sd15", model_family="sdxl")

    def test_validate_options_keeps_generation_settings_and_resolves_lora(self):
        result = validate_options(
            SdGenerationOptions(
                prompt="  1girl  ",
                negative_prompt="bad anatomy",
                model_key="animagine",
                loras=resolve_lora_selections(character_key="character", model_family="sdxl"),
                width=768,
                height=1024,
                steps=28,
                cfg=5.5,
                sampler="euler_ancestral",
                seed=1234,
            )
        )
        self.assertEqual(result.model_family, "sdxl")
        self.assertEqual(result.width, 768)
        self.assertEqual(result.height, 1024)
        self.assertEqual(result.steps, 28)
        self.assertEqual(result.cfg, 5.5)
        self.assertEqual(result.sampler, "euler_ancestral")
        self.assertEqual(result.seed, 1234)
        self.assertEqual(effective_lora_selections(result)[0].key, "character")

    def test_workflow_selection_is_lora_only_and_reference_combination_is_explicitly_rejected(self):
        self.assertEqual(_get_comfy_workflow_file(False), ROOT / "workflows" / "comfy_workflow.json")
        self.assertEqual(
            _get_comfy_workflow_file(False, (LoraSelection("x", "x", "style", "x", "", 0.5),)),
            ROOT / "workflows" / "comfy_workflow_lora.json",
        )
        with self.assertRaisesRegex(ValueError, "同時適用"):
            _get_comfy_workflow_file(True, (LoraSelection("x", "x", "style", "x", "", 0.5),))

    def test_workflow_load_does_not_depend_on_process_cwd(self):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                workflow = _load_comfy_workflow()
            finally:
                os.chdir(original_cwd)

        self.assertTrue(workflow)


if __name__ == "__main__":
    unittest.main()
