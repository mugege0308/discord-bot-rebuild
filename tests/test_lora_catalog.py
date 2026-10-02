import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services.lora_catalog import (
    DEFAULT_STYLE_WORKFLOW,
    build_comfy_styles,
    get_comfy_lora_choice_specs,
    get_comfy_style_choice_specs,
    get_lora_catalog_summary,
    model_family_compatible,
    refresh_comfy_styles,
)
from services.lora_paths import (
    get_comfy_lora_root,
    get_lora_manager_outputs_dir,
    DEFAULT_COMFY_LORA_ROOT,
    DEFAULT_LORA_MANAGER_OUTPUTS_DIR,
    metadata_path_for_lora,
)
from services.lora_training import LORA_TYPE_CHARACTER, LORA_TYPE_STYLE


def _write_model(root: Path, relative: str, metadata: dict | None = None) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"model")
    if metadata is not None:
        metadata_path_for_lora(path).write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )
    return path


class LoraCatalogTestCase(unittest.TestCase):
    def test_built_in_entry_and_recursive_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_model(root, "sample/style_v1.safetensors")
            styles = build_comfy_styles(root)
            self.assertIn("natsume", styles)
            self.assertIn("style_v1", styles)
            self.assertEqual(styles["style_v1"].source_path, path.resolve())
            self.assertEqual(styles["style_v1"].lora_type, LORA_TYPE_STYLE)
            self.assertEqual(styles["style_v1"].default_strength, 0.5)

    def test_metadata_type_trigger_strength_and_family_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_model(
                root,
                "characters/hutao.safetensors",
                {
                    "schema_version": 1,
                    "name": "Hu Tao Character",
                    "lora_type": "character",
                    "trigger_tag": "hu tao",
                    "checkpoint": "animagineXL40Opt_clear.safetensors",
                    "model_family": "sdxl",
                    "recommended_strength": 0.8,
                },
            )
            styles = build_comfy_styles(root)
            entry = styles["hutao"]
            self.assertEqual(entry.lora_type, LORA_TYPE_CHARACTER)
            self.assertEqual(entry.trigger, "hu tao")
            self.assertEqual(entry.default_strength, 0.8)
            self.assertEqual(entry.model_family, "sdxl")
            self.assertEqual(entry.checkpoint, "animagineXL40Opt_clear.safetensors")
            self.assertTrue(entry.metadata_path.is_file())

    def test_missing_or_malformed_metadata_falls_back_to_style_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = _write_model(root, "missing.safetensors")
            malformed = _write_model(root, "broken.safetensors")
            metadata_path_for_lora(malformed).write_text("not json", encoding="utf-8")
            styles = build_comfy_styles(root)
            for path in (missing, malformed):
                entry = styles[path.stem]
                self.assertEqual(entry.lora_type, LORA_TYPE_STYLE)
                self.assertEqual(entry.trigger, "")
                self.assertEqual(entry.default_strength, 0.5)

    def test_pending_training_state_is_not_exposed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_model(root, "pending.safetensors")
            path.with_name(path.name + ".training.json").write_text(
                json.dumps({"metadata_registered": False, "status": "success"}),
                encoding="utf-8",
            )
            styles = build_comfy_styles(root)
            self.assertNotIn("pending", styles)

    def test_duplicate_stems_get_unique_keys_and_display_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _write_model(root, "a/same.safetensors")
            second = _write_model(root, "b/same.safetensors")
            styles = build_comfy_styles(root)
            entries = [value for value in styles.values() if value.source_path is not None]
            self.assertEqual({entry.source_path for entry in entries}, {first.resolve(), second.resolve()})
            self.assertEqual({entry.display_name for entry in entries}, {"a / same", "b / same"})
            self.assertEqual(len({key for key, value in styles.items() if value.source_path is not None}), 2)

    def test_intermediate_training_checkpoint_and_non_model_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_model(root, "model-000002.safetensors")
            (root / "notes.txt").write_text("ignored", encoding="utf-8")
            styles = build_comfy_styles(root)
            self.assertNotIn("model-000002", styles)
            self.assertEqual(len([entry for entry in styles.values() if entry.source_path]), 0)

    def test_shared_comfy_name_and_model_family_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            path = _write_model(
                root,
                "nested/style.safetensors",
                {"lora_type": "style", "name": "Style", "model_family": "sdxl"},
            )
            comfy_root = Path(directory) / "comfy-loras"
            comfy_root.mkdir()
            (comfy_root / path.name).write_bytes(b"shared")
            with mock.patch.dict(os.environ, {"COMFY_LORA_ROOT": str(comfy_root)}, clear=False):
                styles = build_comfy_styles(root)
            self.assertEqual(styles["style"].lora_name, "style.safetensors")
            with mock.patch("services.lora_catalog.COMFY_STYLES", styles):
                self.assertIn("style", {value for _, value in get_comfy_lora_choice_specs("style", "SDXL")})
                self.assertNotIn("style", {value for _, value in get_comfy_lora_choice_specs("style", "sd15")})

    def test_refresh_and_choice_helpers_use_temporary_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_model(root, "one.safetensors")
            old = dict(refresh_comfy_styles(root))
            try:
                self.assertIn("one", {value for _, value in get_comfy_style_choice_specs()})
                summary = get_lora_catalog_summary()
                self.assertGreaterEqual(summary[LORA_TYPE_STYLE], 2)
            finally:
                from services import lora_catalog

                lora_catalog.COMFY_STYLES.clear()
                lora_catalog.COMFY_STYLES.update(old)

    def test_path_defaults_are_environment_aware_and_relative(self):
        with mock.patch.dict(
            os.environ,
            {"LORA_MANAGER_OUTPUTS_DIR": "custom-outputs", "COMFY_LORA_ROOT": "custom-loras"},
            clear=False,
        ):
            self.assertEqual(get_lora_manager_outputs_dir(), Path("custom-outputs"))
            self.assertEqual(get_comfy_lora_root(), Path("custom-loras"))
        self.assertEqual(DEFAULT_STYLE_WORKFLOW, Path("workflows") / "comfy_workflow_lora.json")

    def test_legacy_environment_resolves_project_outputs_and_recognized_comfy_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager_root = root / "manager"
            manager_outputs = manager_root / "outputs"
            manager_outputs.mkdir(parents=True)
            kohya_root = root / "kohya"
            comfy_root = kohya_root / "models" / "Lora"
            comfy_root.mkdir(parents=True)
            with mock.patch.dict(
                os.environ,
                {"LORA_MANAGER_ROOT": str(manager_root), "LORA_KOHYA_ROOT": str(kohya_root)},
                clear=True,
            ):
                self.assertEqual(get_lora_manager_outputs_dir(), manager_outputs)
                self.assertEqual(get_comfy_lora_root(), comfy_root)

    def test_new_environment_takes_precedence_over_legacy_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new_outputs = root / "new-outputs"
            new_outputs.mkdir()
            legacy_outputs = root / "legacy" / "outputs"
            legacy_outputs.mkdir(parents=True)
            new_comfy = root / "new-comfy"
            new_comfy.mkdir()
            legacy_comfy = root / "legacy-kohya" / "models" / "Lora"
            legacy_comfy.mkdir(parents=True)
            with mock.patch.dict(
                os.environ,
                {
                    "LORA_MANAGER_OUTPUTS_DIR": str(new_outputs),
                    "LORA_MANAGER_ROOT": str(legacy_outputs.parent),
                    "COMFY_LORA_ROOT": str(new_comfy),
                    "LORA_KOHYA_ROOT": str(legacy_comfy.parents[1]),
                },
                clear=True,
            ):
                self.assertEqual(get_lora_manager_outputs_dir(), new_outputs)
                self.assertEqual(get_comfy_lora_root(), new_comfy)

    def test_invalid_legacy_paths_fall_back_without_treating_kohya_as_comfy_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid_manager = root / "missing-manager"
            invalid_kohya = root / "missing-kohya"
            with mock.patch.dict(
                os.environ,
                {"LORA_MANAGER_ROOT": str(invalid_manager), "LORA_KOHYA_ROOT": str(invalid_kohya)},
                clear=True,
            ):
                self.assertEqual(get_lora_manager_outputs_dir(), DEFAULT_LORA_MANAGER_OUTPUTS_DIR)
                self.assertEqual(get_comfy_lora_root(), DEFAULT_COMFY_LORA_ROOT)

    def test_legacy_manager_output_catalog_preserves_all_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager_root = root / "manager"
            outputs = manager_root / "outputs"
            _write_model(
                outputs,
                "character/hutao.safetensors",
                {"name": "Hu Tao", "lora_type": "character", "trigger_tag": "hu tao"},
            )
            _write_model(outputs, "style/soft.safetensors", {"lora_type": "style"})
            _write_model(outputs, "additional/detail.safetensors", {"lora_type": "additional"})
            with mock.patch.dict(os.environ, {"LORA_MANAGER_ROOT": str(manager_root)}, clear=True):
                styles = build_comfy_styles()
            self.assertEqual(styles["hutao"].lora_type, "character")
            self.assertEqual(styles["soft"].lora_type, "style")
            self.assertEqual(styles["detail"].lora_type, "additional")

    def test_prompt_and_batch_use_the_shared_catalog_choice_helper(self):
        from commands import generate_batch, prompt
        from services import lora_catalog

        self.assertIs(prompt.get_comfy_lora_choice_specs, lora_catalog.get_comfy_lora_choice_specs)
        self.assertIs(generate_batch.get_comfy_lora_choice_specs, lora_catalog.get_comfy_lora_choice_specs)

    def test_model_family_missing_metadata_is_compatible(self):
        self.assertTrue(model_family_compatible(None, "sdxl"))
        self.assertTrue(model_family_compatible("sdxl", None))
        self.assertTrue(model_family_compatible("SDXL", "sdxl"))
        self.assertFalse(model_family_compatible("sd15", "sdxl"))


if __name__ == "__main__":
    unittest.main()
