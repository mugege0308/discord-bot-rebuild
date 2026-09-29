import json
import tempfile
import unittest
from pathlib import Path

from services.lora_training import (
    LORA_TYPE_CHARACTER,
    LORA_TYPE_STYLE,
    LoraTrainingError,
    apply_lora_captions,
    build_caption_with_trigger,
    has_pending_lora_training_state,
    load_lora_metadata,
    load_lora_registration_defaults,
    metadata_path_for_lora,
    preview_lora_captions,
    resolve_lora_file,
    save_lora_metadata,
    training_state_path_for_lora,
    validate_training_metadata,
)


class LoraCaptionPreparationTestCase(unittest.TestCase):
    def test_trigger_is_added_at_caption_start_and_existing_auto_tags_remain(self):
        self.assertEqual(
            build_caption_with_trigger("1girl, auto blue eyes", "hkmmeruru"),
            "hkmmeruru, 1girl, auto blue eyes",
        )

    def test_existing_trigger_is_not_duplicated_case_insensitively(self):
        self.assertEqual(
            build_caption_with_trigger("1girl, HKMMERURU, smile", "hkmmeruru"),
            "1girl, HKMMERURU, smile",
        )

    def test_missing_caption_is_created_for_character_and_existing_caption_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.png").write_bytes(b"image")
            (root / "two.png").write_bytes(b"image")
            (root / "two.txt").write_text("1girl, auto tag\n", encoding="utf-8")

            preview = preview_lora_captions(
                root,
                lora_type=LORA_TYPE_CHARACTER,
                trigger_tag="hkmmeruru",
            )
            self.assertEqual(
                [item.caption for item in preview.items],
                ["hkmmeruru", "hkmmeruru, 1girl, auto tag"],
            )
            self.assertEqual(apply_lora_captions(preview), 2)
            self.assertEqual((root / "one.txt").read_text(encoding="utf-8"), "hkmmeruru\n")
            self.assertEqual(
                (root / "two.txt").read_text(encoding="utf-8"),
                "hkmmeruru, 1girl, auto tag\n",
            )

    def test_style_does_not_require_trigger_and_missing_caption_becomes_empty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.png").write_bytes(b"image")
            preview = preview_lora_captions(root, lora_type=LORA_TYPE_STYLE)
            self.assertEqual(preview.items[0].caption, "")
            self.assertEqual(apply_lora_captions(preview), 1)
            self.assertEqual((root / "one.txt").read_text(encoding="utf-8"), "")

    def test_character_trigger_is_required_but_style_trigger_is_optional(self):
        with self.assertRaisesRegex(ValueError, "トリガータグが必須"):
            validate_training_metadata(name="char", lora_type=LORA_TYPE_CHARACTER)
        metadata = validate_training_metadata(name="style", lora_type=LORA_TYPE_STYLE)
        self.assertEqual(metadata.trigger_tag, "")


class LoraMetadataPersistenceTestCase(unittest.TestCase):
    def test_pending_training_state_is_used_then_cleared_after_metadata_save(self):
        with tempfile.TemporaryDirectory() as directory:
            lora_path = Path(directory) / "hkmmeruru.safetensors"
            lora_path.write_bytes(b"model")
            state_path = training_state_path_for_lora(lora_path)
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "success",
                        "metadata_registered": False,
                        "lora_name": "hkmmeruru",
                        "lora_metadata": {
                            "lora_type": "character",
                            "trigger_tag": "hkmmeruru",
                            "checkpoint": "animagineXL40Opt_clear.safetensors",
                            "model_family": "sdxl",
                            "recommended_strength": 0.8,
                        },
                    }
                ),
                encoding="utf-8",
            )
            defaults, source = load_lora_registration_defaults(lora_path)
            self.assertEqual(source, "training")
            self.assertEqual(defaults.lora_type, "character")
            self.assertEqual(defaults.trigger_tag, "hkmmeruru")
            self.assertEqual(defaults.model_family, "sdxl")
            self.assertEqual(defaults.recommended_strength, 0.8)
            self.assertTrue(has_pending_lora_training_state(lora_path))
            save_lora_metadata(lora_path, defaults)
            self.assertFalse(state_path.exists())
            self.assertFalse(has_pending_lora_training_state(lora_path))
            self.assertEqual(load_lora_metadata(lora_path), defaults)

    def test_metadata_round_trip_and_legacy_missing_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            lora_path = Path(directory) / "hkmmeruru.safetensors"
            lora_path.write_bytes(b"model")
            metadata = validate_training_metadata(
                name="Hkmmeruru",
                lora_type=LORA_TYPE_CHARACTER,
                trigger_tag="hkmmeruru",
                checkpoint="animagineXL40Opt_clear.safetensors",
                model_family="sdxl",
                recommended_strength=0.8,
            )
            path = save_lora_metadata(lora_path, metadata)
            self.assertEqual(path, metadata_path_for_lora(lora_path))
            self.assertEqual(load_lora_metadata(lora_path), metadata)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["lora_type"], "character")
            self.assertIsNone(load_lora_metadata(Path(directory) / "old.safetensors"))

    def test_filename_only_resolves_from_manager_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lora_path = root / "メルル2" / "hikami_meruru_character_v1.safetensors"
            lora_path.parent.mkdir(parents=True)
            lora_path.write_bytes(b"model")
            self.assertEqual(resolve_lora_file(lora_path.name, search_root=root), lora_path.resolve())

    def test_ambiguous_filename_and_missing_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for parent in ("dataset_a", "dataset_b"):
                path = root / parent / "same.safetensors"
                path.parent.mkdir(parents=True)
                path.write_bytes(b"model")
            with self.assertRaisesRegex(LoraTrainingError, "複数"):
                resolve_lora_file("same.safetensors", search_root=root)
            with self.assertRaisesRegex(LoraTrainingError, "見つかりません"):
                resolve_lora_file("missing.safetensors", search_root=root)

    def test_metadata_save_requires_existing_lora_and_never_overwrites_different_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            lora_path = Path(directory) / "hkmmeruru.safetensors"
            metadata_path = metadata_path_for_lora(lora_path)
            metadata = validate_training_metadata(
                name="Hkmmeruru", lora_type=LORA_TYPE_CHARACTER, trigger_tag="hkmmeruru"
            )
            with self.assertRaisesRegex(LoraTrainingError, "見つかりません"):
                save_lora_metadata(lora_path, metadata)
            lora_path.write_bytes(b"model")
            metadata_path.write_text('{"name": "other"}\n', encoding="utf-8")
            with self.assertRaisesRegex(LoraTrainingError, "異なる"):
                save_lora_metadata(lora_path, metadata)
            self.assertEqual(metadata_path.read_text(encoding="utf-8"), '{"name": "other"}\n')


if __name__ == "__main__":
    unittest.main()
