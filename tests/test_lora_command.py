import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import discord
from discord import app_commands

from commands.lora import (
    LORA_TYPE_CHOICES,
    _format_plan_preview,
    _resolve_dataset_dir_for_prepare,
    resolve_lora_prepare_metadata,
    setup,
)
from services.lora_training import training_state_path_for_lora


class LoraPrepareCommandTestCase(unittest.TestCase):
    def test_weighted_plan_preview_shows_each_subset(self):
        text = _format_plan_preview(
            {
                "dataset_name": "sutera_v7",
                "output_name": "sutera_v7_character_v1",
                "lora_type": "character",
                "trigger_tag": "sutera_starrail",
                "recommended_strength": 0.7,
                "checkpoint": "animagineXL40Opt_clear.safetensors",
                "model_family": "sdxl",
                "image_count": 12,
                "total_steps": 960,
                "settings": {
                    "subsets": [
                        {"path": "images/general", "image_count": 8, "repeats": 8},
                        {"path": "images/identity_detail", "image_count": 4, "repeats": 24},
                    ],
                    "epochs": 6,
                    "resolution": 1024,
                    "network_dim": 32,
                    "network_alpha": 16,
                    "learning_rate": 0.0001,
                    "optimizer": "AdamW8bit",
                    "scheduler": "cosine",
                    "batch_size": 1,
                },
            }
        )
        self.assertIn("images/general: 8枚 × 8", text)
        self.assertIn("images/identity_detail: 4枚 × 24", text)
        self.assertIn("epochs: 6", text)
        self.assertIn("推奨強度: 0.7", text)

    def test_managed_dataset_name_resolves_dataset_toml_image_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            manager_root = Path(directory) / "manager"
            dataset_root = manager_root / "lora_datasets" / "sutera_v2"
            train_dir = dataset_root / "train"
            plan_dir = dataset_root / "training" / "plans" / "sutera_v2_character_v1"
            train_dir.mkdir(parents=True)
            plan_dir.mkdir(parents=True)
            (train_dir / "0001.webp").write_bytes(b"image")
            (plan_dir / "dataset.toml").write_text(
                "[[datasets]]\n"
                "\n"
                "  [[datasets.subsets]]\n"
                f'  image_dir = "{train_dir.as_posix()}"\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"LORA_MANAGER_ROOT": str(manager_root)}, clear=False):
                resolved = _resolve_dataset_dir_for_prepare("sutera_v2")
        self.assertEqual(resolved, train_dir.resolve())

    def test_managed_dataset_name_keeps_legacy_root_images(self):
        with tempfile.TemporaryDirectory() as directory:
            manager_root = Path(directory) / "manager"
            dataset_root = manager_root / "lora_datasets" / "legacy"
            dataset_root.mkdir(parents=True)
            (dataset_root / "0001.png").write_bytes(b"image")
            with patch.dict(os.environ, {"LORA_MANAGER_ROOT": str(manager_root)}, clear=False):
                resolved = _resolve_dataset_dir_for_prepare("legacy")
        self.assertEqual(resolved, dataset_root.resolve())

    def test_prepare_metadata_uses_saved_training_values_and_actual_config_values(self):
        with tempfile.TemporaryDirectory() as directory:
            lora_path = Path(directory) / "hkmmeruru.safetensors"
            lora_path.write_bytes(b"model")
            training_state_path_for_lora(lora_path).write_text(
                json.dumps(
                    {
                        "status": "success",
                        "metadata_registered": False,
                        "lora_name": "hkmmeruru",
                        "lora_metadata": {
                            "lora_type": "character",
                            "trigger_tag": "hkmmeruru",
                            "recommended_strength": 0.8,
                            "checkpoint": "animagineXL40Opt_clear.safetensors",
                            "model_family": "sdxl",
                        },
                    }
                ),
                encoding="utf-8",
            )

            metadata, source = resolve_lora_prepare_metadata(
                lora_path=lora_path,
                lora_name="Hkmmeruru",
            )

        self.assertEqual(source, "training")
        self.assertEqual(metadata.lora_type, "character")
        self.assertEqual(metadata.trigger_tag, "hkmmeruru")
        self.assertEqual(metadata.recommended_strength, 0.8)
        self.assertEqual(metadata.checkpoint, "animagineXL40Opt_clear.safetensors")
        self.assertEqual(metadata.model_family, "sdxl")

    def test_prepare_command_makes_saved_metadata_fields_optional(self):
        client = discord.Client(intents=discord.Intents.none())
        client.tree = app_commands.CommandTree(client)
        setup(client)

        command = client.tree.get_command("lora_prepare")
        self.assertIsNotNone(command)
        parameters = {parameter.name: parameter for parameter in command.parameters}
        self.assertFalse(parameters["lora_type"].required)
        self.assertFalse(parameters["recommended_strength"].required)
        self.assertNotIn("checkpoint", parameters)
        self.assertNotIn("model_family", parameters)
        self.assertEqual(
            {choice.value for choice in LORA_TYPE_CHOICES},
            {"character", "style", "additional"},
        )


if __name__ == "__main__":
    unittest.main()
