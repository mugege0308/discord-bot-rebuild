from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import batch_generation, overnight_generation
from services.generation_reference_assets import REFERENCE_IMAGE_ROOT, storage_root
from services.image_selection import BATCH_DIR
from services.runtime_paths import PROJECT_ROOT, resolve_project_path
from services.sd_control import OUTPUT_DIR, POSE_DEBUG_DIR
from services.shogou_prompt_designer import INSTRUCTION_FILE, load_shogou_instruction


class GenerationRuntimePathTests(unittest.TestCase):
    def test_default_runtime_paths_ignore_external_cwd(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as external:
            try:
                os.chdir(external)
                with patch.dict(os.environ, {"BATCH_GENERATION_DB_PATH": ""}, clear=False):
                    expected = (PROJECT_ROOT / "data" / "batch_generation.db").resolve()
                    self.assertEqual(expected, batch_generation._db_path())
                    self.assertEqual(expected, overnight_generation.resolve_overnight_db_path())
            finally:
                os.chdir(original_cwd)

    def test_relative_runtime_override_is_project_relative(self) -> None:
        self.assertEqual(
            (PROJECT_ROOT / "data" / "custom-batch.db").resolve(),
            resolve_project_path("data/custom-batch.db", "data/batch_generation.db"),
        )
        with patch.object(overnight_generation.OvernightJobStore, "_ensure_schema"):
            store = overnight_generation.OvernightJobStore("data/explicit-overnight.db")
        self.assertEqual((PROJECT_ROOT / "data" / "explicit-overnight.db").resolve(), store.db_path)

    def test_generation_runtime_directories_are_project_root_based(self) -> None:
        expected_root = PROJECT_ROOT.resolve()
        for path in (OUTPUT_DIR, POSE_DEBUG_DIR, BATCH_DIR, REFERENCE_IMAGE_ROOT, storage_root()):
            self.assertTrue(path.is_absolute())
            path.resolve().relative_to(expected_root)

    def test_prompt_resource_is_project_root_based(self) -> None:
        self.assertEqual(PROJECT_ROOT.resolve(), INSTRUCTION_FILE.resolve().parent)

    def test_prompt_resource_load_ignores_external_cwd(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as external:
            try:
                os.chdir(external)
                self.assertTrue(load_shogou_instruction())
            finally:
                os.chdir(original_cwd)

    def test_backend_runner_sets_project_cwd(self) -> None:
        runner = (PROJECT_ROOT / "run_generation_backend.ps1").read_text(encoding="utf-8")
        self.assertIn("Set-Location -LiteralPath $ProjectRoot", runner)
        self.assertIn("$serverScript = Join-Path $ProjectRoot", runner)


if __name__ == "__main__":
    unittest.main()
