from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.image_selection import _load_selected_paths, _selector_environment, _write_selection_diagnostic


class ImageSelectionTests(unittest.TestCase):
    def test_selector_environment_forces_local_model_cache(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            environment = _selector_environment()
        self.assertEqual("1", environment["HF_HUB_OFFLINE"])
        self.assertEqual("1", environment["TRANSFORMERS_OFFLINE"])
        self.assertEqual("1", environment["HF_DATASETS_OFFLINE"])

    def test_selection_failure_diagnostic_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch("services.image_selection.SELECTION_DIAGNOSTIC_DIR", Path(temporary)):
                path = _write_selection_diagnostic(
                    context="overnight-round-2",
                    command=["python", "score_batch_top5.py"],
                    cwd=Path(temporary),
                    batch_dir=Path(temporary) / "batch",
                    returncode=1,
                    stdout="stdout detail",
                    stderr="stderr detail",
                )
            self.assertIsNotNone(path)
            self.assertIn("returncode=1", path.read_text(encoding="utf-8"))
            self.assertIn("stderr detail", path.read_text(encoding="utf-8"))

    def test_bulk_return_count_can_request_four_without_changing_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            output = project / "outputs" / "auto_selected" / "latest_top5"
            output.mkdir(parents=True)
            for index in range(1, 6):
                (output / f"{index:02d}_image.png").write_bytes(b"png")

            self.assertEqual(1, len(_load_selected_paths(project)))
            self.assertEqual(4, len(_load_selected_paths(project, return_count=4)))


if __name__ == "__main__":
    unittest.main()
