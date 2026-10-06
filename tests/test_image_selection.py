from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from services.image_selection import _load_selected_paths


class ImageSelectionTests(unittest.TestCase):
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
