import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import gallery_control


class GalleryControlTests(unittest.TestCase):
    def test_images_root_is_read_from_launcher_state(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            images = project / "output"
            images.mkdir()
            gallery_dir = project / ".gallery"
            gallery_dir.mkdir()
            (gallery_dir / "images_root.txt").write_text(str(images), encoding="utf-8")
            with patch.dict("os.environ", {"GALLERY_PROJECT_ROOT": str(project)}):
                self.assertEqual(images.resolve(), gallery_control.get_gallery_images_root())

    def test_start_gallery_waits_for_health_and_does_not_use_shell(self):
        with (
            patch.object(gallery_control, "_gallery_ready", side_effect=[False, True, True]),
            patch.object(gallery_control, "_launch_start_file") as launch,
            patch.object(gallery_control, "get_gallery_status", return_value=gallery_control.GalleryStatus(
                running=True,
                pid=123,
                port=8765,
                urls=gallery_control.GalleryURLs("http://127.0.0.1:8765/"),
            )),
            patch.object(gallery_control, "DEFAULT_READY_POLL_SECONDS", 0),
        ):
            result = gallery_control.start_gallery()

        self.assertFalse(result.already_running)
        self.assertTrue(result.status.running)
        launch.assert_called_once_with()

    def test_gallery_api_rejects_unsuccessful_response(self):
        with patch.object(
            gallery_control,
            "urlopen",
            return_value=type("Response", (), {
                "__enter__": lambda self: self,
                "__exit__": lambda self, *args: None,
                "read": lambda self: b'{"ok": false, "error": "offline"}',
            })(),
        ):
            with self.assertRaises(gallery_control.GalleryControlError):
                gallery_control.get_gallery_ai_status()


if __name__ == "__main__":
    unittest.main()
