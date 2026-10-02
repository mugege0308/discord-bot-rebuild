import tempfile
import unittest
from pathlib import Path

from services.appearance_resolver import resolve_appearance_text
from services.danbooru_tag_importer import write_snapshot
from services.danbooru_tag_store import DanbooruTagStore


class AppearanceResolverTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "tags.sqlite3"
        write_snapshot(
            self.db_path,
            [
                {"id": 1, "name": "black_hair", "category": 0, "post_count": 1000},
                {"id": 2, "name": "blue_eyes", "category": 0, "post_count": 1000},
                {"id": 3, "name": "silver_hair", "category": 0, "post_count": 1000},
                {"id": 4, "name": "bob_cut", "category": 0, "post_count": 1000},
                {"id": 5, "name": "purple_eyes", "category": 0, "post_count": 1000},
                {"id": 6, "name": "pale_skin", "category": 0, "post_count": 1000},
                {"id": 7, "name": "smile", "category": 0, "post_count": 1000},
                {"id": 8, "name": "artist_only", "category": 1, "post_count": 1000},
            ],
            snapshot_version="appearance-test",
            source="unit-test",
        )
        self.store = DanbooruTagStore(self.db_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_hair_and_eye_colors_resolve_from_active_general_tags(self):
        result = resolve_appearance_text("黒髪、青い目", store=self.store)
        self.assertEqual(("black hair", "blue eyes"), result.tags)
        self.assertEqual((), result.unresolved)

    def test_hair_style_skin_and_eye_resolve_without_request_semantics(self):
        result = resolve_appearance_text("銀髪のボブ、紫の目、色白", store=self.store)
        self.assertEqual(("silver hair", "bob cut", "purple eyes", "pale skin"), result.tags)

    def test_smile_and_background_are_not_appearance_tags(self):
        result = resolve_appearance_text("黒髪、青い目、笑顔で白背景", store=self.store)
        self.assertEqual(("black hair", "blue eyes"), result.tags)
        self.assertNotIn("smile", result.tags)

    def test_unknown_appearance_is_not_invented(self):
        result = resolve_appearance_text("未知の髪質と虹色の瞳", store=self.store)
        self.assertEqual((), result.tags)

    def test_missing_database_fails_closed(self):
        missing = DanbooruTagStore(Path(self.tempdir.name) / "missing.sqlite3")
        result = resolve_appearance_text("黒髪、青い目", store=missing)
        self.assertEqual((), result.tags)
        self.assertEqual(("黒髪、青い目",), result.unresolved)


if __name__ == "__main__":
    unittest.main()
