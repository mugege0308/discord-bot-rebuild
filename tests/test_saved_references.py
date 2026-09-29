import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services import saved_references


class SavedReferencesServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.db_file = root / "history.db"
        self.store = root / "saved_references"
        self.patches = mock.patch.multiple(
            saved_references,
            DB_FILE=self.db_file,
            SAVED_REFERENCES_DIR=self.store,
        )
        self.patches.start()

    def tearDown(self) -> None:
        self.patches.stop()
        self.temp_dir.cleanup()

    def test_save_list_owner_isolation_and_metadata_round_trip(self):
        item = saved_references.save_reference(
            user_id=10,
            name=" pose sample ",
            kind="pose",
            tags="kneeling, kneeling, arms",
            note="test",
            image_bytes=b"png-bytes",
            extension="png",
            metadata={"prompt": "positive", "negative_prompt": "negative", "seed": 42, "model": "animagine", "loras": [{"key": "x"}]},
        )

        self.assertEqual(item.user_id, 10)
        self.assertEqual(item.kind, "pose")
        self.assertEqual(item.tags, ("kneeling", "arms"))
        self.assertEqual(item.prompt, "positive")
        self.assertEqual(item.seed, 42)
        self.assertEqual(saved_references.read_reference_bytes(item), b"png-bytes")
        self.assertEqual([item.id], [value.id for value in saved_references.list_references(10)])
        self.assertEqual([], saved_references.list_references(11))
        self.assertEqual([], saved_references.list_references(10, kind="reference"))

        # A fresh connection sees the same data; no in-memory-only registry is used.
        connection = sqlite3.connect(self.db_file)
        try:
            self.assertEqual(1, connection.execute("SELECT COUNT(*) FROM saved_references").fetchone()[0])
        finally:
            connection.close()

    def test_random_and_delete_remove_only_owned_record(self):
        first = saved_references.save_reference(user_id=1, name="one", kind="reference", image_bytes=b"one")
        second = saved_references.save_reference(user_id=2, name="two", kind="reference", image_bytes=b"two")

        with mock.patch("random.choice", return_value=first):
            self.assertEqual(first.id, saved_references.random_reference(1).id)
        self.assertFalse(saved_references.delete_reference(1, second.id))
        self.assertTrue(saved_references.delete_reference(1, first.id))
        self.assertFalse(first.image_path.exists())
        self.assertIsNotNone(saved_references.get_reference(2, second.id))
        self.assertTrue(second.image_path.exists())

    def test_invalid_kind_and_missing_or_corrupt_file_fail_safely(self):
        with self.assertRaises(ValueError):
            saved_references.save_reference(user_id=1, name="x", kind="other", image_bytes=b"x")

        item = saved_references.save_reference(user_id=1, name="x", kind="reference", image_bytes=b"x")
        item.image_path.unlink()
        with self.assertRaises(RuntimeError):
            saved_references.read_reference_bytes(item)

    def test_limit_and_tag_filter_are_owner_scoped(self):
        saved_references.save_reference(user_id=1, name="a", kind="reference", tags="Beach", image_bytes=b"a")
        saved_references.save_reference(user_id=1, name="b", kind="reference", tags="night", image_bytes=b"b")
        self.assertEqual(["a"], [item.name for item in saved_references.list_references(1, tag="beach")])
        self.assertEqual(1, len(saved_references.list_references(1, limit=1)))
        self.assertIsNone(saved_references.random_reference(99))


if __name__ == "__main__":
    unittest.main()
