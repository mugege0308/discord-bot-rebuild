import tempfile
import unittest
from pathlib import Path

from services.danbooru_tag_importer import write_snapshot
from services.danbooru_tag_store import DanbooruTagStore
from services.verified_character_registry import VerifiedCharacterRegistry


class VerifiedCharacterRegistryTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "tags.sqlite3"
        self.registry_path = Path(self.tempdir.name) / "character_registry.sqlite3"
        write_snapshot(
            self.db_path,
            [
                {"id": 1, "name": "kochiya_sanae", "category": 4, "post_count": 36072},
                {"id": 2, "name": "touhou", "category": 3, "post_count": 911841},
                {"id": 3, "name": "not_a_character", "category": 0, "post_count": 9999},
            ],
            snapshot_version="registry-test",
            source="unit-test",
        )
        self.store = DanbooruTagStore(self.db_path)
        self.registry = VerifiedCharacterRegistry(self.registry_path)
        self.candidate = self.store.search(
            "kochiya sanae",
            limit=1,
            allowed_categories=("character",),
        )[0]

    def tearDown(self):
        self.tempdir.cleanup()

    def test_normalized_aliases_share_one_verified_character(self):
        self.registry.register_verified_candidate(
            self.candidate,
            input_alias="東風谷早苗",
            copyright_tag="touhou",
            store=self.store,
        )
        self.assertEqual(
            self.registry.lookup("Kochiya_Sanaeが笑っている")[0].character.canonical_danbooru_tag,
            "kochiya_sanae",
        )
        self.assertEqual(
            self.registry.lookup("KOCHIYA SANAE")[0].character.canonical_danbooru_tag,
            "kochiya_sanae",
        )

    def test_short_alias_requires_registered_context(self):
        self.registry.register_verified_candidate(
            self.candidate,
            input_alias="早苗",
            copyright_tag="touhou",
            context_aliases=("東方", "touhou"),
            store=self.store,
        )
        self.assertEqual(self.registry.lookup("早苗が笑っている"), ())
        matches = self.registry.lookup("東方の早苗が笑っている")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].character.canonical_danbooru_tag, "kochiya_sanae")
        self.assertEqual(self.registry.lookup("東方の別キャラ"), ())

    def test_registry_never_saves_non_character_candidate(self):
        general = self.store.search(
            "not a character",
            limit=1,
            allowed_categories=("general",),
        )[0]
        with self.assertRaises(ValueError):
            self.registry.register_verified_candidate(
                general,
                input_alias="東風谷早苗",
                store=self.store,
            )


if __name__ == "__main__":
    unittest.main()
