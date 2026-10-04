from __future__ import annotations

import csv
import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from services.danbooru_ja_dictionary import (
    DictionaryFormatError,
    DanbooruJapaneseDictionary,
    parse_dictionary_file,
    parse_dictionary_rows,
)
from services.danbooru_tag_search import clear_tag_translation_cache, search_tag_candidates
from services.danbooru_tag_store import DanbooruTagStore, create_schema


class DanbooruJapaneseDictionaryTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_tag_translation_cache()
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.csv_path = self.root / "danbooru-ja.csv"
        connection = sqlite3.connect(self.root / "tags.sqlite3")
        create_schema(connection)
        rows = [
            (1, "hatsune_miku", "hatsune_miku", "hatsune miku", "character", 1000),
            (2, "touhou", "touhou", "touhou", "copyright", 900),
            (3, "black_hair", "black_hair", "black hair", "general", 800),
            (4, "long_hair", "long_hair", "long hair", "general", 700),
            (5, "glasses", "glasses", "glasses", "general", 600),
            (6, "smile", "smile", "smile", "general", 500),
        ]
        connection.executemany(
            "INSERT INTO tags(tag_id, name, normalized_name, prompt_tag, category, post_count, is_deprecated) VALUES (?, ?, ?, ?, ?, ?, 0)",
            rows,
        )
        connection.executemany(
            "INSERT INTO tags_fts(tag_id, name, prompt_tag, aliases) VALUES (?, ?, ?, ?)",
            [(row[0], row[1], row[3], "") for row in rows],
        )
        connection.commit()
        connection.close()
        self.store = DanbooruTagStore(self.root / "tags.sqlite3")
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(
                [
                    ["hatsune miku", 4, 1000, "miku", "初音ミク"],
                    ["touhou", 3, 900, "", "東方プロジェクト"],
                    ["black hair", 0, 800, "", "黒髪"],
                    ["long hair", 0, 700, "hair past waist", "ロングヘアー"],
                    ["glasses", 0, 600, "eyeglasses", "眼鏡"],
                    ["smile", 0, 500, "", "笑顔"],
                ]
            )

    def tearDown(self) -> None:
        clear_tag_translation_cache()
        self.temporary.cleanup()

    def test_csv_parser_accepts_five_columns_and_aliases(self) -> None:
        entries = parse_dictionary_file(self.csv_path)
        self.assertEqual(6, len(entries))
        self.assertEqual("character", entries[0].category)
        self.assertEqual(("hair past waist",), entries[3].aliases)

    def test_malformed_rows_and_missing_columns_are_rejected(self) -> None:
        with self.assertRaises(DictionaryFormatError):
            parse_dictionary_rows([["tag", "category", "count"]])
        with self.assertRaises(DictionaryFormatError):
            parse_dictionary_rows([["tag", "9", "1", "", "タグ"]])
        with self.assertRaises(DictionaryFormatError):
            parse_dictionary_rows([["tag", "0", "-1", "", "タグ"]])

    def test_japanese_exact_prefix_and_alias_hits_are_canonical(self) -> None:
        dictionary = DanbooruJapaneseDictionary(self.csv_path)
        character = dictionary.search("初音ミク", category="character", store=self.store, limit=10)
        self.assertEqual("hatsune_miku", character[0].candidate.name)
        self.assertEqual("初音ミク", character[0].translation_ja)
        general = dictionary.search("ロングヘア", category="general", store=self.store, limit=10)
        self.assertEqual("long_hair", general[0].candidate.name)
        alias = dictionary.search("hair past waist", category="general", store=self.store, limit=10)
        self.assertEqual("long_hair", alias[0].candidate.name)

    def test_search_api_uses_dictionary_alias_when_canonical_store_has_no_alias(self) -> None:
        dictionary = DanbooruJapaneseDictionary(self.csv_path)
        result = search_tag_candidates("hair past waist", store=self.store, dictionary=dictionary, limit=1)
        self.assertEqual("long_hair", result["results"][0]["tag"])
        self.assertEqual("dictionary", result["search_source"])

    def test_category_filter_prevents_cross_category_fallback(self) -> None:
        dictionary = DanbooruJapaneseDictionary(self.csv_path)
        self.assertEqual((), dictionary.search("初音ミク", category="general", store=self.store, limit=10))
        self.assertEqual("touhou", dictionary.search("東方", category="copyright", store=self.store, limit=10)[0].candidate.name)

    def test_missing_canonical_tag_is_not_returned(self) -> None:
        dictionary = DanbooruJapaneseDictionary(self.csv_path)
        self.assertEqual((), dictionary.search("存在しない", category="general", store=self.store, limit=10))
        rows = parse_dictionary_rows([["not_in_db", 0, 10, "", "未登録"]])
        missing_path = self.root / "missing.csv"
        with missing_path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows([[row.tag, 0, row.post_count, "", row.translation_ja or ""] for row in rows])
        self.assertEqual((), DanbooruJapaneseDictionary(missing_path).search("未登録", category="general", store=self.store, limit=10))

    def test_dictionary_hit_does_not_call_lm_and_preserves_english_output(self) -> None:
        dictionary = DanbooruJapaneseDictionary(self.csv_path)
        calls: list[object] = []

        def interpreter(*_args, **_kwargs):
            calls.append(True)
            return '{"terms":["invented"]}'

        result = search_tag_candidates(
            "初音ミク",
            category="character",
            store=self.store,
            dictionary=dictionary,
            interpreter=interpreter,
        )
        self.assertFalse(calls)
        self.assertEqual("dictionary", result["search_source"])
        self.assertFalse(result["lm_used"])
        self.assertEqual("hatsune_miku", result["results"][0]["tag"])
        self.assertEqual("初音ミク", result["results"][0]["translation_ja"])

    def test_dictionary_translation_precedes_cached_or_lm_translation(self) -> None:
        dictionary = DanbooruJapaneseDictionary(self.csv_path)
        result = search_tag_candidates("smile", store=self.store, dictionary=dictionary, interpreter=lambda *_a, **_k: '{"smile":"別訳"}')
        self.assertEqual("笑顔", result["results"][0]["translation_ja"])

    def test_unknown_japanese_uses_lm_only_after_dictionary_miss(self) -> None:
        dictionary = DanbooruJapaneseDictionary(self.csv_path)
        calls: list[str] = []

        def interpreter(system, _user, **_kwargs):
            calls.append(system)
            if "Translate each supplied" in system:
                return '{"smile":"笑顔"}'
            return '{"terms":["smile"]}'

        result = search_tag_candidates("頬に両手", store=self.store, dictionary=dictionary, interpreter=interpreter, limit=1)
        self.assertTrue(calls)
        self.assertEqual("llm", result["search_source"])
        self.assertTrue(result["lm_used"])
        self.assertEqual("smile", result["results"][0]["tag"])


class UpdateScriptContractTests(unittest.TestCase):
    def test_update_failure_does_not_replace_existing_files(self) -> None:
        from scripts.update_danbooru_ja_dictionary import update_dictionary

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "danbooru-ja.csv"
            meta_path = root / "meta.json"
            csv_path.write_text("old\n", encoding="utf-8")
            meta_path.write_text('{"old":true}\n', encoding="utf-8")
            with patch("scripts.update_danbooru_ja_dictionary._download", side_effect=OSError("offline")):
                with self.assertRaises(OSError):
                    update_dictionary(dictionary_path=csv_path, meta_path=meta_path)
            self.assertEqual("old\n", csv_path.read_text(encoding="utf-8"))
            self.assertEqual('{"old":true}\n', meta_path.read_text(encoding="utf-8"))

    def test_invalid_download_does_not_replace_existing_files(self) -> None:
        from scripts.update_danbooru_ja_dictionary import update_dictionary

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "danbooru-ja.csv"
            meta_path = root / "meta.json"
            csv_path.write_text("old\n", encoding="utf-8")
            meta_path.write_text('{"old":true}\n', encoding="utf-8")

            def invalid_download(_url, destination, _max_bytes):
                if destination.name == "danbooru-ja.csv":
                    destination.write_text("bad,row\n", encoding="utf-8")
                else:
                    destination.write_text('{"generated_at":"now"}\n', encoding="utf-8")

            with patch("scripts.update_danbooru_ja_dictionary._download", side_effect=invalid_download):
                with self.assertRaises(DictionaryFormatError):
                    update_dictionary(dictionary_path=csv_path, meta_path=meta_path)
            self.assertEqual("old\n", csv_path.read_text(encoding="utf-8"))
            self.assertEqual('{"old":true}\n', meta_path.read_text(encoding="utf-8"))

    def test_valid_update_replaces_both_files_atomically(self) -> None:
        from scripts.update_danbooru_ja_dictionary import update_dictionary

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "danbooru-ja.csv"
            meta_path = root / "meta.json"
            csv_path.write_text("old\n", encoding="utf-8")
            meta_path.write_text('{"old":true}\n', encoding="utf-8")

            def valid_download(_url, destination, _max_bytes):
                if destination.name == "danbooru-ja.csv":
                    with destination.open("w", encoding="utf-8", newline="") as handle:
                        csv.writer(handle).writerow(["smile", 0, 1, "", "笑顔"])
                else:
                    destination.write_text('{"generated_at":"now"}\n', encoding="utf-8")

            with patch("scripts.update_danbooru_ja_dictionary._download", side_effect=valid_download), patch(
                "scripts.update_danbooru_ja_dictionary.MIN_ROW_COUNT", 1
            ):
                metadata = update_dictionary(dictionary_path=csv_path, meta_path=meta_path)
            self.assertEqual("smile,0,1,,笑顔\n", csv_path.read_text(encoding="utf-8"))
            self.assertEqual(1, metadata["row_count_validated"])
            self.assertIn("upstream_repository", json.loads(meta_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
