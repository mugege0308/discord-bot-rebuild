from __future__ import annotations

import csv
import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from services.danbooru_ja_dictionary import DanbooruJapaneseDictionary
from services.danbooru_tag_search import search_tag_candidates
from services.danbooru_tag_store import DanbooruTagStore, create_schema


class CharacterTagSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        root = Path(self.temporary.name)
        self.db_path = root / "tags.sqlite3"
        connection = sqlite3.connect(self.db_path)
        create_schema(connection)
        rows = [
            (1, "izayoi_sakuya", "izayoi_sakuya", "izayoi sakuya", "character", 10000),
            (2, "kochiya_sanae", "kochiya_sanae", "kochiya sanae", "character", 9000),
            (3, "hakurei_reimu", "hakurei_reimu", "hakurei reimu", "character", 8000),
            (4, "kirisame_marisa", "kirisame_marisa", "kirisame marisa", "character", 7000),
            (5, "aizawa_sakuya", "aizawa_sakuya", "aizawa sakuya", "character", 100),
            (6, "ootori_sakuya", "ootori_sakuya", "ootori sakuya", "character", 90),
            (7, "sakuya_cosplay", "sakuya_cosplay", "sakuya cosplay", "general", 50000),
        ]
        connection.executemany(
            "INSERT INTO tags(tag_id, name, normalized_name, prompt_tag, category, post_count, is_deprecated) VALUES (?, ?, ?, ?, ?, ?, 0)",
            rows,
        )
        connection.executemany(
            "INSERT INTO tags_fts(tag_id, name, prompt_tag, aliases) VALUES (?, ?, ?, '')",
            [(row[0], row[1], row[3]) for row in rows],
        )
        connection.commit()
        connection.close()
        self.store = DanbooruTagStore(self.db_path)
        self.dictionary = DanbooruJapaneseDictionary(root / "missing.csv")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _interpreter(self, system: str, query: str, **_kwargs: object) -> str:
        if "Translate each supplied" in system:
            tags = json.loads(query)
            return json.dumps({tag: tag for tag in tags}, ensure_ascii=False)
        variants = {
            "咲夜": {
                "reading_hiragana": ["さくや"],
                "reading_katakana": ["サクヤ"],
                "romanized_tokens": ["sakuya"],
                "possible_full_romanized_names": ["izayoi_sakuya"],
            },
            "さくや": {
                "reading_hiragana": ["さくや"],
                "reading_katakana": ["サクヤ"],
                "romanized_tokens": ["sakuya"],
                "possible_full_romanized_names": ["izayoi_sakuya"],
            },
            "早苗": {
                "reading_hiragana": ["さなえ"],
                "reading_katakana": ["サナエ"],
                "romanized_tokens": ["sanae"],
                "possible_full_romanized_names": ["kochiya_sanae"],
            },
            "さなえ": {
                "reading_hiragana": ["さなえ"],
                "reading_katakana": ["サナエ"],
                "romanized_tokens": ["sanae"],
                "possible_full_romanized_names": ["kochiya_sanae"],
            },
            "霊夢": {
                "reading_hiragana": ["れいむ"],
                "reading_katakana": ["レイム"],
                "romanized_tokens": ["reimu"],
                "possible_full_romanized_names": ["hakurei_reimu"],
            },
            "魔理沙": {
                "reading_hiragana": ["まりさ"],
                "reading_katakana": ["マリサ"],
                "romanized_tokens": ["marisa"],
                "possible_full_romanized_names": ["kirisame_marisa"],
            },
        }
        return json.dumps(variants.get(query, {}), ensure_ascii=False)

    def test_kanji_queries_use_reading_and_character_token_matching(self) -> None:
        expected = {
            "咲夜": ("sakuya", "さくや", "izayoi_sakuya"),
            "早苗": ("sanae", "さなえ", "kochiya_sanae"),
            "霊夢": ("reimu", "れいむ", "hakurei_reimu"),
            "魔理沙": ("marisa", "まりさ", "kirisame_marisa"),
        }
        for query, (token, reading, canonical) in expected.items():
            with self.subTest(query=query):
                result = search_tag_candidates(
                    query,
                    category="character",
                    store=self.store,
                    dictionary=self.dictionary,
                    interpreter=self._interpreter,
                )
                self.assertTrue(result["lm_used"])
                self.assertEqual([reading], result["interpreted_variants"]["reading_hiragana"])
                self.assertIn(token, result["interpreted_variants"]["romanized_tokens"])
                self.assertEqual(canonical, result["results"][0]["tag"])
                self.assertTrue(all(item["category"] == "character" for item in result["results"]))
                self.assertNotIn("sakuya_cosplay", {item["tag"] for item in result["results"]})

    def test_kana_romanized_and_full_canonical_queries(self) -> None:
        for query, canonical in (
            ("さくや", "izayoi_sakuya"),
            ("さなえ", "kochiya_sanae"),
            ("sakuya", "izayoi_sakuya"),
            ("sanae", "kochiya_sanae"),
            ("reimu", "hakurei_reimu"),
            ("marisa", "kirisame_marisa"),
            ("izayoi_sakuya", "izayoi_sakuya"),
            ("kochiya_sanae", "kochiya_sanae"),
        ):
            with self.subTest(query=query):
                result = search_tag_candidates(
                    query,
                    category="character",
                    store=self.store,
                    dictionary=self.dictionary,
                    interpreter=self._interpreter,
                )
                self.assertEqual(canonical, result["results"][0]["tag"])
                if query in {"sakuya", "sanae", "reimu", "marisa"}:
                    self.assertFalse(result["lm_used"])

    def test_ascii_character_search_uses_token_boundaries(self) -> None:
        for query, canonical in (("sakuya", "izayoi_sakuya"), ("sanae", "kochiya_sanae")):
            with self.subTest(query=query):
                result = search_tag_candidates(
                    query,
                    category="character",
                    store=self.store,
                    dictionary=self.dictionary,
                    interpreter=self._interpreter,
                )
                self.assertEqual(canonical, result["results"][0]["tag"])
                self.assertFalse(result["lm_used"])

        empty = search_tag_candidates(
            "ana",
            category="character",
            store=self.store,
            dictionary=self.dictionary,
            interpreter=self._interpreter,
        )
        self.assertEqual([], empty["results"])
        self.assertFalse(empty["lm_used"])

    def test_character_token_matching_does_not_use_substring_fuzziness(self) -> None:
        self.assertEqual((), self.store.search_character_tokens(["ana"], limit=20))

    def test_hallucinated_full_name_is_filtered_by_the_canonical_store(self) -> None:
        def hallucinating(_system: str, _query: str, **_kwargs: object) -> str:
            return json.dumps(
                {
                    "romanized_tokens": ["invented"],
                    "possible_full_romanized_names": ["invented_character"],
                }
            )

        result = search_tag_candidates(
            "未知漢字",
            category="character",
            store=self.store,
            dictionary=self.dictionary,
            interpreter=hallucinating,
        )
        self.assertEqual([], result["results"])

    def test_lm_timeout_returns_safe_empty_result(self) -> None:
        def unavailable(*_args: object, **_kwargs: object) -> str:
            raise TimeoutError("LM unavailable")

        result = search_tag_candidates(
            "未知漢字",
            category="character",
            store=self.store,
            dictionary=self.dictionary,
            interpreter=unavailable,
        )
        self.assertEqual([], result["results"])
        self.assertTrue(result["warning"])

    def test_dictionary_matches_can_expand_shared_character_token_without_lm(self) -> None:
        root = Path(self.temporary.name)
        dictionary_path = root / "dictionary.csv"
        with dictionary_path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(
                [
                    ["aizawa_sakuya", 4, 100, "", "咲夜"],
                    ["ootori_sakuya", 4, 90, "", "咲夜"],
                ]
            )
        dictionary = DanbooruJapaneseDictionary(dictionary_path)
        calls: list[bool] = []

        def unexpected(*_args: object, **_kwargs: object) -> str:
            calls.append(True)
            return "{}"

        result = search_tag_candidates(
            "咲夜",
            category="character",
            store=self.store,
            dictionary=dictionary,
            interpreter=unexpected,
        )
        self.assertFalse(calls)
        self.assertEqual("izayoi_sakuya", result["results"][0]["tag"])
        self.assertFalse(result["lm_used"])


if __name__ == "__main__":
    unittest.main()
