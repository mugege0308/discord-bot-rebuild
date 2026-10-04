from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aiohttp.test_utils import TestClient, TestServer

from services.danbooru_tag_search import clear_tag_translation_cache, search_tag_candidates
from services.danbooru_tag_store import DanbooruTagStore, create_schema
from services.generation_api import create_generation_app


class DanbooruTagSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_tag_translation_cache()
        self.temporary = TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "tags.sqlite3"
        connection = sqlite3.connect(self.db_path)
        create_schema(connection)
        connection.executemany(
            """
            INSERT INTO tags(tag_id, name, normalized_name, prompt_tag, category, post_count, is_deprecated)
            VALUES (?, ?, ?, ?, ?, ?, 0)
            """,
            [
                (1, "smile", "smile", "smile", "general", 1000),
                (2, "blue_eyes", "blue_eyes", "blue eyes", "general", 900),
                (3, "hand_on_own_cheek", "hand_on_own_cheek", "hand on own cheek", "general", 800),
                (4, "artist_name", "artist_name", "artist name", "artist", 9999),
                (5, "sanae_kochiya", "sanae_kochiya", "sanae kochiya", "character", 700),
                (6, "sanae_cosplay", "sanae_cosplay", "sanae cosplay", "general", 600),
                (7, "touhou", "touhou", "touhou", "copyright", 500),
            ],
        )
        connection.execute(
            "INSERT INTO tag_aliases(antecedent_name, normalized_antecedent, consequent_name, normalized_consequent, status) VALUES (?, ?, ?, ?, 'active')",
            ("grinning", "grinning", "smile", "smile"),
        )
        connection.executemany(
            "INSERT INTO tags_fts(tag_id, name, prompt_tag, aliases) VALUES (?, ?, ?, ?)",
            [
                (1, "smile", "smile", "grinning"),
                (2, "blue_eyes", "blue eyes", ""),
                (5, "sanae_kochiya", "sanae kochiya", ""),
                (6, "sanae_cosplay", "sanae cosplay", ""),
                (7, "touhou", "touhou", ""),
            ],
        )
        connection.commit()
        connection.close()
        self.store = DanbooruTagStore(self.db_path)

    def tearDown(self) -> None:
        clear_tag_translation_cache()
        self.temporary.cleanup()

    def test_english_search_uses_canonical_general_tags_only(self) -> None:
        result = search_tag_candidates("smile", store=self.store)
        self.assertEqual(["smile"], [item["tag"] for item in result["results"]])
        self.assertEqual("canonical", result["results"][0]["matched_by"])
        self.assertNotIn("artist_name", {item["tag"] for item in result["results"]})

    def test_alias_returns_canonical_tag_only(self) -> None:
        result = search_tag_candidates("grinning", store=self.store)
        self.assertEqual(["smile"], [item["tag"] for item in result["results"]])
        self.assertEqual("alias", result["results"][0]["matched_by"])

    def test_japanese_terms_are_db_validated_and_hallucinated_terms_are_hidden(self) -> None:
        def interpreter(*_args, **_kwargs) -> str:
            return json.dumps({"terms": ["笑顔", "smile", "invented_tag"]}, ensure_ascii=False)

        result = search_tag_candidates("笑顔", store=self.store, interpreter=interpreter)
        self.assertEqual(["smile"], [item["tag"] for item in result["results"]])
        self.assertEqual(["smile"], result["interpreted_terms"])
        self.assertNotIn("invented_tag", json.dumps(result))

    def test_translation_failure_is_safe_empty_result(self) -> None:
        def interpreter(*_args, **_kwargs) -> str:
            raise RuntimeError("LM Studio unavailable")

        result = search_tag_candidates("笑顔", store=self.store, interpreter=interpreter)
        self.assertEqual([], result["results"])
        self.assertTrue(result["warning"])

    def test_category_filter_and_character_search_exclude_general_cosplay(self) -> None:
        character = search_tag_candidates("sanae", category="character", store=self.store, limit=10)
        self.assertEqual(["sanae_kochiya"], [item["tag"] for item in character["results"]])
        self.assertTrue(all(item["category"] == "character" for item in character["results"]))
        general = search_tag_candidates("sanae", category="general", store=self.store, limit=10)
        self.assertEqual(["sanae_cosplay"], [item["tag"] for item in general["results"]])
        all_categories = search_tag_candidates("sanae", category="all", store=self.store, limit=10)
        self.assertEqual({"character", "general"}, {item["category"] for item in all_categories["results"]})

    def test_invalid_category_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            search_tag_candidates("smile", category="artist", store=self.store)

    def test_batch_translation_adds_display_only_label(self) -> None:
        calls = []

        def interpreter(system, user, **_kwargs) -> str:
            calls.append((system, user))
            if "Translate each supplied" in system:
                return json.dumps({"smile": "笑顔"}, ensure_ascii=False)
            return json.dumps({"terms": ["smile"]})

        result = search_tag_candidates("smile", limit=1, store=self.store, interpreter=interpreter)
        self.assertEqual("smile", result["results"][0]["tag"])
        self.assertEqual("笑顔", result["results"][0]["translation_ja"])
        self.assertEqual(1, sum("Translate each supplied" in system for system, _ in calls))

    def test_translation_cache_avoids_second_lm_request(self) -> None:
        calls = 0

        def interpreter(system, _user, **_kwargs) -> str:
            nonlocal calls
            calls += 1
            if "Translate each supplied" in system:
                return '{"smile":"笑顔"}'
            return '{"terms":["smile"]}'

        first = search_tag_candidates("smile", limit=1, store=self.store, interpreter=interpreter)
        second = search_tag_candidates("smile", limit=1, store=self.store, interpreter=interpreter)
        self.assertEqual("笑顔", first["results"][0]["translation_ja"])
        self.assertEqual("笑顔", second["results"][0]["translation_ja"])
        self.assertEqual(1, calls)

    def test_translation_failure_preserves_results(self) -> None:
        result = search_tag_candidates(
            "smile",
            limit=1,
            store=self.store,
            interpreter=lambda *_args, **_kwargs: "not json",
        )
        self.assertEqual("smile", result["results"][0]["tag"])
        self.assertIsNone(result["results"][0]["translation_ja"])
        self.assertTrue(result["warning"])

    def test_japanese_interpretation_prompt_preserves_meaning_elements(self) -> None:
        captured = {}

        def interpreter(system, _query, **_kwargs) -> str:
            captured.setdefault("system", system)
            if "Translate each supplied" in system:
                return '{"hand_on_own_cheek":"頬に手"}'
            return '{"terms":["hand on cheek","hand","cheek"]}'

        result = search_tag_candidates("頬に手", store=self.store, interpreter=interpreter, limit=1)
        self.assertIn("Preserve meaning elements", captured["system"])
        self.assertIn("hand", result["interpreted_terms"])
        self.assertEqual("general", result["category"])

    def test_empty_and_injection_queries_are_safe(self) -> None:
        self.assertEqual([], search_tag_candidates("", store=self.store)["results"])
        result = search_tag_candidates('" OR 1=1 --', store=self.store)
        self.assertEqual([], result["results"])

    def test_long_query_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            search_tag_candidates("x" * 201, store=self.store)

    def test_missing_database_raises_without_exposing_path_in_api_contract(self) -> None:
        missing = DanbooruTagStore(Path(self.temporary.name) / "missing.sqlite3")
        with self.assertRaises(FileNotFoundError):
            search_tag_candidates("smile", store=missing)


class GenerationTagSearchApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = TemporaryDirectory()
        db_path = Path(self.temporary.name) / "tags.sqlite3"
        connection = sqlite3.connect(db_path)
        create_schema(connection)
        connection.execute(
            "INSERT INTO tags(tag_id, name, normalized_name, prompt_tag, category, post_count, is_deprecated) VALUES (1, 'smile', 'smile', 'smile', 'general', 100, 0)"
        )
        connection.commit()
        connection.close()
        self.app = create_generation_app(
            service=object(),
            tag_store=DanbooruTagStore(db_path),
            tag_interpreter=lambda *_args, **_kwargs: '{"terms":["smile"]}',
        )
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()
        self.temporary.cleanup()

    async def test_search_endpoint_returns_safe_canonical_result(self) -> None:
        response = await self.client.get("/api/generation/tags/search?q=smile&limit=20")
        self.assertEqual(200, response.status)
        payload = await response.json()
        self.assertEqual("smile", payload["results"][0]["tag"])
        self.assertNotIn("path", json.dumps(payload).casefold())

    async def test_japanese_search_and_empty_query(self) -> None:
        response = await self.client.get("/api/generation/tags/search?q=%E7%AC%91%E9%A1%94")
        self.assertEqual(200, response.status)
        self.assertEqual(["smile"], (await response.json())["interpreted_terms"])
        empty = await self.client.get("/api/generation/tags/search")
        self.assertEqual(200, empty.status)
        self.assertEqual([], (await empty.json())["results"])

    async def test_category_filter_and_invalid_category(self) -> None:
        character = await self.client.get("/api/generation/tags/search?q=smile&category=character")
        self.assertEqual(200, character.status)
        self.assertEqual("character", (await character.json())["category"])
        invalid = await self.client.get("/api/generation/tags/search?q=smile&category=artist")
        self.assertEqual(400, invalid.status)

    async def test_query_limit_and_length_are_validated(self) -> None:
        invalid_limit = await self.client.get("/api/generation/tags/search?q=smile&limit=0")
        self.assertEqual(400, invalid_limit.status)
        long_query = await self.client.get("/api/generation/tags/search?q=" + ("x" * 201))
        self.assertEqual(400, long_query.status)

    async def test_database_unavailable_is_sanitized(self) -> None:
        missing_app = create_generation_app(
            service=object(),
            tag_store=DanbooruTagStore(Path(self.temporary.name) / "missing.sqlite3"),
        )
        missing_client = TestClient(TestServer(missing_app))
        await missing_client.start_server()
        try:
            response = await missing_client.get("/api/generation/tags/search?q=smile")
            self.assertEqual(503, response.status)
            payload = await response.json()
            self.assertEqual("タグ検索を利用できません。", payload["error"])
            self.assertNotIn("missing.sqlite3", json.dumps(payload))
        finally:
            await missing_client.close()


if __name__ == "__main__":
    unittest.main()
