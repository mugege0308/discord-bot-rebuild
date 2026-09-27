import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services.danbooru_tag_importer import write_snapshot
from services.danbooru_tag_resolver import (
    apply_danbooru_tag_resolution,
    resolve_danbooru_tags,
)
from services.danbooru_tag_store import DanbooruTagStore
from services.prompt_tag_resolver import resolve_prompt_tags
from services import shogou_prompt_designer


class DanbooruTagMvpTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "tags.sqlite3"
        write_snapshot(
            self.db_path,
            [
                {"id": 1, "name": "oversized_clothes", "category": 0, "post_count": 12000},
                {"id": 2, "name": "baggy_clothes", "category": 0, "post_count": 1600},
                {"id": 3, "name": "long_sleeves", "category": 0, "post_count": 1600000},
                {"id": 4, "name": "short_sleeves", "category": 0, "post_count": 900000},
                {"id": 5, "name": "hoodie", "category": 0, "post_count": 130000},
                {"id": 6, "name": "fluffy_hair", "category": 0, "post_count": 1400},
                {"id": 7, "name": "hair_swept_to_the_side", "category": 0, "post_count": 700},
                {"id": 8, "name": "white_coat", "category": 0, "post_count": 5000},
                {"id": 9, "name": "short_skirt", "category": 0, "post_count": 50000},
                {"id": 10, "name": "artist_example", "category": 1, "post_count": 999999},
                {"id": 11, "name": "long_sleeves_canonical", "category": 0, "post_count": 10},
            ],
            [
                {
                    "antecedent_name": "sleeve_long",
                    "consequent_name": "long_sleeves",
                    "status": "active",
                }
            ],
            snapshot_version="test-snapshot",
            source="unit-test",
        )
        self.store = DanbooruTagStore(self.db_path)
        self._tag_store_patch = mock.patch(
            "services.danbooru_tag_resolver.DanbooruTagStore",
            return_value=self.store,
        )
        self._character_store_patch = mock.patch(
            "services.danbooru_character_resolver.DanbooruTagStore",
            return_value=self.store,
        )
        self._tag_store_patch.start()
        self._character_store_patch.start()

    def tearDown(self):
        self._character_store_patch.stop()
        self._tag_store_patch.stop()
        self.tempdir.cleanup()

    def _llm(self, _system, user, response_format, _max_tokens):
        schema_name = response_format["json_schema"]["name"]
        if schema_name == "danbooru_query_plan":
            if "服だぼっと" in user:
                fragments = [
                    {
                        "source": "服だぼっと",
                        "queries": ["oversized clothes", "baggy clothes"],
                        "expected_category": "general",
                        "must_not_infer": ["hoodie", "jacket", "coat"],
                        "confidence": 0.9,
                    },
                    {
                        "source": "袖長い",
                        "queries": ["long sleeves"],
                        "expected_category": "general",
                        "must_not_infer": ["short sleeves"],
                        "confidence": 0.9,
                    },
                ]
            elif "髪ふわっと" in user:
                fragments = [
                    {
                        "source": "髪ふわっと",
                        "queries": ["fluffy hair", "voluminous hair"],
                        "expected_category": "general",
                        "must_not_infer": [],
                        "confidence": 0.8,
                    },
                    {
                        "source": "横に流れる髪",
                        "queries": ["hair swept to the side"],
                        "expected_category": "general",
                        "must_not_infer": [],
                        "confidence": 0.8,
                    },
                ]
            elif "白いパーカー" in user:
                fragments = [
                    {
                        "source": "白いパーカー",
                        "queries": ["hoodie", "white coat"],
                        "expected_category": "general",
                        "must_not_infer": [],
                        "confidence": 0.9,
                    },
                    {
                        "source": "袖長め",
                        "queries": ["long sleeves"],
                        "expected_category": "general",
                        "must_not_infer": [],
                        "confidence": 0.9,
                    },
                ]
            elif "半袖" in user:
                fragments = [
                    {
                        "source": "半袖",
                        "queries": ["short sleeves"],
                        "expected_category": "general",
                        "must_not_infer": ["long sleeves"],
                        "confidence": 0.9,
                    }
                ]
            else:
                fragments = []
            return json.dumps({"fragments": fragments, "unresolved": []})

        payload = json.loads(user)
        decisions = []
        for fragment in payload["fragments"]:
            for candidate in fragment["candidates"]:
                name = candidate["name"]
                decision = "accept" if name in {
                    "oversized_clothes",
                    "baggy_clothes",
                    "long_sleeves",
                    "fluffy_hair",
                    "hair_swept_to_the_side",
                    "hoodie",
                    "short_sleeves",
                    "white_coat",
                } else "reject"
                # A candidate supplied by the DB can still be rejected by the
                # semantic reviewer; no code path invents a candidate ID.
                decisions.append(
                    {
                        "tag_id": candidate["tag_id"],
                        "decision": decision,
                        "confidence": 0.95 if decision == "accept" else 0.2,
                        "reason": "unit test",
                    }
                )
        # The malicious candidate is deliberately outside the set and must be
        # ignored by the Python validator.
        decisions.append(
            {"tag_id": 999999, "decision": "accept", "confidence": 1.0, "reason": "outside"}
        )
        return json.dumps({"decisions": decisions, "abstain": False})

    def test_exact_alias_prefix_and_fts_search_only_return_general(self):
        self.assertEqual(self.store.search("oversized clothes", limit=1)[0].match_type, "exact")
        alias = self.store.search("sleeve_long", limit=1)[0]
        self.assertEqual(alias.name, "long_sleeves")
        self.assertEqual(alias.match_type, "alias_exact")
        self.assertEqual(self.store.search("fluffy hair", limit=1)[0].name, "fluffy_hair")
        self.assertTrue(all(candidate.category == "general" for candidate in self.store.search("artist", limit=20)))

    def test_fragmentary_clothing_input_never_adds_unrequested_hoodie(self):
        result = resolve_danbooru_tags(
            "服だぼっと、袖長い",
            store=self.store,
            llm_json=self._llm,
        )
        self.assertIn("oversized clothes", result.selected_prompt_tags)
        self.assertIn("baggy clothes", result.selected_prompt_tags)
        self.assertIn("long sleeves", result.selected_prompt_tags)
        self.assertNotIn("hoodie", result.selected_prompt_tags)
        self.assertNotIn(999999, result.selected_tag_ids)

    def test_explicit_parka_and_negation(self):
        result = resolve_danbooru_tags("白いパーカー、袖長め", store=self.store, llm_json=self._llm)
        self.assertIn("hoodie", result.selected_prompt_tags)
        self.assertIn("long sleeves", result.selected_prompt_tags)

        negated = resolve_danbooru_tags("長袖ではなく半袖", store=self.store, llm_json=self._llm)
        self.assertIn("short sleeves", negated.selected_prompt_tags)
        self.assertNotIn("long sleeves", negated.selected_prompt_tags)

    def test_core_composition_and_ambiguous_input_do_not_enter_generic_search(self):
        core = resolve_prompt_tags("少し下から振り返る")
        result = resolve_danbooru_tags(
            "少し下から振り返る",
            core_entries=core.entries,
            store=self.store,
            llm_json=self._llm,
        )
        self.assertEqual(result.selected_prompt_tags, ())
        self.assertEqual(
            resolve_danbooru_tags("かわいい女の子", store=self.store, llm_json=self._llm).selected_prompt_tags,
            (),
        )
        self.assertEqual(
            resolve_danbooru_tags(
                "構図・髪型・服装指定なし",
                store=self.store,
                llm_json=self._llm,
            ).selected_prompt_tags,
            (),
        )

    def test_selected_tags_are_added_once_to_positive(self):
        result = resolve_danbooru_tags("髪ふわっと、横に流れる感じ", store=self.store, llm_json=self._llm)
        positive = apply_danbooru_tag_resolution("1girl, fluffy hair, safe", result)
        self.assertEqual(positive.count("fluffy hair"), 1)
        self.assertIn("hair swept to the side", positive)
        self.assertNotIn("999999", positive)

    def test_animagine_generate_path_adds_only_selected_db_tags(self):
        final_response = json.dumps(
            {
                "positive": "1girl, safe, masterpiece",
                "negative": "lowres, bad anatomy",
                "summary": "衣装テスト",
                "assumptions": [],
            }
        )

        def fake_post(_system, user, *, response_format, **_kwargs):
            name = response_format["json_schema"]["name"]
            if name == "danbooru_query_plan":
                return json.dumps(
                    {
                        "fragments": [
                            {
                                "source": "白いパーカー",
                                "queries": ["hoodie"],
                                "expected_category": "general",
                                "must_not_infer": [],
                                "confidence": 0.9,
                            },
                            {
                                "source": "袖長め",
                                "queries": ["long sleeves"],
                                "expected_category": "general",
                                "must_not_infer": [],
                                "confidence": 0.9,
                            },
                        ],
                        "unresolved": [],
                    }
                )
            if name == "danbooru_candidate_selection":
                payload = json.loads(user)
                decisions = []
                for fragment in payload["fragments"]:
                    for candidate in fragment["candidates"]:
                        decisions.append(
                            {
                                "tag_id": candidate["tag_id"],
                                "decision": "accept" if candidate["name"] in {"hoodie", "long_sleeves"} else "reject",
                                "confidence": 0.95,
                                "reason": "direct candidate match",
                            }
                        )
                return json.dumps({"decisions": decisions, "abstain": False})
            return final_response

        with mock.patch.object(shogou_prompt_designer, "_post_chat_completion", side_effect=fake_post):
            result = shogou_prompt_designer.generate_shogou_prompt_result(
                "白いパーカー、袖長め",
                model_key="animagine",
                requested_rating="safe",
                enable_generic_danbooru=True,
            )

        self.assertFalse(result.blocked)
        self.assertIn("hoodie", result.positive)
        self.assertIn("long sleeves", result.positive)
        self.assertTrue(result.danbooru_resolved_tag_ids)
        self.assertEqual(result.danbooru_tag_db_version, "test-snapshot")


if __name__ == "__main__":
    unittest.main()
