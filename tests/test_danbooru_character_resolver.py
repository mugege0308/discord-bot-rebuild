import json
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services.character_tag_dictionary import resolve_all_character_tags
from services.character_appearance_filter import (
    filter_unrequested_character_appearance,
)
from services.danbooru_character_resolver import (
    _build_character_span_attempts,
    _has_character_tail_boundary,
    apply_danbooru_character_resolution,
    DanbooruCharacterResolution,
    apply_danbooru_character_resolution,
    extract_character_context_aliases,
    extract_work_character_context,
    extract_possible_character_names,
    lookup_copyright_aliases,
    lookup_copyright_query_hints,
    resolve_danbooru_character_tags,
    should_search_character_tags,
)
from services.danbooru_tag_importer import write_snapshot
from services.danbooru_tag_store import DanbooruTagStore
from services.verified_character_registry import VerifiedCharacterRegistry
from services import shogou_prompt_designer


class DanbooruCharacterResolverTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "tags.sqlite3"
        write_snapshot(
            self.db_path,
            [
                {
                    "id": 1,
                    "name": "asakura_karen_(heaven_burns_red)",
                    "category": 4,
                    "post_count": 279,
                },
                {
                    "id": 2,
                    "name": "asakura_karen_(other_work)",
                    "category": 4,
                    "post_count": 120,
                },
                {"id": 3, "name": "asakura_toru", "category": 4, "post_count": 2644},
                {"id": 4, "name": "heaven_burns_red", "category": 3, "post_count": 2570},
                {"id": 5, "name": "asakura_karen_general", "category": 0, "post_count": 9999},
                {"id": 6, "name": "kochiya_sanae", "category": 4, "post_count": 36072},
                {"id": 7, "name": "koei_takeru", "category": 4, "post_count": 4},
                {"id": 8, "name": "touhou", "category": 3, "post_count": 911841},
                {"id": 9, "name": "asta_(honkai_star_rail)", "category": 4, "post_count": 8200},
                {"id": 10, "name": "honkai_star_rail", "category": 3, "post_count": 120000},
            ],
            [
                {
                    "antecedent_name": "karen_asakura",
                    "consequent_name": "asakura_karen_(heaven_burns_red)",
                    "status": "active",
                }
            ],
            snapshot_version="character-test-snapshot",
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

    @staticmethod
    def _planner(work_queries=()):
        def llm(_system, _user, _response_format, _max_tokens):
            return json.dumps(
                {
                    "characters": [
                        {
                            "input_name": "朝倉可憐",
                            "queries": ["asakura karen", "karen asakura"],
                            "work_queries": list(work_queries),
                        }
                    ],
                    "unresolved": [],
                },
                ensure_ascii=False,
            )

        return llm

    def test_work_hint_selects_canonical_character_only(self):
        result = resolve_danbooru_character_tags(
            "朝倉可憐が振り返って笑ってる",
            work_hints=("heaven burns red",),
            store=self.store,
            llm_json=self._planner(("heaven burns red",)),
        )
        self.assertEqual(result.selected_character_tag, "asakura_karen_(heaven_burns_red)")
        self.assertEqual(result.selected_prompt_tag, "asakura karen (heaven burns red)")
        self.assertTrue(result.confirmation_required)
        self.assertFalse(result.has_selected_tag)
        self.assertEqual(result.candidate_character_tags[0].category, "character")
        registry = VerifiedCharacterRegistry(Path(self.tempdir.name) / "registry.sqlite3")
        saved = registry.register_verified_candidate(
            result.candidate_character_tags[0],
            input_alias="朝倉可憐",
            copyright_tag=result.selected_copyright_tag,
            context_aliases=result.generated_work_queries,
            store=self.store,
        )
        confirmed = result.with_registry_confirmation(
            character=saved,
            saved_alias="朝倉可憐",
        )
        self.assertEqual(
            apply_danbooru_character_resolution(
                "1girl, 朝倉可憐, asakura karen (other work), looking back, smile, safe",
                confirmed,
            ),
            "1girl, asakura karen (heaven burns red), looking back, smile, safe",
        )

    def test_without_work_ambiguous_candidates_abstain(self):
        result = resolve_danbooru_character_tags(
            "朝倉可憐が笑ってる",
            store=self.store,
            llm_json=self._planner(),
        )
        self.assertTrue(result.abstained)
        self.assertEqual(result.unresolved_reason, "ambiguous_without_work")
        self.assertEqual(result.confirmed_character_tag, "")
        self.assertEqual(result.selected_candidate, "asakura_karen_(heaven_burns_red)")

    def test_manual_dictionary_has_priority_and_never_searches_db(self):
        manual = resolve_all_character_tags("ブルアカのケイ、笑顔")
        self.assertTrue(manual)
        result = resolve_danbooru_character_tags(
            "ブルアカのケイ、笑顔",
            manual_resolutions=manual,
            store=self.store,
            llm_json=mock.Mock(side_effect=AssertionError("DB resolver must not call LLM")),
        )
        self.assertTrue(result.abstained)
        self.assertEqual(result.unresolved_reason, "manual_dictionary_priority")

    def test_english_and_general_appearance_inputs_bypass_character_search(self):
        self.assertFalse(should_search_character_tags("blue eyes, long hair", db_path=self.db_path))
        self.assertFalse(
            should_search_character_tags("白いパーカー、袖長め", db_path=self.db_path)
        )

    def test_katakana_name_with_middle_dot_enters_character_search_gate(self):
        self.assertEqual(
            extract_possible_character_names("アリス・マーガトロイドを笑顔で"),
            ("アリス・マーガトロイド",),
        )
        self.assertEqual(extract_possible_character_names("パーカーを着ている"), ())

    def test_nonexistent_name_does_not_turn_into_another_character(self):
        result = resolve_danbooru_character_tags(
            "架空人物太郎が笑ってる",
            store=self.store,
            llm_json=lambda *_args: json.dumps(
                {
                    "characters": [
                        {
                            "input_name": "架空人物太郎",
                            "queries": ["nonexistent person"],
                            "work_queries": [],
                        }
                    ],
                    "unresolved": [],
                }
            ),
        )
        self.assertTrue(result.abstained)
        self.assertEqual(result.unresolved_reason, "no_character_candidates")

    def test_kochiya_sanae_regression_never_accepts_koei_takeru(self):
        def planner(_system, _user, _response_format, _max_tokens):
            return json.dumps(
                {
                    "characters": [
                        {
                            "input_name": "東風谷早苗",
                            "queries": ["kochiya sanae", "sanae kochiya"],
                            "work_queries": ["touhou"],
                        }
                    ],
                    "unresolved": [],
                },
                ensure_ascii=False,
            )

        result = resolve_danbooru_character_tags(
            "東風谷早苗",
            work_hints=("touhou",),
            store=self.store,
            llm_json=planner,
        )
        names = {candidate.name for candidate in result.candidate_character_tags}
        self.assertIn("kochiya_sanae", names)
        self.assertNotIn("koei_takeru", result.selected_candidate)
        self.assertTrue(result.confirmation_required)
        self.assertFalse(result.has_selected_tag)

    def test_zero_match_planner_retries_with_name_only_query(self):
        calls = []

        def planner(_system, _user, _response_format, _max_tokens):
            calls.append((_system, _user))
            if len(calls) == 1:
                queries = ["fictional taro classroom smile"]
            else:
                queries = ["fictional taro", "taro fictional"]
            return json.dumps(
                {
                    "characters": [
                        {
                            "input_name": "架空人物太郎",
                            "queries": queries,
                            "work_queries": [],
                        }
                    ],
                    "unresolved": [],
                },
                ensure_ascii=False,
            )

        result = resolve_danbooru_character_tags(
            "架空人物太郎が教室で笑顔で",
            store=self.store,
            llm_json=planner,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(result.candidate_character_tags, ())
        self.assertNotIn("koei_takeru", result.selected_candidate)
        self.assertFalse(result.confirmation_required)
        self.assertFalse(result.has_selected_tag)
        self.assertEqual(result.generated_queries[0], "fictional taro classroom smile")
        self.assertIn("fictional taro", result.generated_queries)

    def test_query_hint_recovers_known_japanese_name_without_auto_accepting(self):
        result = resolve_danbooru_character_tags(
            "東風谷早苗を教室で笑顔で",
            store=self.store,
            llm_json=lambda *_args: json.dumps(
                {
                    "characters": [
                        {
                            "input_name": "東風谷早苗",
                            "queries": ["toufukayu saemon classroom smile"],
                            "work_queries": [],
                        }
                    ],
                    "unresolved": [],
                },
                ensure_ascii=False,
            ),
        )

        self.assertIn("kochiya_sanae", {candidate.name for candidate in result.candidate_character_tags})
        self.assertTrue(result.confirmation_required)
        self.assertFalse(result.has_selected_tag)
        self.assertIn("kochiya sanae", result.generated_queries)

    def test_explicit_japanese_work_context_is_retained_for_registry(self):
        self.assertIn(
            "東方",
            extract_character_context_aliases("東方の東風谷早苗", "東風谷早苗"),
        )

    def test_work_of_character_is_split_before_character_search(self):
        context = extract_work_character_context("崩壊スターレイルのアスター")
        self.assertEqual(context.work_name, "崩壊スターレイル")
        self.assertEqual(context.character_name, "アスター")
        self.assertEqual(
            extract_possible_character_names("崩壊スターレイルのアスター"),
            ("アスター",),
        )
        self.assertNotIn("崩壊スターレイル", extract_possible_character_names("崩壊スターレイルのアスター"))

    def test_copyright_aliases_are_query_only_until_character_confirmation(self):
        self.assertIn("honkai star rail", lookup_copyright_query_hints("崩壊：スターレイル"))
        self.assertIn("スタレ", lookup_copyright_aliases("崩壊スターレイル"))
        self.assertFalse(should_search_character_tags("崩壊スターレイル", db_path=self.db_path))

    def test_work_context_filters_character_candidates_and_keeps_trace(self):
        def planner(_system, user, _response_format, _max_tokens):
            payload = json.loads(user)
            self.assertEqual(payload["input"], "アスター")
            self.assertEqual(payload["candidate_names"], ["アスター"])
            self.assertIn("崩壊スターレイル", payload["extracted_work_name"])
            return json.dumps(
                {
                    "characters": [
                        {
                            "input_name": "アスター",
                            "queries": ["asta"],
                            "work_queries": ["honkai star rail"],
                        }
                    ],
                    "unresolved": [],
                },
                ensure_ascii=False,
            )

        result = resolve_danbooru_character_tags(
            "崩壊スターレイルのアスター",
            store=self.store,
            llm_json=planner,
        )

        self.assertEqual(result.extracted_work_name, "崩壊スターレイル")
        self.assertEqual(result.extracted_character_name, "アスター")
        self.assertTrue(result.work_alias_hit)
        self.assertEqual(result.selected_character_tag, "asta_(honkai_star_rail)")
        self.assertEqual(result.selected_copyright_tag, "honkai_star_rail")
        self.assertTrue(result.confirmation_required)
        self.assertEqual(
            {candidate.name for candidate in result.candidate_character_tags},
            {"asta_(honkai_star_rail)"},
        )
        trace = result.to_trace_dict()
        self.assertEqual(trace["extracted_work_name"], "崩壊スターレイル")
        self.assertEqual(trace["extracted_character_name"], "アスター")
        self.assertTrue(trace["copyright_candidates"])

    def test_wrong_llm_spelling_uses_copyright_scoped_romanized_fallback(self):
        def planner(_system, _user, _response_format, _max_tokens):
            # Reproduce the production miss: the model hears アスター as an
            # English-looking ``aster`` family and never emits ``asta``.
            return json.dumps(
                {
                    "characters": [
                        {
                            "input_name": "アスター",
                            "queries": ["aster", "asterra", "asterran", "astera"],
                            "work_queries": ["honkai star rail"],
                        }
                    ],
                    "unresolved": [],
                },
                ensure_ascii=False,
            )

        result = resolve_danbooru_character_tags(
            "崩壊スターレイルのアスター 後ろで手を組んで前かがみ",
            store=self.store,
            llm_json=planner,
        )

        self.assertEqual(result.input_character_name, "アスター")
        self.assertEqual(result.remaining_prompt_text, "後ろで手を組んで前かがみ")
        self.assertEqual(result.deterministic_romanization, "asutaa")
        self.assertTrue(result.fallback_used)
        self.assertGreaterEqual(result.copyright_scoped_candidate_count, 1)
        self.assertEqual(result.selected_character_tag, "asta_(honkai_star_rail)")
        self.assertTrue(result.confirmation_required)
        self.assertTrue(result.fuzzy_candidate_scores)
        self.assertEqual(
            result.to_trace_dict()["llm_character_queries"],
            ["aster", "asterra", "asterran", "astera"],
        )

    def test_action_text_after_character_is_remaining_prompt(self):
        def planner(_system, user, _response_format, _max_tokens):
            payload = json.loads(user)
            self.assertEqual(payload["candidate_names"], ["アスター"])
            return json.dumps(
                {
                    "characters": [
                        {
                            "input_name": "アスター",
                            "queries": ["asta"],
                            "work_queries": ["honkai star rail"],
                        }
                    ],
                    "unresolved": [],
                },
                ensure_ascii=False,
            )

        result = resolve_danbooru_character_tags(
            "崩壊スターレイルのアスター 後ろで手を組んで前かがみ",
            store=self.store,
            llm_json=planner,
        )

        self.assertEqual(result.extracted_work_name, "崩壊スターレイル")
        self.assertEqual(result.raw_character_span, "アスター 後ろで手を組んで前かがみ")
        self.assertEqual(result.extracted_character_name, "アスター")
        self.assertEqual(result.input_character_name, "アスター")
        self.assertEqual(result.remaining_prompt_text, "後ろで手を組んで前かがみ")
        self.assertEqual(result.selected_character_tag, "asta_(honkai_star_rail)")
        self.assertEqual(result.character_candidate_attempts[-1]["status"], "matched")
        self.assertEqual(
            result.character_candidate_attempts[0]["status"],
            "syntax_action_tail",
        )

    def test_particles_and_compound_character_names_keep_the_correct_span(self):
        cases = (
            ("崩壊スターレイルのアスターが笑っている", "アスター", "笑っている"),
            ("崩壊スターレイルのアスターを下から見る", "アスター", "下から見る"),
            ("東方Projectのアリス・マーガトロイドが座っている", "アリス・マーガトロイド", "座っている"),
            ("東方Projectの東風谷早苗 振り返って笑う", "東風谷早苗", "振り返って笑う"),
            ("崩壊スターレイルのMarch 7thが笑っている", "March 7th", "笑っている"),
            ("崩壊スターレイルのSilver Wolfが座っている", "Silver Wolf", "座っている"),
        )
        for text, expected_name, expected_remaining in cases:
            with self.subTest(text=text):
                context = extract_work_character_context(text)
                self.assertEqual(context.work_name, text.split("の", 1)[0])
                self.assertEqual(extract_possible_character_names(text), (expected_name,))
                attempts = _build_character_span_attempts(context)
                usable = [
                    (name, remaining)
                    for name, remaining in attempts
                    if not _has_character_tail_boundary(name)
                ]
                self.assertEqual(usable[0], (expected_name, expected_remaining))

    def test_work_only_and_ordinary_scene_do_not_trigger_character_search(self):
        self.assertFalse(should_search_character_tags("崩壊スターレイル", db_path=self.db_path))
        self.assertFalse(should_search_character_tags("教室の女の子が笑っている", db_path=self.db_path))
        self.assertEqual(extract_possible_character_names("教室の女の子が笑っている"), ())

    def test_explicit_work_context_is_saved_for_short_alias_reuse(self):
        result = resolve_danbooru_character_tags(
            "崩壊スターレイルのアスター",
            store=self.store,
            llm_json=lambda *_args: json.dumps(
                {
                    "characters": [
                        {
                            "input_name": "アスター",
                            "queries": ["asta"],
                            "work_queries": ["honkai star rail"],
                        }
                    ],
                    "unresolved": [],
                },
                ensure_ascii=False,
            ),
        )
        self.assertIn("崩壊スターレイル", result.context_aliases)
        self.assertIn("スタレ", result.context_aliases)
        registry = VerifiedCharacterRegistry(Path(self.tempdir.name) / "registry.sqlite3")
        registry.register_verified_candidate(
            result.candidate_character_tags[0],
            input_alias="アスター",
            copyright_tag=result.selected_copyright_tag,
            context_aliases=result.context_aliases,
            store=self.store,
        )
        self.assertEqual(
            registry.lookup("スタレのアスター")[0].character.canonical_danbooru_tag,
            "asta_(honkai_star_rail)",
        )

    def test_generate_ai_path_inserts_only_db_character_tag(self):
        final_response = json.dumps(
            {
                "positive": "1girl, 朝倉可憐, looking back, smile, safe",
                "negative": "lowres, bad anatomy",
                "summary": "キャラクター確認",
                "assumptions": [],
            },
            ensure_ascii=False,
        )

        def fake_post(_system, _user, *, response_format, **_kwargs):
            schema_name = response_format["json_schema"]["name"]
            if schema_name == "danbooru_character_query_plan":
                return json.dumps(
                    {
                        "characters": [
                            {
                                "input_name": "朝倉可憐",
                                "queries": ["asakura karen"],
                                "work_queries": ["heaven burns red"],
                            }
                        ],
                        "unresolved": [],
                    },
                    ensure_ascii=False,
                )
            return final_response

        with mock.patch.object(shogou_prompt_designer, "_post_chat_completion", side_effect=fake_post):
            result = shogou_prompt_designer.generate_shogou_prompt_result(
                "朝倉可憐が振り返って笑ってる",
                model_key="animagine",
                requested_rating="safe",
                enable_generic_danbooru=True,
            )

        self.assertFalse(result.blocked)
        self.assertTrue(result.character_confirmation_required)
        self.assertEqual(result.positive, "")
        self.assertEqual(result.danbooru_character_resolved_tags, ())

    def test_registry_hit_skips_character_search_and_resolves_on_second_generation(self):
        registry = VerifiedCharacterRegistry(Path(self.tempdir.name) / "registry.sqlite3")
        candidate = self.store.search(
            "asakura karen",
            limit=1,
            allowed_categories=("character",),
        )[0]
        registry.register_verified_candidate(
            candidate,
            input_alias="朝倉可憐",
            copyright_tag="heaven_burns_red",
            store=self.store,
        )
        final_response = json.dumps(
            {
                "positive": "1girl, 朝倉可憐, looking back, safe",
                "negative": "lowres",
                "summary": "registry hit",
                "assumptions": [],
            },
            ensure_ascii=False,
        )

        with mock.patch.object(shogou_prompt_designer, "VerifiedCharacterRegistry", return_value=registry), \
             mock.patch.object(shogou_prompt_designer, "_post_chat_completion", return_value=final_response) as post:
            result = shogou_prompt_designer.generate_shogou_prompt_result(
                "朝倉可憐が振り返ってる",
                model_key="animagine",
                requested_rating="safe",
                enable_generic_danbooru=True,
            )

        self.assertFalse(result.character_confirmation_required)
        self.assertIn("asakura karen (heaven burns red)", result.positive)
        self.assertNotIn("朝倉可憐", result.positive)
        self.assertEqual(result.danbooru_character_resolution.registry_hit, True)
        self.assertEqual(post.call_count, 1)

    def test_confirmed_character_suppresses_unrequested_hair_identity_completion(self):
        resolution = DanbooruCharacterResolution(
            selected_character_tag="kochiya_sanae",
            selected_prompt_tag="kochiya sanae",
            confirmed_character_tag="kochiya_sanae",
        )

        result = filter_unrequested_character_appearance(
            "1girl, kochiya sanae, long white hair, smile, peace sign, safe",
            user_input="東風谷早苗が笑顔でピース",
            character_resolution=resolution,
        )

        self.assertIn("kochiya sanae", result.positive)
        self.assertIn("smile", result.positive)
        self.assertIn("peace sign", result.positive)
        self.assertNotIn("long white hair", result.positive)
        self.assertEqual(result.removed_tags, ("long white hair",))

    def test_explicit_hair_color_is_preserved_for_confirmed_character(self):
        resolution = DanbooruCharacterResolution(
            selected_character_tag="kochiya_sanae",
            selected_prompt_tag="kochiya sanae",
            confirmed_character_tag="kochiya_sanae",
        )

        result = filter_unrequested_character_appearance(
            "1girl, kochiya sanae, long white hair, white hair, smile, peace sign, safe",
            user_input="東風谷早苗が白髪になって笑顔でピース",
            character_resolution=resolution,
        )

        self.assertIn("white hair", result.positive)
        self.assertNotIn("long white hair", result.positive)
        self.assertEqual(result.removed_tags, ("long white hair",))

    def test_confirmed_character_keeps_explicit_composition_and_filters_hair(self):
        resolution = DanbooruCharacterResolution(
            selected_character_tag="kochiya_sanae",
            selected_prompt_tag="kochiya sanae",
            confirmed_character_tag="kochiya_sanae",
        )

        result = filter_unrequested_character_appearance(
            "1girl, kochiya sanae, long white hair, from below, looking back, smile, safe",
            user_input="東風谷早苗が少し下から振り返って笑う",
            character_resolution=resolution,
        )

        self.assertNotIn("long white hair", result.positive)
        self.assertIn("from below", result.positive)
        self.assertIn("looking back", result.positive)
        self.assertIn("smile", result.positive)

    def test_confirmed_character_filters_eye_color_and_constant_accessory(self):
        resolution = DanbooruCharacterResolution(
            selected_character_tag="kochiya_sanae",
            selected_prompt_tag="kochiya sanae",
            confirmed_character_tag="kochiya_sanae",
        )

        result = filter_unrequested_character_appearance(
            "1girl, kochiya sanae, white hair, blue eyes, black hairband, smile, safe",
            user_input="東風谷早苗が目キラキラして笑顔で",
            character_resolution=resolution,
        )

        self.assertNotIn("white hair", result.positive)
        self.assertNotIn("blue eyes", result.positive)
        self.assertNotIn("black hairband", result.positive)
        self.assertEqual(
            result.removed_tags,
            ("white hair", "blue eyes", "black hairband"),
        )

    def test_verified_generic_tag_is_not_removed_by_character_filter(self):
        resolution = DanbooruCharacterResolution(
            selected_character_tag="kochiya_sanae",
            selected_prompt_tag="kochiya sanae",
            confirmed_character_tag="kochiya_sanae",
        )

        result = filter_unrequested_character_appearance(
            "1girl, kochiya sanae, white hair, smile, safe",
            user_input="東風谷早苗が笑顔で",
            character_resolution=resolution,
            verified_prompt_tags=("white hair",),
        )

        self.assertIn("white hair", result.positive)
        self.assertEqual(result.removed_tags, ())

    def test_registry_hit_filters_llm_appearance_in_prompt_designer(self):
        registry = VerifiedCharacterRegistry(Path(self.tempdir.name) / "registry.sqlite3")
        candidate = self.store.search(
            "kochiya sanae",
            limit=1,
            allowed_categories=("character",),
        )[0]
        registry.register_verified_candidate(
            candidate,
            input_alias="東風谷早苗",
            copyright_tag="touhou",
            store=self.store,
        )
        final_response = json.dumps(
            {
                "positive": "1girl, kochiya sanae, long white hair, smile, safe",
                "negative": "lowres",
                "summary": "appearance suppression",
                "assumptions": [],
            },
            ensure_ascii=False,
        )

        with mock.patch.object(shogou_prompt_designer, "VerifiedCharacterRegistry", return_value=registry), \
             mock.patch.object(shogou_prompt_designer, "_post_chat_completion", return_value=final_response):
            result = shogou_prompt_designer.generate_shogou_prompt_result(
                "東風谷早苗が笑顔でピース",
                model_key="animagine",
                requested_rating="safe",
                enable_generic_danbooru=True,
            )

        self.assertFalse(result.blocked)
        self.assertTrue(result.danbooru_character_resolution.registry_hit)
        self.assertIn("kochiya sanae", result.positive)
        self.assertNotIn("long white hair", result.positive)
        self.assertEqual(result.character_appearance_suppressed_tags, ("long white hair",))

    def test_registry_hit_cleans_invalid_and_duplicate_positive_tags(self):
        registry = VerifiedCharacterRegistry(Path(self.tempdir.name) / "registry.sqlite3")
        candidate = self.store.search(
            "kochiya sanae",
            limit=1,
            allowed_categories=("character",),
        )[0]
        registry.register_verified_candidate(
            candidate,
            input_alias="東風谷早苗",
            copyright_tag="touhou",
            store=self.store,
        )
        final_response = json.dumps(
            {
                "positive": (
                    "2, kochiya sanae, safe, white hair, smiley face, v, "
                    "long_hair, smile, anime style, masterpiece, high score, great score, absurdres"
                ),
                "negative": "lowres",
                "summary": "explicit appearance change",
                "assumptions": [],
            },
            ensure_ascii=False,
        )

        with mock.patch.object(shogou_prompt_designer, "VerifiedCharacterRegistry", return_value=registry), \
             mock.patch.object(shogou_prompt_designer, "_post_chat_completion", return_value=final_response):
            result = shogou_prompt_designer.generate_shogou_prompt_result(
                "東風谷早苗が白髪になって笑顔でピース",
                model_key="animagine",
                requested_rating="safe",
                enable_generic_danbooru=True,
            )

        self.assertFalse(result.blocked)
        self.assertIn("kochiya sanae", result.positive)
        self.assertIn("white hair", result.positive)
        self.assertIn("smile", result.positive)
        self.assertIn("peace sign", result.positive)
        self.assertNotIn("long_hair", result.positive)
        self.assertNotIn("smiley face", result.positive)
        self.assertNotIn(", v", ", " + result.positive)
        self.assertNotIn(", 2,", ", " + result.positive + ", ")
        self.assertEqual(result.character_appearance_suppressed_tags, ("long_hair",))
        self.assertEqual(result.positive_cleanup_removed_tags, ("2", "smiley face", "v", "smile"))


if __name__ == "__main__":
    unittest.main()
