import unittest

from services.character_tag_dictionary import (
    _alias_present,
    _build_character_entries,
    _build_series_entries,
    _normalize,
    find_characters_by_name,
    get_character_detail,
    known_character_and_series_tags,
    list_characters,
    resolve_all_character_tags,
    resolve_character_tags,
    search_characters,
)


class KomeijiKoishiResolutionTestCase(unittest.TestCase):
    def test_japanese_full_name_with_series_resolves(self):
        result = resolve_character_tags("東方の古明地こいし")

        self.assertIsNotNone(result)
        self.assertEqual(result.character_tag, "komeiji koishi")
        self.assertEqual(result.series_tag, "touhou")

    def test_english_full_name_with_series_resolves(self):
        result = resolve_character_tags("Touhou Project Koishi Komeiji")

        self.assertIsNotNone(result)
        self.assertEqual(result.character_tag, "komeiji koishi")
        self.assertEqual(result.series_tag, "touhou")

    def test_bare_given_name_without_series_does_not_resolve(self):
        # "koishi"/"こいし" alone can also read as an ordinary word/name, so it
        # must not resolve without the series also being present.
        result = resolve_character_tags("こいしという名前の由来")

        self.assertIsNone(result)


class SparkleResolutionTestCase(unittest.TestCase):
    def test_japanese_alias_with_series_resolves(self):
        result = resolve_character_tags("崩壊:スターレイルの火花")

        self.assertIsNotNone(result)
        self.assertEqual(result.character_tag, "sparkle")
        self.assertEqual(result.series_tag, "honkai: star rail")

    def test_fullwidth_colon_series_alias_also_resolves(self):
        result = resolve_character_tags("崩壊：スターレイルの火花、笑顔")

        self.assertIsNotNone(result)
        self.assertEqual(result.character_tag, "sparkle")
        self.assertEqual(result.series_tag, "honkai: star rail")

    def test_generic_word_without_series_does_not_resolve(self):
        result = resolve_character_tags("火花が散る夜空")

        self.assertIsNone(result)

    def test_unrelated_english_word_does_not_resolve(self):
        result = resolve_character_tags("spark effect")

        self.assertIsNone(result)


class KinakoNanobanaResolutionTestCase(unittest.TestCase):
    """character_tag is "nanobana kinako" (surname-first), matching
    Danbooru's real `nanobana_kinako` tag confirmed by external research.
    "kinako nanobana" (the dictionary's original, reversed order) remains
    accepted as a backward-compatible input alias - see
    test_english_alias_with_series_resolves below."""

    def test_full_name_with_series_resolves(self):
        result = resolve_character_tags("イナズマイレブンGOの菜花きなこ")

        self.assertIsNotNone(result)
        self.assertEqual(result.character_tag, "nanobana kinako")
        self.assertEqual(result.series_tag, "inazuma eleven go")

    def test_english_alias_with_series_resolves(self):
        result = resolve_character_tags("inazuma eleven go kinako nanobana")

        self.assertIsNotNone(result)
        self.assertEqual(result.character_tag, "nanobana kinako")
        self.assertEqual(result.series_tag, "inazuma eleven go")

    def test_common_noun_does_not_resolve_to_character(self):
        result = resolve_character_tags("きなこ餅がおいしい")

        self.assertIsNone(result)


class NoMatchTestCase(unittest.TestCase):
    def test_unrelated_text_resolves_to_none(self):
        self.assertIsNone(resolve_character_tags("猫耳の女の子が公園にいる"))

    def test_empty_text_resolves_to_none(self):
        self.assertIsNone(resolve_character_tags(""))

    def test_long_unrelated_word_containing_a_short_alias_is_not_falsely_matched(self):
        # "koishi" is a substring inside neither of these, but guards against a
        # naive implementation that would match on partial/root substrings
        # instead of whole aliases.
        self.assertIsNone(resolve_character_tags("this is a sparkling clean kitchen"))


class SparkleStandaloneEnglishNameTestCase(unittest.TestCase):
    """The English name "sparkle" was promoted to a safe alias (this batch):
    unlike the generic Japanese word "花火", it is specific enough to resolve
    without the series also being present, per the request that added the
    other 11 characters below."""

    def test_bare_english_name_resolves_without_series(self):
        result = resolve_character_tags("sparkle")

        self.assertIsNotNone(result)
        self.assertEqual(result.character_tag, "sparkle")
        self.assertEqual(result.series_tag, "honkai: star rail")

    def test_series_abbreviation_alias_also_resolves(self):
        result = resolve_character_tags("スタレの花火")

        self.assertIsNotNone(result)
        self.assertEqual(result.character_tag, "sparkle")
        self.assertEqual(result.series_tag, "honkai: star rail")

    def test_bare_hanabi_spelling_without_series_does_not_resolve(self):
        # "花火" ("fireworks") alone is an ordinary word (e.g. a genuine
        # fireworks-festival request) and must stay series-gated just like
        # "火花".
        self.assertIsNone(resolve_character_tags("夏祭りの花火を見上げる"))


class KinakoFullKanjiNameTestCase(unittest.TestCase):
    """The character's full formal kanji name "菜花黄名子" must resolve on its
    own, without the series also being present (this batch added it as a
    safe alias alongside the pre-existing "菜花きなこ")."""

    def test_full_kanji_name_resolves_without_series(self):
        result = resolve_character_tags("菜花黄名子")

        self.assertIsNotNone(result)
        self.assertEqual(result.character_tag, "nanobana kinako")
        self.assertEqual(result.series_tag, "inazuma eleven go")


# (input_text, expected_character_tag, expected_series_tag) for cases that
# must resolve. Table-driven so all 14 characters get the same baseline
# coverage: full Japanese name / romaji name / series-qualified input.
_RESOLVES_CASES: tuple[tuple[str, str, str, str], ...] = (
    ("nanami-chiaki-full-name", "七海千秋", "nanami chiaki", "danganronpa"),
    ("nanami-chiaki-romaji", "nanami chiaki", "nanami chiaki", "danganronpa"),
    ("nanami-chiaki-with-series", "ダンガンロンパの七海千秋", "nanami chiaki", "danganronpa"),
    ("nanami-chiaki-uppercase", "NANAMI CHIAKI", "nanami chiaki", "danganronpa"),
    ("nanami-chiaki-padded", "  七海千秋  ", "nanami chiaki", "danganronpa"),
    ("shimoe-koharu-full-name", "下江コハル", "shimoe koharu", "blue archive"),
    ("shimoe-koharu-romaji", "shimoe koharu", "shimoe koharu", "blue archive"),
    ("shimoe-koharu-abbrev-series", "ブルアカのコハル", "shimoe koharu", "blue archive"),
    ("kisaki-full-name", "竜華キサキ", "kisaki", "blue archive"),
    ("kisaki-romaji-full-name", "ryuuge kisaki", "kisaki", "blue archive"),
    ("kisaki-full-series-name", "ブルーアーカイブの竜華キサキ", "kisaki", "blue archive"),
    ("kisaki-abbrev-series-japanese", "ブルアカのキサキ", "kisaki", "blue archive"),
    ("kisaki-abbrev-series-romaji", "blue archive kisaki", "kisaki", "blue archive"),
    ("kei-abbrev-series", "ブルアカのケイ", "kei", "blue archive"),
    ("kei-uppercase-abbrev-series", "ブルアカのKEI", "kei", "blue archive"),
    ("leafa-full-series-name", "ソードアート・オンラインのリーファ", "leafa", "sword art online"),
    ("leafa-abbrev-series-no-space", "SAOのリーファ", "leafa", "sword art online"),
    ("leafa-romaji-abbrev-series", "sao leafa", "leafa", "sword art online"),
    ("hu-tao-with-series", "原神の胡桃", "hu tao", "genshin impact"),
    ("hu-tao-romaji-standalone", "hu tao", "hu tao", "genshin impact"),
    ("hu-tao-romaji-nospace", "genshin impact hutao", "hu tao", "genshin impact"),
    ("suzuna-with-series", "ポケモンのスズナ", "suzuna", "pokemon"),
    ("suzuna-romaji-with-series", "pokemon suzuna", "suzuna", "pokemon"),
    ("suzuki-hana-full-name", "鈴木羽那", "suzuki hana", "idolmaster shiny colors"),
    ("suzuki-hana-romaji", "suzuki hana", "suzuki hana", "idolmaster shiny colors"),
    ("suzuki-hana-abbrev-series", "シャニマスの鈴木羽那", "suzuki hana", "idolmaster shiny colors"),
    ("izumi-kie-full-name", "和泉妃愛", "izumi kie", "hamidashi creative"),
    ("izumi-kie-romaji", "izumi kie", "izumi kie", "hamidashi creative"),
    ("izumi-kie-with-series", "ハミダシクリエイティブの和泉妃愛", "izumi kie", "hamidashi creative"),
    ("meruru-with-series", "まのさばのメルル", "meruru", "mano saba"),
    ("meruru-romaji-with-series", "mano saba meruru", "meruru", "mano saba"),
    ("shikano-chinatsu-full-name", "鹿野千夏", "shikano chinatsu", "ao no hako"),
    ("shikano-chinatsu-romaji", "shikano chinatsu", "shikano chinatsu", "ao no hako"),
    ("shikano-chinatsu-abbrev-series", "アオのハコの千夏", "shikano chinatsu", "ao no hako"),
    ("shikano-chinatsu-english-series-alias", "blue box chinatsu", "shikano chinatsu", "ao no hako"),
)


class NewCharacterResolutionTableTestCase(unittest.TestCase):
    def test_each_case_resolves_to_the_expected_tags(self):
        for case_name, input_text, expected_character_tag, expected_series_tag in _RESOLVES_CASES:
            with self.subTest(case=case_name):
                result = resolve_character_tags(input_text)

                self.assertIsNotNone(result, f"expected {input_text!r} to resolve")
                self.assertEqual(result.character_tag, expected_character_tag)
                self.assertEqual(result.series_tag, expected_series_tag)


# Bare/ambiguous aliases that must NOT resolve without their series also
# being present in the same text - the "誤検出しやすい短い名前" safety
# requirement for this batch.
_DOES_NOT_RESOLVE_ALONE_CASES: tuple[tuple[str, str], ...] = (
    ("koharu-bare", "コハル"),
    ("kisaki-bare-japanese", "キサキ"),
    ("kisaki-bare-romaji", "kisaki"),
    ("kei-bare-japanese", "ケイ"),
    ("kei-bare-romaji", "kei"),
    ("leafa-bare-japanese", "リーファ"),
    ("leafa-bare-romaji", "leafa"),
    ("hu-tao-bare-kanji", "胡桃"),
    ("suzuna-bare-japanese", "スズナ"),
    ("suzuna-bare-romaji", "suzuna"),
    ("meruru-bare-japanese", "メルル"),
    ("meruru-bare-romaji", "meruru"),
    ("chinatsu-bare", "千夏"),
)


class AmbiguousAliasRequiresSeriesTestCase(unittest.TestCase):
    def test_bare_ambiguous_alias_does_not_resolve_without_series(self):
        for case_name, input_text in _DOES_NOT_RESOLVE_ALONE_CASES:
            with self.subTest(case=case_name):
                self.assertIsNone(resolve_character_tags(input_text), f"{input_text!r} must not resolve alone")

    def test_ambiguous_alias_inside_unrelated_sentence_does_not_resolve(self):
        self.assertIsNone(resolve_character_tags("胡桃を食べるのが好きです"))
        self.assertIsNone(resolve_character_tags("コハルという名前の由来"))


class WordBoundaryFalsePositiveTestCase(unittest.TestCase):
    """Short ASCII aliases (e.g. "kei") must not match a substring inside an
    unrelated longer ASCII word. Directly exercises the normalize/match
    helpers rather than only the end-to-end resolver, so a future change to
    _normalize's whitespace handling that reopens this gap fails loudly."""

    def test_short_ascii_alias_does_not_match_inside_longer_word(self):
        normalized_text = _normalize("steak eight")
        self.assertFalse(_alias_present(normalized_text, _normalize("kei")))

    def test_short_ascii_alias_matches_as_its_own_word(self):
        normalized_text = _normalize("ブルアカのkeiが好き")
        self.assertTrue(_alias_present(normalized_text, _normalize("kei")))

    def test_ascii_alias_adjacent_to_japanese_text_without_space_still_matches(self):
        # Japanese text conventionally has no space between an ASCII
        # abbreviation and the following particle ("SAOのリーファ"); the
        # boundary check must not require a space there.
        normalized_text = _normalize("SAOのリーファ")
        self.assertTrue(_alias_present(normalized_text, _normalize("sao")))


class MultiCharacterResolutionTestCase(unittest.TestCase):
    """A single request may mention more than one known character; the
    resolver must return every one of them rather than deciding on just one
    (safe multi-character handling for Prompt Designer)."""

    def test_two_different_characters_both_resolve(self):
        results = resolve_all_character_tags("下江コハルと竜華キサキが一緒に写っている")

        character_tags = {result.character_tag for result in results}
        self.assertEqual(character_tags, {"shimoe koharu", "kisaki"})

    def test_same_character_referenced_by_multiple_aliases_is_deduplicated(self):
        results = resolve_all_character_tags("古明地こいしとkoishiとこいしが一緒にいる東方のシーン")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].character_tag, "komeiji koishi")

    def test_singular_resolver_returns_only_the_first_match(self):
        multi_text = "下江コハルと竜華キサキが一緒に写っている"
        results = resolve_all_character_tags(multi_text)
        single = resolve_character_tags(multi_text)

        self.assertIsNotNone(single)
        self.assertEqual(single.character_tag, results[0].character_tag)

    def test_no_match_returns_empty_tuple_not_none(self):
        self.assertEqual(resolve_all_character_tags("猫耳の女の子が公園にいる"), ())


class DisplayNameAndSeriesNameTestCase(unittest.TestCase):
    def test_resolution_carries_display_name_and_series_name(self):
        result = resolve_character_tags("竜華キサキ")

        self.assertIsNotNone(result)
        self.assertEqual(result.display_name, "竜華キサキ")
        self.assertEqual(result.series_name, "ブルーアーカイブ")


class KnownCharacterAndSeriesTagsIncludesNewEntriesTestCase(unittest.TestCase):
    def test_new_character_and_series_tags_are_included(self):
        tags = known_character_and_series_tags()

        for expected_tag in ("nanami chiaki", "kei", "leafa", "hu tao", "blue archive", "genshin impact"):
            self.assertIn(expected_tag, tags)


class BrokenDictionaryEntryIsolationTestCase(unittest.TestCase):
    """A malformed raw entry (missing required key, unknown series_tag) must
    be skipped with a logged warning rather than raising - so one broken
    entry can never take down the whole dictionary (or the bot) at import
    time. Exercises the builder functions directly with intentionally
    broken input."""

    def test_series_entry_missing_required_key_is_skipped_not_raised(self):
        raw = (
            {"series_tag": "ok series", "series_name": "OK", "aliases": ("ok",)},
            {"series_tag": "broken series"},  # missing "series_name" and "aliases"
        )

        built = _build_series_entries(raw)

        self.assertEqual([entry.series_tag for entry in built], ["ok series"])

    def test_character_entry_missing_required_key_is_skipped_not_raised(self):
        raw = (
            {
                "character_tag": "ok character",
                "display_name": "OK",
                "series_tag": "ok series",
                "safe_aliases": ("ok character",),
            },
            {"character_tag": "broken character"},  # missing display_name/series_tag/safe_aliases
        )

        built = _build_character_entries(raw, frozenset({"ok series"}))

        self.assertEqual([entry.character_tag for entry in built], ["ok character"])

    def test_character_entry_with_unknown_series_tag_is_skipped_not_raised(self):
        raw = (
            {
                "character_tag": "orphan character",
                "display_name": "Orphan",
                "series_tag": "nonexistent series",
                "safe_aliases": ("orphan character",),
            },
        )

        built = _build_character_entries(raw, frozenset({"ok series"}))

        self.assertEqual(built, ())


class ListCharactersTestCase(unittest.TestCase):
    def test_returns_every_registered_character_with_series_name(self):
        entries = list_characters()

        self.assertEqual(len(entries), 14)
        by_tag = {entry.character_tag: entry for entry in entries}
        self.assertEqual(by_tag["kisaki"].display_name, "竜華キサキ")
        self.assertEqual(by_tag["kisaki"].series_tag, "blue archive")
        self.assertEqual(by_tag["kisaki"].series_name, "ブルーアーカイブ")

    def test_does_not_mutate_between_calls(self):
        first = list_characters()
        second = list_characters()

        self.assertEqual(first, second)
        self.assertIsNot(first, second)


class GetCharacterDetailTestCase(unittest.TestCase):
    def test_known_character_tag_returns_full_detail(self):
        detail = get_character_detail("sparkle")

        self.assertIsNotNone(detail)
        self.assertEqual(detail.display_name, "花火")
        self.assertEqual(detail.series_tag, "honkai: star rail")
        self.assertEqual(detail.series_name, "崩壊：スターレイル")
        self.assertIn("safe_aliases", detail.extra_fields)
        self.assertIn("ambiguous_aliases", detail.extra_fields)
        self.assertIn("notes", detail.extra_fields)
        self.assertEqual(detail.extra_fields["safe_aliases"], ("sparkle",))

    def test_unknown_character_tag_returns_none(self):
        self.assertIsNone(get_character_detail("no such character"))

    def test_extra_fields_excludes_core_identity_fields(self):
        detail = get_character_detail("kei")

        self.assertIsNotNone(detail)
        self.assertNotIn("character_tag", detail.extra_fields)
        self.assertNotIn("display_name", detail.extra_fields)
        self.assertNotIn("series_tag", detail.extra_fields)


class FindCharactersByNameTestCase(unittest.TestCase):
    def test_partial_display_name_match_finds_character(self):
        matches = find_characters_by_name("こいし")

        self.assertEqual([m.character_tag for m in matches], ["komeiji koishi"])

    def test_partial_character_tag_match_finds_character(self):
        matches = find_characters_by_name("nanobana")

        self.assertEqual([m.character_tag for m in matches], ["nanobana kinako"])

    def test_series_name_alone_does_not_match_any_character(self):
        # find_characters_by_name only looks at the character's own identity
        # (display_name/character_tag/aliases), not the series - that broader
        # match is search_characters' job.
        self.assertEqual(find_characters_by_name("ブルーアーカイブ"), ())

    def test_empty_query_returns_no_matches(self):
        self.assertEqual(find_characters_by_name(""), ())

    def test_no_match_returns_empty_tuple(self):
        self.assertEqual(find_characters_by_name("該当なしのクエリ"), ())


class SearchCharactersTestCase(unittest.TestCase):
    def test_series_alias_returns_every_character_in_that_series(self):
        matches = search_characters("ブルアカ")

        self.assertEqual(
            {m.character_tag for m in matches},
            {"shimoe koharu", "kisaki", "kei"},
        )

    def test_series_name_returns_every_character_in_that_series(self):
        matches = search_characters("東方Project")

        self.assertEqual({m.character_tag for m in matches}, {"komeiji koishi"})

    def test_character_alias_still_matches(self):
        matches = search_characters("hu tao")

        self.assertEqual([m.character_tag for m in matches], ["hu tao"])

    def test_empty_query_returns_no_matches(self):
        self.assertEqual(search_characters(""), ())

    def test_no_match_returns_empty_tuple(self):
        self.assertEqual(search_characters("該当なしのクエリ"), ())


if __name__ == "__main__":
    unittest.main()
