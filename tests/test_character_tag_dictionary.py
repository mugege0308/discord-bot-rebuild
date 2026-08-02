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
    ("kei-abbrev-series", "ブルアカのケイ", "kei (blue archive)", "blue archive"),
    ("kei-uppercase-abbrev-series", "ブルアカのKEI", "kei (blue archive)", "blue archive"),
    ("kei-student-explicit-no-series", "生徒ケイ", "kei (blue archive)", "blue archive"),
    ("kei-student-explicit-alt-order-no-series", "ケイ 生徒", "kei (blue archive)", "blue archive"),
    ("kei-former-explicit-no-series", "旧ケイ", "kei (former) (blue archive)", "blue archive"),
    ("kei-former-explicit-phrase-no-series", "アリス身体のケイ", "kei (former) (blue archive)", "blue archive"),
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
    ("chiaki-motomiya-full-name", "元宮チアキ", "chiaki (blue archive)", "blue archive"),
    ("chiaki-motomiya-romaji", "motomiya chiaki", "chiaki (blue archive)", "blue archive"),
    ("chiaki-motomiya-abbrev-series", "ブルアカのチアキ", "chiaki (blue archive)", "blue archive"),
    ("hinana-full-name", "市川雛菜", "ichikawa hinana", "idolmaster shiny colors"),
    ("hinana-romaji", "ichikawa hinana", "ichikawa hinana", "idolmaster shiny colors"),
    ("hinana-abbrev-series", "シャニマスの雛菜", "ichikawa hinana", "idolmaster shiny colors"),
    ("natsume-full-name", "四季ナツメ", "shiki natsume", "cafe stella to shinigami no chou"),
    ("natsume-romaji", "shiki natsume", "shiki natsume", "cafe stella to shinigami no chou"),
    ("natsume-abbrev-series", "喫茶ステラの四季ナツメ", "shiki natsume", "cafe stella to shinigami no chou"),
    ("raika-full-name", "忍原来夏", "shinohara raika", "inazuma eleven victory road"),
    ("raika-romaji", "shinohara raika", "shinohara raika", "inazuma eleven victory road"),
    ("raika-abbrev-series", "イナイレVの来夏", "shinohara raika", "inazuma eleven victory road"),
    ("castorice-japanese", "キャストリス", "castorice", "honkai: star rail"),
    ("castorice-romaji-standalone", "castorice", "castorice", "honkai: star rail"),
    ("castorice-with-series", "崩壊スターレイルのキャストリス", "castorice", "honkai: star rail"),
    ("remielle-full-japanese", "レミエール・ダン", "remielle dan", "zenless zone zero"),
    ("remielle-short-japanese", "レミエール", "remielle dan", "zenless zone zero"),
    ("remielle-romaji-standalone", "remielle", "remielle dan", "zenless zone zero"),
    ("remielle-with-series", "ゼンゼロのレミエール", "remielle dan", "zenless zone zero"),
)


class NewCharacterResolutionTableTestCase(unittest.TestCase):
    def test_each_case_resolves_to_the_expected_tags(self):
        for case_name, input_text, expected_character_tag, expected_series_tag in _RESOLVES_CASES:
            with self.subTest(case=case_name):
                result = resolve_character_tags(input_text)

                self.assertIsNotNone(result, f"expected {input_text!r} to resolve")
                self.assertEqual(result.character_tag, expected_character_tag)
                self.assertEqual(result.series_tag, expected_series_tag)


class KeiStudentAndFormerSplitTestCase(unittest.TestCase):
    """The single, ambiguous "kei (blue archive)" entry was split into two:
    "kei (blue archive)" (2026's student-ver. Kei, the common case) and
    "kei (former) (blue archive)" (the old Alice-body-side Kei), reachable
    only via its own explicit aliases. Covers the exact scenarios requested
    when the split was designed: bare ambiguous resolution is unchanged,
    student-ver. resolves under series/explicit-context, the former form
    resolves only via its own explicit aliases, and mentioning both an
    explicit former-Kei alias and the series name together must resolve to
    the former only - not both at once.
    """

    def test_bare_kei_alone_still_does_not_resolve(self):
        # Unchanged from before the split: "kei"/"ケイ" alone remain too
        # generic to resolve without series or another explicit alias.
        self.assertIsNone(resolve_character_tags("kei"))
        self.assertIsNone(resolve_character_tags("ケイ"))

    def test_series_qualified_bare_kei_resolves_to_student_version(self):
        result = resolve_character_tags("ブルアカのケイ")

        self.assertIsNotNone(result)
        self.assertEqual(result.character_tag, "kei (blue archive)")
        self.assertEqual(
            result.signature_tags,
            ("white hair", "very long hair", "pink eyes", "sharp eyes", "asymmetrical bangs"),
        )

    def test_student_explicit_aliases_resolve_without_series(self):
        for text in ("生徒ケイ", "ケイ 生徒"):
            with self.subTest(text=text):
                result = resolve_character_tags(text)

                self.assertIsNotNone(result)
                self.assertEqual(result.character_tag, "kei (blue archive)")

    def test_former_explicit_aliases_resolve_without_series(self):
        for text in ("旧ケイ", "アリス身体のケイ"):
            with self.subTest(text=text):
                result = resolve_character_tags(text)

                self.assertIsNotNone(result)
                self.assertEqual(result.character_tag, "kei (former) (blue archive)")
                self.assertEqual(result.signature_tags, ())

    def test_former_alias_with_series_resolves_only_to_former_not_both(self):
        # Alias collision case: "旧ケイ"/"アリス身体のケイ" both literally
        # contain "ケイ" (Japanese matching is plain substring containment),
        # so student-ver. Kei's own ambiguous alias "ケイ" is also present in
        # the same text. Combined with a series mention, both characters'
        # match conditions would technically be satisfied at once; the
        # dictionary must resolve this to the former only.
        for text in ("ブルアカの旧ケイ", "ブルアカのアリス身体のケイ", "blue archive 旧ケイ"):
            with self.subTest(text=text):
                results = resolve_all_character_tags(text)

                self.assertEqual(len(results), 1, f"expected exactly one resolution for {text!r}: {results!r}")
                self.assertEqual(results[0].character_tag, "kei (former) (blue archive)")

    def test_two_genuinely_different_blue_archive_characters_still_both_resolve(self):
        # Confirms the alias-subsumption safety valve added for the two kei
        # entries does not suppress an unrelated pair of characters from the
        # same series when mentioned together (pre-existing behavior).
        results = resolve_all_character_tags("下江コハルと竜華キサキが一緒に写っている")

        self.assertEqual({r.character_tag for r in results}, {"shimoe koharu", "kisaki"})

    def test_get_character_detail_exposes_signature_tags_only_for_student_version(self):
        student_detail = get_character_detail("kei (blue archive)")
        former_detail = get_character_detail("kei (former) (blue archive)")

        self.assertIsNotNone(student_detail)
        self.assertIsNotNone(former_detail)
        self.assertEqual(
            student_detail.extra_fields["signature_tags"],
            ("white hair", "very long hair", "pink eyes", "sharp eyes", "asymmetrical bangs"),
        )
        self.assertEqual(former_detail.extra_fields["signature_tags"], ())


class NewSixCharacterBatchTestCase(unittest.TestCase):
    """The 6-character batch added for character-dictionary expansion:
    元宮チアキ (chiaki (blue archive)), 市川雛菜 (ichikawa hinana),
    四季ナツメ (shiki natsume), 忍原来夏 (shinohara raika), キャストリス
    (castorice), レミエール・ダン (remielle dan). Covers the specific risks
    identified while designing the batch: "chiaki" must not collide with the
    pre-existing "nanami chiaki" (Danganronpa), the two "inazuma eleven"
    series_tags must stay distinct, and signature_tags reach
    CharacterTagResolution/get_character_detail correctly (including the
    empty-tuple case for 市川雛菜, whose appearance is unconfirmed).
    """

    def test_chiaki_motomiya_full_name_does_not_collide_with_nanami_chiaki(self):
        # Realistic inputs only: nanami chiaki has no bare "チアキ"/"chiaki"
        # alias registered (safe_aliases only), so it can only ever resolve
        # via its own full-name aliases, never via chiaki_motomiya's bare
        # ambiguous "チアキ"/"chiaki".
        motomiya = resolve_character_tags("元宮チアキ")
        self.assertIsNotNone(motomiya)
        self.assertEqual(motomiya.character_tag, "chiaki (blue archive)")

        nanami = resolve_character_tags("七海千秋")
        self.assertIsNotNone(nanami)
        self.assertEqual(nanami.character_tag, "nanami chiaki")

        nanami_romaji = resolve_character_tags("chiaki nanami")
        self.assertIsNotNone(nanami_romaji)
        self.assertEqual(nanami_romaji.character_tag, "nanami chiaki")

        # Bare "チアキ"/"chiaki" without series or full-name context resolves
        # to neither character.
        self.assertIsNone(resolve_character_tags("チアキ"))
        self.assertIsNone(resolve_character_tags("chiaki"))

        # "ダンガンロンパのチアキ" does not resolve nanami chiaki (no bare
        # ambiguous alias registered for it) and must not accidentally
        # resolve chiaki_motomiya either (wrong series).
        self.assertIsNone(resolve_character_tags("ダンガンロンパのチアキ"))

        # "chiaki nanami" alone (no blue archive series alias present) must
        # not also resolve chiaki_motomiya via its ambiguous "chiaki",
        # despite "chiaki" being a literal substring of "chiaki nanami".
        results = resolve_all_character_tags("chiaki nanami")
        self.assertEqual({r.character_tag for r in results}, {"nanami chiaki"})

    def test_inazuma_eleven_go_and_victory_road_stay_separate_series(self):
        raika = resolve_character_tags("イナイレVの来夏")
        self.assertIsNotNone(raika)
        self.assertEqual(raika.series_tag, "inazuma eleven victory road")

        kinako = resolve_character_tags("イナイレGOの菜花きなこ")
        self.assertIsNotNone(kinako)
        self.assertEqual(kinako.series_tag, "inazuma eleven go")

        # raika's bare given name "来夏" is an ambiguous_alias (no safe_alias
        # match here), so pairing it with the WRONG series (GO, not Victory
        # Road) must not resolve at all.
        self.assertIsNone(resolve_character_tags("イナイレGOの来夏"))

        # nanobana_kinako's full name IS a safe_alias, so she still resolves
        # even when paired with the wrong series text - but her series_tag
        # must stay her own real "inazuma eleven go", never get silently
        # overwritten by the unrelated "inazuma eleven victory road" mention
        # (see test_wrong_series_text_never_overwrites_a_safe_alias_characters_series_tag
        # below for the general-purpose regression this depends on).
        kinako_wrong_series_text = resolve_character_tags("イナイレVの菜花きなこ")
        self.assertIsNotNone(kinako_wrong_series_text)
        self.assertEqual(kinako_wrong_series_text.character_tag, "nanobana kinako")
        self.assertEqual(kinako_wrong_series_text.series_tag, "inazuma eleven go")
        self.assertEqual(kinako_wrong_series_text.match_type, "character_only")

    def test_wrong_series_text_never_overwrites_a_safe_alias_characters_series_tag(self):
        # General-purpose regression (not specific to the inazuma eleven
        # pair): a safe_alias match must always report the character's own,
        # fixed series_tag - resolving via a full name must never let an
        # unrelated series mention elsewhere in the same text corrupt it.
        # Exercised here with genshin impact's hu tao (English "hu tao" is a
        # safe_alias) paired with an unrelated but registered series alias
        # ("pokemon").
        result = resolve_character_tags("pokemon hu tao")

        self.assertIsNotNone(result)
        self.assertEqual(result.character_tag, "hu tao")
        self.assertEqual(result.series_tag, "genshin impact")
        self.assertEqual(result.match_type, "character_only")

    def test_natsume_and_raika_ambiguous_aliases_require_series(self):
        for text in ("ナツメ", "natsume", "らいか", "raika", "来夏"):
            with self.subTest(text=text):
                self.assertIsNone(resolve_character_tags(text))

    def test_signature_tags_reach_resolution_for_each_new_character(self):
        expectations = (
            ("元宮チアキ", "chiaki (blue archive)", ("long hair", "horns")),
            ("四季ナツメ", "shiki natsume", ("black hair", "long hair")),
            ("忍原来夏", "shinohara raika", ("pink hair", "ponytail", "pink eyes")),
            ("キャストリス", "castorice", ("purple hair", "very long hair", "purple eyes", "pointy ears")),
            ("レミエール・ダン", "remielle dan", ("pink hair", "very long hair", "pink eyes", "wings")),
        )
        for text, expected_tag, expected_signature_tags in expectations:
            with self.subTest(text=text):
                result = resolve_character_tags(text)

                self.assertIsNotNone(result)
                self.assertEqual(result.character_tag, expected_tag)
                self.assertEqual(result.signature_tags, expected_signature_tags)

    def test_hinana_has_empty_signature_tags_and_still_resolves(self):
        result = resolve_character_tags("市川雛菜")

        self.assertIsNotNone(result)
        self.assertEqual(result.character_tag, "ichikawa hinana")
        self.assertEqual(result.signature_tags, ())

    def test_get_character_detail_exposes_signature_tags_for_new_batch(self):
        expectations = {
            "chiaki (blue archive)": ("long hair", "horns"),
            "ichikawa hinana": (),
            "shiki natsume": ("black hair", "long hair"),
            "shinohara raika": ("pink hair", "ponytail", "pink eyes"),
            "castorice": ("purple hair", "very long hair", "purple eyes", "pointy ears"),
            "remielle dan": ("pink hair", "very long hair", "pink eyes", "wings"),
        }
        for character_tag, expected_signature_tags in expectations.items():
            with self.subTest(character_tag=character_tag):
                detail = get_character_detail(character_tag)

                self.assertIsNotNone(detail)
                self.assertEqual(detail.extra_fields["signature_tags"], expected_signature_tags)


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
    ("chiaki-motomiya-bare-japanese", "チアキ"),
    ("chiaki-motomiya-bare-romaji", "chiaki"),
    ("hinana-bare-kanji", "雛菜"),
    ("hinana-bare-katakana", "ひなな"),
    ("hinana-bare-romaji", "hinana"),
    ("natsume-bare-katakana", "ナツメ"),
    ("natsume-bare-romaji", "natsume"),
    ("raika-bare-kanji", "来夏"),
    ("raika-bare-hiragana", "らいか"),
    ("raika-bare-romaji", "raika"),
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

        for expected_tag in (
            "nanami chiaki", "kei (blue archive)", "leafa", "hu tao", "blue archive", "genshin impact"
        ):
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

        self.assertEqual(len(entries), 21)
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
        detail = get_character_detail("kei (blue archive)")

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
            {
                "shimoe koharu",
                "kisaki",
                "kei (blue archive)",
                "kei (former) (blue archive)",
                "chiaki (blue archive)",
            },
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
