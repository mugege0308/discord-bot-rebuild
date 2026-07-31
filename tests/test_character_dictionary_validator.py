import unittest

from services.character_dictionary_validator import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    validate_character_dictionary,
)
from services.character_tag_dictionary import CharacterDetail, SeriesListEntry


def _character(
    character_tag: str,
    display_name: str,
    series_tag: str = "s1",
    series_name: str = "作品1",
    safe_aliases: tuple[str, ...] = (),
    ambiguous_aliases: tuple[str, ...] = (),
    notes: str = "",
    extra: dict | None = None,
) -> CharacterDetail:
    extra_fields = {"safe_aliases": safe_aliases, "ambiguous_aliases": ambiguous_aliases, "notes": notes}
    if extra:
        extra_fields.update(extra)
    return CharacterDetail(
        character_tag=character_tag,
        display_name=display_name,
        series_tag=series_tag,
        series_name=series_name,
        extra_fields=extra_fields,
    )


def _series(series_tag: str = "s1", series_name: str = "作品1", aliases: tuple[str, ...] = ("作品1エイリアス",)) -> SeriesListEntry:
    return SeriesListEntry(series_tag=series_tag, series_name=series_name, aliases=aliases)


def _codes(report, severity: str | None = None) -> set[str]:
    return {issue.code for issue in report.issues if severity is None or issue.severity == severity}


class RealDictionaryHasNoErrorsTestCase(unittest.TestCase):
    def test_real_dictionary_has_zero_errors(self):
        report = validate_character_dictionary()

        self.assertEqual(report.error_count, 0, msg=[i.message for i in report.issues if i.severity == SEVERITY_ERROR])
        self.assertTrue(report.is_usable)
        self.assertGreater(report.character_count, 0)
        self.assertGreater(report.series_count, 0)


class DuplicateDisplayNameTestCase(unittest.TestCase):
    def test_duplicate_display_name_is_warning_not_error(self):
        characters = (
            _character("char-a", "同じ名前", safe_aliases=("char-a-alias",)),
            _character("char-b", "同じ名前", safe_aliases=("char-b-alias",)),
        )
        report = validate_character_dictionary(characters=characters, series=(_series(),))

        self.assertIn("duplicate_display_name", _codes(report, SEVERITY_WARNING))
        self.assertNotIn("duplicate_display_name", _codes(report, SEVERITY_ERROR))


class DuplicateCharacterTagTestCase(unittest.TestCase):
    def test_duplicate_character_tag_is_error(self):
        characters = (
            _character("dup-tag", "名前A", safe_aliases=("alias-a",)),
            _character("dup-tag", "名前B", safe_aliases=("alias-b",)),
        )
        report = validate_character_dictionary(characters=characters, series=(_series(),))

        self.assertIn("duplicate_character_tag", _codes(report, SEVERITY_ERROR))
        self.assertFalse(report.is_usable)


class CrossCharacterAliasCollisionTestCase(unittest.TestCase):
    def test_same_alias_on_two_characters_is_error(self):
        characters = (
            _character("char-a", "名前A", safe_aliases=("shared alias",)),
            _character("char-b", "名前B", safe_aliases=("shared alias",)),
        )
        report = validate_character_dictionary(characters=characters, series=(_series(),))

        self.assertIn("cross_character_alias_collision", _codes(report, SEVERITY_ERROR))

    def test_alias_collision_after_nfkc_casefold_normalization_is_still_caught(self):
        # "ABC" vs "abc" are different raw strings but the same alias once
        # normalized the same way resolve_character_tags does.
        characters = (
            _character("char-a", "名前A", safe_aliases=("ABC",)),
            _character("char-b", "名前B", safe_aliases=("abc",)),
        )
        report = validate_character_dictionary(characters=characters, series=(_series(),))

        self.assertIn("cross_character_alias_collision", _codes(report, SEVERITY_ERROR))


class SafeAmbiguousAliasOverlapTestCase(unittest.TestCase):
    def test_same_alias_in_both_lists_is_warning(self):
        characters = (
            _character("char-a", "名前A", safe_aliases=("x",), ambiguous_aliases=("x",)),
        )
        report = validate_character_dictionary(characters=characters, series=(_series(),))

        self.assertIn("safe_ambiguous_alias_overlap", _codes(report, SEVERITY_WARNING))
        self.assertEqual(report.error_count, 0)


class EmptyStringFieldTestCase(unittest.TestCase):
    def test_empty_character_tag_is_error(self):
        characters = (_character("", "名前A", safe_aliases=("alias-a",)),)
        report = validate_character_dictionary(characters=characters, series=(_series(),))

        self.assertIn("empty_core_identity_field", _codes(report, SEVERITY_ERROR))

    def test_empty_series_tag_is_error(self):
        series = (_series(series_tag=""),)
        report = validate_character_dictionary(characters=(), series=series)

        self.assertIn("empty_core_identity_field", _codes(report, SEVERITY_ERROR))

    def test_empty_alias_is_warning(self):
        characters = (_character("char-a", "名前A", safe_aliases=("valid", "")),)
        report = validate_character_dictionary(characters=characters, series=(_series(),))

        self.assertIn("empty_alias", _codes(report, SEVERITY_WARNING))
        self.assertEqual(report.error_count, 0)


class UnconfirmedNotesTestCase(unittest.TestCase):
    def test_notes_with_unconfirmed_keyword_is_warning(self):
        characters = (
            _character("char-a", "名前A", safe_aliases=("alias-a",), notes="character_tagは未確認。要確認。"),
        )
        report = validate_character_dictionary(characters=characters, series=(_series(),))

        self.assertIn("unconfirmed_notes", _codes(report, SEVERITY_WARNING))

    def test_notes_without_unconfirmed_keyword_is_not_flagged(self):
        characters = (
            _character("char-a", "名前A", safe_aliases=("alias-a",), notes="確認済みの公式名です。"),
        )
        report = validate_character_dictionary(characters=characters, series=(_series(),))

        self.assertNotIn("unconfirmed_notes", _codes(report))


class FutureFieldDoesNotBreakValidatorTestCase(unittest.TestCase):
    def test_unknown_extra_field_is_ignored_without_raising(self):
        characters = (
            _character(
                "char-a",
                "名前A",
                safe_aliases=("alias-a",),
                extra={"appearance_tags": ("long hair",), "identity_hints": ()},
            ),
        )

        report = validate_character_dictionary(characters=characters, series=(_series(),))

        self.assertEqual(report.error_count, 0)
        self.assertEqual(report.character_count, 1)


class EmptySafeAliasesTestCase(unittest.TestCase):
    def test_character_with_no_safe_aliases_is_warning(self):
        characters = (_character("char-a", "名前A", safe_aliases=(), ambiguous_aliases=("x",)),)
        report = validate_character_dictionary(characters=characters, series=(_series(),))

        self.assertIn("empty_safe_aliases", _codes(report, SEVERITY_WARNING))


class SeriesAliasAvailabilityTestCase(unittest.TestCase):
    def test_ambiguous_character_with_no_series_aliases_is_warning(self):
        characters = (_character("char-a", "名前A", series_tag="lonely", ambiguous_aliases=("x",)),)
        series = (SeriesListEntry(series_tag="lonely", series_name="孤立作品", aliases=()),)
        report = validate_character_dictionary(characters=characters, series=series)

        self.assertIn("series_alias_missing_for_ambiguous_character", _codes(report, SEVERITY_WARNING))


class SeriesAliasCollisionTestCase(unittest.TestCase):
    def test_same_series_alias_on_two_series_is_error(self):
        series = (
            SeriesListEntry(series_tag="s1", series_name="作品1", aliases=("shared",)),
            SeriesListEntry(series_tag="s2", series_name="作品2", aliases=("shared",)),
        )
        report = validate_character_dictionary(characters=(), series=series)

        self.assertIn("cross_series_alias_collision", _codes(report, SEVERITY_ERROR))

    def test_duplicate_alias_within_one_series_after_normalization_is_warning(self):
        series = (SeriesListEntry(series_tag="s1", series_name="作品1", aliases=("Foo", "foo")),)
        report = validate_character_dictionary(characters=(), series=series)

        self.assertIn("duplicate_series_alias", _codes(report, SEVERITY_WARNING))


if __name__ == "__main__":
    unittest.main()
