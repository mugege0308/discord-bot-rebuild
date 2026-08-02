import unittest
from unittest import mock

from services.character_missing_finder import (
    SAMPLE_REQUEST_TEXT_MAX_LENGTH,
    find_missing_character_candidates,
)
from services.generation_history import RequestTextRow


def _row(id_: int, created_at: str, request_text: str, user_id: int = 1) -> RequestTextRow:
    return RequestTextRow(id=id_, created_at=created_at, user_id=user_id, request_text=request_text)


class AggregatesBlockedMentionsTestCase(unittest.TestCase):
    def test_repeated_series_gated_mention_is_counted_and_ranked(self):
        rows = [
            _row(1, "2026-01-01T10:00:00+09:00", "スズナがいる公園"),
            _row(2, "2026-01-02T10:00:00+09:00", "スズナと歩く"),
            _row(3, "2026-01-03T10:00:00+09:00", "スズナが笑う"),
        ]

        candidates = find_missing_character_candidates(rows=rows)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].character_tag, "suzuna")
        self.assertEqual(candidates[0].occurrence_count, 3)

    def test_already_resolvable_mention_is_excluded(self):
        # "ポケモンのスズナ" resolves fully (series co-mentioned), so it must
        # never appear as a missing candidate.
        rows = [_row(1, "2026-01-01T10:00:00+09:00", "ポケモンのスズナ")]

        candidates = find_missing_character_candidates(rows=rows)

        self.assertEqual(candidates, [])

    def test_ordinary_sentence_produces_no_candidates(self):
        rows = [_row(1, "2026-01-01T10:00:00+09:00", "猫耳の女の子が公園にいる、構図はバストアップで")]

        candidates = find_missing_character_candidates(rows=rows)

        self.assertEqual(candidates, [])

    def test_no_rows_returns_empty_list(self):
        self.assertEqual(find_missing_character_candidates(rows=[]), [])

    def test_different_aliases_of_the_same_character_are_aggregated_together(self):
        # "ケイ" (katakana) and "kei" (romaji) are both registered aliases of
        # the same character_tag - this is the "表記揺れの集約" this module
        # supports (aggregating by character_tag, not by literal alias text).
        rows = [
            _row(1, "2026-01-01T10:00:00+09:00", "ケイが街にいる"),
            _row(2, "2026-01-02T10:00:00+09:00", "a portrait of kei walking"),
        ]

        candidates = find_missing_character_candidates(rows=rows)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].character_tag, "kei (blue archive)")
        self.assertEqual(candidates[0].occurrence_count, 2)

    def test_former_kei_fully_resolved_with_series_produces_no_candidate(self):
        # "ブルアカの旧ケイ" fully resolves to the former-Kei entry (its own
        # explicit alias). The generic student-Kei entry's ambiguous alias
        # "ケイ" is also a literal substring of "旧ケイ" and the series name
        # is present too, but this must not be reported as a "series not
        # co-mentioned" candidate for student-Kei - it was correctly resolved
        # elsewhere (subsumed by the more specific former-Kei match), not
        # blocked by a missing series.
        rows = [_row(1, "2026-01-01T10:00:00+09:00", "ブルアカの旧ケイが立っている")]

        candidates = find_missing_character_candidates(rows=rows)

        self.assertEqual(candidates, [])

    def test_multiple_characters_are_sorted_by_occurrence_count_descending(self):
        rows = (
            [_row(i, f"2026-01-0{i}T10:00:00+09:00", "スズナがいる") for i in range(1, 4)]
            + [_row(10, "2026-02-01T10:00:00+09:00", "リーファが戦う")]
        )

        candidates = find_missing_character_candidates(rows=rows)

        self.assertEqual([c.character_tag for c in candidates], ["suzuna", "leafa"])
        self.assertEqual(candidates[0].occurrence_count, 3)
        self.assertEqual(candidates[1].occurrence_count, 1)

    def test_last_used_at_is_the_most_recent_occurrence_regardless_of_row_order(self):
        rows = [
            _row(1, "2026-01-05T10:00:00+09:00", "スズナが笑う（新しい）"),
            _row(2, "2026-01-01T10:00:00+09:00", "スズナが歩く（古い）"),
        ]

        candidates = find_missing_character_candidates(rows=rows)

        self.assertEqual(candidates[0].last_used_at, "2026-01-05T10:00:00+09:00")
        self.assertIn("新しい", candidates[0].sample_request_text)

    def test_sample_request_text_is_truncated_not_full_raw_text(self):
        long_text = "スズナが" + ("あ" * 500) + "という長い依頼文の中に登場する"
        rows = [_row(1, "2026-01-01T10:00:00+09:00", long_text)]

        candidates = find_missing_character_candidates(rows=rows)

        self.assertLess(len(candidates[0].sample_request_text), len(long_text))
        self.assertLessEqual(len(candidates[0].sample_request_text), SAMPLE_REQUEST_TEXT_MAX_LENGTH + 1)

    def test_result_respects_limit(self):
        rows = [
            _row(1, "2026-01-01T10:00:00+09:00", "スズナがいる"),
            _row(2, "2026-01-01T10:00:00+09:00", "リーファが戦う"),
            _row(3, "2026-01-01T10:00:00+09:00", "ケイが立つ"),
        ]

        candidates = find_missing_character_candidates(rows=rows, limit=2)

        self.assertEqual(len(candidates), 2)

    def test_reason_names_the_missing_series(self):
        rows = [_row(1, "2026-01-01T10:00:00+09:00", "スズナがいる")]

        candidates = find_missing_character_candidates(rows=rows)

        self.assertIn("ポケットモンスター", candidates[0].reason)

    def test_history_read_failure_returns_empty_list_without_raising(self):
        with mock.patch(
            "services.character_missing_finder.list_request_texts",
            side_effect=RuntimeError("db exploded"),
        ):
            self.assertEqual(find_missing_character_candidates(), [])

    def test_row_scan_failure_is_skipped_not_fatal(self):
        rows = [
            _row(1, "2026-01-01T10:00:00+09:00", "スズナがいる"),
            _row(2, "2026-01-02T10:00:00+09:00", "スズナが歩く"),
        ]
        real_find = __import__(
            "services.character_tag_dictionary", fromlist=["find_blocked_alias_mentions"]
        ).find_blocked_alias_mentions

        def flaky(text):
            if "歩く" in text:
                raise RuntimeError("boom")
            return real_find(text)

        with mock.patch("services.character_missing_finder.find_blocked_alias_mentions", side_effect=flaky):
            candidates = find_missing_character_candidates(rows=rows)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].occurrence_count, 1)


if __name__ == "__main__":
    unittest.main()
