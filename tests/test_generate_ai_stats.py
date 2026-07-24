import unittest
from unittest import mock

import discord

import commands.prompt as prompt_module
from commands.prompt import (
    SD_DENIED_MESSAGE,
    STATS_FAILED_MESSAGE,
    STATS_LOW_SAMPLE_NOTE,
    STATS_NO_HISTORY_MESSAGE,
    STATS_NO_RATINGS_NOTE,
    _build_generation_stats_embed,
    _format_percentage,
    _format_rating_line,
    _select_top_tag_phrases,
)
from services.generation_history import (
    MIN_GROUP_SIZE_FOR_TAG_COMPARISON,
    MIN_RATED_FOR_TREND,
    NORMAL_GENERATION_LABEL,
    GenerationStats,
    RatingCounts,
    RegenerationTypeStats,
    TagTrendEntry,
)


def _make_stats(**overrides) -> GenerationStats:
    defaults = dict(
        total_requests=0,
        total_images=0,
        rated_images=0,
        unrated_images=0,
        ratings_all_time=RatingCounts(),
        ratings_7d=RatingCounts(),
        ratings_30d=RatingCounts(),
        regeneration_breakdown=(),
        tag_trend=(),
        good_sample_count=0,
        low_sample_count=0,
    )
    defaults.update(overrides)
    return GenerationStats(**defaults)


class FormatPercentageTestCase(unittest.TestCase):
    def test_zero_denominator_is_a_dash_not_a_division_error(self):
        self.assertEqual(_format_percentage(0, 0), "-")

    def test_typical_ratio_is_formatted_to_one_decimal(self):
        self.assertEqual(_format_percentage(43, 82), "52.4%")

    def test_full_ratio_is_100_percent(self):
        self.assertEqual(_format_percentage(5, 5), "100.0%")


class FormatRatingLineTestCase(unittest.TestCase):
    def test_all_four_ratings_appear_in_fixed_order_with_icons(self):
        line = _format_rating_line(RatingCounts(favorite=12, like=31, meh=24, off_target=15))
        self.assertEqual(line, "⭐ 12 / 👍 31 / 👎 24 / 🎯 15")

    def test_zero_counts_are_still_shown_not_omitted(self):
        line = _format_rating_line(RatingCounts())
        self.assertEqual(line, "⭐ 0 / 👍 0 / 👎 0 / 🎯 0")


class SelectTopTagPhrasesTestCase(unittest.TestCase):
    def test_phrase_more_common_in_good_group_is_reported_as_good_leaning(self):
        tag_trend = (TagTrendEntry(phrase="natural lighting", good_count=8, low_count=1),)

        good, low = _select_top_tag_phrases(tag_trend, good_sample_count=10, low_sample_count=10)

        self.assertEqual(good, ["natural lighting"])
        self.assertEqual(low, [])

    def test_phrase_more_common_in_low_group_is_reported_as_low_leaning(self):
        tag_trend = (TagTrendEntry(phrase="strong backlight", good_count=1, low_count=9),)

        good, low = _select_top_tag_phrases(tag_trend, good_sample_count=10, low_sample_count=10)

        self.assertEqual(good, [])
        self.assertEqual(low, ["strong backlight"])

    def test_ranking_uses_share_of_each_groups_own_sample_size(self):
        # Small low-sample group (3) with 2/3 occurrence should still beat a
        # large good-sample group (20) with only 4/20 occurrence, since raw
        # counts alone would rank this backwards.
        tag_trend = (
            TagTrendEntry(phrase="rarely_in_good_often_in_low", good_count=4, low_count=2),
        )

        good, low = _select_top_tag_phrases(tag_trend, good_sample_count=20, low_sample_count=3)

        self.assertEqual(good, [])
        self.assertEqual(low, ["rarely_in_good_often_in_low"])

    def test_equal_ratio_is_neither_good_nor_low_leaning(self):
        tag_trend = (TagTrendEntry(phrase="neutral lighting", good_count=5, low_count=5),)

        good, low = _select_top_tag_phrases(tag_trend, good_sample_count=10, low_sample_count=10)

        self.assertEqual(good, [])
        self.assertEqual(low, [])

    def test_result_is_limited_to_the_configured_top_n(self):
        tag_trend = tuple(
            TagTrendEntry(phrase=f"tag_{index}", good_count=10 - index, low_count=0) for index in range(5)
        )

        good, _ = _select_top_tag_phrases(tag_trend, good_sample_count=10, low_sample_count=10, limit=3)

        self.assertEqual(len(good), 3)
        self.assertEqual(good, ["tag_0", "tag_1", "tag_2"])


class BuildGenerationStatsEmbedTestCase(unittest.TestCase):
    def test_no_history_shows_guidance_and_nothing_else(self):
        embed = _build_generation_stats_embed(_make_stats(total_requests=0))

        self.assertEqual(embed.description, STATS_NO_HISTORY_MESSAGE)
        self.assertEqual(len(embed.fields), 0)

    def test_history_without_ratings_shows_counts_and_a_no_ratings_note(self):
        stats = _make_stats(total_requests=3, total_images=10, rated_images=0, unrated_images=10)

        embed = _build_generation_stats_embed(stats)

        field_values = {field.name: field.value for field in embed.fields}
        self.assertIn("3件", field_values["全体件数"])
        self.assertIn("10枚", field_values["全体件数"])
        self.assertEqual(field_values["評価別件数"], STATS_NO_RATINGS_NOTE)

    def test_mixed_ratings_show_counts_and_percentage(self):
        stats = _make_stats(
            total_requests=40,
            total_images=156,
            rated_images=82,
            unrated_images=74,
            ratings_all_time=RatingCounts(favorite=12, like=31, meh=24, off_target=15),
        )

        embed = _build_generation_stats_embed(stats)

        field_values = {field.name: field.value for field in embed.fields}
        self.assertIn("⭐ 12 / 👍 31 / 👎 24 / 🎯 15", field_values["評価別件数"])
        self.assertIn(_format_percentage(43, 82), field_values["評価別件数"])

    def test_low_overall_sample_adds_a_caution_note_to_recent_trend(self):
        stats = _make_stats(
            total_requests=2,
            total_images=5,
            rated_images=MIN_RATED_FOR_TREND - 1,
            unrated_images=0,
            ratings_all_time=RatingCounts(like=MIN_RATED_FOR_TREND - 1),
        )

        embed = _build_generation_stats_embed(stats)

        field_values = {field.name: field.value for field in embed.fields}
        self.assertIn(STATS_LOW_SAMPLE_NOTE, field_values["直近の傾向"])

    def test_sufficient_overall_sample_has_no_caution_note_in_recent_trend(self):
        stats = _make_stats(
            total_requests=10,
            total_images=20,
            rated_images=MIN_RATED_FOR_TREND,
            unrated_images=0,
            ratings_all_time=RatingCounts(like=MIN_RATED_FOR_TREND),
        )

        embed = _build_generation_stats_embed(stats)

        field_values = {field.name: field.value for field in embed.fields}
        self.assertNotIn(STATS_LOW_SAMPLE_NOTE, field_values["直近の傾向"])

    def test_regeneration_breakdown_lists_each_type_with_low_count_marker(self):
        stats = _make_stats(
            total_requests=5,
            total_images=5,
            rated_images=1,
            unrated_images=4,
            ratings_all_time=RatingCounts(like=1),
            regeneration_breakdown=(
                RegenerationTypeStats(
                    regeneration_type=NORMAL_GENERATION_LABEL, request_count=3, ratings=RatingCounts(like=1)
                ),
                RegenerationTypeStats(
                    regeneration_type="same_prompt", request_count=1, ratings=RatingCounts()
                ),
                RegenerationTypeStats(
                    regeneration_type="change_composition", request_count=1, ratings=RatingCounts()
                ),
            ),
        )

        embed = _build_generation_stats_embed(stats)

        field_values = {field.name: field.value for field in embed.fields}
        breakdown = field_values["再生成傾向"]
        self.assertIn("通常生成: 3件", breakdown)
        self.assertIn("同じ条件で再生成: 1件", breakdown)
        self.assertIn("構図を変えて再生成: 1件", breakdown)
        # Below MIN_GROUP_SIZE_FOR_TAG_COMPARISON ratings for a type: flagged,
        # not silently presented as if it were a confident comparison.
        self.assertLess(0, breakdown.count("（件数少なめ）"))

    def test_regeneration_breakdown_lists_custom_instruction_type(self):
        stats = _make_stats(
            total_requests=1,
            total_images=1,
            rated_images=1,
            unrated_images=0,
            ratings_all_time=RatingCounts(like=1),
            regeneration_breakdown=(
                RegenerationTypeStats(
                    regeneration_type="custom_instruction", request_count=1, ratings=RatingCounts(like=1)
                ),
            ),
        )

        embed = _build_generation_stats_embed(stats)

        field_values = {field.name: field.value for field in embed.fields}
        breakdown = field_values["再生成傾向"]
        self.assertIn("指示を追加して再生成: 1件", breakdown)

    def test_tag_trend_hidden_behind_sufficient_sample_thresholds(self):
        stats = _make_stats(
            total_requests=5,
            total_images=5,
            rated_images=MIN_RATED_FOR_TREND,
            unrated_images=0,
            ratings_all_time=RatingCounts(like=MIN_RATED_FOR_TREND),
            good_sample_count=MIN_GROUP_SIZE_FOR_TAG_COMPARISON - 1,
            low_sample_count=MIN_GROUP_SIZE_FOR_TAG_COMPARISON - 1,
            tag_trend=(TagTrendEntry(phrase="natural lighting", good_count=2, low_count=0),),
        )

        embed = _build_generation_stats_embed(stats)

        field_values = {field.name: field.value for field in embed.fields}
        self.assertEqual(field_values["Prompt Designerのテキスト傾向"], STATS_LOW_SAMPLE_NOTE)

    def test_tag_trend_is_shown_once_all_thresholds_are_met(self):
        stats = _make_stats(
            total_requests=20,
            total_images=40,
            rated_images=MIN_RATED_FOR_TREND,
            unrated_images=0,
            ratings_all_time=RatingCounts(like=6, meh=4),
            good_sample_count=6,
            low_sample_count=4,
            tag_trend=(
                TagTrendEntry(phrase="natural lighting", good_count=5, low_count=0),
                TagTrendEntry(phrase="strong backlight", good_count=0, low_count=3),
            ),
        )

        embed = _build_generation_stats_embed(stats)

        field_values = {field.name: field.value for field in embed.fields}
        trend = field_values["Prompt Designerのテキスト傾向"]
        self.assertIn("natural lighting", trend)
        self.assertIn("strong backlight", trend)
        self.assertNotEqual(trend, STATS_LOW_SAMPLE_NOTE)

    def test_embed_field_values_stay_within_discord_limits(self):
        stats = _make_stats(
            total_requests=40,
            total_images=156,
            rated_images=82,
            unrated_images=74,
            ratings_all_time=RatingCounts(favorite=12, like=31, meh=24, off_target=15),
            ratings_7d=RatingCounts(like=3),
            ratings_30d=RatingCounts(like=10, meh=2),
            good_sample_count=43,
            low_sample_count=39,
            regeneration_breakdown=(
                RegenerationTypeStats(
                    regeneration_type=NORMAL_GENERATION_LABEL, request_count=30, ratings=RatingCounts(like=20)
                ),
                RegenerationTypeStats(
                    regeneration_type="same_prompt", request_count=6, ratings=RatingCounts(like=8)
                ),
                RegenerationTypeStats(
                    regeneration_type="change_composition", request_count=4, ratings=RatingCounts(off_target=3)
                ),
            ),
            tag_trend=(
                TagTrendEntry(phrase="natural lighting", good_count=20, low_count=2),
                TagTrendEntry(phrase="warm daylight", good_count=15, low_count=1),
                TagTrendEntry(phrase="strong backlight", good_count=1, low_count=18),
            ),
        )

        embed = _build_generation_stats_embed(stats)

        self.assertLessEqual(len(embed.title or ""), 256)
        for embed_field in embed.fields:
            self.assertLessEqual(len(embed_field.name), 256)
            self.assertLessEqual(len(embed_field.value), 1024)
        self.assertLessEqual(len(embed.fields), 25)


class FakeCommandTree:
    def __init__(self) -> None:
        self.commands: dict[str, object] = {}

    def command(self, *, name: str, description: str):
        del description

        def decorator(callback):
            self.commands[name] = callback
            return callback

        return decorator


class FakeClient:
    def __init__(self) -> None:
        self.tree = FakeCommandTree()


class GenerateAiStatsCommandTestCase(unittest.IsolatedAsyncioTestCase):
    def _interaction(self, user_id: int = 555):
        interaction = mock.MagicMock()
        interaction.user.id = user_id
        interaction.response.defer = mock.AsyncMock()
        interaction.followup.send = mock.AsyncMock()
        return interaction

    def _command(self):
        client = FakeClient()
        prompt_module.setup(client)
        return client.tree.commands["generate_ai_stats"]

    async def test_denied_user_is_rejected_ephemerally_without_querying_stats(self):
        command = self._command()
        interaction = self._interaction()

        with mock.patch("commands.prompt.is_pc_control_allowed", return_value=False), mock.patch(
            "commands.prompt.get_generation_stats"
        ) as stats_mock:
            await command(interaction)

        stats_mock.assert_not_called()
        interaction.followup.send.assert_awaited_once()
        _, kwargs = interaction.followup.send.call_args
        self.assertEqual(kwargs["content"], SD_DENIED_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])

    async def test_allowed_user_receives_an_ephemeral_embed(self):
        command = self._command()
        interaction = self._interaction()
        stats = _make_stats(total_requests=1, total_images=1)

        with mock.patch("commands.prompt.is_pc_control_allowed", return_value=True), mock.patch(
            "commands.prompt.get_generation_stats", return_value=stats
        ) as stats_mock:
            await command(interaction)

        stats_mock.assert_called_once_with(555)
        interaction.followup.send.assert_awaited_once()
        _, kwargs = interaction.followup.send.call_args
        self.assertIsInstance(kwargs["embed"], discord.Embed)
        self.assertTrue(kwargs["ephemeral"])

    async def test_db_failure_reports_a_short_message_without_leaking_detail(self):
        command = self._command()
        interaction = self._interaction()

        with mock.patch("commands.prompt.is_pc_control_allowed", return_value=True), mock.patch(
            "commands.prompt.get_generation_stats", return_value=None
        ):
            await command(interaction)

        _, kwargs = interaction.followup.send.call_args
        self.assertEqual(kwargs["content"], STATS_FAILED_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])
        for leaked in ("Traceback", "sqlite3", "OperationalError"):
            self.assertNotIn(leaked, kwargs["content"])

    async def test_unexpected_exception_does_not_propagate(self):
        command = self._command()
        interaction = self._interaction()

        with mock.patch("commands.prompt.is_pc_control_allowed", return_value=True), mock.patch(
            "commands.prompt.get_generation_stats", side_effect=RuntimeError("db exploded")
        ):
            await command(interaction)  # must not raise

        _, kwargs = interaction.followup.send.call_args
        self.assertEqual(kwargs["content"], STATS_FAILED_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])


if __name__ == "__main__":
    unittest.main()
