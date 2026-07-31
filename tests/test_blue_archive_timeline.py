import math
import unittest
from decimal import Decimal

from services.blue_archive_timeline import (
    MAX_TIMELINE_ACTIONS,
    TimelineAction,
    TimelineConfig,
    TimelineValidationError,
    format_cost,
    format_remaining_time,
    parse_remaining_time,
    parse_timeline,
    simulate_timeline,
)


def _config(**overrides):
    values = {
        "battle_time": 240,
        "start_cost": 0,
        "max_cost": 10,
        "cost_per_second": "0.7",
        "recovery_multiplier": 1,
    }
    values.update(overrides)
    return TimelineConfig.from_values(**values)


class BlueArchiveTimelineParsingTestCase(unittest.TestCase):
    def test_parse_time_supports_integer_and_fractional_seconds(self):
        self.assertEqual(parse_remaining_time("3:55"), Decimal("235"))
        self.assertEqual(parse_remaining_time("1:02.5"), Decimal("62.5"))

    def test_parse_timeline_supports_names_spaces_actual_cost_and_memo(self):
        actions = parse_timeline("3:55 水着 ホシノ 6/5 | コスト軽減後")

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].display_name, "水着 ホシノ")
        self.assertEqual(actions[0].base_cost, Decimal("6"))
        self.assertEqual(actions[0].actual_cost, Decimal("5"))
        self.assertEqual(actions[0].memo, "コスト軽減後")
        self.assertEqual(actions[0].line_number, 1)

    def test_pipe_separated_columns_support_names_spaces_and_memo(self):
        actions = parse_timeline(
            "3:55 | 水着 ホシノ | 6/5 | コスト軽減後; "
            "3:51 | アコ | 3"
        )

        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0].display_name, "水着 ホシノ")
        self.assertEqual(actions[0].base_cost, Decimal("6"))
        self.assertEqual(actions[0].actual_cost, Decimal("5"))
        self.assertEqual(actions[0].memo, "コスト軽減後")
        self.assertEqual(actions[1].display_name, "アコ")
        self.assertEqual(actions[1].actual_cost, Decimal("3"))

    def test_semicolon_and_blank_lines_are_supported(self):
        actions = parse_timeline("3:55 ヒマリ 3;\n\n3:51 アコ 3")

        self.assertEqual([action.display_name for action in actions], ["ヒマリ", "アコ"])

    def test_time_format_and_seconds_range_are_validated(self):
        for value in ("235", "3:5", "3:60", "-1:00", "abc"):
            with self.subTest(value=value):
                with self.assertRaises(TimelineValidationError):
                    parse_remaining_time(value)

    def test_malformed_line_and_cost_are_rejected_with_line_number(self):
        with self.assertRaisesRegex(TimelineValidationError, "2行目"):
            parse_timeline("3:55 ヒマリ 3\ninvalid")
        with self.assertRaisesRegex(TimelineValidationError, "1行目"):
            parse_timeline("3:55 ヒマリ 3/2/1")
        with self.assertRaisesRegex(TimelineValidationError, "3列形式"):
            parse_timeline("3:55 | ヒマリ | 3 | メモ | 余分")

    def test_empty_and_oversized_timeline_are_rejected(self):
        with self.assertRaisesRegex(TimelineValidationError, "1行以上"):
            parse_timeline(" \n ")

        too_many = "\n".join(f"3:55 生徒{i} 1" for i in range(MAX_TIMELINE_ACTIONS + 1))
        with self.assertRaisesRegex(TimelineValidationError, f"{MAX_TIMELINE_ACTIONS}行以内"):
            parse_timeline(too_many)

    def test_display_format_rounds_only_for_output(self):
        self.assertEqual(format_cost(Decimal("1.235")), "1.24")
        self.assertEqual(format_cost(Decimal("1.2")), "1.2")
        self.assertEqual(format_remaining_time(Decimal("62.54")), "1:02.5")
        self.assertEqual(format_remaining_time(Decimal("-0.06")), "-0:00.1")


class BlueArchiveTimelineSimulationTestCase(unittest.TestCase):
    def test_requested_example_detects_shortage_without_consuming_cost(self):
        actions = parse_timeline(
            "\n".join(
                (
                    "3:55 ヒマリ 3",
                    "3:51 アコ 3",
                    "3:47 ドレスヒナ 6",
                    "3:38 ココナ 2",
                )
            )
        )

        result = simulate_timeline(actions, _config())

        first, second, third, fourth = result.actions
        self.assertEqual((first.cost_before, first.cost_spent, first.cost_after), (
            Decimal("3.5"),
            Decimal("3"),
            Decimal("0.5"),
        ))
        self.assertEqual((second.cost_before, second.cost_after), (Decimal("3.3"), Decimal("0.3")))
        self.assertFalse(third.is_available)
        self.assertEqual(third.cost_before, Decimal("3.1"))
        self.assertEqual(third.cost_spent, Decimal("0"))
        self.assertEqual(third.cost_after, Decimal("3.1"))
        self.assertEqual(third.shortage, Decimal("2.9"))
        self.assertEqual(
            third.earliest_available_time,
            Decimal("227") - Decimal("2.9") / Decimal("0.7"),
        )
        self.assertTrue(fourth.is_available)
        self.assertEqual(fourth.cost_before, Decimal("9.4"))
        self.assertEqual(fourth.cost_after, Decimal("7.4"))
        self.assertFalse(result.is_valid)
        self.assertEqual(result.total_recovered_cost, Decimal("15.4"))
        self.assertEqual(result.total_spent_cost, Decimal("8"))
        self.assertEqual(result.total_overflow_cost, Decimal("0"))
        self.assertEqual(result.final_cost, Decimal("7.4"))

    def test_overflow_is_recorded_separately_from_gross_recovery(self):
        actions = parse_timeline("3:50 ヒマリ 3")
        result = simulate_timeline(
            actions,
            _config(start_cost=9, cost_per_second=1),
        )

        action = result.actions[0]
        self.assertEqual(action.cost_before, Decimal("10"))
        self.assertEqual(action.overflow_cost, Decimal("9"))
        self.assertEqual(action.cost_after, Decimal("7"))
        self.assertEqual(result.total_recovered_cost, Decimal("10"))
        self.assertEqual(result.total_overflow_cost, Decimal("9"))

    def test_same_timestamp_is_processed_in_input_order_without_recovery(self):
        result = simulate_timeline(
            parse_timeline("3:55 ヒマリ 3\n3:55 アコ 3"),
            _config(start_cost=6, cost_per_second=1),
        )

        self.assertEqual(result.actions[0].cost_after, Decimal("7"))
        self.assertEqual(result.actions[1].cost_before, Decimal("7"))
        self.assertEqual(result.actions[1].cost_after, Decimal("4"))
        self.assertEqual(result.total_recovered_cost, Decimal("5"))

    def test_zero_recovery_reports_unavailable_shortest_time(self):
        result = simulate_timeline(
            parse_timeline("3:55 ヒマリ 3"),
            _config(cost_per_second=0),
        )

        action = result.actions[0]
        self.assertFalse(action.is_available)
        self.assertIsNone(action.earliest_available_time)
        self.assertIn("実効回復速度が0", action.warning or "")
        self.assertTrue(any("実効回復速度が0" in warning for warning in result.warnings))

    def test_cost_above_maximum_can_never_become_available(self):
        result = simulate_timeline(
            parse_timeline("3:55 アタッカー 11"),
            _config(max_cost=10, cost_per_second=1),
        )

        self.assertFalse(result.actions[0].is_available)
        self.assertIsNone(result.actions[0].earliest_available_time)
        self.assertIn("最大コストを超える", result.actions[0].warning or "")

    def test_shortest_time_below_zero_is_returned_with_warning(self):
        result = simulate_timeline(
            parse_timeline("0:01 アタッカー 10"),
            _config(battle_time=2, cost_per_second="0.1"),
        )

        action = result.actions[0]
        self.assertFalse(action.is_available)
        self.assertIsNotNone(action.earliest_available_time)
        self.assertLess(action.earliest_available_time or Decimal("0"), Decimal("0"))
        self.assertIn("戦闘時間内に使用できません", action.warning or "")

    def test_recovery_multiplier_is_applied(self):
        result = simulate_timeline(
            parse_timeline("3:55 ヒマリ 3"),
            _config(cost_per_second="0.3", recovery_multiplier="2"),
        )

        self.assertEqual(result.effective_recovery_rate, Decimal("0.6"))
        self.assertEqual(result.actions[0].cost_before, Decimal("3.0"))
        self.assertTrue(result.is_valid)

    def test_base_and_actual_cost_are_kept_distinct(self):
        action = parse_timeline("3:55 ヒマリ 5/3")[0]
        result = simulate_timeline((action,), _config(cost_per_second=1))

        self.assertEqual(action.base_cost, Decimal("5"))
        self.assertEqual(action.actual_cost, Decimal("3"))
        self.assertEqual(result.actions[0].cost_spent, Decimal("3"))


class BlueArchiveTimelineValidationTestCase(unittest.TestCase):
    def test_timestamp_order_start_bound_and_negative_time_are_rejected(self):
        with self.assertRaisesRegex(TimelineValidationError, "大きい順"):
            simulate_timeline(
                parse_timeline("3:50 ヒマリ 3\n3:55 アコ 3"),
                _config(),
            )
        with self.assertRaisesRegex(TimelineValidationError, "戦闘開始時刻以下"):
            simulate_timeline(parse_timeline("4:01 ヒマリ 3"), _config())

        action = TimelineAction(
            remaining_time=Decimal("-1"),
            display_name="ヒマリ",
            base_cost=Decimal("3"),
            actual_cost=Decimal("3"),
            line_number=1,
        )
        with self.assertRaisesRegex(TimelineValidationError, "0:00以上"):
            simulate_timeline((action,), _config())

    def test_invalid_config_values_are_rejected(self):
        invalid_configs = (
            ({"battle_time": 0}, "戦闘時間"),
            ({"start_cost": -1}, "開始コスト"),
            ({"max_cost": 0}, "最大コスト"),
            ({"start_cost": 11, "max_cost": 10}, "開始コストは最大コスト以下"),
            ({"cost_per_second": -1}, "1秒当たり"),
            ({"recovery_multiplier": -1}, "回復速度倍率"),
            ({"cost_per_second": math.nan}, "NaNや無限大"),
            ({"cost_per_second": math.inf}, "NaNや無限大"),
        )
        for overrides, message in invalid_configs:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(TimelineValidationError, message):
                    simulate_timeline(parse_timeline("3:55 ヒマリ 3"), _config(**overrides))

    def test_float_conversion_does_not_use_binary_float_decimal(self):
        config = _config(cost_per_second=0.1, recovery_multiplier=1.2)

        self.assertEqual(config.cost_per_second, Decimal("0.1"))
        self.assertEqual(config.recovery_multiplier, Decimal("1.2"))
        self.assertEqual(config.effective_recovery_rate, Decimal("0.12"))

    def test_delay_mode_is_explicitly_rejected_as_not_implemented(self):
        with self.assertRaisesRegex(TimelineValidationError, "delayモードは現在未対応"):
            simulate_timeline(
                parse_timeline("3:55 ヒマリ 3"),
                _config(mode="delay"),
            )

    def test_extreme_allowed_values_finish_without_large_output(self):
        actions = parse_timeline("\n".join(f"59:59 生徒{i} 1000" for i in range(MAX_TIMELINE_ACTIONS)))
        result = simulate_timeline(
            actions,
            _config(
                battle_time=3600,
                max_cost=1000,
                cost_per_second=1000,
                recovery_multiplier=1000,
            ),
        )

        self.assertEqual(len(result.actions), MAX_TIMELINE_ACTIONS)
        self.assertLess(len(format_cost(result.total_recovered_cost)), 30)


if __name__ == "__main__":
    unittest.main()
