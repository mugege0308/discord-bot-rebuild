import math
import unittest
from decimal import Decimal

from services.blue_archive_damage import (
    DamageCalculationInput,
    DamageValidationError,
    calculate_damage,
    round_damage,
)


def _calculate(**overrides):
    values = {
        "attack": 1000,
        "skill_percent": 200,
        "enemy_defense": 0,
    }
    values.update(overrides)
    return calculate_damage(DamageCalculationInput.from_percentages(**values))


class BlueArchiveDamageCalculationTestCase(unittest.TestCase):
    def test_basic_damage_without_buffs_or_defense(self):
        result = _calculate()

        self.assertEqual((result.non_critical_per_hit.minimum, result.non_critical_per_hit.maximum), (2000, 2000))
        self.assertEqual(result.non_critical_per_hit.average, 2000)
        self.assertEqual((result.critical_per_hit.minimum, result.critical_per_hit.maximum), (4000, 4000))
        self.assertEqual(result.critical_per_hit.average, 4000)

    def test_attack_buff_is_applied_to_effective_attack(self):
        result = _calculate(attack_buff_percent=50)

        self.assertEqual(result.intermediate.effective_attack, Decimal("1500.0"))
        self.assertEqual(result.non_critical_per_hit.average, 3000)

    def test_damage_buff_is_applied_multiplicatively(self):
        result = _calculate(damage_buff_percent=25)

        self.assertEqual(result.intermediate.damage_buff_multiplier, Decimal("1.25"))
        self.assertEqual(result.non_critical_per_hit.average, 2500)

    def test_defense_debuff_reduces_effective_defense(self):
        result = _calculate(enemy_defense=1000, defense_debuff_percent=50)

        self.assertEqual(result.intermediate.effective_defense, Decimal("500.0"))
        self.assertEqual(result.non_critical_per_hit.average, 1538)

    def test_attribute_multiplier_is_applied(self):
        result = _calculate(attribute_multiplier=1.5)

        self.assertEqual(result.non_critical_per_hit.average, 3000)

    def test_terrain_multiplier_is_applied(self):
        result = _calculate(terrain_multiplier=1.2)

        self.assertEqual(result.non_critical_per_hit.average, 2400)

    def test_critical_damage_multiplier_is_applied(self):
        result = _calculate(critical_damage_percent=250, is_critical=True)

        self.assertEqual(result.critical_per_hit.average, 5000)
        self.assertEqual(result.selected_per_hit.average, 5000)

    def test_multiple_hits_return_totals(self):
        result = _calculate(hit_count=3)

        self.assertEqual(result.non_critical_total.average, 6000)
        self.assertEqual(result.critical_total.average, 12000)
        self.assertEqual(result.hit_count, 3)

    def test_random_range_returns_minimum_maximum_and_average(self):
        result = _calculate(random_min_percent=90, random_max_percent=110)

        self.assertEqual(
            (
                result.non_critical_per_hit.minimum,
                result.non_critical_per_hit.maximum,
                result.non_critical_per_hit.average,
            ),
            (1800, 2200, 2000),
        )
        self.assertEqual(
            (
                result.critical_per_hit.minimum,
                result.critical_per_hit.maximum,
                result.critical_per_hit.average,
            ),
            (3600, 4400, 4000),
        )

    def test_one_hundred_percent_defense_debuff_makes_defense_zero(self):
        result = _calculate(enemy_defense=1000, defense_debuff_percent=100)

        self.assertEqual(result.intermediate.effective_defense, Decimal("0"))
        self.assertEqual(result.intermediate.defense_multiplier, Decimal("1"))
        self.assertEqual(result.non_critical_per_hit.average, 2000)

    def test_defense_debuff_over_one_hundred_percent_is_clamped_at_zero_defense(self):
        result = _calculate(enemy_defense=1000, defense_debuff_percent=150)

        self.assertEqual(result.intermediate.effective_defense, Decimal("0"))
        self.assertEqual(result.non_critical_per_hit.average, 2000)

    def test_rounding_uses_half_up_instead_of_bankers_rounding(self):
        result = _calculate(attack=1, skill_percent=150)

        self.assertEqual(result.non_critical_per_hit.average, 2)
        self.assertEqual(round_damage(Decimal("2.5")), 3)

    def test_very_large_allowed_values_do_not_crash(self):
        result = _calculate(
            attack=1_000_000_000,
            skill_percent=1_000_000,
            attack_buff_percent=1_000_000,
            damage_buff_percent=1_000_000,
            attribute_multiplier=1000,
            terrain_multiplier=1000,
            critical_damage_percent=1_000_000,
            hit_count=100,
            random_max_percent=1_000_000,
        )

        self.assertGreater(result.critical_total.maximum, 10**30)


class BlueArchiveDamageValidationTestCase(unittest.TestCase):
    def assert_invalid(self, expected_message_part, **overrides):
        with self.assertRaisesRegex(DamageValidationError, expected_message_part):
            _calculate(**overrides)

    def test_negative_attack_is_rejected(self):
        self.assert_invalid("基礎攻撃力は0以上", attack=-1)

    def test_negative_enemy_defense_is_rejected(self):
        self.assert_invalid("敵の防御力は0以上", enemy_defense=-1)

    def test_zero_skill_multiplier_is_rejected(self):
        self.assert_invalid("スキル倍率は0より大きい", skill_percent=0)

    def test_negative_skill_multiplier_is_rejected(self):
        self.assert_invalid("スキル倍率は0より大きい", skill_percent=-1)

    def test_zero_hit_count_is_rejected(self):
        self.assert_invalid("ヒット数は1以上", hit_count=0)

    def test_negative_hit_count_is_rejected(self):
        self.assert_invalid("ヒット数は1以上", hit_count=-1)

    def test_nan_is_rejected(self):
        for field_name, overrides in (
            ("基礎攻撃力", {"attack": math.nan}),
            ("スキル倍率", {"skill_percent": math.nan}),
            ("敵の防御力", {"enemy_defense": math.nan}),
        ):
            with self.subTest(field_name=field_name):
                self.assert_invalid("NaNや無限大", **overrides)

    def test_infinity_is_rejected(self):
        for value in (math.inf, -math.inf):
            with self.subTest(value=value):
                self.assert_invalid("NaNや無限大", attack=value)

    def test_reversed_random_range_is_rejected(self):
        self.assert_invalid("最小値は最大値以下", random_min_percent=110, random_max_percent=90)

    def test_values_above_command_limits_are_rejected_by_engine(self):
        self.assert_invalid("基礎攻撃力は1,000,000,000以下", attack=1_000_000_001)
        self.assert_invalid("ヒット数は100以下", hit_count=101)
        self.assert_invalid("属性相性倍率が入力上限", attribute_multiplier=1001)

    def test_excessive_defense_debuff_is_rejected_at_safety_limit(self):
        self.assert_invalid("防御力デバフ率が入力上限", defense_debuff_percent=1_000_001)

    def test_negative_buff_rate_is_rejected(self):
        self.assert_invalid("攻撃力バフ率は0以上", attack_buff_percent=-1)

    def test_non_finite_internal_decimal_is_rejected(self):
        values = DamageCalculationInput(
            base_attack=Decimal("NaN"),
            skill_multiplier=Decimal("1"),
        )

        with self.assertRaisesRegex(DamageValidationError, "NaNや無限大"):
            calculate_damage(values)


if __name__ == "__main__":
    unittest.main()
