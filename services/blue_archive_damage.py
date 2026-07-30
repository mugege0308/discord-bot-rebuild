"""ブルーアーカイブ向けの手動入力ダメージ計算エンジン。

このモジュールはDiscordや外部通信に依存しない。割合はすべて内部では
``0.5 == 50%``、スキル倍率は ``8.0 == 800%`` のような倍率で扱う。

防御補正 ``1666.6666666667 / (1666.6666666667 + 実効防御力)`` は、
コミュニティで広く使われている検証式であり、公式公開式ではない。
攻撃力、スキル倍率、各種乗算補正を掛ける順序も一般的な計算モデルに
基づく近似である。個別スキル固有処理、レベル差、敵固有補正、ゲーム内の
段階的な端数処理は未対応。検証後に差し替えやすいよう、基礎・防御・会心の
計算はそれぞれ独立したpublic関数に分離している。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


NumericInput = int | float | Decimal | str

DEFENSE_SCALING_CONSTANT = Decimal("1666.6666666667")
MAX_STAT = Decimal("1000000000")
MAX_RATE = Decimal("10000")
MAX_MULTIPLIER = Decimal("1000")
MAX_RANDOM_MULTIPLIER = Decimal("10000")
MAX_HIT_COUNT = 100


class DamageValidationError(ValueError):
    """ダメージ計算の入力値が不正な場合に送出する例外。"""


@dataclass(frozen=True)
class DamageCalculationInput:
    """内部倍率へ正規化済みのダメージ計算入力。"""

    base_attack: Decimal
    skill_multiplier: Decimal
    attack_buff_rate: Decimal = Decimal("0")
    damage_buff_rate: Decimal = Decimal("0")
    enemy_defense: Decimal = Decimal("0")
    defense_debuff_rate: Decimal = Decimal("0")
    attribute_multiplier: Decimal = Decimal("1")
    terrain_multiplier: Decimal = Decimal("1")
    critical_damage_multiplier: Decimal = Decimal("2")
    random_min_multiplier: Decimal = Decimal("1")
    random_max_multiplier: Decimal = Decimal("1")
    hit_count: int = 1
    is_critical: bool = False

    @classmethod
    def from_percentages(
        cls,
        *,
        attack: NumericInput,
        skill_percent: NumericInput,
        enemy_defense: NumericInput,
        attack_buff_percent: NumericInput = 0,
        damage_buff_percent: NumericInput = 0,
        defense_debuff_percent: NumericInput = 0,
        attribute_multiplier: NumericInput = 1,
        terrain_multiplier: NumericInput = 1,
        critical_damage_percent: NumericInput = 200,
        random_min_percent: NumericInput = 100,
        random_max_percent: NumericInput = 100,
        hit_count: int = 1,
        is_critical: bool = False,
    ) -> "DamageCalculationInput":
        """ユーザー向けの百分率入力を内部倍率へ変換する。

        ``50`` は50%のバフ率（内部では0.5）、``800`` は800%の
        スキル倍率（内部では8.0）として変換する。
        """

        one_hundred = Decimal("100")
        return cls(
            base_attack=_to_decimal(attack, "基礎攻撃力"),
            skill_multiplier=_to_decimal(skill_percent, "スキル倍率") / one_hundred,
            attack_buff_rate=_to_decimal(attack_buff_percent, "攻撃力バフ率") / one_hundred,
            damage_buff_rate=_to_decimal(damage_buff_percent, "与ダメージ増加率") / one_hundred,
            enemy_defense=_to_decimal(enemy_defense, "敵の防御力"),
            defense_debuff_rate=_to_decimal(defense_debuff_percent, "防御力デバフ率") / one_hundred,
            attribute_multiplier=_to_decimal(attribute_multiplier, "属性相性倍率"),
            terrain_multiplier=_to_decimal(terrain_multiplier, "地形適性倍率"),
            critical_damage_multiplier=_to_decimal(critical_damage_percent, "会心ダメージ倍率") / one_hundred,
            random_min_multiplier=_to_decimal(random_min_percent, "ダメージ乱数の最小値") / one_hundred,
            random_max_multiplier=_to_decimal(random_max_percent, "ダメージ乱数の最大値") / one_hundred,
            hit_count=hit_count,
            is_critical=is_critical,
        )


@dataclass(frozen=True)
class DamageRange:
    """整数へ丸めたダメージ範囲。"""

    minimum: int
    maximum: int
    average: int


@dataclass(frozen=True)
class DamageIntermediateValues:
    """表示や検証に利用できる計算途中の補正値。"""

    attack_buff_multiplier: Decimal
    effective_attack: Decimal
    effective_defense: Decimal
    defense_multiplier: Decimal
    base_damage: Decimal
    damage_buff_multiplier: Decimal
    attribute_multiplier: Decimal
    terrain_multiplier: Decimal
    non_critical_before_random: Decimal
    critical_damage_multiplier: Decimal


@dataclass(frozen=True)
class DamageCalculationResult:
    """非会心・会心の1ヒット値とヒット合計をまとめた計算結果。"""

    non_critical_per_hit: DamageRange
    critical_per_hit: DamageRange
    non_critical_total: DamageRange
    critical_total: DamageRange
    hit_count: int
    is_critical: bool
    intermediate: DamageIntermediateValues

    @property
    def selected_per_hit(self) -> DamageRange:
        """入力された会心有無に対応する1ヒットの範囲を返す。"""

        return self.critical_per_hit if self.is_critical else self.non_critical_per_hit

    @property
    def selected_total(self) -> DamageRange:
        """入力された会心有無に対応するヒット合計の範囲を返す。"""

        return self.critical_total if self.is_critical else self.non_critical_total


def calculate_effective_defense(enemy_defense: Decimal, defense_debuff_rate: Decimal) -> Decimal:
    """防御デバフ適用後の実効防御力を0以上で返す。

    100%を超える防御デバフは許容するが、実効防御力は負数にせず0へ
    クランプする。
    """

    return max(enemy_defense * (Decimal("1") - defense_debuff_rate), Decimal("0"))


def calculate_defense_multiplier(effective_defense: Decimal) -> Decimal:
    """実効防御力から与ダメージ側の防御倍率を計算する。

    現在の式はコミュニティ検証で広く使われる近似式
    ``1666.6666666667 / (1666.6666666667 + effective_defense)``。
    公式公開式ではないため、ゲーム内検証で差異が見つかった場合はこの関数を
    差し替える。
    """

    return DEFENSE_SCALING_CONSTANT / (DEFENSE_SCALING_CONSTANT + effective_defense)


def calculate_base_damage(effective_attack: Decimal, skill_multiplier: Decimal) -> Decimal:
    """実効攻撃力と1ヒット当たりのスキル倍率から基礎ダメージを返す。

    固定加算ダメージや個別スキル固有の段階的丸めは未対応。
    """

    return effective_attack * skill_multiplier


def calculate_critical_damage(non_critical_damage: Decimal, critical_damage_multiplier: Decimal) -> Decimal:
    """非会心ダメージへ会心ダメージ倍率を乗算する。

    会心抵抗やスキル固有の会心制限は未対応。
    """

    return non_critical_damage * critical_damage_multiplier


def round_damage(value: Decimal) -> int:
    """表示用ダメージを小数第1位で四捨五入して整数にする。

    ゲーム内の各計算段階での切り捨て順序は未確定のため、エンジンでは最後に
    ``ROUND_HALF_UP`` を1回だけ適用する。
    """

    return int(value.to_integral_value(rounding=ROUND_HALF_UP))


def calculate_damage(values: DamageCalculationInput) -> DamageCalculationResult:
    """検証済み入力から非会心・会心のダメージ範囲とヒット合計を計算する。"""

    _validate_input(values)

    attack_buff_multiplier = Decimal("1") + values.attack_buff_rate
    effective_attack = values.base_attack * attack_buff_multiplier
    effective_defense = calculate_effective_defense(values.enemy_defense, values.defense_debuff_rate)
    defense_multiplier = calculate_defense_multiplier(effective_defense)
    base_damage = calculate_base_damage(effective_attack, values.skill_multiplier)
    damage_buff_multiplier = Decimal("1") + values.damage_buff_rate
    non_critical_before_random = (
        base_damage
        * damage_buff_multiplier
        * defense_multiplier
        * values.attribute_multiplier
        * values.terrain_multiplier
    )
    critical_before_random = calculate_critical_damage(
        non_critical_before_random,
        values.critical_damage_multiplier,
    )

    non_critical_per_hit = _build_range(
        non_critical_before_random,
        values.random_min_multiplier,
        values.random_max_multiplier,
    )
    critical_per_hit = _build_range(
        critical_before_random,
        values.random_min_multiplier,
        values.random_max_multiplier,
    )
    hit_multiplier = Decimal(values.hit_count)
    non_critical_total = _build_range(
        non_critical_before_random * hit_multiplier,
        values.random_min_multiplier,
        values.random_max_multiplier,
    )
    critical_total = _build_range(
        critical_before_random * hit_multiplier,
        values.random_min_multiplier,
        values.random_max_multiplier,
    )

    return DamageCalculationResult(
        non_critical_per_hit=non_critical_per_hit,
        critical_per_hit=critical_per_hit,
        non_critical_total=non_critical_total,
        critical_total=critical_total,
        hit_count=values.hit_count,
        is_critical=values.is_critical,
        intermediate=DamageIntermediateValues(
            attack_buff_multiplier=attack_buff_multiplier,
            effective_attack=effective_attack,
            effective_defense=effective_defense,
            defense_multiplier=defense_multiplier,
            base_damage=base_damage,
            damage_buff_multiplier=damage_buff_multiplier,
            attribute_multiplier=values.attribute_multiplier,
            terrain_multiplier=values.terrain_multiplier,
            non_critical_before_random=non_critical_before_random,
            critical_damage_multiplier=values.critical_damage_multiplier,
        ),
    )


def _to_decimal(value: NumericInput, field_name: str) -> Decimal:
    if isinstance(value, bool):
        raise DamageValidationError(f"{field_name}には数値を入力してください。")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise DamageValidationError(f"{field_name}には有効な数値を入力してください。") from exc
    if not result.is_finite():
        raise DamageValidationError(f"{field_name}にはNaNや無限大を入力できません。")
    return result


def _validate_input(values: DamageCalculationInput) -> None:
    decimal_fields = (
        ("基礎攻撃力", values.base_attack),
        ("スキル倍率", values.skill_multiplier),
        ("攻撃力バフ率", values.attack_buff_rate),
        ("与ダメージ増加率", values.damage_buff_rate),
        ("敵の防御力", values.enemy_defense),
        ("防御力デバフ率", values.defense_debuff_rate),
        ("属性相性倍率", values.attribute_multiplier),
        ("地形適性倍率", values.terrain_multiplier),
        ("会心ダメージ倍率", values.critical_damage_multiplier),
        ("ダメージ乱数の最小値", values.random_min_multiplier),
        ("ダメージ乱数の最大値", values.random_max_multiplier),
    )
    for field_name, value in decimal_fields:
        if not isinstance(value, Decimal) or not value.is_finite():
            raise DamageValidationError(f"{field_name}にはNaNや無限大を入力できません。")

    if values.base_attack < 0:
        raise DamageValidationError("基礎攻撃力は0以上で入力してください。")
    if values.base_attack > MAX_STAT:
        raise DamageValidationError(f"基礎攻撃力は{int(MAX_STAT):,}以下で入力してください。")
    if values.enemy_defense < 0:
        raise DamageValidationError("敵の防御力は0以上で入力してください。")
    if values.enemy_defense > MAX_STAT:
        raise DamageValidationError(f"敵の防御力は{int(MAX_STAT):,}以下で入力してください。")
    if values.skill_multiplier <= 0:
        raise DamageValidationError("スキル倍率は0より大きい値を入力してください。")
    if values.skill_multiplier > MAX_RATE:
        raise DamageValidationError("スキル倍率が入力上限を超えています。")

    for field_name, rate in (
        ("攻撃力バフ率", values.attack_buff_rate),
        ("与ダメージ増加率", values.damage_buff_rate),
        ("防御力デバフ率", values.defense_debuff_rate),
    ):
        if rate < 0:
            raise DamageValidationError(f"{field_name}は0以上で入力してください。")
        if rate > MAX_RATE:
            raise DamageValidationError(f"{field_name}が入力上限を超えています。")

    for field_name, multiplier in (
        ("属性相性倍率", values.attribute_multiplier),
        ("地形適性倍率", values.terrain_multiplier),
        ("会心ダメージ倍率", values.critical_damage_multiplier),
    ):
        if multiplier <= 0:
            raise DamageValidationError(f"{field_name}は0より大きい値を入力してください。")
        if multiplier > MAX_MULTIPLIER and field_name != "会心ダメージ倍率":
            raise DamageValidationError(f"{field_name}が入力上限を超えています。")
    if values.critical_damage_multiplier > MAX_RATE:
        raise DamageValidationError("会心ダメージ倍率が入力上限を超えています。")

    if values.random_min_multiplier < 0:
        raise DamageValidationError("ダメージ乱数の最小値は0以上で入力してください。")
    if values.random_max_multiplier <= 0:
        raise DamageValidationError("ダメージ乱数の最大値は0より大きい値を入力してください。")
    if (
        values.random_min_multiplier > MAX_RANDOM_MULTIPLIER
        or values.random_max_multiplier > MAX_RANDOM_MULTIPLIER
    ):
        raise DamageValidationError("ダメージ乱数が入力上限を超えています。")
    if values.random_min_multiplier > values.random_max_multiplier:
        raise DamageValidationError("ダメージ乱数の最小値は最大値以下にしてください。")

    if isinstance(values.hit_count, bool) or not isinstance(values.hit_count, int):
        raise DamageValidationError("ヒット数には整数を入力してください。")
    if values.hit_count <= 0:
        raise DamageValidationError("ヒット数は1以上で入力してください。")
    if values.hit_count > MAX_HIT_COUNT:
        raise DamageValidationError(f"ヒット数は{MAX_HIT_COUNT}以下で入力してください。")
    if not isinstance(values.is_critical, bool):
        raise DamageValidationError("会心の有無にはtrueまたはfalseを指定してください。")


def _build_range(base_damage: Decimal, random_min: Decimal, random_max: Decimal) -> DamageRange:
    average_random = (random_min + random_max) / Decimal("2")
    return DamageRange(
        minimum=round_damage(base_damage * random_min),
        maximum=round_damage(base_damage * random_max),
        average=round_damage(base_damage * average_random),
    )
