"""ブルーアーカイブ向けの手動入力TLコスト検査エンジン。

このモジュールはDiscordや外部通信に依存しない。ゲーム固有の回復速度を
正式値として仮定せず、1秒当たりの基礎回復量と回復速度倍率を利用者から
受け取る汎用シミュレーターである。

時刻は戦闘終了までの残り秒数を :class:`~decimal.Decimal` で保持する。
現在実装しているのはstrictモードのみで、指定時刻にコストが不足していた
場合は消費せず、その時刻の残高から後続行の検査を続ける。delayモードは
型として予約しているが、後続行を自動的にずらす処理は今後の実装対象。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Literal


NumericInput = int | float | Decimal | str
TimelineMode = Literal["strict", "delay"]

MAX_TIMELINE_TEXT_LENGTH = 2_000
MAX_TIMELINE_ACTIONS = 15
MAX_BATTLE_TIME = Decimal("3600")
MAX_COST_VALUE = Decimal("1000")
MAX_RECOVERY_VALUE = Decimal("1000")
MAX_NAME_LENGTH = 80
MAX_MEMO_LENGTH = 200

_TIME_PATTERN = re.compile(r"^(?P<minutes>\d+):(?P<seconds>\d{2}(?:\.\d{1,3})?)$")


class TimelineValidationError(ValueError):
    """TL解析またはコスト計算の入力値が不正な場合に送出する例外。"""


@dataclass(frozen=True)
class TimelineAction:
    """TLの1行を表す、時刻とEXコスト使用情報。"""

    remaining_time: Decimal
    display_name: str
    base_cost: Decimal
    actual_cost: Decimal
    memo: str = ""
    line_number: int = 0


@dataclass(frozen=True)
class TimelineConfig:
    """TLシミュレーション全体の設定。"""

    battle_time: Decimal
    start_cost: Decimal
    max_cost: Decimal
    cost_per_second: Decimal
    recovery_multiplier: Decimal = Decimal("1")
    mode: TimelineMode = "strict"

    @classmethod
    def from_values(
        cls,
        *,
        battle_time: NumericInput = 240,
        start_cost: NumericInput = 0,
        max_cost: NumericInput = 10,
        cost_per_second: NumericInput,
        recovery_multiplier: NumericInput = 1,
        mode: TimelineMode = "strict",
    ) -> "TimelineConfig":
        """一般的な数値入力を精度を保ったDecimal設定へ変換する。"""

        return cls(
            battle_time=_to_decimal(battle_time, "戦闘時間"),
            start_cost=_to_decimal(start_cost, "開始コスト"),
            max_cost=_to_decimal(max_cost, "最大コスト"),
            cost_per_second=_to_decimal(cost_per_second, "1秒当たりのコスト回復量"),
            recovery_multiplier=_to_decimal(recovery_multiplier, "回復速度倍率"),
            mode=mode,
        )

    @property
    def effective_recovery_rate(self) -> Decimal:
        """倍率適用後の1秒当たりの実効回復量を返す。"""

        return self.cost_per_second * self.recovery_multiplier


@dataclass(frozen=True)
class TimelineActionResult:
    """TLの1行について、使用可否とその時点のコストを表す結果。"""

    line_number: int
    remaining_time: Decimal
    display_name: str
    base_cost: Decimal
    required_cost: Decimal
    cost_before: Decimal
    cost_spent: Decimal
    cost_after: Decimal
    is_available: bool
    shortage: Decimal
    earliest_available_time: Decimal | None
    overflow_cost: Decimal
    error_message: str | None = None
    warning: str | None = None
    memo: str = ""


@dataclass(frozen=True)
class TimelineSimulationResult:
    """各行の検査結果とTL全体のコスト集計。"""

    actions: tuple[TimelineActionResult, ...]
    is_valid: bool
    total_recovered_cost: Decimal
    total_spent_cost: Decimal
    total_overflow_cost: Decimal
    final_cost: Decimal
    effective_recovery_rate: Decimal
    warnings: tuple[str, ...]


def parse_remaining_time(value: str) -> Decimal:
    """``M:SS`` または ``M:SS.s`` を残り秒数へ変換する。

    秒部分は00以上60未満、小数は最大3桁まで受け付ける。
    """

    if not isinstance(value, str):
        raise TimelineValidationError("時刻は「3:55」の形式で入力してください。")
    match = _TIME_PATTERN.fullmatch(value.strip())
    if match is None:
        raise TimelineValidationError(f"時刻「{value}」は「3:55」または「1:02.5」の形式で入力してください。")

    minutes = Decimal(match.group("minutes"))
    seconds = Decimal(match.group("seconds"))
    if seconds >= Decimal("60"):
        raise TimelineValidationError(f"時刻「{value}」の秒部分は60未満にしてください。")
    return minutes * Decimal("60") + seconds


def format_remaining_time(value: Decimal) -> str:
    """残り秒数を表示用の ``M:SS`` または ``M:SS.s`` に整形する。"""

    if not isinstance(value, Decimal) or not value.is_finite():
        raise TimelineValidationError("表示する時刻には有限のDecimalを指定してください。")

    rounded = value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    sign = "-" if rounded < 0 else ""
    absolute = abs(rounded)
    minutes = int(absolute // Decimal("60"))
    seconds = absolute - Decimal(minutes * 60)
    if seconds == seconds.to_integral_value():
        seconds_text = f"{int(seconds):02d}"
    else:
        seconds_text = f"{seconds:04.1f}"
    return f"{sign}{minutes}:{seconds_text}"


def format_cost(value: Decimal) -> str:
    """コストを小数第2位まで、不要な末尾0を除いて表示する。"""

    if not isinstance(value, Decimal) or not value.is_finite():
        raise TimelineValidationError("表示するコストには有限のDecimalを指定してください。")
    rounded = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    text = f"{rounded:.2f}".rstrip("0").rstrip(".")
    return text or "0"


def parse_timeline(timeline: str) -> tuple[TimelineAction, ...]:
    """複数行またはセミコロン区切りのTL文字列を解析する。

    ``3:55 ヒマリ 3`` と ``3:55 | ヒマリ | 3`` の両形式を
    受け付ける。コストを ``5/3`` と書くと基本コスト5、実消費3として
    扱う。空白形式では末尾の ``| メモ``、列形式では4列目が任意メモ。
    生徒名・表示名には空白を含められる。
    """

    if not isinstance(timeline, str):
        raise TimelineValidationError("timelineにはTL文字列を入力してください。")
    if not timeline.strip():
        raise TimelineValidationError("timelineを1行以上入力してください。")
    if len(timeline) > MAX_TIMELINE_TEXT_LENGTH:
        raise TimelineValidationError(f"timelineは{MAX_TIMELINE_TEXT_LENGTH:,}文字以内で入力してください。")

    normalized = timeline.replace("\r\n", "\n").replace("\r", "\n").replace(";", "\n")
    parsed: list[TimelineAction] = []
    for source_line_number, raw_line in enumerate(normalized.split("\n"), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if len(parsed) >= MAX_TIMELINE_ACTIONS:
            raise TimelineValidationError(f"TLは{MAX_TIMELINE_ACTIONS}行以内で入力してください。")

        pipe_count = stripped.count("|")
        if pipe_count >= 2:
            columns = [column.strip() for column in stripped.split("|")]
            if len(columns) not in (3, 4) or not all(columns[:3]):
                raise TimelineValidationError(
                    f"{source_line_number}行目: 「3:55 | ヒマリ | 3」の3列形式で入力してください。"
                )
            time_text, display_name, cost_text = columns[:3]
            memo = columns[3] if len(columns) == 4 else ""
        else:
            action_text, separator, memo_text = stripped.partition("|")
            memo = memo_text.strip() if separator else ""
            parts = action_text.split()
            if len(parts) < 3:
                raise TimelineValidationError(
                    f"{source_line_number}行目: 「3:55 ヒマリ 3」の形式で入力してください。"
                )
            time_text = parts[0]
            display_name = " ".join(parts[1:-1]).strip()
            cost_text = parts[-1]

        if len(memo) > MAX_MEMO_LENGTH:
            raise TimelineValidationError(
                f"{source_line_number}行目: メモは{MAX_MEMO_LENGTH}文字以内で入力してください。"
            )

        try:
            remaining_time = parse_remaining_time(time_text)
        except TimelineValidationError as exc:
            raise TimelineValidationError(f"{source_line_number}行目: {exc}") from exc

        if not display_name:
            raise TimelineValidationError(f"{source_line_number}行目: 表示名を入力してください。")
        if len(display_name) > MAX_NAME_LENGTH:
            raise TimelineValidationError(
                f"{source_line_number}行目: 表示名は{MAX_NAME_LENGTH}文字以内で入力してください。"
            )

        base_cost, actual_cost = _parse_cost_token(cost_text, source_line_number)
        parsed.append(
            TimelineAction(
                remaining_time=remaining_time,
                display_name=display_name,
                base_cost=base_cost,
                actual_cost=actual_cost,
                memo=memo,
                line_number=source_line_number,
            )
        )

    if not parsed:
        raise TimelineValidationError("timelineを1行以上入力してください。")
    return tuple(parsed)


def simulate_timeline(
    actions: tuple[TimelineAction, ...] | list[TimelineAction],
    config: TimelineConfig,
) -> TimelineSimulationResult:
    """strictモードで時間回復・上限溢れ・EX消費を順番に検査する。

    コスト不足の行ではコストを消費しない。後続行は指定時刻のまま検査し、
    不足分と実効回復速度から最短使用可能時刻を計算する。
    """

    _validate_config(config)
    normalized_actions = tuple(actions)
    _validate_actions(normalized_actions, config)

    if config.mode == "delay":
        raise TimelineValidationError(
            "delayモードは現在未対応です。strictモードで最短使用可能時刻を確認してください。"
        )

    effective_rate = config.effective_recovery_rate
    previous_time = config.battle_time
    current_cost = config.start_cost
    total_recovered = Decimal("0")
    total_spent = Decimal("0")
    total_overflow = Decimal("0")
    results: list[TimelineActionResult] = []
    warnings: list[str] = []

    for index, action in enumerate(normalized_actions, start=1):
        elapsed_time = previous_time - action.remaining_time
        recovered_cost = elapsed_time * effective_rate
        total_recovered += recovered_cost

        uncapped_cost = current_cost + recovered_cost
        cost_before = min(config.max_cost, uncapped_cost)
        overflow_cost = max(uncapped_cost - config.max_cost, Decimal("0"))
        total_overflow += overflow_cost

        is_available = cost_before >= action.actual_cost
        shortage = max(action.actual_cost - cost_before, Decimal("0"))
        warning: str | None = None
        error_message: str | None = None

        if is_available:
            cost_spent = action.actual_cost
            cost_after = cost_before - cost_spent
            earliest_available_time: Decimal | None = action.remaining_time
            total_spent += cost_spent
        else:
            cost_spent = Decimal("0")
            cost_after = cost_before
            error_message = f"コストが{format_cost(shortage)}不足しています。"
            earliest_available_time, warning = _calculate_earliest_available_time(
                action=action,
                shortage=shortage,
                effective_rate=effective_rate,
                max_cost=config.max_cost,
            )
            warning_prefix = f"{action.line_number or index}行目: "
            warnings.append(warning_prefix + error_message)
            if warning:
                warnings.append(warning_prefix + warning)

        results.append(
            TimelineActionResult(
                line_number=action.line_number or index,
                remaining_time=action.remaining_time,
                display_name=action.display_name,
                base_cost=action.base_cost,
                required_cost=action.actual_cost,
                cost_before=cost_before,
                cost_spent=cost_spent,
                cost_after=cost_after,
                is_available=is_available,
                shortage=shortage,
                earliest_available_time=earliest_available_time,
                overflow_cost=overflow_cost,
                error_message=error_message,
                warning=warning,
                memo=action.memo,
            )
        )
        current_cost = cost_after
        previous_time = action.remaining_time

    return TimelineSimulationResult(
        actions=tuple(results),
        is_valid=all(result.is_available for result in results),
        total_recovered_cost=total_recovered,
        total_spent_cost=total_spent,
        total_overflow_cost=total_overflow,
        final_cost=current_cost,
        effective_recovery_rate=effective_rate,
        warnings=tuple(warnings),
    )


def _parse_cost_token(value: str, line_number: int) -> tuple[Decimal, Decimal]:
    parts = value.split("/")
    if len(parts) == 1:
        base_cost = _to_decimal(parts[0], f"{line_number}行目の消費コスト")
        actual_cost = base_cost
    elif len(parts) == 2 and all(part.strip() for part in parts):
        base_cost = _to_decimal(parts[0], f"{line_number}行目の基本コスト")
        actual_cost = _to_decimal(parts[1], f"{line_number}行目の実消費コスト")
    else:
        raise TimelineValidationError(
            f"{line_number}行目: コストは「3」または「5/3」の形式で入力してください。"
        )

    for field_name, cost in (("基本コスト", base_cost), ("実消費コスト", actual_cost)):
        if cost < 0:
            raise TimelineValidationError(f"{line_number}行目: {field_name}は0以上で入力してください。")
        if cost > MAX_COST_VALUE:
            raise TimelineValidationError(
                f"{line_number}行目: {field_name}は{format_cost(MAX_COST_VALUE)}以下で入力してください。"
            )
    return base_cost, actual_cost


def _calculate_earliest_available_time(
    *,
    action: TimelineAction,
    shortage: Decimal,
    effective_rate: Decimal,
    max_cost: Decimal,
) -> tuple[Decimal | None, str | None]:
    if action.actual_cost > max_cost:
        return None, "実消費コストが最大コストを超えるため、この設定では使用できません。"
    if effective_rate <= 0:
        return None, "実効回復速度が0のため、最短使用可能時刻を算出できません。"

    wait_seconds = shortage / effective_rate
    earliest_time = action.remaining_time - wait_seconds
    if earliest_time < 0:
        return earliest_time, "最短使用可能時刻が0:00より後になり、戦闘時間内に使用できません。"
    return earliest_time, None


def _to_decimal(value: NumericInput, field_name: str) -> Decimal:
    if isinstance(value, bool):
        raise TimelineValidationError(f"{field_name}には数値を入力してください。")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise TimelineValidationError(f"{field_name}には有効な数値を入力してください。") from exc
    if not result.is_finite():
        raise TimelineValidationError(f"{field_name}にはNaNや無限大を入力できません。")
    return result


def _validate_config(config: TimelineConfig) -> None:
    if not isinstance(config, TimelineConfig):
        raise TimelineValidationError("TL設定が不正です。")

    fields = (
        ("戦闘時間", config.battle_time),
        ("開始コスト", config.start_cost),
        ("最大コスト", config.max_cost),
        ("1秒当たりのコスト回復量", config.cost_per_second),
        ("回復速度倍率", config.recovery_multiplier),
    )
    for field_name, value in fields:
        if not isinstance(value, Decimal) or not value.is_finite():
            raise TimelineValidationError(f"{field_name}にはNaNや無限大を入力できません。")

    if config.battle_time <= 0 or config.battle_time > MAX_BATTLE_TIME:
        raise TimelineValidationError(
            f"戦闘時間は0より大きく{format_cost(MAX_BATTLE_TIME)}秒以下で入力してください。"
        )
    if config.start_cost < 0 or config.start_cost > MAX_COST_VALUE:
        raise TimelineValidationError(
            f"開始コストは0以上{format_cost(MAX_COST_VALUE)}以下で入力してください。"
        )
    if config.max_cost <= 0 or config.max_cost > MAX_COST_VALUE:
        raise TimelineValidationError(
            f"最大コストは0より大きく{format_cost(MAX_COST_VALUE)}以下で入力してください。"
        )
    if config.start_cost > config.max_cost:
        raise TimelineValidationError("開始コストは最大コスト以下にしてください。")
    if config.cost_per_second < 0 or config.cost_per_second > MAX_RECOVERY_VALUE:
        raise TimelineValidationError(
            f"1秒当たりのコスト回復量は0以上{format_cost(MAX_RECOVERY_VALUE)}以下で入力してください。"
        )
    if config.recovery_multiplier < 0 or config.recovery_multiplier > MAX_RECOVERY_VALUE:
        raise TimelineValidationError(
            f"回復速度倍率は0以上{format_cost(MAX_RECOVERY_VALUE)}以下で入力してください。"
        )
    if config.mode not in ("strict", "delay"):
        raise TimelineValidationError("モードはstrictまたはdelayを指定してください。")


def _validate_actions(actions: tuple[TimelineAction, ...], config: TimelineConfig) -> None:
    if not actions:
        raise TimelineValidationError("TLを1行以上入力してください。")
    if len(actions) > MAX_TIMELINE_ACTIONS:
        raise TimelineValidationError(f"TLは{MAX_TIMELINE_ACTIONS}行以内で入力してください。")

    previous_time = config.battle_time
    for index, action in enumerate(actions, start=1):
        line_number = action.line_number or index
        if not isinstance(action.remaining_time, Decimal) or not action.remaining_time.is_finite():
            raise TimelineValidationError(f"{line_number}行目: 時刻には有限のDecimalを指定してください。")
        if action.remaining_time < 0:
            raise TimelineValidationError(f"{line_number}行目: 時刻は0:00以上にしてください。")
        if action.remaining_time > config.battle_time:
            raise TimelineValidationError(f"{line_number}行目: 時刻は戦闘開始時刻以下にしてください。")
        if action.remaining_time > previous_time:
            raise TimelineValidationError(
                f"{line_number}行目: TLは残り時間が大きい順に並べてください。"
            )
        if not action.display_name or len(action.display_name) > MAX_NAME_LENGTH:
            raise TimelineValidationError(
                f"{line_number}行目: 表示名は1〜{MAX_NAME_LENGTH}文字で入力してください。"
            )
        if len(action.memo) > MAX_MEMO_LENGTH:
            raise TimelineValidationError(
                f"{line_number}行目: メモは{MAX_MEMO_LENGTH}文字以内で入力してください。"
            )
        for field_name, value in (("基本コスト", action.base_cost), ("実消費コスト", action.actual_cost)):
            if not isinstance(value, Decimal) or not value.is_finite():
                raise TimelineValidationError(f"{line_number}行目: {field_name}には有限のDecimalを指定してください。")
            if value < 0 or value > MAX_COST_VALUE:
                raise TimelineValidationError(
                    f"{line_number}行目: {field_name}は0以上{format_cost(MAX_COST_VALUE)}以下にしてください。"
                )
        previous_time = action.remaining_time
