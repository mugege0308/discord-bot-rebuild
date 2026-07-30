"""ブルーアーカイブの手動入力ダメージ計算コマンド。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

import discord
from discord import app_commands

from services.blue_archive_damage import (
    DamageCalculationInput,
    DamageCalculationResult,
    DamageRange,
    DamageValidationError,
    calculate_damage,
    round_damage,
)


logger = logging.getLogger("secretary-bot.blue_archive")

CALCULATION_FAILED_MESSAGE = "ダメージ計算に失敗しました。入力値を確認して、もう一度試してください。"
CAUTION_MESSAGE = "実際のゲーム内表示とは、端数処理、個別スキル仕様、敵固有補正などにより差が出る可能性があります。"


@dataclass(frozen=True)
class DamageEmbedField:
    """Discord Embedへ変換する前の表示フィールド。"""

    name: str
    value: str
    inline: bool = False


@dataclass(frozen=True)
class DamageEmbedData:
    """Discord APIを呼ばずにテストできるEmbed表示データ。"""

    title: str
    description: str
    fields: tuple[DamageEmbedField, ...]
    footer: str


def format_damage(value: int) -> str:
    """整数ダメージを3桁区切りで表示する。"""

    return f"{value:,}"


def build_damage_embed_data(
    values: DamageCalculationInput,
    result: DamageCalculationResult,
) -> DamageEmbedData:
    """計算入力と結果からDiscord非依存の表示データを作る。"""

    selection_label = "会心" if result.is_critical else "非会心"
    fields = (
        DamageEmbedField(
            name="入力条件",
            value="\n".join(
                (
                    f"基礎攻撃力: {format_damage(round_damage(values.base_attack))}",
                    f"スキル倍率（1ヒット）: {_format_percent(values.skill_multiplier)}",
                    f"攻撃力バフ: {_format_percent(values.attack_buff_rate)}",
                    f"与ダメージ増加: {_format_percent(values.damage_buff_rate)}",
                    f"敵防御力: {format_damage(round_damage(values.enemy_defense))}",
                    f"防御デバフ: {_format_percent(values.defense_debuff_rate)}",
                )
            ),
            inline=True,
        ),
        DamageEmbedField(
            name="倍率条件",
            value="\n".join(
                (
                    f"属性相性: ×{_format_decimal(values.attribute_multiplier)}",
                    f"地形適性: ×{_format_decimal(values.terrain_multiplier)}",
                    f"会心ダメージ: {_format_percent(values.critical_damage_multiplier)}",
                    f"乱数: {_format_percent(values.random_min_multiplier)}〜{_format_percent(values.random_max_multiplier)}",
                    f"ヒット数: {result.hit_count}",
                    f"選択結果: {selection_label}",
                )
            ),
            inline=True,
        ),
        DamageEmbedField(
            name="計算途中の補正値",
            value="\n".join(
                (
                    f"実効攻撃力: {format_damage(round_damage(result.intermediate.effective_attack))}",
                    f"実効防御力: {format_damage(round_damage(result.intermediate.effective_defense))}",
                    f"防御倍率: ×{_format_decimal(result.intermediate.defense_multiplier, places=6)}",
                    f"基礎ダメージ: {format_damage(round_damage(result.intermediate.base_damage))}",
                    "乗算順: 基礎 × 与ダメ × 防御 × 属性 × 地形 × 乱数 × 会心",
                )
            ),
            inline=False,
        ),
        DamageEmbedField(
            name="非会心ダメージ（1ヒット）",
            value=_format_range(result.non_critical_per_hit),
            inline=True,
        ),
        DamageEmbedField(
            name="会心ダメージ（1ヒット）",
            value=_format_range(result.critical_per_hit),
            inline=True,
        ),
        DamageEmbedField(
            name=f"合計ダメージ（{result.hit_count}ヒット）",
            value="\n".join(
                (
                    f"非会心: {_format_range(result.non_critical_total)}",
                    f"会心: {_format_range(result.critical_total)}",
                    f"選択中（{selection_label}）: {_format_range(result.selected_total)}",
                )
            ),
            inline=False,
        ),
    )
    return DamageEmbedData(
        title="ブルーアーカイブ ダメージ計算",
        description="手動入力による概算です。スキル倍率は1ヒット当たりとして計算します。",
        fields=fields,
        footer=CAUTION_MESSAGE,
    )


def build_damage_embed(data: DamageEmbedData) -> discord.Embed:
    """表示データをdiscord.pyのEmbedへ変換する。"""

    embed = discord.Embed(
        title=data.title,
        description=data.description,
        color=discord.Color.blurple(),
    )
    for field in data.fields:
        embed.add_field(name=field.name, value=field.value, inline=field.inline)
    embed.set_footer(text=data.footer)
    return embed


def setup(client: discord.Client) -> None:
    """``/ba_damage`` をクライアントのCommandTreeへ登録する。"""

    @client.tree.command(name="ba_damage", description="ブルーアーカイブの想定ダメージを手動入力で計算します。")
    @app_commands.describe(
        attack="基礎攻撃力（0〜1,000,000,000）",
        skill_percent="1ヒット当たりのスキル倍率（例: 800%なら800）",
        enemy_defense="敵の防御力（0〜1,000,000,000）",
        attack_buff_percent="攻撃力バフ率（例: 50%なら50）",
        damage_buff_percent="与ダメージ増加率（例: 25%なら25）",
        defense_debuff_percent="敵への防御力デバフ率（100%超は実効防御0）",
        attribute_multiplier="属性相性倍率（例: Weakなら2.0）",
        terrain_multiplier="地形適性倍率（例: Sなら1.2）",
        critical_damage_percent="会心ダメージ倍率（例: 200%なら200）",
        hit_count="ヒット数（1〜100）",
        random_min_percent="ダメージ乱数の最小値（例: 90%なら90）",
        random_max_percent="ダメージ乱数の最大値（例: 110%なら110）",
        is_critical="選択結果を会心として表示するか",
    )
    async def slash_ba_damage(
        interaction: discord.Interaction,
        attack: app_commands.Range[int, 0, 1_000_000_000],
        skill_percent: app_commands.Range[float, 0.01, 1_000_000.0],
        enemy_defense: app_commands.Range[int, 0, 1_000_000_000],
        attack_buff_percent: app_commands.Range[float, 0.0, 1_000_000.0] = 0.0,
        damage_buff_percent: app_commands.Range[float, 0.0, 1_000_000.0] = 0.0,
        defense_debuff_percent: app_commands.Range[float, 0.0, 1_000_000.0] = 0.0,
        attribute_multiplier: app_commands.Range[float, 0.0001, 1000.0] = 1.0,
        terrain_multiplier: app_commands.Range[float, 0.0001, 1000.0] = 1.0,
        critical_damage_percent: app_commands.Range[float, 0.01, 1_000_000.0] = 200.0,
        hit_count: app_commands.Range[int, 1, 100] = 1,
        random_min_percent: app_commands.Range[float, 0.0, 1_000_000.0] = 100.0,
        random_max_percent: app_commands.Range[float, 0.01, 1_000_000.0] = 100.0,
        is_critical: bool = False,
    ) -> None:
        try:
            values = DamageCalculationInput.from_percentages(
                attack=attack,
                skill_percent=skill_percent,
                enemy_defense=enemy_defense,
                attack_buff_percent=attack_buff_percent,
                damage_buff_percent=damage_buff_percent,
                defense_debuff_percent=defense_debuff_percent,
                attribute_multiplier=attribute_multiplier,
                terrain_multiplier=terrain_multiplier,
                critical_damage_percent=critical_damage_percent,
                hit_count=hit_count,
                random_min_percent=random_min_percent,
                random_max_percent=random_max_percent,
                is_critical=is_critical,
            )
            result = calculate_damage(values)
        except DamageValidationError as exc:
            await interaction.response.send_message(f"入力エラー: {exc}", ephemeral=True)
            return
        except Exception:
            logger.exception("Failed to calculate /ba_damage.")
            await interaction.response.send_message(CALCULATION_FAILED_MESSAGE, ephemeral=True)
            return

        embed_data = build_damage_embed_data(values, result)
        await interaction.response.send_message(embed=build_damage_embed(embed_data))


def _format_range(values: DamageRange) -> str:
    return (
        f"{format_damage(values.minimum)}〜{format_damage(values.maximum)}"
        f"（平均 {format_damage(values.average)}）"
    )


def _format_percent(value: Decimal) -> str:
    return f"{_format_decimal(value * Decimal('100'))}%"


def _format_decimal(value: Decimal, *, places: int = 4) -> str:
    text = f"{value:.{places}f}".rstrip("0").rstrip(".")
    return text or "0"
