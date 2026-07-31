"""キャラクター辞書の管理補助コマンド（未登録候補集計・整合性検査）。

`services/character_tag_dictionary.py` を辞書データの唯一の情報源とし、
このモジュール側で辞書データを複製しない。辞書の編集・追加・削除は行わない
読み取り専用。generation history の読み取りは既存の
`services/generation_history.py` を、辞書との突き合わせは
`services/character_missing_finder.py` を、整合性検査は
`services/character_dictionary_validator.py` をそれぞれ利用する。
"""

from __future__ import annotations

import asyncio
import io
import logging

import discord
from discord import app_commands

from services.character_dictionary_admin import is_character_admin_allowed
from services.character_dictionary_validator import (
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    ValidationReport,
    validate_character_dictionary,
)
from services.character_missing_finder import MissingCandidate, find_missing_character_candidates


logger = logging.getLogger("secretary-bot.character_dictionary")

CHARACTER_ADMIN_DENIED_MESSAGE = "このコマンドは許可ユーザーだけが使えます。"
NO_MISSING_CANDIDATES_MESSAGE = (
    "未登録候補は見つかりませんでした。スキャン対象の生成履歴の範囲では、現在の辞書でカバーできています。"
)
MISSING_SCAN_FAILED_MESSAGE = "未登録候補の集計に失敗しました。"
VALIDATION_FAILED_MESSAGE = "辞書検証に失敗しました。"

# Discordの1メッセージ本文は2000文字までなので、embed分を差し引いた安全域。
# これを超える詳細は添付テキストファイルに切り替える。
VALIDATION_DETAIL_INLINE_LIMIT = 1500
_SEVERITY_LABELS = {SEVERITY_ERROR: "ERROR", SEVERITY_WARNING: "WARNING", SEVERITY_INFO: "INFO"}

EMBED_FIELD_VALUE_LIMIT = 1024


def _truncate_field_value(value: str, limit: int = EMBED_FIELD_VALUE_LIMIT) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def build_missing_candidates_embed(candidates: list[MissingCandidate]) -> discord.Embed:
    embed = discord.Embed(
        title="未登録キャラクター候補",
        description=(
            f"上位{len(candidates)}件。辞書に登録済みだが、作品名が同時に指定されていないため"
            "現在は解決できていない候補です（未登録の新規キャラクター名の自動抽出は行っていません）。"
        ),
        color=discord.Color.orange(),
    )
    for candidate in candidates:
        value = "\n".join(
            (
                f"使用回数: {candidate.occurrence_count}回",
                f"最終使用日時: {candidate.last_used_at}",
                f"代表的な元入力: {candidate.sample_request_text}",
                f"作品名らしき語: {candidate.series_name}",
                f"理由: {candidate.reason}",
            )
        )
        embed.add_field(
            name=f"{candidate.matched_alias} — {candidate.occurrence_count}回",
            value=_truncate_field_value(value),
            inline=False,
        )
    return embed


def build_validation_summary_embed(report: ValidationReport) -> discord.Embed:
    usable = report.is_usable
    embed = discord.Embed(
        title="キャラクター辞書検証",
        color=discord.Color.green() if usable else discord.Color.red(),
    )
    embed.add_field(name="登録キャラ", value=str(report.character_count), inline=True)
    embed.add_field(name="登録作品", value=str(report.series_count), inline=True)
    embed.add_field(name="結果", value="使用可能" if usable else "要修正", inline=True)
    embed.add_field(name="ERROR", value=str(report.error_count), inline=True)
    embed.add_field(name="WARNING", value=str(report.warning_count), inline=True)
    embed.add_field(name="INFO", value=str(report.info_count), inline=True)
    return embed


def build_validation_detail_text(report: ValidationReport) -> str:
    if not report.issues:
        return "検出された問題はありません。"
    lines: list[str] = []
    for severity in (SEVERITY_ERROR, SEVERITY_WARNING, SEVERITY_INFO):
        matching = [issue for issue in report.issues if issue.severity == severity]
        if not matching:
            continue
        lines.append(f"[{_SEVERITY_LABELS[severity]}] {len(matching)}件")
        for issue in matching:
            lines.append(f"- ({issue.code}) {issue.message}")
        lines.append("")
    return "\n".join(lines).strip()


def _validation_detail_file(text: str) -> discord.File:
    buffer = io.BytesIO(text.encode("utf-8"))
    return discord.File(buffer, filename="character_validate_detail.txt")


def setup(client: discord.Client) -> None:
    """``/character_missing`` ``/character_validate`` をCommandTreeへ登録する。"""

    @client.tree.command(
        name="character_missing",
        description="生成履歴の中から、辞書で未解決のキャラクター候補を集計します（許可ユーザー限定）。",
    )
    async def slash_character_missing(interaction: discord.Interaction) -> None:
        if not is_character_admin_allowed(interaction.user.id):
            logger.warning("Denied /character_missing by user_id=%s", interaction.user.id)
            await interaction.response.send_message(CHARACTER_ADMIN_DENIED_MESSAGE, ephemeral=True)
            return

        await interaction.response.defer(ephemeral=True)
        try:
            candidates = await asyncio.to_thread(find_missing_character_candidates)
        except Exception:
            logger.exception("Failed to build /character_missing report. user_id=%s", interaction.user.id)
            await interaction.followup.send(MISSING_SCAN_FAILED_MESSAGE, ephemeral=True)
            return

        if not candidates:
            await interaction.followup.send(NO_MISSING_CANDIDATES_MESSAGE, ephemeral=True)
            return

        await interaction.followup.send(embed=build_missing_candidates_embed(candidates), ephemeral=True)

    @client.tree.command(name="character_validate", description="キャラクター辞書全体の整合性を読み取り専用で検査します。")
    async def slash_character_validate(interaction: discord.Interaction) -> None:
        try:
            report = validate_character_dictionary()
            embed = build_validation_summary_embed(report)
            detail_text = build_validation_detail_text(report)

            if len(detail_text) <= VALIDATION_DETAIL_INLINE_LIMIT:
                await interaction.response.send_message(content=detail_text, embed=embed, ephemeral=True)
                return

            file = _validation_detail_file(detail_text)
            try:
                await interaction.response.send_message(
                    content="詳細は添付ファイルを確認してください。",
                    embed=embed,
                    file=file,
                    ephemeral=True,
                )
            finally:
                file.close()
        except Exception:
            logger.exception("Failed to build /character_validate report.")
            await interaction.response.send_message(VALIDATION_FAILED_MESSAGE, ephemeral=True)
