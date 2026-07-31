"""キャラクター辞書の閲覧・診断コマンド（一覧・詳細・検索・未登録候補集計・整合性検査）。

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
from collections import OrderedDict

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
from services.character_tag_dictionary import (
    CharacterDetail,
    CharacterListEntry,
    find_characters_by_name,
    get_character_detail,
    list_characters,
    search_characters,
)


logger = logging.getLogger("secretary-bot.character_dictionary")

SERIES_PER_PAGE = 8
SEARCH_RESULTS_PER_PAGE = 20

NOT_FOUND_MESSAGE = "該当するキャラクターが見つかりませんでした。"
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

# `/character_info` `/character_search` が表示する extra_fields のラベル。
# 未登録のキー（将来追加される appearance_tags / identity_hints など）は
# ここに無くても _humanize_field_name() でそのまま表示できるので、
# この辞書自体を更新しなくても新フィールドの表示は壊れない。
_FIELD_LABELS: dict[str, str] = {
    "safe_aliases": "確定エイリアス（単独で解決）",
    "ambiguous_aliases": "曖昧エイリアス（作品名と併記時のみ解決）",
    "notes": "備考",
    "appearance_tags": "外見タグ",
    "identity_hints": "識別ヒント",
}


def _humanize_field_name(name: str) -> str:
    return _FIELD_LABELS.get(name, name)


EMBED_FIELD_VALUE_LIMIT = 1024


def _truncate_field_value(value: str, limit: int = EMBED_FIELD_VALUE_LIMIT) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _format_field_value(value: object) -> str:
    if value is None:
        return "未設定"
    if isinstance(value, (tuple, list, set, frozenset)):
        items = [str(item) for item in value if str(item).strip()]
        text = "、".join(items) if items else "未設定"
    else:
        text = str(value).strip() or "未設定"
    return _truncate_field_value(text)


def _group_by_series(entries: tuple[CharacterListEntry, ...]) -> "OrderedDict[tuple[str, str], list[CharacterListEntry]]":
    grouped: "OrderedDict[tuple[str, str], list[CharacterListEntry]]" = OrderedDict()
    for entry in entries:
        key = (entry.series_tag, entry.series_name)
        grouped.setdefault(key, []).append(entry)
    return grouped


def build_character_list_embed(entries: tuple[CharacterListEntry, ...], page: int) -> tuple[discord.Embed, int, int]:
    """一覧embedと、実際に表示したページ番号・総ページ数を返す。"""

    grouped = _group_by_series(entries)
    series_keys = list(grouped.keys())
    total_pages = max((len(series_keys) + SERIES_PER_PAGE - 1) // SERIES_PER_PAGE, 1)
    page = min(max(page, 1), total_pages)
    selected_keys = series_keys[(page - 1) * SERIES_PER_PAGE : page * SERIES_PER_PAGE]

    embed = discord.Embed(
        title="キャラクター辞書 一覧",
        description=f"総登録数: {len(entries)}件（{len(series_keys)}作品）",
        color=discord.Color.blurple(),
    )
    for series_tag, series_name in selected_keys:
        characters = grouped[(series_tag, series_name)]
        value = _truncate_field_value("\n".join(f"・{c.display_name}" for c in characters))
        embed.add_field(name=f"{series_name}（{len(characters)}件）", value=value, inline=False)
    embed.set_footer(text=f"ページ {page}/{total_pages}（別ページは /character_list page:番号）")
    return embed, page, total_pages


def build_character_detail_embed(detail: CharacterDetail) -> discord.Embed:
    embed = discord.Embed(
        title=detail.display_name,
        description=f"作品: {detail.series_name}",
        color=discord.Color.blurple(),
    )
    embed.add_field(name="character_tag", value=detail.character_tag, inline=True)
    embed.add_field(name="series_tag", value=detail.series_tag, inline=True)
    for field_name, value in detail.extra_fields.items():
        embed.add_field(name=_humanize_field_name(field_name), value=_format_field_value(value), inline=False)
    return embed


def build_character_search_embed(
    query: str, entries: tuple[CharacterListEntry, ...], page: int
) -> tuple[discord.Embed, int, int]:
    grouped = _group_by_series(entries)
    total_pages = max((len(entries) + SEARCH_RESULTS_PER_PAGE - 1) // SEARCH_RESULTS_PER_PAGE, 1)
    page = min(max(page, 1), total_pages)

    embed = discord.Embed(
        title=f"キャラクター検索: 「{query}」",
        description=f"該当 {len(entries)}件（{len(grouped)}作品）",
        color=discord.Color.blurple(),
    )
    selected = entries[(page - 1) * SEARCH_RESULTS_PER_PAGE : page * SEARCH_RESULTS_PER_PAGE]
    selected_grouped = _group_by_series(selected)
    for (series_tag, series_name), characters in selected_grouped.items():
        value = _truncate_field_value("\n".join(f"・{c.display_name}" for c in characters))
        embed.add_field(name=f"{series_name}（{len(characters)}件）", value=value, inline=False)
    embed.set_footer(text=f"ページ {page}/{total_pages}（別ページは /character_search query:{query} page:番号）")
    return embed, page, total_pages


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
    """``/character_list`` ``/character_info`` ``/character_search`` をCommandTreeへ登録する。"""

    @client.tree.command(name="character_list", description="登録済みキャラクター一覧を作品別に表示します。")
    @app_commands.describe(page="表示するページ番号（既定: 1）")
    async def slash_character_list(interaction: discord.Interaction, page: int = 1) -> None:
        try:
            entries = list_characters()
            if not entries:
                await interaction.response.send_message("キャラクター辞書はまだ空です。", ephemeral=True)
                return
            embed, _, _ = build_character_list_embed(entries, page)
            await interaction.response.send_message(embed=embed, ephemeral=True)
        except Exception:
            logger.exception("Failed to build /character_list.")
            await interaction.response.send_message("一覧の取得に失敗しました。", ephemeral=True)

    @client.tree.command(name="character_info", description="キャラクターの詳細情報を表示します（部分一致検索可）。")
    @app_commands.describe(character="表示したいキャラクター名（部分一致）")
    async def slash_character_info(interaction: discord.Interaction, character: str) -> None:
        try:
            matches = find_characters_by_name(character)
            if not matches:
                await interaction.response.send_message(NOT_FOUND_MESSAGE, ephemeral=True)
                return
            if len(matches) > 1:
                lines = [f"・{m.display_name}（{m.series_name}）" for m in matches[:20]]
                message = (
                    f"「{character}」に複数のキャラクターが該当しました。もう少し詳しく入力してください。\n"
                    + "\n".join(lines)
                )
                await interaction.response.send_message(message, ephemeral=True)
                return

            detail = get_character_detail(matches[0].character_tag)
            if detail is None:
                await interaction.response.send_message(NOT_FOUND_MESSAGE, ephemeral=True)
                return
            await interaction.response.send_message(embed=build_character_detail_embed(detail), ephemeral=True)
        except Exception:
            logger.exception("Failed to build /character_info.")
            await interaction.response.send_message("詳細情報の取得に失敗しました。", ephemeral=True)

    @client.tree.command(name="character_search", description="作品名・タグ・alias・display_nameからキャラクターを検索します。")
    @app_commands.describe(query="検索語（作品名・タグ・alias・display_nameのいずれか）", page="表示するページ番号（既定: 1）")
    async def slash_character_search(interaction: discord.Interaction, query: str, page: int = 1) -> None:
        try:
            entries = search_characters(query)
            if not entries:
                await interaction.response.send_message(NOT_FOUND_MESSAGE, ephemeral=True)
                return
            embed, _, _ = build_character_search_embed(query, entries, page)
            await interaction.response.send_message(embed=embed, ephemeral=True)
        except Exception:
            logger.exception("Failed to build /character_search.")
            await interaction.response.send_message("検索に失敗しました。", ephemeral=True)

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
