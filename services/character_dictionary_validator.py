"""Read-only consistency check for the character/series dictionary.

Combines `services/character_tag_dictionary.py`'s own bulk read APIs
(`list_series` / `list_character_details`) with its own normalization
function (`_normalize`, reused rather than reimplemented so "same text after
normalization" here always means the exact same thing it means to
`resolve_character_tags`) to look for data-authoring mistakes: duplicate or
colliding identifiers/aliases, blank/untrimmed fields, characters that can
never resolve, and characters whose notes still flag them as unconfirmed.

This module never mutates the dictionary and never calls
`resolve_character_tags` / `resolve_all_character_tags` - it only reads the
same read-only snapshots `/character_list` etc. already use, so it cannot
change resolution behavior.
"""

import logging
from dataclasses import dataclass
from typing import Sequence

from services.character_tag_dictionary import (
    CharacterDetail,
    SeriesListEntry,
    _normalize,
    list_character_details,
    list_series,
)


logger = logging.getLogger("secretary-bot.character_dictionary_validator")

SEVERITY_ERROR = "ERROR"
SEVERITY_WARNING = "WARNING"
SEVERITY_INFO = "INFO"
_SEVERITY_ORDER = {SEVERITY_ERROR: 0, SEVERITY_WARNING: 1, SEVERITY_INFO: 2}

# notes conventions already used throughout _RAW_CHARACTERS (see
# character_tag_dictionary.py's own module docstring/entries) to flag a
# character_tag/series_tag/reading that could not be confirmed from anything
# in this repository yet.
_UNCONFIRMED_NOTES_KEYWORDS = ("未確認", "要確認")


@dataclass(frozen=True)
class ValidationIssue:
    """One finding. `subject` is the character_tag/series_tag/alias text the
    issue is about, for grouping or drill-down display - never the full
    request/user text (this module never reads generation history)."""

    severity: str
    code: str
    message: str
    subject: str


@dataclass(frozen=True)
class ValidationReport:
    character_count: int
    series_count: int
    issues: tuple[ValidationIssue, ...]

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == SEVERITY_ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == SEVERITY_WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == SEVERITY_INFO)

    @property
    def is_usable(self) -> bool:
        """False only when at least one ERROR was found - matches this
        module's ERROR definition ("解決結果が不定、起動時障害、誤キャラ解決に
        つながる可能性が高い"). WARNING/INFO never affect this."""
        return self.error_count == 0


def _sorted_issues(issues: Sequence[ValidationIssue]) -> tuple[ValidationIssue, ...]:
    return tuple(sorted(issues, key=lambda issue: (_SEVERITY_ORDER.get(issue.severity, 99), issue.code, issue.subject)))


def _is_blank(text: str) -> bool:
    return text.strip() == ""


def _is_untrimmed(text: str) -> bool:
    return text != "" and text != text.strip()


def _norm(text: str) -> str:
    # Delegates to character_tag_dictionary._normalize (NFKC + casefold +
    # whitespace collapse) rather than reimplementing it, so "collide after
    # normalization" here means exactly what it means to alias resolution.
    return _normalize(text)


def _check_core_identity_fields(characters: Sequence[CharacterDetail], series: Sequence[SeriesListEntry]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for character in characters:
        for field_name, value in (
            ("character_tag", character.character_tag),
            ("display_name", character.display_name),
            ("series_tag", character.series_tag),
        ):
            if _is_blank(value):
                issues.append(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="empty_core_identity_field",
                        message=f"キャラクターの{field_name}が空です。",
                        subject=character.character_tag or character.display_name or "(unknown)",
                    )
                )
            elif _is_untrimmed(value):
                issues.append(
                    ValidationIssue(
                        severity=SEVERITY_WARNING,
                        code="untrimmed_field",
                        message=f"キャラクターの{field_name}に前後の空白があります: {value!r}",
                        subject=character.character_tag,
                    )
                )
    for series_entry in series:
        for field_name, value in (
            ("series_tag", series_entry.series_tag),
            ("series_name", series_entry.series_name),
        ):
            if _is_blank(value):
                issues.append(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="empty_core_identity_field",
                        message=f"作品の{field_name}が空です。",
                        subject=series_entry.series_tag or series_entry.series_name or "(unknown)",
                    )
                )
            elif _is_untrimmed(value):
                issues.append(
                    ValidationIssue(
                        severity=SEVERITY_WARNING,
                        code="untrimmed_field",
                        message=f"作品の{field_name}に前後の空白があります: {value!r}",
                        subject=series_entry.series_tag,
                    )
                )
    return issues


def _check_duplicate_series_tags(series: Sequence[SeriesListEntry]) -> list[ValidationIssue]:
    counts: dict[str, int] = {}
    for series_entry in series:
        counts[series_entry.series_tag] = counts.get(series_entry.series_tag, 0) + 1
    issues: list[ValidationIssue] = []
    for series_tag, count in counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    severity=SEVERITY_ERROR,
                    code="duplicate_series_tag",
                    message=(
                        f"series_tag「{series_tag}」が{count}件重複しています。series_tagは作品情報の参照キーのため、"
                        "重複すると後に定義された方だけが有効になります。"
                    ),
                    subject=series_tag,
                )
            )
    return issues


def _check_duplicate_character_tags(characters: Sequence[CharacterDetail]) -> list[ValidationIssue]:
    by_tag: dict[str, list[CharacterDetail]] = {}
    for character in characters:
        by_tag.setdefault(character.character_tag, []).append(character)
    issues: list[ValidationIssue] = []
    for character_tag, group in by_tag.items():
        if len(group) <= 1:
            continue
        series_tags = {c.series_tag for c in group}
        scope = "同一作品内で" if len(series_tags) == 1 else "複数作品にまたがって"
        issues.append(
            ValidationIssue(
                severity=SEVERITY_ERROR,
                code="duplicate_character_tag",
                message=(
                    f"character_tag「{character_tag}」が{scope}{len(group)}件重複しています"
                    f"（display_name: {', '.join(c.display_name for c in group)}）。"
                ),
                subject=character_tag,
            )
        )
    return issues


def _check_duplicate_display_names(characters: Sequence[CharacterDetail]) -> list[ValidationIssue]:
    by_name: dict[str, list[CharacterDetail]] = {}
    for character in characters:
        by_name.setdefault(character.display_name, []).append(character)
    issues: list[ValidationIssue] = []
    for display_name, group in by_name.items():
        if len(group) <= 1 or _is_blank(display_name):
            continue
        issues.append(
            ValidationIssue(
                severity=SEVERITY_WARNING,
                code="duplicate_display_name",
                message=(
                    f"display_name「{display_name}」が{len(group)}件重複しています"
                    f"（character_tag: {', '.join(c.character_tag for c in group)}）。"
                ),
                subject=display_name,
            )
        )
    return issues


def _alias_lists(character: CharacterDetail) -> tuple[tuple[str, ...], tuple[str, ...]]:
    safe = character.extra_fields.get("safe_aliases") or ()
    ambiguous = character.extra_fields.get("ambiguous_aliases") or ()
    return tuple(safe), tuple(ambiguous)


def _check_intra_character_alias_issues(characters: Sequence[CharacterDetail]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for character in characters:
        safe_aliases, ambiguous_aliases = _alias_lists(character)

        for label, aliases, code in (
            ("safe_aliases", safe_aliases, "duplicate_safe_alias"),
            ("ambiguous_aliases", ambiguous_aliases, "duplicate_ambiguous_alias"),
        ):
            seen_normalized: dict[str, str] = {}
            for alias in aliases:
                if _is_blank(alias):
                    issues.append(
                        ValidationIssue(
                            severity=SEVERITY_WARNING,
                            code="empty_alias",
                            message=f"{character.character_tag}の{label}に空文字のエイリアスがあります。",
                            subject=character.character_tag,
                        )
                    )
                    continue
                if _is_untrimmed(alias):
                    issues.append(
                        ValidationIssue(
                            severity=SEVERITY_WARNING,
                            code="untrimmed_field",
                            message=f"{character.character_tag}の{label}に前後の空白があるエイリアスがあります: {alias!r}",
                            subject=character.character_tag,
                        )
                    )
                normalized = _norm(alias)
                if normalized in seen_normalized:
                    previous = seen_normalized[normalized]
                    detail = "(表記は異なりますが正規化後に同一)" if previous != alias else ""
                    issues.append(
                        ValidationIssue(
                            severity=SEVERITY_WARNING,
                            code=code,
                            message=f"{character.character_tag}の{label}内でエイリアスが重複しています: {previous!r} / {alias!r} {detail}".strip(),
                            subject=character.character_tag,
                        )
                    )
                else:
                    seen_normalized[normalized] = alias

        safe_normalized = {_norm(alias) for alias in safe_aliases if not _is_blank(alias)}
        for alias in ambiguous_aliases:
            if _is_blank(alias):
                continue
            if _norm(alias) in safe_normalized:
                issues.append(
                    ValidationIssue(
                        severity=SEVERITY_WARNING,
                        code="safe_ambiguous_alias_overlap",
                        message=(
                            f"{character.character_tag}: エイリアス{alias!r}がsafe_aliasesとambiguous_aliasesの両方に"
                            "含まれています（safe側が優先されambiguous指定は無意味になります）。"
                        ),
                        subject=character.character_tag,
                    )
                )
    return issues


def _check_cross_character_alias_collisions(characters: Sequence[CharacterDetail]) -> list[ValidationIssue]:
    # normalized alias text -> list of (character_tag, raw alias) that use it.
    owners: dict[str, list[tuple[str, str]]] = {}
    for character in characters:
        safe_aliases, ambiguous_aliases = _alias_lists(character)
        for alias in (*safe_aliases, *ambiguous_aliases):
            if _is_blank(alias):
                continue
            owners.setdefault(_norm(alias), []).append((character.character_tag, alias))

    issues: list[ValidationIssue] = []
    for normalized_alias, owner_list in owners.items():
        distinct_characters = {tag for tag, _raw in owner_list}
        if len(distinct_characters) <= 1:
            continue
        detail = ", ".join(f"{tag}({raw!r})" for tag, raw in owner_list)
        issues.append(
            ValidationIssue(
                severity=SEVERITY_ERROR,
                code="cross_character_alias_collision",
                message=f"同じエイリアスが複数のキャラクターに割り当てられています: {detail}",
                subject=normalized_alias,
            )
        )
    return issues


def _check_series_aliases(series: Sequence[SeriesListEntry]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    owners: dict[str, list[tuple[str, str]]] = {}
    for series_entry in series:
        seen_normalized: dict[str, str] = {}
        for alias in series_entry.aliases:
            if _is_blank(alias):
                issues.append(
                    ValidationIssue(
                        severity=SEVERITY_WARNING,
                        code="empty_alias",
                        message=f"作品「{series_entry.series_tag}」のaliasesに空文字があります。",
                        subject=series_entry.series_tag,
                    )
                )
                continue
            if _is_untrimmed(alias):
                issues.append(
                    ValidationIssue(
                        severity=SEVERITY_WARNING,
                        code="untrimmed_field",
                        message=f"作品「{series_entry.series_tag}」のaliasesに前後の空白があります: {alias!r}",
                        subject=series_entry.series_tag,
                    )
                )
            normalized = _norm(alias)
            if normalized in seen_normalized:
                previous = seen_normalized[normalized]
                detail = "(表記は異なりますが正規化後に同一)" if previous != alias else ""
                issues.append(
                    ValidationIssue(
                        severity=SEVERITY_WARNING,
                        code="duplicate_series_alias",
                        message=f"作品「{series_entry.series_tag}」内でaliasが重複しています: {previous!r} / {alias!r} {detail}".strip(),
                        subject=series_entry.series_tag,
                    )
                )
            else:
                seen_normalized[normalized] = alias
            owners.setdefault(normalized, []).append((series_entry.series_tag, alias))

    for normalized_alias, owner_list in owners.items():
        distinct_series = {tag for tag, _raw in owner_list}
        if len(distinct_series) <= 1:
            continue
        detail = ", ".join(f"{tag}({raw!r})" for tag, raw in owner_list)
        issues.append(
            ValidationIssue(
                severity=SEVERITY_ERROR,
                code="cross_series_alias_collision",
                message=(
                    f"同じaliasが複数の作品に割り当てられています: {detail}。"
                    "series判定は先勝ちのため、一方の作品が常に隠れます。"
                ),
                subject=normalized_alias,
            )
        )
    return issues


def _check_empty_safe_aliases(characters: Sequence[CharacterDetail]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for character in characters:
        safe_aliases, _ambiguous = _alias_lists(character)
        if not safe_aliases:
            issues.append(
                ValidationIssue(
                    severity=SEVERITY_WARNING,
                    code="empty_safe_aliases",
                    message=(
                        f"{character.character_tag}（{character.display_name}）にsafe_aliasesがありません。"
                        "作品名の同時指定がないと解決できません。"
                    ),
                    subject=character.character_tag,
                )
            )
    return issues


def _check_series_alias_availability(characters: Sequence[CharacterDetail], series: Sequence[SeriesListEntry]) -> list[ValidationIssue]:
    aliases_by_series_tag = {series_entry.series_tag: series_entry.aliases for series_entry in series}
    issues: list[ValidationIssue] = []
    for character in characters:
        _safe_aliases, ambiguous_aliases = _alias_lists(character)
        if not ambiguous_aliases:
            continue
        series_aliases = aliases_by_series_tag.get(character.series_tag, ())
        if not series_aliases:
            issues.append(
                ValidationIssue(
                    severity=SEVERITY_WARNING,
                    code="series_alias_missing_for_ambiguous_character",
                    message=(
                        f"{character.character_tag}（{character.display_name}）はambiguous_aliasesの解決に"
                        f"作品「{character.series_tag}」のaliasが必要ですが、その作品にaliasが登録されていません。"
                        "このキャラクターのambiguous_aliasesは常に解決不能です。"
                    ),
                    subject=character.character_tag,
                )
            )
    return issues


def _check_unconfirmed_notes(characters: Sequence[CharacterDetail]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for character in characters:
        notes = character.extra_fields.get("notes") or ""
        if not isinstance(notes, str) or not notes:
            continue
        if any(keyword in notes for keyword in _UNCONFIRMED_NOTES_KEYWORDS):
            issues.append(
                ValidationIssue(
                    severity=SEVERITY_WARNING,
                    code="unconfirmed_notes",
                    message=f"{character.character_tag}（{character.display_name}）のnotesに未確認/要確認の記載があります。",
                    subject=character.character_tag,
                )
            )
    return issues


def validate_character_dictionary(
    *,
    characters: Sequence[CharacterDetail] | None = None,
    series: Sequence[SeriesListEntry] | None = None,
) -> ValidationReport:
    """Runs every consistency check against the dictionary and returns a
    single report. Read-only: never mutates the dictionary and never touches
    `resolve_character_tags`/`resolve_all_character_tags`.

    `characters`/`series` default to the live dictionary
    (`list_character_details()`/`list_series()`) but can be overridden with
    synthetic data for testing individual failure modes in isolation.

    Never raises: any unexpected failure while checking is logged and
    reported as a single INFO issue rather than propagating, so a bug in this
    diagnostic tool can never itself take down `/character_validate`.
    """
    resolved_characters = list(list_character_details()) if characters is None else list(characters)
    resolved_series = list(list_series()) if series is None else list(series)

    try:
        issues: list[ValidationIssue] = []
        issues.extend(_check_core_identity_fields(resolved_characters, resolved_series))
        issues.extend(_check_duplicate_series_tags(resolved_series))
        issues.extend(_check_duplicate_character_tags(resolved_characters))
        issues.extend(_check_duplicate_display_names(resolved_characters))
        issues.extend(_check_intra_character_alias_issues(resolved_characters))
        issues.extend(_check_cross_character_alias_collisions(resolved_characters))
        issues.extend(_check_series_aliases(resolved_series))
        issues.extend(_check_empty_safe_aliases(resolved_characters))
        issues.extend(_check_series_alias_availability(resolved_characters, resolved_series))
        issues.extend(_check_unconfirmed_notes(resolved_characters))
    except Exception:
        logger.exception("Character dictionary validation failed unexpectedly.")
        issues = [
            ValidationIssue(
                severity=SEVERITY_INFO,
                code="validation_internal_error",
                message="検証処理中に想定外のエラーが発生したため、一部の検査結果が欠けている可能性があります。",
                subject="",
            )
        ]

    return ValidationReport(
        character_count=len(resolved_characters),
        series_count=len(resolved_series),
        issues=_sorted_issues(issues),
    )
