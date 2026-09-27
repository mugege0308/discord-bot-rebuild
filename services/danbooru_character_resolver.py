"""Conservative Japanese-character-name resolution through the local Danbooru DB.

The hand-maintained ``character_tag_dictionary`` remains the first authority.
This module is deliberately narrower than the generic appearance resolver:
the only positive candidates it can return are active ``category=character``
rows from the local SQLite snapshot.  The LLM may translate/search-plan a
fragment, but it never supplies a final character tag.
"""

from __future__ import annotations

import json
import difflib
import logging
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from services.character_tag_dictionary import CharacterTagResolution
from services.danbooru_tag_store import (
    DEFAULT_DANBOORU_DB_PATH,
    DanbooruTagCandidate,
    DanbooruTagStore,
    normalize_search_text,
    normalize_tag_name,
)
from services.verified_character_registry import VerifiedCharacter


logger = logging.getLogger(__name__)

CHARACTER_RESOLVER_VERSION = "0.2.0"
CHARACTER_SEARCH_INTERNAL_LIMIT = 50
CHARACTER_CANDIDATE_LIMIT = 8
CHARACTER_ACCEPT_CONFIDENCE = 0.90
CHARACTER_FALLBACK_MIN_SCORE = 0.78
CHARACTER_FALLBACK_MARGIN = 0.06
CHARACTER_CATEGORY = "character"
COPYRIGHT_CATEGORY = "copyright"
CHARACTER_QUERY_HINTS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "danbooru" / "character_query_hints.json"
)
_CHARACTER_QUERY_HINTS: dict[str, tuple[str, ...]] | None = None
COPYRIGHT_QUERY_HINTS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "danbooru" / "copyright_query_hints.json"
)
_COPYRIGHT_QUERY_HINTS: dict[str, tuple[str, ...]] | None = None

# This gate prevents ordinary Japanese visual nouns ("パーカー", "女の子",
# "上半身") from causing a proper-name search.  A future version can replace
# it with a token classifier; the MVP intentionally prefers abstention.
_CHARACTER_NAME_RE = re.compile(
    r"(?<![\w一-龯々ぁ-んァ-ヶー])([一-龯々]{2,12}[ぁ-んァ-ヶー]{0,6})(?="
    r"(?:が|は|を|の|と|に|へ|も|、|，|,|。|！|!|\?|？|：|:|（|\(|$))"
)
_KATAKANA_CHARACTER_NAME_RE = re.compile(
    r"(?<![\w一-龯々ぁ-んァ-ヶー・])([ァ-ヶー]{2,16}・[ァ-ヶー]{2,16})(?="
    r"(?:が|は|を|の|と|に|へ|も|、|，|,|。|！|!|\?|？|：|:|（|\(|$))"
)
_WORK_CHARACTER_RE = re.compile(
    r"(?<![\w一-龯々ぁ-んァ-ヶー])"
    r"(?P<work>[^、,。！？!?\n]{2,40}?)の"
    r"(?P<character>[一-龯々ぁ-んァ-ヶーA-Za-z0-9][^、,。！？!?\n]{0,48}?)"
    r"(?=(?:、|，|,|。|！|!|\?|？|：|:|（|\(|$))"
)
_CHARACTER_ACTION_MARKERS = (
    "振り返って",
    "しゃがんで",
    "前かがみ",
    "後ろで",
    "前で",
    "横で",
    "上から",
    "下から",
    "笑って",
    "座って",
    "立って",
    "歩いて",
    "走って",
    "見て",
    "手を",
)
_CHARACTER_ACTION_BOUNDARY_RE = re.compile(
    r"\s+(?=(?:" + "|".join(map(re.escape, _CHARACTER_ACTION_MARKERS)) + r"))"
)
_LEADING_CHARACTER_PARTICLE_RE = re.compile(r"^(?:が|は|を|に|で|と|へ|から|より|も)\s*")
_CHARACTER_PARTICLE_BOUNDARY_RE = re.compile(
    r"(?P<particle>から|より|が|は|を|に|で|と|へ|も)(?=[一-龯々ぁ-んァ-ヶーA-Za-z0-9])"
)


def _has_character_tail_boundary(value: str) -> bool:
    return bool(
        _CHARACTER_ACTION_BOUNDARY_RE.search(value)
        or _CHARACTER_PARTICLE_BOUNDARY_RE.search(value)
    )
_CHARACTER_STOPWORDS = {
    "上半身",
    "下半身",
    "振り返",
    "見下ろ",
    "見上げ",
    "動きある",
    "構図",
    "髪型",
    "服装",
    "女の子",
    "男の子",
    "キャラクター",
    "指定なし",
    "指定しない",
    "服装指定なし",
    "笑ってる",
    "笑顔",
    "袖長め",
    "袖長い",
    "長袖",
    "半袖",
    "髪長い",
    "髪ふわっと",
    "白い上着",
    "短め",
    "横顔",
    "表情",
    # Katakana-only visual nouns must not become proper-name searches when
    # the gate is broad enough to catch names such as
    # ``アリス・マーガトロイド``.
    "パーカー",
    "スカート",
    "ジャケット",
    "コート",
    "ワンピース",
    "セーラー服",
    "制服",
    "シャツ",
    "ブラウス",
    "リボン",
    "マフラー",
    "ショートヘア",
    "ロングヘア",
    "ツインテール",
    "ポニーテール",
    "ストレートヘア",
    "ウェーブヘア",
}
QUERY_PLANNER_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "danbooru_character_query_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "characters": {
                    "type": "array",
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "input_name": {"type": "string", "minLength": 1},
                            "queries": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 5,
                                "items": {"type": "string", "minLength": 1},
                            },
                            "work_queries": {
                                "type": "array",
                                "maxItems": 5,
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["input_name", "queries", "work_queries"],
                        "additionalProperties": False,
                    },
                },
                "unresolved": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {"type": "string"},
                },
            },
            "required": ["characters", "unresolved"],
            "additionalProperties": False,
        },
    },
}


LlmJsonCallback = Callable[[str, str, Mapping[str, Any], int], str]


@dataclass(frozen=True)
class CharacterSearchIntent:
    input_name: str
    queries: tuple[str, ...]
    work_queries: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtractedCharacterContext:
    work_name: str = ""
    character_name: str = ""
    raw_character_span: str = ""
    remaining_prompt_text: str = ""


@dataclass(frozen=True)
class DanbooruCharacterResolution:
    resolver_version: str = CHARACTER_RESOLVER_VERSION
    snapshot_version: str = "unknown"
    input_character_name: str = ""
    normalized_character_name: str = ""
    extracted_work_name: str = ""
    normalized_work_name: str = ""
    extracted_character_name: str = ""
    raw_character_span: str = ""
    remaining_prompt_text: str = ""
    character_candidate_attempts: tuple[dict[str, object], ...] = ()
    work_alias_hit: bool = False
    copyright_candidates: tuple[DanbooruTagCandidate, ...] = ()
    registry_hit: bool = False
    registry_character_id: int | None = None
    generated_queries: tuple[str, ...] = ()
    llm_character_queries: tuple[str, ...] = ()
    generated_work_queries: tuple[str, ...] = ()
    context_aliases: tuple[str, ...] = ()
    candidate_character_tags: tuple[DanbooruTagCandidate, ...] = ()
    selected_tag_id: int | None = None
    selected_character_tag: str = ""
    selected_prompt_tag: str = ""
    selected_candidate: str = ""
    selected_copyright_tag: str = ""
    candidate_match_strength: float = 0.0
    deterministic_romanization: str = ""
    copyright_scoped_candidate_count: int = 0
    fuzzy_candidate_scores: tuple[dict[str, object], ...] = ()
    fallback_used: bool = False
    confirmation_required: bool = False
    confirmed_character_tag: str = ""
    saved_alias: str = ""
    confidence: float = 0.0
    unresolved_reason: str = ""
    rejected_tag_ids: tuple[int, ...] = ()
    abstained: bool = False
    error: str = ""

    @property
    def has_selected_tag(self) -> bool:
        return bool(self.confirmed_character_tag and self.selected_prompt_tag)

    @property
    def has_candidate(self) -> bool:
        return bool(self.candidate_character_tags)

    @property
    def selected_tag_ids(self) -> tuple[int, ...]:
        return (self.selected_tag_id,) if self.selected_tag_id is not None else ()

    @property
    def selected_prompt_tags(self) -> tuple[str, ...]:
        return (self.selected_prompt_tag,) if self.has_selected_tag else ()

    def to_trace_dict(self) -> dict[str, object]:
        return {
            "resolver_version": self.resolver_version,
            "snapshot_version": self.snapshot_version,
            "input_character_name": self.input_character_name,
            "normalized_character_name": self.normalized_character_name,
            "extracted_work_name": self.extracted_work_name,
            "normalized_work_name": self.normalized_work_name,
            "extracted_character_name": self.extracted_character_name,
            "raw_character_span": self.raw_character_span,
            "remaining_prompt_text": self.remaining_prompt_text,
            "character_candidate_attempts": list(self.character_candidate_attempts),
            "work_alias_hit": self.work_alias_hit,
            "copyright_candidates": [
                candidate.to_trace_dict() for candidate in self.copyright_candidates
            ],
            "registry_hit": self.registry_hit,
            "registry_character_id": self.registry_character_id,
            "generated_queries": list(self.generated_queries),
            "llm_character_queries": list(self.llm_character_queries or self.generated_queries),
            "generated_work_queries": list(self.generated_work_queries),
            "context_aliases": list(self.context_aliases),
            "candidate_character_tags": [
                candidate.to_trace_dict() for candidate in self.candidate_character_tags
            ],
            "selected_tag_id": self.selected_tag_id,
            "selected_character_tag": self.selected_character_tag,
            "selected_prompt_tag": self.selected_prompt_tag,
            "selected_candidate": self.selected_candidate,
            "selected_copyright_tag": self.selected_copyright_tag,
            "candidate_match_strength": round(self.candidate_match_strength, 4),
            "deterministic_romanization": self.deterministic_romanization,
            "copyright_scoped_candidate_count": self.copyright_scoped_candidate_count,
            "fuzzy_candidate_scores": list(self.fuzzy_candidate_scores),
            "fallback_used": self.fallback_used,
            "confirmation_required": self.confirmation_required,
            "confirmed_character_tag": self.confirmed_character_tag,
            "saved_alias": self.saved_alias,
            "confidence": round(self.confidence, 4),
            "rejected_tag_ids": list(self.rejected_tag_ids),
            "abstained": self.abstained,
            "unresolved_reason": self.unresolved_reason,
            "error": self.error,
        }

    @classmethod
    def from_verified_registry(
        cls,
        *,
        input_character_name: str,
        character: VerifiedCharacter,
        matched_alias: str = "",
    ) -> "DanbooruCharacterResolution":
        return cls(
            input_character_name=input_character_name,
            normalized_character_name=normalize_search_text(input_character_name),
            registry_hit=True,
            registry_character_id=character.character_id,
            selected_character_tag=character.canonical_danbooru_tag,
            selected_prompt_tag=character.prompt_tag,
            selected_candidate=character.canonical_danbooru_tag,
            selected_copyright_tag=character.copyright_tag,
            candidate_match_strength=1.0,
            confirmed_character_tag=character.canonical_danbooru_tag,
            saved_alias=matched_alias or input_character_name,
            confidence=1.0,
            unresolved_reason="",
        )

    def with_registry_confirmation(
        self,
        *,
        character: VerifiedCharacter,
        saved_alias: str,
    ) -> "DanbooruCharacterResolution":
        return replace(
            self,
            registry_hit=True,
            registry_character_id=character.character_id,
            selected_character_tag=character.canonical_danbooru_tag,
            selected_prompt_tag=character.prompt_tag,
            selected_candidate=character.canonical_danbooru_tag,
            selected_copyright_tag=character.copyright_tag or self.selected_copyright_tag,
            candidate_match_strength=max(self.candidate_match_strength, self.confidence),
            confirmation_required=False,
            confirmed_character_tag=character.canonical_danbooru_tag,
            saved_alias=saved_alias,
            confidence=1.0,
            unresolved_reason="",
        )


def extract_work_character_context(text: str) -> ExtractedCharacterContext:
    """Split the explicit Japanese ``work の character`` construction.

    The split is deliberately syntax-first and conservative.  It does not
    claim that either side is real; copyright and character DB validation are
    still performed later.  Returning the character side separately prevents
    a work title such as ``崩壊スターレイル`` from entering the character
    planner as a name.
    """

    for match in _WORK_CHARACTER_RE.finditer(str(text or "")):
        work_name = match.group("work").strip()
        character_name = match.group("character").strip()
        if work_name and character_name:
            remaining = str(text or "")[match.end() :].strip()
            remaining = _LEADING_CHARACTER_PARTICLE_RE.sub("", remaining, count=1).strip()
            return ExtractedCharacterContext(
                work_name=work_name,
                character_name=character_name,
                raw_character_span=character_name,
                remaining_prompt_text=remaining,
            )
    return ExtractedCharacterContext()


def _build_character_span_attempts(
    context: ExtractedCharacterContext,
) -> tuple[tuple[str, str], ...]:
    """Return longest-to-shorter character spans with their remaining text.

    The raw span is retained for traceability, but an obvious action phrase
    after a whitespace boundary is tried as a shorter name candidate before
    handing the text to the planner.  This preserves compound English names
    such as ``Silver Wolf`` because no split is made unless the following
    token is a known action/direction phrase.
    """

    raw = str(context.raw_character_span or context.character_name or "").strip()
    if not raw:
        return ()
    suffix = str(context.remaining_prompt_text or "").strip()
    attempts: list[tuple[str, str]] = [(raw, suffix)]

    for match in _CHARACTER_ACTION_BOUNDARY_RE.finditer(raw):
        candidate = raw[: match.start()].strip()
        remaining = raw[match.end() :].strip()
        if suffix:
            remaining = f"{remaining} {suffix}".strip()
        if candidate:
            attempts.append((candidate, remaining))

    for match in _CHARACTER_PARTICLE_BOUNDARY_RE.finditer(raw):
        candidate = raw[: match.start()].strip()
        remaining = raw[match.end() :].strip()
        if suffix:
            remaining = f"{remaining} {suffix}".strip()
        if candidate:
            attempts.append((candidate, remaining))

    # If an action marker was not recognized, keep conservative whitespace
    # prefixes as fallbacks.  They are only accepted after DB candidate
    # validation, so compound names are not split merely because they contain
    # a space.
    for index, character in enumerate(raw):
        if not character.isspace():
            continue
        candidate = raw[:index].strip()
        remaining = raw[index:].strip()
        if suffix:
            remaining = f"{remaining} {suffix}".strip()
        if candidate:
            attempts.append((candidate, remaining))

    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for candidate, remaining in attempts:
        key = normalize_search_text(candidate)
        if key and key not in seen:
            seen.add(key)
            unique.append((candidate, remaining))
    return tuple(unique)


def extract_possible_character_names(text: str) -> tuple[str, ...]:
    """Extract name-shaped spans after removing an explicit work prefix."""

    context = extract_work_character_context(text)
    if context.work_name:
        for candidate, _remaining in _build_character_span_attempts(context):
            if _has_character_tail_boundary(candidate):
                continue
            if candidate and candidate not in _CHARACTER_STOPWORDS:
                return (candidate,)
        return ()

    source_text = str(text or "")

    found: list[str] = []
    for pattern in (_CHARACTER_NAME_RE, _KATAKANA_CHARACTER_NAME_RE):
        for match in pattern.finditer(source_text):
            value = match.group(1).strip()
            if not value or value in _CHARACTER_STOPWORDS:
                continue
            if normalize_search_text(value) not in {
                normalize_search_text(item) for item in found
            }:
                found.append(value)
    return tuple(found[:4])


def extract_character_context_aliases(text: str, character_name: str) -> tuple[str, ...]:
    """Keep explicit Japanese work/context wording for future short aliases."""

    escaped_name = re.escape(str(character_name or "").strip())
    if not escaped_name:
        return ()
    patterns = (
        re.compile(rf"([^、,。！？!?\n]{{2,40}})の(?:{escaped_name})"),
        re.compile(rf"[作品シリーズタイトル作][:：]\s*([^、,。！？!?\n]{{2,40}})"),
        re.compile(rf"[（(]([^）)]{{2,40}})[）)]"),
    )
    found: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(str(text or "")):
            value = match.group(1).strip()
            if value and normalize_search_text(value) not in {
                normalize_search_text(item) for item in found
            }:
                found.append(value)
    return tuple(found[:4])


def _load_character_query_hints() -> dict[str, tuple[str, ...]]:
    global _CHARACTER_QUERY_HINTS
    if _CHARACTER_QUERY_HINTS is not None:
        return _CHARACTER_QUERY_HINTS
    try:
        payload = json.loads(CHARACTER_QUERY_HINTS_PATH.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        _CHARACTER_QUERY_HINTS = {}
        return _CHARACTER_QUERY_HINTS
    if not isinstance(payload, Mapping):
        _CHARACTER_QUERY_HINTS = {}
        return _CHARACTER_QUERY_HINTS
    hints: dict[str, tuple[str, ...]] = {}
    for raw_name, raw_queries in payload.items():
        if not isinstance(raw_name, str) or not isinstance(raw_queries, list):
            continue
        queries = tuple(
            str(query).strip()[:100]
            for query in raw_queries[:5]
            if isinstance(query, str) and query.strip()
        )
        if queries:
            hints[normalize_search_text(raw_name)] = tuple(dict.fromkeys(queries))
    _CHARACTER_QUERY_HINTS = hints
    return hints


def lookup_character_query_hints(character_name: str) -> tuple[str, ...]:
    """Return query-only hints; never treat them as verified character tags."""

    return _load_character_query_hints().get(normalize_search_text(character_name), ())


def _load_copyright_query_hints() -> dict[str, tuple[str, ...]]:
    global _COPYRIGHT_QUERY_HINTS
    if _COPYRIGHT_QUERY_HINTS is not None:
        return _COPYRIGHT_QUERY_HINTS
    try:
        payload = json.loads(COPYRIGHT_QUERY_HINTS_PATH.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        _COPYRIGHT_QUERY_HINTS = {}
        return _COPYRIGHT_QUERY_HINTS
    if not isinstance(payload, Mapping):
        _COPYRIGHT_QUERY_HINTS = {}
        return _COPYRIGHT_QUERY_HINTS
    hints: dict[str, tuple[str, ...]] = {}
    for raw_name, raw_queries in payload.items():
        if not isinstance(raw_name, str) or not isinstance(raw_queries, list):
            continue
        queries = tuple(
            str(query).strip()[:100]
            for query in raw_queries[:5]
            if isinstance(query, str) and query.strip()
        )
        if queries:
            hints[normalize_search_text(raw_name)] = tuple(dict.fromkeys(queries))
    _COPYRIGHT_QUERY_HINTS = hints
    return hints


def lookup_copyright_query_hints(work_name: str) -> tuple[str, ...]:
    """Translate a curated work alias into English/DB search queries."""

    return _load_copyright_query_hints().get(normalize_search_text(work_name), ())


def lookup_copyright_aliases(work_name: str) -> tuple[str, ...]:
    """Return aliases sharing a curated copyright query group.

    These aliases are only persisted after the user confirms a character;
    they never auto-confirm a character or a work on their own.
    """

    normalized_name = normalize_search_text(work_name)
    queries = set(_load_copyright_query_hints().get(normalized_name, ()))
    if not queries:
        return ()
    aliases = [
        alias
        for alias, alias_queries in _load_copyright_query_hints().items()
        if queries.intersection(alias_queries)
    ]
    return tuple(dict.fromkeys(aliases))


def should_search_character_tags(
    text: str,
    *,
    manual_resolutions: Sequence[CharacterTagResolution] = (),
    selected_character_name: str | None = None,
    db_path=DEFAULT_DANBOORU_DB_PATH,
) -> bool:
    if selected_character_name or manual_resolutions:
        return False
    if any(cue in str(text or "") for cue in ("指定なし", "指定しない", "特にない", "おまかせ", "お任せ")):
        return False
    clean_text = str(text or "").strip()
    if lookup_copyright_query_hints(clean_text):
        # A work-only request must not turn the work title into a character.
        return False
    if not extract_possible_character_names(clean_text):
        return False
    return DanbooruTagStore(db_path).available


def build_character_query_planner_instruction(
    candidate_names: Iterable[str],
    work_hints: Iterable[str] = (),
) -> str:
    names = ", ".join(str(value).strip() for value in candidate_names if str(value).strip()) or "none"
    works = ", ".join(str(value).strip() for value in work_hints if str(value).strip()) or "none"
    return (
        "You are a conservative Japanese character-name search planner for a Danbooru tag database.\n"
        "Return only the requested JSON schema.\n"
        f"The only possible Japanese character-name fragments are: {names}.\n"
        f"Possible work/title hints are: {works}.\n"
        "For each supplied character fragment, generate 1-5 English search queries only; "
        "do not write a final Danbooru tag. Use romanized name order variants when useful.\n"
        "Generate work_queries only when the input or supplied context supports a work/title; "
        "do not invent a franchise.\n"
        "If a name is too vague or not identifiable, omit its character entry and put it in unresolved."
    )


def build_character_query_retry_instruction(
    candidate_names: Iterable[str],
    work_hints: Iterable[str] = (),
    previous_queries: Iterable[str] = (),
) -> str:
    """Make one conservative retry when the first plan has no DB matches.

    The retry is still only a query planner.  It cannot introduce a tag, and
    the local character DB remains the sole source of candidates.  Keeping the
    retry name-only is important: scene words caused the live model to invent
    a phonetic spelling for 東風谷早苗 instead of searching for the canonical
    ``kochiya sanae`` tag.
    """

    base = build_character_query_planner_instruction(candidate_names, work_hints)
    previous = ", ".join(
        str(value).strip() for value in previous_queries if str(value).strip()
    ) or "none"
    return (
        f"{base}\n"
        "A previous plan returned zero active Danbooru character rows. Retry once.\n"
        "Use name-only English/romaji queries; remove scene, pose, expression, and clothing words.\n"
        "Prefer the standard romanization of the supplied Japanese name. Do not invent a phonetic spelling, "
        "do not translate the scene, and do not output a final Danbooru tag.\n"
        f"Previous queries that failed: {previous}."
    )


def _extract_json(raw: str) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        match = re.search(r"\{.*\}", str(raw or ""), flags=re.DOTALL)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(0))
        except (TypeError, ValueError):
            return None
    return payload if isinstance(payload, Mapping) else None


def _parse_intents(
    payload: Mapping[str, Any],
    allowed_names: Sequence[str],
    supplied_work_hints: Sequence[str],
) -> tuple[tuple[CharacterSearchIntent, ...], tuple[str, ...]]:
    allowed = {normalize_search_text(value): value for value in allowed_names}
    intents: list[CharacterSearchIntent] = []
    raw_characters = payload.get("characters")
    if isinstance(raw_characters, list):
        for raw in raw_characters[:4]:
            if not isinstance(raw, Mapping):
                continue
            raw_name = str(raw.get("input_name") or "").strip()
            canonical_name = allowed.get(normalize_search_text(raw_name))
            if not canonical_name:
                continue
            raw_queries = raw.get("queries")
            if not isinstance(raw_queries, list):
                continue
            queries = tuple(
                str(query).strip()[:100]
                for query in raw_queries[:5]
                if isinstance(query, str) and query.strip()
            )
            if not queries:
                continue
            raw_work_queries = raw.get("work_queries")
            if not isinstance(raw_work_queries, list):
                raw_work_queries = []
            work_queries = tuple(
                str(query).strip()[:100]
                for query in raw_work_queries[:5]
                if isinstance(query, str) and query.strip()
            )
            # Keep caller-provided context available even if the model omitted
            # it from one character entry.  The model still controls the
            # translation of Japanese work hints.
            if supplied_work_hints and not work_queries:
                work_queries = tuple(str(value).strip()[:100] for value in supplied_work_hints if str(value).strip())
            intents.append(
                CharacterSearchIntent(
                    input_name=canonical_name,
                    queries=tuple(dict.fromkeys(queries)),
                    work_queries=tuple(dict.fromkeys(work_queries)),
                )
            )
    unresolved = tuple(
        str(value).strip()[:160]
        for value in payload.get("unresolved", [])[:8]
        if isinstance(value, str) and value.strip()
    )
    return tuple(intents), unresolved


def _work_key(value: str) -> str:
    return normalize_tag_name(value).strip("_()")


def _candidate_matches_work(candidate: DanbooruTagCandidate, work_candidates: Sequence[DanbooruTagCandidate], work_queries: Sequence[str]) -> bool:
    candidate_key = normalize_tag_name(candidate.name)
    if any(_work_key(work.name) in candidate_key for work in work_candidates if _work_key(work.name)):
        return True
    return any(_work_key(query) and _work_key(query) in candidate_key for query in work_queries)


def _candidate_base_name(value: str) -> str:
    return normalize_tag_name(value).split("_(", 1)[0]


# This is intentionally a small, deterministic reading aid rather than a
# Japanese name generator.  Its output is only compared with characters that
# already came from the copyright-scoped Danbooru table; it is never emitted
# as a prompt tag.  Keeping the table here also avoids making the resolver
# depend on an optional transliteration package at runtime.
_KANA_ROMAJI: dict[str, str] = {
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "カ": "ka", "キ": "ki", "ク": "ku", "ケ": "ke", "コ": "ko",
    "ガ": "ga", "ギ": "gi", "グ": "gu", "ゲ": "ge", "ゴ": "go",
    "サ": "sa", "シ": "shi", "ス": "su", "セ": "se", "ソ": "so",
    "ザ": "za", "ジ": "ji", "ズ": "zu", "ゼ": "ze", "ゾ": "zo",
    "タ": "ta", "チ": "chi", "ツ": "tsu", "テ": "te", "ト": "to",
    "ダ": "da", "ヂ": "ji", "ヅ": "zu", "デ": "de", "ド": "do",
    "ナ": "na", "ニ": "ni", "ヌ": "nu", "ネ": "ne", "ノ": "no",
    "ハ": "ha", "ヒ": "hi", "フ": "fu", "ヘ": "he", "ホ": "ho",
    "バ": "ba", "ビ": "bi", "ブ": "bu", "ベ": "be", "ボ": "bo",
    "パ": "pa", "ピ": "pi", "プ": "pu", "ペ": "pe", "ポ": "po",
    "マ": "ma", "ミ": "mi", "ム": "mu", "メ": "me", "モ": "mo",
    "ヤ": "ya", "ユ": "yu", "ヨ": "yo",
    "ラ": "ra", "リ": "ri", "ル": "ru", "レ": "re", "ロ": "ro",
    "ワ": "wa", "ヲ": "o", "ン": "n", "ヷ": "va", "ヺ": "vo",
    "ヴ": "vu", "ヰ": "wi", "ヱ": "we",
}
_KANA_DIGRAPHS: dict[str, str] = {
    "キャ": "kya", "キュ": "kyu", "キョ": "kyo",
    "ギャ": "gya", "ギュ": "gyu", "ギョ": "gyo",
    "シャ": "sha", "シュ": "shu", "ショ": "sho",
    "ジャ": "ja", "ジュ": "ju", "ジョ": "jo",
    "チャ": "cha", "チュ": "chu", "チョ": "cho",
    "ニャ": "nya", "ニュ": "nyu", "ニョ": "nyo",
    "ヒャ": "hya", "ヒュ": "hyu", "ヒョ": "hyo",
    "ビャ": "bya", "ビュ": "byu", "ビョ": "byo",
    "ピャ": "pya", "ピュ": "pyu", "ピョ": "pyo",
    "ミャ": "mya", "ミュ": "myu", "ミョ": "myo",
    "リャ": "rya", "リュ": "ryu", "リョ": "ryo",
    "ファ": "fa", "フィ": "fi", "フェ": "fe", "フォ": "fo",
    "フュ": "fyu", "ウィ": "wi", "ウェ": "we", "ウォ": "wo",
    "ティ": "ti", "ディ": "di", "トゥ": "tu", "ドゥ": "du",
    "ツァ": "tsa", "ツィ": "tsi", "ツェ": "tse", "ツォ": "tso",
    "ヴァ": "va", "ヴィ": "vi", "ヴェ": "ve", "ヴォ": "vo",
    "シェ": "she", "ジェ": "je", "チェ": "che",
}


def deterministic_character_romanization(value: str) -> str:
    """Return a stable kana reading used only for scoped candidate matching."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    converted: list[str] = []
    for char in text:
        codepoint = ord(char)
        if 0x3041 <= codepoint <= 0x3096:
            converted.append(chr(codepoint + 0x60))
        else:
            converted.append(char)
    katakana = "".join(converted)

    result: list[str] = []
    pending_geminate = False
    last_vowel = ""
    index = 0
    while index < len(katakana):
        char = katakana[index]
        if char == "ッ":
            pending_geminate = True
            index += 1
            continue
        if char == "ー":
            if last_vowel:
                result.append(last_vowel)
            index += 1
            continue
        roman = _KANA_DIGRAPHS.get(katakana[index:index + 2])
        consumed = 2 if roman else 1
        if roman is None:
            roman = _KANA_ROMAJI.get(char, char if char.isascii() else "")
        if roman:
            if pending_geminate:
                first_consonant = next((item for item in roman if item not in "aeiou"), "")
                if first_consonant:
                    result.append(first_consonant)
                pending_geminate = False
            result.append(roman)
            vowels = [item for item in reversed(roman) if item in "aeiou"]
            if vowels:
                last_vowel = vowels[0]
        index += consumed
    return re.sub(r"[^a-z0-9]", "", "".join(result))


def _romanization_name_key(value: str) -> str:
    """Normalize common Japanese epenthetic/long-vowel spelling differences."""

    key = re.sub(r"[^a-z0-9]", "", str(value or "").casefold())
    key = re.sub(r"([aeiou])\1+", r"\1", key)
    # Japanese readings often retain a devoiced ``u`` (su/tsu/ku, etc.) while
    # Danbooru's romanized tag omits it: アスター -> asutaa -> asta.
    key = re.sub(r"([bcdfghjklmnpqrstvwxyz])u", r"\1", key)
    return key


def _romanized_candidate_score(input_reading: str, candidate_base: str) -> float:
    raw_input = re.sub(r"[^a-z0-9]", "", input_reading.casefold())
    raw_candidate = re.sub(r"[^a-z0-9]", "", candidate_base.casefold())
    if not raw_input or not raw_candidate:
        return 0.0
    normalized_input = _romanization_name_key(raw_input)
    normalized_candidate = _romanization_name_key(raw_candidate)
    return max(
        difflib.SequenceMatcher(None, raw_input, raw_candidate).ratio(),
        difflib.SequenceMatcher(None, normalized_input, normalized_candidate).ratio(),
    )


def _retrieve_copyright_scoped_character_candidates(
    search_store: DanbooruTagStore,
    *,
    work_queries: Sequence[str],
    work_candidates: Sequence[DanbooruTagCandidate],
) -> tuple[DanbooruTagCandidate, ...]:
    """Find active character rows tied to an already-resolved copyright."""

    scoped_by_id: dict[int, DanbooruTagCandidate] = {}
    search_terms = tuple(
        dict.fromkeys(
            (
                *(str(query).strip() for query in work_queries if str(query).strip()),
                *(candidate.name for candidate in work_candidates),
            )
        )
    )
    # The DB search remains bounded.  It is deliberately performed only after
    # work context is known, rather than fuzzy-matching the complete character
    # table.
    for term in search_terms:
        for candidate in search_store.search(
            term,
            limit=CHARACTER_SEARCH_INTERNAL_LIMIT * 10,
            allowed_categories=(CHARACTER_CATEGORY,),
        ):
            if _candidate_matches_work(candidate, work_candidates, work_queries):
                scoped_by_id[candidate.tag_id] = candidate
    return tuple(
        sorted(
            scoped_by_id.values(),
            key=lambda candidate: (-candidate.post_count, candidate.name),
        )
    )


def _choose_romanized_fallback_candidate(
    candidates: Sequence[DanbooruTagCandidate],
    *,
    input_name: str,
) -> tuple[DanbooruTagCandidate | None, tuple[dict[str, object], ...]]:
    reading = deterministic_character_romanization(input_name)
    if not reading:
        return None, ()
    scored = [
        (
            _romanized_candidate_score(reading, _candidate_base_name(candidate.name)),
            candidate,
        )
        for candidate in candidates
    ]
    scored.sort(key=lambda item: (-item[0], -item[1].post_count, item[1].name))
    trace = tuple(
        {
            "tag_id": candidate.tag_id,
            "name": candidate.name,
            "base_name": _candidate_base_name(candidate.name),
            "score": round(score, 4),
        }
        for score, candidate in scored[:CHARACTER_CANDIDATE_LIMIT]
    )
    if not scored or scored[0][0] < CHARACTER_FALLBACK_MIN_SCORE:
        return None, trace
    if len(scored) > 1 and scored[0][0] - scored[1][0] < CHARACTER_FALLBACK_MARGIN:
        return None, trace
    return scored[0][1], trace


def _select_candidate(
    candidates: Sequence[DanbooruTagCandidate],
    *,
    queries: Sequence[str],
    work_queries: Sequence[str],
    work_candidates: Sequence[DanbooruTagCandidate],
) -> tuple[DanbooruTagCandidate | None, float, str]:
    if not candidates:
        return None, 0.0, "no_character_candidates"

    if work_queries:
        work_matches = [
            candidate
            for candidate in candidates
            if _candidate_matches_work(candidate, work_candidates, work_queries)
        ]
        if len(work_matches) == 1:
            return work_matches[0], 0.98, ""
        if not work_matches:
            return None, 0.0, "work_not_matched"
        return None, 0.0, "ambiguous_with_work"

    if len(candidates) == 1:
        return candidates[0], 0.92, ""

    exact_base_matches: list[DanbooruTagCandidate] = []
    query_keys = {_work_key(query) for query in queries if _work_key(query)}
    for candidate in candidates:
        base = _candidate_base_name(candidate.name)
        if base in query_keys:
            exact_base_matches.append(candidate)
    if len(exact_base_matches) == 1:
        return exact_base_matches[0], 0.93, ""
    return None, 0.0, "ambiguous_without_work"


def resolve_danbooru_character_tags(
    text: str,
    *,
    work_hints: Sequence[str] = (),
    manual_resolutions: Sequence[CharacterTagResolution] = (),
    selected_character_name: str | None = None,
    store: DanbooruTagStore | None = None,
    llm_json: LlmJsonCallback | None = None,
) -> DanbooruCharacterResolution:
    search_store = store or DanbooruTagStore()
    snapshot = search_store.snapshot_version
    extracted_context = extract_work_character_context(text)
    extracted_work_name = extracted_context.work_name
    extracted_character_name = extracted_context.character_name
    work_alias_queries = lookup_copyright_query_hints(extracted_work_name)
    effective_work_hints = tuple(
        dict.fromkeys(
            (
                *tuple(str(value).strip() for value in work_hints if str(value).strip()),
                *work_alias_queries,
                *((extracted_work_name,) if extracted_work_name else ()),
            )
        )
    )
    span_attempts = _build_character_span_attempts(extracted_context)
    usable_span_attempts = tuple(
        attempt
        for attempt in span_attempts
        if not _has_character_tail_boundary(attempt[0])
    )
    preferred_span, preferred_remaining = (
        usable_span_attempts[0]
        if usable_span_attempts
        else ("", "")
    )
    character_input = preferred_span if extracted_work_name else str(text or "")
    if extracted_work_name:
        names = (
            (preferred_span,)
            if preferred_span and preferred_span not in _CHARACTER_STOPWORDS
            else ()
        )
    else:
        names = extract_possible_character_names(character_input)
    candidate_attempts: list[dict[str, object]] = []
    if extracted_work_name:
        for candidate, remaining in span_attempts:
            if candidate == preferred_span and candidate_attempts:
                continue
            if _has_character_tail_boundary(candidate):
                candidate_attempts.append(
                    {
                        "span": candidate,
                        "remaining_prompt_text": remaining,
                        "status": "syntax_action_tail",
                    }
                )
        if preferred_span:
            candidate_attempts.append(
                {
                    "span": preferred_span,
                    "remaining_prompt_text": preferred_remaining,
                    "status": "pending",
                }
            )
    trace_context = {
        "extracted_work_name": extracted_work_name,
        "normalized_work_name": normalize_search_text(extracted_work_name),
        "extracted_character_name": names[0] if names else extracted_character_name,
        "raw_character_span": extracted_context.raw_character_span,
        "remaining_prompt_text": preferred_remaining,
        "character_candidate_attempts": tuple(candidate_attempts),
        "work_alias_hit": bool(work_alias_queries),
    }
    if not extracted_work_name and lookup_copyright_query_hints(str(text or "")):
        return DanbooruCharacterResolution(
            snapshot_version=snapshot,
            **trace_context,
            unresolved_reason="work_only_input",
            abstained=True,
        )
    if manual_resolutions:
        return DanbooruCharacterResolution(
            snapshot_version=snapshot,
            **trace_context,
            unresolved_reason="manual_dictionary_priority",
            abstained=True,
        )
    if selected_character_name:
        return DanbooruCharacterResolution(
            snapshot_version=snapshot,
            **trace_context,
            unresolved_reason="selected_character_preset_priority",
            abstained=True,
        )
    if not search_store.available:
        return DanbooruCharacterResolution(
            snapshot_version=snapshot,
            **trace_context,
            input_character_name=names[0] if names else "",
            normalized_character_name=normalize_search_text(names[0]) if names else "",
            unresolved_reason="character_db_unavailable",
            abstained=True,
        )
    if not names:
        return DanbooruCharacterResolution(snapshot_version=snapshot, **trace_context)
    if llm_json is None:
        return DanbooruCharacterResolution(
            snapshot_version=snapshot,
            **trace_context,
            input_character_name=names[0],
            normalized_character_name=normalize_search_text(names[0]),
            unresolved_reason="llm_json_not_configured",
            abstained=True,
        )

    try:
        planner_raw = llm_json(
            build_character_query_planner_instruction(names, effective_work_hints),
            json.dumps(
                {
                    "input": character_input,
                    "candidate_names": list(names),
                    "work_hints": list(effective_work_hints),
                    "extracted_work_name": extracted_work_name,
                },
                ensure_ascii=False,
            ),
            QUERY_PLANNER_RESPONSE_FORMAT,
            384,
        )
    except Exception as exc:  # noqa: BLE001 - character lookup must not break generation
        logger.warning("Danbooru character query planner failed: %s", exc)
        return DanbooruCharacterResolution(
            snapshot_version=snapshot,
            **trace_context,
            input_character_name=names[0],
            normalized_character_name=normalize_search_text(names[0]),
            unresolved_reason="query_planner_failed",
            error=f"{type(exc).__name__}",
            abstained=True,
        )
    payload = _extract_json(planner_raw)
    if payload is None:
        return DanbooruCharacterResolution(
            snapshot_version=snapshot,
            **trace_context,
            input_character_name=names[0],
            normalized_character_name=normalize_search_text(names[0]),
            unresolved_reason="query_planner_invalid_json",
            abstained=True,
        )
    intents, unresolved = _parse_intents(payload, names, effective_work_hints)
    if not intents:
        return DanbooruCharacterResolution(
            snapshot_version=snapshot,
            **trace_context,
            input_character_name=names[0],
            normalized_character_name=normalize_search_text(names[0]),
            unresolved_reason=";".join(unresolved) or "no_search_intent",
            abstained=True,
        )

    # MVP handles the first unresolved proper name conservatively.  Multiple
    # character support can be added later without mixing categories.
    intent = intents[0]
    context_aliases = list(extract_character_context_aliases(text, intent.input_name))
    if extracted_work_name:
        context_aliases.extend(lookup_copyright_aliases(extracted_work_name))
    context_aliases = tuple(dict.fromkeys(context_aliases))
    def _retrieve_candidates(
        search_intent: CharacterSearchIntent,
    ) -> tuple[tuple[DanbooruTagCandidate, ...], list[DanbooruTagCandidate]]:
        candidate_by_id: dict[int, DanbooruTagCandidate] = {}
        for query in search_intent.queries:
            for candidate in search_store.search(
                query,
                limit=CHARACTER_SEARCH_INTERNAL_LIMIT,
                allowed_categories=(CHARACTER_CATEGORY,),
            ):
                previous = candidate_by_id.get(candidate.tag_id)
                if previous is None or candidate.retrieval_score > previous.retrieval_score:
                    candidate_by_id[candidate.tag_id] = candidate
            for work_query in search_intent.work_queries:
                combined = f"{query} {work_query}".strip()
                for candidate in search_store.search(
                    combined,
                    limit=CHARACTER_SEARCH_INTERNAL_LIMIT,
                    allowed_categories=(CHARACTER_CATEGORY,),
                ):
                    previous = candidate_by_id.get(candidate.tag_id)
                    if previous is None or candidate.retrieval_score > previous.retrieval_score:
                        candidate_by_id[candidate.tag_id] = candidate
        resolved_candidates = tuple(
            sorted(
                candidate_by_id.values(),
                key=lambda candidate: (-candidate.retrieval_score, -candidate.post_count, candidate.name),
            )[:CHARACTER_CANDIDATE_LIMIT]
        )
        resolved_work_candidates: list[DanbooruTagCandidate] = []
        for work_query in search_intent.work_queries:
            resolved_work_candidates.extend(
                search_store.search(
                    work_query,
                    limit=10,
                    allowed_categories=(COPYRIGHT_CATEGORY,),
                )
            )
        return resolved_candidates, resolved_work_candidates

    generated_queries = list(intent.queries)
    generated_work_queries = list(intent.work_queries)
    llm_character_queries = list(intent.queries)
    candidates, work_candidates = _retrieve_candidates(intent)
    if intent.work_queries:
        # A character query can return similarly named characters from other
        # works.  With explicit work context, expose only candidates whose
        # canonical tag is tied to a matching copyright row/query.
        candidates = tuple(
            candidate
            for candidate in candidates
            if _candidate_matches_work(candidate, work_candidates, intent.work_queries)
        )

    attempt_record = next(
        (
            attempt
            for attempt in candidate_attempts
            if attempt.get("span") == intent.input_name
            and attempt.get("status") == "pending"
        ),
        None,
    )
    if attempt_record is None:
        attempt_record = {"span": intent.input_name}
        candidate_attempts.append(attempt_record)
    attempt_record.update(
        {
            "remaining_prompt_text": preferred_remaining,
            "queries": list(intent.queries),
            "candidate_character_tags": [candidate.to_trace_dict() for candidate in candidates],
            "status": "searched",
        }
    )

    # Keep a tiny, separately editable query-hint layer for known Japanese
    # names whose standard reading is routinely mistransliterated by a small
    # local model.  Hints only add DB search queries; they never bypass the
    # candidate validation or the first-use confirmation flow.
    if not candidates:
        hint_queries = lookup_character_query_hints(intent.input_name)
        if hint_queries:
            intent = replace(
                intent,
                queries=tuple(dict.fromkeys((*intent.queries, *hint_queries))),
            )
            generated_queries.extend(hint_queries)
            candidates, work_candidates = _retrieve_candidates(intent)

    # A local DB miss must never fall through to the free-form prompt LLM.
    # Give the planner one name-only retry because small local models can
    # mistransliterate a Japanese name when scene words are present.  The
    # retry still yields queries only; every final candidate is re-read from
    # and validated against the local character table below.
    if not candidates:
        try:
            retry_raw = llm_json(
                build_character_query_retry_instruction(
                    names,
                    effective_work_hints,
                    previous_queries=intent.queries,
                ),
                json.dumps(
                    {
                        "input": character_input,
                        "candidate_names": list(names),
                        "work_hints": list(effective_work_hints),
                        "previous_queries": list(intent.queries),
                        "retry": True,
                    },
                    ensure_ascii=False,
                ),
                QUERY_PLANNER_RESPONSE_FORMAT,
                384,
            )
            retry_payload = _extract_json(retry_raw)
            if retry_payload is not None:
                retry_intents, _retry_unresolved = _parse_intents(
                    retry_payload,
                    names,
                    effective_work_hints,
                )
                if retry_intents:
                    retry_intent = retry_intents[0]
                    if normalize_search_text(retry_intent.input_name) == normalize_search_text(intent.input_name):
                        intent = retry_intent
                        context_aliases = list(extract_character_context_aliases(text, intent.input_name))
                        if extracted_work_name:
                            context_aliases.extend(lookup_copyright_aliases(extracted_work_name))
                        context_aliases = tuple(dict.fromkeys(context_aliases))
                        generated_queries.extend(intent.queries)
                        llm_character_queries.extend(intent.queries)
                        generated_work_queries.extend(intent.work_queries)
                        candidates, work_candidates = _retrieve_candidates(intent)
                        if intent.work_queries:
                            candidates = tuple(
                                candidate
                                for candidate in candidates
                                if _candidate_matches_work(
                                    candidate,
                                    work_candidates,
                                    intent.work_queries,
                                )
                            )
        except Exception as exc:  # noqa: BLE001 - safe fallback is abstention
            logger.warning("Danbooru character query retry failed: %s", exc)

    # A miss from the LLM's English spelling is not enough reason to give up
    # when the work is already known.  Scope the search to active character
    # rows belonging to that work, then compare their base names with a
    # deterministic kana reading.  This remains a suggestion only: the normal
    # confirmation UI still gates every unregistered character.
    deterministic_romanization = ""
    copyright_scoped_candidates: tuple[DanbooruTagCandidate, ...] = ()
    fuzzy_candidate_scores: tuple[dict[str, object], ...] = ()
    fallback_used = False
    preview_selected, _preview_confidence, _preview_reason = _select_candidate(
        candidates,
        queries=intent.queries,
        work_queries=intent.work_queries,
        work_candidates=work_candidates,
    )
    if intent.work_queries and preview_selected is None:
        fallback_used = True
        deterministic_romanization = deterministic_character_romanization(intent.input_name)
        fallback_work_queries = tuple(
            dict.fromkeys(
                (
                    *intent.work_queries,
                    *effective_work_hints,
                )
            )
        )
        if not work_candidates:
            fallback_work_candidates: list[DanbooruTagCandidate] = []
            for work_query in fallback_work_queries:
                fallback_work_candidates.extend(
                    search_store.search(
                        work_query,
                        limit=10,
                        allowed_categories=(COPYRIGHT_CATEGORY,),
                    )
                )
            work_candidates = list(
                {
                    candidate.tag_id: candidate for candidate in fallback_work_candidates
                }.values()
            )
        copyright_scoped_candidates = _retrieve_copyright_scoped_character_candidates(
            search_store,
            work_queries=fallback_work_queries,
            work_candidates=work_candidates,
        )
        fallback_candidate, fuzzy_candidate_scores = _choose_romanized_fallback_candidate(
            copyright_scoped_candidates,
            input_name=intent.input_name,
        )
        if fallback_candidate is not None:
            candidates = (fallback_candidate,)
            # Keep the candidate's query visible in trace/debug even though
            # it was found from work scope rather than the LLM spelling.
            attempt_record["fallback"] = {
                "input_name": intent.input_name,
                "deterministic_romanization": deterministic_romanization,
                "copyright_scoped_candidate_count": len(copyright_scoped_candidates),
                "fuzzy_candidate_scores": list(fuzzy_candidate_scores),
                "selected_candidate": fallback_candidate.name,
                "status": "fallback_candidate",
            }
        else:
            attempt_record["fallback"] = {
                "input_name": intent.input_name,
                "deterministic_romanization": deterministic_romanization,
                "copyright_scoped_candidate_count": len(copyright_scoped_candidates),
                "fuzzy_candidate_scores": list(fuzzy_candidate_scores),
                "selected_candidate": "",
                "status": "fallback_abstained",
            }

    selected, confidence, unresolved_reason = _select_candidate(
        candidates,
        queries=intent.queries,
        work_queries=intent.work_queries,
        work_candidates=work_candidates,
    )
    candidate_ids = {candidate.tag_id for candidate in candidates}
    if selected is not None:
        selected_rows = search_store.validate_candidate_ids(
            (selected.tag_id,),
            candidate_ids=candidate_ids,
            allowed_categories=(CHARACTER_CATEGORY,),
        )
        if len(selected_rows) != 1:
            selected = None
            confidence = 0.0
            unresolved_reason = "selected_candidate_failed_validation"
    if selected is not None and confidence < CHARACTER_ACCEPT_CONFIDENCE:
        selected = None
        confidence = 0.0
        unresolved_reason = "confidence_below_threshold"
    attempt_record.update(
        {
            "candidate_match_strength": round(confidence, 4),
            "selected_character_tag": selected.name if selected is not None else "",
            "unresolved_reason": unresolved_reason,
            "status": "matched" if selected is not None else "no_match",
        }
    )
    trace_context["character_candidate_attempts"] = tuple(candidate_attempts)
    selected_copyright_tag = ""
    if selected is not None:
        for work_candidate in work_candidates:
            work_key = _work_key(work_candidate.name)
            if work_key and work_key in normalize_tag_name(selected.name):
                selected_copyright_tag = work_candidate.name
                break
    if not selected_copyright_tag and work_candidates:
        exact_work_candidates = [
            work_candidate
            for work_candidate in work_candidates
            if normalize_tag_name(work_candidate.name)
            in {_work_key(query) for query in intent.work_queries if _work_key(query)}
        ]
        if len(exact_work_candidates) == 1:
            selected_copyright_tag = exact_work_candidates[0].name
    # A Danbooru candidate is intentionally never confirmed here.  The
    # selected value is only a UI suggestion; Positive may use it only after
    # the registry save performed by the confirmation view.
    confirmation_required = bool(candidates)
    if selected is not None:
        unresolved_reason = "confirmation_required"
        attempt_record["unresolved_reason"] = unresolved_reason
    suggested = selected or (candidates[0] if candidates else None)
    rejected_ids = tuple(
        candidate.tag_id
        for candidate in candidates
        if selected is not None and candidate.tag_id != selected.tag_id
    )
    return DanbooruCharacterResolution(
        snapshot_version=snapshot,
        extracted_work_name=extracted_work_name,
        normalized_work_name=normalize_search_text(extracted_work_name),
        extracted_character_name=names[0] if names else extracted_character_name,
        raw_character_span=extracted_context.raw_character_span,
        remaining_prompt_text=preferred_remaining,
        character_candidate_attempts=tuple(candidate_attempts),
        work_alias_hit=bool(work_alias_queries),
        copyright_candidates=tuple(dict.fromkeys(work_candidates)),
        input_character_name=intent.input_name,
        normalized_character_name=normalize_search_text(intent.input_name),
        generated_queries=tuple(dict.fromkeys(generated_queries)),
        llm_character_queries=tuple(dict.fromkeys(llm_character_queries)),
        generated_work_queries=tuple(dict.fromkeys(generated_work_queries)),
        context_aliases=context_aliases,
        candidate_character_tags=candidates,
        selected_tag_id=suggested.tag_id if suggested is not None else None,
        selected_character_tag=suggested.name if suggested is not None else "",
        selected_prompt_tag=suggested.prompt_tag if suggested is not None else "",
        selected_candidate=suggested.name if suggested is not None else "",
        selected_copyright_tag=selected_copyright_tag,
        candidate_match_strength=confidence,
        deterministic_romanization=deterministic_romanization,
        copyright_scoped_candidate_count=len(copyright_scoped_candidates),
        fuzzy_candidate_scores=fuzzy_candidate_scores,
        fallback_used=fallback_used,
        confirmation_required=confirmation_required,
        confidence=confidence,
        unresolved_reason=unresolved_reason,
        rejected_tag_ids=rejected_ids,
        abstained=selected is None,
    )


def build_danbooru_character_instruction(
    resolution: DanbooruCharacterResolution | None,
) -> str:
    if resolution is None or not resolution.has_selected_tag:
        return ""
    return "\n".join(
        (
            "## Danbooru character tag (authoritative)",
            "The following canonical character tag was selected from the active local Danbooru character DB.",
            "Keep it exactly in Positive; do not translate, abbreviate, replace, or invent another character tag.",
            "Do not output the Japanese character name as a separate tag. The associated copyright tag is search context only and need not be added.",
            f"- selected character tag: {resolution.selected_prompt_tag}",
            f"- selected tag ID: {resolution.selected_tag_id}",
        )
    )


def apply_danbooru_character_resolution(
    positive: str,
    resolution: DanbooruCharacterResolution | None,
) -> str:
    """Remove the resolved Japanese identity and insert the canonical tag."""

    if resolution is None or not resolution.has_selected_tag:
        return str(positive or "").strip()
    input_name = normalize_search_text(resolution.input_character_name)
    candidate_keys = {
        normalize_search_text(candidate.prompt_tag)
        for candidate in resolution.candidate_character_tags
    }
    kept: list[str] = []
    seen: set[str] = set()
    for raw_token in str(positive or "").split(","):
        token = raw_token.strip()
        if not token:
            continue
        key = normalize_search_text(token)
        if input_name and (key == input_name or input_name in key):
            continue
        # The LM is not allowed to turn a rejected DB candidate back into a
        # character tag in the final Positive.  Unrelated general tags remain
        # untouched; only the resolver's own candidate set is filtered here.
        if key in candidate_keys and key != normalize_search_text(resolution.selected_prompt_tag):
            continue
        selected_base_key = normalize_search_text(
            resolution.selected_prompt_tag.split("(", 1)[0].strip()
        )
        if selected_base_key and key.startswith(f"{selected_base_key} ("):
            continue
        if key in seen:
            continue
        kept.append(token)
        seen.add(key)

    selected_key = normalize_search_text(resolution.selected_prompt_tag)
    if selected_key in seen:
        kept = [token for token in kept if normalize_search_text(token) != selected_key]
        seen.discard(selected_key)
    insert_at = 0
    if kept and re.fullmatch(r"\d+(?:girl|girls|boy|boys|other)", kept[0].casefold()):
        insert_at = 1
    kept[insert_at:insert_at] = [resolution.selected_prompt_tag]
    return ", ".join(kept)
