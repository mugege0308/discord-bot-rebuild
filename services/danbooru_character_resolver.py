"""Conservative Japanese-character-name resolution through the local Danbooru DB.

The hand-maintained ``character_tag_dictionary`` remains the first authority.
This module is deliberately narrower than the generic appearance resolver:
the only positive candidates it can return are active ``category=character``
rows from the local SQLite snapshot.  The LLM may translate/search-plan a
fragment, but it never supplies a final character tag.
"""

from __future__ import annotations

import json
import logging
import re
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

CHARACTER_RESOLVER_VERSION = "0.1.0"
CHARACTER_SEARCH_INTERNAL_LIMIT = 50
CHARACTER_CANDIDATE_LIMIT = 8
CHARACTER_ACCEPT_CONFIDENCE = 0.90
CHARACTER_CATEGORY = "character"
COPYRIGHT_CATEGORY = "copyright"
CHARACTER_QUERY_HINTS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "danbooru" / "character_query_hints.json"
)
_CHARACTER_QUERY_HINTS: dict[str, tuple[str, ...]] | None = None

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
class DanbooruCharacterResolution:
    resolver_version: str = CHARACTER_RESOLVER_VERSION
    snapshot_version: str = "unknown"
    input_character_name: str = ""
    normalized_character_name: str = ""
    registry_hit: bool = False
    registry_character_id: int | None = None
    generated_queries: tuple[str, ...] = ()
    generated_work_queries: tuple[str, ...] = ()
    context_aliases: tuple[str, ...] = ()
    candidate_character_tags: tuple[DanbooruTagCandidate, ...] = ()
    selected_tag_id: int | None = None
    selected_character_tag: str = ""
    selected_prompt_tag: str = ""
    selected_candidate: str = ""
    selected_copyright_tag: str = ""
    candidate_match_strength: float = 0.0
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
            "registry_hit": self.registry_hit,
            "registry_character_id": self.registry_character_id,
            "generated_queries": list(self.generated_queries),
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


def extract_possible_character_names(text: str) -> tuple[str, ...]:
    """Extract only name-shaped Japanese spans for the planner gate.

    This is not a resolver.  It only decides whether an additional character
    search is worth attempting; the DB and canonical-row validation remain the
    actual authority.
    """

    found: list[str] = []
    for pattern in (_CHARACTER_NAME_RE, _KATAKANA_CHARACTER_NAME_RE):
        for match in pattern.finditer(str(text or "")):
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
    if not extract_possible_character_names(text):
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
    names = extract_possible_character_names(text)
    snapshot = search_store.snapshot_version
    if manual_resolutions:
        return DanbooruCharacterResolution(
            snapshot_version=snapshot,
            unresolved_reason="manual_dictionary_priority",
            abstained=True,
        )
    if selected_character_name:
        return DanbooruCharacterResolution(
            snapshot_version=snapshot,
            unresolved_reason="selected_character_preset_priority",
            abstained=True,
        )
    if not search_store.available:
        return DanbooruCharacterResolution(
            snapshot_version=snapshot,
            input_character_name=names[0] if names else "",
            normalized_character_name=normalize_search_text(names[0]) if names else "",
            unresolved_reason="character_db_unavailable",
            abstained=True,
        )
    if not names:
        return DanbooruCharacterResolution(snapshot_version=snapshot)
    if llm_json is None:
        return DanbooruCharacterResolution(
            snapshot_version=snapshot,
            input_character_name=names[0],
            normalized_character_name=normalize_search_text(names[0]),
            unresolved_reason="llm_json_not_configured",
            abstained=True,
        )

    try:
        planner_raw = llm_json(
            build_character_query_planner_instruction(names, work_hints),
            json.dumps(
                {"input": str(text or ""), "candidate_names": list(names), "work_hints": list(work_hints)},
                ensure_ascii=False,
            ),
            QUERY_PLANNER_RESPONSE_FORMAT,
            384,
        )
    except Exception as exc:  # noqa: BLE001 - character lookup must not break generation
        logger.warning("Danbooru character query planner failed: %s", exc)
        return DanbooruCharacterResolution(
            snapshot_version=snapshot,
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
            input_character_name=names[0],
            normalized_character_name=normalize_search_text(names[0]),
            unresolved_reason="query_planner_invalid_json",
            abstained=True,
        )
    intents, unresolved = _parse_intents(payload, names, work_hints)
    if not intents:
        return DanbooruCharacterResolution(
            snapshot_version=snapshot,
            input_character_name=names[0],
            normalized_character_name=normalize_search_text(names[0]),
            unresolved_reason=";".join(unresolved) or "no_search_intent",
            abstained=True,
        )

    # MVP handles the first unresolved proper name conservatively.  Multiple
    # character support can be added later without mixing categories.
    intent = intents[0]
    context_aliases = extract_character_context_aliases(text, intent.input_name)
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
    candidates, work_candidates = _retrieve_candidates(intent)

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
                    work_hints,
                    previous_queries=intent.queries,
                ),
                json.dumps(
                    {
                        "input": str(text or ""),
                        "candidate_names": list(names),
                        "work_hints": list(work_hints),
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
                    work_hints,
                )
                if retry_intents:
                    retry_intent = retry_intents[0]
                    if normalize_search_text(retry_intent.input_name) == normalize_search_text(intent.input_name):
                        intent = retry_intent
                        context_aliases = extract_character_context_aliases(text, intent.input_name)
                        generated_queries.extend(intent.queries)
                        generated_work_queries.extend(intent.work_queries)
                        candidates, work_candidates = _retrieve_candidates(intent)
        except Exception as exc:  # noqa: BLE001 - safe fallback is abstention
            logger.warning("Danbooru character query retry failed: %s", exc)

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
    suggested = selected or (candidates[0] if candidates else None)
    rejected_ids = tuple(
        candidate.tag_id
        for candidate in candidates
        if selected is not None and candidate.tag_id != selected.tag_id
    )
    return DanbooruCharacterResolution(
        snapshot_version=snapshot,
        input_character_name=intent.input_name,
        normalized_character_name=normalize_search_text(intent.input_name),
        generated_queries=tuple(dict.fromkeys(generated_queries)),
        generated_work_queries=tuple(dict.fromkeys(generated_work_queries)),
        context_aliases=context_aliases,
        candidate_character_tags=candidates,
        selected_tag_id=suggested.tag_id if suggested is not None else None,
        selected_character_tag=suggested.name if suggested is not None else "",
        selected_prompt_tag=suggested.prompt_tag if suggested is not None else "",
        selected_candidate=suggested.name if suggested is not None else "",
        selected_copyright_tag=selected_copyright_tag,
        candidate_match_strength=confidence,
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
