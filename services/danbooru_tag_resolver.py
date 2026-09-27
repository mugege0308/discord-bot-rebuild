"""LLM-assisted query planning and candidate selection for Danbooru tags.

The model may produce English search queries and choose from returned IDs, but
it is never allowed to invent a final tag.  The final selection is reloaded
from SQLite and validated here before it can reach a prompt.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from services.danbooru_tag_store import (
    DEFAULT_DANBOORU_DB_PATH,
    DanbooruTagCandidate,
    DanbooruTagStore,
    normalize_search_text,
    normalize_tag_name,
)


logger = logging.getLogger(__name__)
GENERIC_RESOLVER_VERSION = "0.1.0"
GENERIC_SEARCH_INTERNAL_LIMIT = 50
GENERIC_RERANK_LIMIT = 8
GENERIC_ACCEPT_CONFIDENCE = 0.80

# This gate keeps the new exploratory path away from ordinary composition-only
# requests and preserves the existing one-call behavior for old tests/flows.
GENERIC_TRIGGER_CUES = (
    "服",
    "衣装",
    "上着",
    "袖",
    "スカート",
    "ズボン",
    "パーカー",
    "フード",
    "コート",
    "シャツ",
    "制服",
    "髪",
    "髪型",
    "前髪",
    "後髪",
    "目",
    "瞳",
    "まつげ",
    "眼鏡",
    "メガネ",
    "靴",
    "帽子",
    "リボン",
    "アクセサリー",
    "小物",
    "背景",
)
GENERIC_NO_REQUEST_CUES = (
    "指定なし",
    "指定しない",
    "指定はない",
    "特にない",
    "おまかせ",
    "お任せ",
)
_CORE_CATEGORY_CONFLICT_NAMES = {
    "framing": ("portrait", "upper_body", "cowboy_shot", "full_body", "medium_shot"),
    "camera_elevation": ("from_above", "from_below", "high_angle", "low_angle"),
    "camera_horizontal": (
        "from_side",
        "from_behind",
        "three-quarter_view",
        "straight-on",
        "side_view",
        "back_view",
        "front_view",
    ),
    "gaze_direction": (
        "looking_at_viewer",
        "looking_away",
        "looking_back",
        "looking_to_the_side",
    ),
    "basic_pose": (
        "standing",
        "sitting",
        "kneeling",
        "squatting",
        "lying",
        "walking",
        "running",
        "jumping",
    ),
    "expression": ("smile", "grin", "serious", "surprised", "blush", "closed_eyes", "open_mouth"),
    "dynamic_composition": ("dynamic_pose", "dutch_angle", "foreshortening"),
}

QUERY_PLANNER_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "danbooru_query_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "fragments": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "minLength": 1},
                            "queries": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 5,
                                "items": {"type": "string", "minLength": 1},
                            },
                            "expected_category": {"type": "string", "enum": ["general"]},
                            "must_not_infer": {
                                "type": "array",
                                "maxItems": 8,
                                "items": {"type": "string"},
                            },
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": [
                            "source",
                            "queries",
                            "expected_category",
                            "must_not_infer",
                            "confidence",
                        ],
                        "additionalProperties": False,
                    },
                },
                "unresolved": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {"type": "string"},
                },
            },
            "required": ["fragments", "unresolved"],
            "additionalProperties": False,
        },
    },
}

RERANK_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "danbooru_candidate_selection",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "maxItems": 64,
                    "items": {
                        "type": "object",
                        "properties": {
                            "tag_id": {"type": "integer"},
                            "decision": {"type": "string", "enum": ["accept", "maybe", "reject"]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "reason": {"type": "string", "maxLength": 240},
                        },
                        "required": ["tag_id", "decision", "confidence", "reason"],
                        "additionalProperties": False,
                    },
                },
                "abstain": {"type": "boolean"},
            },
            "required": ["decisions", "abstain"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class DanbooruSearchIntent:
    source: str
    queries: tuple[str, ...]
    expected_category: str = "general"
    must_not_infer: tuple[str, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True)
class DanbooruCandidateDecision:
    tag_id: int
    decision: str
    confidence: float
    reason: str = ""


@dataclass(frozen=True)
class DanbooruTagResolution:
    resolver_version: str = GENERIC_RESOLVER_VERSION
    snapshot_version: str = "unknown"
    fragments: tuple[str, ...] = ()
    intents: tuple[DanbooruSearchIntent, ...] = ()
    queries: tuple[str, ...] = ()
    candidates: tuple[DanbooruTagCandidate, ...] = ()
    decisions: tuple[DanbooruCandidateDecision, ...] = ()
    selected_tag_ids: tuple[int, ...] = ()
    selected_prompt_tags: tuple[str, ...] = ()
    rejected_tag_ids: tuple[int, ...] = ()
    maybe_tag_ids: tuple[int, ...] = ()
    unresolved_fragments: tuple[str, ...] = ()
    abstained: bool = False
    error: str = ""

    @property
    def has_selected_tags(self) -> bool:
        return bool(self.selected_tag_ids)

    def candidates_by_fragment(self) -> dict[str, list[dict[str, object]]]:
        grouped: dict[str, list[dict[str, object]]] = {}
        for candidate in self.candidates:
            grouped.setdefault(candidate.query, []).append(candidate.to_trace_dict())
        return grouped

    def to_trace_dict(self) -> dict[str, object]:
        return {
            "resolver_version": self.resolver_version,
            "snapshot_version": self.snapshot_version,
            "fragments": list(self.fragments),
            "intents": [
                {
                    "source": intent.source,
                    "queries": list(intent.queries),
                    "expected_category": intent.expected_category,
                    "must_not_infer": list(intent.must_not_infer),
                    "confidence": intent.confidence,
                }
                for intent in self.intents
            ],
            "queries": list(self.queries),
            "candidates": [candidate.to_trace_dict() for candidate in self.candidates],
            "decisions": [decision.__dict__ for decision in self.decisions],
            "selected_tag_ids": list(self.selected_tag_ids),
            "selected_prompt_tags": list(self.selected_prompt_tags),
            "rejected_tag_ids": list(self.rejected_tag_ids),
            "maybe_tag_ids": list(self.maybe_tag_ids),
            "unresolved_fragments": list(self.unresolved_fragments),
            "abstained": self.abstained,
            "error": self.error,
        }


LlmJsonCallback = Callable[[str, str, Mapping[str, Any], int], str]


def should_search_generic_tags(text: str, *, db_path=DEFAULT_DANBOORU_DB_PATH) -> bool:
    clean_text = str(text or "").strip()
    if not clean_text or any(cue in clean_text for cue in GENERIC_NO_REQUEST_CUES):
        return False
    if not any(cue in clean_text for cue in GENERIC_TRIGGER_CUES):
        return False
    return DanbooruTagStore(db_path).available


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


def _parse_intents(payload: Mapping[str, Any], source_text: str) -> tuple[tuple[DanbooruSearchIntent, ...], tuple[str, ...]]:
    raw_fragments = payload.get("fragments")
    if not isinstance(raw_fragments, list):
        return (), (source_text[:120],) if source_text.strip() else ()
    intents: list[DanbooruSearchIntent] = []
    for raw in raw_fragments[:8]:
        if not isinstance(raw, Mapping):
            continue
        source = str(raw.get("source") or "").strip()[:160]
        queries_raw = raw.get("queries")
        if not source or not isinstance(queries_raw, list):
            continue
        queries = tuple(
            query.strip()[:100]
            for query in queries_raw[:5]
            if isinstance(query, str) and query.strip()
        )
        if not queries:
            continue
        category = str(raw.get("expected_category") or "general").strip().casefold()
        if category != "general":
            continue
        raw_confidence = raw.get("confidence", 0.0)
        try:
            confidence = max(0.0, min(1.0, float(raw_confidence)))
        except (TypeError, ValueError):
            confidence = 0.0
        must_not_infer = tuple(
            value.strip()[:100]
            for value in raw.get("must_not_infer", [])
            if isinstance(value, str) and value.strip()
        )
        intents.append(
            DanbooruSearchIntent(
                source=source,
                queries=queries,
                expected_category="general",
                must_not_infer=must_not_infer,
                confidence=confidence,
            )
        )
    unresolved = tuple(
        value.strip()[:160]
        for value in payload.get("unresolved", [])
        if isinstance(value, str) and value.strip()
    )
    return tuple(intents), unresolved


def _locked_names(core_entries: Iterable[object]) -> set[str]:
    names: set[str] = set()
    for entry in core_entries:
        category = str(getattr(entry, "category", ""))
        names.update(normalize_tag_name(value) for value in _CORE_CATEGORY_CONFLICT_NAMES.get(category, ()))
        for attribute in ("danbooru_tag", "prompt_tag", "output_aliases"):
            value = getattr(entry, attribute, "")
            values = value if isinstance(value, (tuple, list, set)) else (value,)
            for item in values:
                if item:
                    names.add(normalize_tag_name(str(item)))
    return names


def build_query_planner_instruction(locked_names: Iterable[str]) -> str:
    locked = ", ".join(sorted(set(locked_names))) or "none"
    return (
        "You are a Japanese-to-search-query planner for a Danbooru tag database.\n"
        "Return only the requested JSON schema.\n"
        "Split fragmentary Japanese, omitted words, colloquial wording, and onomatopoeia into small visual intents.\n"
        "Generate 1-5 English search queries per intent; queries are only search text, never final tags.\n"
        "Use only category general. Do not infer a concrete object that the user did not mention: "
        "for example, 'oversized clothes' must not become hoodie, jacket, or coat.\n"
        "Preserve negation in must_not_infer; a negated concept must not become a positive query.\n"
        f"Core dictionary locked names that generic search must not compete with: {locked}.\n"
        "If the fragment is too vague, put it in unresolved and omit an intent."
    )


def build_candidate_rerank_instruction(locked_names: Iterable[str]) -> str:
    locked = ", ".join(sorted(set(locked_names))) or "none"
    return (
        "You are a conservative Danbooru candidate reviewer.\n"
        "Return only the requested JSON schema.\n"
        "Select only candidate tag_id values present in the supplied candidate list.\n"
        "Use accept only when the candidate directly matches the fragment and the query evidence.\n"
        "Use maybe for plausible but insufficiently certain candidates, reject for mismatch, and abstain when none are safe.\n"
        "Do not add a narrower object, action, or clothing type that the Japanese fragment did not say.\n"
        f"Never select a candidate that competes with a core locked name: {locked}.\n"
        "A high post_count is only a weak popularity prior, not evidence of meaning."
    )


def build_danbooru_tag_instruction(resolution: DanbooruTagResolution) -> str:
    if not resolution.selected_prompt_tags:
        return ""
    return "\n".join(
        (
            "## Generic Danbooru supplemental tags (authoritative candidates)",
            "These tags were selected from active general tags in the local Danbooru DB.",
            "Keep them in Positive when relevant; do not replace them with invented tags or infer extra objects.",
            "- selected prompt tags: " + ", ".join(resolution.selected_prompt_tags),
            "- selected tag IDs: " + ", ".join(str(tag_id) for tag_id in resolution.selected_tag_ids),
        )
    )


def _candidate_payload(intent: DanbooruSearchIntent, candidates: Sequence[DanbooruTagCandidate]) -> dict[str, object]:
    return {
        "fragment": intent.source,
        "queries": list(intent.queries),
        "must_not_infer": list(intent.must_not_infer),
        "candidates": [
            {
                "tag_id": candidate.tag_id,
                "name": candidate.name,
                "prompt_tag": candidate.prompt_tag,
                "post_count": candidate.post_count,
                "match_type": candidate.match_type,
                "query": candidate.query,
            }
            for candidate in candidates[:GENERIC_RERANK_LIMIT]
        ],
    }


def _parse_decisions(payload: Mapping[str, Any]) -> tuple[tuple[DanbooruCandidateDecision, ...], bool]:
    decisions: list[DanbooruCandidateDecision] = []
    raw_decisions = payload.get("decisions")
    if isinstance(raw_decisions, list):
        for raw in raw_decisions[:64]:
            if not isinstance(raw, Mapping):
                continue
            try:
                tag_id = int(raw.get("tag_id"))
                confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
            except (TypeError, ValueError):
                continue
            decision = str(raw.get("decision") or "reject").casefold()
            if decision not in {"accept", "maybe", "reject"}:
                decision = "reject"
            decisions.append(
                DanbooruCandidateDecision(
                    tag_id=tag_id,
                    decision=decision,
                    confidence=confidence,
                    reason=str(raw.get("reason") or "")[:240],
                )
            )
    return tuple(decisions), bool(payload.get("abstain", False))


def resolve_danbooru_tags(
    text: str,
    *,
    core_entries: Iterable[object] = (),
    store: DanbooruTagStore | None = None,
    llm_json: LlmJsonCallback | None = None,
) -> DanbooruTagResolution:
    """Resolve generic appearance fragments through DB candidates only."""

    search_store = store or DanbooruTagStore()
    if not should_search_generic_tags(text, db_path=search_store.path):
        return DanbooruTagResolution(snapshot_version=search_store.snapshot_version)
    if llm_json is None:
        return DanbooruTagResolution(
            snapshot_version=search_store.snapshot_version,
            unresolved_fragments=(str(text or "")[:160],),
            error="llm_json callback is not configured",
        )

    locked_names = _locked_names(core_entries)
    try:
        planner_raw = llm_json(
            build_query_planner_instruction(locked_names),
            str(text or ""),
            QUERY_PLANNER_RESPONSE_FORMAT,
            512,
        )
    except Exception as exc:  # noqa: BLE001 - search must not break normal generation
        logger.warning("Danbooru query planner failed: %s", exc)
        return DanbooruTagResolution(
            snapshot_version=search_store.snapshot_version,
            unresolved_fragments=(str(text or "")[:160],),
            error=f"query_planner:{type(exc).__name__}",
        )
    planner_payload = _extract_json(planner_raw)
    if planner_payload is None:
        return DanbooruTagResolution(
            snapshot_version=search_store.snapshot_version,
            unresolved_fragments=(str(text or "")[:160],),
            error="query_planner:invalid_json",
        )
    intents, unresolved = _parse_intents(planner_payload, str(text or ""))
    if not intents:
        return DanbooruTagResolution(
            snapshot_version=search_store.snapshot_version,
            unresolved_fragments=unresolved or (str(text or "")[:160],),
        )

    candidates: list[DanbooruTagCandidate] = []
    candidates_by_source: dict[str, list[DanbooruTagCandidate]] = {}
    for intent in intents:
        by_id: dict[int, DanbooruTagCandidate] = {}
        for query in intent.queries:
            for candidate in search_store.search(query, limit=GENERIC_SEARCH_INTERNAL_LIMIT):
                if normalize_tag_name(candidate.name) in locked_names:
                    continue
                previous = by_id.get(candidate.tag_id)
                if previous is None or candidate.retrieval_score > previous.retrieval_score:
                    by_id[candidate.tag_id] = candidate
        ordered = sorted(
            by_id.values(),
            key=lambda candidate: (
                -candidate.retrieval_score,
                -math_log_post_count(candidate.post_count),
                candidate.name,
            ),
        )
        candidates_by_source[intent.source] = ordered[:GENERIC_RERANK_LIMIT]
        candidates.extend(ordered)

    if not candidates:
        return DanbooruTagResolution(
            snapshot_version=search_store.snapshot_version,
            fragments=tuple(intent.source for intent in intents),
            intents=intents,
            queries=tuple(query for intent in intents for query in intent.queries),
            unresolved_fragments=unresolved,
        )

    rerank_input = {
        "fragments": [
            _candidate_payload(intent, candidates_by_source.get(intent.source, ()))
            for intent in intents
            if candidates_by_source.get(intent.source)
        ]
    }
    try:
        rerank_raw = llm_json(
            build_candidate_rerank_instruction(locked_names),
            json.dumps(rerank_input, ensure_ascii=False),
            RERANK_RESPONSE_FORMAT,
            768,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Danbooru candidate reranker failed: %s", exc)
        return DanbooruTagResolution(
            snapshot_version=search_store.snapshot_version,
            fragments=tuple(intent.source for intent in intents),
            intents=intents,
            queries=tuple(query for intent in intents for query in intent.queries),
            candidates=tuple(candidates),
            unresolved_fragments=unresolved,
            error=f"candidate_reranker:{type(exc).__name__}",
        )
    rerank_payload = _extract_json(rerank_raw)
    if rerank_payload is None:
        return DanbooruTagResolution(
            snapshot_version=search_store.snapshot_version,
            fragments=tuple(intent.source for intent in intents),
            intents=intents,
            queries=tuple(query for intent in intents for query in intent.queries),
            candidates=tuple(candidates),
            unresolved_fragments=unresolved,
            error="candidate_reranker:invalid_json",
        )

    decisions, abstained = _parse_decisions(rerank_payload)
    candidate_ids = {candidate.tag_id for candidate in candidates}
    accepted_ids = tuple(
        dict.fromkeys(
            decision.tag_id
            for decision in decisions
            if decision.tag_id in candidate_ids
            and decision.decision == "accept"
            and decision.confidence >= GENERIC_ACCEPT_CONFIDENCE
        )
    )
    valid_rows = search_store.validate_candidate_ids(
        accepted_ids,
        candidate_ids=candidate_ids,
        locked_names=locked_names,
    )
    selected_ids = tuple(row.tag_id for row in valid_rows)
    selected_prompt_tags = tuple(row.prompt_tag for row in valid_rows)
    rejected_ids = tuple(
        decision.tag_id
        for decision in decisions
        if decision.tag_id in candidate_ids and decision.decision == "reject"
    )
    maybe_ids = tuple(
        decision.tag_id
        for decision in decisions
        if decision.tag_id in candidate_ids and decision.decision == "maybe"
    )
    return DanbooruTagResolution(
        snapshot_version=search_store.snapshot_version,
        fragments=tuple(intent.source for intent in intents),
        intents=intents,
        queries=tuple(query for intent in intents for query in intent.queries),
        candidates=tuple(candidates),
        decisions=decisions,
        selected_tag_ids=selected_ids,
        selected_prompt_tags=selected_prompt_tags,
        rejected_tag_ids=rejected_ids,
        maybe_tag_ids=maybe_ids,
        unresolved_fragments=unresolved,
        abstained=abstained,
    )


def math_log_post_count(value: int) -> float:
    # Kept as a tiny helper to make the ranking policy explicit and testable.
    import math

    return math.log1p(max(int(value), 0))


def apply_danbooru_tag_resolution(
    positive: str,
    resolution: DanbooruTagResolution | None,
) -> str:
    if resolution is None or not resolution.selected_prompt_tags:
        return positive.strip()
    kept: list[str] = []
    seen: set[str] = set()
    selected = {normalize_search_text(tag) for tag in resolution.selected_prompt_tags}
    for raw_token in str(positive or "").split(","):
        token = raw_token.strip()
        if not token:
            continue
        key = normalize_search_text(token)
        if key in selected or key in seen:
            continue
        kept.append(token)
        seen.add(key)
    for prompt_tag in resolution.selected_prompt_tags:
        key = normalize_search_text(prompt_tag)
        if key not in seen:
            kept.append(prompt_tag)
            seen.add(key)
    return ", ".join(kept)
