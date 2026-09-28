from __future__ import annotations

import contextvars
import copy
import json
import re
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


DEBUG_HISTORY_PER_USER = 3
CONTEXT_LIMIT_DEFAULT = 8192
DETAIL_FIELD_MAX_CHARS = 12000
DISCORD_DETAIL_MESSAGE_LIMIT = 1900
DETAIL_STAGE_ORDER = (
    "03_lm_studio",
    "04_json_parse",
    "06_rating",
    "08_character",
    "10_comfyui",
)
STAGE_ORDER = (
    "01_request",
    "02_prompt_build",
    "03_lm_studio",
    "04_json_parse",
    "06_rating",
    "08_character",
    "10_comfyui",
)
STAGE_LABELS = {
    "01_request": "01 Request",
    "02_prompt_build": "02 Prompt build",
    "03_lm_studio": "03 LM Studio",
    "04_json_parse": "04 JSON parse",
    "06_rating": "06 Rating",
    "08_character": "08 Character",
    "10_comfyui": "10 ComfyUI",
}


@dataclass
class GenerateAiStageDetail:
    positive: str | None = None
    negative: str | None = None
    raw_imported_positive: str | None = None
    raw_imported_negative: str | None = None
    parsed_imported_tags: tuple[str, ...] = ()
    canonicalized_tags: tuple[str, ...] = ()
    imported_prompt_tags: tuple[str, ...] = ()
    conflict_removed_tags: tuple[str, ...] = ()
    character_resolution: tuple[str, ...] = ()
    framing_resolution: tuple[str, ...] = ()
    llm_added_tags: tuple[str, ...] = ()
    removed_llm_only_tags: tuple[str, ...] = ()
    rating: str | None = None
    character_name: str | None = None
    added_tags: tuple[str, ...] = ()
    appearance_removed_tags: tuple[str, ...] = ()
    raw_llm_positive: str | None = None
    raw_summary: str | None = None
    final_summary: str | None = None
    raw_assumptions: tuple[str, ...] = ()
    final_assumptions: tuple[str, ...] = ()
    semantic_grounding_result: tuple[str, ...] = ()
    user_grounded_tags: tuple[str, ...] = ()
    removed_unrequested_semantic_tags: tuple[str, ...] = ()
    pose_normalizations: tuple[str, ...] = ()
    cleanup_removed_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerateAiValidationAttempt:
    attempt: int
    missing_tags: tuple[str, ...] = ()
    conflicting_tags: tuple[str, ...] = ()
    framing_missing_tags: tuple[str, ...] = ()
    framing_conflicting_tags: tuple[str, ...] = ()
    background_conflicting_tags: tuple[str, ...] = ()
    lighting_missing_tags: tuple[str, ...] = ()
    lighting_conflicting_tags: tuple[str, ...] = ()
    assumption_violations: tuple[str, ...] = ()


@dataclass
class GenerateAiDebugRecord:
    trace_id: str
    user_id: int
    model_key: str
    model_name: str
    requested_rating: str
    final_rating: str | None = None
    result: str = "running"
    stages: dict[str, str] = field(
        default_factory=lambda: {stage: "pending" for stage in STAGE_ORDER}
    )
    last_stage: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    failure_subsystem: str | None = None
    http_status: int | None = None
    input_tokens: int | None = None
    context_limit: int | None = None
    timeout_seconds: float | None = None
    character_merge: str | None = None
    selected_character: str | None = None
    comfyui_reached: bool = False
    final_positive_tags: int | None = None
    prompt_tag_dictionary_version: str | None = None
    resolved_tag_ids: tuple[str, ...] = ()
    resolved_prompt_tags: tuple[str, ...] = ()
    unresolved_japanese_phrases: tuple[str, ...] = ()
    danbooru_tag_db_version: str | None = None
    danbooru_tag_resolution: dict[str, Any] = field(default_factory=dict)
    danbooru_character_tag_db_version: str | None = None
    danbooru_character_resolution: dict[str, Any] = field(default_factory=dict)
    stage_details: dict[str, GenerateAiStageDetail] = field(default_factory=dict)
    validation_attempts: list[GenerateAiValidationAttempt] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


_lock = threading.RLock()
_records_by_user: dict[int, deque[GenerateAiDebugRecord]] = defaultdict(
    lambda: deque(maxlen=DEBUG_HISTORY_PER_USER)
)
_records_by_trace: dict[str, GenerateAiDebugRecord] = {}
_active_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "generate_ai_debug_trace_id", default=None
)


def start_generate_ai_debug(
    user_id: int,
    trace_id: str,
    *,
    model_key: str,
    model_name: str,
    requested_rating: str,
) -> GenerateAiDebugRecord:
    record = GenerateAiDebugRecord(
        trace_id=trace_id,
        user_id=int(user_id),
        model_key=model_key,
        model_name=model_name,
        requested_rating=requested_rating,
    )
    with _lock:
        previous = _records_by_trace.get(trace_id)
        if previous is not None:
            try:
                _records_by_user[previous.user_id].remove(previous)
            except ValueError:
                pass
        records = _records_by_user[int(user_id)]
        if len(records) == records.maxlen:
            evicted = records[0]
            _records_by_trace.pop(evicted.trace_id, None)
        records.append(record)
        _records_by_trace[trace_id] = record
    return copy.deepcopy(record)


def activate_generate_ai_debug(trace_id: str) -> contextvars.Token[str | None]:
    return _active_trace_id.set(trace_id)


def deactivate_generate_ai_debug(token: contextvars.Token[str | None]) -> None:
    _active_trace_id.reset(token)


def get_active_generate_ai_trace_id() -> str | None:
    return _active_trace_id.get()


def _record(trace_id: str | None) -> GenerateAiDebugRecord | None:
    if not trace_id:
        return None
    return _records_by_trace.get(trace_id)


def mark_generate_ai_stage(
    trace_id: str | None,
    stage: str,
    status: str = "ok",
    *,
    final_rating: str | None = None,
    character_merge: str | None = None,
    comfyui_reached: bool | None = None,
    final_positive_tags: int | None = None,
) -> None:
    if stage not in STAGE_ORDER or status not in {"pending", "ok", "error"}:
        return
    with _lock:
        record = _record(trace_id)
        if record is None:
            return
        record.stages[stage] = status
        record.last_stage = stage
        if final_rating is not None:
            record.final_rating = final_rating
        if character_merge is not None:
            record.character_merge = character_merge
        if comfyui_reached is not None:
            record.comfyui_reached = comfyui_reached
        if final_positive_tags is not None:
            record.final_positive_tags = final_positive_tags


def set_generate_ai_debug_character(
    trace_id: str | None,
    character_name: str | None,
) -> None:
    with _lock:
        record = _record(trace_id)
        if record is not None:
            record.selected_character = character_name.strip() if character_name else None


def store_generate_ai_prompt_tag_resolution(
    trace_id: str | None,
    *,
    dictionary_version: str,
    resolved_tag_ids: tuple[str, ...] | list[str] = (),
    resolved_prompt_tags: tuple[str, ...] | list[str] = (),
    unresolved_japanese_phrases: tuple[str, ...] | list[str] = (),
) -> None:
    """Store compact prompt-tag diagnostics without changing Discord output."""

    def bounded(values: tuple[str, ...] | list[str], limit: int = 40) -> tuple[str, ...]:
        return tuple(
            value.strip()[:300]
            for value in values[:limit]
            if isinstance(value, str) and value.strip()
        )

    with _lock:
        record = _record(trace_id)
        if record is None:
            return
        record.prompt_tag_dictionary_version = str(dictionary_version).strip() or None
        record.resolved_tag_ids = bounded(resolved_tag_ids)
        record.resolved_prompt_tags = bounded(resolved_prompt_tags)
        record.unresolved_japanese_phrases = bounded(unresolved_japanese_phrases)


def store_generate_ai_danbooru_tag_resolution(
    trace_id: str | None,
    resolution: Mapping[str, Any] | None,
) -> None:
    """Store bounded generic Danbooru search diagnostics without Discord noise."""

    if not resolution:
        return

    def bounded_strings(values: object, limit: int = 40) -> list[str]:
        if not isinstance(values, (list, tuple)):
            return []
        return [
            str(value).strip()[:300]
            for value in values[:limit]
            if str(value).strip()
        ]

    compact: dict[str, Any] = {
        "input_fragments": bounded_strings(resolution.get("fragments")),
        "generated_search_queries": bounded_strings(resolution.get("queries")),
        "selected_tag_ids": bounded_strings(resolution.get("selected_tag_ids")),
        "rejected_tag_ids": bounded_strings(resolution.get("rejected_tag_ids")),
        "maybe_tag_ids": bounded_strings(resolution.get("maybe_tag_ids")),
        "final_resolved_danbooru_tags": bounded_strings(resolution.get("selected_prompt_tags")),
        "unresolved_fragments": bounded_strings(resolution.get("unresolved_fragments")),
        "abstained": bool(resolution.get("abstained", False)),
        "error": str(resolution.get("error") or "")[:300],
    }
    raw_candidates = resolution.get("candidates")
    if isinstance(raw_candidates, list):
        compact["db_candidates"] = [
            value for value in raw_candidates[:80] if isinstance(value, Mapping)
        ]
    raw_intents = resolution.get("intents")
    if isinstance(raw_intents, list):
        compact["intents"] = [
            value for value in raw_intents[:8] if isinstance(value, Mapping)
        ]
    try:
        # Ensure the stored trace remains JSON-compatible and bounded enough for
        # the in-memory debug record.
        compact = json.loads(json.dumps(compact, ensure_ascii=False))
    except (TypeError, ValueError):
        compact = {"error": "danbooru_trace_not_serializable"}

    with _lock:
        record = _record(trace_id)
        if record is None:
            return
        record.danbooru_tag_db_version = str(resolution.get("snapshot_version") or "unknown")[:200]
        record.danbooru_tag_resolution = compact


def store_generate_ai_character_resolution(
    trace_id: str | None,
    resolution: Mapping[str, Any] | None,
) -> None:
    """Store bounded Japanese-character resolver diagnostics without Discord noise."""

    if not resolution:
        return

    def bounded_strings(values: object, limit: int = 40) -> list[str]:
        if not isinstance(values, (list, tuple)):
            return []
        return [
            str(value).strip()[:300]
            for value in values[:limit]
            if str(value).strip()
        ]

    compact: dict[str, Any] = {
        "input_character_name": str(resolution.get("input_character_name") or "")[:200],
        "normalized_character_name": str(resolution.get("normalized_character_name") or "")[:200],
        "registry_hit": bool(resolution.get("registry_hit", False)),
        "registry_character_id": resolution.get("registry_character_id"),
        "generated_queries": bounded_strings(resolution.get("generated_queries")),
        "generated_work_queries": bounded_strings(resolution.get("generated_work_queries")),
        "candidate_match_strength": resolution.get("candidate_match_strength", 0.0),
        "selected_candidate": str(resolution.get("selected_candidate") or "")[:300],
        "selected_character_tag": str(resolution.get("selected_character_tag") or "")[:300],
        "confirmation_required": bool(resolution.get("confirmation_required", False)),
        "confirmed_character_tag": str(resolution.get("confirmed_character_tag") or "")[:300],
        "saved_alias": str(resolution.get("saved_alias") or "")[:200],
        "selected_tag_id": resolution.get("selected_tag_id"),
        "confidence": resolution.get("confidence", 0.0),
        "unresolved_reason": str(resolution.get("unresolved_reason") or "")[:300],
        "abstained": bool(resolution.get("abstained", False)),
        "rejected_tag_ids": bounded_strings(resolution.get("rejected_tag_ids")),
    }
    raw_candidates = resolution.get("candidate_character_tags")
    if isinstance(raw_candidates, list):
        compact["candidate_character_tags"] = [
            value for value in raw_candidates[:20] if isinstance(value, Mapping)
        ]
    try:
        compact = json.loads(json.dumps(compact, ensure_ascii=False))
    except (TypeError, ValueError):
        compact = {"error": "danbooru_character_trace_not_serializable"}

    with _lock:
        record = _record(trace_id)
        if record is None:
            return
        record.danbooru_character_tag_db_version = str(
            resolution.get("snapshot_version") or "unknown"
        )[:200]
        record.danbooru_character_resolution = compact


def _bounded_detail_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if len(text) <= DETAIL_FIELD_MAX_CHARS:
        return text
    return text[:DETAIL_FIELD_MAX_CHARS].rstrip() + " ... <truncated>"


def store_generate_ai_stage_detail(
    trace_id: str | None,
    stage: str,
    *,
    positive: str | None = None,
    negative: str | None = None,
    raw_imported_positive: str | None = None,
    raw_imported_negative: str | None = None,
    parsed_imported_tags: tuple[str, ...] | list[str] = (),
    canonicalized_tags: tuple[str, ...] | list[str] = (),
    imported_prompt_tags: tuple[str, ...] | list[str] = (),
    conflict_removed_tags: tuple[str, ...] | list[str] = (),
    character_resolution: tuple[str, ...] | list[str] = (),
    framing_resolution: tuple[str, ...] | list[str] = (),
    llm_added_tags: tuple[str, ...] | list[str] = (),
    removed_llm_only_tags: tuple[str, ...] | list[str] = (),
    rating: str | None = None,
    character_name: str | None = None,
    added_tags: tuple[str, ...] | list[str] = (),
    appearance_removed_tags: tuple[str, ...] | list[str] = (),
    raw_llm_positive: str | None = None,
    raw_summary: str | None = None,
    final_summary: str | None = None,
    raw_assumptions: tuple[str, ...] | list[str] = (),
    final_assumptions: tuple[str, ...] | list[str] = (),
    semantic_grounding_result: tuple[str, ...] | list[str] = (),
    user_grounded_tags: tuple[str, ...] | list[str] = (),
    removed_unrequested_semantic_tags: tuple[str, ...] | list[str] = (),
    pose_normalizations: tuple[str, ...] | list[str] = (),
    cleanup_removed_tags: tuple[str, ...] | list[str] = (),
) -> None:
    """Stores only bounded parsed prompt fields needed by the Discord detail view."""
    if stage not in DETAIL_STAGE_ORDER:
        return
    clean_added = tuple(
        tag.strip()[:500]
        for tag in added_tags
        if isinstance(tag, str) and tag.strip()
    )
    clean_appearance_removed = tuple(
        tag.strip()[:500]
        for tag in appearance_removed_tags
        if isinstance(tag, str) and tag.strip()
    )
    clean_cleanup_removed = tuple(
        tag.strip()[:500]
        for tag in cleanup_removed_tags
        if isinstance(tag, str) and tag.strip()
    )
    def clean_values(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        return tuple(
            tag.strip()[:500]
            for tag in values
            if isinstance(tag, str) and tag.strip()
        )
    detail = GenerateAiStageDetail(
        positive=_bounded_detail_text(positive),
        negative=_bounded_detail_text(negative),
        raw_imported_positive=_bounded_detail_text(raw_imported_positive),
        raw_imported_negative=_bounded_detail_text(raw_imported_negative),
        parsed_imported_tags=clean_values(parsed_imported_tags),
        canonicalized_tags=clean_values(canonicalized_tags),
        imported_prompt_tags=clean_values(imported_prompt_tags),
        conflict_removed_tags=clean_values(conflict_removed_tags),
        character_resolution=clean_values(character_resolution),
        framing_resolution=clean_values(framing_resolution),
        llm_added_tags=clean_values(llm_added_tags),
        removed_llm_only_tags=clean_values(removed_llm_only_tags),
        rating=rating.strip() if rating else None,
        character_name=character_name.strip() if character_name else None,
        added_tags=clean_added,
        appearance_removed_tags=clean_appearance_removed,
        raw_llm_positive=_bounded_detail_text(raw_llm_positive),
        raw_summary=_bounded_detail_text(raw_summary),
        final_summary=_bounded_detail_text(final_summary),
        raw_assumptions=clean_values(raw_assumptions),
        final_assumptions=clean_values(final_assumptions),
        semantic_grounding_result=clean_values(semantic_grounding_result),
        user_grounded_tags=clean_values(user_grounded_tags),
        removed_unrequested_semantic_tags=clean_values(removed_unrequested_semantic_tags),
        pose_normalizations=clean_values(pose_normalizations),
        cleanup_removed_tags=clean_cleanup_removed,
    )
    with _lock:
        record = _record(trace_id)
        if record is not None:
            record.stage_details[stage] = detail


def store_generate_ai_validation_attempt(
    trace_id: str | None,
    attempt: int,
    *,
    missing_tags: tuple[str, ...] | list[str] = (),
    conflicting_tags: tuple[str, ...] | list[str] = (),
    framing_missing_tags: tuple[str, ...] | list[str] = (),
    framing_conflicting_tags: tuple[str, ...] | list[str] = (),
    background_conflicting_tags: tuple[str, ...] | list[str] = (),
    lighting_missing_tags: tuple[str, ...] | list[str] = (),
    lighting_conflicting_tags: tuple[str, ...] | list[str] = (),
    assumption_violations: tuple[str, ...] | list[str] = (),
) -> None:
    """Stores bounded validation diagnostics without retaining the request."""

    if attempt < 1:
        return

    def bounded(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        return tuple(
            value.strip()[:300]
            for value in values[:20]
            if isinstance(value, str) and value.strip()
        )

    clean_framing_missing = bounded(framing_missing_tags or missing_tags)
    clean_framing_conflicting = bounded(framing_conflicting_tags or conflicting_tags)
    clean_background_conflicting = bounded(background_conflicting_tags)
    clean_lighting_missing = bounded(lighting_missing_tags)
    clean_lighting_conflicting = bounded(lighting_conflicting_tags)
    item = GenerateAiValidationAttempt(
        attempt=attempt,
        missing_tags=tuple(dict.fromkeys((*clean_framing_missing, *clean_lighting_missing))),
        conflicting_tags=tuple(
            dict.fromkeys(
                (*clean_framing_conflicting, *clean_background_conflicting, *clean_lighting_conflicting)
            )
        ),
        framing_missing_tags=clean_framing_missing,
        framing_conflicting_tags=clean_framing_conflicting,
        background_conflicting_tags=clean_background_conflicting,
        lighting_missing_tags=clean_lighting_missing,
        lighting_conflicting_tags=clean_lighting_conflicting,
        assumption_violations=bounded(assumption_violations),
    )
    with _lock:
        record = _record(trace_id)
        if record is None:
            return
        record.validation_attempts = [
            existing
            for existing in record.validation_attempts
            if existing.attempt != attempt
        ]
        record.validation_attempts.append(item)
        record.validation_attempts.sort(key=lambda existing: existing.attempt)
        del record.validation_attempts[3:]


def update_generate_ai_metrics(
    trace_id: str | None,
    *,
    input_tokens: int | None = None,
    context_limit: int | None = None,
    http_status: int | None = None,
    timeout_seconds: float | None = None,
) -> None:
    with _lock:
        record = _record(trace_id)
        if record is None:
            return
        if input_tokens is not None:
            record.input_tokens = input_tokens
        if context_limit is not None:
            record.context_limit = context_limit
        if http_status is not None:
            record.http_status = http_status
        if timeout_seconds is not None:
            record.timeout_seconds = timeout_seconds


def fail_generate_ai_debug(
    trace_id: str | None,
    *,
    error_type: str,
    message: str,
    subsystem: str,
    stage: str | None = None,
    http_status: int | None = None,
    input_tokens: int | None = None,
    context_limit: int | None = None,
    timeout_seconds: float | None = None,
) -> None:
    with _lock:
        record = _record(trace_id)
        if record is None:
            return
        record.result = "failed"
        record.error_type = error_type
        record.error_message = " ".join(message.split())[:500]
        record.failure_subsystem = subsystem
        if stage in STAGE_ORDER:
            record.stages[stage] = "error"
            record.last_stage = stage
        if http_status is not None:
            record.http_status = http_status
        if input_tokens is not None:
            record.input_tokens = input_tokens
        if context_limit is not None:
            record.context_limit = context_limit
        if timeout_seconds is not None:
            record.timeout_seconds = timeout_seconds


def complete_generate_ai_debug(trace_id: str | None) -> None:
    with _lock:
        record = _record(trace_id)
        if record is not None:
            record.result = "success"
            record.error_type = None
            record.error_message = None
            record.failure_subsystem = None


def get_latest_generate_ai_debug(user_id: int) -> GenerateAiDebugRecord | None:
    with _lock:
        records = _records_by_user.get(int(user_id))
        if not records:
            return None
        return copy.deepcopy(records[-1])


def clear_generate_ai_debug_records() -> None:
    """Test helper. Runtime records are intentionally memory-only and bounded."""
    with _lock:
        _records_by_user.clear()
        _records_by_trace.clear()


def parse_lm_context_metrics(body: str) -> tuple[int | None, int | None]:
    """Extracts prompt/context token counts without retaining the HTTP body."""
    text = body or ""
    input_patterns = (
        r'"n_keep"\s*:\s*(\d+)',
        r"n_keep\D{0,12}(\d+)",
        r"(?:prompt|input)\D{0,20}(\d+)\s*tokens?",
    )
    limit_patterns = (
        r'"n_ctx"\s*:\s*(\d+)',
        r"n_ctx\D{0,12}(\d+)",
        r"context(?: window| limit)?\D{0,20}(\d+)\s*tokens?",
    )

    def first(patterns: tuple[str, ...]) -> int | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    return first(input_patterns), first(limit_patterns)


def is_context_overflow_body(body: str) -> bool:
    lowered = (body or "").casefold()
    return (
        ("n_keep" in lowered and "n_ctx" in lowered)
        or (
            "context" in lowered
            and any(term in lowered for term in ("overflow", "too large", "exceed"))
        )
    )


def classify_comfy_error(error: BaseException) -> tuple[str, str]:
    text = " ".join(str(error).split())
    lowered = text.casefold()
    if "参考画像" in text and "アップロード" in text:
        return "reference_image_upload_error", "Reference image upload failed"
    if any(term in lowered for term in ("connection", "connect", "接続", "winerror 10061")):
        return "comfyui_connection_error", "ComfyUI connection error"
    if any(term in lowered for term in ("workflow", "/prompt", "prompt_id", "ワークフロー")):
        return "comfyui_workflow_error", "ComfyUI workflow/prompt error"
    return "unknown_error", text[:300] or "Unknown image generation error"


def format_generate_ai_debug(record: GenerateAiDebugRecord) -> str:
    result_label = {
        "success": "Success",
        "failed": "Failed",
        "running": "Running",
    }.get(record.result, record.result)
    lines = [
        f"Trace: {record.trace_id}",
        f"Result: {result_label}",
        f"Model: {record.model_name}",
        f"Rating: {record.final_rating or record.requested_rating}",
        f"Last stage: {STAGE_LABELS.get(record.last_stage or '', '-')}",
        "",
    ]
    status_labels = {"ok": "OK", "error": "ERROR", "pending": "-"}
    for stage in STAGE_ORDER:
        lines.append(f"{STAGE_LABELS[stage]:<17} {status_labels[record.stages.get(stage, 'pending')]}")

    if record.result == "failed":
        lines.extend(["", "Error:", record.error_message or record.error_type or "Unknown error"])
        if record.failure_subsystem:
            lines.append(f"Subsystem: {record.failure_subsystem}")
        if record.http_status is not None:
            lines.append(f"HTTP: {record.http_status}")
        if record.timeout_seconds is not None:
            lines.append(f"Timeout: {record.timeout_seconds:g}s")

    if record.input_tokens is not None:
        if record.context_limit is not None:
            lines.append(f"Input tokens: {record.input_tokens} / {record.context_limit}")
        else:
            lines.append(f"Input tokens: {record.input_tokens}")
    elif record.context_limit is not None:
        lines.append(f"Context limit: {record.context_limit}")
    lines.append(f"Character merge: {record.character_merge or 'No'}")
    if record.prompt_tag_dictionary_version:
        lines.append(f"Prompt tag dictionary: {record.prompt_tag_dictionary_version}")
        if record.resolved_prompt_tags:
            lines.append("Resolved prompt tags: " + ", ".join(record.resolved_prompt_tags))
        if record.unresolved_japanese_phrases:
            lines.append(f"Unresolved composition phrases: {len(record.unresolved_japanese_phrases)}")
    if record.danbooru_tag_db_version:
        lines.append(f"Danbooru tag DB: {record.danbooru_tag_db_version}")
        danbooru_trace = record.danbooru_tag_resolution
        queries = danbooru_trace.get("generated_search_queries", [])
        selected = danbooru_trace.get("final_resolved_danbooru_tags", [])
        if queries:
            lines.append("Danbooru queries: " + ", ".join(str(value) for value in queries[:8]))
        if selected:
            lines.append("Danbooru selected: " + ", ".join(str(value) for value in selected[:8]))
        if danbooru_trace.get("abstained"):
            lines.append("Danbooru candidate review: abstained")
    if record.danbooru_character_tag_db_version:
        character_trace = record.danbooru_character_resolution
        confirmed_character = character_trace.get("confirmed_character_tag")
        selected_candidate = character_trace.get("selected_candidate")
        if confirmed_character:
            lines.append("Danbooru character confirmed: " + str(confirmed_character))
        elif selected_candidate:
            lines.append("Danbooru character candidate: " + str(selected_candidate))
            if character_trace.get("confirmation_required"):
                lines.append("Danbooru character review: confirmation required")
        elif character_trace.get("unresolved_reason"):
            lines.append(
                "Danbooru character: unresolved ("
                + str(character_trace.get("unresolved_reason"))
                + ")"
            )
    lines.append(f"ComfyUI reached: {'Yes' if record.comfyui_reached else 'No'}")
    if record.final_positive_tags is not None:
        lines.append(f"Final positive tags: {record.final_positive_tags}")
    return "```text\n" + "\n".join(lines) + "\n```"


def _split_detail_value(value: str, max_chars: int) -> list[str]:
    if len(value) <= max_chars:
        return [value]
    tags = [tag.strip() for tag in value.split(",") if tag.strip()]
    if len(tags) <= 1:
        return [value[index : index + max_chars] for index in range(0, len(value), max_chars)]
    chunks: list[str] = []
    current = ""
    for tag in tags:
        candidate = tag if not current else f"{current}, {tag}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(tag) <= max_chars:
            current = tag
        else:
            chunks.extend(tag[index : index + max_chars] for index in range(0, len(tag), max_chars))
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _detail_stage_fields(detail: GenerateAiStageDetail, stage: str) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    if stage == "04_json_parse":
        if detail.raw_imported_positive:
            fields.append(("Raw imported Positive", detail.raw_imported_positive))
        if detail.raw_imported_negative:
            fields.append(("Raw imported Negative", detail.raw_imported_negative))
        if detail.parsed_imported_tags:
            fields.append(("Parsed imported tags", ", ".join(detail.parsed_imported_tags)))
        if detail.canonicalized_tags:
            fields.append(("Canonicalized tags", " | ".join(detail.canonicalized_tags)))
        if detail.imported_prompt_tags:
            fields.append(("Imported prompt tags", ", ".join(detail.imported_prompt_tags)))
        if detail.conflict_removed_tags:
            fields.append(("Conflict removed", ", ".join(detail.conflict_removed_tags)))
        if detail.character_resolution:
            fields.append(("Character resolution", " | ".join(detail.character_resolution)))
        if detail.framing_resolution:
            fields.append(("Framing resolution", " | ".join(detail.framing_resolution)))
        if detail.tag_provenance:
            fields.append(("Tag provenance", " | ".join(detail.tag_provenance)))
        fields.append(("LLM-added tags", ", ".join(detail.llm_added_tags) if detail.llm_added_tags else "None"))
        fields.append(
            (
                "LLM-only removed",
                ", ".join(detail.removed_llm_only_tags) if detail.removed_llm_only_tags else "None",
            )
        )
    if stage == "06_rating" and detail.rating:
        fields.append(("Rating", detail.rating))
    if stage == "06_rating" and detail.appearance_removed_tags:
        fields.append(("Appearance removed", ", ".join(detail.appearance_removed_tags)))
    if stage == "06_rating" and detail.raw_llm_positive:
        fields.append(("Raw LLM Positive", detail.raw_llm_positive))
    if stage == "06_rating" and detail.raw_summary:
        fields.append(("Raw Summary", detail.raw_summary))
    if stage == "06_rating" and detail.final_summary:
        fields.append(("Final Summary", detail.final_summary))
    if stage == "06_rating" and detail.raw_assumptions:
        fields.append(("Raw Assumptions", ", ".join(detail.raw_assumptions)))
    if stage == "06_rating" and detail.final_assumptions:
        fields.append(("Final Assumptions", ", ".join(detail.final_assumptions)))
    if stage == "06_rating" and detail.semantic_grounding_result:
        fields.append(("Semantic grounding", " | ".join(detail.semantic_grounding_result)))
    if stage == "06_rating" and detail.user_grounded_tags:
        fields.append(("User-grounded tags", ", ".join(detail.user_grounded_tags)))
    if stage == "06_rating" and detail.removed_unrequested_semantic_tags:
        fields.append(("Unrequested semantic removed", ", ".join(detail.removed_unrequested_semantic_tags)))
    if stage == "06_rating" and detail.pose_normalizations:
        fields.append(("Pose normalizations", " | ".join(detail.pose_normalizations)))
    if stage == "06_rating" and detail.cleanup_removed_tags:
        fields.append(("Cleanup removed", ", ".join(detail.cleanup_removed_tags)))
    if stage == "08_character":
        fields.append(("Character merge", detail.character_name or "None"))
        fields.append(("Added", ", ".join(detail.added_tags) if detail.added_tags else "None"))
        if detail.positive:
            fields.append(("Final Positive", detail.positive))
        return fields
    if detail.positive:
        fields.append(("Positive", detail.positive))
    if detail.negative and stage in {"03_lm_studio", "04_json_parse", "10_comfyui"}:
        fields.append(("Negative", detail.negative))
    return fields


def _format_detail_stage_chunks(
    stage: str,
    detail: GenerateAiStageDetail,
    *,
    limit: int,
) -> list[str]:
    title = f"**{STAGE_LABELS[stage]}**"
    fields = _detail_stage_fields(detail, stage)
    combined_lines = [title]
    for label, value in fields:
        safe_value = value.replace("```", "``\u200b`")
        combined_lines.extend([f"**{label}:**", f"```\n{safe_value}\n```"])
    combined = "\n".join(combined_lines)
    if len(combined) <= limit:
        return [combined]

    chunks: list[str] = []
    for label, value in fields:
        prefix = f"{title}\n**{label}:**\n```\n"
        suffix = "\n```"
        available = max(100, limit - len(prefix) - len(suffix) - 24)
        value_parts = _split_detail_value(value.replace("```", "``\u200b`"), available)
        for index, part in enumerate(value_parts, start=1):
            part_label = label if len(value_parts) == 1 else f"{label} ({index}/{len(value_parts)})"
            message = f"{title}\n**{part_label}:**\n```\n{part}\n```"
            chunks.append(message)
    return chunks or [title + "\n（保持されたフィールドはありません）"]


def _format_validation_chunks(
    attempts: list[GenerateAiValidationAttempt],
    *,
    limit: int,
) -> list[str]:
    chunks: list[str] = []
    for item in attempts:
        fields = (
            (
                "Framing missing",
                ", ".join(item.framing_missing_tags)
                if item.framing_missing_tags
                else "None",
            ),
            (
                "Framing conflicts",
                ", ".join(item.framing_conflicting_tags)
                if item.framing_conflicting_tags
                else "None",
            ),
            (
                "Background conflicts",
                ", ".join(item.background_conflicting_tags)
                if item.background_conflicting_tags
                else "None",
            ),
            (
                "Lighting missing",
                ", ".join(item.lighting_missing_tags)
                if item.lighting_missing_tags
                else "None",
            ),
            (
                "Lighting conflicts",
                ", ".join(item.lighting_conflicting_tags)
                if item.lighting_conflicting_tags
                else "None",
            ),
        )
        if item.assumption_violations:
            fields += (("Lighting assumptions", " | ".join(item.assumption_violations)),)
        block_lines = ["**Validation**", f"**Attempt {item.attempt}:**"]
        block_lines.extend(f"{label}: {value}" for label, value in fields)
        block = "\n".join(block_lines)
        if len(block) <= limit:
            chunks.append(block)
            continue
        for label, value in fields:
            prefix = f"**Validation — Attempt {item.attempt}**\n{label}: "
            available = max(100, limit - len(prefix) - 12)
            chunks.extend(prefix + part for part in _split_detail_value(value, available))
    return chunks


def format_generate_ai_debug_detail_chunks(
    record: GenerateAiDebugRecord,
    *,
    limit: int = DISCORD_DETAIL_MESSAGE_LIMIT,
) -> list[str]:
    if limit < 300:
        raise ValueError("detail message limit must be at least 300 characters")
    result_label = {
        "success": "Success",
        "failed": "Failed",
        "running": "Running",
    }.get(record.result, record.result)
    header_lines = [
        f"**Trace:** `{record.trace_id}`",
        f"**Result:** {result_label}",
        f"**Model:** {record.model_name}",
        f"**Character:** {record.selected_character or '-'}",
    ]
    if record.result == "failed":
        header_lines.append(f"**Error:** {record.error_message or record.error_type or 'Unknown error'}")
    chunks = ["\n".join(header_lines)[:limit]]
    chunks.extend(_format_validation_chunks(record.validation_attempts, limit=limit))
    detail_count = 0
    for stage in DETAIL_STAGE_ORDER:
        detail = record.stage_details.get(stage)
        if detail is None:
            continue
        chunks.extend(_format_detail_stage_chunks(stage, detail, limit=limit))
        detail_count += 1
    if detail_count == 0:
        chunks.append("到達済みstageに表示可能なPositive / Negative情報はありません。")
    return chunks
