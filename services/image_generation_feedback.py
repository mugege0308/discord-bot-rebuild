"""Self-improvement Phase 1: recording and reading structured image
generation feedback.

This module is deliberately read/write only - it never adjusts a prompt,
character dictionary entry, LoRA choice, or model based on what it reads.
Aggregation functions here return raw counts for a human (or a later,
separately-approved phase) to look at; they never make an automatic
decision. See services/selection_preference.py and services/prompt_preference.py
for the (also human-gated, mode-controlled) systems that already do apply
past image_ratings signal to generation - image_generation_feedback is a new,
separate signal and does not feed either of them.

DB access itself (schema, CRUD, validation of raw values) lives in
services/generation_history.py, matching how services/selection_preference.py
depends on generation_history.get_rated_prompt_rows() rather than opening
sqlite3 directly. This module adds the parts that are specific to feedback:
building the generation_snapshot_json payload, and aggregating stored
feedback rows in Python (the same approach generation_history.get_generation_stats
already uses instead of complex SQL, appropriate at this bot's data scale).
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from services.character_tag_dictionary import resolve_all_character_tags
from services.generation_history import (
    ALLOWED_ISSUE_CATEGORIES,
    FeedbackImageTarget,
    GenerationRequestDetail,
    ImageFeedbackRecord,
    list_all_feedback_rows,
    upsert_image_feedback,
)
from services.image_generation_profiles import get_image_generation_profile


logger = logging.getLogger("secretary-bot.image_generation_feedback")

# Used to bucket feedback whose request_text did not resolve to any known
# character (e.g. a background/scenery-only request) in the per-character
# summary, so those rows are still visible rather than silently dropped.
UNKNOWN_CHARACTER_LABEL = "(未特定)"

GENERATION_SNAPSHOT_SCHEMA_VERSION = 1

# Cap applied to get_accepted_generation_snapshots - a future consumer must
# ask for a bounded amount of data, matching every read function in
# generation_history.py.
DEFAULT_ACCEPTED_SNAPSHOT_LIMIT = 50


def _json_safe(value: Any) -> Any:
    """Fallback for json.dumps(default=...): stringifies anything that is
    not natively JSON-serializable (e.g. a stray Path or Enum) instead of
    raising, so a snapshot build can never crash a feedback save over one
    unexpected field.
    """
    return str(value)


def _character_resolutions_snapshot(request_text: str) -> list[dict[str, Any]]:
    try:
        resolutions = resolve_all_character_tags(request_text)
    except Exception:
        logger.exception("Character tag resolution failed while building a feedback snapshot; recording none.")
        return []
    return [
        {
            "character_tag": resolution.character_tag,
            "display_name": resolution.display_name,
            "series_tag": resolution.series_tag,
            "series_name": resolution.series_name,
            "match_type": resolution.match_type,
            "signature_tags": list(resolution.signature_tags),
        }
        for resolution in resolutions
    ]


def _model_profile_snapshot(model_key: str) -> dict[str, Any] | None:
    try:
        profile = get_image_generation_profile(model_key)
    except ValueError:
        return None
    return {
        "model_key": profile.model_key,
        "display_name": profile.display_name,
        "checkpoint": profile.checkpoint,
        "default_width": profile.width,
        "default_height": profile.height,
        "default_steps": profile.steps,
        "default_cfg": profile.cfg,
        "default_sampler": profile.sampler,
        "default_scheduler": profile.scheduler,
        "default_denoise": profile.denoise,
    }


def build_generation_snapshot(request: GenerationRequestDetail, image: FeedbackImageTarget) -> dict[str, Any]:
    """Builds the JSON-safe dict stored as image_generation_feedback.generation_snapshot_json.

    Only fields already present in generation_requests/generation_images (or
    deterministically derivable from them, such as the model profile lookup
    and the character-dictionary resolution below) are included. Nothing is
    invented: LoRA, IP-Adapter, and denoise are not tracked anywhere in this
    codebase today (services/sd_control.GeneratedImage has no such fields),
    so they are recorded as explicit null rather than guessed - a later
    change that starts tracking them only needs to fill these keys in, not
    add them.

    character_tags is a best-effort addition, not a verbatim historical
    record: generation_requests does not persist which character(s) were
    resolved at generation time, only the final request_text/positive
    prompt. This re-resolves request_text against the character dictionary
    as it exists right now (i.e. at feedback-save time) and says so
    explicitly in the "resolved_at" field, so a future reader does not
    mistake it for what the dictionary looked like when the image was
    actually generated.
    """
    return {
        "schema_version": GENERATION_SNAPSHOT_SCHEMA_VERSION,
        "request": {
            "request_id": request.id,
            "created_at": request.created_at,
            "user_id": request.user_id,
            "request_text": request.request_text,
            "reference_id": request.reference_id,
            "model_key": request.model_key,
            "parent_request_id": request.parent_request_id,
            "regeneration_type": request.regeneration_type,
            "custom_instruction": request.custom_instruction,
            "selection_used": request.selection_used,
            "selection_message": request.selection_message,
        },
        "prompt": {
            "positive": request.positive,
            "negative": request.negative,
            "summary": request.summary,
            "assumptions": list(request.assumptions),
        },
        "image": {
            "image_id": image.image_id,
            "image_index": image.image_index,
            "image_path": image.image_path,
            "seed": image.seed,
            "checkpoint_name": image.checkpoint_name,
            "sampler_name": image.sampler_name,
            "scheduler": image.scheduler,
            "steps": image.steps,
            "cfg": image.cfg,
            "width": image.width,
            "height": image.height,
            "was_selected_by_ai": image.was_selected_by_ai,
            # Not tracked anywhere in this codebase yet - see docstring.
            "denoise": None,
        },
        # Not tracked anywhere in this codebase yet - see docstring.
        "lora": None,
        "ip_adapter": None,
        "model_profile": _model_profile_snapshot(request.model_key),
        "character_tags": {
            "resolved_at": "feedback_save_time",
            "note": (
                "request_text をキャラクター辞書（services/character_tag_dictionary）で"
                "評価保存時点に解決した結果です。生成時点の辞書とは異なる場合があります。"
            ),
            "resolutions": _character_resolutions_snapshot(request.request_text),
        },
    }


def _dump_snapshot(snapshot: dict[str, Any]) -> str:
    try:
        return json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=_json_safe)
    except Exception:
        logger.exception("Failed to serialize generation snapshot; recording an empty one instead.")
        return json.dumps({"schema_version": GENERATION_SNAPSHOT_SCHEMA_VERSION, "error": "snapshot_serialization_failed"})


def save_image_feedback(
    *,
    request: GenerationRequestDetail,
    image: FeedbackImageTarget,
    user_id: int,
    guild_id: int | None,
    channel_id: int | None,
    message_id: int | None,
    accepted: bool,
    issue_categories: Sequence[str] = (),
    comment: str | None,
) -> bool:
    """Builds this image's generation snapshot and persists one user's
    feedback for it in a single call - the entry point the Discord UI layer
    should use instead of touching generation_history.upsert_image_feedback
    directly, so snapshot construction and validation always happen
    together. Returns True only when stored; never raises (delegates to
    upsert_image_feedback, which follows the same convention).
    """
    snapshot = build_generation_snapshot(request, image)
    return upsert_image_feedback(
        image_id=image.image_id,
        image_index=image.image_index,
        image_path=image.image_path,
        user_id=user_id,
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=message_id,
        accepted=accepted,
        issue_categories=issue_categories,
        comment=comment,
        generation_snapshot_json=_dump_snapshot(snapshot),
    )


@dataclass(frozen=True)
class CharacterFeedbackSummary:
    """Phase 1 aggregate for one character - raw counts only, no derived
    recommendation. A row's problem-category counters are incremented once
    per category it was tagged with (a row can add to more than one, since
    issue_categories is multi-select); accepted and total are each
    incremented at most once per row.
    """

    character_tag: str
    display_name: str
    total: int
    accepted: int
    face_mismatch: int
    hair_mismatch: int
    outfit_mismatch: int
    composition_mismatch: int
    low_quality: int
    other: int

    @property
    def acceptance_rate(self) -> float | None:
        return None if self.total == 0 else self.accepted / self.total


def _characters_in_row(row: ImageFeedbackRecord) -> list[tuple[str, str]]:
    """Returns [(character_tag, display_name), ...] resolved for one
    feedback row from its own frozen snapshot (not a fresh dictionary
    lookup - see build_generation_snapshot's docstring on why the snapshot
    is the source of truth for what a given row is "about"). A row with no
    resolved character is reported as [(UNKNOWN_CHARACTER_LABEL, UNKNOWN_CHARACTER_LABEL)]
    so it is still counted somewhere instead of silently vanishing from the
    summary.
    """
    try:
        resolutions = row.generation_snapshot.get("character_tags", {}).get("resolutions", [])
    except AttributeError:
        resolutions = []
    if not resolutions:
        return [(UNKNOWN_CHARACTER_LABEL, UNKNOWN_CHARACTER_LABEL)]
    seen: dict[str, str] = {}
    for entry in resolutions:
        tag = entry.get("character_tag")
        name = entry.get("display_name") or tag
        if tag:
            seen[tag] = name
    return list(seen.items()) or [(UNKNOWN_CHARACTER_LABEL, UNKNOWN_CHARACTER_LABEL)]


def get_character_feedback_summary(limit: int | None = None) -> tuple[CharacterFeedbackSummary, ...]:
    """Aggregates every stored feedback row (across all users) by character,
    for a human to review - never used to change generation automatically.

    A row that resolves to more than one character (a multi-character
    request) adds to every one of those characters' totals, so per-character
    totals can sum to more than the number of feedback rows scanned; this
    matches how the row's own snapshot recorded the request, and is
    documented here rather than picking one character arbitrarily.
    """
    rows = list_all_feedback_rows() if limit is None else list_all_feedback_rows(limit)
    buckets: dict[str, CharacterFeedbackSummary] = {}
    for row in rows:
        for character_tag, display_name in _characters_in_row(row):
            current = buckets.get(character_tag)
            if current is None:
                current = CharacterFeedbackSummary(
                    character_tag=character_tag,
                    display_name=display_name,
                    total=0,
                    accepted=0,
                    face_mismatch=0,
                    hair_mismatch=0,
                    outfit_mismatch=0,
                    composition_mismatch=0,
                    low_quality=0,
                    other=0,
                )
            buckets[character_tag] = CharacterFeedbackSummary(
                character_tag=current.character_tag,
                display_name=current.display_name,
                total=current.total + 1,
                accepted=current.accepted + (1 if row.accepted else 0),
                face_mismatch=current.face_mismatch + (1 if "face_mismatch" in row.issue_categories else 0),
                hair_mismatch=current.hair_mismatch + (1 if "hair_mismatch" in row.issue_categories else 0),
                outfit_mismatch=current.outfit_mismatch + (1 if "outfit_mismatch" in row.issue_categories else 0),
                composition_mismatch=current.composition_mismatch
                + (1 if "composition_mismatch" in row.issue_categories else 0),
                low_quality=current.low_quality + (1 if "low_quality" in row.issue_categories else 0),
                other=current.other + (1 if "other" in row.issue_categories else 0),
            )
    return tuple(
        sorted(buckets.values(), key=lambda summary: (-summary.total, summary.character_tag))
    )


@dataclass(frozen=True)
class FailureCategoryCount:
    category: str
    count: int


def get_common_failure_categories(character_tag: str | None = None) -> tuple[FailureCategoryCount, ...]:
    """Counts how many stored feedback rows carry each issue category,
    across all users, optionally restricted to rows whose snapshot resolved
    the given character_tag. Every known category is included even at zero,
    so a caller cannot mistake "not reported yet" for "no such category" -
    and, per the Phase 1 scope, this returns counts only; it never ranks,
    recommends, or triggers anything.
    """
    counts = {category: 0 for category in ALLOWED_ISSUE_CATEGORIES}
    for row in list_all_feedback_rows():
        if character_tag is not None:
            row_tags = {tag for tag, _ in _characters_in_row(row)}
            if character_tag not in row_tags:
                continue
        for category in row.issue_categories:
            if category in counts:
                counts[category] += 1
    return tuple(
        FailureCategoryCount(category=category, count=counts[category]) for category in ALLOWED_ISSUE_CATEGORIES
    )


def get_accepted_generation_snapshots(limit: int = DEFAULT_ACCEPTED_SNAPSHOT_LIMIT) -> list[dict[str, Any]]:
    """Returns the stored generation_snapshot for every accepted=True
    feedback row, most recently updated first, capped at `limit` - the
    read API a later, separately-approved self-improvement phase would use
    to look at "what generation conditions were adopted", without this
    module itself drawing any conclusion from them.
    """
    rows = list_all_feedback_rows()
    accepted = [row for row in rows if row.accepted]
    return [row.generation_snapshot for row in accepted[: max(0, limit)]]
