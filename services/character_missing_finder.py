"""/character_missing: known-but-currently-unresolved character alias scan.

## What this module does, and deliberately does NOT do

`generation_requests` (services/generation_history.py) has no
character_name / resolved_character / character_tag / series_tag column -
only `request_text` (the user's own Japanese wording), plus LM Studio's own
output (positive/negative/summary/assumptions_json). Character-tag
resolution (services/character_tag_dictionary.py) happens once, transiently,
while building the Prompt Designer system prompt
(services/shogou_prompt_designer.py's `_append_known_character_context`) -
its result is logged at INFO for characters that DID resolve, but a request
where resolution found nothing leaves no trace anywhere. There is therefore
no structured, already-verified signal for "the user meant a character name
here, and it wasn't recognized" for a genuinely novel/unregistered name.

Given that, this module implements only the one candidate source that is
safe (not a guess): a character ALREADY in the dictionary
(character_tag_dictionary.find_blocked_alias_mentions) whose ambiguous_alias
appears in request_text without its series also being mentioned - e.g. a
user typing "スズナ" without "ポケモン", where "suzuna"/"スズナ" is already a
curated dictionary entry that only resolves when its series is co-mentioned.
"Did this resolve or not" is the exact same ground truth
resolve_character_tags itself uses, not a separate heuristic.

It does NOT attempt to detect a wholly novel/never-registered character name
(e.g. "羽那" as a partial reading of "鈴木羽那", or "七海" alone) from free
text. Recognizing those reliably needs either a curated seed list this
dictionary doesn't have yet, or genuine NLP guesswork this codebase's own
"推測精度の低い自動抽出を勝手に本実装しない" ground rule rules out. See this
module's report in the assistant's final summary for what history fields
would remove this limitation (e.g. storing resolve_all_character_tags'
*attempted* aliases, not just its successful matches).
"""

import logging
from dataclasses import dataclass

from services.character_tag_dictionary import find_blocked_alias_mentions
from services.generation_history import RequestTextRow, list_request_texts


logger = logging.getLogger("secretary-bot.character_missing_finder")

DEFAULT_CANDIDATE_LIMIT = 20
# Bounds how much of any one user's raw request_text ever leaves this module,
# even for the permission-gated /character_missing admin view - "ユーザー入力
# 全文を不用意に公開しない".
SAMPLE_REQUEST_TEXT_MAX_LENGTH = 60


@dataclass(frozen=True)
class MissingCandidate:
    """Aggregated evidence for one known-but-currently-unresolved character
    alias, across every scanned generation_requests row.

    `sample_request_text` is truncated to SAMPLE_REQUEST_TEXT_MAX_LENGTH and
    is the most recent occurrence's request_text, not a concatenation of
    every occurrence - only one bounded sample is ever kept.
    """

    character_tag: str
    display_name: str
    series_tag: str
    series_name: str
    matched_alias: str
    occurrence_count: int
    last_used_at: str
    sample_request_text: str
    reason: str


def _truncate_sample(text: str, limit: int = SAMPLE_REQUEST_TEXT_MAX_LENGTH) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit].rstrip() + "…"


def find_missing_character_candidates(
    *,
    rows: list[RequestTextRow] | None = None,
    limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> list[MissingCandidate]:
    """Scans generation history for dictionary characters whose
    ambiguous_alias was mentioned without their series, aggregates by
    character_tag, and returns the top `limit` candidates ordered by
    occurrence_count descending (ties broken by character_tag for a stable
    order).

    `rows` is exposed for testing (see module docstring's dependency-
    injection convention used throughout this codebase's other
    generation-history-derived services); live callers should omit it and
    let this read generation_history.list_request_texts() itself.

    Never raises: any failure while reading history or scanning a row is
    logged and either yields an empty list (history read failure) or simply
    skips that row (single-row scan failure) - one malformed row can never
    take down the whole scan.
    """
    try:
        source_rows = list_request_texts() if rows is None else list(rows)
    except Exception:
        logger.exception("Failed to load request texts for /character_missing scan.")
        return []

    aggregates: dict[str, dict[str, object]] = {}
    for row in source_rows:
        try:
            mentions = find_blocked_alias_mentions(row.request_text)
        except Exception:
            logger.exception("Failed to scan request_id=%s for blocked alias mentions.", row.id)
            continue
        for mention in mentions:
            bucket = aggregates.get(mention.character_tag)
            if bucket is None:
                bucket = {
                    "display_name": mention.display_name,
                    "series_tag": mention.series_tag,
                    "series_name": mention.series_name,
                    "matched_alias": mention.matched_alias,
                    "count": 0,
                    "last_used_at": row.created_at,
                    "sample_request_text": row.request_text,
                }
                aggregates[mention.character_tag] = bucket
            bucket["count"] = int(bucket["count"]) + 1
            if row.created_at > str(bucket["last_used_at"]):
                bucket["last_used_at"] = row.created_at
                bucket["sample_request_text"] = row.request_text
                bucket["matched_alias"] = mention.matched_alias

    candidates = [
        MissingCandidate(
            character_tag=character_tag,
            display_name=str(bucket["display_name"]),
            series_tag=str(bucket["series_tag"]),
            series_name=str(bucket["series_name"]),
            matched_alias=str(bucket["matched_alias"]),
            occurrence_count=int(bucket["count"]),
            last_used_at=str(bucket["last_used_at"]),
            sample_request_text=_truncate_sample(str(bucket["sample_request_text"])),
            reason=(
                f"作品名「{bucket['series_name']}」が同時に指定されていないため、"
                "現在の辞書では解決できません。"
            ),
        )
        for character_tag, bucket in aggregates.items()
    ]
    candidates.sort(key=lambda candidate: (-candidate.occurrence_count, candidate.character_tag))
    return candidates[:limit]
