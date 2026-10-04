"""Safe Danbooru tag search for the Gallery generation form.

English queries are resolved locally from the read-only Danbooru snapshot.
Japanese queries may be translated into a few search terms, but every term
and every returned candidate is still validated against that snapshot.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from services.danbooru_tag_store import DanbooruTagCandidate, DanbooruTagStore
from services.prompt_generator import post_prompt_designer_completion


TAG_SEARCH_DEFAULT_LIMIT = 20
TAG_SEARCH_MAX_LIMIT = 50
TAG_SEARCH_MAX_QUERY_LENGTH = 200
TAG_SEARCH_MAX_TERMS = 5
TAG_SEARCH_WARNING = "日本語のタグ解釈を利用できません。英語タグで検索してください。"


class TagSearchError(ValueError):
    """A sanitized tag-search validation or availability error."""


def _is_japanese_or_non_ascii(query: str) -> bool:
    return any(ord(character) > 127 for character in query)


def _public_candidate(candidate: DanbooruTagCandidate) -> dict[str, object]:
    match_type = candidate.match_type
    if match_type.startswith("alias"):
        matched_by = "alias"
    elif match_type == "exact":
        matched_by = "canonical"
    else:
        matched_by = "search"
    return {
        "tag": candidate.name,
        "category": candidate.category,
        "post_count": candidate.post_count,
        "matched_by": matched_by,
    }


def _parse_terms(raw: str) -> tuple[str, ...]:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty translation")
    # The translator contract is JSON-only.  Accept a fenced JSON object as a
    # defensive parser convenience, but never expose the raw response.
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
        if text.casefold().startswith("json"):
            text = text[4:].lstrip()
    value = json.loads(text)
    if not isinstance(value, dict) or not isinstance(value.get("terms"), list):
        raise ValueError("translation schema is invalid")
    terms: list[str] = []
    seen: set[str] = set()
    for item in value["terms"]:
        if not isinstance(item, str):
            continue
        term = re.sub(r"\s+", " ", item).strip()
        if not term or any(ord(character) > 127 for character in term):
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term[:TAG_SEARCH_MAX_QUERY_LENGTH])
        if len(terms) >= TAG_SEARCH_MAX_TERMS:
            break
    return tuple(terms)


def _translate_query(query: str, interpreter: Callable[..., str]) -> tuple[str, ...]:
    system = (
        "Return a JSON object only. Convert the Japanese image-tag phrase into "
        "at most 5 short English Danbooru search terms. Do not invent tags, "
        "explain, or generate an image prompt. Exact schema: {\"terms\":[\"...\"]}."
    )
    raw = interpreter(system, query, temperature=0.0, max_tokens=256)
    return _parse_terms(raw)


def search_tag_candidates(
    query: str,
    *,
    limit: int = TAG_SEARCH_DEFAULT_LIMIT,
    store: DanbooruTagStore | None = None,
    interpreter: Callable[..., str] | None = None,
) -> dict[str, object]:
    clean_query = str(query or "").strip()
    if len(clean_query) > TAG_SEARCH_MAX_QUERY_LENGTH:
        raise TagSearchError("検索語が長すぎます。")
    try:
        requested_limit = int(limit)
    except (TypeError, ValueError) as exc:
        raise TagSearchError("limitが不正です。") from exc
    if requested_limit < 1:
        raise TagSearchError("limitは1以上で指定してください。")
    safe_limit = min(requested_limit, TAG_SEARCH_MAX_LIMIT)
    if not clean_query:
        return {"query": "", "interpreted_terms": [], "results": []}

    tag_store = store or DanbooruTagStore()
    if _is_japanese_or_non_ascii(clean_query):
        translate = interpreter or post_prompt_designer_completion
        try:
            terms = _translate_query(clean_query, translate)
        except Exception:  # noqa: BLE001 - translator failure is a safe empty result
            return {
                "query": clean_query,
                "interpreted_terms": [],
                "results": [],
                "warning": TAG_SEARCH_WARNING,
            }
    else:
        terms = (clean_query,)

    results: list[dict[str, object]] = []
    matched_terms: list[str] = []
    seen_ids: set[int] = set()
    for term in terms:
        candidates = tag_store.search(
            term,
            limit=safe_limit,
            allowed_categories=("general",),
            include_deprecated=False,
        )
        term_had_match = False
        for candidate in candidates:
            term_had_match = True
            if candidate.tag_id in seen_ids:
                continue
            seen_ids.add(candidate.tag_id)
            results.append(_public_candidate(candidate))
            if len(results) >= safe_limit:
                break
        if term_had_match:
            matched_terms.append(term)
        if len(results) >= safe_limit:
            break
    response: dict[str, object] = {
        "query": clean_query,
        "interpreted_terms": matched_terms,
        "results": results,
    }
    return response


__all__ = [
    "TAG_SEARCH_DEFAULT_LIMIT",
    "TAG_SEARCH_MAX_LIMIT",
    "TAG_SEARCH_MAX_QUERY_LENGTH",
    "TAG_SEARCH_MAX_TERMS",
    "TAG_SEARCH_WARNING",
    "TagSearchError",
    "search_tag_candidates",
]
