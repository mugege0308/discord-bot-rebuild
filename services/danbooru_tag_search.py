"""Safe Danbooru tag search for the Gallery generation form.

English queries are resolved locally from the read-only Danbooru snapshot.
Japanese queries may be translated into a few search terms, but every term
and every returned candidate is still validated against that snapshot.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from threading import RLock
from typing import Callable, Sequence

from services.danbooru_tag_store import (
    DEFAULT_DANBOORU_DB_PATH,
    DanbooruTagCandidate,
    DanbooruTagStore,
)
from services.danbooru_ja_dictionary import (
    DanbooruJapaneseDictionary,
    get_default_dictionary,
)
from services.prompt_generator import post_prompt_designer_completion


TAG_SEARCH_DEFAULT_LIMIT = 20
TAG_SEARCH_MAX_LIMIT = 50
TAG_SEARCH_MAX_QUERY_LENGTH = 200
TAG_SEARCH_MAX_TERMS = 5
TAG_SEARCH_WARNING = "日本語のタグ解釈を利用できません。英語タグで検索してください。"
TAG_TRANSLATION_WARNING = "日本語訳を取得できませんでした。英語タグを表示します。"
TAG_TRANSLATION_CACHE_SIZE = 256
TAG_SEARCH_CATEGORIES = frozenset({"all", "general", "character", "copyright"})
TAG_SEARCH_DEFAULT_CATEGORY = "general"

_translation_cache: OrderedDict[str, str] = OrderedDict()
_translation_cache_lock = RLock()


class TagSearchError(ValueError):
    """A sanitized tag-search validation or availability error."""


def _is_japanese_or_non_ascii(query: str) -> bool:
    return any(ord(character) > 127 for character in query)


def _public_candidate(
    candidate: DanbooruTagCandidate,
    *,
    translation_ja: str | None = None,
    matched_by: str | None = None,
) -> dict[str, object]:
    if matched_by is None:
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
        "translation_ja": translation_ja,
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
        "at most 5 short English Danbooru search terms. Preserve meaning elements "
        "and useful multi-word phrases; do not reduce a phrase to one noun. "
        "For example, 頬に手 can produce hand on cheek, hand, cheek, and "
        "頬に両手 can produce hands on cheeks, both hands, cheek. Do not invent "
        "tags, explain, or generate an image prompt. Exact schema: "
        "{\"terms\":[\"...\"]}."
    )
    raw = interpreter(system, query, temperature=0.0, max_tokens=256)
    return _parse_terms(raw)


def _cache_key(tag: str) -> str:
    return re.sub(r"\s+", "_", str(tag or "").strip().casefold())


def clear_tag_translation_cache() -> None:
    """Clear the process-local translation cache (primarily for tests)."""

    with _translation_cache_lock:
        _translation_cache.clear()


def _cached_translations(tags: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    with _translation_cache_lock:
        for tag in tags:
            value = _translation_cache.get(_cache_key(tag))
            if value:
                _translation_cache.move_to_end(_cache_key(tag))
                result[tag] = value
    return result


def _store_translations(translations: dict[str, str]) -> None:
    with _translation_cache_lock:
        for tag, value in translations.items():
            key = _cache_key(tag)
            if not key or not value:
                continue
            _translation_cache[key] = value
            _translation_cache.move_to_end(key)
        while len(_translation_cache) > TAG_TRANSLATION_CACHE_SIZE:
            _translation_cache.popitem(last=False)


def _parse_translation_map(raw: str, tags: Sequence[str]) -> dict[str, str]:
    value = json.loads(str(raw or "").strip())
    if not isinstance(value, dict):
        raise ValueError("translation schema is invalid")
    requested = {_cache_key(tag): tag for tag in tags}
    result: dict[str, str] = {}
    for raw_tag, raw_label in value.items():
        if not isinstance(raw_tag, str) or not isinstance(raw_label, str):
            continue
        tag = requested.get(_cache_key(raw_tag))
        label = re.sub(r"\s+", " ", raw_label).strip()
        if tag and label:
            result[tag] = label[:120]
    return result


def _translate_result_tags(
    tags: Sequence[str],
    interpreter: Callable[..., str],
    dictionary: DanbooruJapaneseDictionary | None = None,
) -> tuple[dict[str, str], bool]:
    dictionary_labels: dict[str, str] = {}
    if dictionary is not None and dictionary.available:
        for tag in tags:
            label = dictionary.translation_for(tag)
            if label:
                dictionary_labels[tag] = label
    cached = _cached_translations(tags)
    translated = {**cached, **dictionary_labels}
    missing = [tag for tag in tags if tag not in translated]
    if not missing:
        return translated, False
    system = (
        "Return one JSON object only. Translate each supplied canonical Danbooru "
        "tag into a concise Japanese display label. Keep the exact tag keys, "
        "do not add keys, explanations, prompt text, or values for missing tags. "
        "The labels are display-only, not prompt tags."
    )
    user = json.dumps(missing, ensure_ascii=False, separators=(",", ":"))
    try:
        raw = interpreter(system, user, temperature=0.0, max_tokens=512)
        translated = _parse_translation_map(raw, missing)
    except Exception:  # noqa: BLE001 - translation failure must not hide results
        return translated, True
    _store_translations(translated)
    return translated, len(translated) < len(missing)


def search_tag_candidates(
    query: str,
    *,
    limit: int = TAG_SEARCH_DEFAULT_LIMIT,
    category: str = TAG_SEARCH_DEFAULT_CATEGORY,
    store: DanbooruTagStore | None = None,
    interpreter: Callable[..., str] | None = None,
    dictionary: DanbooruJapaneseDictionary | None = None,
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
    selected_category = str(category or TAG_SEARCH_DEFAULT_CATEGORY).strip().casefold()
    if selected_category not in TAG_SEARCH_CATEGORIES:
        raise TagSearchError("categoryが不正です。")
    safe_limit = min(requested_limit, TAG_SEARCH_MAX_LIMIT)
    if not clean_query:
        return {"query": "", "category": selected_category, "interpreted_terms": [], "results": []}

    tag_store = store or DanbooruTagStore()
    if dictionary is not None:
        ja_dictionary = dictionary
    elif tag_store.path.resolve() == DEFAULT_DANBOORU_DB_PATH.resolve():
        ja_dictionary = get_default_dictionary()
    else:
        ja_dictionary = None
    direct_dictionary_matches = ()
    search_source = "canonical"
    if _is_japanese_or_non_ascii(clean_query):
        if ja_dictionary is not None:
            direct_dictionary_matches = ja_dictionary.search(
                clean_query,
                category=selected_category,
                store=tag_store,
                limit=safe_limit,
            )
        if direct_dictionary_matches:
            terms = (clean_query,)
            search_source = "dictionary"
        else:
            translate = interpreter or post_prompt_designer_completion
            try:
                terms = _translate_query(clean_query, translate)
            except Exception:  # noqa: BLE001 - translator failure is a safe empty result
                return {
                    "query": clean_query,
                    "category": selected_category,
                    "interpreted_terms": [],
                    "results": [],
                    "search_source": "llm",
                    "lm_used": True,
                    "warning": TAG_SEARCH_WARNING,
                }
            search_source = "llm"
    else:
        terms = (clean_query,)

    results: list[DanbooruTagCandidate] = [item.candidate for item in direct_dictionary_matches]
    dictionary_labels = {item.candidate.name: item.translation_ja for item in direct_dictionary_matches if item.translation_ja}
    matched_terms: list[str] = []
    seen_ids: set[int] = set()
    for candidate in results:
        seen_ids.add(candidate.tag_id)
    if direct_dictionary_matches:
        matched_terms.append(clean_query)
    allowed_categories = None if selected_category == "all" else (selected_category,)
    if not direct_dictionary_matches:
        for term in terms:
            candidates = tag_store.search(
                term,
                limit=safe_limit,
                allowed_categories=allowed_categories,
                include_deprecated=False,
            )
            term_had_match = False
            for candidate in candidates:
                term_had_match = True
                if candidate.tag_id in seen_ids:
                    continue
                seen_ids.add(candidate.tag_id)
                results.append(candidate)
                if len(results) >= safe_limit:
                    break
            if term_had_match:
                matched_terms.append(term)
            if len(results) >= safe_limit:
                break
        if not results and ja_dictionary is not None:
            direct_dictionary_matches = ja_dictionary.search(
                clean_query,
                category=selected_category,
                store=tag_store,
                limit=safe_limit,
            )
            if direct_dictionary_matches:
                search_source = "dictionary"
                results = [item.candidate for item in direct_dictionary_matches]
                dictionary_labels = {
                    item.candidate.name: item.translation_ja
                    for item in direct_dictionary_matches
                    if item.translation_ja
                }
                seen_ids = {candidate.tag_id for candidate in results}
                matched_terms.append(clean_query)
        if direct_dictionary_matches and _is_japanese_or_non_ascii(clean_query):
            translations = dictionary_labels
            translation_failed = False
        else:
            translations, translation_failed = _translate_result_tags(
                [candidate.name for candidate in results],
                interpreter or post_prompt_designer_completion,
                ja_dictionary,
            )
    else:
        translations = dictionary_labels
        translation_failed = False
    response: dict[str, object] = {
        "query": clean_query,
        "category": selected_category,
        "interpreted_terms": matched_terms,
        "search_source": search_source,
        "lm_used": search_source == "llm",
        "results": [
            _public_candidate(
                candidate,
                translation_ja=translations.get(candidate.name),
                matched_by=("dictionary" if any(item.candidate.tag_id == candidate.tag_id for item in direct_dictionary_matches) else None),
            )
            for candidate in results
        ],
    }
    if translation_failed:
        response["warning"] = TAG_TRANSLATION_WARNING
    return response


__all__ = [
    "TAG_SEARCH_DEFAULT_LIMIT",
    "TAG_SEARCH_MAX_LIMIT",
    "TAG_SEARCH_MAX_QUERY_LENGTH",
    "TAG_SEARCH_MAX_TERMS",
    "TAG_SEARCH_WARNING",
    "TAG_TRANSLATION_WARNING",
    "TAG_TRANSLATION_CACHE_SIZE",
    "TAG_SEARCH_CATEGORIES",
    "TAG_SEARCH_DEFAULT_CATEGORY",
    "TagSearchError",
    "clear_tag_translation_cache",
    "search_tag_candidates",
]
