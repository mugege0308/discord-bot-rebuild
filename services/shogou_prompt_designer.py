import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from services.character_tag_dictionary import CharacterTagResolution, resolve_all_character_tags
from services.image_memory import build_image_memory_prompt_context
from services.image_generation_profiles import (
    DEFAULT_IMAGE_GENERATION_MODEL_KEY,
    get_image_generation_profile,
)
from services.memory import append_memory_to_system_prompt
from services.prompt_generator import (
    PromptGeneratorError,
    post_prompt_designer_completion as _post_chat_completion,
)
from services.prompt_tag_resolver import (
    PromptTagResolution,
    apply_prompt_tag_resolution,
    build_prompt_tag_instruction,
    resolve_prompt_tags,
)
from services.danbooru_tag_resolver import (
    DanbooruTagResolution,
    apply_danbooru_tag_resolution,
    build_danbooru_tag_instruction,
    resolve_danbooru_tags,
    should_search_generic_tags,
)
from services.danbooru_character_resolver import (
    DanbooruCharacterResolution,
    apply_danbooru_character_resolution,
    build_danbooru_character_instruction,
    resolve_danbooru_character_tags,
    should_search_character_tags,
)
from services.character_appearance_filter import (
    enforce_confirmed_character_appearance,
    filter_unrequested_character_appearance,
)
from services.prompt_positive_cleanup import (
    cleanup_positive_prompt,
    enforce_atomic_prompt_tags,
    normalize_negative_prompt,
    normalize_prompt_tag_key,
)
from services.prompt_semantic_grounding import ground_prompt_semantics
from services.verified_character_registry import VerifiedCharacterRegistry
from services.generate_ai_debug import (
    complete_generate_ai_debug,
    mark_generate_ai_stage,
    start_generate_ai_debug,
    store_generate_ai_character_resolution,
    store_generate_ai_danbooru_tag_resolution,
    store_generate_ai_prompt_tag_resolution,
    store_generate_ai_stage_detail,
)


logger = logging.getLogger("secretary-bot.shogou_prompt_designer")

INSTRUCTION_FILE = Path("shogou_chan_prompt_designer_instruction.md")
REQUIRED_KEYS = ("positive", "negative", "summary", "assumptions")
SHOGOU_RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "shogou_prompt_result", "schema": {"type": "object"}}}
PROMPT_RATING_AUTO = "auto"
PROMPT_RATING_VALUES = ("safe", "sensitive", "nsfw", "explicit")
PROMPT_RATING_CHOICES = (PROMPT_RATING_AUTO, *PROMPT_RATING_VALUES)
LEGACY_PROMPT_RATING = "questionable"
_ALL_PROMPT_RATING_TAGS = (*PROMPT_RATING_VALUES, LEGACY_PROMPT_RATING)
_PROMPT_RATING_PRIORITY = {rating: index for index, rating in enumerate(PROMPT_RATING_VALUES)}


@dataclass(frozen=True)
class ShogouPromptResult:
    positive: str = ""
    negative: str = ""
    summary: str = ""
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    blocked_reason: str = ""
    prompt_tag_dictionary_version: str = ""
    resolved_tag_ids: tuple[str, ...] = field(default_factory=tuple)
    resolved_prompt_tags: tuple[str, ...] = field(default_factory=tuple)
    unresolved_japanese_phrases: tuple[str, ...] = field(default_factory=tuple)
    danbooru_tag_db_version: str = ""
    danbooru_resolved_tag_ids: tuple[str, ...] = field(default_factory=tuple)
    danbooru_resolved_prompt_tags: tuple[str, ...] = field(default_factory=tuple)
    danbooru_character_tag_db_version: str = ""
    danbooru_character_resolved_tag_ids: tuple[str, ...] = field(default_factory=tuple)
    danbooru_character_resolved_tags: tuple[str, ...] = field(default_factory=tuple)
    danbooru_character_resolution: DanbooruCharacterResolution | None = None
    confirmed_character_appearance_tags: tuple[str, ...] = field(default_factory=tuple)
    character_appearance_suppressed_tags: tuple[str, ...] = field(default_factory=tuple)
    raw_llm_positive: str = ""
    raw_summary: str = ""
    raw_assumptions: tuple[str, ...] = field(default_factory=tuple)
    final_summary: str = ""
    user_grounded_tags: tuple[str, ...] = field(default_factory=tuple)
    removed_unrequested_semantic_tags: tuple[str, ...] = field(default_factory=tuple)
    semantic_grounding_result: tuple[str, ...] = field(default_factory=tuple)
    semantic_grounding_removed_assumptions: tuple[str, ...] = field(default_factory=tuple)
    pose_normalizations: tuple[str, ...] = field(default_factory=tuple)
    positive_cleanup_removed_tags: tuple[str, ...] = field(default_factory=tuple)
    candidate_final_tags: tuple[str, ...] = field(default_factory=tuple)
    tag_provenance: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    grounded_tags: tuple[str, ...] = field(default_factory=tuple)
    llm_only_tags: tuple[str, ...] = field(default_factory=tuple)
    removed_ungrounded_tags: tuple[str, ...] = field(default_factory=tuple)
    character_confirmation_required: bool = False
    character_unresolved_reason: str = ""

    @property
    def blocked(self) -> bool:
        return bool(self.blocked_reason)


_SUMMARY_OMIT_TAGS = frozenset(
    {
        "safe",
        "sensitive",
        "nsfw",
        "explicit",
        "masterpiece",
        "best quality",
        "high quality",
        "high score",
        "great score",
        "absurdres",
    }
)


def validate_requested_prompt_rating(requested_rating: str | None, *, is_adult: bool) -> str:
    rating = (requested_rating or PROMPT_RATING_AUTO).strip().casefold()
    if rating not in PROMPT_RATING_CHOICES:
        raise PromptGeneratorError("ratingはAuto / Safe / Sensitive / NSFW / Explicitから選んでください。")
    if not is_adult and rating not in (PROMPT_RATING_AUTO, "safe"):
        raise PromptGeneratorError("このratingはAdultモデルでのみ使用できます。")
    return rating


def extract_prompt_rating(positive: str) -> str | None:
    found: list[str] = []
    for raw_tag in positive.split(","):
        tag = raw_tag.strip().casefold()
        if tag == LEGACY_PROMPT_RATING:
            tag = "nsfw"
        if tag in PROMPT_RATING_VALUES:
            found.append(tag)
    if not found:
        return None
    return max(found, key=_PROMPT_RATING_PRIORITY.__getitem__)


def _build_grounded_summary(positive: str) -> str:
    """Build a summary from the same positive that will be sent to ComfyUI."""

    visible_tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in str(positive or "").split(","):
        tag = raw_tag.strip()
        key = normalize_prompt_tag_key(tag)
        if not tag or not key or key in seen or key in _SUMMARY_OMIT_TAGS:
            continue
        visible_tags.append(tag)
        seen.add(key)
    if not visible_tags:
        return "生成条件を確定しました。"
    return "生成条件: " + ", ".join(visible_tags)

def load_shogou_instruction(model_key: str = DEFAULT_IMAGE_GENERATION_MODEL_KEY) -> str:
    try:
        profile = get_image_generation_profile(model_key)
    except ValueError as exc:
        raise PromptGeneratorError(str(exc)) from exc
    instruction_file = profile.prompt_designer_instruction
    if not instruction_file.exists():
        raise PromptGeneratorError(f"{instruction_file} が見つかりません。")
    instruction = instruction_file.read_text(encoding="utf-8")
    addendum_file = profile.prompt_designer_addendum
    if addendum_file is None:
        return instruction
    if not addendum_file.exists():
        raise PromptGeneratorError(f"{addendum_file} が見つかりません。")
    addendum = addendum_file.read_text(encoding="utf-8")
    return instruction.rstrip() + "\n\n" + addendum.lstrip()


def _strip_think_blocks(text: str) -> str:
    # Only removes matched <think>...</think> pairs. If a </think> is never
    # found, the regex simply does not match, so the original text is
    # returned unchanged instead of being truncated at the opening tag.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _build_known_character_context(resolutions: Sequence[CharacterTagResolution]) -> str:
    lines = ["Known character:"]
    for resolution in resolutions:
        lines.append(f"- name: {resolution.display_name} ({resolution.series_name})")
        lines.append(f"- character tag: {resolution.character_tag}")
        lines.append(f"- series tag: {resolution.series_tag}")
        if resolution.signature_tags:
            signature_tags_text = ", ".join(resolution.signature_tags)
            lines.append(f"- confirmed appearance tags for {resolution.character_tag}: {signature_tags_text}")
    lines.append("- these tags must appear near the beginning of positive, right after the count tag (1girl/1boy/1other)")
    lines.append("- do not translate, split, abbreviate, replace, or generalize these tags")
    lines.append("- the name/series text above is identification context only; never add it as a separate tag itself")
    if any(resolution.signature_tags for resolution in resolutions):
        lines.append(
            "- confirmed appearance tags listed above must also be added to positive verbatim (each as its own "
            "tag, anywhere after the character/series tags), unless the user's own request explicitly specifies "
            "a conflicting appearance for that character"
        )
    if len(resolutions) > 1:
        lines.append(
            "- multiple known characters were detected in this request; keep them as distinct, separate "
            "characters and do not merge them into one"
        )
    return "\n".join(lines)


def _append_known_character_context(system_prompt: str, text: str) -> str:
    resolutions = resolve_all_character_tags(text)
    if not resolutions:
        return system_prompt
    for resolution in resolutions:
        logger.debug(
            "Character tag resolved: input_character=%s character_tag=%s series_tag=%s match_type=%s",
            resolution.matched_character_alias,
            resolution.character_tag,
            resolution.series_tag,
            resolution.match_type,
        )
    return system_prompt + "\n\n" + _build_known_character_context(resolutions)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_shogou_prompt_response(raw_response: str) -> ShogouPromptResult:
    cleaned = _strip_think_blocks(raw_response)
    cleaned = _strip_code_fence(cleaned)
    payload = _extract_json_object(cleaned)
    if payload is None:
        logger.warning("Shogou prompt JSON extraction failed.")
        return ShogouPromptResult(blocked_reason="LM Studioの返答からJSONを取り出せませんでした。")

    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        logger.warning("Shogou prompt response missing keys: %s", missing)
        return ShogouPromptResult(blocked_reason=f"必須キーが不足しています: {', '.join(missing)}")

    positive_raw = payload["positive"]
    negative_raw = payload["negative"]
    summary_raw = payload["summary"]
    assumptions_raw = payload["assumptions"]

    if not isinstance(positive_raw, str):
        return ShogouPromptResult(blocked_reason="positive が文字列ではありませんでした。")
    if not isinstance(negative_raw, str):
        return ShogouPromptResult(blocked_reason="negative が文字列ではありませんでした。")
    if not isinstance(summary_raw, str):
        return ShogouPromptResult(blocked_reason="summary が文字列ではありませんでした。")
    if not isinstance(assumptions_raw, list):
        return ShogouPromptResult(blocked_reason="assumptions が配列ではありませんでした。")

    positive = positive_raw.strip()
    negative = negative_raw.strip()
    if not positive:
        return ShogouPromptResult(blocked_reason="positive が空でした。")
    if not negative:
        return ShogouPromptResult(blocked_reason="negative が空でした。")

    assumptions = tuple(str(item).strip() for item in assumptions_raw if str(item).strip())

    logger.info(
        "Shogou prompt response parsed: positive_length=%s negative_length=%s summary_length=%s assumptions_count=%s",
        len(positive),
        len(negative),
        len(summary_raw.strip()),
        len(assumptions),
    )
    return ShogouPromptResult(positive=positive, negative=negative, summary=summary_raw.strip(), assumptions=assumptions)


def generate_shogou_prompt_result(
    user_input: str,
    user_id: int | None = None,
    reference_id: int | None = None,
    model_key: str = DEFAULT_IMAGE_GENERATION_MODEL_KEY,
    requested_rating: str = "auto",
    *,
    enable_generic_danbooru: bool = False,
) -> ShogouPromptResult:
    text = user_input.strip()
    if not text:
        raise PromptGeneratorError("内容を入力してください。例: `/generate_ai request:雨の日の小さな喫茶店`")
    if len(text) > 1000:
        raise PromptGeneratorError("少し長すぎます。1000文字以内にしてください。")

    profile = get_image_generation_profile(model_key)
    requested_rating = validate_requested_prompt_rating(
        requested_rating,
        is_adult=profile.is_adult,
    )
    trace_id = hashlib.sha256(
        f"{user_id}|{profile.model_key}|{text}".encode("utf-8", errors="replace")
    ).hexdigest()[:16]
    if user_id is not None:
        start_generate_ai_debug(
            user_id,
            trace_id,
            model_key=profile.model_key,
            model_name=profile.display_name,
            requested_rating="auto",
        )
        mark_generate_ai_stage(trace_id, "01_request")

    prompt_resolution: PromptTagResolution | None = None
    generic_resolution: DanbooruTagResolution | None = None
    character_resolution: DanbooruCharacterResolution | None = None
    manual_characters = resolve_all_character_tags(text)

    if profile.model_key == DEFAULT_IMAGE_GENERATION_MODEL_KEY:
        prompt_resolution = resolve_prompt_tags(text)
        if user_id is not None:
            store_generate_ai_prompt_tag_resolution(
                trace_id,
                dictionary_version=prompt_resolution.dictionary_version,
                resolved_tag_ids=prompt_resolution.resolved_tag_ids,
                resolved_prompt_tags=prompt_resolution.resolved_prompt_tags,
                unresolved_japanese_phrases=prompt_resolution.unresolved_japanese_phrases,
            )
        if enable_generic_danbooru:
            def _call_danbooru_llm_json(
                _system_prompt: str,
                user_prompt: str,
                _response_format: Mapping[str, Any],
                _max_tokens: int,
            ) -> str:
                # The resolver owns the allowlist; the LLM is only a query/
                # candidate-comparison helper, so plain JSON text is enough.
                try:
                    return _post_chat_completion(
                        _system_prompt,
                        user_prompt,
                        temperature=0.20,
                        response_format=_response_format,
                        max_tokens=_max_tokens,
                    )
                except TypeError:
                    # Keep compatibility with the older direct LM helper.
                    return _post_chat_completion(
                        _system_prompt,
                        user_prompt,
                        temperature=0.20,
                    )

            if not manual_characters:
                registry_matches = VerifiedCharacterRegistry().lookup(text, context_text=text)
                if len(registry_matches) == 1:
                    matched = registry_matches[0]
                    character_resolution = DanbooruCharacterResolution.from_verified_registry(
                        input_character_name=matched.matched_alias,
                        character=matched.character,
                        matched_alias=matched.matched_alias,
                    )
                elif len(registry_matches) > 1:
                    character_resolution = DanbooruCharacterResolution(
                        input_character_name=text[:160],
                        normalized_character_name=text.casefold().strip(),
                        unresolved_reason="ambiguous_verified_registry",
                        abstained=True,
                    )
                elif should_search_character_tags(text):
                    character_resolution = resolve_danbooru_character_tags(
                        text,
                        llm_json=_call_danbooru_llm_json,
                    )

            if character_resolution is not None and user_id is not None:
                store_generate_ai_character_resolution(
                    trace_id,
                    character_resolution.to_trace_dict(),
                )
            if character_resolution is not None and (
                character_resolution.confirmation_required
                or character_resolution.unresolved_reason
            ):
                if user_id is not None:
                    complete_generate_ai_debug(trace_id)
                return ShogouPromptResult(
                    summary="キャラクター確認待ち",
                    danbooru_character_tag_db_version=character_resolution.snapshot_version,
                    danbooru_character_resolved_tag_ids=tuple(
                        str(tag_id) for tag_id in character_resolution.selected_tag_ids
                    ),
                    danbooru_character_resolved_tags=character_resolution.selected_prompt_tags,
                    danbooru_character_resolution=character_resolution,
                    character_confirmation_required=character_resolution.confirmation_required,
                    character_unresolved_reason=character_resolution.unresolved_reason,
                )

            if should_search_generic_tags(text):
                generic_resolution = resolve_danbooru_tags(
                    text,
                    core_entries=prompt_resolution.entries if prompt_resolution is not None else (),
                    llm_json=_call_danbooru_llm_json,
                )
                if user_id is not None:
                    store_generate_ai_danbooru_tag_resolution(
                        trace_id,
                        generic_resolution.to_trace_dict(),
                    )

    logger.info("Shogou system prompt generation start: user_id=%s input_length=%s", user_id, len(text))
    system_prompt = append_memory_to_system_prompt(
        load_shogou_instruction(model_key), user_id, text, mode="prompt"
    )
    if requested_rating != PROMPT_RATING_AUTO:
        system_prompt += (
            f"\n\nRequested rating: {requested_rating}. Use only this rating tag in positive; "
            "do not add or intensify content to justify it."
        )
    if user_id is not None:
        try:
            reference_context = build_image_memory_prompt_context(user_id, text, reference_id)
        except ValueError as exc:
            raise PromptGeneratorError(str(exc)) from exc
        if reference_context:
            system_prompt += "\n\n" + reference_context
    system_prompt = _append_known_character_context(system_prompt, text)
    if prompt_resolution is not None:
        system_prompt += "\n\n" + build_prompt_tag_instruction(prompt_resolution)
    if generic_resolution is not None:
        system_prompt += "\n\n" + build_danbooru_tag_instruction(generic_resolution)
    if character_resolution is not None:
        system_prompt += "\n\n" + build_danbooru_character_instruction(character_resolution)
    system_prompt = system_prompt.rstrip()
    if user_id is not None:
        mark_generate_ai_stage(trace_id, "02_prompt_build")

    try:
        raw_content = _post_chat_completion(
            system_prompt,
            text,
            temperature=0.35,
            response_format=SHOGOU_RESPONSE_FORMAT,
            max_tokens=512,
        )
    except TypeError:
        raw_content = _post_chat_completion(system_prompt, text, temperature=0.35)
    if user_id is not None:
        mark_generate_ai_stage(trace_id, "03_lm_studio")
        store_generate_ai_stage_detail(trace_id, "03_lm_studio", positive=raw_content)
    result = parse_shogou_prompt_response(raw_content)
    if result.blocked:
        if user_id is not None:
            complete_generate_ai_debug(trace_id)
        return result

    verified_prompt_tags = (
        tuple(prompt_resolution.resolved_prompt_tags) if prompt_resolution is not None else ()
    ) + (
        tuple(generic_resolution.selected_prompt_tags) if generic_resolution is not None else ()
    )
    character_prompt_tags = (
        tuple(character_resolution.selected_prompt_tags)
        if character_resolution is not None
        else ()
    )
    confirmed_appearance_tags = tuple(
        dict.fromkeys(
            tag
            for resolution in manual_characters
            for tag in resolution.signature_tags
            if str(tag or "").strip()
        )
    )
    confirmed_character_identity_tags = tuple(
        dict.fromkeys(
            tag
            for resolution in manual_characters
            for tag in (resolution.character_tag, resolution.series_tag)
            if str(tag or "").strip()
        )
    )
    authoritative_character_tags = character_prompt_tags + confirmed_appearance_tags
    authoritative_prompt_tags = verified_prompt_tags + authoritative_character_tags
    authoritative_tag_sources: dict[str, str] = {
        tag: "core_resolver"
        for tag in (tuple(prompt_resolution.resolved_prompt_tags) if prompt_resolution else ())
    }
    authoritative_tag_sources.update(
        {
            tag: "general_resolver"
            for tag in (tuple(generic_resolution.selected_prompt_tags) if generic_resolution else ())
        }
    )
    authoritative_tag_sources.update(
        {tag: "confirmed_character" for tag in character_prompt_tags}
    )
    authoritative_tag_sources.update(
        {tag: "confirmed_character_appearance" for tag in confirmed_appearance_tags}
    )
    initial_cleanup = cleanup_positive_prompt(
        result.positive,
        canonical_prompt_tags=authoritative_prompt_tags,
        user_input=text,
    )
    semantic_grounding = (
        ground_prompt_semantics(
            initial_cleanup.positive,
            user_input=text,
            authoritative_prompt_tags=authoritative_prompt_tags,
            authoritative_tag_sources=authoritative_tag_sources,
            assumptions=result.assumptions,
            defer_character_appearance=(
                character_resolution is not None
                and character_resolution.has_selected_tag
            ),
            strict_unknown=True,
        )
        if prompt_resolution is not None
        else None
    )
    semantic_positive = (
        semantic_grounding.positive
        if semantic_grounding is not None
        else result.positive
    )
    appearance = filter_unrequested_character_appearance(
        semantic_positive,
        user_input=text,
        character_resolution=character_resolution,
        verified_prompt_tags=authoritative_prompt_tags,
    )
    final_positive = appearance.positive
    if prompt_resolution is not None:
        final_positive = apply_prompt_tag_resolution(final_positive, prompt_resolution)
    if generic_resolution is not None:
        final_positive = apply_danbooru_tag_resolution(final_positive, generic_resolution)
    if character_resolution is not None:
        final_positive = apply_danbooru_character_resolution(final_positive, character_resolution)
    confirmed_appearance = enforce_confirmed_character_appearance(
        final_positive,
        confirmed_appearance_tags,
        user_input=text,
        explicit_prompt_tags=verified_prompt_tags,
        identity_tags=confirmed_character_identity_tags,
    )
    final_positive = confirmed_appearance.positive
    final_positive = enforce_atomic_prompt_tags(final_positive, character_prompt_tags)
    final_semantic_grounding = (
        ground_prompt_semantics(
            final_positive,
            user_input=text,
            authoritative_prompt_tags=authoritative_prompt_tags,
            authoritative_tag_sources=authoritative_tag_sources,
            assumptions=(
                semantic_grounding.assumptions
                if semantic_grounding is not None
                else result.assumptions
            ),
            defer_character_appearance=(
                character_resolution is not None
                and character_resolution.has_selected_tag
            ),
            strict_unknown=True,
        )
        if semantic_grounding is not None
        else None
    )
    if final_semantic_grounding is not None:
        final_positive = final_semantic_grounding.positive
    final_confirmed_appearance = enforce_confirmed_character_appearance(
        final_positive,
        confirmed_appearance_tags,
        user_input=text,
        explicit_prompt_tags=verified_prompt_tags,
        identity_tags=confirmed_character_identity_tags,
    )
    final_positive = final_confirmed_appearance.positive
    final_positive = enforce_atomic_prompt_tags(final_positive, character_prompt_tags)
    cleanup = cleanup_positive_prompt(
        final_positive,
        canonical_prompt_tags=authoritative_prompt_tags,
        user_input=text,
    )
    result = replace(
        result,
        positive=cleanup.positive,
        negative=normalize_negative_prompt(result.negative),
        raw_llm_positive=result.positive,
        raw_summary=result.summary,
        raw_assumptions=result.assumptions,
        prompt_tag_dictionary_version=prompt_resolution.dictionary_version if prompt_resolution else "",
        resolved_tag_ids=prompt_resolution.resolved_tag_ids if prompt_resolution else (),
        resolved_prompt_tags=prompt_resolution.resolved_prompt_tags if prompt_resolution else (),
        unresolved_japanese_phrases=prompt_resolution.unresolved_japanese_phrases if prompt_resolution else (),
        danbooru_tag_db_version=generic_resolution.snapshot_version if generic_resolution else "",
        danbooru_resolved_tag_ids=tuple(str(tag_id) for tag_id in generic_resolution.selected_tag_ids) if generic_resolution else (),
        danbooru_resolved_prompt_tags=generic_resolution.selected_prompt_tags if generic_resolution else (),
        danbooru_character_tag_db_version=character_resolution.snapshot_version if character_resolution else "",
        danbooru_character_resolved_tag_ids=tuple(str(tag_id) for tag_id in character_resolution.selected_tag_ids) if character_resolution else (),
        danbooru_character_resolved_tags=character_resolution.selected_prompt_tags if character_resolution else (),
        danbooru_character_resolution=character_resolution,
        confirmed_character_appearance_tags=confirmed_appearance_tags,
        character_appearance_suppressed_tags=tuple(
            dict.fromkeys(
                appearance.removed_tags
                + confirmed_appearance.removed_tags
                + final_confirmed_appearance.removed_tags
            )
        ),
        positive_cleanup_removed_tags=tuple(
            dict.fromkeys(initial_cleanup.removed_tags + cleanup.removed_tags)
        ),
    )
    semantic_passes = tuple(
        grounding
        for grounding in (semantic_grounding, final_semantic_grounding)
        if grounding is not None
    )
    semantic_removed_tags = tuple(
        dict.fromkeys(
            tag
            for grounding in semantic_passes
            for tag in grounding.removed_tags
        )
    )
    semantic_removed_assumptions = tuple(
        dict.fromkeys(
            assumption
            for grounding in semantic_passes
            for assumption in grounding.removed_assumptions
        )
    )
    semantic_user_grounded_tags = tuple(
        dict.fromkeys(
            tag
            for grounding in semantic_passes
            for tag in grounding.user_grounded_tags
        )
    )
    semantic_pose_normalizations = tuple(
        dict.fromkeys(
            item
            for grounding in semantic_passes
            for item in grounding.pose_normalizations
        )
    )
    semantic_grounding_result = tuple(
        dict.fromkeys(
            [
                f"{tag} -> {reason}"
                for grounding in semantic_passes
                for tag, reason in grounding.removed_reasons
            ]
            + [
                f"assumption:{assumption} -> unrequested_semantic"
                for assumption in semantic_removed_assumptions
            ]
        )
    )
    final_assumptions = (
        final_semantic_grounding.assumptions
        if final_semantic_grounding is not None
        else (
            semantic_grounding.assumptions
            if semantic_grounding is not None
            else result.assumptions
        )
    )
    summary_was_rebuilt = bool(
        semantic_removed_tags
        or semantic_removed_assumptions
        or initial_cleanup.removed_tags
        or cleanup.removed_tags
    )
    final_summary = (
        _build_grounded_summary(cleanup.positive)
        if summary_was_rebuilt
        else result.summary
    )
    semantic_tag_provenance = tuple(
        dict.fromkeys(
            f"{tag} -> {source}"
            for grounding in semantic_passes
            for tag, source in grounding.tag_provenance
        )
    )
    semantic_grounded_tags = tuple(
        dict.fromkeys(
            tag for grounding in semantic_passes for tag in grounding.grounded_tags
        )
    )
    semantic_llm_only_tags = tuple(
        dict.fromkeys(
            tag for grounding in semantic_passes for tag in grounding.llm_only_tags
        )
    )
    semantic_removed_ungrounded_tags = tuple(
        dict.fromkeys(
            tag for grounding in semantic_passes for tag in grounding.removed_ungrounded_tags
        )
    )
    semantic_candidate_final_tags = tuple(
        dict.fromkeys(
            tag for grounding in semantic_passes for tag in grounding.candidate_final_tags
        )
    )
    result = replace(
        result,
        summary=final_summary,
        final_summary=final_summary,
        user_grounded_tags=semantic_user_grounded_tags,
        removed_unrequested_semantic_tags=semantic_removed_tags,
        semantic_grounding_result=semantic_grounding_result,
        semantic_grounding_removed_assumptions=semantic_removed_assumptions,
        pose_normalizations=semantic_pose_normalizations,
        candidate_final_tags=semantic_candidate_final_tags,
        tag_provenance=semantic_tag_provenance,
        grounded_tags=semantic_grounded_tags,
        llm_only_tags=semantic_llm_only_tags,
        removed_ungrounded_tags=semantic_removed_ungrounded_tags,
        assumptions=final_assumptions,
    )
    if user_id is not None:
        store_generate_ai_stage_detail(
            trace_id,
            "06_rating",
            positive=result.positive,
            appearance_removed_tags=result.character_appearance_suppressed_tags,
            cleanup_removed_tags=result.positive_cleanup_removed_tags,
            raw_llm_positive=result.raw_llm_positive,
            raw_summary=result.raw_summary,
            final_summary=result.final_summary,
            raw_assumptions=result.raw_assumptions,
            final_assumptions=result.assumptions,
            semantic_grounding_result=result.semantic_grounding_result,
            user_grounded_tags=result.user_grounded_tags,
            removed_unrequested_semantic_tags=result.removed_unrequested_semantic_tags,
            pose_normalizations=result.pose_normalizations,
        )
        mark_generate_ai_stage(trace_id, "06_rating", final_positive_tags=len(result.positive.split(",")))
        complete_generate_ai_debug(trace_id)
    logger.info(
        "Shogou postprocess complete: positive_length=%s negative_length=%s blocked=%s",
        len(result.positive),
        len(result.negative),
        result.blocked,
    )
    return result

def _build_composition_variation_instruction(
    parent_request_text: str,
    parent_positive: str,
    parent_negative: str,
    parent_summary: str,
    parent_assumptions: Sequence[str],
) -> str:
    return (
        "\n\n---\n"
        "これは既存の生成結果をもとにした「構図を変えて再生成」リクエストです。"
        "以下の元の生成条件をもとに、キャラクター・衣装・雰囲気・重要な属性、"
        "および元の色彩・明るさ・光の方向性（色温度や逆光の強さなど）はそのまま維持しつつ、"
        "構図・カメラ角度・ポーズ・画面内配置だけを変更した新しいpositive/negative/summary/assumptionsを作成してください。\n\n"
        f"元のリクエスト: {parent_request_text}\n"
        f"元のPositive: {parent_positive}\n"
        f"元のNegative: {parent_negative}\n"
        f"元のSummary: {parent_summary}\n"
        f"元のAssumptions: {json.dumps(list(parent_assumptions), ensure_ascii=False)}\n"
    )


def _build_custom_instruction_variation_instruction(
    parent_request_text: str,
    parent_positive: str,
    parent_negative: str,
    parent_summary: str,
    parent_assumptions: Sequence[str],
    custom_instruction: str,
) -> str:
    return (
        "\n\n---\n"
        "これは既存の生成結果をもとにした「追加指示付き再生成」リクエストです。"
        "元の生成方針をできるだけ維持し、追加指示に関係する部分のみを変更してください。"
        "追加指示と矛盾しない限り、元の色彩・ライティング・構図・雰囲気・衣装などは維持してください。\n\n"
        f"元のリクエスト: {parent_request_text}\n"
        f"元のPositive: {parent_positive}\n"
        f"元のNegative: {parent_negative}\n"
        f"元のSummary: {parent_summary}\n"
        f"元のAssumptions: {json.dumps(list(parent_assumptions), ensure_ascii=False)}\n\n"
        f"今回の追加指示: {custom_instruction}\n"
    )


def generate_shogou_custom_instruction_variation(
    parent_request_text: str,
    parent_positive: str,
    parent_negative: str,
    parent_summary: str,
    parent_assumptions: Sequence[str],
    custom_instruction: str,
    user_id: int | None = None,
    reference_id: int | None = None,
    model_key: str = DEFAULT_IMAGE_GENERATION_MODEL_KEY,
) -> ShogouPromptResult:
    """"指示を追加して再生成": reuses the same shogou instruction/system
    prompt as a normal /generate_ai call, but appends the parent request's
    positive/negative/summary/assumptions plus the user's own free-text
    instruction (from the regenerate modal), with an explicit instruction to
    keep the original generation policy and only change what the new
    instruction targets.
    """
    instruction_text = custom_instruction.strip()
    if not instruction_text:
        raise PromptGeneratorError("追加指示を入力してください。")
    if len(instruction_text) > 1000:
        raise PromptGeneratorError("追加指示は1000文字以内にしてください。")

    text = parent_request_text.strip() or "(元のリクエスト文なし)"
    logger.info("Shogou custom instruction variation generation start: user_id=%s", user_id)
    system_prompt = append_memory_to_system_prompt(load_shogou_instruction(model_key), user_id, text, mode="prompt")
    if user_id is not None:
        try:
            reference_context = build_image_memory_prompt_context(user_id, text, reference_id)
        except ValueError as exc:
            raise PromptGeneratorError(str(exc)) from exc
        if reference_context:
            system_prompt += "\n\n" + reference_context
    system_prompt = _append_known_character_context(system_prompt, text)
    system_prompt += _build_custom_instruction_variation_instruction(
        parent_request_text, parent_positive, parent_negative, parent_summary, parent_assumptions, instruction_text
    )
    logger.info(
        "Shogou custom instruction variation system prompt complete: user_id=%s length=%s", user_id, len(system_prompt)
    )

    raw_content = _post_chat_completion(system_prompt, text, temperature=0.35)
    logger.info("Shogou custom instruction variation postprocess start: raw_content_length=%s", len(raw_content))
    result = parse_shogou_prompt_response(raw_content)
    logger.info(
        "Shogou custom instruction variation postprocess complete: positive_length=%s negative_length=%s blocked=%s",
        len(result.positive),
        len(result.negative),
        result.blocked,
    )
    return result


def generate_shogou_composition_variation(
    parent_request_text: str,
    parent_positive: str,
    parent_negative: str,
    parent_summary: str,
    parent_assumptions: Sequence[str],
    user_id: int | None = None,
    reference_id: int | None = None,
    model_key: str = DEFAULT_IMAGE_GENERATION_MODEL_KEY,
) -> ShogouPromptResult:
    """Phase 1C "change_composition" regeneration: reuses the same shogou
    instruction/system prompt as a normal /generate_ai call, but appends the
    parent request's positive/negative/summary/assumptions plus an explicit
    instruction to keep character/clothing/mood and only vary composition.
    """
    text = parent_request_text.strip() or "(元のリクエスト文なし)"
    logger.info("Shogou composition variation generation start: user_id=%s", user_id)
    system_prompt = append_memory_to_system_prompt(load_shogou_instruction(model_key), user_id, text, mode="prompt")
    if user_id is not None:
        try:
            reference_context = build_image_memory_prompt_context(user_id, text, reference_id)
        except ValueError as exc:
            raise PromptGeneratorError(str(exc)) from exc
        if reference_context:
            system_prompt += "\n\n" + reference_context
    system_prompt = _append_known_character_context(system_prompt, text)
    system_prompt += _build_composition_variation_instruction(
        parent_request_text, parent_positive, parent_negative, parent_summary, parent_assumptions
    )
    logger.info("Shogou composition variation system prompt complete: user_id=%s length=%s", user_id, len(system_prompt))

    raw_content = _post_chat_completion(system_prompt, text, temperature=0.35)
    logger.info("Shogou composition variation postprocess start: raw_content_length=%s", len(raw_content))
    result = parse_shogou_prompt_response(raw_content)
    logger.info(
        "Shogou composition variation postprocess complete: positive_length=%s negative_length=%s blocked=%s",
        len(result.positive),
        len(result.negative),
        result.blocked,
    )
    return result
