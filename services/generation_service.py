"""Shared, Discord-independent generation orchestration.

This module is intentionally a thin boundary around the generation services
that already exist in this repository.  It does not own Discord interactions,
Gallery HTTP, or ComfyUI workflow construction.  Those remain adapters around
the request/prepare/submit contract defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from services.appearance_resolver import (
    AppearanceResolution,
    appearance_tag_category,
    resolve_appearance_text,
)
from services.danbooru_tag_resolver import DanbooruTagResolution
from services.generation_variation import MAX_GENERATION_COUNT
from services.image_generation_profiles import (
    DEFAULT_IMAGE_GENERATION_MODEL_KEY,
    get_image_generation_profile,
)
from services.prompt_generator import PromptGeneratorError
from services.prompt_tag_resolver import (
    PromptTagResolution,
    apply_prompt_tag_resolution,
    resolve_prompt_tags,
)
from services.prompt_conflict_resolver import (
    PromptConflict,
    PromptFramingSemantics,
    PromptPeopleSemantics,
    PromptTagRecord,
    SuppressedPromptTag,
    detect_people_semantics,
    has_non_solo_constraint,
    merge_prompt_framing_semantics,
    resolve_prompt_tag_conflicts,
)
from services.sd_control import (
    GeneratedImage,
    LoraSelection,
    SdGenerationOptions,
    apply_lora_triggers,
    generate_images_with_metadata,
    get_negative_prompt,
    resolve_saved_lora_selections,
)
from services.shogou_prompt_designer import (
    PROMPT_RATING_AUTO,
    ShogouPromptResult,
    generate_shogou_prompt_result,
    validate_requested_prompt_rating,
)


PromptResolver = Callable[..., ShogouPromptResult]
AppearanceResolver = Callable[[str], "AppearanceResolution | Sequence[str]"]
LoraResolver = Callable[..., tuple[LoraSelection, ...]]
ImageGenerator = Callable[[SdGenerationOptions], Sequence[GeneratedImage]]


def _normalise_text(value: object) -> str:
    return str(value or "").strip()


def _split_tag_values(values: Sequence[str]) -> tuple[str, ...]:
    """Split list items on top-level commas without breaking weighted tags."""

    result: list[str] = []
    current: list[str] = []
    depths = {"(": 0, "[": 0, "{": 0}
    closing = {")": "(", "]": "[", "}": "{",
    }
    for raw_value in values:
        for char in str(raw_value or ""):
            if char in depths:
                depths[char] += 1
                current.append(char)
                continue
            if char in closing:
                opener = closing[char]
                if depths[opener] > 0:
                    depths[opener] -= 1
                current.append(char)
                continue
            if char == "," and not any(depths.values()):
                token = "".join(current).strip()
                if token:
                    result.append(token)
                current.clear()
                continue
            current.append(char)
        if current and len(values) > 1:
            # A list element is already a natural boundary.  The comma parser
            # above still handles the uncommon case where one item contains a
            # weighted expression.
            token = "".join(current).strip()
            if token:
                result.append(token)
            current.clear()
    token = "".join(current).strip()
    if token:
        result.append(token)
    return tuple(result)


def _tag_key(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().replace("_", " ").split())


def _dedupe_tags(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        tag = _normalise_text(raw)
        key = _tag_key(tag)
        if not tag or not key or key in seen:
            continue
        result.append(tag)
        seen.add(key)
    return tuple(result)


def _append_tags(positive: str, tags: Sequence[str]) -> str:
    """Append authoritative tags once while preserving the existing order."""

    tokens = [token.strip() for token in str(positive or "").split(",") if token.strip()]
    seen = {_tag_key(token) for token in tokens}
    for tag in tags:
        clean = _normalise_text(tag)
        key = _tag_key(clean)
        if clean and key and key not in seen:
            tokens.append(clean)
            seen.add(key)
    return ", ".join(tokens)


_SYSTEM_TAG_KEYS = frozenset(
    {
        "1girl",
        "1boy",
        "1other",
        "solo",
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
_RATING_TAG_KEYS = frozenset({"safe", "sensitive", "nsfw", "explicit"})
_QUALITY_TAG_KEYS = frozenset(
    {"masterpiece", "best quality", "high quality", "high score", "great score", "absurdres"}
)
_MULTI_SUBJECT_HINTS = (
    "複数",
    "二人",
    "ふたり",
    "3人",
    "三人",
    "group",
    "multiple",
    "couple",
    "two girls",
    "two boys",
)
_FEMALE_SUBJECT_HINTS = (
    "女の子",
    "少女",
    "女性",
    "girl",
    "girls",
    "woman",
    "female",
)


@dataclass(frozen=True)
class GenerationRequest:
    """Structured input shared by Discord, Gallery, and future clients.

    The model deliberately contains no Discord objects.  ``requester_id`` is
    an optional trace-owner value; ``source`` and ``source_metadata`` describe
    the caller without requiring that caller to be Discord.
    """

    character_preset: str | None = None
    freeform_character: str | None = None
    explicit_tags: tuple[str, ...] = ()
    appearance_text: str | None = None
    request_text: str | None = None
    model_key: str = DEFAULT_IMAGE_GENERATION_MODEL_KEY
    rating: str = PROMPT_RATING_AUTO
    loras: tuple[object, ...] = ()
    size_preset: str | None = None
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg: float = 7.0
    sampler: str = "Euler a"
    count: int = 1
    variation: str = "off"
    variation_offset: int = 0
    seed: int = -1
    requester_id: int | str | None = None
    source: str = "unknown"
    source_metadata: Mapping[str, Any] = field(default_factory=dict)
    reference_id: int | None = None
    enable_generic_danbooru: bool = True
    preserve_prompt_text: bool = True
    model_selection_explicit: bool = False
    explicit_generation_fields: frozenset[str] = frozenset()
    # Reference/Pose is carried as a small, typed-by-contract mapping so the
    # public API can keep the image itself behind an opaque asset id while the
    # service receives validated bytes.  Expected keys are mode, image_id,
    # image_bytes, content_type, width, height, strength and fit.
    reference: Mapping[str, Any] | None = None
    pose: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "explicit_tags", _split_tag_values(tuple(self.explicit_tags)))
        object.__setattr__(self, "loras", tuple(self.loras or ()))
        object.__setattr__(self, "source_metadata", dict(self.source_metadata or {}))

    @property
    def prompt_input(self) -> str:
        """Return only the free-form text intended for Prompt Designer.

        Explicit tags and appearance text are kept out of this string so they
        can be resolved and merged with their own provenance.
        """

        parts = (
            _normalise_text(self.character_preset),
            _normalise_text(self.freeform_character),
            _normalise_text(self.request_text),
        )
        for part in parts:
            if part:
                return ", ".join(value for value in parts if value)
        if self.explicit_tags:
            return ", ".join(self.explicit_tags)
        return ""

    def validate(self, *, require_prompt_input: bool = True) -> None:
        if not _normalise_text(self.model_key):
            raise PromptGeneratorError("model_keyを指定してください。")
        if self.count < 1 or self.count > MAX_GENERATION_COUNT:
            raise PromptGeneratorError(f"countは1～{MAX_GENERATION_COUNT}枚で指定してください。")
        if (
            isinstance(self.variation_offset, bool)
            or not isinstance(self.variation_offset, int)
            or self.variation_offset < 0
            or self.variation_offset + self.count > MAX_GENERATION_COUNT
        ):
            raise PromptGeneratorError("variationの開始位置と枚数の組み合わせが不正です。")
        if self.width < 256 or self.height < 256:
            raise PromptGeneratorError("画像サイズは256以上で指定してください。")
        if self.steps < 1 or self.cfg < 0:
            raise PromptGeneratorError("生成設定が不正です。")
        if require_prompt_input and not self.prompt_input:
            raise PromptGeneratorError("characterまたはrequest_textを指定してください。")


@dataclass(frozen=True)
class TagProvenance:
    tag: str
    source: str
    category: str = ""


def _positive_tokens(value: object) -> list[str]:
    return list(_dedupe_tags(_split_tag_values((str(value or ""),))))


def _subject_system_tags(request: GenerationRequest, tokens: Sequence[str]) -> tuple[str, ...]:
    """Return only structural subject tags that the request supports."""

    keys = {_tag_key(token) for token in tokens}
    subject = " ".join(
        value
        for value in (
            _normalise_text(request.character_preset),
            _normalise_text(request.freeform_character),
            _normalise_text(request.request_text),
        )
        if value
    ).casefold()
    multi_subject = any(hint.casefold() in subject for hint in _MULTI_SUBJECT_HINTS)
    multi_subject = multi_subject or has_non_solo_constraint(
        tuple(tokens),
        context_text=subject,
    )

    result: list[str] = []
    if "1girl" in keys:
        result.append("1girl")
    elif "1boy" in keys:
        result.append("1boy")
    elif "1other" in keys:
        result.append("1other")
    elif not multi_subject and any(hint.casefold() in subject for hint in _FEMALE_SUBJECT_HINTS):
        result.append("1girl")

    if "solo" in keys:
        result.append("solo")
    elif subject and not multi_subject and not any(
        key in keys for key in ("2girls", "2boys", "multiple girls", "multiple boys")
    ):
        result.append("solo")
    return tuple(result)


def _finalize_positive(
    request: GenerationRequest,
    profile: Any,
    positive: object,
) -> tuple[str, tuple[TagProvenance, ...]]:
    """Apply deterministic system tags after the LLM/resolver merge."""

    tokens = _positive_tokens(positive)
    explicit_keys = {_tag_key(tag) for tag in request.explicit_tags}
    subject_tags = _subject_system_tags(request, tokens)
    subject_keys = {_tag_key(tag) for tag in subject_tags}

    requested_rating = _normalise_text(request.rating).casefold()
    rating_tags = (
        (requested_rating,)
        if requested_rating and requested_rating != PROMPT_RATING_AUTO
        else ()
    )
    requested_rating_keys = {_tag_key(tag) for tag in rating_tags}

    existing_quality = tuple(token for token in tokens if _tag_key(token) in _QUALITY_TAG_KEYS)
    quality_tags = _dedupe_tags((*existing_quality, *getattr(profile, "system_positive_tags", ())))
    quality_keys = {_tag_key(tag) for tag in quality_tags}

    body: list[str] = []
    for token in tokens:
        key = _tag_key(token)
        if key in subject_keys or key in quality_keys:
            continue
        if key in _RATING_TAG_KEYS and requested_rating != PROMPT_RATING_AUTO and key not in explicit_keys:
            continue
        body.append(token)

    final_tokens = _dedupe_tags((*subject_tags, *rating_tags, *body, *quality_tags))
    system_keys = subject_keys | requested_rating_keys | quality_keys
    system_provenance = tuple(
        TagProvenance(tag, "system")
        for tag in final_tokens
        if _tag_key(tag) in system_keys
    )
    return ", ".join(final_tokens), system_provenance


@dataclass(frozen=True)
class ResolvedGeneration:
    request: GenerationRequest
    prompt_result: ShogouPromptResult
    explicit_tags: tuple[str, ...] = ()
    appearance_tags: tuple[str, ...] = ()
    provenance: tuple[TagProvenance, ...] = ()
    suppressed_tags: tuple[SuppressedPromptTag, ...] = ()
    prompt_conflicts: tuple[PromptConflict, ...] = ()
    people_semantics: PromptPeopleSemantics = PromptPeopleSemantics()
    framing_semantics: PromptFramingSemantics = PromptFramingSemantics()


@dataclass(frozen=True)
class PreparedGeneration:
    request: GenerationRequest
    resolved: ResolvedGeneration
    options: SdGenerationOptions
    requested_seed: int


@dataclass(frozen=True)
class GenerationResult:
    preparation: PreparedGeneration
    images: tuple[GeneratedImage, ...] = ()
    requested_seed: int = -1
    actual_seeds: tuple[int | None, ...] = ()

    @property
    def actual_seed(self) -> int | None:
        values = tuple(value for value in self.actual_seeds if value is not None)
        return values[0] if len(values) == 1 else None


class GenerationService:
    """Thin orchestration facade over the existing prompt and SD services."""

    def __init__(
        self,
        *,
        prompt_resolver: PromptResolver | None = None,
        appearance_resolver: AppearanceResolver | None = None,
    ) -> None:
        self._prompt_resolver = prompt_resolver
        self._appearance_resolver = appearance_resolver or resolve_appearance_text

    @staticmethod
    def _requester_for_prompt(request: GenerationRequest) -> int | None:
        # Existing Prompt Designer trace/history APIs use an integer owner.
        # Non-Discord callers can keep a string requester_id in the request;
        # it is not coerced into a fake Discord user ID.
        return request.requester_id if isinstance(request.requester_id, int) else None

    @staticmethod
    def _canonicalize_explicit_tags(
        tags: Sequence[str],
    ) -> tuple[tuple[str, ...], PromptTagResolution]:
        raw_tags = _split_tag_values(tuple(tags))
        if not raw_tags:
            return (), resolve_prompt_tags("")
        resolution = resolve_prompt_tags(", ".join(raw_tags))
        recognised: set[str] = set()
        for match in resolution.matches:
            recognised.update(
                {
                    _tag_key(match.matched_alias),
                    _tag_key(match.entry.prompt_tag),
                    _tag_key(match.entry.danbooru_tag),
                }
            )
        unknown = tuple(tag for tag in raw_tags if _tag_key(tag) not in recognised)
        return _dedupe_tags((*resolution.resolved_prompt_tags, *unknown)), resolution

    def _resolve_appearance(self, text: str | None) -> AppearanceResolution:
        clean = _normalise_text(text)
        if not clean:
            return AppearanceResolution(input_text="")
        resolver = self._appearance_resolver
        if resolver is None:
            return AppearanceResolution(input_text=clean, unresolved=(clean,), resolver="unconfigured")
        resolved = resolver(clean)
        if isinstance(resolved, AppearanceResolution):
            return resolved
        return AppearanceResolution(
            input_text=clean,
            tags=_dedupe_tags(tuple(str(tag) for tag in resolved)),
        )

    def _merge_resolved_prompt(
        self,
        request: GenerationRequest,
        prompt_result: ShogouPromptResult,
    ) -> ResolvedGeneration:
        explicit_tags, explicit_resolution = self._canonicalize_explicit_tags(request.explicit_tags)
        appearance = self._resolve_appearance(request.appearance_text)

        confirmed_appearance = tuple(
            tag.strip()
            for tag in getattr(prompt_result, "confirmed_character_appearance_tags", ())
            if str(tag or "").strip()
        )
        explicit_categories = {
            category
            for category in (appearance_tag_category(tag) for tag in explicit_tags)
            if category is not None
        }
        confirmed_categories = {
            category
            for category in (appearance_tag_category(tag) for tag in confirmed_appearance)
            if category is not None
        }
        overridden_confirmed = {
            _tag_key(tag)
            for tag in confirmed_appearance
            if appearance_tag_category(tag) in explicit_categories
            and _tag_key(tag) not in {_tag_key(item) for item in explicit_tags}
        }
        if overridden_confirmed:
            positive_tokens = [
                token.strip()
                for token in str(prompt_result.positive or "").split(",")
                if token.strip() and _tag_key(token) not in overridden_confirmed
            ]
            prompt_result = replace(prompt_result, positive=", ".join(positive_tokens))

        appearance_tags = tuple(
            tag
            for tag in appearance.tags
            if (
                appearance_tag_category(tag) not in confirmed_categories
                and appearance_tag_category(tag) not in explicit_categories
            )
        )

        positive = prompt_result.positive
        if explicit_resolution.has_resolved_tags:
            positive = apply_prompt_tag_resolution(positive, explicit_resolution)
        positive = _append_tags(positive, explicit_tags)
        positive = _append_tags(positive, appearance_tags)

        provenance: list[TagProvenance] = []
        for tag in explicit_tags:
            provenance.append(TagProvenance(tag, "user_explicit"))
        for tag in appearance_tags:
            provenance.append(TagProvenance(tag, "appearance_resolver"))
        for tag in prompt_result.danbooru_character_resolved_tags:
            provenance.append(TagProvenance(tag, "character_confirmed"))
        for tag in prompt_result.resolved_prompt_tags:
            provenance.append(TagProvenance(tag, "request_resolver"))
        for tag in prompt_result.danbooru_resolved_prompt_tags:
            provenance.append(TagProvenance(tag, "request_resolver"))
        for token in positive.split(","):
            clean = token.strip()
            if clean and _tag_key(clean) not in {_tag_key(item.tag) for item in provenance}:
                provenance.append(TagProvenance(clean, "request_resolver"))

        merged_result = replace(
            prompt_result,
            positive=positive,
            user_grounded_tags=_dedupe_tags(
                (*prompt_result.user_grounded_tags, *explicit_tags, *appearance_tags)
            ),
            tag_provenance=tuple(
                f"{item.tag} -> {item.source}"
                for item in dict.fromkeys(provenance)
            ),
        )
        return ResolvedGeneration(
            request=request,
            prompt_result=merged_result,
            explicit_tags=explicit_tags,
            appearance_tags=appearance_tags,
            provenance=tuple(dict.fromkeys(provenance)),
            people_semantics=detect_people_semantics(
                _positive_tokens(positive),
                context_text=" ".join(
                    value
                    for value in (request.request_text, request.character_preset, request.freeform_character)
                    if _normalise_text(value)
                ),
                explicit_values=explicit_tags,
            ),
        )

    def resolve_request(
        self,
        request: GenerationRequest,
        *,
        prompt_resolver: PromptResolver | None = None,
    ) -> ResolvedGeneration:
        request.validate()
        profile = get_image_generation_profile(request.model_key.strip())
        validate_requested_prompt_rating(request.rating, is_adult=profile.is_adult)
        resolver = prompt_resolver or self._prompt_resolver or generate_shogou_prompt_result
        prompt_result = resolver(
            request.prompt_input,
            self._requester_for_prompt(request),
            request.reference_id,
            profile.model_key,
            request.rating,
            enable_generic_danbooru=request.enable_generic_danbooru,
        )
        if not isinstance(prompt_result, ShogouPromptResult):
            raise PromptGeneratorError("Prompt Designerの結果が不正です。")
        return self._merge_resolved_prompt(request, prompt_result)

    def prepare_generation(
        self,
        request: GenerationRequest,
        *,
        resolved: ResolvedGeneration | None = None,
        prompt_result: ShogouPromptResult | None = None,
        positive_override: str | None = None,
        negative_prompt_override: str | None = None,
        variation_tags: Sequence[str] = (),
        resolved_loras: tuple[LoraSelection, ...] | None = None,
        lora_resolver: LoraResolver | None = None,
        gallery_output_root: Path | None = None,
        gallery_metadata: Mapping[str, Any] | None = None,
    ) -> PreparedGeneration:
        request.validate(
            require_prompt_input=prompt_result is None and positive_override is None
        )
        profile = get_image_generation_profile(request.model_key.strip())
        validate_requested_prompt_rating(request.rating, is_adult=profile.is_adult)
        if resolved is None:
            if prompt_result is not None:
                resolved = self._merge_resolved_prompt(request, prompt_result)
            else:
                resolved = self.resolve_request(request)

        loras = resolved_loras
        if loras is None:
            resolve_loras = lora_resolver or resolve_saved_lora_selections
            loras = tuple(resolve_loras(request.loras, model_family=profile.model_family))
        positive_source = positive_override if positive_override is not None else resolved.prompt_result.positive
        base_provenance = tuple(
            PromptTagRecord(item.tag, item.source, item.category)
            for item in resolved.provenance
        )
        pre_final_conflicts = resolve_prompt_tag_conflicts(
            positive_source,
            explicit_tags=resolved.explicit_tags,
            request_text=request.request_text or "",
            character_text=" ".join(
                value
                for value in (request.character_preset, request.freeform_character)
                if _normalise_text(value)
            ),
            base_provenance=base_provenance,
            variation_tags=variation_tags,
        )
        positive, system_provenance = _finalize_positive(
            request,
            profile,
            pre_final_conflicts.positive,
        )
        system_records = tuple(
            PromptTagRecord(item.tag, item.source, item.category)
            for item in system_provenance
        )
        final_conflicts = resolve_prompt_tag_conflicts(
            positive,
            explicit_tags=resolved.explicit_tags,
            request_text=request.request_text or "",
            character_text=" ".join(
                value
                for value in (request.character_preset, request.freeform_character)
                if _normalise_text(value)
            ),
            base_provenance=(*base_provenance, *pre_final_conflicts.provenance, *system_records),
            variation_tags=variation_tags,
        )
        positive = final_conflicts.positive
        positive = apply_lora_triggers(positive, loras)
        # Lora triggers are another final tag source.  Run the same semantic
        # invariant once more after them so a late auto/inferred ``solo`` can
        # never survive merely because it was appended after the first pass.
        post_lora_conflicts = resolve_prompt_tag_conflicts(
            positive,
            explicit_tags=resolved.explicit_tags,
            request_text=request.request_text or "",
            character_text=" ".join(
                value
                for value in (request.character_preset, request.freeform_character)
                if _normalise_text(value)
            ),
            base_provenance=(
                *base_provenance,
                *pre_final_conflicts.provenance,
                *system_records,
                *final_conflicts.provenance,
            ),
            variation_tags=variation_tags,
        )
        positive = post_lora_conflicts.positive
        negative = (
            negative_prompt_override
            if negative_prompt_override is not None
            else resolved.prompt_result.negative
        )
        if not _normalise_text(negative):
            negative = get_negative_prompt(profile.negative_default)
        else:
            negative = get_negative_prompt(str(negative))

        prepared_provenance = [
            TagProvenance(item.tag, item.source, item.category)
            for item in post_lora_conflicts.provenance
        ]
        prepared_provenance.extend(
            item
            for item in resolved.provenance
            if _tag_key(item.tag) not in {_tag_key(value.tag) for value in prepared_provenance}
        )
        source_overrides: dict[str, str] = {}
        for selection in loras:
            trigger = _normalise_text(selection.trigger)
            if trigger:
                source_overrides[_tag_key(trigger)] = "lora_trigger"
        for token in str(positive or "").split(","):
            clean = token.strip()
            key = _tag_key(clean)
            if key in _SYSTEM_TAG_KEYS:
                source_overrides[key] = "system"
        prepared_provenance = [
            TagProvenance(
                item.tag,
                item.source
                if item.source in {"user_explicit", "user_request"}
                else source_overrides.get(_tag_key(item.tag), item.source),
                item.category,
            )
            for item in prepared_provenance
        ]
        known_provenance_keys = {_tag_key(item.tag) for item in prepared_provenance}
        for token in str(positive or "").split(","):
            clean = token.strip()
            key = _tag_key(clean)
            source = source_overrides.get(key)
            if clean and source and key not in known_provenance_keys:
                prepared_provenance.append(TagProvenance(clean, source))
                known_provenance_keys.add(key)

        suppressed_tags = tuple(
            dict.fromkeys(
                (
                    *resolved.suppressed_tags,
                    *pre_final_conflicts.suppressed_tags,
                    *final_conflicts.suppressed_tags,
                    *post_lora_conflicts.suppressed_tags,
                )
            )
        )
        prompt_conflicts = tuple(
            dict.fromkeys(
                (
                    *resolved.prompt_conflicts,
                    *pre_final_conflicts.prompt_conflicts,
                    *final_conflicts.prompt_conflicts,
                    *post_lora_conflicts.prompt_conflicts,
                )
            )
        )

        reference_values = dict(request.reference or {})
        reference_mode = str(reference_values.get("mode") or "none").strip().casefold()
        reference_bytes = reference_values.get("image_bytes")
        reference_content_type = reference_values.get("content_type")
        reference_strength = reference_values.get("strength")
        reference_fit = str(reference_values.get("fit") or "contain").strip().casefold()
        reference_options: dict[str, Any] = {}
        if reference_mode != "none":
            if not isinstance(reference_bytes, bytes) or not reference_bytes:
                raise PromptGeneratorError("参考画像が読み込めませんでした。")
            if reference_mode == "composition":
                reference_options = {
                    "reference_image_bytes": reference_bytes,
                    "reference_image_content_type": reference_content_type,
                    "reference_strength": reference_strength,
                    "reference_mode": reference_mode,
                }
            elif reference_mode == "pose":
                reference_options = {
                    "pose_reference_bytes": reference_bytes,
                    "pose_reference_content_type": reference_content_type,
                    "pose_reference_width": reference_values.get("width"),
                    "pose_reference_height": reference_values.get("height"),
                    "pose_strength": 0.7 if reference_strength is None else reference_strength,
                    "pose_fit_mode": reference_fit,
                    "reference_mode": reference_mode,
                }
            else:
                raise PromptGeneratorError("reference modeが不正です。")

        options = SdGenerationOptions(
            prompt=positive,
            negative_prompt=str(negative),
            model_key=profile.model_key,
            model_family=profile.model_family,
            filename_prefix=profile.filename_prefix,
            model_selection_explicit=request.model_selection_explicit,
            preserve_prompt_text=request.preserve_prompt_text,
            count=request.count,
            width=request.width,
            height=request.height,
            steps=request.steps,
            cfg=request.cfg,
            sampler=request.sampler,
            seed=request.seed,
            loras=loras,
            gallery_output_root=gallery_output_root,
            gallery_metadata=(None if gallery_metadata is None else dict(gallery_metadata)),
            explicit_generation_fields=request.explicit_generation_fields,
            **reference_options,
        )
        prepared_result = replace(
            resolved.prompt_result,
            positive=positive,
            negative=str(negative),
            tag_provenance=tuple(
                f"{item.tag} -> {item.source}"
                for item in dict.fromkeys(prepared_provenance)
            ),
        )
        prepared_resolved = replace(
            resolved,
            prompt_result=prepared_result,
            provenance=tuple(dict.fromkeys(prepared_provenance)),
            suppressed_tags=suppressed_tags,
            prompt_conflicts=prompt_conflicts,
            people_semantics=post_lora_conflicts.people_semantics,
            framing_semantics=replace(
                merge_prompt_framing_semantics(
                    resolved.framing_semantics,
                    pre_final_conflicts.framing_semantics,
                    final_conflicts.framing_semantics,
                    post_lora_conflicts.framing_semantics,
                ),
                # Only the last pass describes the effective final prompt;
                # requested/source/suppressed traces remain additive.
                effective=post_lora_conflicts.framing_semantics.effective,
            ),
        )
        return PreparedGeneration(
            request=request,
            resolved=prepared_resolved,
            options=options,
            requested_seed=request.seed,
        )

    def submit_generation(
        self,
        request: GenerationRequest,
        *,
        prepared: PreparedGeneration | None = None,
        generator: ImageGenerator = generate_images_with_metadata,
        **prepare_kwargs: Any,
    ) -> GenerationResult:
        preparation = prepared or self.prepare_generation(request, **prepare_kwargs)
        images = tuple(generator(preparation.options))
        return GenerationResult(
            preparation=preparation,
            images=images,
            requested_seed=preparation.requested_seed,
            actual_seeds=tuple(image.seed for image in images),
        )


__all__ = [
    "AppearanceResolution",
    "GenerationRequest",
    "GenerationResult",
    "GenerationService",
    "PreparedGeneration",
    "ResolvedGeneration",
    "TagProvenance",
]
