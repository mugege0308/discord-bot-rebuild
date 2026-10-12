"""Shared size and composition settings for /generate_ai."""

from __future__ import annotations

from dataclasses import dataclass

from services.prompt_context_conflicts import analyze_explicit_visual_constraints


DEFAULT_SIZE_KEY = "square"
DEFAULT_COMPOSITION_PRESET = "none"


@dataclass(frozen=True)
class ImageSizeSpec:
    key: str
    label: str
    width: int
    height: int


IMAGE_SIZE_SPECS = {
    "square": ImageSizeSpec("square", "Square", 1024, 1024),
    "portrait": ImageSizeSpec("portrait", "Portrait", 768, 1024),
    "landscape": ImageSizeSpec("landscape", "Landscape", 1024, 768),
}

COMPOSITION_LABELS = {
    "none": "None",
    "face_closeup": "Face close-up",
    "upper_body": "Upper body",
    "cowboy": "Cowboy shot",
    "full_body": "Full body",
    "full_body_tight": "Full body tight",
}

_COMPOSITION_INSTRUCTIONS = {
    "face_closeup": "Use a face close-up with the face large and the character centered.",
    "upper_body": "Use an upper-body composition centered on the character.",
    "cowboy": "Use a cowboy-shot composition showing the people from about mid-thigh up.",
    "full_body": "Show the complete character from head to toe, including both feet.",
    "full_body_tight": (
        "Show the complete character from head to toe, centered and filling most of the frame, "
        "with minimal empty space and balanced framing."
    ),
}


def get_image_size_spec(size_key: str | None) -> ImageSizeSpec:
    key = (size_key or DEFAULT_SIZE_KEY).strip().casefold()
    try:
        return IMAGE_SIZE_SPECS[key]
    except KeyError as exc:
        raise ValueError(f"unknown size: {size_key}") from exc


def validate_composition_preset(value: str | None) -> str:
    key = (value or DEFAULT_COMPOSITION_PRESET).strip().casefold()
    if key not in COMPOSITION_LABELS:
        raise ValueError(f"unknown composition_preset: {value}")
    return key


def infer_size_key(width: int | None, height: int | None) -> str:
    if width is None or height is None or width == height:
        return DEFAULT_SIZE_KEY
    return "portrait" if height > width else "landscape"


def build_composition_preset_instruction(
    request_text: str,
    composition_preset: str | None,
) -> str:
    """Returns a weak framing hint unless the current request owns framing."""

    preset = validate_composition_preset(composition_preset)
    if preset == DEFAULT_COMPOSITION_PRESET:
        return ""
    if analyze_explicit_visual_constraints(request_text).controls_composition:
        return ""
    return (
        "Composition preset (weak helper; the user's explicit composition always wins): "
        + _COMPOSITION_INSTRUCTIONS[preset]
    )
