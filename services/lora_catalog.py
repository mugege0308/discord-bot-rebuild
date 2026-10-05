"""Portable LoRA catalog discovery and compatibility helpers.

This is intentionally a data-layer module.  It discovers managed LoRA files,
reads the metadata core, and produces choice data for later UI/generation
integration.  It does not import ``sd_control`` or select a workflow.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from services.lora_paths import (
    DEFAULT_COMFY_LORA_ROOT,
    DEFAULT_LORA_MANAGER_OUTPUTS_DIR,
    get_comfy_lora_root,
    get_lora_manager_outputs_dir,
    metadata_path_for_lora,
)
from services.lora_training import (
    LORA_TYPE_STYLE,
    LORA_TYPES,
    LoraMetadata,
    has_pending_lora_training_state,
    load_lora_metadata,
    normalize_lora_type,
)


logger = logging.getLogger("secretary-bot.lora_catalog")

KOHYA_INTERMEDIATE_CHECKPOINT_SUFFIX = re.compile(r"-\d{6}$")
DEFAULT_STYLE_WORKFLOW = Path("workflows") / "comfy_workflow_lora.json"
MAX_DISCORD_CHOICES = 25


@dataclass(frozen=True)
class ComfyStyle:
    """A catalog entry; workflow application belongs to a later commit."""

    display_name: str
    lora_name: str
    trigger: str
    default_strength: float
    workflow_file: Path | None = None
    source_path: Path | None = None
    load_error: str | None = None
    lora_type: str = LORA_TYPE_STYLE
    checkpoint: str | None = None
    model_family: str | None = None
    metadata_path: Path | None = None


def model_family_compatible(
    lora_family: str | None,
    requested_family: str | None,
) -> bool:
    """Return whether catalog filtering may expose the LoRA.

    Missing metadata is intentionally compatible: legacy LoRAs remain
    selectable until a later generation layer can apply stricter policy.
    """

    if not lora_family or not requested_family:
        return True
    return lora_family.strip().casefold() == requested_family.strip().casefold()


def _static_comfy_styles() -> dict[str, ComfyStyle]:
    # Keep the existing built-in catalog entry as data.  The workflow is only
    # metadata here; this commit does not ship or execute that workflow.
    return {
        "natsume": ComfyStyle(
            display_name="Natsume Soft",
            lora_name="natsume_soft_style_v1.safetensors",
            trigger="natsume_soft_style",
            default_strength=0.5,
            workflow_file=DEFAULT_STYLE_WORKFLOW,
            lora_type=LORA_TYPE_STYLE,
            checkpoint="animagineXL40Opt_clear.safetensors",
            model_family="sdxl",
        )
    }


def _resolve_comfy_lora_name(path: Path, relative_name: str) -> str:
    """Use a flat shared ComfyUI copy when it is actually present."""

    comfy_root = get_comfy_lora_root()
    shared_copy = comfy_root / path.name
    if shared_copy.is_file():
        return path.name
    return relative_name.replace(os.sep, "/")


def _metadata_path_if_present(path: Path) -> Path | None:
    primary = metadata_path_for_lora(path)
    if primary.is_file():
        return primary
    legacy = path.with_suffix(".json")
    return legacy if legacy.is_file() else None


def _managed_key(path: Path, relative: Path, stem_counts: dict[str, int], reserved: set[str]) -> str:
    stem = path.stem
    normalized = stem.casefold()
    if stem_counts.get(normalized, 0) == 1 and normalized not in reserved:
        return stem
    digest = hashlib.sha256(str(relative).casefold().encode("utf-8")).hexdigest()[:12]
    return f"managed-{digest}"


def _display_name(path: Path, relative: Path, stem_counts: dict[str, int], metadata: LoraMetadata | None) -> str:
    if metadata is not None and metadata.name:
        return metadata.name
    stem = path.stem
    if stem_counts.get(stem.casefold(), 0) > 1:
        return f"{relative.parent.as_posix()} / {stem}" if relative.parent.as_posix() != "." else stem
    return stem


def build_comfy_styles(outputs_dir: Path | None = None) -> dict[str, ComfyStyle]:
    """Build the catalog from managed output files and compatible metadata."""

    root = Path(outputs_dir) if outputs_dir is not None else get_lora_manager_outputs_dir()
    styles = _static_comfy_styles()
    if not root.exists() or not root.is_dir():
        return styles

    try:
        candidates = sorted(
            (
                path
                for path in root.rglob("*.safetensors")
                if path.is_file() and not KOHYA_INTERMEDIATE_CHECKPOINT_SUFFIX.search(path.stem)
            ),
            key=lambda path: str(path).casefold(),
        )
    except OSError as exc:
        logger.warning("Could not scan LoRA catalog root=%s: %s", root, exc)
        return styles

    stem_counts: dict[str, int] = {}
    for path in candidates:
        stem_counts[path.stem.casefold()] = stem_counts.get(path.stem.casefold(), 0) + 1
    reserved = {entry.lora_name.casefold() for entry in styles.values()}

    for path in candidates:
        if has_pending_lora_training_state(path):
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = Path(path.name)
        metadata = load_lora_metadata(path)
        key = _managed_key(path, relative, stem_counts, reserved)
        while key in styles:
            key = f"{key}-managed"
        metadata_path = _metadata_path_if_present(path)
        styles[key] = ComfyStyle(
            display_name=_display_name(path, relative, stem_counts, metadata),
            lora_name=_resolve_comfy_lora_name(path, relative.as_posix()),
            trigger=metadata.trigger_tag if metadata else "",
            default_strength=metadata.recommended_strength if metadata else 0.5,
            source_path=path.resolve(),
            load_error=None,
            lora_type=metadata.lora_type if metadata else LORA_TYPE_STYLE,
            checkpoint=metadata.checkpoint if metadata else None,
            model_family=metadata.model_family if metadata else None,
            metadata_path=metadata_path,
        )
    return styles


COMFY_STYLES: dict[str, ComfyStyle] = build_comfy_styles()


def refresh_comfy_styles(outputs_dir: Path | None = None) -> dict[str, ComfyStyle]:
    """Refresh the in-memory catalog and return the new mapping."""

    refreshed = build_comfy_styles(outputs_dir)
    COMFY_STYLES.clear()
    COMFY_STYLES.update(refreshed)
    return COMFY_STYLES


def get_comfy_style(key: str | None) -> ComfyStyle | None:
    """Look up a catalog entry without making catalog keys case-sensitive."""

    normalized = str(key or "").strip().casefold()
    if not normalized:
        return None
    for catalog_key, style in COMFY_STYLES.items():
        if catalog_key.casefold() == normalized:
            return style
    return None


def get_comfy_style_choice_specs() -> list[tuple[str, str]]:
    """Return display/value pairs for style entries only."""

    entries = [
        (style.display_name, key)
        for key, style in COMFY_STYLES.items()
        if style.lora_type == LORA_TYPE_STYLE
    ]
    return sorted(entries, key=lambda item: item[0].casefold())[:MAX_DISCORD_CHOICES]


def get_comfy_lora_choice_specs(
    lora_type: str,
    model_family: str | None = None,
) -> list[tuple[str, str]]:
    """Return typed catalog choices, optionally filtered by model family."""

    normalized_type = normalize_lora_type(lora_type)
    entries = [
        (style.display_name, key)
        for key, style in COMFY_STYLES.items()
        if style.lora_type == normalized_type
        and model_family_compatible(style.model_family, model_family)
    ]
    # Keep one slot available for a future explicit "none" option in Discord.
    return sorted(entries, key=lambda item: item[0].casefold())[: MAX_DISCORD_CHOICES - 1]


def get_lora_catalog_summary() -> dict[str, int]:
    """Return stable counts useful to diagnostics without exposing paths."""

    summary = {lora_type: 0 for lora_type in LORA_TYPES}
    for style in COMFY_STYLES.values():
        summary[style.lora_type] = summary.get(style.lora_type, 0) + 1
    return summary


__all__ = [
    "COMFY_STYLES",
    "ComfyStyle",
    "DEFAULT_COMFY_LORA_ROOT",
    "DEFAULT_LORA_MANAGER_OUTPUTS_DIR",
    "DEFAULT_STYLE_WORKFLOW",
    "build_comfy_styles",
    "get_comfy_lora_choice_specs",
    "get_comfy_style",
    "get_comfy_lora_root",
    "get_comfy_style_choice_specs",
    "get_lora_catalog_summary",
    "get_lora_manager_outputs_dir",
    "model_family_compatible",
    "refresh_comfy_styles",
]
