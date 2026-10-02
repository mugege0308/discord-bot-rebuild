"""Environment-aware path helpers for the LoRA core.

This module deliberately contains no Discord, ComfyUI, or generation imports.
It gives training, catalog, and future job code one place to resolve LoRA
manager output paths without embedding a machine-specific Windows path.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


LORA_METADATA_SUFFIX = ".metadata.json"
LORA_TRAINING_STATE_SUFFIX = ".training.json"
DEFAULT_LORA_MANAGER_OUTPUTS_DIR = Path("outputs")
DEFAULT_COMFY_LORA_ROOT = Path("models") / "Lora"

logger = logging.getLogger("secretary-bot.lora_paths")


class LoraPathError(ValueError):
    """Raised when a LoRA path cannot be resolved safely."""


def _configured_env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip().strip('"')
    return Path(value).expanduser() if value else None


def _existing_directory(path: Path) -> Path | None:
    try:
        return path if path.is_dir() else None
    except OSError:
        return None


def _legacy_manager_outputs_dir() -> Path | None:
    """Resolve the old manager project root to its managed output folder."""

    manager_root = _configured_env_path("LORA_MANAGER_ROOT")
    if manager_root is None:
        return None
    outputs = manager_root / "outputs"
    if _existing_directory(outputs) is not None:
        return outputs
    logger.warning("Ignoring invalid legacy LORA_MANAGER_ROOT for LoRA catalog resolution")
    return None


def _legacy_comfy_lora_root() -> Path | None:
    """Resolve only an explicitly recognizable legacy ComfyUI layout.

    The old ``LORA_KOHYA_ROOT`` points at the kohya training project, not at
    ComfyUI's model directory.  Treating it as the latter would make shared
    model lookup point at the wrong tree.  A nested ``models/Lora`` directory
    is supported for legacy deployments that colocated both layouts.
    """

    kohya_root = _configured_env_path("LORA_KOHYA_ROOT")
    if kohya_root is None:
        return None
    for relative in (Path("models") / "Lora", Path("models") / "lora"):
        candidate = kohya_root / relative
        if _existing_directory(candidate) is not None:
            return candidate
    return None


def get_lora_manager_outputs_dir() -> Path:
    """Return the configured LoRA manager output directory.

    ``LORA_MANAGER_OUTPUTS_DIR`` is preferred.  When it is absent, the legacy
    ``LORA_MANAGER_ROOT`` is interpreted as a project root and its ``outputs``
    child is used only when that directory exists.  The relative fallback
    keeps the module portable for development and tests; deployment-specific
    paths belong in the process environment or external configuration.
    """

    configured = _configured_env_path("LORA_MANAGER_OUTPUTS_DIR")
    if configured is not None:
        return configured
    legacy = _legacy_manager_outputs_dir()
    return legacy if legacy is not None else DEFAULT_LORA_MANAGER_OUTPUTS_DIR


def get_comfy_lora_root() -> Path:
    """Return the ComfyUI LoRA root used for shared-file detection.

    ``COMFY_LORA_ROOT`` is preferred.  ``LORA_KOHYA_ROOT`` is not a direct
    alias because the legacy variable denotes the training project.  Only a
    real nested ``models/Lora`` layout is accepted as a compatibility hint;
    otherwise the portable default remains in effect.
    """

    configured = _configured_env_path("COMFY_LORA_ROOT")
    if configured is not None:
        return configured
    legacy = _legacy_comfy_lora_root()
    return legacy if legacy is not None else DEFAULT_COMFY_LORA_ROOT


def metadata_path_for_lora(lora_path: Path) -> Path:
    return Path(lora_path).with_name(Path(lora_path).name + LORA_METADATA_SUFFIX)


def training_state_path_for_lora(lora_path: Path) -> Path:
    return Path(lora_path).with_name(Path(lora_path).name + LORA_TRAINING_STATE_SUFFIX)


def resolve_lora_file(lora_file: object, *, search_root: Path) -> Path:
    """Resolve one existing ``.safetensors`` file without guessing duplicates."""

    if not isinstance(lora_file, str) or not lora_file.strip():
        raise LoraPathError("lora_fileは必須です。")
    raw_value = lora_file.strip()
    requested = Path(raw_value)
    if requested.suffix.casefold() != ".safetensors":
        raise LoraPathError("lora_fileは.safetensorsのLoRA本体を指定してください。")

    root = Path(search_root)
    try:
        root = root.resolve()
    except OSError as exc:
        raise LoraPathError(f"LoRA管理出力ディレクトリを解決できませんでした: {root}") from exc

    candidates: list[Path] = []

    def add_candidate(candidate: Path) -> None:
        try:
            resolved = candidate.resolve()
        except OSError:
            return
        if (
            resolved.suffix.casefold() == ".safetensors"
            and resolved.is_file()
            and resolved not in candidates
        ):
            candidates.append(resolved)

    if requested.is_absolute():
        add_candidate(requested)
    elif len(requested.parts) == 1:
        if not root.exists() or not root.is_dir():
            raise LoraPathError(f"LoRA管理出力ディレクトリが見つかりません: {root}")
        try:
            for candidate in root.rglob(requested.name):
                add_candidate(candidate)
        except OSError as exc:
            raise LoraPathError(f"LoRA管理出力を検索できませんでした: {root}") from exc
    else:
        add_candidate(requested)
        add_candidate(root / requested)

    if not candidates:
        raise LoraPathError(f"指定されたLoRA本体が見つかりません: {raw_value}")
    if len(candidates) > 1:
        choices = "、".join(str(path) for path in sorted(candidates, key=lambda path: str(path).casefold())[:5])
        suffix = "" if len(candidates) <= 5 else "、ほか"
        raise LoraPathError(
            f"同名LoRAが複数見つかりました。フルパスで指定してください: {choices}{suffix}"
        )
    return candidates[0]
