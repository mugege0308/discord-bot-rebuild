"""Environment-aware path helpers for the LoRA core.

This module deliberately contains no Discord, ComfyUI, or generation imports.
It gives training, catalog, and future job code one place to resolve LoRA
manager output paths without embedding a machine-specific Windows path.
"""

from __future__ import annotations

import os
from pathlib import Path


LORA_METADATA_SUFFIX = ".metadata.json"
LORA_TRAINING_STATE_SUFFIX = ".training.json"
DEFAULT_LORA_MANAGER_OUTPUTS_DIR = Path("outputs")
DEFAULT_COMFY_LORA_ROOT = Path("models") / "Lora"


class LoraPathError(ValueError):
    """Raised when a LoRA path cannot be resolved safely."""


def _env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else fallback


def get_lora_manager_outputs_dir() -> Path:
    """Return the configured LoRA manager output directory.

    ``LORA_MANAGER_OUTPUTS_DIR`` is preferred.  The relative fallback keeps
    the module portable for development and tests; deployment-specific paths
    belong in the process environment or external configuration.
    """

    return _env_path("LORA_MANAGER_OUTPUTS_DIR", DEFAULT_LORA_MANAGER_OUTPUTS_DIR)


def get_comfy_lora_root() -> Path:
    """Return the ComfyUI LoRA root used for shared-file detection."""

    return _env_path("COMFY_LORA_ROOT", DEFAULT_COMFY_LORA_ROOT)


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
