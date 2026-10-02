"""Resolve catalog LoRA names against ComfyUI's accepted loader values.

The catalog is built from the manager's filesystem, while ComfyUI validates
``LoraLoader.lora_name`` against the values returned by ``/object_info``.
Those values may use Windows separators even when the catalog uses POSIX
separators.  This module keeps that representation difference at the API
boundary: comparison is normalized, but the value returned to the workflow is
always the exact candidate string supplied by ComfyUI.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import requests


logger = logging.getLogger("secretary-bot.comfy_lora_resolver")

_CACHE_TTL_SECONDS = 5.0
_cache_lock = threading.Lock()
_cached_candidates: tuple[str, ...] = ()
_cached_at = 0.0


class ComfyLoraResolutionError(RuntimeError):
    """Raised before submission when ComfyUI cannot accept a selected LoRA."""


@dataclass(frozen=True)
class ComfyLoraInventory:
    """The accepted LoraLoader values returned by one ComfyUI snapshot."""

    candidates: tuple[str, ...]


def _api_url(path: str) -> str:
    base = os.getenv("SD_API_URL", "http://127.0.0.1:8188").strip().rstrip("/")
    return f"{base}/{path.lstrip('/')}"


def normalize_lora_name(name: str) -> str:
    """Normalize separators for comparison without changing the API value."""

    value = str(name or "").strip().replace("\\", "/")
    value = re.sub(r"/+", "/", value)
    while value.startswith("./"):
        value = value[2:]
    return value


def _casefold_key(name: str) -> str:
    return normalize_lora_name(name).casefold()


def _extract_lora_candidates(payload: Any) -> tuple[str, ...]:
    loader = payload.get("LoraLoader") if isinstance(payload, dict) else None
    if not isinstance(loader, dict):
        return ()
    input_spec = loader.get("input")
    required = input_spec.get("required") if isinstance(input_spec, dict) else None
    lora_spec = required.get("lora_name") if isinstance(required, dict) else None
    values = lora_spec[0] if isinstance(lora_spec, (list, tuple)) and lora_spec else lora_spec
    if not isinstance(values, (list, tuple)):
        return ()

    candidates: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip():
            candidates.append(value)
    return tuple(dict.fromkeys(candidates))


def fetch_comfy_lora_inventory(*, refresh: bool = False, timeout: float = 10.0) -> ComfyLoraInventory:
    """Fetch and briefly cache the currently accepted ComfyUI LoRA names."""

    global _cached_at, _cached_candidates
    now = time.monotonic()
    with _cache_lock:
        if not refresh and _cached_candidates and now - _cached_at < _CACHE_TTL_SECONDS:
            return ComfyLoraInventory(_cached_candidates)

    try:
        response = requests.get(_api_url("/object_info"), timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Failed to fetch ComfyUI LoRA candidates: %s", exc)
        raise ComfyLoraResolutionError(
            "ComfyUIのLoRA一覧を取得できないため、LoRA生成を開始できません。"
        ) from exc

    candidates = _extract_lora_candidates(payload)
    if not candidates:
        raise ComfyLoraResolutionError(
            "ComfyUIが認識しているLoRA一覧を取得できないため、LoRA生成を開始できません。"
        )
    with _cache_lock:
        _cached_candidates = candidates
        _cached_at = time.monotonic()
    logger.debug("Fetched ComfyUI LoRA candidates: count=%s", len(candidates))
    return ComfyLoraInventory(candidates)


def clear_comfy_lora_inventory_cache() -> None:
    global _cached_at, _cached_candidates
    with _cache_lock:
        _cached_candidates = ()
        _cached_at = 0.0


def _resolve_against_candidates(catalog_name: str, candidates: Sequence[str]) -> str | None:
    requested = str(catalog_name or "").strip()
    if not requested:
        return None

    # Prefer the literal API value when the catalog already has it.
    for candidate in candidates:
        if candidate == requested:
            return candidate

    requested_normalized = normalize_lora_name(requested)
    for candidate in candidates:
        if normalize_lora_name(candidate) == requested_normalized:
            return candidate

    # Windows model paths are case-insensitive.  Keep this as a final
    # fallback so a case-preserving match always wins when available.
    requested_folded = _casefold_key(requested)
    for candidate in candidates:
        if _casefold_key(candidate) == requested_folded:
            return candidate
    return None


def resolve_comfy_lora_names(
    catalog_names: Iterable[str],
    *,
    candidates: Sequence[str] | None = None,
    refresh: bool = False,
) -> tuple[str, ...]:
    """Return exact ComfyUI values for every catalog name, or fail atomically."""

    names = tuple(str(name or "").strip() for name in catalog_names)
    if not names:
        return ()
    accepted = tuple(candidates) if candidates is not None else fetch_comfy_lora_inventory(refresh=refresh).candidates
    resolved: list[str] = []
    for name in names:
        match = _resolve_against_candidates(name, accepted)
        if match is None:
            logger.warning(
                "Selected LoRA is not accepted by ComfyUI: catalog_name=%r candidate_count=%s",
                name,
                len(accepted),
            )
            raise ComfyLoraResolutionError(
                "選択したLoRAを現在のComfyUIが認識していません。LoRA一覧を更新してから再試行してください。"
            )
        resolved.append(match)
    return tuple(resolved)


__all__ = [
    "ComfyLoraInventory",
    "ComfyLoraResolutionError",
    "clear_comfy_lora_inventory_cache",
    "fetch_comfy_lora_inventory",
    "normalize_lora_name",
    "resolve_comfy_lora_names",
]
