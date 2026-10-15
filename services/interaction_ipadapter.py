"""Validation and preparation for released interaction IPAdapter profiles.

This module deliberately keeps the internal reference path out of request and
Gallery metadata.  The ledger stores a relative ComfyUI input path and a
checksum; the runtime root is supplied by the process environment.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from services.interaction_profiles import (
    InteractionChecksumMismatch,
    InteractionProfile,
    InteractionProfileCompatibilityError,
    InteractionProfileError,
    InteractionProfileModeUnsupported,
    InteractionReferenceMissing,
)
from services.runtime_paths import PROJECT_ROOT


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_IPADAPTER_FIELDS = (
    "preset",
    "model",
    "clip_vision",
    "weight",
    "weight_type",
    "start",
    "end",
    "embeds_scaling",
    "combine_embeds",
)


@dataclass(frozen=True)
class InteractionReferencePreflight:
    """Internal reference bytes plus safe execution metadata."""

    image_bytes: bytes
    content_type: str
    reference_sha256: str
    preprocess: str
    ipadapter: dict[str, Any]

    def metadata(self) -> dict[str, Any]:
        return {
            "reference_sha256": self.reference_sha256,
            "preprocess": self.preprocess,
            **self.ipadapter,
        }


def _reference_root() -> Path:
    for name in ("INTERACTION_REFERENCE_ROOT", "COMFYUI_ROOT", "COMFY_ROOT"):
        value = os.getenv(name, "").strip().strip('"')
        if value:
            return Path(value).expanduser().resolve()
    # A local checkout may provide input/<file> directly.  Production always
    # supplies INTERACTION_REFERENCE_ROOT from the ComfyUI package root.
    return PROJECT_ROOT


def _require_number(value: object, field: str, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InteractionProfileError(f"IPAdapter {field}が不正です。")
    result = float(value)
    if not minimum <= result <= maximum:
        raise InteractionProfileError(f"IPAdapter {field}が不正です。")
    return result


def _validate_ipadapter(profile: InteractionProfile) -> dict[str, Any]:
    settings = profile.ipadapter
    if not isinstance(settings, dict) or any(
        key not in settings or settings[key] in (None, "") for key in _REQUIRED_IPADAPTER_FIELDS
    ):
        raise InteractionProfileError("interaction_profile_schema_invalid: IPAdapter設定が不足しています。")
    normalized = {key: settings[key] for key in _REQUIRED_IPADAPTER_FIELDS}
    normalized["weight"] = _require_number(normalized["weight"], "weight")
    normalized["start"] = _require_number(normalized["start"], "start")
    normalized["end"] = _require_number(normalized["end"], "end")
    if normalized["start"] > normalized["end"]:
        raise InteractionProfileError("interaction_profile_schema_invalid: IPAdapter start/endが不正です。")
    for key in ("preset", "model", "clip_vision", "weight_type", "embeds_scaling", "combine_embeds"):
        if not isinstance(normalized[key], str) or not normalized[key].strip():
            raise InteractionProfileError(f"interaction_profile_schema_invalid: IPAdapter {key}が不正です。")
        normalized[key] = normalized[key].strip()
    return normalized


def preflight_ipadapter_profile(profile: InteractionProfile) -> InteractionReferencePreflight:
    """Validate, hash, and load the internal reference for one profile."""

    if profile.mode != "ipadapter_profile":
        raise InteractionProfileModeUnsupported(f"IPAdapter modeではありません: {profile.mode}")
    if not profile.dedicated_positive or not profile.dedicated_negative:
        raise InteractionProfileError("interaction_profile_schema_invalid: dedicated promptが不足しています。")
    reference = profile.reference
    if not isinstance(reference, dict):
        raise InteractionProfileError("interaction_profile_schema_invalid: reference設定が不足しています。")
    relative_file = reference.get("file")
    expected = str(reference.get("sha256") or "").strip().casefold()
    preprocess = str(reference.get("preprocess") or "").strip()
    if not isinstance(relative_file, str) or not relative_file.strip():
        raise InteractionProfileError("interaction_profile_schema_invalid: reference fileが不足しています。")
    if not _SHA256_RE.fullmatch(expected):
        raise InteractionProfileError("interaction_profile_schema_invalid: SHA-256が不正です。")
    if not preprocess:
        raise InteractionProfileError("interaction_profile_schema_invalid: reference preprocessが不足しています。")
    ipadapter = _validate_ipadapter(profile)
    candidate = (_reference_root() / relative_file).resolve()
    try:
        candidate.relative_to(_reference_root())
    except ValueError as exc:
        raise InteractionProfileError("interaction_profile_schema_invalid: reference pathがroot外です。") from exc
    if not candidate.is_file():
        raise InteractionReferenceMissing("interaction referenceが見つかりません。")
    image_bytes = candidate.read_bytes()
    actual = hashlib.sha256(image_bytes).hexdigest()
    if actual != expected:
        raise InteractionChecksumMismatch("interaction referenceのchecksumが一致しません。")
    return InteractionReferencePreflight(
        image_bytes=image_bytes,
        content_type="image/png",
        reference_sha256=actual,
        preprocess=preprocess,
        ipadapter=ipadapter,
    )


def validate_hugging_compatibility(request: Any) -> None:
    """Enforce the initial Phase 2 hugging request contract."""

    if int(request.count) != 1:
        raise InteractionProfileCompatibilityError(
            "huggingは1枚生成のみ対応です。", code="interaction_batch_unsupported"
        )
    if int(request.width) != 768 or int(request.height) != 1024:
        raise InteractionProfileCompatibilityError(
            "huggingは768x1024のみ対応です。", code="interaction_size_unsupported"
        )
    if tuple(request.loras or ()):
        raise InteractionProfileCompatibilityError(
            "huggingではLoRAを利用できません。", code="interaction_lora_unsupported"
        )
    reference = request.reference or {}
    mode = str(reference.get("mode") or "none").casefold().strip()
    if mode != "none":
        code = {
            "composition": "interaction_reference_conflict",
            "pose": "interaction_pose_reference_conflict",
            "edit": "interaction_reference_conflict",
        }.get(mode, "interaction_reference_conflict")
        raise InteractionProfileCompatibilityError(
            "huggingではユーザーreferenceを併用できません。", code=code
        )
    if request.pose:
        raise InteractionProfileCompatibilityError(
            "huggingではpose referenceを併用できません。",
            code="interaction_pose_reference_conflict",
        )


__all__ = [
    "InteractionReferencePreflight",
    "preflight_ipadapter_profile",
    "validate_hugging_compatibility",
]
