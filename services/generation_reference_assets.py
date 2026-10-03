"""Opaque runtime storage for images submitted to the Generation API.

The public API never accepts a filesystem path.  Uploads are decoded and
validated here, then stored under a generated identifier so a later job can
refer to the same bytes without exposing the runtime directory.
"""

from __future__ import annotations

import os
import re
import secrets
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image


MAX_REFERENCE_IMAGE_BYTES = 10 * 1024 * 1024
REFERENCE_IMAGE_ROOT = Path("data") / "generation_references"
REFERENCE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
_FORMAT_TO_CONTENT_TYPE = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}
_CONTENT_TYPE_TO_SUFFIX = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}


@dataclass(frozen=True)
class ReferenceAsset:
    asset_id: str
    content_type: str
    image_bytes: bytes
    width: int
    height: int


def storage_root() -> Path:
    configured = os.getenv("GENERATION_REFERENCE_ROOT", "").strip()
    return Path(configured) if configured else REFERENCE_IMAGE_ROOT


def _decode_image(image_bytes: bytes, claimed_content_type: str | None) -> tuple[str, int, int]:
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("参考画像が空です。")
    if len(image_bytes) > MAX_REFERENCE_IMAGE_BYTES:
        raise ValueError("参考画像が大きすぎます。")
    content_type = str(claimed_content_type or "").split(";", 1)[0].strip().casefold()
    if content_type not in _CONTENT_TYPE_TO_SUFFIX:
        raise ValueError("参考画像はJPEG / PNG / WebPを指定してください。")
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            actual_content_type = _FORMAT_TO_CONTENT_TYPE.get(str(image.format or "").upper())
            width, height = int(image.width), int(image.height)
    except (OSError, ValueError, SyntaxError) as exc:
        raise ValueError("有効なJPEG / PNG / WebP画像ではありません。") from exc
    if actual_content_type != content_type or width <= 0 or height <= 0:
        raise ValueError("参考画像の形式がContent-Typeと一致しません。")
    return actual_content_type, width, height


def save_reference_asset(image_bytes: bytes, content_type: str | None) -> ReferenceAsset:
    normalized_type, width, height = _decode_image(image_bytes, content_type)
    root = storage_root()
    root.mkdir(parents=True, exist_ok=True)
    asset_id = secrets.token_urlsafe(16)
    destination = root / f"{asset_id}{_CONTENT_TYPE_TO_SUFFIX[normalized_type]}"
    destination.write_bytes(image_bytes)
    return ReferenceAsset(asset_id, normalized_type, image_bytes, width, height)


def load_reference_asset(asset_id: str) -> ReferenceAsset:
    if not isinstance(asset_id, str) or not REFERENCE_ID_RE.fullmatch(asset_id):
        raise ValueError("参考画像IDが不正です。")
    root = storage_root()
    for content_type, suffix in _CONTENT_TYPE_TO_SUFFIX.items():
        path = root / f"{asset_id}{suffix}"
        if not path.is_file():
            continue
        image_bytes = path.read_bytes()
        normalized_type, width, height = _decode_image(image_bytes, content_type)
        return ReferenceAsset(asset_id, normalized_type, image_bytes, width, height)
    raise ValueError("参考画像が見つかりません。")


__all__ = [
    "MAX_REFERENCE_IMAGE_BYTES",
    "REFERENCE_IMAGE_ROOT",
    "ReferenceAsset",
    "load_reference_asset",
    "save_reference_asset",
    "storage_root",
]
