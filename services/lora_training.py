"""Caption preparation and metadata persistence for trained LoRAs.

The repository does not own a LoRA trainer; training is performed by an
external LoRA manager.  This module owns the stable metadata sidecar format,
training state, and caption helpers shared by that manager and future
generation/catalog integrations.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.lora_paths import (
    LORA_METADATA_SUFFIX,
    LORA_TRAINING_STATE_SUFFIX,
    LoraPathError,
    metadata_path_for_lora,
    resolve_lora_file as _resolve_lora_file,
    training_state_path_for_lora,
)


logger = logging.getLogger("secretary-bot.lora_training")

LORA_TYPE_STYLE = "style"
LORA_TYPE_CHARACTER = "character"
LORA_TYPE_ADDITIONAL = "additional"
LORA_TYPES = (LORA_TYPE_STYLE, LORA_TYPE_CHARACTER, LORA_TYPE_ADDITIONAL)
LORA_TYPE_LABELS = {
    LORA_TYPE_STYLE: "絵柄",
    LORA_TYPE_CHARACTER: "キャラクター",
    LORA_TYPE_ADDITIONAL: "追加",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PREVIEW_MAX_ITEMS = 100


class LoraTrainingError(ValueError):
    """Raised when a caption or metadata training input is invalid."""


@dataclass(frozen=True)
class LoraMetadata:
    name: str
    lora_type: str
    trigger_tag: str
    checkpoint: str | None
    model_family: str | None
    recommended_strength: float


@dataclass(frozen=True)
class LoraTrainingState:
    metadata: LoraMetadata
    status: str
    metadata_registered: bool
    path: Path
    expected_output: Path | None
    training_info: Path | None
    history: Path | None
    config: Path | None


@dataclass(frozen=True)
class CaptionPreviewItem:
    image_path: Path
    caption_path: Path
    caption: str
    existed: bool
    changed: bool


@dataclass(frozen=True)
class CaptionPreview:
    dataset_dir: Path
    lora_type: str
    trigger_tag: str
    items: tuple[CaptionPreviewItem, ...]


def normalize_lora_type(value: object) -> str:
    if not isinstance(value, str):
        raise LoraTrainingError("LoRA種別は style / character / additional のいずれかです。")
    normalized = value.strip().casefold()
    aliases = {
        "style": LORA_TYPE_STYLE,
        "絵柄": LORA_TYPE_STYLE,
        "character": LORA_TYPE_CHARACTER,
        "キャラクター": LORA_TYPE_CHARACTER,
        "additional": LORA_TYPE_ADDITIONAL,
        "追加": LORA_TYPE_ADDITIONAL,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise LoraTrainingError("LoRA種別は絵柄 / キャラクター / 追加から選択してください。") from exc


def _clean_text(value: object, field: str, *, required: bool = False) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise LoraTrainingError(f"{field}は文字列で指定してください。")
    if required and not text:
        raise LoraTrainingError(f"{field}は必須です。")
    return text


def _clean_strength(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LoraTrainingError("推奨強度は0.0～1.0の数値で指定してください。")
    strength = float(value)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise LoraTrainingError("推奨強度は0.0～1.0の範囲で指定してください。")
    return strength


def validate_training_metadata(
    *,
    name: object,
    lora_type: object,
    trigger_tag: object = "",
    checkpoint: object = None,
    model_family: object = None,
    recommended_strength: object = 0.5,
) -> LoraMetadata:
    normalized_type = normalize_lora_type(lora_type)
    cleaned_name = _clean_text(name, "LoRA名", required=True)
    cleaned_trigger = _clean_text(trigger_tag, "トリガータグ")
    if normalized_type == LORA_TYPE_CHARACTER and not cleaned_trigger:
        raise LoraTrainingError("キャラクターLoRAではトリガータグが必須です。")
    cleaned_checkpoint = _clean_text(checkpoint, "checkpoint") or None
    cleaned_family = _clean_text(model_family, "モデル系統") or None
    return LoraMetadata(
        name=cleaned_name,
        lora_type=normalized_type,
        trigger_tag=cleaned_trigger,
        checkpoint=cleaned_checkpoint,
        model_family=cleaned_family,
        recommended_strength=_clean_strength(recommended_strength),
    )


def resolve_lora_file(lora_file: object, *, search_root: Path) -> Path:
    """Compatibility wrapper exposing path-core resolution with training errors."""

    try:
        return _resolve_lora_file(lora_file, search_root=search_root)
    except LoraPathError as exc:
        raise LoraTrainingError(str(exc)) from exc


def _metadata_candidates(lora_path: Path) -> tuple[Path, ...]:
    return (
        metadata_path_for_lora(lora_path),
        Path(lora_path).with_suffix(".json"),
    )


def _optional_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value.strip())


def load_lora_training_state(lora_path: Path) -> LoraTrainingState | None:
    """Load a pending trainer state without treating it as final metadata."""

    lora_path = Path(lora_path)
    state_path = training_state_path_for_lora(lora_path)
    if not state_path.is_file():
        return None
    try:
        value = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise LoraTrainingError("training state JSON must be an object")
        metadata_value = value.get("lora_metadata")
        raw = {**value, **metadata_value} if isinstance(metadata_value, dict) else value
        metadata = validate_training_metadata(
            name=raw.get("name", raw.get("lora_name", lora_path.stem)),
            lora_type=raw.get("lora_type", LORA_TYPE_STYLE),
            trigger_tag=raw.get("trigger_tag", ""),
            checkpoint=raw.get("checkpoint"),
            model_family=raw.get("model_family"),
            recommended_strength=raw.get(
                "recommended_strength", raw.get("default_strength", 0.5)
            ),
        )
        return LoraTrainingState(
            metadata=metadata,
            status=str(value.get("status") or "unknown"),
            metadata_registered=value.get("metadata_registered") is True,
            path=state_path,
            expected_output=_optional_path(value.get("expected_output")),
            training_info=_optional_path(value.get("training_info")),
            history=_optional_path(value.get("history")),
            config=_optional_path(value.get("config")),
        )
    except (OSError, json.JSONDecodeError, LoraTrainingError, TypeError) as exc:
        logger.warning("Ignoring invalid LoRA training state: %s (%s)", state_path, exc)
        return None


def has_pending_lora_training_state(lora_path: Path) -> bool:
    state_path = training_state_path_for_lora(Path(lora_path))
    if not state_path.is_file():
        return False
    state = load_lora_training_state(lora_path)
    return state is None or not state.metadata_registered


def load_lora_registration_defaults(lora_path: Path) -> tuple[LoraMetadata, str]:
    lora_path = Path(lora_path)
    metadata = load_lora_metadata(lora_path)
    if metadata is not None:
        return metadata, "metadata"
    state = load_lora_training_state(lora_path)
    if state is not None and not state.metadata_registered:
        return state.metadata, "training"
    return (
        validate_training_metadata(
            name=lora_path.stem,
            lora_type=LORA_TYPE_STYLE,
            trigger_tag="",
            checkpoint=None,
            model_family=None,
            recommended_strength=0.5,
        ),
        "legacy",
    )


def clear_lora_training_state(lora_path: Path) -> bool:
    try:
        training_state_path_for_lora(Path(lora_path)).unlink(missing_ok=True)
        return True
    except OSError as exc:
        logger.error(
            "Could not remove LoRA training state after metadata save: %s (%s)",
            lora_path,
            exc,
            exc_info=True,
        )
        return False


def _metadata_to_object(metadata: LoraMetadata) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": metadata.name,
        "lora_type": metadata.lora_type,
        "trigger_tag": metadata.trigger_tag,
        "checkpoint": metadata.checkpoint,
        "model_family": metadata.model_family,
        "recommended_strength": metadata.recommended_strength,
    }


def save_lora_metadata(
    lora_path: Path,
    metadata: LoraMetadata,
    *,
    operation_id: str | None = None,
) -> Path:
    """Atomically write metadata beside an existing LoRA without overwriting."""

    lora_path = Path(lora_path)
    operation = operation_id or "-"
    started_at = time.perf_counter()
    logger.info(
        "LoRA metadata file save start operation_id=%s lora_path=%s",
        operation,
        lora_path,
    )
    try:
        if lora_path.suffix.casefold() != ".safetensors" or not lora_path.is_file():
            raise LoraTrainingError(f"指定されたLoRA本体が見つかりません: {lora_path}")
        destination = metadata_path_for_lora(lora_path)
        payload = _metadata_to_object(metadata)
        logger.info(
            "LoRA metadata file save step=validate_target complete operation_id=%s destination=%s",
            operation,
            destination,
        )
        if destination.exists():
            logger.info(
                "LoRA metadata file save step=existing_sidecar_detected operation_id=%s destination=%s",
                operation,
                destination,
            )
            try:
                existing = json.loads(destination.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise LoraTrainingError(
                    f"同名のLoRAメタデータが既にあり、内容を確認できないため上書きしません: {destination}"
                ) from exc
            if existing == payload:
                training_state_cleanup_ok = clear_lora_training_state(lora_path)
                logger.info(
                    "LoRA metadata file save complete operation_id=%s elapsed_ms=%s destination=%s reused=true training_state_cleanup_ok=%s",
                    operation,
                    int((time.perf_counter() - started_at) * 1000),
                    destination,
                    training_state_cleanup_ok,
                )
                return destination
            raise LoraTrainingError(
                f"同名のLoRAメタデータが既にあり、内容が異なるため上書きしません: {destination}"
            )

        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if destination.exists():
                raise LoraTrainingError(
                    f"同名のLoRAメタデータが既に作成されたため上書きしません: {destination}"
                )
            os.replace(temporary, destination)
        except LoraTrainingError:
            raise
        except OSError as exc:
            raise LoraTrainingError(f"LoRAメタデータを保存できませんでした: {destination}") from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logger.error(
                    "LoRA metadata temporary cleanup failed operation_id=%s temporary=%s",
                    operation,
                    temporary,
                    exc_info=True,
                )

        training_state_cleanup_ok = clear_lora_training_state(lora_path)
        logger.info(
            "LoRA metadata file save complete operation_id=%s elapsed_ms=%s destination=%s reused=false training_state_cleanup_ok=%s",
            operation,
            int((time.perf_counter() - started_at) * 1000),
            destination,
            training_state_cleanup_ok,
        )
        return destination
    except Exception:
        logger.exception(
            "LoRA metadata file save failed operation_id=%s elapsed_ms=%s lora_path=%s",
            operation,
            int((time.perf_counter() - started_at) * 1000),
            lora_path,
        )
        raise


def load_lora_metadata(lora_path: Path) -> LoraMetadata | None:
    """Load current or legacy sidecars; invalid metadata never hides a LoRA."""

    for candidate in _metadata_candidates(Path(lora_path)):
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise LoraTrainingError("metadata JSON must be an object")
            return validate_training_metadata(
                name=value.get("name", Path(lora_path).stem),
                lora_type=value.get("lora_type", value.get("type", LORA_TYPE_STYLE)),
                trigger_tag=value.get("trigger_tag", value.get("trigger", "")),
                checkpoint=value.get("checkpoint"),
                model_family=value.get("model_family"),
                recommended_strength=value.get(
                    "recommended_strength", value.get("default_strength", 0.5)
                ),
            )
        except (OSError, json.JSONDecodeError, LoraTrainingError):
            logger.warning("Ignoring invalid LoRA metadata: %s", candidate, exc_info=True)
            return None
    return None


def _trigger_pattern(trigger_tag: str) -> re.Pattern[str]:
    return re.compile(r"(?<![\w-])" + re.escape(trigger_tag) + r"(?![\w-])", re.IGNORECASE)


def _caption_contains_trigger(caption: str, trigger_tag: str) -> bool:
    return bool(trigger_tag and _trigger_pattern(trigger_tag).search(caption))


def build_caption_with_trigger(caption: str, trigger_tag: str) -> str:
    body = caption.strip()
    trigger = trigger_tag.strip()
    if not trigger or _caption_contains_trigger(body, trigger):
        return body
    return trigger if not body else f"{trigger}, {body}"


def _image_files(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise LoraTrainingError(f"dataset directoryが見つかりません: {dataset_dir}")
    return sorted(
        (
            path
            for path in dataset_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )


def preview_lora_captions(
    dataset_dir: Path,
    *,
    lora_type: str,
    trigger_tag: str = "",
) -> CaptionPreview:
    normalized_type = normalize_lora_type(lora_type)
    trigger = _clean_text(trigger_tag, "トリガータグ")
    if normalized_type == LORA_TYPE_CHARACTER and not trigger:
        raise LoraTrainingError("キャラクターLoRAではトリガータグが必須です。")

    items: list[CaptionPreviewItem] = []
    for image_path in _image_files(Path(dataset_dir))[:PREVIEW_MAX_ITEMS]:
        caption_path = image_path.with_suffix(".txt")
        existed = caption_path.exists()
        try:
            existing = caption_path.read_text(encoding="utf-8") if existed else ""
        except OSError as exc:
            raise LoraTrainingError(f"captionを読み込めませんでした: {caption_path}") from exc
        final_caption = build_caption_with_trigger(existing, trigger) if trigger else existing.strip()
        items.append(
            CaptionPreviewItem(
                image_path=image_path,
                caption_path=caption_path,
                caption=final_caption,
                existed=existed,
                changed=(not existed) or existing.strip() != final_caption,
            )
        )
    if not items:
        raise LoraTrainingError(f"dataset directoryに対応画像がありません: {dataset_dir}")
    return CaptionPreview(
        dataset_dir=Path(dataset_dir),
        lora_type=normalized_type,
        trigger_tag=trigger,
        items=tuple(items),
    )


def apply_lora_captions(
    preview: CaptionPreview, *, operation_id: str | None = None
) -> int:
    operation = operation_id or "-"
    started_at = time.perf_counter()
    logger.info(
        "LoRA caption apply start operation_id=%s dataset_dir=%s item_count=%s",
        operation,
        preview.dataset_dir,
        len(preview.items),
    )
    changed_count = 0
    try:
        for item in preview.items:
            if not item.changed:
                continue
            try:
                item.caption_path.write_text(
                    item.caption + "\n" if item.caption else "", encoding="utf-8"
                )
            except OSError as exc:
                raise LoraTrainingError(f"captionを保存できませんでした: {item.caption_path}") from exc
            changed_count += 1
    except Exception:
        logger.exception(
            "LoRA caption apply failed operation_id=%s elapsed_ms=%s dataset_dir=%s",
            operation,
            int((time.perf_counter() - started_at) * 1000),
            preview.dataset_dir,
        )
        raise
    logger.info(
        "LoRA caption apply complete operation_id=%s elapsed_ms=%s changed_count=%s",
        operation,
        int((time.perf_counter() - started_at) * 1000),
        changed_count,
    )
    return changed_count


def format_caption_preview(
    preview: CaptionPreview, *, limit: int = 12, caption_limit: int = 500
) -> str:
    lines = [
        f"dataset: {preview.dataset_dir}",
        f"type: {LORA_TYPE_LABELS[preview.lora_type]}",
        f"trigger: {preview.trigger_tag or 'なし'}",
        f"caption件数: {len(preview.items)}件（表示は最大{limit}件）",
        "",
    ]
    for item in preview.items[:limit]:
        caption = item.caption[:caption_limit]
        if len(item.caption) > caption_limit:
            caption += "..."
        status = "作成" if not item.existed else "更新" if item.changed else "変更なし"
        lines.append(f"[{status}] {item.caption_path.name}: {caption or '（空）'}")
    remaining = len(preview.items) - min(limit, len(preview.items))
    if remaining > 0:
        lines.append(f"...（他{remaining}件省略）")
    return "\n".join(lines)
