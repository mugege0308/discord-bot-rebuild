"""Owner-scoped persistent reference image storage.

The library is deliberately independent from Discord, ComfyUI, and prompt
generation.  It stores image metadata in the existing generation-history
database without changing its schema version.
"""

from __future__ import annotations

import json
import logging
import secrets
import shutil
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from services.generation_history import DB_FILE
from services.weather import JST


logger = logging.getLogger("secretary-bot.saved_references")

SAVED_REFERENCES_DIR = Path("data") / "saved_references"
REFERENCE_KINDS = ("pose", "reference", "style", "derive")
MAX_TAGS = 20
MAX_NAME_LENGTH = 80
MAX_NOTE_LENGTH = 500


@dataclass(frozen=True)
class SavedReference:
    id: int
    user_id: int
    name: str
    image_path: Path
    kind: str
    tags: tuple[str, ...]
    note: str
    prompt: str
    negative_prompt: str
    seed: int | None
    model_name: str
    lora_json: str
    created_at: str


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_FILE, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_references (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            kind TEXT NOT NULL,
            tags_json TEXT NOT NULL DEFAULT '[]',
            note TEXT NOT NULL DEFAULT '',
            prompt TEXT NOT NULL DEFAULT '',
            negative_prompt TEXT NOT NULL DEFAULT '',
            seed INTEGER,
            model_name TEXT NOT NULL DEFAULT '',
            lora_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_saved_references_user_kind "
        "ON saved_references(user_id, kind, created_at)"
    )
    return connection


@contextmanager
def _managed_connection():
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _normalize_tags(tags: Sequence[str] | str | None) -> tuple[str, ...]:
    values = tags.split(",") if isinstance(tags, str) else list(tags or ())
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = " ".join(str(raw).split()).strip()
        key = value.casefold()
        if not value or key in seen:
            continue
        seen.add(key)
        result.append(value[:50])
        if len(result) >= MAX_TAGS:
            break
    return tuple(result)


def _row_to_reference(row: sqlite3.Row) -> SavedReference:
    try:
        raw_tags = json.loads(row["tags_json"] or "[]")
    except (TypeError, json.JSONDecodeError):
        raw_tags = []
    tags = _normalize_tags(raw_tags if isinstance(raw_tags, list) else ())
    raw_seed = row["seed"]
    seed = int(raw_seed) if isinstance(raw_seed, int) else None
    return SavedReference(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        name=str(row["name"]),
        image_path=Path(str(row["image_path"])),
        kind=str(row["kind"]),
        tags=tags,
        note=str(row["note"] or ""),
        prompt=str(row["prompt"] or ""),
        negative_prompt=str(row["negative_prompt"] or ""),
        seed=seed,
        model_name=str(row["model_name"] or ""),
        lora_json=str(row["lora_json"] or "[]"),
        created_at=str(row["created_at"]),
    )


def _metadata_value(metadata: dict[str, Any] | None, key: str, default: Any = "") -> Any:
    value = (metadata or {}).get(key, default)
    return default if value is None else value


def save_reference(
    *,
    user_id: int,
    name: str,
    kind: str,
    tags: Sequence[str] | str | None = None,
    note: str = "",
    source_path: Path | None = None,
    image_bytes: bytes | None = None,
    extension: str = ".png",
    metadata: dict[str, Any] | None = None,
) -> SavedReference:
    normalized_kind = str(kind or "").strip().casefold()
    if normalized_kind not in REFERENCE_KINDS:
        raise ValueError(f"kindは{', '.join(REFERENCE_KINDS)}のいずれかで指定してください。")
    clean_name = " ".join(str(name or "").split()).strip()
    if not clean_name:
        raise ValueError("保存名を入力してください。")
    clean_name = clean_name[:MAX_NAME_LENGTH]
    clean_note = str(note or "").strip()[:MAX_NOTE_LENGTH]
    normalized_tags = _normalize_tags(tags)

    if image_bytes is not None:
        if not image_bytes:
            raise ValueError("保存する画像データが空です。")
        suffix = str(extension or ".png").lower()
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".png"
    elif source_path is not None:
        source = Path(source_path)
        if not source.is_file():
            raise ValueError(f"保存元画像が見つかりません: {source}")
        suffix = source.suffix.lower() if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} else ".png"
    else:
        raise ValueError("保存元画像がありません。")

    destination_dir = SAVED_REFERENCES_DIR / str(int(user_id))
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(6)}{suffix}"
    if image_bytes is not None:
        destination.write_bytes(image_bytes)
    else:
        shutil.copy2(Path(source_path), destination)

    metadata = metadata or {}
    seed_value = _metadata_value(metadata, "seed", None)
    seed = int(seed_value) if isinstance(seed_value, int) else None
    loras = _metadata_value(metadata, "loras", _metadata_value(metadata, "lora", []))
    try:
        lora_json = json.dumps(loras if isinstance(loras, list) else [], ensure_ascii=False)
    except (TypeError, ValueError):
        lora_json = "[]"

    created_at = datetime.now(JST).isoformat(timespec="seconds")
    try:
        with _managed_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO saved_references(
                    user_id, name, image_path, kind, tags_json, note,
                    prompt, negative_prompt, seed, model_name, lora_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(user_id), clean_name, str(destination), normalized_kind,
                    json.dumps(list(normalized_tags), ensure_ascii=False), clean_note,
                    str(_metadata_value(metadata, "prompt", _metadata_value(metadata, "positive_prompt", "")))[:10000],
                    str(_metadata_value(metadata, "negative_prompt", ""))[:10000],
                    seed,
                    str(_metadata_value(metadata, "model_name", _metadata_value(metadata, "model", "")))[:200],
                    lora_json, created_at,
                ),
            )
            reference_id = int(cursor.lastrowid)
        logger.info("Saved reference created: id=%s user_id=%s kind=%s path=%s", reference_id, user_id, normalized_kind, destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    saved = get_reference(int(user_id), reference_id)
    if saved is None:
        destination.unlink(missing_ok=True)
        raise RuntimeError("保存済み参考画像の登録結果を読み込めませんでした。")
    return saved


def get_reference(user_id: int, reference_id: int) -> SavedReference | None:
    with _managed_connection() as connection:
        row = connection.execute(
            "SELECT * FROM saved_references WHERE user_id = ? AND id = ?",
            (int(user_id), int(reference_id)),
        ).fetchone()
    return None if row is None else _row_to_reference(row)


def list_references(user_id: int, *, kind: str | None = None, tag: str | None = None, limit: int = 25) -> list[SavedReference]:
    normalized_kind = str(kind or "").strip().casefold() or None
    if normalized_kind is not None and normalized_kind not in REFERENCE_KINDS:
        raise ValueError("不正なkindです。")
    normalized_tag = " ".join(str(tag or "").split()).casefold() or None
    query = "SELECT * FROM saved_references WHERE user_id = ?"
    params: list[Any] = [int(user_id)]
    if normalized_kind:
        query += " AND kind = ?"
        params.append(normalized_kind)
    query += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(max(1, min(int(limit), 100)))
    with _managed_connection() as connection:
        rows = connection.execute(query, params).fetchall()
    references = [_row_to_reference(row) for row in rows]
    if normalized_tag:
        references = [item for item in references if any(normalized_tag == tag.casefold() for tag in item.tags)]
    return references


def random_reference(user_id: int, *, kind: str | None = None, tag: str | None = None) -> SavedReference | None:
    import random
    candidates = list_references(user_id, kind=kind, tag=tag, limit=100)
    return random.choice(candidates) if candidates else None


def read_reference_bytes(reference: SavedReference) -> bytes:
    try:
        return reference.image_path.read_bytes()
    except OSError as exc:
        raise RuntimeError("保存済み参考画像のファイルを読み込めませんでした。") from exc


def delete_reference(user_id: int, reference_id: int) -> bool:
    reference = get_reference(user_id, reference_id)
    if reference is None:
        return False
    with _managed_connection() as connection:
        connection.execute("DELETE FROM saved_references WHERE user_id = ? AND id = ?", (int(user_id), int(reference_id)))
    try:
        root = SAVED_REFERENCES_DIR.resolve()
        path = reference.image_path.resolve()
        if path.is_relative_to(root):
            path.unlink(missing_ok=True)
        else:
            logger.warning("Refused to delete saved reference outside storage root: %s", path)
    except OSError:
        logger.exception("Failed to remove saved reference image: id=%s", reference_id)
    logger.info("Saved reference deleted: id=%s user_id=%s", reference_id, user_id)
    return True


__all__ = [
    "REFERENCE_KINDS", "SAVED_REFERENCES_DIR", "SavedReference",
    "delete_reference", "get_reference", "list_references", "random_reference",
    "read_reference_bytes", "save_reference",
]
