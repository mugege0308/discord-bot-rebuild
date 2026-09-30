"""Persistent lifecycle primitives for batch image-generation jobs.

This module deliberately stops at durable job state.  It does not start an
image-generation worker, call ComfyUI, register Discord commands, or enqueue
Gallery work.  Those integrations belong to later commits and can consume
the small, owner-checked API defined here.

Batch state lives in its own SQLite database so the existing generation
history schema remains unchanged.  The database path is configurable through
``BATCH_GENERATION_DB_PATH`` and defaults to ``data/batch_generation.db``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import threading
import uuid
from typing import Any, Iterable

from services.image_generation_profiles import (
    DEFAULT_IMAGE_GENERATION_MODEL_KEY,
    get_image_generation_profile,
)
from services.sd_control import LoraSelection, lora_selections_to_dicts
from services.shogou_prompt_designer import (
    PROMPT_RATING_AUTO,
    validate_requested_prompt_rating,
)


logger = logging.getLogger("secretary-bot.batch_generation")

DEFAULT_MAX_JOB_COUNT = 100
MAX_JOB_COUNT = DEFAULT_MAX_JOB_COUNT
MAX_IMAGES_PER_PROMPT = 10
MAX_VARIATION_BATCHES = DEFAULT_MAX_JOB_COUNT
BATCH_ERROR_STORAGE_LIMIT = 8000
BATCH_ERROR_DISPLAY_LIMIT = 360

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_COMPLETED = "completed"
JOB_FAILED = "failed"
JOB_CANCELLED = "cancelled"
TERMINAL_JOB_STATES = frozenset({JOB_COMPLETED, JOB_FAILED, JOB_CANCELLED})
ACTIVE_JOB_STATES = frozenset({JOB_QUEUED, JOB_RUNNING})
VALID_JOB_STATES = ACTIVE_JOB_STATES | TERMINAL_JOB_STATES

VARIATION_COMPOSITIONS = ("portrait", "upper body", "cowboy shot", "full body")
VARIATION_ANGLES = ("front view", "three-quarter view", "side view")
VARIATION_EXPRESSIONS = (
    "neutral expression",
    "gentle smile",
    "smile",
    "surprised expression",
    "serious expression",
)
VARIATION_POSES = ("standing", "sitting", "walking", "relaxed pose", "natural pose")
VARIATION_SCENES = ("simple background", "cafe", "city street", "classroom", "park")


class BatchJobError(RuntimeError):
    """Base error for invalid or unavailable persisted Batch state."""


class BatchJobNotFoundError(BatchJobError):
    """Raised when a requested job does not exist."""


class BatchJobOwnershipError(BatchJobError, PermissionError):
    """Raised when a user attempts to access another user's job."""


def configured_max_job_count() -> int:
    """Return the configured safety limit, clamped to a safe operator range."""

    raw = os.getenv("BATCH_MAX_JOB_COUNT", str(DEFAULT_MAX_JOB_COUNT)).strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_JOB_COUNT
    return max(1, min(value, 500))


def build_variation_patterns(count: int) -> tuple[tuple[str, ...], ...]:
    """Build deterministic variation labels without starting generation."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        return ()
    capacity = (
        len(VARIATION_COMPOSITIONS)
        * len(VARIATION_ANGLES)
        * len(VARIATION_EXPRESSIONS)
        * len(VARIATION_POSES)
        * len(VARIATION_SCENES)
    )
    if count > capacity:
        raise ValueError(f"variation_countは1～{capacity}種類で指定してください。")
    patterns: list[tuple[str, ...]] = []
    for index in range(count):
        remainder = index
        composition = VARIATION_COMPOSITIONS[remainder % len(VARIATION_COMPOSITIONS)]
        remainder //= len(VARIATION_COMPOSITIONS)
        angle = VARIATION_ANGLES[remainder % len(VARIATION_ANGLES)]
        remainder //= len(VARIATION_ANGLES)
        expression = VARIATION_EXPRESSIONS[remainder % len(VARIATION_EXPRESSIONS)]
        remainder //= len(VARIATION_EXPRESSIONS)
        pose = VARIATION_POSES[remainder % len(VARIATION_POSES)]
        remainder //= len(VARIATION_POSES)
        scene = VARIATION_SCENES[remainder % len(VARIATION_SCENES)]
        patterns.append((composition, angle, expression, pose, scene))
    return tuple(patterns)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _db_path() -> Path:
    raw = os.getenv("BATCH_GENERATION_DB_PATH", "data/batch_generation.db").strip().strip('"')
    if not raw:
        raw = "data/batch_generation.db"
    return Path(raw).expanduser().resolve()


@contextmanager
def _db_connection(path: Path | None = None):
    """Open a short-lived SQLite transaction and always close its handle."""

    destination = path or _db_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(destination, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    try:
        yield connection
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()
    finally:
        connection.close()


def _json_load(value: object, fallback: Any) -> Any:
    try:
        decoded = json.loads(str(value or ""))
    except (TypeError, json.JSONDecodeError):
        return fallback
    return decoded


def _stored_error_text(error: object) -> str:
    raw = str(error or "").strip()
    return (raw or "原因不明のエラー")[:BATCH_ERROR_STORAGE_LIMIT]


def compact_error_text(error: object, *, limit: int = BATCH_ERROR_DISPLAY_LIMIT) -> str:
    """Bound an error suitable for a future Discord status message."""

    raw = " ".join(str(error or "").split()) or "原因不明のエラー"
    return raw if len(raw) <= limit else raw[: limit - 1].rstrip() + "…"


@dataclass(frozen=True)
class BatchGenerationRequest:
    """Immutable request snapshot persisted when a Batch job is created."""

    requested_by: int
    character: str
    total_count: int
    images_per_prompt: int
    model_key: str = DEFAULT_IMAGE_GENERATION_MODEL_KEY
    loras: tuple[LoraSelection, ...] = ()
    request_text: str = ""
    variation_enabled: bool = True
    auto_review_enabled: bool = False
    variation_count: int | None = None
    images_per_variation: int | None = None
    base_prompt: str = ""
    base_negative_prompt: str = ""
    repeat_of_job_id: str = ""
    rating: str = PROMPT_RATING_AUTO
    generation_config: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_images_per_variation(self) -> int:
        return self.images_per_variation or self.images_per_prompt

    @property
    def effective_variation_count(self) -> int:
        if not self.variation_enabled:
            return 1
        if self.variation_count is not None:
            return self.variation_count
        return math.ceil(self.total_count / self.effective_images_per_variation)


@dataclass(frozen=True)
class BatchSpec:
    """Persistable future-worker batch description, with no generation logic."""

    batch_id: str
    job_id: str
    prompt: str = ""
    negative_prompt: str = ""
    character: str = ""
    loras: tuple[LoraSelection, ...] = ()
    checkpoint: str = ""
    requested_count: int = 0
    created_at: str = ""
    variation_index: int = 1


def _serialize_loras(loras: Iterable[LoraSelection] | None) -> list[dict[str, Any]]:
    values = tuple(loras or ())
    if len(values) > 3:
        raise ValueError("lorasは最大3件まで指定できます。")
    try:
        serialized = list(lora_selections_to_dicts(values))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("LoRA設定が不正です。") from exc
    if not all(isinstance(item, dict) for item in serialized):
        raise ValueError("LoRA設定が不正です。")
    return [dict(item) for item in serialized]


def _deserialize_loras(value: object) -> tuple[LoraSelection, ...]:
    if value in (None, [], ()):
        return ()
    if not isinstance(value, list):
        raise BatchJobError("保存されたLoRA設定が不正です。")
    selections: list[LoraSelection] = []
    for item in value:
        if not isinstance(item, dict):
            raise BatchJobError("保存されたLoRA設定が不正です。")
        required = ("key", "lora_name", "lora_type", "display_name", "trigger", "strength")
        if any(key not in item for key in required):
            raise BatchJobError("保存されたLoRA設定が不完全です。")
        try:
            selections.append(
                LoraSelection(
                    key=str(item["key"]),
                    lora_name=str(item["lora_name"]),
                    lora_type=str(item["lora_type"]),
                    display_name=str(item["display_name"]),
                    trigger=str(item["trigger"]),
                    strength=float(item["strength"]),
                    model_family=(
                        None
                        if item.get("model_family") is None
                        else str(item["model_family"])
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise BatchJobError("保存されたLoRA設定が不正です。") from exc
    return tuple(selections)


def _request_json(request: BatchGenerationRequest) -> dict[str, Any]:
    return {
        "character": request.character,
        "total_count": request.total_count,
        "images_per_prompt": request.effective_images_per_variation,
        "images_per_variation": request.effective_images_per_variation,
        "variation_count": request.effective_variation_count,
        "model_key": request.model_key,
        "loras": _serialize_loras(request.loras),
        "request_text": request.request_text,
        "variation_enabled": request.variation_enabled,
        "auto_review_enabled": request.auto_review_enabled,
        "base_prompt": request.base_prompt,
        "base_negative_prompt": request.base_negative_prompt,
        "repeat_of_job_id": request.repeat_of_job_id,
        "rating": request.rating,
        "generation_config": dict(request.generation_config),
    }


def _validate_request(request: BatchGenerationRequest) -> None:
    if isinstance(request.requested_by, bool) or not isinstance(request.requested_by, int):
        raise ValueError("requested_byが不正です。")
    if not request.character.strip() and not request.base_prompt.strip():
        raise ValueError("characterを指定してください。")
    if not isinstance(request.model_key, str) or not request.model_key.strip():
        raise ValueError("model_keyを指定してください。")
    for name, value in (
        ("total_count", request.total_count),
        ("images_per_prompt", request.images_per_prompt),
        ("variation_count", request.variation_count),
        ("images_per_variation", request.images_per_variation),
    ):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"{name}は整数で指定してください。")
    if not isinstance(request.variation_enabled, bool):
        raise ValueError("variation_enabledが不正です。")
    if not isinstance(request.auto_review_enabled, bool):
        raise ValueError("auto_review_enabledが不正です。")
    if not isinstance(request.generation_config, dict):
        raise ValueError("generation_configが不正です。")
    try:
        json.dumps(request.generation_config, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("generation_configはJSON化できる値で指定してください。") from exc
    _serialize_loras(request.loras)

    max_job_count = configured_max_job_count()
    images_per_variation = request.effective_images_per_variation
    variation_count = request.effective_variation_count
    if request.total_count < 1 or request.total_count > max_job_count:
        raise ValueError(f"total_countは1～{max_job_count}枚で指定してください。")
    if images_per_variation < 1 or images_per_variation > MAX_IMAGES_PER_PROMPT:
        raise ValueError(f"images_per_variationは1～{MAX_IMAGES_PER_PROMPT}枚で指定してください。")
    if variation_count < 1 or variation_count > max_job_count:
        raise ValueError(f"variation_countは1～{max_job_count}種類で指定してください。")
    if request.variation_enabled:
        build_variation_patterns(variation_count)
        if request.total_count > variation_count * images_per_variation:
            raise ValueError("total_countがvariation_count×images_per_variationを超えています。")

    profile = get_image_generation_profile(request.model_key.strip())
    try:
        validate_requested_prompt_rating(request.rating, is_adult=profile.is_adult)
    except Exception as exc:
        raise ValueError(str(exc)) from exc


def _row_config(row: sqlite3.Row) -> dict[str, Any]:
    config = _json_load(row["config_json"], {})
    return dict(config) if isinstance(config, dict) else {}


def _request_from_row(row: sqlite3.Row) -> BatchGenerationRequest:
    config = _row_config(row)
    try:
        requested_by = int(row["requested_by"])
    except (TypeError, ValueError) as exc:
        raise BatchJobError("保存されたjobの所有者が不正です。") from exc
    return BatchGenerationRequest(
        requested_by=requested_by,
        character=str(row["character"] or config.get("character") or ""),
        total_count=int(row["total_count"]),
        images_per_prompt=int(config.get("images_per_prompt") or 1),
        model_key=str(config.get("model_key") or DEFAULT_IMAGE_GENERATION_MODEL_KEY),
        loras=_deserialize_loras(config.get("loras", [])),
        request_text=str(config.get("request_text") or ""),
        variation_enabled=bool(row["variation_enabled"]),
        auto_review_enabled=bool(row["auto_review_enabled"]),
        variation_count=(
            None if config.get("variation_count") is None else int(config["variation_count"])
        ),
        images_per_variation=(
            None
            if config.get("images_per_variation") is None
            else int(config["images_per_variation"])
        ),
        base_prompt=str(config.get("base_prompt") or ""),
        base_negative_prompt=str(config.get("base_negative_prompt") or ""),
        repeat_of_job_id=str(config.get("repeat_of_job_id") or ""),
        rating=str(config.get("rating") or PROMPT_RATING_AUTO),
        generation_config=dict(config.get("generation_config") or {}),
    )


class BatchGenerationManager:
    """Thread-safe persistence and state API for future Batch workers."""

    def __init__(self) -> None:
        self._db_lock = threading.RLock()
        self._schema_paths: set[Path] = set()

    def _init_schema(self) -> None:
        path = _db_path()
        with self._db_lock:
            if path in self._schema_paths:
                return
            with _db_connection(path) as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS generation_jobs (
                        job_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        requested_by TEXT NOT NULL,
                        character TEXT NOT NULL,
                        total_count INTEGER NOT NULL,
                        completed_count INTEGER NOT NULL DEFAULT 0,
                        failed_count INTEGER NOT NULL DEFAULT 0,
                        status TEXT NOT NULL,
                        started_at TEXT,
                        finished_at TEXT,
                        auto_review_enabled INTEGER NOT NULL DEFAULT 0,
                        variation_enabled INTEGER NOT NULL DEFAULT 1,
                        current_batch_id TEXT NOT NULL DEFAULT '',
                        current_prompt TEXT NOT NULL DEFAULT '',
                        cancel_requested INTEGER NOT NULL DEFAULT 0,
                        review_status TEXT NOT NULL DEFAULT 'not_requested',
                        review_job_id TEXT NOT NULL DEFAULT '',
                        error TEXT NOT NULL DEFAULT '',
                        config_json TEXT NOT NULL DEFAULT '{}'
                    );
                    CREATE INDEX IF NOT EXISTS idx_generation_jobs_requested_by
                        ON generation_jobs(requested_by, created_at);
                    CREATE INDEX IF NOT EXISTS idx_generation_jobs_status
                        ON generation_jobs(status);
                    CREATE TABLE IF NOT EXISTS generation_batches (
                        batch_id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL REFERENCES generation_jobs(job_id) ON DELETE CASCADE,
                        prompt TEXT NOT NULL DEFAULT '',
                        negative_prompt TEXT NOT NULL DEFAULT '',
                        character TEXT NOT NULL DEFAULT '',
                        loras_json TEXT NOT NULL DEFAULT '[]',
                        checkpoint TEXT NOT NULL DEFAULT '',
                        requested_count INTEGER NOT NULL,
                        completed_count INTEGER NOT NULL DEFAULT 0,
                        failed_count INTEGER NOT NULL DEFAULT 0,
                        variation_index INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        error TEXT NOT NULL DEFAULT ''
                    );
                    CREATE INDEX IF NOT EXISTS idx_generation_batches_job
                        ON generation_batches(job_id, created_at);
                    """
                )
                columns = {
                    str(row[1])
                    for row in connection.execute(
                        "PRAGMA table_info(generation_batches)"
                    ).fetchall()
                }
                if "failed_count" not in columns:
                    connection.execute(
                        "ALTER TABLE generation_batches ADD COLUMN failed_count INTEGER NOT NULL DEFAULT 0"
                    )
                if "variation_index" not in columns:
                    connection.execute(
                        "ALTER TABLE generation_batches ADD COLUMN variation_index INTEGER NOT NULL DEFAULT 1"
                    )
            self._schema_paths.add(path)

    def _fetch_job_row(self, job_id: str) -> sqlite3.Row | None:
        self._init_schema()
        with self._db_lock, _db_connection() as connection:
            return connection.execute(
                "SELECT * FROM generation_jobs WHERE job_id = ?",
                (job_id.strip(),),
            ).fetchone()

    @staticmethod
    def _owner_matches(row: sqlite3.Row, requested_by: int) -> bool:
        return str(row["requested_by"]) == str(requested_by)

    def _require_owned_row(self, job_id: str, requested_by: int) -> sqlite3.Row:
        row = self._fetch_job_row(job_id)
        if row is None:
            raise BatchJobNotFoundError("指定されたjob_idが見つかりません。")
        if not self._owner_matches(row, requested_by):
            raise BatchJobOwnershipError("他のユーザーのjobは操作できません。")
        return row

    def create_job(self, request: BatchGenerationRequest) -> str:
        """Validate and atomically persist a new queued job."""

        _validate_request(request)
        self._init_schema()
        job_id = uuid.uuid4().hex[:16]
        now = _now_iso()
        config = json.dumps(_request_json(request), ensure_ascii=False)
        try:
            with self._db_lock, _db_connection() as connection:
                connection.execute(
                    """
                    INSERT INTO generation_jobs
                        (job_id, created_at, requested_by, character, total_count,
                         status, auto_review_enabled, variation_enabled, config_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        now,
                        str(request.requested_by),
                        request.character.strip(),
                        request.total_count,
                        JOB_QUEUED,
                        int(request.auto_review_enabled),
                        int(request.variation_enabled),
                        config,
                    ),
                )
        except Exception:
            logger.exception("Failed to persist batch job: job_id=%s", job_id)
            raise
        return job_id

    def get_job(self, job_id: str, requested_by: int | None = None) -> dict[str, Any] | None:
        """Return a job, optionally enforcing its owner.

        Internal callers may omit ``requested_by``.  User-facing callers must
        use ``get_job_for_user`` so an explicit foreign ID is rejected.
        """

        row = self._fetch_job_row(job_id)
        if row is None:
            return None
        if requested_by is not None and not self._owner_matches(row, requested_by):
            raise BatchJobOwnershipError("他のユーザーのjobは閲覧できません。")
        config = _row_config(row)
        with self._db_lock, _db_connection() as connection:
            batches = connection.execute(
                "SELECT * FROM generation_batches WHERE job_id = ? ORDER BY created_at, batch_id",
                (job_id.strip(),),
            ).fetchall()
        return {
            "job_id": row["job_id"],
            "created_at": row["created_at"],
            "requested_by": row["requested_by"],
            "character": row["character"],
            "total_count": row["total_count"],
            "completed_count": row["completed_count"],
            "failed_count": row["failed_count"],
            "status": row["status"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "auto_review_enabled": bool(row["auto_review_enabled"]),
            "variation_enabled": bool(row["variation_enabled"]),
            "current_batch_id": row["current_batch_id"],
            "current_prompt": row["current_prompt"],
            "cancel_requested": bool(row["cancel_requested"]),
            "review_status": row["review_status"],
            "review_job_id": row["review_job_id"],
            "error": row["error"],
            "variation_count": int(config.get("variation_count") or 0),
            "images_per_variation": int(
                config.get("images_per_variation") or config.get("images_per_prompt") or 1
            ),
            "repeat_of_job_id": str(config.get("repeat_of_job_id") or ""),
            "rating": str(config.get("rating") or PROMPT_RATING_AUTO),
            "config": config,
            "batches": [
                {
                    "batch_id": batch["batch_id"],
                    "job_id": batch["job_id"],
                    "prompt": batch["prompt"],
                    "negative_prompt": batch["negative_prompt"],
                    "character": batch["character"],
                    "loras": _json_load(batch["loras_json"], []),
                    "checkpoint": batch["checkpoint"],
                    "requested_count": batch["requested_count"],
                    "completed_count": batch["completed_count"],
                    "failed_count": batch["failed_count"],
                    "variation_index": batch["variation_index"],
                    "created_at": batch["created_at"],
                    "error": batch["error"],
                }
                for batch in batches
            ],
        }

    def get_job_for_user(self, job_id: str, requested_by: int) -> dict[str, Any]:
        job = self.get_job(job_id, requested_by=requested_by)
        if job is None:
            raise BatchJobNotFoundError("指定されたjob_idが見つかりません。")
        return job

    def get_request(self, job_id: str, requested_by: int | None = None) -> BatchGenerationRequest | None:
        row = self._fetch_job_row(job_id)
        if row is None:
            return None
        if requested_by is not None and not self._owner_matches(row, requested_by):
            raise BatchJobOwnershipError("他のユーザーのjobは閲覧できません。")
        return _request_from_row(row)

    def get_request_for_user(self, job_id: str, requested_by: int) -> BatchGenerationRequest:
        request = self.get_request(job_id, requested_by=requested_by)
        if request is None:
            raise BatchJobNotFoundError("指定されたjob_idが見つかりません。")
        return request

    def latest_job_id(self, requested_by: int) -> str | None:
        self._init_schema()
        with self._db_lock, _db_connection() as connection:
            row = connection.execute(
                "SELECT job_id FROM generation_jobs WHERE requested_by = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (str(requested_by),),
            ).fetchone()
        return str(row["job_id"]) if row else None

    def mark_running(self, job_id: str, requested_by: int) -> dict[str, Any]:
        row = self._require_owned_row(job_id, requested_by)
        state = str(row["status"])
        if state in TERMINAL_JOB_STATES:
            raise BatchJobError(f"jobはすでに{state}です。")
        if state == JOB_RUNNING:
            return self.get_job_for_user(job_id, requested_by)
        with self._db_lock, _db_connection() as connection:
            connection.execute(
                "UPDATE generation_jobs SET status = ?, started_at = ? WHERE job_id = ? AND status = ?",
                (JOB_RUNNING, _now_iso(), job_id.strip(), JOB_QUEUED),
            )
        return self.get_job_for_user(job_id, requested_by)

    def try_claim_job(self, job_id: str, requested_by: int) -> bool:
        """Atomically claim a queued job for one in-process worker.

        ``False`` means another worker already claimed it or the job is
        terminal. Ownership and missing-job errors remain explicit.
        """

        self._init_schema()
        with self._db_lock, _db_connection() as connection:
            row = connection.execute(
                "SELECT requested_by, status FROM generation_jobs WHERE job_id = ?",
                (job_id.strip(),),
            ).fetchone()
            if row is None:
                raise BatchJobNotFoundError("指定されたjob_idが見つかりません。")
            if str(row["requested_by"]) != str(requested_by):
                raise BatchJobOwnershipError("他のユーザーのjobは操作できません。")
            if str(row["status"]) != JOB_QUEUED:
                return False
            cursor = connection.execute(
                "UPDATE generation_jobs SET status = ?, started_at = ? "
                "WHERE job_id = ? AND status = ?",
                (JOB_RUNNING, _now_iso(), job_id.strip(), JOB_QUEUED),
            )
            return cursor.rowcount == 1

    def update_progress(
        self,
        job_id: str,
        requested_by: int,
        *,
        completed_count: int,
        failed_count: int,
        current_batch_id: str = "",
        current_prompt: str = "",
        error: str = "",
    ) -> dict[str, Any]:
        row = self._require_owned_row(job_id, requested_by)
        if str(row["status"]) not in ACTIVE_JOB_STATES:
            raise BatchJobError("終了済みjobの進捗は更新できません。")
        total = int(row["total_count"])
        if (
            isinstance(completed_count, bool)
            or isinstance(failed_count, bool)
            or not isinstance(completed_count, int)
            or not isinstance(failed_count, int)
            or completed_count < 0
            or failed_count < 0
            or completed_count + failed_count > total
        ):
            raise ValueError("job進捗が不正です。")
        with self._db_lock, _db_connection() as connection:
            connection.execute(
                """UPDATE generation_jobs
                   SET completed_count = ?, failed_count = ?, current_batch_id = ?,
                       current_prompt = ?, error = ?
                   WHERE job_id = ?""",
                (
                    completed_count,
                    failed_count,
                    str(current_batch_id or ""),
                    str(current_prompt or ""),
                    _stored_error_text(error) if error else "",
                    job_id.strip(),
                ),
            )
        return self.get_job_for_user(job_id, requested_by)

    def finish_job(
        self,
        job_id: str,
        requested_by: int,
        *,
        status: str,
        error: str = "",
    ) -> dict[str, Any]:
        if status not in TERMINAL_JOB_STATES:
            raise ValueError("終了状態はcompleted / failed / cancelledのいずれかです。")
        row = self._require_owned_row(job_id, requested_by)
        if str(row["status"]) in TERMINAL_JOB_STATES:
            if str(row["status"]) != status:
                raise BatchJobError("jobは別の終了状態になっています。")
            return self.get_job_for_user(job_id, requested_by)
        with self._db_lock, _db_connection() as connection:
            connection.execute(
                "UPDATE generation_jobs SET status = ?, finished_at = ?, error = ?, "
                "current_batch_id = '', current_prompt = '' WHERE job_id = ?",
                (status, _now_iso(), _stored_error_text(error) if error else "", job_id.strip()),
            )
        return self.get_job_for_user(job_id, requested_by)

    def cancel_job(self, job_id: str, requested_by: int) -> tuple[bool, str]:
        row = self._require_owned_row(job_id, requested_by)
        state = str(row["status"])
        if state in TERMINAL_JOB_STATES:
            return False, f"jobはすでに{state}です。"
        with self._db_lock, _db_connection() as connection:
            if state == JOB_QUEUED:
                connection.execute(
                    "UPDATE generation_jobs SET status = ?, cancel_requested = 1, finished_at = ? "
                    "WHERE job_id = ? AND status = ?",
                    (JOB_CANCELLED, _now_iso(), job_id.strip(), JOB_QUEUED),
                )
                return True, "キュー待ちのjobをキャンセルしました。生成済み画像はありません。"
            connection.execute(
                "UPDATE generation_jobs SET cancel_requested = 1 WHERE job_id = ? AND status = ?",
                (job_id.strip(), JOB_RUNNING),
            )
        return True, "キャンセルを受け付けました。現在の処理が終わり次第、未処理分を停止します。"

    def repeat_job(self, source_job_id: str, requested_by: int) -> str:
        source = self._require_owned_row(source_job_id, requested_by)
        request = _request_from_row(source)
        repeated = replace(
            request,
            requested_by=requested_by,
            repeat_of_job_id=source_job_id.strip(),
            generation_config={**request.generation_config, "seed": -1},
        )
        return self.create_job(repeated)

    def create_batch(
        self,
        job_id: str,
        requested_by: int,
        *,
        batch_id: str | None = None,
        prompt: str = "",
        negative_prompt: str = "",
        character: str = "",
        loras: Iterable[LoraSelection] = (),
        checkpoint: str = "",
        requested_count: int,
        variation_index: int = 1,
    ) -> str:
        row = self._require_owned_row(job_id, requested_by)
        if str(row["status"]) in TERMINAL_JOB_STATES:
            raise BatchJobError("終了済みjobにはbatchを追加できません。")
        if isinstance(requested_count, bool) or not isinstance(requested_count, int) or requested_count < 1:
            raise ValueError("requested_countは1以上の整数で指定してください。")
        if isinstance(variation_index, bool) or not isinstance(variation_index, int) or variation_index < 1:
            raise ValueError("variation_indexは1以上の整数で指定してください。")
        serialized_loras = json.dumps(_serialize_loras(loras), ensure_ascii=False)
        resolved_batch_id = (batch_id or uuid.uuid4().hex[:16]).strip()
        if not resolved_batch_id:
            raise ValueError("batch_idが不正です。")
        with self._db_lock, _db_connection() as connection:
            connection.execute(
                """INSERT INTO generation_batches
                   (batch_id, job_id, prompt, negative_prompt, character, loras_json,
                    checkpoint, requested_count, variation_index, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    resolved_batch_id,
                    job_id.strip(),
                    str(prompt or ""),
                    str(negative_prompt or ""),
                    str(character or ""),
                    serialized_loras,
                    str(checkpoint or ""),
                    requested_count,
                    variation_index,
                    _now_iso(),
                ),
            )
        return resolved_batch_id

    def update_batch_progress(
        self,
        batch_id: str,
        requested_by: int,
        *,
        completed_count: int,
        failed_count: int = 0,
        error: str = "",
    ) -> None:
        self._init_schema()
        with self._db_lock, _db_connection() as connection:
            row = connection.execute(
                """SELECT b.requested_count, j.requested_by
                   FROM generation_batches b JOIN generation_jobs j ON j.job_id = b.job_id
                   WHERE b.batch_id = ?""",
                (batch_id.strip(),),
            ).fetchone()
            if row is None:
                raise BatchJobNotFoundError("指定されたbatch_idが見つかりません。")
            if str(row["requested_by"]) != str(requested_by):
                raise BatchJobOwnershipError("他のユーザーのbatchは操作できません。")
            if (
                isinstance(completed_count, bool)
                or not isinstance(completed_count, int)
                or completed_count < 0
                or isinstance(failed_count, bool)
                or not isinstance(failed_count, int)
                or failed_count < 0
                or completed_count + failed_count > int(row["requested_count"])
            ):
                raise ValueError("batch進捗が不正です。")
            connection.execute(
                "UPDATE generation_batches SET completed_count = ?, failed_count = ?, error = ? "
                "WHERE batch_id = ?",
                (
                    completed_count,
                    failed_count,
                    _stored_error_text(error) if error else "",
                    batch_id.strip(),
                ),
            )

    def get_batch_specs_for_user(
        self,
        job_id: str,
        requested_by: int,
    ) -> tuple[BatchSpec, ...]:
        """Return persisted Batch specs in deterministic variation order."""

        job = self.get_job_for_user(job_id, requested_by)
        specs: list[BatchSpec] = []
        for batch in sorted(
            job["batches"],
            key=lambda item: (
                int(item.get("variation_index") or 1),
                str(item.get("created_at") or ""),
                str(item.get("batch_id") or ""),
            ),
        ):
            specs.append(
                BatchSpec(
                    batch_id=str(batch["batch_id"]),
                    job_id=str(batch["job_id"]),
                    prompt=str(batch.get("prompt") or ""),
                    negative_prompt=str(batch.get("negative_prompt") or ""),
                    character=str(batch.get("character") or ""),
                    loras=_deserialize_loras(batch.get("loras", [])),
                    checkpoint=str(batch.get("checkpoint") or ""),
                    requested_count=int(batch.get("requested_count") or 0),
                    created_at=str(batch.get("created_at") or ""),
                    variation_index=int(batch.get("variation_index") or 1),
                )
            )
        return tuple(specs)

    def recover_incomplete_jobs(self) -> int:
        """Mark jobs from a previous process as failed; never resume them."""

        self._init_schema()
        with self._db_lock, _db_connection() as connection:
            cursor = connection.execute(
                """UPDATE generation_jobs
                   SET status = ?, finished_at = ?,
                       error = 'Bot再起動時に実行中だったため終了しました。'
                   WHERE status IN (?, ?)""",
                (JOB_FAILED, _now_iso(), JOB_QUEUED, JOB_RUNNING),
            )
            return int(cursor.rowcount)


manager = BatchGenerationManager()


def recover_incomplete_jobs() -> int:
    """Compatibility wrapper for the future Bot startup hook."""

    return manager.recover_incomplete_jobs()


async def shutdown() -> None:
    """Reserved lifecycle hook; no worker is started by this commit."""

    return None
