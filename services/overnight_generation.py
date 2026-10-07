"""Durable state primitives for long-running overnight generation jobs.

The module owns orchestration state only.  Image preparation, ComfyUI
submission, and AI selection remain in the existing generation/selection
services and are injected by ``generation_api``.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Mapping
import uuid

from services.runtime_paths import resolve_project_path


OVERNIGHT_TOTALS = frozenset({128, 256, 512})
OVERNIGHT_DEFAULT_TOTAL = 256
OVERNIGHT_ROUND_SIZE = 32
OVERNIGHT_MAX_RETRIES = 2
OVERNIGHT_MAX_CONSECUTIVE_INFRA_FAILURES = 3
OVERNIGHT_ACTIVE_STATES = frozenset({"queued", "running", "selecting", "paused", "paused_error", "stopping"})
OVERNIGHT_TERMINAL_STATES = frozenset({"completed", "completed_selection_failed", "stopped", "failed"})


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_overnight_db_path() -> Path:
    raw = os.getenv("BATCH_GENERATION_DB_PATH", "").strip().strip('"')
    return resolve_project_path(raw or None, "data/batch_generation.db")


def _json_load(value: str, default: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class OvernightJobError(RuntimeError):
    """Raised when durable overnight state cannot be created or updated."""


class OvernightJobStore:
    """Small SQLite repository using the existing batch runtime database."""

    _JSON_COLUMNS = {
        "request_json",
        "errors_json",
        "rounds_json",
        "final_selected_json",
        "timing_json",
        "in_flight_json",
    }
    _COLUMNS = frozenset(
        {
            "request_json",
            "total_count",
            "round_size",
            "total_rounds",
            "current_round",
            "current_slot",
            "generated_count",
            "failed_count",
            "state",
            "selection_state",
            "error_summary",
            "errors_json",
            "rounds_json",
            "final_selected_json",
            "timing_json",
            "started_at",
            "updated_at",
            "finished_at",
            "pause_requested",
            "stop_requested",
            "in_flight_json",
        }
    )

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = (
            resolve_project_path(db_path, "data/batch_generation.db")
            if db_path is not None
            else resolve_overnight_db_path()
        )
        self._lock = threading.RLock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.db_path), timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _ensure_schema(self) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS overnight_generation_jobs (
                    job_id TEXT PRIMARY KEY,
                    request_json TEXT NOT NULL,
                    total_count INTEGER NOT NULL,
                    round_size INTEGER NOT NULL,
                    total_rounds INTEGER NOT NULL,
                    current_round INTEGER NOT NULL DEFAULT 1,
                    current_slot INTEGER NOT NULL DEFAULT 1,
                    generated_count INTEGER NOT NULL DEFAULT 0,
                    failed_count INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL,
                    selection_state TEXT NOT NULL DEFAULT 'pending',
                    error_summary TEXT NOT NULL DEFAULT '',
                    errors_json TEXT NOT NULL DEFAULT '[]',
                    rounds_json TEXT NOT NULL DEFAULT '[]',
                    final_selected_json TEXT NOT NULL DEFAULT '[]',
                    timing_json TEXT NOT NULL DEFAULT '{}',
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL DEFAULT '',
                    pause_requested INTEGER NOT NULL DEFAULT 0,
                    stop_requested INTEGER NOT NULL DEFAULT 0,
                    in_flight_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(overnight_generation_jobs)").fetchall()
            }
            if "timing_json" not in columns:
                connection.execute(
                    "ALTER TABLE overnight_generation_jobs ADD COLUMN timing_json TEXT NOT NULL DEFAULT '{}'"
                )

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        value = dict(row)
        value["request"] = _json_load(value.pop("request_json"), {})
        value["errors"] = _json_load(value.pop("errors_json"), [])
        value["rounds"] = _json_load(value.pop("rounds_json"), [])
        value["final_selected"] = _json_load(value.pop("final_selected_json"), [])
        value["timing"] = _json_load(value.pop("timing_json"), {})
        value["in_flight"] = _json_load(value.pop("in_flight_json"), {})
        value["pause_requested"] = bool(value["pause_requested"])
        value["stop_requested"] = bool(value["stop_requested"])
        return value

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM overnight_generation_jobs WHERE job_id = ?",
                (str(job_id),),
            ).fetchone()
        return self._decode(row)

    def list_active(self) -> list[dict[str, Any]]:
        placeholders = ",".join("?" for _ in OVERNIGHT_ACTIVE_STATES)
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM overnight_generation_jobs WHERE state IN ({placeholders}) ORDER BY started_at",
                tuple(sorted(OVERNIGHT_ACTIVE_STATES)),
            ).fetchall()
        return [item for item in (self._decode(row) for row in rows) if item is not None]

    def create(self, request: Mapping[str, Any], total_count: int) -> dict[str, Any]:
        if total_count not in OVERNIGHT_TOTALS:
            raise OvernightJobError("夜間生成は128、256、512枚だけ指定できます。")
        active = self.list_active()
        if active:
            raise OvernightJobError("夜間生成は同時に1件だけ実行できます。")
        now = utc_now_iso()
        record = {
            "job_id": uuid.uuid4().hex[:20],
            "request": dict(request),
            "total_count": int(total_count),
            "round_size": OVERNIGHT_ROUND_SIZE,
            "total_rounds": total_count // OVERNIGHT_ROUND_SIZE,
            "current_round": 1,
            "current_slot": 1,
            "generated_count": 0,
            "failed_count": 0,
            "state": "queued",
            "selection_state": "pending",
            "error_summary": "",
            "errors": [],
            "rounds": [],
            "final_selected": [],
            "timing": {},
            "started_at": now,
            "updated_at": now,
            "finished_at": "",
            "pause_requested": False,
            "stop_requested": False,
            "in_flight": {},
        }
        columns = (
            "job_id", "request_json", "total_count", "round_size", "total_rounds",
            "current_round", "current_slot", "generated_count", "failed_count", "state",
            "selection_state", "error_summary", "errors_json", "rounds_json",
            "final_selected_json", "timing_json", "started_at", "updated_at", "finished_at",
            "pause_requested", "stop_requested", "in_flight_json",
        )
        values = (
            record["job_id"], _json_dump(record["request"]), record["total_count"],
            record["round_size"], record["total_rounds"], record["current_round"],
            record["current_slot"], record["generated_count"], record["failed_count"],
            record["state"], record["selection_state"], record["error_summary"],
            _json_dump(record["errors"]), _json_dump(record["rounds"]),
            _json_dump(record["final_selected"]), _json_dump(record["timing"]),
            record["started_at"], record["updated_at"],
            record["finished_at"], int(record["pause_requested"]), int(record["stop_requested"]),
            _json_dump(record["in_flight"]),
        )
        with self._lock, self._connection() as connection:
            connection.execute(
                f"INSERT INTO overnight_generation_jobs ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                values,
            )
        return record

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        unknown = set(changes) - self._COLUMNS - {
            "request",
            "errors",
            "rounds",
            "final_selected",
            "timing",
            "in_flight",
        }
        if unknown:
            raise OvernightJobError(f"夜間生成state項目が不正です: {sorted(unknown)}")
        with self._lock, self._connection() as connection:
            current = self._decode(
                connection.execute(
                    "SELECT * FROM overnight_generation_jobs WHERE job_id = ?",
                    (str(job_id),),
                ).fetchone()
            )
            if current is None:
                raise OvernightJobError("夜間生成jobが見つかりません。")
            updated = dict(current)
            updated.update(changes)
            updated["updated_at"] = utc_now_iso()
            encoded = {
                "request_json": _json_dump(updated["request"]),
                "errors_json": _json_dump(updated["errors"]),
                "rounds_json": _json_dump(updated["rounds"]),
                "final_selected_json": _json_dump(updated["final_selected"]),
                "timing_json": _json_dump(updated["timing"]),
                "in_flight_json": _json_dump(updated["in_flight"]),
                "pause_requested": int(bool(updated["pause_requested"])),
                "stop_requested": int(bool(updated["stop_requested"])),
            }
            for key in self._COLUMNS:
                if key in encoded:
                    continue
                if key in updated:
                    encoded[key] = updated[key]
            assignments = ",".join(f"{key} = ?" for key in encoded)
            connection.execute(
                f"UPDATE overnight_generation_jobs SET {assignments} WHERE job_id = ?",
                tuple(encoded.values()) + (str(job_id),),
            )
        return updated


def overnight_response(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a browser-safe state view without filesystem paths."""

    total = int(record.get("total_count") or 0)
    generated = int(record.get("generated_count") or 0)
    failed = int(record.get("failed_count") or 0)
    rounds = list(record.get("rounds") or [])
    final_selected = list(record.get("final_selected") or [])
    return {
        "overnight_job_id": str(record.get("job_id") or ""),
        "job_id": str(record.get("job_id") or ""),
        "state": str(record.get("state") or "unknown"),
        "progress": (generated + failed) / total if total else 0.0,
        "total_requested": total,
        "round_size": int(record.get("round_size") or OVERNIGHT_ROUND_SIZE),
        "total_rounds": int(record.get("total_rounds") or 0),
        "current_round": int(record.get("current_round") or 1),
        "current_slot": int(record.get("current_slot") or 1),
        "generated_count": generated,
        "failed_count": failed,
        "selection_state": str(record.get("selection_state") or "pending"),
        "error_summary": str(record.get("error_summary") or ""),
        "errors": list(record.get("errors") or [])[:4],
        "rounds": rounds,
        "round_best_count": sum(len(item.get("selected") or []) for item in rounds if isinstance(item, dict)),
        "final_selected": final_selected,
        "pause_requested": bool(record.get("pause_requested")),
        "stop_requested": bool(record.get("stop_requested")),
        "timing": dict(record.get("timing") or {}),
        "started_at": str(record.get("started_at") or ""),
        "updated_at": str(record.get("updated_at") or ""),
        "finished_at": str(record.get("finished_at") or ""),
    }


__all__ = [
    "OVERNIGHT_ACTIVE_STATES",
    "OVERNIGHT_DEFAULT_TOTAL",
    "OVERNIGHT_MAX_CONSECUTIVE_INFRA_FAILURES",
    "OVERNIGHT_MAX_RETRIES",
    "OVERNIGHT_ROUND_SIZE",
    "OVERNIGHT_TERMINAL_STATES",
    "OVERNIGHT_TOTALS",
    "OvernightJobError",
    "OvernightJobStore",
    "overnight_response",
    "resolve_overnight_db_path",
    "utc_now_iso",
]
