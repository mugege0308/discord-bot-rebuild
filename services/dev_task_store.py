from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

from services.dev_task_router import DevTaskEngine


logger = logging.getLogger("secretary-bot.dev_task_store")

DATA_DIR = Path("data/dev_tasks")

_store_lock = threading.Lock()


class DevTaskStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Phase 3+ can extend this table with new statuses/transitions without touching
# update_task()'s logic. A status with no entry (or an empty set) accepts no
# further transitions, including transitioning to itself.
DEV_TASK_STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    DevTaskStatus.PENDING_APPROVAL.value: frozenset(
        {DevTaskStatus.RUNNING.value, DevTaskStatus.FAILED.value}
    ),
    DevTaskStatus.RUNNING.value: frozenset(
        {DevTaskStatus.COMPLETED.value, DevTaskStatus.FAILED.value}
    ),
    DevTaskStatus.COMPLETED.value: frozenset(),
    DevTaskStatus.FAILED.value: frozenset(),
}

# Fields update_task() may change, and fields it must always reject.
_UPDATABLE_FIELDS = frozenset(
    {
        "status",
        "error_message",
        "routed_engine",
        "routing_reason",
        "engine_overridden",
        "timeout_seconds",
    }
)
_IMMUTABLE_FIELDS = frozenset(
    {
        "task_id",
        "requested_by_user_id",
        "guild_id",
        "channel_id",
        "requested_at",
        "prompt_text",
    }
)

# Per-field audit lock: which of the task's *current* statuses a given
# updatable field may still be changed from. A field with no entry here
# (currently only "status") is governed solely by DEV_TASK_STATUS_TRANSITIONS
# instead of a simple current-status membership check.
#
# routed_engine / routing_reason / engine_overridden / timeout_seconds record
# the routing decision made at task creation and are locked as soon as the
# task leaves pending_approval, so they can never be rewritten once a run has
# started. error_message may still be attached while the task is on its way
# to a terminal state (e.g. running -> failed), but is locked once the task
# reaches a terminal status (completed/failed), preserving it as an audit
# record of how the task actually finished.
FIELD_ALLOWED_STATUSES: dict[str, frozenset[str]] = {
    "routed_engine": frozenset({DevTaskStatus.PENDING_APPROVAL.value}),
    "routing_reason": frozenset({DevTaskStatus.PENDING_APPROVAL.value}),
    "engine_overridden": frozenset({DevTaskStatus.PENDING_APPROVAL.value}),
    "timeout_seconds": frozenset({DevTaskStatus.PENDING_APPROVAL.value}),
    "error_message": frozenset({DevTaskStatus.PENDING_APPROVAL.value, DevTaskStatus.RUNNING.value}),
}

_VALID_STATUS_VALUES = {member.value for member in DevTaskStatus}
_VALID_ENGINE_VALUES = {member.value for member in DevTaskEngine}

# Discord snowflake ids are 64-bit unsigned integers; used for every
# user/guild/channel id field validated in this module.
MAX_SNOWFLAKE_ID = 2**64 - 1

# Operational ceiling on how long a dev task run may be allowed to take.
MAX_DEV_TASK_TIMEOUT_SECONDS = 86400

# Stable, machine-checkable reasons so callers can branch on why an operation
# failed instead of parsing the Discord-facing Japanese message.
ERROR_INVALID_TASK_ID = "invalid_task_id"
ERROR_TASK_NOT_FOUND = "task_not_found"
ERROR_READ_FAILED = "read_failed"
ERROR_CORRUPTED = "corrupted"
ERROR_UNKNOWN_FIELD = "unknown_field"
ERROR_IMMUTABLE_FIELD = "immutable_field"
ERROR_INVALID_VALUE = "invalid_value"
ERROR_INVALID_TRANSITION = "invalid_transition"
ERROR_FIELD_LOCKED = "field_locked"
ERROR_SAVE_FAILED = "save_failed"

_ERROR_MESSAGES: dict[str, str] = {
    ERROR_INVALID_TASK_ID: "タスクIDの形式が不正です。",
    ERROR_TASK_NOT_FOUND: "タスクが見つかりません。",
    ERROR_READ_FAILED: "タスクデータの読み込みに失敗しました。",
    ERROR_CORRUPTED: "タスクデータが壊れています。",
    ERROR_UNKNOWN_FIELD: "更新できない項目が指定されています。",
    ERROR_IMMUTABLE_FIELD: "変更できない項目が指定されています。",
    ERROR_INVALID_VALUE: "更新内容が不正です。",
    ERROR_INVALID_TRANSITION: "その状態遷移は許可されていません。",
    ERROR_FIELD_LOCKED: "現在の状態ではこの項目は変更できません。",
    ERROR_SAVE_FAILED: "タスクの保存に失敗しました。",
}


@dataclass(frozen=True)
class DevTask:
    task_id: str
    status: str
    requested_by_user_id: int
    guild_id: int | None
    channel_id: int
    requested_at: str
    prompt_text: str
    routed_engine: str
    routing_reason: str
    engine_overridden: bool
    timeout_seconds: int
    error_message: str | None = None


@dataclass(frozen=True)
class DevTaskResult:
    success: bool
    task: DevTask | None
    error_message: str | None = None
    error_code: str | None = None


def _result_error(code: str) -> DevTaskResult:
    return DevTaskResult(success=False, task=None, error_message=_ERROR_MESSAGES[code], error_code=code)


def _task_path(task_id: str) -> Path:
    return DATA_DIR / f"{task_id}.json"


def _normalize_task_id(task_id: Any) -> str | None:
    """Parse `task_id` as a strict UUID4 string.

    This is the single UUID4 validation path used for every source of a task
    id in this module: user-supplied ids, JSON-embedded task_id fields, and
    filenames read back off disk. A syntactically valid UUID of any other
    version (UUID1/3/5) or variant is rejected, not just malformed strings.

    Returns the canonical lowercase, hyphenated string form on success, or
    None if `task_id` is not a valid UUID4.
    """
    try:
        candidate = uuid.UUID(str(task_id))
    except (ValueError, AttributeError, TypeError):
        return None
    if candidate.version != 4 or candidate.variant != uuid.RFC_4122:
        return None
    return str(candidate)


def _generate_task_id() -> str:
    """Generate a new task id through the exact same UUID4 validation path
    used for every other id source, so create_task() never bypasses it even
    for locally generated ids."""
    candidate = str(uuid.uuid4())
    validated = _normalize_task_id(candidate)
    if validated is None:  # pragma: no cover - uuid.uuid4() always yields a UUID4
        raise RuntimeError("uuid.uuid4() produced a value that failed UUID4 validation")
    return validated


def _is_strict_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_strict_int(value: Any) -> bool:
    # bool is a subclass of int in Python, so it must be excluded explicitly
    # or True/False would silently pass as 0/1.
    return isinstance(value, int) and not isinstance(value, bool)


def _is_utf8_safe_str(value: Any) -> bool:
    """True only for a str that can round-trip through the UTF-8 encoding
    _write_task_atomic() uses to persist every record.

    A value that reads back from json.loads() as a str but contains an
    unpaired surrogate (e.g. produced by a raw "\\udXXX" escape in the JSON
    text) is not valid UTF-8 text and must be treated as corrupted data, not
    a legitimate string, everywhere a str field is accepted: stored-record
    parsing, create_task()/update_task() input, and update_task()'s merge of
    the two.
    """
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _is_valid_status(value: Any) -> bool:
    return _is_utf8_safe_str(value) and value in _VALID_STATUS_VALUES


def _is_valid_engine(value: Any) -> bool:
    return _is_utf8_safe_str(value) and value in _VALID_ENGINE_VALUES


def _is_valid_timeout_seconds(value: Any) -> bool:
    return _is_strict_int(value) and 1 <= value <= MAX_DEV_TASK_TIMEOUT_SECONDS


def _is_valid_snowflake_id(value: Any) -> bool:
    # Discord snowflake ids (user/guild/channel) are always positive
    # integers within the 64-bit unsigned range.
    return _is_strict_int(value) and 0 < value <= MAX_SNOWFLAKE_ID


def _is_valid_optional_snowflake_id(value: Any) -> bool:
    return value is None or _is_valid_snowflake_id(value)


def _parse_requested_at(value: Any) -> datetime | None:
    """Parse `value` as the ISO 8601 UTC timestamp requested_at must be,
    returning an aware, UTC-normalized datetime on success.

    This is the single parsing path shared by _is_valid_requested_at()
    (existence/validity check) and get_latest_task_for_user() (actual
    chronological comparison), so "valid" and "comparable" never disagree.
    create_task() always writes datetime.now(timezone.utc).isoformat(), so a
    naive or non-UTC-offset timestamp is corrupted data rather than a
    legitimate variant this module ever produces. Never raises: any
    unparseable input yields None.
    """
    if not _is_utf8_safe_str(value):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.utcoffset() != timedelta(0):
        return None
    return parsed.astimezone(timezone.utc)


def _is_valid_requested_at(value: Any) -> bool:
    return _parse_requested_at(value) is not None


def _is_valid_error_message(value: Any) -> bool:
    return value is None or _is_utf8_safe_str(value)


def _is_valid_status_error_message_pair(status: Any, error_message: Any) -> bool:
    """Enforce the invariant between a task's terminal status and its
    error_message, on top of each field's own per-field validation:

    - completed tasks must never carry an error_message (must be None)
    - failed tasks must always explain why (a non-empty string)
    - pending_approval/running are unconstrained beyond the individual
      error_message validator (None or any valid str)

    Called only after `status` and `error_message` have already individually
    passed _is_valid_status()/_is_valid_error_message(), so this only needs
    to check the pairing, not re-validate either field's own type/shape.
    """
    if status == DevTaskStatus.COMPLETED.value:
        return error_message is None
    if status == DevTaskStatus.FAILED.value:
        return isinstance(error_message, str) and error_message != ""
    return True


# Single source of truth for the semantic type/value of every DevTask field
# except task_id, which is validated separately via _normalize_task_id() and
# cross-checked against the filename (see _load_task()). Used, without
# coercion, by record parsing (_task_from_dict), field-level update
# validation (_validate_update_fields), and task creation
# (_validate_create_fields) so all three paths agree on what "valid" means.
_FIELD_VALIDATORS: dict[str, Callable[[Any], bool]] = {
    "status": _is_valid_status,
    "requested_by_user_id": _is_valid_snowflake_id,
    "guild_id": _is_valid_optional_snowflake_id,
    "channel_id": _is_valid_snowflake_id,
    "requested_at": _is_valid_requested_at,
    "prompt_text": _is_utf8_safe_str,
    "routed_engine": _is_valid_engine,
    "routing_reason": _is_utf8_safe_str,
    "engine_overridden": _is_strict_bool,
    "timeout_seconds": _is_valid_timeout_seconds,
    "error_message": _is_valid_error_message,
}

_CREATE_TASK_FIELDS = frozenset(
    {
        "requested_by_user_id",
        "guild_id",
        "channel_id",
        "prompt_text",
        "routed_engine",
        "routing_reason",
        "engine_overridden",
        "timeout_seconds",
    }
)


def _normalize_field_values(fields: dict[str, Any]) -> dict[str, Any]:
    """Normalize any str Enum member (DevTaskStatus, DevTaskEngine, ...) in
    `fields` to its plain str .value.

    Public callers of create_task()/update_task() may pass either a str
    Enum member or a plain str for fields like routed_engine/status - both
    validate successfully since these Enums subclass str. But the DevTask
    dataclass and its on-disk JSON form must only ever store plain str, so
    that a task's field type is identical immediately after creation and
    after being reloaded from disk. Values that aren't Enum members pass
    through unchanged.
    """
    return {key: (value.value if isinstance(value, Enum) else value) for key, value in fields.items()}


def _task_from_dict(normalized_task_id: str, data: dict[str, Any]) -> DevTask | None:
    """Strictly parse a stored task record for the already-verified
    `normalized_task_id`.

    No implicit type coercion is performed: a field with the wrong Python
    type (e.g. engine_overridden="false", timeout_seconds=True) is rejected
    rather than coerced, since a value that round-tripped through JSON with
    the wrong type indicates corrupted data rather than a legitimate variant.
    `normalized_task_id` is used as-is for the returned DevTask's task_id;
    callers must have already established it via filename/content
    cross-checking (see _load_task()) before calling this function.
    """
    for field_name, validator in _FIELD_VALIDATORS.items():
        if field_name not in data:
            logger.error(
                "Dev task JSON missing field: task_id=%s field=%s", normalized_task_id, field_name
            )
            return None
        if not validator(data[field_name]):
            logger.error(
                "Dev task JSON has invalid field: task_id=%s field=%s", normalized_task_id, field_name
            )
            return None

    if not _is_valid_status_error_message_pair(data["status"], data["error_message"]):
        logger.error(
            "Dev task JSON violates status/error_message invariant: task_id=%s status=%s",
            normalized_task_id,
            data["status"],
        )
        return None

    return DevTask(
        task_id=normalized_task_id,
        status=data["status"],
        requested_by_user_id=data["requested_by_user_id"],
        guild_id=data["guild_id"],
        channel_id=data["channel_id"],
        requested_at=data["requested_at"],
        prompt_text=data["prompt_text"],
        routed_engine=data["routed_engine"],
        routing_reason=data["routing_reason"],
        engine_overridden=data["engine_overridden"],
        timeout_seconds=data["timeout_seconds"],
        error_message=data["error_message"],
    )


def _read_task_file(path: Path) -> dict[str, Any] | None:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        logger.exception("Failed to read dev task file: path=%s", path)
        return None
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.exception("Failed to parse dev task JSON: path=%s", path)
        return None
    if not isinstance(value, dict):
        logger.error("Dev task JSON is not an object: path=%s", path)
        return None
    return value


def _load_task(normalized_id: str) -> tuple[str | None, DevTask | None]:
    """Load, verify, and strictly parse the single task file for a
    filename-derived, already-UUID4-validated `normalized_id`.

    This is the one common path get_task(), update_task(),
    get_latest_task_for_user(), and mark_running_tasks_failed_on_startup()
    all use to turn a verified id into a DevTask: the filename-derived id is
    always the identifier used to build the filesystem path, and the JSON
    body's task_id is untrusted content that must match it before the record
    is accepted. There is no code path anywhere in this module that derives
    a filesystem path from a JSON-embedded task_id.

    Returns (None, task) on success, or (error_code, None) on failure where
    error_code is one of ERROR_TASK_NOT_FOUND / ERROR_READ_FAILED /
    ERROR_CORRUPTED.
    """
    path = _task_path(normalized_id)
    if not path.exists():
        return ERROR_TASK_NOT_FOUND, None

    data = _read_task_file(path)
    if data is None:
        return ERROR_READ_FAILED, None

    content_task_id = _normalize_task_id(data.get("task_id"))
    if content_task_id is None or content_task_id != normalized_id:
        logger.error(
            "Rejecting dev task file with mismatched or invalid content task_id: path=%s content_task_id=%r",
            path,
            data.get("task_id"),
        )
        return ERROR_CORRUPTED, None

    task = _task_from_dict(normalized_id, data)
    if task is None:
        return ERROR_CORRUPTED, None

    return None, task


def _write_task_atomic(task_id: str, payload: dict[str, Any]) -> bool:
    """Serialize `payload` as JSON and atomically replace the on-disk record
    for `task_id`.

    Serialization (json.dumps + UTF-8 encoding) and filesystem operations are
    handled as two separate, independently-caught stages: a payload
    containing something JSON can't serialize (TypeError/ValueError) or that
    can't be UTF-8 encoded (e.g. a lone surrogate, UnicodeError) fails before
    any file is touched, while OSError covers directory creation, temp file
    creation/write, and the final replace. Every temp file used here is
    created by this call with a unique random name (tempfile.mkstemp) inside
    DATA_DIR, and only that exact temp file is ever removed on failure - an
    unrelated stray .tmp file, or one created by a concurrent call, is never
    touched.

    The raw fd returned by tempfile.mkstemp() is tracked separately from the
    file object os.fdopen() wraps it in: if os.fdopen() itself fails, the fd
    was never handed off to anything that would close it on cleanup, so it
    is closed explicitly here. On Windows in particular, leaving that fd
    open would keep the temp file locked and make the tmp_path.unlink()
    cleanup below fail.
    """
    target_path = _task_path(task_id)

    try:
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        logger.exception("Failed to serialize dev task to JSON: task_id=%s", task_id)
        return False

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Failed to create dev task data directory: task_id=%s", task_id)
        return False

    fd: int | None = None
    tmp_path: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=f"{task_id}.", suffix=".tmp", dir=str(DATA_DIR))
        tmp_path = Path(tmp_name)
        tmp_file = os.fdopen(fd, "wb")
        fd = None  # os.fdopen() succeeded: tmp_file now owns closing the fd.
        with tmp_file:
            tmp_file.write(encoded)
        os.replace(tmp_path, target_path)
        tmp_path = None
        return True
    except OSError:
        logger.exception("Failed to save dev task: task_id=%s", task_id)
        return False
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                logger.exception(
                    "Failed to close dev task temp file descriptor: task_id=%s fd=%s", task_id, fd
                )
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                logger.exception(
                    "Failed to clean up dev task temp file: task_id=%s path=%s", task_id, tmp_path
                )


def _iter_stored_tasks() -> Iterable[tuple[str, DevTask]]:
    """Yield (verified_task_id, task) for every file in DATA_DIR that passes
    the same filename/content UUID4 consistency check and strict field
    validation as _load_task() - the same common verification path get_task()
    and update_task() use for single-record lookups.
    """
    for path in sorted(DATA_DIR.glob("*.json")):
        filename_task_id = _normalize_task_id(path.stem)
        if filename_task_id is None:
            logger.warning("Skipping dev task file with non-UUID4 filename: path=%s", path)
            continue

        error_code, task = _load_task(filename_task_id)
        if error_code is not None:
            logger.warning("Skipping dev task file during scan: path=%s reason=%s", path, error_code)
            continue

        yield filename_task_id, task


def _validate_update_fields(fields: dict[str, Any]) -> str | None:
    for key in fields:
        if key in _IMMUTABLE_FIELDS:
            return ERROR_IMMUTABLE_FIELD
        if key not in _UPDATABLE_FIELDS:
            return ERROR_UNKNOWN_FIELD

    for key, value in fields.items():
        if not _FIELD_VALIDATORS[key](value):
            return ERROR_INVALID_VALUE

    return None


def _validate_create_fields(fields: dict[str, Any]) -> str | None:
    for key in _CREATE_TASK_FIELDS:
        if not _FIELD_VALIDATORS[key](fields[key]):
            return ERROR_INVALID_VALUE
    return None


def _validate_field_locks(current_status: str, fields: dict[str, Any]) -> str | None:
    """Check every governed field in `fields` against FIELD_ALLOWED_STATUSES
    for the task's current status, before any field is written. All-or­-
    nothing: the first field that isn't editable in the current status
    rejects the whole update and nothing is written.
    """
    for field_name in fields:
        allowed_statuses = FIELD_ALLOWED_STATUSES.get(field_name)
        if allowed_statuses is not None and current_status not in allowed_statuses:
            return ERROR_FIELD_LOCKED
    return None


def _is_allowed_transition(current_status: str, new_status: str) -> bool:
    return new_status in DEV_TASK_STATUS_TRANSITIONS.get(current_status, frozenset())


def create_task(
    *,
    requested_by_user_id: int,
    guild_id: int | None,
    channel_id: int,
    prompt_text: str,
    routed_engine: str,
    routing_reason: str,
    engine_overridden: bool,
    timeout_seconds: int,
) -> DevTaskResult:
    candidate_fields = _normalize_field_values(
        {
            "requested_by_user_id": requested_by_user_id,
            "guild_id": guild_id,
            "channel_id": channel_id,
            "prompt_text": prompt_text,
            "routed_engine": routed_engine,
            "routing_reason": routing_reason,
            "engine_overridden": engine_overridden,
            "timeout_seconds": timeout_seconds,
        }
    )
    field_error_code = _validate_create_fields(candidate_fields)
    if field_error_code is not None:
        logger.warning("Rejected dev task creation: reason=%s", field_error_code)
        return _result_error(field_error_code)

    task_id = _generate_task_id()
    task = DevTask(
        task_id=task_id,
        status=DevTaskStatus.PENDING_APPROVAL.value,
        requested_by_user_id=candidate_fields["requested_by_user_id"],
        guild_id=candidate_fields["guild_id"],
        channel_id=candidate_fields["channel_id"],
        requested_at=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        prompt_text=candidate_fields["prompt_text"],
        routed_engine=candidate_fields["routed_engine"],
        routing_reason=candidate_fields["routing_reason"],
        engine_overridden=candidate_fields["engine_overridden"],
        timeout_seconds=candidate_fields["timeout_seconds"],
        error_message=None,
    )

    with _store_lock:
        saved = _write_task_atomic(task_id, asdict(task))
    if not saved:
        return _result_error(ERROR_SAVE_FAILED)

    logger.info(
        "Dev task created: task_id=%s user_id=%s engine=%s",
        task_id,
        candidate_fields["requested_by_user_id"],
        candidate_fields["routed_engine"],
    )
    return DevTaskResult(success=True, task=task, error_message=None, error_code=None)


def get_task(task_id: str) -> DevTaskResult:
    normalized_id = _normalize_task_id(task_id)
    if normalized_id is None:
        return _result_error(ERROR_INVALID_TASK_ID)

    error_code, task = _load_task(normalized_id)
    if error_code is not None:
        return _result_error(error_code)

    return DevTaskResult(success=True, task=task, error_message=None, error_code=None)


def update_task(task_id: str, **fields: Any) -> DevTaskResult:
    normalized_id = _normalize_task_id(task_id)
    if normalized_id is None:
        return _result_error(ERROR_INVALID_TASK_ID)

    fields = _normalize_field_values(fields)

    field_error_code = _validate_update_fields(fields)
    if field_error_code is not None:
        logger.warning(
            "Rejected dev task update: task_id=%s reason=%s fields=%s",
            normalized_id,
            field_error_code,
            sorted(fields.keys()),
        )
        return _result_error(field_error_code)

    with _store_lock:
        error_code, current_task = _load_task(normalized_id)
        if error_code is not None:
            return _result_error(error_code)

        if "status" in fields:
            new_status = fields["status"]
            if not _is_allowed_transition(current_task.status, new_status):
                logger.warning(
                    "Rejected dev task status transition: task_id=%s from=%s to=%s",
                    normalized_id,
                    current_task.status,
                    new_status,
                )
                return _result_error(ERROR_INVALID_TRANSITION)

        lock_error_code = _validate_field_locks(current_task.status, fields)
        if lock_error_code is not None:
            logger.warning(
                "Rejected dev task update due to field lock: task_id=%s status=%s fields=%s",
                normalized_id,
                current_task.status,
                sorted(fields.keys()),
            )
            return _result_error(lock_error_code)

        merged = asdict(current_task)
        merged.update(fields)
        updated_task = _task_from_dict(normalized_id, merged)
        if updated_task is None:
            return _result_error(ERROR_INVALID_VALUE)

        saved = _write_task_atomic(normalized_id, asdict(updated_task))
        if not saved:
            return _result_error(ERROR_SAVE_FAILED)

    logger.info("Dev task updated: task_id=%s fields=%s", normalized_id, sorted(fields.keys()))
    return DevTaskResult(success=True, task=updated_task, error_message=None, error_code=None)


def get_latest_task_for_user(user_id: int) -> DevTaskResult:
    if not _is_valid_snowflake_id(user_id):
        logger.warning("Rejected get_latest_task_for_user: invalid user_id=%r", user_id)
        return _result_error(ERROR_INVALID_VALUE)

    if not DATA_DIR.exists():
        return _result_error(ERROR_TASK_NOT_FOUND)

    latest_task: DevTask | None = None
    latest_requested_at: datetime | None = None
    for _verified_task_id, task in _iter_stored_tasks():
        if task.requested_by_user_id != user_id:
            continue
        # requested_at was already validated by _iter_stored_tasks() ->
        # _load_task() -> _task_from_dict(), so this should always parse;
        # skip defensively rather than let a comparison against None escape.
        parsed_requested_at = _parse_requested_at(task.requested_at)
        if parsed_requested_at is None:
            logger.warning(
                "Skipping dev task with unparseable requested_at in get_latest_task_for_user: task_id=%s",
                task.task_id,
            )
            continue
        if latest_task is None or parsed_requested_at > latest_requested_at:
            latest_task = task
            latest_requested_at = parsed_requested_at

    if latest_task is None:
        return _result_error(ERROR_TASK_NOT_FOUND)
    return DevTaskResult(success=True, task=latest_task, error_message=None, error_code=None)


def mark_running_tasks_failed_on_startup() -> int:
    if not DATA_DIR.exists():
        return 0

    changed_count = 0
    with _store_lock:
        for verified_task_id, task in _iter_stored_tasks():
            if task.status != DevTaskStatus.RUNNING.value:
                continue

            failed_payload = asdict(task)
            failed_payload["status"] = DevTaskStatus.FAILED.value
            failed_payload["error_message"] = "Bot再起動により処理が中断されました。"

            if _write_task_atomic(verified_task_id, failed_payload):
                changed_count += 1
            else:
                logger.error("Failed to mark dev task as failed on startup: task_id=%s", verified_task_id)

    logger.info("Dev tasks marked failed on startup: count=%s", changed_count)
    return changed_count
