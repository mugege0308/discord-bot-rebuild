from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger("secretary-bot.dev_copy_workspace")

# This module never touches services.dev_worktree/services.dev_git_runner
# or any real .git repository at all - it is the Phase 4A-v2 replacement
# environment-prep step for an actual Claude Code trial: instead of
# building a git worktree branched off the real repo, it copies an
# allowlisted subset of plain files into a disposable, independent
# directory that shares nothing (not even a hardlink) with the real
# repository, so a Claude Code run against it can never touch .git,
# secrets, or the real working tree - even by accident, even via a bug in
# Claude's own tool use.
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MAX_FILES = 5000
MIN_MAX_FILES = 1
MAX_MAX_FILES = 200_000

DEFAULT_MAX_TOTAL_BYTES = 524_288_000  # 500 MiB
MIN_MAX_TOTAL_BYTES = 1024  # 1 KiB
MAX_MAX_TOTAL_BYTES = 10 * 1024 * 1024 * 1024  # 10 GiB ceiling, sanity bound only

WORKSPACE_MODE_ISOLATED_COPY = "isolated_copy"
_VALID_WORKSPACE_MODES = frozenset({WORKSPACE_MODE_ISOLATED_COPY})

_BASELINE_SUFFIX = ".baseline.json"

# --- allowlist (checked first) -----------------------------------------

_ALLOWED_EXTENSIONS = frozenset({".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json"})
_ALLOWED_EXACT_FILENAMES = frozenset(
    {
        "requirements.txt",
        "pyproject.toml",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "Pipfile",
        "Pipfile.lock",
        ".env.example",
    }
)

# --- explicit denials (defense in depth on top of the allowlist) -------

_DENIED_EXTENSIONS = frozenset({".key", ".pem", ".pfx", ".p12", ".db", ".sqlite", ".sqlite3", ".log"})

_DENIED_DIR_NAMES_LOWER = frozenset(
    name.lower()
    for name in (
        ".git",
        "data",
        "outputs",
        "logs",
        "work",
        "secrets",
        "credentials",
        "tokens",
        "cache",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "venv",
        ".venv",
        "node_modules",
        "dist",
        "build",
        # Not named explicitly in the operator's deny list, but excluded as
        # defense in depth: these are exactly where Claude Code/Codex/this
        # repo's own tooling keep local session state, auth-adjacent data,
        # or attachments - never something a sandboxed trial copy needs,
        # and exactly the kind of "Claude credential" data the operator
        # asked to keep out.
        ".claude",
        ".codex",
        ".codex-remote-attachments",
        ".agents",
        ".github",
        ".idea",
        ".vscode",
    )
)

_ENV_FILENAME_PATTERN = re.compile(r"^\.env(\..+)?$", re.IGNORECASE)

# --- error codes ---------------------------------------------------------

ERROR_INVALID_TASK_ID = "invalid_task_id"
ERROR_SANDBOX_ROOT_NOT_CONFIGURED = "sandbox_root_not_configured"
ERROR_SANDBOX_ROOT_CONFLICTS_WITH_REPO = "sandbox_root_conflicts_with_repo"
ERROR_WORKSPACE_MODE_INVALID = "workspace_mode_invalid"
ERROR_PATH_BOUNDARY_VIOLATION = "path_boundary_violation"
ERROR_PATH_ALREADY_EXISTS = "path_already_exists"
ERROR_COPY_FAILED = "copy_failed"
ERROR_LIMIT_EXCEEDED = "limit_exceeded"
ERROR_SANDBOX_NOT_FOUND = "sandbox_not_found"
ERROR_FORBIDDEN_CONTENT_FOUND = "forbidden_content_found"
ERROR_VERIFY_FAILED = "verify_failed"
ERROR_EXECUTION_IN_PROGRESS = "execution_in_progress"
ERROR_TERMINATION_UNCONFIRMED = "termination_unconfirmed"
ERROR_DISCARD_FAILED = "discard_failed"

_ERROR_MESSAGES: dict[str, str] = {
    ERROR_INVALID_TASK_ID: "タスクIDの形式が不正です。",
    ERROR_SANDBOX_ROOT_NOT_CONFIGURED: "DEV_TASK_SANDBOX_ROOTが設定されていません。",
    ERROR_SANDBOX_ROOT_CONFLICTS_WITH_REPO: (
        "DEV_TASK_SANDBOX_ROOTが本体リポジトリと重複または包含関係にあるため使用できません。"
    ),
    ERROR_WORKSPACE_MODE_INVALID: "未対応のworkspaceモードが指定されています。",
    ERROR_PATH_BOUNDARY_VIOLATION: "サンドボックスのパスが安全な範囲外を指しているため中止しました。",
    ERROR_PATH_ALREADY_EXISTS: "サンドボックスのパスが既に存在します。",
    ERROR_COPY_FAILED: "サンドボックスへのコピー中にエラーが発生しました。",
    ERROR_LIMIT_EXCEEDED: "ファイル数または合計サイズが上限を超えたため中止しました。",
    ERROR_SANDBOX_NOT_FOUND: "サンドボックスが見つかりません。",
    ERROR_FORBIDDEN_CONTENT_FOUND: "サンドボックス内に許可されない内容が見つかりました。",
    ERROR_VERIFY_FAILED: "サンドボックスの安全確認に失敗しました。",
    ERROR_EXECUTION_IN_PROGRESS: "実行中のため破棄できません。",
    ERROR_TERMINATION_UNCONFIRMED: "プロセス終了が未確認のため破棄できません。",
    ERROR_DISCARD_FAILED: "サンドボックスの削除に失敗しました。",
}


# --- config ---------------------------------------------------------------


def get_sandbox_root() -> Path | None:
    raw_value = os.getenv("DEV_TASK_SANDBOX_ROOT", "").strip()
    if not raw_value:
        return None
    candidate = Path(raw_value)
    if not candidate.is_absolute():
        logger.warning("DEV_TASK_SANDBOX_ROOT must be an absolute path: %r", raw_value)
        return None
    return candidate


def get_max_files() -> int:
    raw_value = os.getenv("DEV_TASK_SANDBOX_MAX_FILES", str(DEFAULT_MAX_FILES)).strip()
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid DEV_TASK_SANDBOX_MAX_FILES: %r", raw_value)
        return DEFAULT_MAX_FILES
    if value < MIN_MAX_FILES or value > MAX_MAX_FILES:
        logger.warning("DEV_TASK_SANDBOX_MAX_FILES out of range: %s. Using default.", value)
        return DEFAULT_MAX_FILES
    return value


def get_max_total_bytes() -> int:
    raw_value = os.getenv("DEV_TASK_SANDBOX_MAX_TOTAL_BYTES", str(DEFAULT_MAX_TOTAL_BYTES)).strip()
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid DEV_TASK_SANDBOX_MAX_TOTAL_BYTES: %r", raw_value)
        return DEFAULT_MAX_TOTAL_BYTES
    if value < MIN_MAX_TOTAL_BYTES or value > MAX_MAX_TOTAL_BYTES:
        logger.warning("DEV_TASK_SANDBOX_MAX_TOTAL_BYTES out of range: %s. Using default.", value)
        return DEFAULT_MAX_TOTAL_BYTES
    return value


def get_workspace_mode() -> str:
    raw_value = os.getenv("CLAUDE_HEADLESS_WORKSPACE_MODE", "").strip()
    return raw_value or WORKSPACE_MODE_ISOLATED_COPY


def is_workspace_mode_valid(mode: str) -> bool:
    return mode in _VALID_WORKSPACE_MODES


# --- task id / path helpers (same UUID4 discipline as dev_worktree) -------


def _normalize_task_id(task_id: Any) -> str | None:
    try:
        candidate = uuid.UUID(str(task_id))
    except (ValueError, AttributeError, TypeError):
        return None
    if candidate.version != 4 or candidate.variant != uuid.RFC_4122:
        return None
    return str(candidate)


def _sandbox_path_for(root: Path, normalized_task_id: str) -> Path:
    return root / normalized_task_id


def _baseline_path_for(root: Path, normalized_task_id: str) -> Path:
    # Stored as a SIBLING of the sandbox directory, never inside it - so a
    # Claude Code run whose tools are confined to the sandbox directory
    # itself can never see or tamper with its own baseline manifest.
    return root / f"{normalized_task_id}{_BASELINE_SUFFIX}"


# --- reparse point / path boundary safety (duplicated from dev_worktree.py
# deliberately, not imported - see that module's own note on why Phase 2's
# hardened code is not refactored to share helpers with new callers) -------


def _path_entry_exists(path: Path) -> bool:
    try:
        return os.path.lexists(path)
    except OSError:
        return True


def _is_reparse_point(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
    except OSError:
        return True

    if os.name == "nt":
        try:
            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))  # type: ignore[attr-defined]
        except OSError:
            return False
        invalid_file_attributes = 0xFFFFFFFF
        file_attribute_reparse_point = 0x400
        if attrs == invalid_file_attributes:
            return False
        return bool(attrs & file_attribute_reparse_point)

    return False


@dataclass(frozen=True)
class _ValidationResult:
    success: bool
    error_message: str | None = None
    error_code: str | None = None


def _validation_error(code: str) -> _ValidationResult:
    return _ValidationResult(success=False, error_message=_ERROR_MESSAGES[code], error_code=code)


def _check_no_reparse_points_in_ancestry(root: Path, target: Path) -> bool:
    if not _path_entry_exists(root):
        return True
    if _is_reparse_point(root):
        return False
    try:
        relative_parts = target.relative_to(root).parts
    except ValueError:
        return False
    current = root
    for part in relative_parts[:-1]:
        current = current / part
        if _path_entry_exists(current) and _is_reparse_point(current):
            return False
    return True


def _resolve_best_effort(path: Path) -> Path | None:
    try:
        return path.resolve(strict=False)
    except OSError:
        return None


def _sandbox_root_conflicts_with_repo_root(root: Path) -> bool:
    """True iff DEV_TASK_SANDBOX_ROOT (`root`) and REPO_ROOT are the same
    directory, or either one is nested inside the other, once BOTH are
    fully resolved via Path.resolve() - never a raw string/prefix
    comparison, which a trailing slash, a relative `..` component, or a
    differently-cased drive letter would trivially defeat.
    REPO_ROOT (this module's own top-of-file constant) is already a fully
    resolved absolute path, so only `root` needs resolving here; on
    Windows, PureWindowsPath equality/`in` is itself case-insensitive, so
    resolving `root` is also what makes "same path, different casing or
    '/' vs '\\\\'" collapse to the same comparison as an exact match,
    with no separate casing-specific code needed.

    Checked in BOTH directions, because each direction breaks a different
    guarantee this module exists to provide:
    - sandbox root inside (or equal to) REPO_ROOT: every per-task sandbox
      directory create_sandbox() creates would then sit inside REPO_ROOT
      itself. os.walk() re-lists each directory's children at the moment
      it actually descends into them (not a snapshot taken once up
      front), so the still-growing sandbox directory - already created on
      disk by the time the walk reaches that part of the tree - would
      itself be walked into and copied into itself: an unbounded,
      self-referential recursive copy caused purely by this
      misconfiguration, independent of anything in the allowlist.
    - REPO_ROOT inside (or equal to) sandbox root: this does not trigger
      the recursive-copy failure mode above, but it means the sandbox
      root - the directory tree a Claude Code trial's tools are meant to
      be confined to - would now contain the entire real repository
      (including .git and any real secrets) as a sibling of every task's
      sandbox, defeating the "shares nothing - not even a hardlink - with
      the real repository" guarantee described at the top of this file.

    Never treats an unresolvable `root` as safe: if it cannot be resolved
    at all, this returns True (conflict) so callers fail closed, the same
    posture every other path-safety check in this module already takes.
    """
    resolved_root = _resolve_best_effort(root)
    if resolved_root is None:
        return True

    if resolved_root == REPO_ROOT:
        return True
    if REPO_ROOT in resolved_root.parents:
        return True
    if resolved_root in REPO_ROOT.parents:
        return True
    return False


def _validate_path_boundary(root: Path, target_path: Path) -> _ValidationResult:
    # Checked first, before any reparse-point/existence inspection below:
    # a DEV_TASK_SANDBOX_ROOT that is the real repository, sits inside it,
    # or contains it is a configuration error that must be rejected
    # regardless of what target_path even is - see
    # _sandbox_root_conflicts_with_repo_root()'s own docstring for why
    # each direction of this relationship is unsafe.
    if _sandbox_root_conflicts_with_repo_root(root):
        logger.error(
            "DEV_TASK_SANDBOX_ROOT conflicts with REPO_ROOT (same path, or nested in "
            "either direction, once resolved), refusing: root=%s repo_root=%s",
            root,
            REPO_ROOT,
        )
        return _validation_error(ERROR_SANDBOX_ROOT_CONFLICTS_WITH_REPO)

    if _path_entry_exists(root) and _is_reparse_point(root):
        logger.error("DEV_TASK_SANDBOX_ROOT is a symlink/junction/reparse point, refusing: %s", root)
        return _validation_error(ERROR_PATH_BOUNDARY_VIOLATION)

    if not _check_no_reparse_points_in_ancestry(root, target_path):
        logger.error(
            "A symlink/junction/reparse point was found between sandbox root and target: root=%s target=%s",
            root,
            target_path,
        )
        return _validation_error(ERROR_PATH_BOUNDARY_VIOLATION)

    if _path_entry_exists(target_path) and _is_reparse_point(target_path):
        logger.error("Sandbox target path itself is a symlink/junction/reparse point: target=%s", target_path)
        return _validation_error(ERROR_PATH_BOUNDARY_VIOLATION)

    if target_path.exists():
        try:
            resolved_target = target_path.resolve(strict=True)
            resolved_root = root.resolve(strict=True)
        except OSError:
            return _validation_error(ERROR_PATH_BOUNDARY_VIOLATION)
        if resolved_root != resolved_target and resolved_root not in resolved_target.parents:
            logger.error(
                "Sandbox target resolves outside root: target=%s resolved=%s root=%s",
                target_path,
                resolved_target,
                resolved_root,
            )
            return _validation_error(ERROR_PATH_BOUNDARY_VIOLATION)

    return _ValidationResult(success=True)


# --- per-task_id lock, same ref-counted pattern as dev_worktree.py --------

_task_locks_guard = threading.Lock()


class _RefCountedLock:
    __slots__ = ("lock", "ref_count")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ref_count = 0


_task_locks: dict[str, _RefCountedLock] = {}


@contextlib.contextmanager
def _task_lock(normalized_task_id: str):
    with _task_locks_guard:
        entry = _task_locks.get(normalized_task_id)
        if entry is None:
            entry = _RefCountedLock()
            _task_locks[normalized_task_id] = entry
        entry.ref_count += 1

    entry.lock.acquire()
    try:
        yield
    finally:
        entry.lock.release()
        with _task_locks_guard:
            entry.ref_count -= 1
            if entry.ref_count == 0 and _task_locks.get(normalized_task_id) is entry:
                del _task_locks[normalized_task_id]


# --- allowlist filtering ---------------------------------------------------


def _is_allowed_filename(filename: str) -> bool:
    if filename in _ALLOWED_EXACT_FILENAMES:
        return True
    if _ENV_FILENAME_PATTERN.match(filename):
        # Any other dotenv-style file (.env, .env.local, .env.production,
        # ...) is always denied - .env.example is the only exception, and
        # it was already accepted above.
        return False
    suffix = Path(filename).suffix.lower()
    if suffix in _DENIED_EXTENSIONS:
        return False
    return suffix in _ALLOWED_EXTENSIONS


def _hash_file(path: Path) -> str | None:
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
    except OSError:
        logger.exception("Failed to hash file: path=%s", path)
        return None
    return hasher.hexdigest()


def _copy_file_with_hash(source_path: Path, dest_path: Path) -> str | None:
    hasher = hashlib.sha256()
    try:
        with open(source_path, "rb") as src, open(dest_path, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
                dst.write(chunk)
    except OSError:
        logger.exception("Failed to copy file into sandbox: source=%s", source_path)
        return None

    # Defense in depth: a genuine independent copy must never share the
    # source's file identity (st_dev/st_ino) - a hardlink would. The
    # chunked read/write above always produces a fresh file on both
    # Windows (NTFS) and POSIX; if this ever trips, something unexpected
    # happened (e.g. filesystem-level dedup) and must be treated as a
    # failure, never silently accepted.
    try:
        src_stat = source_path.stat()
        dst_stat = dest_path.stat()
        if (src_stat.st_dev, src_stat.st_ino) == (dst_stat.st_dev, dst_stat.st_ino):
            logger.error("Sandbox copy shares file identity with source (unexpected): source=%s", source_path)
            return None
    except OSError:
        return None

    return hasher.hexdigest()


def _write_baseline(baseline_path: Path, manifest: dict[str, dict[str, Any]]) -> None:
    encoded = json.dumps(manifest, ensure_ascii=False)
    baseline_path.write_text(encoded, encoding="utf-8")


def _read_baseline(baseline_path: Path) -> dict[str, Any] | None:
    try:
        raw_text = baseline_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _best_effort_rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except OSError:
        logger.exception("Failed to clean up sandbox after a failed create: path=%s", path)


# --- create -----------------------------------------------------------


@dataclass(frozen=True)
class SandboxCreateResult:
    success: bool
    task_id: str | None
    sandbox_path: Path | None
    file_count: int
    total_bytes: int
    error_message: str | None = None
    error_code: str | None = None


def create_sandbox(task_id: Any) -> SandboxCreateResult:
    """Copy an allowlisted subset of REPO_ROOT into a fresh, independent
    directory under DEV_TASK_SANDBOX_ROOT/<task_id>, recording a SHA256
    baseline manifest (stored as a SIBLING file, never inside the sandbox
    itself) for later diffing.

    Never follows symlinks/junctions (source-side or destination-side),
    never hardlinks, default-denies anything not matching the allowlist,
    and aborts (deleting only the directory THIS call just created) if
    file-count/total-byte limits are exceeded or any step fails. Ends with
    a full re-verification via verify_sandbox_safe() before reporting
    success - a sandbox is never reported as ready for Claude to use
    without that.
    """
    normalized_id = _normalize_task_id(task_id)
    if normalized_id is None:
        return SandboxCreateResult(False, None, None, 0, 0, *_error_tuple(ERROR_INVALID_TASK_ID))

    mode = get_workspace_mode()
    if not is_workspace_mode_valid(mode):
        logger.error("Refusing to create sandbox: unsupported workspace mode=%r", mode)
        return SandboxCreateResult(
            False, normalized_id, None, 0, 0, *_error_tuple(ERROR_WORKSPACE_MODE_INVALID)
        )

    root = get_sandbox_root()
    if root is None:
        return SandboxCreateResult(
            False, normalized_id, None, 0, 0, *_error_tuple(ERROR_SANDBOX_ROOT_NOT_CONFIGURED)
        )

    with _task_lock(normalized_id):
        sandbox_path = _sandbox_path_for(root, normalized_id)
        baseline_path = _baseline_path_for(root, normalized_id)

        boundary_check = _validate_path_boundary(root, sandbox_path)
        if not boundary_check.success:
            return SandboxCreateResult(
                False, normalized_id, sandbox_path, 0, 0, boundary_check.error_message, boundary_check.error_code
            )

        if _path_entry_exists(sandbox_path):
            return SandboxCreateResult(
                False, normalized_id, sandbox_path, 0, 0, *_error_tuple(ERROR_PATH_ALREADY_EXISTS)
            )

        try:
            # exist_ok=False makes this the atomic ownership claim: only
            # one caller can ever win this mkdir for a given task_id, so
            # only the winner ever proceeds to (and is allowed to) clean up
            # on a later failure.
            sandbox_path.mkdir(parents=True, exist_ok=False)
        except OSError:
            logger.exception("Failed to create sandbox directory: task_id=%s", normalized_id)
            return SandboxCreateResult(
                False, normalized_id, sandbox_path, 0, 0, *_error_tuple(ERROR_COPY_FAILED)
            )

        max_files = get_max_files()
        max_total_bytes = get_max_total_bytes()
        manifest: dict[str, dict[str, Any]] = {}
        file_count = 0
        total_bytes = 0
        limit_exceeded = False

        try:
            for dirpath, dirnames, filenames in os.walk(REPO_ROOT, topdown=True, followlinks=False):
                current_dir = Path(dirpath)
                dirnames[:] = [
                    name
                    for name in dirnames
                    if name.lower() not in _DENIED_DIR_NAMES_LOWER
                    and not _is_reparse_point(current_dir / name)
                ]

                for filename in filenames:
                    source_path = current_dir / filename
                    if _is_reparse_point(source_path):
                        continue
                    if not _is_allowed_filename(filename):
                        continue

                    try:
                        file_size = source_path.stat().st_size
                    except OSError:
                        continue

                    file_count += 1
                    total_bytes += file_size
                    if file_count > max_files or total_bytes > max_total_bytes:
                        limit_exceeded = True
                        break

                    relative_path = source_path.relative_to(REPO_ROOT)
                    dest_path = sandbox_path / relative_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                    file_hash = _copy_file_with_hash(source_path, dest_path)
                    if file_hash is None:
                        raise OSError(f"failed to copy {source_path}")

                    manifest[str(relative_path).replace("\\", "/")] = {"sha256": file_hash, "size": file_size}

                if limit_exceeded:
                    break
        except OSError:
            logger.exception("Sandbox copy failed: task_id=%s", normalized_id)
            _best_effort_rmtree(sandbox_path)
            return SandboxCreateResult(
                False, normalized_id, sandbox_path, file_count, total_bytes, *_error_tuple(ERROR_COPY_FAILED)
            )

        if limit_exceeded:
            logger.warning(
                "Sandbox copy limit exceeded, aborting: task_id=%s file_count=%s total_bytes=%s",
                normalized_id,
                file_count,
                total_bytes,
            )
            _best_effort_rmtree(sandbox_path)
            return SandboxCreateResult(
                False, normalized_id, sandbox_path, file_count, total_bytes, *_error_tuple(ERROR_LIMIT_EXCEEDED)
            )

        try:
            _write_baseline(baseline_path, manifest)
        except OSError:
            logger.exception("Failed to write sandbox baseline manifest: task_id=%s", normalized_id)
            _best_effort_rmtree(sandbox_path)
            return SandboxCreateResult(
                False, normalized_id, sandbox_path, file_count, total_bytes, *_error_tuple(ERROR_COPY_FAILED)
            )

        verify_result = verify_sandbox_safe(normalized_id)
        if not verify_result.success:
            logger.error(
                "Newly created sandbox failed post-copy verification: task_id=%s error_code=%s",
                normalized_id,
                verify_result.error_code,
            )
            _best_effort_rmtree(sandbox_path)
            with contextlib.suppress(OSError):
                baseline_path.unlink(missing_ok=True)
            return SandboxCreateResult(
                False,
                normalized_id,
                sandbox_path,
                file_count,
                total_bytes,
                verify_result.error_message,
                verify_result.error_code,
            )

        logger.info(
            "Sandbox created: task_id=%s file_count=%s total_bytes=%s", normalized_id, file_count, total_bytes
        )
        return SandboxCreateResult(True, normalized_id, sandbox_path, file_count, total_bytes)


def _error_tuple(code: str) -> tuple[str, str]:
    return _ERROR_MESSAGES[code], code


# --- verify -----------------------------------------------------------


@dataclass(frozen=True)
class SandboxVerifyResult:
    success: bool
    task_id: str | None = None
    sandbox_path: Path | None = None
    error_message: str | None = None
    error_code: str | None = None


def verify_sandbox_safe(task_id: Any) -> SandboxVerifyResult:
    """Re-check, from scratch against live filesystem state (never trusting
    a cached create_sandbox() result), that a sandbox is safe to launch
    Claude Code against. This is called both at the end of create_sandbox()
    and, independently, by services.claude_headless.run() immediately
    before ever launching the CLI - so the check is always re-run right
    before the moment it actually matters, not just once at creation time.
    """
    normalized_id = _normalize_task_id(task_id)
    if normalized_id is None:
        return SandboxVerifyResult(False, None, None, *_error_tuple(ERROR_INVALID_TASK_ID))

    mode = get_workspace_mode()
    if not is_workspace_mode_valid(mode):
        return SandboxVerifyResult(False, normalized_id, None, *_error_tuple(ERROR_WORKSPACE_MODE_INVALID))

    root = get_sandbox_root()
    if root is None:
        return SandboxVerifyResult(False, normalized_id, None, *_error_tuple(ERROR_SANDBOX_ROOT_NOT_CONFIGURED))

    sandbox_path = _sandbox_path_for(root, normalized_id)

    boundary_check = _validate_path_boundary(root, sandbox_path)
    if not boundary_check.success:
        return SandboxVerifyResult(
            False, normalized_id, sandbox_path, boundary_check.error_message, boundary_check.error_code
        )

    if not sandbox_path.is_dir():
        return SandboxVerifyResult(False, normalized_id, sandbox_path, *_error_tuple(ERROR_SANDBOX_NOT_FOUND))

    max_files = get_max_files()
    max_total_bytes = get_max_total_bytes()

    try:
        resolved_root = sandbox_path.resolve(strict=True)
    except OSError:
        return SandboxVerifyResult(False, normalized_id, sandbox_path, *_error_tuple(ERROR_VERIFY_FAILED))

    file_count = 0
    total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(sandbox_path, topdown=True, followlinks=False):
        current_dir = Path(dirpath)
        # Never descend into a reparse point that may have been planted
        # inside the sandbox after creation - prune it and flag it below.
        safe_dirnames = []
        for name in dirnames:
            child = current_dir / name
            if _is_reparse_point(child):
                return SandboxVerifyResult(
                    False, normalized_id, sandbox_path, *_error_tuple(ERROR_FORBIDDEN_CONTENT_FOUND)
                )
            safe_dirnames.append(name)
        dirnames[:] = safe_dirnames

        relative_dir = current_dir.relative_to(sandbox_path)
        if str(relative_dir) != "." and any(
            part.lower() in _DENIED_DIR_NAMES_LOWER for part in relative_dir.parts
        ):
            logger.error(
                "Forbidden directory found inside sandbox: task_id=%s dir=%s", normalized_id, relative_dir
            )
            return SandboxVerifyResult(
                False, normalized_id, sandbox_path, *_error_tuple(ERROR_FORBIDDEN_CONTENT_FOUND)
            )

        for filename in filenames:
            file_path = current_dir / filename

            if _is_reparse_point(file_path):
                logger.error("Symlink/junction found inside sandbox: task_id=%s", normalized_id)
                return SandboxVerifyResult(
                    False, normalized_id, sandbox_path, *_error_tuple(ERROR_FORBIDDEN_CONTENT_FOUND)
                )

            if filename.lower() == ".git":
                return SandboxVerifyResult(
                    False, normalized_id, sandbox_path, *_error_tuple(ERROR_FORBIDDEN_CONTENT_FOUND)
                )

            if _ENV_FILENAME_PATTERN.match(filename) and filename not in _ALLOWED_EXACT_FILENAMES:
                logger.error("Forbidden .env-style file found inside sandbox: task_id=%s", normalized_id)
                return SandboxVerifyResult(
                    False, normalized_id, sandbox_path, *_error_tuple(ERROR_FORBIDDEN_CONTENT_FOUND)
                )

            suffix = Path(filename).suffix.lower()
            if suffix in _DENIED_EXTENSIONS:
                logger.error("Forbidden extension found inside sandbox: task_id=%s file=%s", normalized_id, filename)
                return SandboxVerifyResult(
                    False, normalized_id, sandbox_path, *_error_tuple(ERROR_FORBIDDEN_CONTENT_FOUND)
                )

            file_count += 1
            try:
                file_size = file_path.stat().st_size
            except OSError:
                return SandboxVerifyResult(False, normalized_id, sandbox_path, *_error_tuple(ERROR_VERIFY_FAILED))
            total_bytes += file_size

            if file_count > max_files or total_bytes > max_total_bytes:
                return SandboxVerifyResult(False, normalized_id, sandbox_path, *_error_tuple(ERROR_LIMIT_EXCEEDED))

            try:
                resolved_file = file_path.resolve(strict=True)
            except OSError:
                return SandboxVerifyResult(False, normalized_id, sandbox_path, *_error_tuple(ERROR_VERIFY_FAILED))
            if resolved_root != resolved_file and resolved_root not in resolved_file.parents:
                logger.error("File resolves outside sandbox root: task_id=%s file=%s", normalized_id, filename)
                return SandboxVerifyResult(
                    False, normalized_id, sandbox_path, *_error_tuple(ERROR_FORBIDDEN_CONTENT_FOUND)
                )

            # Never share file identity (hardlink) with the corresponding
            # source-repo file.
            relative_path = file_path.relative_to(sandbox_path)
            source_equivalent = REPO_ROOT / relative_path
            if source_equivalent.exists():
                try:
                    src_stat = source_equivalent.stat()
                    dst_stat = file_path.stat()
                    if (src_stat.st_dev, src_stat.st_ino) == (dst_stat.st_dev, dst_stat.st_ino):
                        logger.error(
                            "Sandbox file shares identity with source repo file: task_id=%s file=%s",
                            normalized_id,
                            relative_path,
                        )
                        return SandboxVerifyResult(
                            False, normalized_id, sandbox_path, *_error_tuple(ERROR_FORBIDDEN_CONTENT_FOUND)
                        )
                except OSError:
                    pass

    return SandboxVerifyResult(True, normalized_id, sandbox_path)


# --- diff (hash-based, never git) --------------------------------------


_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Unicode bidi override/embedding/isolate control characters (LRM, RLM,
# LRE/RLE/PDF/LRO/RLO, LRI/RLI/FSI/PDI). Written as explicit \uXXXX escapes
# rather than literal glyphs, since these are by definition invisible
# characters and would otherwise be an unreviewable landmine sitting in
# this source file.
_BIDI_CONTROL_PATTERN = re.compile("[\u200e\u200f\u202a-\u202e\u2066-\u2069]")
_DISCORD_MENTION_LIKE_PATTERN = re.compile(r"[@<>`]")

_MAX_SAMPLE_NAMES = 20
_MAX_DISPLAY_NAME_LENGTH = 200


def sanitize_filename_for_display(name: str, *, max_length: int = _MAX_DISPLAY_NAME_LENGTH) -> str:
    """Strip everything that could make a filename dangerous or misleading
    to render verbatim in a Discord message: ANSI escape sequences,
    control characters, Unicode bidi override/embedding/isolate
    characters, '@'/'<'/'>'/backtick (mention/markup/code-fence
    injection), and newlines/carriage returns (line injection).

    This is a defense on the TEXT itself; callers actually sending to
    Discord must still pass allowed_mentions=discord.AllowedMentions.none()
    - stripping '@' here is extra insurance, not a substitute for that.
    """
    if not isinstance(name, str):
        return ""
    text = _ANSI_ESCAPE_PATTERN.sub("", name)
    text = _CONTROL_CHAR_PATTERN.sub("", text)
    text = _BIDI_CONTROL_PATTERN.sub("", text)
    text = _DISCORD_MENTION_LIKE_PATTERN.sub("", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text[:max_length]


@dataclass(frozen=True)
class SandboxDiffResult:
    success: bool
    added_count: int = 0
    modified_count: int = 0
    removed_count: int = 0
    changed_file_count: int = 0
    sample_changed_names: tuple[str, ...] = ()
    manual_review_required: bool = True
    error_message: str | None = None
    error_code: str | None = None


def diff_sandbox_against_baseline(task_id: Any) -> SandboxDiffResult:
    """Read-only comparison of the sandbox's CURRENT file hashes against
    the SHA256 baseline manifest recorded at create_sandbox() time - never
    git, never a subprocess; this is the simpler, git-free alternative to
    running a git repo inside the sandbox, exactly as permitted.
    """
    normalized_id = _normalize_task_id(task_id)
    if normalized_id is None:
        return SandboxDiffResult(False, error_message=_ERROR_MESSAGES[ERROR_INVALID_TASK_ID], error_code=ERROR_INVALID_TASK_ID)

    root = get_sandbox_root()
    if root is None:
        return SandboxDiffResult(
            False,
            error_message=_ERROR_MESSAGES[ERROR_SANDBOX_ROOT_NOT_CONFIGURED],
            error_code=ERROR_SANDBOX_ROOT_NOT_CONFIGURED,
        )

    sandbox_path = _sandbox_path_for(root, normalized_id)
    baseline_path = _baseline_path_for(root, normalized_id)

    baseline = _read_baseline(baseline_path)
    if baseline is None:
        return SandboxDiffResult(
            False, error_message="baselineが読み込めませんでした。", error_code=ERROR_VERIFY_FAILED
        )

    if not sandbox_path.is_dir():
        return SandboxDiffResult(
            False, error_message=_ERROR_MESSAGES[ERROR_SANDBOX_NOT_FOUND], error_code=ERROR_SANDBOX_NOT_FOUND
        )

    current: dict[str, str] = {}
    try:
        for dirpath, dirnames, filenames in os.walk(sandbox_path, topdown=True, followlinks=False):
            current_dir = Path(dirpath)
            dirnames[:] = [name for name in dirnames if not _is_reparse_point(current_dir / name)]
            for filename in filenames:
                file_path = current_dir / filename
                if _is_reparse_point(file_path):
                    continue
                relative_path = str(file_path.relative_to(sandbox_path)).replace("\\", "/")
                file_hash = _hash_file(file_path)
                if file_hash is None:
                    return SandboxDiffResult(
                        False, error_message="変更確認中にエラーが発生しました。", error_code=ERROR_VERIFY_FAILED
                    )
                current[relative_path] = file_hash
    except OSError:
        logger.exception("Failed to walk sandbox for diff: task_id=%s", normalized_id)
        return SandboxDiffResult(
            False, error_message="変更確認中にエラーが発生しました。", error_code=ERROR_VERIFY_FAILED
        )

    baseline_paths = set(baseline.keys())
    current_paths = set(current.keys())

    added = current_paths - baseline_paths
    removed = baseline_paths - current_paths
    modified = {
        path
        for path in (baseline_paths & current_paths)
        if not isinstance(baseline.get(path), dict) or baseline[path].get("sha256") != current[path]
    }

    changed = added | removed | modified
    sample_names = tuple(sorted(sanitize_filename_for_display(name) for name in changed)[:_MAX_SAMPLE_NAMES])

    return SandboxDiffResult(
        success=True,
        added_count=len(added),
        modified_count=len(modified),
        removed_count=len(removed),
        changed_file_count=len(changed),
        sample_changed_names=sample_names,
        manual_review_required=False,
    )


# --- execution-in-progress discard guard --------------------------------

_execution_state_lock = threading.Lock()
_execution_state: dict[str, dict[str, bool]] = {}


def mark_execution_started(task_id: Any) -> None:
    normalized_id = _normalize_task_id(task_id)
    if normalized_id is None:
        return
    with _execution_state_lock:
        _execution_state[normalized_id] = {"in_progress": True, "termination_confirmed": True}


def mark_execution_finished(task_id: Any, *, termination_confirmed: bool) -> None:
    normalized_id = _normalize_task_id(task_id)
    if normalized_id is None:
        return
    with _execution_state_lock:
        _execution_state[normalized_id] = {
            "in_progress": False,
            "termination_confirmed": termination_confirmed,
        }


def _clear_execution_state(normalized_task_id: str) -> None:
    with _execution_state_lock:
        _execution_state.pop(normalized_task_id, None)


def _discard_guard_reason(normalized_task_id: str) -> str | None:
    with _execution_state_lock:
        state = _execution_state.get(normalized_task_id)
    if state is None:
        return None
    if state["in_progress"]:
        return ERROR_EXECUTION_IN_PROGRESS
    if not state["termination_confirmed"]:
        return ERROR_TERMINATION_UNCONFIRMED
    return None


# --- discard ------------------------------------------------------------


@dataclass(frozen=True)
class SandboxDiscardResult:
    success: bool
    task_id: str | None = None
    error_message: str | None = None
    error_code: str | None = None


def discard_sandbox(task_id: Any) -> SandboxDiscardResult:
    """Remove a sandbox and its baseline manifest - refusing outright,
    touching nothing, while an execution is still marked in_progress for
    this task_id, or while the most recent execution's process-tree
    termination was never confirmed (see mark_execution_started()/
    mark_execution_finished()). Idempotent: discarding an already-absent
    sandbox is reported as success.
    """
    normalized_id = _normalize_task_id(task_id)
    if normalized_id is None:
        return SandboxDiscardResult(False, None, *_error_tuple(ERROR_INVALID_TASK_ID))

    with _task_lock(normalized_id):
        guard_code = _discard_guard_reason(normalized_id)
        if guard_code is not None:
            logger.warning("Refusing to discard sandbox: task_id=%s reason=%s", normalized_id, guard_code)
            return SandboxDiscardResult(False, normalized_id, *_error_tuple(guard_code))

        root = get_sandbox_root()
        if root is None:
            return SandboxDiscardResult(False, normalized_id, *_error_tuple(ERROR_SANDBOX_ROOT_NOT_CONFIGURED))

        sandbox_path = _sandbox_path_for(root, normalized_id)
        baseline_path = _baseline_path_for(root, normalized_id)

        boundary_check = _validate_path_boundary(root, sandbox_path)
        if not boundary_check.success:
            return SandboxDiscardResult(
                False, normalized_id, boundary_check.error_message, boundary_check.error_code
            )

        if not _path_entry_exists(sandbox_path):
            _clear_execution_state(normalized_id)
            return SandboxDiscardResult(True, normalized_id)

        try:
            shutil.rmtree(sandbox_path)
        except OSError:
            logger.exception("Failed to discard sandbox: task_id=%s", normalized_id)
            return SandboxDiscardResult(False, normalized_id, *_error_tuple(ERROR_DISCARD_FAILED))

        with contextlib.suppress(OSError):
            baseline_path.unlink(missing_ok=True)

        if _path_entry_exists(sandbox_path):
            logger.error("Sandbox residue remains after discard: task_id=%s", normalized_id)
            return SandboxDiscardResult(False, normalized_id, *_error_tuple(ERROR_DISCARD_FAILED))

        _clear_execution_state(normalized_id)
        logger.info("Sandbox discarded: task_id=%s", normalized_id)
        return SandboxDiscardResult(True, normalized_id)
