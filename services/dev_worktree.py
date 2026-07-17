from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.dev_git_runner import (
    ERROR_CWD_NOT_FOUND,
    ERROR_GIT_COMMAND_FAILED,
    ERROR_GIT_LAUNCH_FAILED,
    ERROR_GIT_NOT_FOUND,
    ERROR_GIT_TIMEOUT,
    check_ref_exists,
    map_git_error,
    run_git,
)
from services.dev_worktree_parser import WorktreeRecord, parse_worktree_list_porcelain_z


logger = logging.getLogger("secretary-bot.dev_worktree")

# Path(__file__).resolve().parents[1] is services/../ == the repo root this
# file lives under, independent of the process's current working directory.
#
# REPO_ROOT is intentionally read fresh from this module attribute (never
# cached into a frozen constant derived at import time) everywhere below,
# because tests reassign it directly (`wt.REPO_ROOT = <temp repo>`) to
# exercise this module against a throwaway git repository.
REPO_ROOT = Path(__file__).resolve().parents[1]

WORK_DIR_NAME = "work"
BRANCH_PREFIX = "dev-task/"

_GIT_TIMEOUT_SECONDS = 30.0
_WORKTREE_ADD_TIMEOUT_SECONDS = 60.0

# Git already rejects all of these via `check-ref-format`, but rejecting them
# up front avoids spawning a subprocess for obviously-bad input and keeps the
# rule set self-documenting.
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x1f\x7f]")

# Stable, machine-checkable reasons so callers can branch on why an operation
# failed instead of parsing the Discord-facing Japanese message.
ERROR_INVALID_TASK_ID = "invalid_task_id"
ERROR_INVALID_BASE_BRANCH_TYPE = "invalid_base_branch_type"
ERROR_INVALID_BASE_BRANCH_FORMAT = "invalid_base_branch_format"
ERROR_BASE_BRANCH_NOT_FOUND = "base_branch_not_found"
ERROR_PATH_ALREADY_EXISTS = "path_already_exists"
ERROR_BRANCH_ALREADY_EXISTS = "branch_already_exists"
ERROR_ALREADY_REGISTERED = "already_registered"
ERROR_PATH_BOUNDARY_VIOLATION = "path_boundary_violation"
ERROR_WORKTREE_ADD_FAILED = "worktree_add_failed"
ERROR_WORKTREE_NOT_FOUND = "worktree_not_found"
ERROR_WORKTREE_REMOVE_FAILED = "worktree_remove_failed"
ERROR_WORKTREE_REMOVE_INCOMPLETE = "worktree_remove_incomplete"
ERROR_FOREIGN_WORKTREE = "foreign_worktree"
ERROR_BRANCH_REMOVE_FAILED = "branch_remove_failed"
ERROR_INFO_FAILED = "info_failed"
ERROR_PRUNE_FAILED = "prune_failed"
ERROR_GIT_OUTPUT_TRUNCATED = "git_output_truncated"

_ERROR_MESSAGES: dict[str, str] = {
    ERROR_INVALID_TASK_ID: "タスクIDの形式が不正です。",
    ERROR_INVALID_BASE_BRANCH_TYPE: "ベースブランチの指定形式が不正です。",
    ERROR_INVALID_BASE_BRANCH_FORMAT: "ベースブランチ名が不正です。",
    ERROR_BASE_BRANCH_NOT_FOUND: "指定されたベースブランチが見つかりません。",
    ERROR_PATH_ALREADY_EXISTS: "作業ディレクトリが既に存在します。",
    ERROR_BRANCH_ALREADY_EXISTS: "同名のブランチが既に存在します。",
    ERROR_ALREADY_REGISTERED: "既にworktreeとして登録されています。",
    ERROR_PATH_BOUNDARY_VIOLATION: "作業ディレクトリのパスが安全な範囲外を指しているため中止しました。",
    ERROR_WORKTREE_ADD_FAILED: "worktreeの作成に失敗しました。",
    ERROR_WORKTREE_NOT_FOUND: "worktreeが見つかりません。",
    ERROR_WORKTREE_REMOVE_FAILED: "worktreeの削除に失敗しました。",
    ERROR_WORKTREE_REMOVE_INCOMPLETE: "worktreeの削除が完了したように見えましたが、実際には残っていました。",
    ERROR_FOREIGN_WORKTREE: "このパスには別のブランチのworktreeが登録されているため中止しました。",
    ERROR_BRANCH_REMOVE_FAILED: "ブランチの削除に失敗しました。",
    ERROR_INFO_FAILED: "worktree情報の取得に失敗しました。",
    ERROR_PRUNE_FAILED: "worktree情報の整理に失敗しました。",
    ERROR_GIT_OUTPUT_TRUNCATED: "Gitの出力が長すぎて安全に読み取れませんでした。",
    ERROR_GIT_NOT_FOUND: "Gitの実行ファイルが見つかりません。",
    ERROR_GIT_TIMEOUT: "Gitコマンドがタイムアウトしました。",
    ERROR_GIT_LAUNCH_FAILED: "Gitコマンドの起動に失敗しました。",
    ERROR_GIT_COMMAND_FAILED: "Gitコマンドが失敗しました。",
    ERROR_CWD_NOT_FOUND: "作業ディレクトリが見つかりません。",
}

# get_worktree_info().consistency values.
CONSISTENCY_CONSISTENT = "consistent"
CONSISTENCY_NOT_FOUND = "not_found"
CONSISTENCY_MISSING_PATH = "missing_path"
CONSISTENCY_MISSING_REGISTRATION = "missing_registration"
CONSISTENCY_ORPHAN_BRANCH = "orphan_branch"
CONSISTENCY_PARTIAL_STATE = "partial_state"
CONSISTENCY_FOREIGN_WORKTREE = "foreign_worktree"


def _map_git_error(error_code: str | None, *, default: str) -> str:
    return map_git_error(error_code, default=default)


@dataclass(frozen=True)
class CreateCleanupResult:
    """Outcome of the post-failure check after a failed `git worktree add`.

    Phase 2 performs ZERO automatic deletion here. Earlier designs tried to
    prove "this create attempt owns what's now at this path" by checking
    whether the registered branch ref matched this task_id's own derived
    branch name - but that check is worthless against the one race that
    actually matters: two concurrent create_worktree() calls for the SAME
    task_id derive the identical path AND the identical branch name, so the
    loser's post-check would see a registration whose branch matches its
    own expected branch not because the loser made it, but because the
    winner did. "The name matches what I would have created" is not proof
    of authorship when task_id can be shared between concurrent callers,
    and Phase 2 has no call-specific nonce or ownership marker that could
    actually prove it. So this function never calls `git worktree remove`,
    `git branch -D`, `git worktree prune`, or shutil.rmtree - it only
    re-queries state and reports it.

    Invariant: manual_cleanup_required is False if and only if this
    function positively confirmed - by a fresh post-check, not by assuming
    anything from before the failed attempt - that no residue exists at
    all (path_entry_exists_after, registration_exists_after, and
    branch_exists_after are all False). Any residue, or any failure to even
    determine the state, sets manual_cleanup_required True. There is no
    third outcome: cleanup is never reported as "done" while a post-check
    still shows something remaining.
    """

    # None on any of these three means the post-check itself failed and
    # this specific piece of state could not be determined (see
    # cleanup_error_code) - never guessed as False.
    path_entry_exists_after: bool | None
    registration_exists_after: bool | None
    branch_exists_after: bool | None
    # Always False in Phase 2: no nonce/ownership-marker mechanism exists
    # yet to prove a specific call created a specific piece of state, so
    # ownership is never claimed as verified.
    ownership_verified: bool
    # Always False in Phase 2: no automatic deletion is ever performed.
    automatic_cleanup_performed: bool
    manual_cleanup_required: bool
    cleanup_error_code: str | None


@dataclass(frozen=True)
class WorktreeResult:
    success: bool
    task_id: str | None
    worktree_path: Path | None
    branch_name: str | None
    error_message: str | None = None
    error_code: str | None = None
    cleanup: CreateCleanupResult | None = None


@dataclass(frozen=True)
class WorktreeRemoveResult:
    success: bool
    task_id: str | None
    worktree_path: Path | None
    branch_name: str | None
    # worktree_removed/branch_removed are always set from a fresh, measured
    # get_worktree_info() re-check after `git worktree remove`/`git branch
    # -D` report success - never assumed from those commands' exit codes
    # alone. success=True only when both are confirmed True by that
    # measurement.
    worktree_removed: bool
    branch_removed: bool
    # True whenever git reported (or appeared to report) success but a
    # fresh measurement afterward still found residue, or the measurement
    # itself could not be completed - i.e. exactly the situations where an
    # operator may need to look, distinct from a plain, well-understood
    # command failure (which already carries its own error_code/message).
    manual_cleanup_required: bool = False
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class WorktreeInfo:
    task_id: str
    worktree_path: Path
    branch_name: str
    path_entry_exists: bool
    branch_exists: bool
    registered_in_git_worktree: bool
    # The exact "branch" field (full ref, e.g. "refs/heads/foo") reported by
    # `git worktree list` for the entry registered at this path, or None if
    # nothing is registered there, or the registration is detached.
    registered_branch_ref: str | None
    registered_detached: bool
    # True iff something is registered at this path but it is NOT this
    # task's own branch (a different branch, or a detached HEAD) - i.e. a
    # worktree belonging to someone/something else occupies our path.
    branch_mismatch: bool
    consistency: str


@dataclass(frozen=True)
class WorktreeInfoResult:
    success: bool
    info: WorktreeInfo | None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class WorktreeValidationResult:
    success: bool
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class PruneResult:
    success: bool
    error_message: str | None = None
    error_code: str | None = None


def _validation_error(code: str) -> WorktreeValidationResult:
    return WorktreeValidationResult(success=False, error_message=_ERROR_MESSAGES[code], error_code=code)


def _normalize_task_id(task_id: Any) -> str | None:
    """Parse `task_id` as a strict UUID4 string, identical in spirit to
    Phase 1's dev_task_store._normalize_task_id(): a syntactically valid
    UUID of any other version/variant is rejected, not just malformed
    strings. Returns the canonical lowercase, hyphenated form, or None.
    """
    try:
        candidate = uuid.UUID(str(task_id))
    except (ValueError, AttributeError, TypeError):
        return None
    if candidate.version != 4 or candidate.variant != uuid.RFC_4122:
        return None
    return str(candidate)


def _work_root() -> Path:
    return REPO_ROOT / WORK_DIR_NAME


def _worktree_path_for(normalized_task_id: str) -> Path:
    return _work_root() / normalized_task_id


def _branch_name_for(normalized_task_id: str) -> str:
    # The full UUID4 (36 characters, not a truncated prefix) is used so two
    # different task ids can never derive the same branch name: an 8-char
    # prefix has a real, if small, collision probability, while the full
    # UUID's collision probability is negligible, and every caller of this
    # function - create_worktree(), remove_worktree(), get_worktree_info(),
    # the failed-create cleanup path - shares this single derivation, so
    # they can never disagree about what "the" branch for a task id is.
    return f"{BRANCH_PREFIX}{normalized_task_id}"


# Per-task_id locks serializing create_worktree()/remove_worktree() for
# the SAME task_id within this process, so a concurrent create+create,
# create+remove, or remove+remove for one task_id can never interleave.
# This only protects against other threads in this same process; it does
# NOT provide cross-process mutual exclusion (e.g. two separate bot
# processes, or an external `git worktree` invocation) - that is out of
# scope for Phase 2.
#
# Entries are reference-counted rather than left to accumulate forever:
# _task_lock() increments the count before acquiring, and decrements it
# after releasing, deleting the dict entry only when the count reaches
# zero - so a long-running process handling many distinct task_ids over
# time does not grow this dict without bound. All dict mutations (lookup,
# insert, refcount changes, delete) happen only while holding
# _task_locks_guard, which is what makes the increment-before-acquire /
# decrement-after-release protocol race-free: see _task_lock()'s docstring
# for why a lock can never be deleted while another caller is still using
# (or about to use) it.
_task_locks_guard = threading.Lock()


class _RefCountedLock:
    __slots__ = ("lock", "ref_count")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ref_count = 0


_task_locks: dict[str, _RefCountedLock] = {}


@contextlib.contextmanager
def _task_lock(normalized_task_id: str):
    """Serialize execution across every caller using this exact task_id,
    without leaking a permanent dict entry per task_id ever seen.

    Safety argument for why a lock is never deleted while still in use:
    ref_count is only ever incremented/decremented while holding
    _task_locks_guard, and the *decrement* (by the caller currently holding
    `entry.lock`) always happens strictly after that caller has already
    called entry.lock.release() - so by the time ref_count can reach zero
    under the guard, the thread that just finished has fully relinquished
    the lock. Any other thread that looked up (and thus incremented) the
    same entry before that decrement is guaranteed to see a non-zero count
    and prevent deletion; any thread that looks it up afterward finds no
    entry and creates a fresh one, which is correct because the previous
    holder is by then already done. Both the increment and the
    decrement-with-conditional-delete are single atomic critical sections
    under the same guard, so the two can never interleave with each other
    for the same task_id.
    """
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


def _normalized_path_key(path: Path) -> str:
    # os.path.normcase is a no-op on POSIX and lowercases + normalizes
    # separators on Windows, so paths that differ only in drive-letter case
    # or slash direction (as git's porcelain output does: forward slashes
    # even on Windows) still compare equal.
    return os.path.normcase(str(path.resolve()))


def _path_entry_exists(path: Path) -> bool:
    """True iff a directory entry exists at `path` at all - including a
    broken symlink or a junction whose target no longer exists.

    Path.exists() (and plain os.path.exists()) follows symlinks/junctions
    and reports False for a BROKEN one, which hides the fact that
    something is still sitting there on disk: a broken junction at
    work/<uuid> would make Path.exists() say "nothing here", incorrectly
    signaling a path is free to claim. This uses os.path.lexists() (the
    lstat-based check, which does not follow the final link) instead, so a
    broken link/junction is still reported as present.
    """
    try:
        return os.path.lexists(path)
    except OSError:
        return True  # Cannot safely stat it; treat as present/suspect.


def _is_reparse_point(path: Path) -> bool:
    """True if `path` is a symlink, Windows junction/mount point, or any
    other filesystem reparse point.

    Path.is_symlink() alone is not sufficient on Windows: NTFS junctions
    and mount-point reparse points use a different reparse tag from
    symlinks, so Path.is_symlink() reports False for them even though -
    exactly like a symlink - the path can silently resolve somewhere
    outside the directory tree it lexically appears to be under. This
    additionally checks the FILE_ATTRIBUTE_REPARSE_POINT flag via
    GetFileAttributesW on Windows to catch that case too.
    """
    try:
        if path.is_symlink():
            return True
    except OSError:
        return True  # Cannot even stat it safely; treat as suspect.

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


def _check_no_reparse_points_in_ancestry(root: Path, target: Path) -> bool:
    """True iff `root` itself and every existing intermediate directory
    between `root` and `target` (exclusive of `target` itself) are free of
    symlinks/junctions/reparse points. `target` need not exist yet.

    Existence here is checked via _path_entry_exists() (lexists-based), not
    Path.exists(), so a broken symlink/junction sitting in the ancestry is
    still caught instead of being silently skipped as "not there".
    """
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


def _validate_path_boundary(target_path: Path) -> WorktreeValidationResult:
    """Confirm `target_path` is - and, short of an active TOCTOU attack,
    will remain - genuinely inside REPO_ROOT/work on the real filesystem,
    not merely by lexical path prefix.

    Checks, in order: WORK_ROOT itself is not a symlink/junction/reparse
    point; every existing intermediate directory between WORK_ROOT and
    target_path is free of symlinks/junctions/reparse points; target_path
    ITSELF is not a symlink/junction/reparse point; and, if target_path
    resolves to something real, its fully resolved real path is still
    inside WORK_ROOT's fully resolved real path.

    The target-path-itself check is what actually matters for the common
    attack/failure shape: a junction/symlink planted exactly at
    work/<uuid>. It is checked via _path_entry_exists() (lexists-based),
    not Path.exists(), specifically because a BROKEN junction/symlink (one
    whose target no longer resolves) makes Path.exists() report False even
    though the reparse point itself is still sitting on disk - that must
    never be read as "the path is free to use" or "there is nothing left
    to worry about here".

    This is used as a pre-condition gate before create_worktree() ever
    calls `git worktree add`, and remove_worktree() calls it twice (at the
    start of removal, and again immediately before `git worktree remove`)
    since this module still never deletes a path via a direct filesystem
    call (shutil.rmtree or equivalent) - see CreateCleanupResult and
    _cleanup_failed_create().
    """
    work_root = _work_root()

    if _path_entry_exists(work_root) and _is_reparse_point(work_root):
        logger.error("WORK_ROOT is a symlink/junction/reparse point, refusing to proceed: %s", work_root)
        return _validation_error(ERROR_PATH_BOUNDARY_VIOLATION)

    if not _check_no_reparse_points_in_ancestry(work_root, target_path):
        logger.error(
            "A symlink/junction/reparse point was found between WORK_ROOT and the target path: root=%s target=%s",
            work_root,
            target_path,
        )
        return _validation_error(ERROR_PATH_BOUNDARY_VIOLATION)

    if _path_entry_exists(target_path) and _is_reparse_point(target_path):
        logger.error(
            "Target path itself is a symlink/junction/reparse point (possibly broken): target=%s",
            target_path,
        )
        return _validation_error(ERROR_PATH_BOUNDARY_VIOLATION)

    if target_path.exists():
        try:
            resolved_target = target_path.resolve(strict=True)
            resolved_root = work_root.resolve(strict=True)
        except OSError:
            return _validation_error(ERROR_PATH_BOUNDARY_VIOLATION)
        if resolved_root != resolved_target and resolved_root not in resolved_target.parents:
            logger.error(
                "Target path resolves outside WORK_ROOT: target=%s resolved=%s root=%s",
                target_path,
                resolved_target,
                resolved_root,
            )
            return _validation_error(ERROR_PATH_BOUNDARY_VIOLATION)

    return WorktreeValidationResult(success=True, error_message=None, error_code=None)


def _has_disallowed_base_branch_chars(value: str) -> bool:
    if _CONTROL_CHAR_PATTERN.search(value):
        return True
    if ".." in value:
        return True
    if "@{" in value:
        return True
    if "\\" in value:
        return True
    if value.startswith("-"):
        return True
    return False


def validate_base_branch(base_branch: Any) -> WorktreeValidationResult:
    """Strictly validate a (potentially user-supplied) base branch name.

    Rejects, in order: non-str/bool input, empty/whitespace-only strings,
    control characters (including newlines), a leading hyphen, "..", "@{",
    backslashes - all before ever spawning a subprocess - then defers to
    `git check-ref-format --branch` for the full git ref grammar, and
    finally requires the name to resolve to an existing local branch. A
    base branch that passes every check here is safe to interpolate into a
    fixed git argument list elsewhere in this module.
    """
    if isinstance(base_branch, bool) or not isinstance(base_branch, str):
        return _validation_error(ERROR_INVALID_BASE_BRANCH_TYPE)

    if not base_branch.strip():
        return _validation_error(ERROR_INVALID_BASE_BRANCH_FORMAT)

    if _has_disallowed_base_branch_chars(base_branch):
        return _validation_error(ERROR_INVALID_BASE_BRANCH_FORMAT)

    format_check = run_git(
        ["check-ref-format", "--branch", base_branch], cwd=REPO_ROOT, timeout=_GIT_TIMEOUT_SECONDS
    )
    if not format_check.success:
        if format_check.error_code == ERROR_GIT_COMMAND_FAILED:
            return _validation_error(ERROR_INVALID_BASE_BRANCH_FORMAT)
        return _validation_error(format_check.error_code)

    exists_check = check_ref_exists(f"refs/heads/{base_branch}", cwd=REPO_ROOT, timeout=_GIT_TIMEOUT_SECONDS)
    if exists_check.exists is None:
        return _validation_error(exists_check.error_code)
    if not exists_check.exists:
        return _validation_error(ERROR_BASE_BRANCH_NOT_FOUND)

    return WorktreeValidationResult(success=True, error_message=None, error_code=None)


def _list_worktrees_porcelain() -> tuple[list[WorktreeRecord] | None, str | None]:
    """List every registered worktree, or fail safely - never returning a
    list assembled from incomplete data.

    stdout_truncated is checked BEFORE ever attempting to parse: a
    truncated stream can drop trailing records entirely (including,
    possibly, the very record a caller is looking for), and a truncation
    that happens to land on a NUL boundary can look structurally complete
    to the parser even though it isn't - so stdout_truncated (set by
    dev_git_runner from the actual byte count git produced, not guessed
    from output shape) is the only signal trusted here. This means every
    caller of this function (get_worktree_info(), and transitively
    create_worktree()/remove_worktree()'s pre-checks) fails closed instead
    of ever concluding "not registered" from a truncated listing.
    """
    result = run_git(["worktree", "list", "--porcelain", "-z"], cwd=REPO_ROOT, timeout=_GIT_TIMEOUT_SECONDS)
    if not result.success:
        return None, _map_git_error(result.error_code, default=ERROR_INFO_FAILED)

    if result.stdout_truncated:
        logger.error(
            "git worktree list --porcelain -z output was truncated (exceeded the capture limit); "
            "refusing to parse a partial worktree listing"
        )
        return None, ERROR_GIT_OUTPUT_TRUNCATED

    entries = parse_worktree_list_porcelain_z(result.stdout)
    if entries is None:
        logger.error("Failed to parse `git worktree list --porcelain -z` output")
        return None, ERROR_INFO_FAILED
    return entries, None


def _find_registered_record(entries: list[WorktreeRecord], target_path: Path) -> WorktreeRecord | None:
    target_key = _normalized_path_key(target_path)
    for entry in entries:
        if _normalized_path_key(entry.path) == target_key:
            return entry
    return None


def _classify_consistency(
    *, path_entry_exists: bool, registered: bool, branch_exists: bool, branch_mismatch: bool = False
) -> str:
    if registered and branch_mismatch:
        return CONSISTENCY_FOREIGN_WORKTREE
    if not path_entry_exists and not registered and not branch_exists:
        return CONSISTENCY_NOT_FOUND
    if path_entry_exists and registered and branch_exists:
        return CONSISTENCY_CONSISTENT
    if registered and not path_entry_exists:
        return CONSISTENCY_MISSING_PATH
    if not registered and path_entry_exists and branch_exists:
        return CONSISTENCY_MISSING_REGISTRATION
    if not registered and not path_entry_exists and branch_exists:
        return CONSISTENCY_ORPHAN_BRANCH
    return CONSISTENCY_PARTIAL_STATE


def get_worktree_info(task_id: Any) -> WorktreeInfoResult:
    """Return a structured snapshot of a dev task's worktree/branch state,
    cross-checking the filesystem, `git worktree list --porcelain -z`, and
    `git show-ref` against each other so callers (including this module's
    own create_worktree()/remove_worktree()) can detect drift between them
    instead of trusting any single source.

    Crucially, "registered at this path" is not treated as synonymous with
    "is this task's own worktree": registered_branch_ref/registered_detached
    record exactly what git reports is registered there, and
    branch_mismatch is True whenever that registration is NOT this task's
    expected branch (a different branch, or a detached HEAD) - letting
    callers refuse to touch a foreign worktree that merely happens to share
    this task's derived path.
    """
    normalized_id = _normalize_task_id(task_id)
    if normalized_id is None:
        return WorktreeInfoResult(
            success=False,
            info=None,
            error_message=_ERROR_MESSAGES[ERROR_INVALID_TASK_ID],
            error_code=ERROR_INVALID_TASK_ID,
        )

    worktree_path = _worktree_path_for(normalized_id)
    branch_name = _branch_name_for(normalized_id)
    expected_ref = f"refs/heads/{branch_name}"

    entries, list_error_code = _list_worktrees_porcelain()
    if entries is None:
        return WorktreeInfoResult(
            success=False,
            info=None,
            error_message=_ERROR_MESSAGES[list_error_code],
            error_code=list_error_code,
        )

    # lexists-based, not Path.exists(): a broken symlink/junction sitting at
    # this path must still be reported as "something is here", not silently
    # treated as absent (see _path_entry_exists()'s docstring).
    path_entry_exists = _path_entry_exists(worktree_path)

    record = _find_registered_record(entries, worktree_path)
    registered = record is not None
    registered_branch_ref = record.branch if record is not None else None
    registered_detached = record.detached if record is not None else False

    branch_check = check_ref_exists(expected_ref, cwd=REPO_ROOT, timeout=_GIT_TIMEOUT_SECONDS)
    if branch_check.exists is None:
        mapped_code = _map_git_error(branch_check.error_code, default=ERROR_INFO_FAILED)
        return WorktreeInfoResult(
            success=False,
            info=None,
            error_message=_ERROR_MESSAGES[mapped_code],
            error_code=mapped_code,
        )
    branch_exists = branch_check.exists

    branch_mismatch = registered and (registered_detached or registered_branch_ref != expected_ref)

    consistency = _classify_consistency(
        path_entry_exists=path_entry_exists,
        registered=registered,
        branch_exists=branch_exists,
        branch_mismatch=branch_mismatch,
    )

    info = WorktreeInfo(
        task_id=normalized_id,
        worktree_path=worktree_path,
        branch_name=branch_name,
        path_entry_exists=path_entry_exists,
        branch_exists=branch_exists,
        registered_in_git_worktree=registered,
        registered_branch_ref=registered_branch_ref,
        registered_detached=registered_detached,
        branch_mismatch=branch_mismatch,
        consistency=consistency,
    )
    return WorktreeInfoResult(success=True, info=info, error_message=None, error_code=None)


def _cleanup_failed_create(normalized_id: str, worktree_path: Path, branch_name: str) -> CreateCleanupResult:
    """Post-failure state check after a failed `git worktree add` - never
    deletes anything (see CreateCleanupResult's docstring for why).

    Called only from create_worktree() after `git worktree add` itself has
    already failed. Simply re-queries the current state via
    get_worktree_info() and reports it: if the post-check itself can't be
    completed, or if it finds any residue at all, manual_cleanup_required
    is True. This never assumes "the pre-check found nothing, so anything
    here now must be mine" - a same-task_id race (see the class docstring)
    is exactly the scenario where that assumption is false.
    """
    post_info_result = get_worktree_info(normalized_id)
    if not post_info_result.success:
        logger.warning(
            "Could not determine post-failure state after failed worktree create; leaving everything in "
            "place for manual review: task_id=%s error_code=%s",
            normalized_id,
            post_info_result.error_code,
        )
        return CreateCleanupResult(
            path_entry_exists_after=None,
            registration_exists_after=None,
            branch_exists_after=None,
            ownership_verified=False,
            automatic_cleanup_performed=False,
            manual_cleanup_required=True,
            cleanup_error_code=post_info_result.error_code,
        )

    post = post_info_result.info
    no_residue = not post.path_entry_exists and not post.registered_in_git_worktree and not post.branch_exists

    if no_residue:
        logger.info("No residue left after failed worktree create: task_id=%s", normalized_id)
    else:
        logger.warning(
            "Residue left after failed worktree create - Phase 2 performs no automatic deletion, manual "
            "cleanup required: task_id=%s path=%s path_entry_exists=%s registered=%s branch=%s branch_exists=%s",
            normalized_id,
            worktree_path,
            post.path_entry_exists,
            post.registered_in_git_worktree,
            branch_name,
            post.branch_exists,
        )

    return CreateCleanupResult(
        path_entry_exists_after=post.path_entry_exists,
        registration_exists_after=post.registered_in_git_worktree,
        branch_exists_after=post.branch_exists,
        ownership_verified=False,
        automatic_cleanup_performed=False,
        manual_cleanup_required=not no_residue,
        cleanup_error_code=None,
    )


def validate_worktree_boundary_for_task(task_id: Any) -> WorktreeValidationResult:
    """Public wrapper around the same path-boundary validation
    create_worktree()/remove_worktree() perform internally
    (_validate_path_boundary()), for callers outside this module that need
    to re-verify - immediately before doing something as consequential as
    launching an external command with this path as cwd - that a task's
    worktree path is still genuinely inside WORK_ROOT on the real
    filesystem, without reaching into this module's private
    _validate_path_boundary()/_worktree_path_for() helpers directly.
    """
    normalized_id = _normalize_task_id(task_id)
    if normalized_id is None:
        return _validation_error(ERROR_INVALID_TASK_ID)
    return _validate_path_boundary(_worktree_path_for(normalized_id))


def create_worktree(task_id: Any, base_branch: Any) -> WorktreeResult:
    """Create a dev task worktree at work/<task_id> on a new branch
    dev-task/<task_id> (the full UUID4) branched from `base_branch`.

    No Discord interaction, no agent execution, no commits: this only
    prepares the isolated working directory. Every path/branch this
    function ever touches is derived solely from the already-validated
    task_id, never from user-controlled text.
    """
    normalized_id = _normalize_task_id(task_id)
    if normalized_id is None:
        return WorktreeResult(
            success=False,
            task_id=None,
            worktree_path=None,
            branch_name=None,
            error_message=_ERROR_MESSAGES[ERROR_INVALID_TASK_ID],
            error_code=ERROR_INVALID_TASK_ID,
        )

    branch_validation = validate_base_branch(base_branch)
    if not branch_validation.success:
        return WorktreeResult(
            success=False,
            task_id=normalized_id,
            worktree_path=None,
            branch_name=None,
            error_message=branch_validation.error_message,
            error_code=branch_validation.error_code,
        )

    worktree_path = _worktree_path_for(normalized_id)
    branch_name = _branch_name_for(normalized_id)

    # Serialize every create/remove for this exact task_id within this
    # process: this closes the in-process window of the same-task_id race
    # described in CreateCleanupResult's docstring (two concurrent
    # create_worktree() calls, or a create racing a remove, for one
    # task_id). It does not protect against a second OS process or an
    # external `git worktree` invocation - out of scope for Phase 2.
    with _task_lock(normalized_id):
        info_result = get_worktree_info(normalized_id)
        if not info_result.success:
            return WorktreeResult(
                success=False,
                task_id=normalized_id,
                worktree_path=worktree_path,
                branch_name=branch_name,
                error_message=info_result.error_message,
                error_code=info_result.error_code,
            )
        info = info_result.info

        # Never overwrite or reuse an existing path/branch/registration: any
        # one of these already present means this task_id (or, in the
        # registration case, this exact path) is not free to claim.
        if info.path_entry_exists:
            return WorktreeResult(
                success=False,
                task_id=normalized_id,
                worktree_path=worktree_path,
                branch_name=branch_name,
                error_message=_ERROR_MESSAGES[ERROR_PATH_ALREADY_EXISTS],
                error_code=ERROR_PATH_ALREADY_EXISTS,
            )
        if info.registered_in_git_worktree:
            return WorktreeResult(
                success=False,
                task_id=normalized_id,
                worktree_path=worktree_path,
                branch_name=branch_name,
                error_message=_ERROR_MESSAGES[ERROR_ALREADY_REGISTERED],
                error_code=ERROR_ALREADY_REGISTERED,
            )
        if info.branch_exists:
            return WorktreeResult(
                success=False,
                task_id=normalized_id,
                worktree_path=worktree_path,
                branch_name=branch_name,
                error_message=_ERROR_MESSAGES[ERROR_BRANCH_ALREADY_EXISTS],
                error_code=ERROR_BRANCH_ALREADY_EXISTS,
            )

        boundary_check = _validate_path_boundary(worktree_path)
        if not boundary_check.success:
            return WorktreeResult(
                success=False,
                task_id=normalized_id,
                worktree_path=worktree_path,
                branch_name=branch_name,
                error_message=boundary_check.error_message,
                error_code=boundary_check.error_code,
            )

        add_result = run_git(
            ["worktree", "add", "-b", branch_name, str(worktree_path), str(base_branch)],
            cwd=REPO_ROOT,
            timeout=_WORKTREE_ADD_TIMEOUT_SECONDS,
        )
        if not add_result.success:
            cleanup_result = _cleanup_failed_create(normalized_id, worktree_path, branch_name)
            mapped_code = _map_git_error(add_result.error_code, default=ERROR_WORKTREE_ADD_FAILED)
            return WorktreeResult(
                success=False,
                task_id=normalized_id,
                worktree_path=worktree_path,
                branch_name=branch_name,
                error_message=_ERROR_MESSAGES[mapped_code],
                error_code=mapped_code,
                cleanup=cleanup_result,
            )

        logger.info(
            "Created dev worktree: task_id=%s branch=%s path=%s", normalized_id, branch_name, worktree_path
        )
        return WorktreeResult(
            success=True,
            task_id=normalized_id,
            worktree_path=worktree_path,
            branch_name=branch_name,
            error_message=None,
            error_code=None,
        )


def remove_worktree(task_id: Any) -> WorktreeRemoveResult:
    """Remove a dev task's worktree and, only once that succeeds, its
    branch - operating solely on the path/branch derived from `task_id`.

    Refuses outright, touching nothing, if the worktree registered at this
    path belongs to a different branch or is a detached checkout: only a
    registration whose branch exactly matches this task's own derived
    branch is ever removed. If `git worktree remove` fails, the branch is
    never touched. If the branch delete subsequently fails (e.g. it's
    checked out by some other worktree), that is reported as an explicit
    partial failure (worktree_removed=True, branch_removed=False) rather
    than silently treated as success or failure.

    worktree_removed/branch_removed/success are never taken directly from
    a git command's exit code: after `git worktree remove` and `git branch
    -D` each report success, this function re-measures the actual state via
    get_worktree_info()/check_ref_exists() and only reports success when
    that measurement confirms nothing remains. A junction or other reparse
    point can leave a dangling directory entry behind even though git
    itself reports a clean removal - that must never be reported back as a
    successful remove.
    """
    normalized_id = _normalize_task_id(task_id)
    if normalized_id is None:
        return WorktreeRemoveResult(
            success=False,
            task_id=None,
            worktree_path=None,
            branch_name=None,
            worktree_removed=False,
            branch_removed=False,
            error_message=_ERROR_MESSAGES[ERROR_INVALID_TASK_ID],
            error_code=ERROR_INVALID_TASK_ID,
        )

    worktree_path = _worktree_path_for(normalized_id)
    branch_name = _branch_name_for(normalized_id)

    # Serialize with any concurrent create_worktree()/remove_worktree() for
    # this exact task_id within this process - see create_worktree()'s
    # matching comment. Same lock object (keyed by task_id), so a create
    # and a remove for the same task never interleave either.
    with _task_lock(normalized_id):
        info_result = get_worktree_info(normalized_id)
        if not info_result.success:
            return WorktreeRemoveResult(
                success=False,
                task_id=normalized_id,
                worktree_path=worktree_path,
                branch_name=branch_name,
                worktree_removed=False,
                branch_removed=False,
                error_message=info_result.error_message,
                error_code=info_result.error_code,
            )
        info = info_result.info

        if not info.path_entry_exists and not info.registered_in_git_worktree:
            return WorktreeRemoveResult(
                success=False,
                task_id=normalized_id,
                worktree_path=worktree_path,
                branch_name=branch_name,
                worktree_removed=False,
                branch_removed=False,
                error_message=_ERROR_MESSAGES[ERROR_WORKTREE_NOT_FOUND],
                error_code=ERROR_WORKTREE_NOT_FOUND,
            )

        # Boundary check #1, at the start of removal processing: WORK_ROOT
        # and this path must be genuinely inside it on the real filesystem,
        # not just by lexical prefix - see _validate_path_boundary().
        boundary_check = _validate_path_boundary(worktree_path)
        if not boundary_check.success:
            logger.error(
                "Refusing to remove worktree: path boundary check failed at start of removal: "
                "task_id=%s path=%s",
                normalized_id,
                worktree_path,
            )
            return WorktreeRemoveResult(
                success=False,
                task_id=normalized_id,
                worktree_path=worktree_path,
                branch_name=branch_name,
                worktree_removed=False,
                branch_removed=False,
                error_message=boundary_check.error_message,
                error_code=boundary_check.error_code,
            )

        if info.registered_in_git_worktree and info.branch_mismatch:
            logger.warning(
                "Refusing to remove worktree: a foreign branch (or detached HEAD) is registered at this "
                "path: task_id=%s path=%s expected_branch=%s registered_branch_ref=%s detached=%s",
                normalized_id,
                worktree_path,
                branch_name,
                info.registered_branch_ref,
                info.registered_detached,
            )
            return WorktreeRemoveResult(
                success=False,
                task_id=normalized_id,
                worktree_path=worktree_path,
                branch_name=branch_name,
                worktree_removed=False,
                branch_removed=False,
                error_message=_ERROR_MESSAGES[ERROR_FOREIGN_WORKTREE],
                error_code=ERROR_FOREIGN_WORKTREE,
            )

        # Boundary check #2, immediately before the actual removal: closes
        # the TOCTOU window between the check above (and the foreign-branch
        # decision, which itself takes some time) and the filesystem-
        # affecting `git worktree remove --force` call below. If this
        # re-check fails, `git worktree remove` is never called.
        reverify_boundary_check = _validate_path_boundary(worktree_path)
        if not reverify_boundary_check.success:
            logger.error(
                "Refusing to remove worktree: path boundary re-check immediately before removal failed: "
                "task_id=%s path=%s",
                normalized_id,
                worktree_path,
            )
            return WorktreeRemoveResult(
                success=False,
                task_id=normalized_id,
                worktree_path=worktree_path,
                branch_name=branch_name,
                worktree_removed=False,
                branch_removed=False,
                error_message=reverify_boundary_check.error_message,
                error_code=reverify_boundary_check.error_code,
            )

        remove_result = run_git(
            ["worktree", "remove", "--force", str(worktree_path)], cwd=REPO_ROOT, timeout=_GIT_TIMEOUT_SECONDS
        )
        if not remove_result.success:
            mapped_code = _map_git_error(remove_result.error_code, default=ERROR_WORKTREE_REMOVE_FAILED)
            return WorktreeRemoveResult(
                success=False,
                task_id=normalized_id,
                worktree_path=worktree_path,
                branch_name=branch_name,
                worktree_removed=False,
                branch_removed=False,
                manual_cleanup_required=False,
                error_message=_ERROR_MESSAGES[mapped_code],
                error_code=mapped_code,
            )

        # `git worktree remove` returning success is not, on its own, proof
        # that the path/registration are actually gone: a junction or other
        # reparse point can leave a dangling directory entry behind even
        # though git itself reports a clean removal. Always re-measure via
        # a fresh get_worktree_info() call rather than trusting the exit
        # code - this single re-check also tells us the real, current
        # branch_exists, replacing what used to be a separate check_ref_exists()
        # call right after remove_result.success.
        verify_result = get_worktree_info(normalized_id)
        if not verify_result.success:
            logger.error(
                "Could not verify post-removal state after `git worktree remove` reported success: "
                "task_id=%s error_code=%s",
                normalized_id,
                verify_result.error_code,
            )
            return WorktreeRemoveResult(
                success=False,
                task_id=normalized_id,
                worktree_path=worktree_path,
                branch_name=branch_name,
                worktree_removed=False,
                branch_removed=False,
                manual_cleanup_required=True,
                error_message=verify_result.error_message,
                error_code=verify_result.error_code,
            )
        verified = verify_result.info

        worktree_removed = not verified.path_entry_exists and not verified.registered_in_git_worktree
        if not worktree_removed:
            logger.error(
                "`git worktree remove` reported success but residue remains after verification: "
                "task_id=%s path=%s path_entry_exists=%s registered=%s",
                normalized_id,
                worktree_path,
                verified.path_entry_exists,
                verified.registered_in_git_worktree,
            )
            return WorktreeRemoveResult(
                success=False,
                task_id=normalized_id,
                worktree_path=worktree_path,
                branch_name=branch_name,
                worktree_removed=False,
                branch_removed=False,
                manual_cleanup_required=True,
                error_message=_ERROR_MESSAGES[ERROR_WORKTREE_REMOVE_INCOMPLETE],
                error_code=ERROR_WORKTREE_REMOVE_INCOMPLETE,
            )

        if not verified.branch_exists:
            logger.info(
                "Removed dev worktree; branch already absent: task_id=%s branch=%s",
                normalized_id,
                branch_name,
            )
            return WorktreeRemoveResult(
                success=True,
                task_id=normalized_id,
                worktree_path=worktree_path,
                branch_name=branch_name,
                worktree_removed=True,
                branch_removed=True,
                manual_cleanup_required=False,
                error_message=None,
                error_code=None,
            )

        branch_delete_result = run_git(
            ["branch", "-D", branch_name], cwd=REPO_ROOT, timeout=_GIT_TIMEOUT_SECONDS
        )
        if not branch_delete_result.success:
            mapped_code = _map_git_error(branch_delete_result.error_code, default=ERROR_BRANCH_REMOVE_FAILED)
            logger.warning(
                "Worktree removed but branch delete failed: task_id=%s branch=%s error_code=%s",
                normalized_id,
                branch_name,
                mapped_code,
            )
            return WorktreeRemoveResult(
                success=False,
                task_id=normalized_id,
                worktree_path=worktree_path,
                branch_name=branch_name,
                worktree_removed=True,
                branch_removed=False,
                manual_cleanup_required=False,
                error_message=_ERROR_MESSAGES[mapped_code],
                error_code=mapped_code,
            )

        # As with the worktree removal above, `git branch -D` reporting
        # success is not itself proof the branch is actually gone - verify.
        final_branch_check = check_ref_exists(
            f"refs/heads/{branch_name}", cwd=REPO_ROOT, timeout=_GIT_TIMEOUT_SECONDS
        )
        if final_branch_check.exists is None:
            mapped_code = _map_git_error(final_branch_check.error_code, default=ERROR_BRANCH_REMOVE_FAILED)
            return WorktreeRemoveResult(
                success=False,
                task_id=normalized_id,
                worktree_path=worktree_path,
                branch_name=branch_name,
                worktree_removed=True,
                branch_removed=False,
                manual_cleanup_required=True,
                error_message=_ERROR_MESSAGES[mapped_code],
                error_code=mapped_code,
            )
        if final_branch_check.exists:
            logger.error(
                "`git branch -D` reported success but the branch still exists after verification: "
                "task_id=%s branch=%s",
                normalized_id,
                branch_name,
            )
            return WorktreeRemoveResult(
                success=False,
                task_id=normalized_id,
                worktree_path=worktree_path,
                branch_name=branch_name,
                worktree_removed=True,
                branch_removed=False,
                manual_cleanup_required=True,
                error_message=_ERROR_MESSAGES[ERROR_BRANCH_REMOVE_FAILED],
                error_code=ERROR_BRANCH_REMOVE_FAILED,
            )

        logger.info("Removed dev worktree and branch: task_id=%s branch=%s", normalized_id, branch_name)
        return WorktreeRemoveResult(
            success=True,
            task_id=normalized_id,
            worktree_path=worktree_path,
            branch_name=branch_name,
            worktree_removed=True,
            branch_removed=True,
            manual_cleanup_required=False,
            error_message=None,
            error_code=None,
        )


def cleanup_stale_worktree_metadata() -> PruneResult:
    """Run `git worktree prune`: a whole-repository operation that drops
    administrative entries for every worktree in this repo whose path no
    longer exists, not just this module's own dev-task worktrees.

    This is the ONLY function in this module that ever calls `git worktree
    prune`. create_worktree()/remove_worktree()/get_worktree_info() never
    call it, including on their own failure paths (see
    _cleanup_failed_create()), specifically because prune's effect is
    repo-wide and not scoped to a single task_id. Nothing in this module
    calls this function automatically; it is exposed for a caller (e.g. a
    future scheduled maintenance job) to invoke deliberately and is not
    wired into Phase 3's per-task flow.
    """
    result = run_git(["worktree", "prune"], cwd=REPO_ROOT, timeout=_GIT_TIMEOUT_SECONDS)
    if not result.success:
        mapped_code = _map_git_error(result.error_code, default=ERROR_PRUNE_FAILED)
        logger.warning("git worktree prune failed: error_code=%s", mapped_code)
        return PruneResult(success=False, error_message=_ERROR_MESSAGES[mapped_code], error_code=mapped_code)

    logger.info("git worktree prune completed")
    return PruneResult(success=True, error_message=None, error_code=None)
