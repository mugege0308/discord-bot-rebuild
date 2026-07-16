from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from dataclasses import dataclass

from services import dev_task_store as store
from services.dev_worktree import (
    ERROR_FOREIGN_WORKTREE,
    ERROR_PATH_BOUNDARY_VIOLATION,
    ERROR_WORKTREE_NOT_FOUND,
    remove_worktree,
)


logger = logging.getLogger("secretary-bot.dev_task_decision")

# Never force past a foreign-worktree/boundary-safety refusal from Phase 2 -
# these mean removal was refused specifically to avoid touching something
# that might not belong to this task. The decision is deliberately left
# unrecorded (None) in that case so the requester can investigate and retry.
_SAFETY_REFUSAL_CODES = frozenset({ERROR_FOREIGN_WORKTREE, ERROR_PATH_BOUNDARY_VIOLATION})

# This module is deliberately discord.py-independent: it is the "keep or
# discard this task's worktree" decision logic factored out of
# views/dev_task.py specifically so it (and the concurrency/CAS behavior
# that matters most - a second Codex review flagged that a plain
# per-View Python bool was not enough to make concurrent keep/discard clicks
# mutually exclusive) can be unit-tested without discord.py installed.


@dataclass(frozen=True)
class DevTaskDecisionOutcome:
    """success=True iff this call is the one that durably recorded the
    decision. already_decided=True means some other call (concurrent or
    earlier) already recorded a decision first - this call changed
    nothing. refused=True means a discard was blocked by a Phase 2 safety
    refusal (foreign worktree / boundary violation) - decision is still
    None, nothing was touched, and a later retry remains possible.
    message is always safe to show verbatim in Discord.
    """

    success: bool
    decision: str | None
    already_decided: bool
    refused: bool
    manual_cleanup_required: bool
    message: str
    error_message: str | None = None


# Per-task_id locks serializing keep/discard attempts for the SAME task_id
# within this process. Refcounted so the dict does not grow without bound
# over a long-running process - the same pattern services.dev_worktree uses
# for its own per-task_id worktree lock, after an earlier Phase 2 review
# specifically flagged an unbounded version of this idea.
_decision_locks_guard = threading.Lock()


class _RefCountedLock:
    __slots__ = ("lock", "ref_count")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.ref_count = 0


_decision_locks: dict[str, _RefCountedLock] = {}


@contextlib.asynccontextmanager
async def _task_decision_lock(task_id: str):
    with _decision_locks_guard:
        entry = _decision_locks.get(task_id)
        if entry is None:
            entry = _RefCountedLock()
            _decision_locks[task_id] = entry
        entry.ref_count += 1

    await entry.lock.acquire()
    try:
        yield
    finally:
        entry.lock.release()
        with _decision_locks_guard:
            entry.ref_count -= 1
            if entry.ref_count == 0 and _decision_locks.get(task_id) is entry:
                del _decision_locks[task_id]


async def keep_worktree(task_id: str) -> DevTaskDecisionOutcome:
    """Record "kept" for `task_id`'s worktree - never touches the
    filesystem, only the store. Concurrent calls for the same task_id
    (keep vs. keep, or keep vs. discard) serialize on the per-task lock;
    whichever loses the race sees decision already set and changes
    nothing.
    """
    async with _task_decision_lock(task_id):
        task_result = store.get_task(task_id)
        if not task_result.success:
            return DevTaskDecisionOutcome(
                success=False,
                decision=None,
                already_decided=False,
                refused=False,
                manual_cleanup_required=False,
                message="タスクが見つかりません。",
                error_message=task_result.error_message,
            )
        if task_result.task.decision is not None:
            return DevTaskDecisionOutcome(
                success=False,
                decision=task_result.task.decision,
                already_decided=True,
                refused=False,
                manual_cleanup_required=False,
                message="既に判断済みです。",
            )

        finalize_result = store.finalize_decision(task_id, decision="kept")
        if not finalize_result.success:
            if finalize_result.error_code == store.ERROR_DECISION_ALREADY_MADE:
                return DevTaskDecisionOutcome(
                    success=False,
                    decision=None,
                    already_decided=True,
                    refused=False,
                    manual_cleanup_required=False,
                    message="既に判断済みです。",
                )
            return DevTaskDecisionOutcome(
                success=False,
                decision=None,
                already_decided=False,
                refused=False,
                manual_cleanup_required=False,
                message=f"保持の記録に失敗しました: {finalize_result.error_message}",
                error_message=finalize_result.error_message,
            )

        logger.info("Dev task worktree kept: task_id=%s", task_id)
        return DevTaskDecisionOutcome(
            success=True,
            decision="kept",
            already_decided=False,
            refused=False,
            manual_cleanup_required=False,
            message="worktreeを保持しました。",
        )


async def discard_worktree(task_id: str) -> DevTaskDecisionOutcome:
    """Record "discarded" for `task_id`'s worktree and remove it via
    services.dev_worktree.remove_worktree() - decision is only ever
    recorded once remove_worktree() has actually run and its outcome is
    known, and only when that outcome isn't a foreign-worktree/boundary
    safety refusal.
    """
    async with _task_decision_lock(task_id):
        task_result = store.get_task(task_id)
        if not task_result.success:
            return DevTaskDecisionOutcome(
                success=False,
                decision=None,
                already_decided=False,
                refused=False,
                manual_cleanup_required=False,
                message="タスクが見つかりません。",
                error_message=task_result.error_message,
            )
        if task_result.task.decision is not None:
            return DevTaskDecisionOutcome(
                success=False,
                decision=task_result.task.decision,
                already_decided=True,
                refused=False,
                manual_cleanup_required=False,
                message="既に判断済みです。",
            )

        remove_result = await asyncio.to_thread(remove_worktree, task_id)

        if remove_result.error_code in _SAFETY_REFUSAL_CODES:
            logger.warning(
                "Discard refused by worktree safety check, decision left unrecorded: task_id=%s code=%s",
                task_id,
                remove_result.error_code,
            )
            return DevTaskDecisionOutcome(
                success=False,
                decision=None,
                already_decided=False,
                refused=True,
                manual_cleanup_required=False,
                message=f"破棄を中止しました（安全確認のため）: {remove_result.error_message}",
                error_message=remove_result.error_message,
            )

        # Any other remove_worktree() outcome (fully removed, already gone,
        # or a non-safety partial failure) still means "the requester's
        # choice to discard" is the correct thing to persist - only the
        # safety-refusal case above leaves the decision unrecorded.
        finalize_result = store.finalize_decision(task_id, decision="discarded")
        if not finalize_result.success:
            if finalize_result.error_code == store.ERROR_DECISION_ALREADY_MADE:
                return DevTaskDecisionOutcome(
                    success=False,
                    decision=None,
                    already_decided=True,
                    refused=False,
                    manual_cleanup_required=remove_result.manual_cleanup_required,
                    message="既に判断済みです。",
                )
            return DevTaskDecisionOutcome(
                success=False,
                decision=None,
                already_decided=False,
                refused=False,
                manual_cleanup_required=remove_result.manual_cleanup_required,
                message=f"破棄の記録に失敗しました: {finalize_result.error_message}",
                error_message=finalize_result.error_message,
            )

        if remove_result.success:
            message = "worktreeを破棄しました。"
        elif remove_result.error_code == ERROR_WORKTREE_NOT_FOUND:
            message = "対象のworktreeは既に存在しませんでした。"
        else:
            manual_note = (
                " 手動での確認が必要な残骸が残っている可能性があります。"
                if remove_result.manual_cleanup_required
                else ""
            )
            message = (
                f"破棄処理を試みましたが完全には成功しませんでした: {remove_result.error_message}{manual_note}"
            )

        logger.info("Dev task worktree discarded: task_id=%s success=%s", task_id, remove_result.success)
        return DevTaskDecisionOutcome(
            success=True,
            decision="discarded",
            already_decided=False,
            refused=False,
            manual_cleanup_required=remove_result.manual_cleanup_required,
            message=message,
        )
