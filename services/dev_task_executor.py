from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from services import dev_task_store as store
from services.dev_git_runner import run_git
from services.dev_worktree import create_worktree


logger = logging.getLogger("secretary-bot.dev_task_executor")

# Phase 3 never runs Codex or Claude Code - this module only ever performs a
# short, read-only, no-op "execution" so the rest of the pipeline (worktree
# lifecycle, Discord approval/result UI, notification delivery) can be built
# and tested end-to-end before any real headless engine exists. It never
# writes to the worktree, never uses shell=True, and only ever runs `git
# status`/`git diff` (both read-only) via services.dev_git_runner.run_git(),
# the same hardened subprocess wrapper Phase 2 uses.
_DUMMY_EXECUTION_SECONDS = 2.0
_GIT_READ_TIMEOUT_SECONDS = 30.0
_MAX_DIFF_SUMMARY_LENGTH = 1500
_MAX_RESULT_SUMMARY_LENGTH = 200

# Global, single-slot concurrency gate: Phase 3 allows at most one dev task
# executing at a time, rejects a second attempt outright, and implements no
# queue.
#
# reserve_slot()/release_slot() split the "claim the slot" step out from
# "actually run the dummy work" specifically so callers can reserve the slot
# and only THEN attempt the store-level claim_task_for_execution() CAS and
# worktree creation - never the other way around. A second Codex review
# found that reserving the slot only inside run_dummy_execution() (i.e.
# after the worktree already existed) let two concurrently-approved tasks
# both create worktrees before either noticed the other.
_execution_lock = asyncio.Lock()
_current_task_id: str | None = None
_current_asyncio_task: "asyncio.Task | None" = None


@dataclass(frozen=True)
class DevTaskExecutionResult:
    success: bool
    changed_file_count: int
    diff_summary: str
    duration_seconds: float
    error_message: str | None
    cancelled: bool = False


@dataclass(frozen=True)
class DevTaskExecutionOutcome:
    """The full result of execute_approved_task(): everything a caller
    needs to build a Discord notification, without that caller ever having
    to reason about slot reservation, store CAS calls, or worktree
    creation order itself.

    claimed=False means no worktree was ever created (the execution slot
    could not be reserved, or the store claim itself lost a race) - never
    offer a keep/discard UI in that case, there is nothing to decide about.

    final_task is None only when a store write that SHOULD have succeeded
    did not (e.g. disk failure) - callers MUST treat this as "the outcome
    is not durably confirmed" and must not tell the user the task
    completed or failed, since that was never actually saved.
    """

    task_id: str
    claimed: bool
    worktree_created: bool
    branch_name: str | None
    execution_result: DevTaskExecutionResult | None
    final_task: "store.DevTask | None"
    failure_reason: str | None


def is_busy() -> bool:
    return _execution_lock.locked()


def get_current_task_id() -> str | None:
    return _current_task_id


async def reserve_slot(task_id: str) -> bool:
    """Reserve the single global execution slot for `task_id`. Returns
    False immediately (never waits, never queues) if another task already
    holds it.

    Checking `_execution_lock.locked()` and then calling `acquire()` never
    yields to the event loop in between when the lock is free
    (asyncio.Lock.acquire() only ever truly awaits if it must wait), so
    this check-then-acquire is race-free within this single-process,
    single-event-loop bot. On success, the caller owns the slot until it
    calls release_slot(task_id) - directly, if the reservation ends up
    unused (e.g. the following store claim fails), or implicitly via
    run_dummy_execution()'s own cleanup once execution actually starts.
    """
    if _execution_lock.locked():
        return False
    await _execution_lock.acquire()
    global _current_task_id
    _current_task_id = task_id
    return True


def release_slot(task_id: str) -> None:
    """Release a slot previously reserved via reserve_slot(task_id) for the
    SAME task_id. A mismatched task_id indicates a caller bug (logged), but
    the lock is still released - leaving the executor stuck "busy" forever
    would be worse than an over-eager release.
    """
    global _current_task_id, _current_asyncio_task
    if task_id != _current_task_id:
        logger.error(
            "release_slot called for a task_id that does not hold the slot: called_with=%s holder=%s",
            task_id,
            _current_task_id,
        )
    _current_task_id = None
    _current_asyncio_task = None
    if _execution_lock.locked():
        _execution_lock.release()


async def cancel_dummy_execution_and_wait(task_id: str, *, timeout: float = 10.0) -> bool:
    """Request cancellation of the currently-running DUMMY execution for
    `task_id` and wait for it to actually finish observing the
    cancellation before returning, rather than firing-and-forgetting.

    Returns False immediately if `task_id` is not the one currently
    executing. Returns True once cancellation was requested - regardless
    of whether the wait completed before `timeout` - since the request
    itself was made either way; a timeout is logged so an operator can
    notice if the dummy executor is ever stuck.

    Named specifically for Phase 3's dummy execution. A future Phase 4+
    stop of a REAL headless engine process is a fundamentally different
    operation (terminating an actual OS process, not cancelling an asyncio
    Task) and must use its own, differently-named API rather than being
    folded into this one.
    """
    if task_id != _current_task_id or _current_asyncio_task is None:
        return False

    target_task = _current_asyncio_task
    target_task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(target_task), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(
            "Timed out waiting for dummy execution to observe cancellation: task_id=%s", task_id
        )
    except Exception:
        logger.exception("Unexpected error while waiting for cancelled dummy execution: task_id=%s", task_id)
    return True


def _count_changed_files(git_status_short_output: str) -> int:
    return len([line for line in git_status_short_output.splitlines() if line.strip()])


async def run_dummy_execution(task_id: str, worktree_path: Path, timeout_seconds: int) -> DevTaskExecutionResult:
    """Perform Phase 3's dummy "execution": wait briefly, then read (never
    write) `git status`/`git diff` in `worktree_path`, and return a
    structured result.

    The caller MUST have already reserved the execution slot for this
    exact task_id via reserve_slot() before calling this - this function
    does not reserve it itself (see the module-level note on
    reserve_slot()/release_slot()), and always releases it via
    release_slot() before returning, regardless of outcome.
    """
    global _current_asyncio_task
    _current_asyncio_task = asyncio.current_task()
    start = time.monotonic()
    try:
        sleep_seconds = min(_DUMMY_EXECUTION_SECONDS, max(0.1, float(timeout_seconds)))
        await asyncio.sleep(sleep_seconds)

        status_result = await asyncio.to_thread(
            run_git, ["status", "--short"], cwd=worktree_path, timeout=_GIT_READ_TIMEOUT_SECONDS
        )
        diff_result = await asyncio.to_thread(
            run_git, ["diff", "--stat"], cwd=worktree_path, timeout=_GIT_READ_TIMEOUT_SECONDS
        )

        if not status_result.success or not diff_result.success:
            logger.error(
                "Dummy execution could not read worktree state: task_id=%s status_ok=%s diff_ok=%s",
                task_id,
                status_result.success,
                diff_result.success,
            )
            return DevTaskExecutionResult(
                success=False,
                changed_file_count=0,
                diff_summary="",
                duration_seconds=time.monotonic() - start,
                error_message="worktree内の状態確認に失敗しました。",
            )

        status_text = status_result.stdout.decode("utf-8", errors="replace")
        diff_text = diff_result.stdout.decode("utf-8", errors="replace").strip()
        changed_file_count = _count_changed_files(status_text)
        diff_summary = diff_text[:_MAX_DIFF_SUMMARY_LENGTH] or "変更はありません（ダミー実行のため）。"

        logger.info(
            "Dummy execution completed: task_id=%s changed_file_count=%s", task_id, changed_file_count
        )
        return DevTaskExecutionResult(
            success=True,
            changed_file_count=changed_file_count,
            diff_summary=diff_summary,
            duration_seconds=time.monotonic() - start,
            error_message=None,
        )
    except asyncio.CancelledError:
        logger.info("Dummy execution cancelled: task_id=%s", task_id)
        return DevTaskExecutionResult(
            success=False,
            changed_file_count=0,
            diff_summary="",
            duration_seconds=time.monotonic() - start,
            error_message="ユーザーによりキャンセルされました。",
            cancelled=True,
        )
    finally:
        release_slot(task_id)


async def execute_approved_task(
    task_id: str,
    *,
    base_branch: str,
    timeout_seconds: int,
    branch_name: str,
    engine_override: str | None = None,
    routing_reason_override: str | None = None,
) -> DevTaskExecutionOutcome:
    """Full approval -> slot reservation -> store claim -> worktree
    creation -> dummy execution -> terminal-state save pipeline, entirely
    independent of discord.py (no Discord objects are touched anywhere in
    this module) so it can be exercised and unit-tested without discord.py
    installed. Callers (views/dev_task.py) are responsible for everything
    Discord-specific: building/sending the notification from the returned
    DevTaskExecutionOutcome, and recording how that notification was
    delivered via store.set_notification_method_once() afterward.

    Every step is ordered so that nothing later ever runs unless the step
    before it durably succeeded: the execution slot is reserved before the
    store claim; the store claim (pending_approval -> running) is confirmed
    before create_worktree() is ever called; and the terminal
    completed/failed state is saved via finish_task() - and its success
    checked - before this function returns anything a caller could mistake
    for a confirmed outcome (see DevTaskExecutionOutcome.final_task).
    """
    reserved = await reserve_slot(task_id)
    if not reserved:
        reason = "他の開発タスクが実行中のため開始できませんでした。"
        fail_result = store.fail_pending_approval(task_id, reason=reason)
        return DevTaskExecutionOutcome(
            task_id=task_id,
            claimed=False,
            worktree_created=False,
            branch_name=None,
            execution_result=None,
            final_task=fail_result.task if fail_result.success else None,
            failure_reason=reason,
        )

    claim_result = store.claim_task_for_execution(
        task_id,
        branch_name=branch_name,
        engine_override=engine_override,
        routing_reason_override=routing_reason_override,
    )
    if not claim_result.success:
        release_slot(task_id)
        reason = f"タスクの確保に失敗しました: {claim_result.error_message}"
        return DevTaskExecutionOutcome(
            task_id=task_id,
            claimed=False,
            worktree_created=False,
            branch_name=None,
            execution_result=None,
            final_task=None,
            failure_reason=reason,
        )

    worktree_result = await asyncio.to_thread(create_worktree, task_id, base_branch)
    if not worktree_result.success:
        release_slot(task_id)
        reason = worktree_result.error_message or "worktreeの作成に失敗しました。"
        finish_result = store.finish_task(task_id, success=False, error_message=reason)
        return DevTaskExecutionOutcome(
            task_id=task_id,
            claimed=True,
            worktree_created=False,
            branch_name=worktree_result.branch_name,
            execution_result=None,
            final_task=finish_result.task if finish_result.success else None,
            failure_reason=reason,
        )

    execution_result = await run_dummy_execution(task_id, worktree_result.worktree_path, timeout_seconds)

    if execution_result.success:
        result_summary = (
            f"{execution_result.changed_file_count}件のファイル変更 / "
            f"{execution_result.diff_summary[:_MAX_RESULT_SUMMARY_LENGTH]}"
        )
        finish_result = store.finish_task(task_id, success=True, result_summary=result_summary)
        failure_reason = None if finish_result.success else finish_result.error_message
    else:
        reason = execution_result.error_message or "ダミー実行に失敗しました。"
        finish_result = store.finish_task(task_id, success=False, error_message=reason)
        failure_reason = reason if finish_result.success else finish_result.error_message

    return DevTaskExecutionOutcome(
        task_id=task_id,
        claimed=True,
        worktree_created=True,
        branch_name=worktree_result.branch_name,
        execution_result=execution_result,
        final_task=finish_result.task if finish_result.success else None,
        failure_reason=failure_reason,
    )
