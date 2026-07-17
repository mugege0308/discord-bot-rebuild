from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from services import dev_task_store as store
from services.dev_git_runner import run_git
from services.dev_task_router import DevTaskEngine


logger = logging.getLogger("secretary-bot.dev_task_executor")

# Phase 3 never ran Codex or Claude Code - it only performed a short,
# read-only, no-op "execution" so the rest of the pipeline (worktree
# lifecycle, Discord approval/result UI, notification delivery) could be
# built and tested end-to-end before any real headless engine existed. That
# dummy path (run_dummy_execution()/reserve_slot()/release_slot()/
# cancel_dummy_execution_and_wait()) is kept exactly as-is below - still
# fully functional and still covered by its own tests.
#
# Phase 4A pivoted real Claude Code execution away from a git worktree of
# the real repository to a disposable, allowlisted isolated-copy sandbox
# (services.dev_copy_workspace + services.claude_headless) that shares
# nothing - not even a hardlink - with the real repository or its secrets.
# That mechanism is deliberately NOT wired into this Store-backed
# execute_approved_task() pipeline yet: doing so naively would also require
# updating the keep/discard Result View (services.dev_task_decision /
# views.dev_task) to call services.dev_copy_workspace.discard_sandbox()
# instead of services.dev_worktree.remove_worktree() for a sandbox-based
# task - without that, a sandbox could be silently left orphaned on disk
# forever while the Store still records "discarded". That follow-up is
# left to a future phase; until then, execute_approved_task() safely
# rejects every engine (see _engine_gate_failure_reason()), and real
# execution is only ever exercised directly via services.claude_headless.run().
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
    # Additional fields some future execution mechanism may want to
    # populate - always the dummy-path defaults (False/None/False/False/
    # False) when this result came from run_dummy_execution().
    timed_out: bool = False
    returncode: int | None = None
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    manual_review_required: bool = False


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


async def cancel_current_execution_and_wait(task_id: str, *, timeout: float = 30.0) -> bool:
    """Request cancellation of whichever execution is currently running for
    `task_id` and wait for it to actually finish observing the
    cancellation before returning. This is what /dev_task_stop calls.

    Currently identical to cancel_dummy_execution_and_wait() (only the
    Phase 3 dummy asyncio-task path is ever reachable via the live
    /dev_task pipeline right now - see the module-level note on why real
    Claude execution isn't wired into execute_approved_task() yet). Kept
    as its own, separately-named function so a future phase that wires a
    real, cancel-token-based execution mechanism into this module has a
    stable name to extend without another rename.
    """
    if task_id != _current_task_id or _current_asyncio_task is None:
        return False

    target_task = _current_asyncio_task
    target_task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(target_task), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("Timed out waiting for execution to observe cancellation: task_id=%s", task_id)
    except Exception:
        logger.exception("Unexpected error while waiting for cancelled execution: task_id=%s", task_id)
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


def _engine_gate_failure_reason(engine: str) -> str | None:
    """The live /dev_task Store-backed pipeline does not execute ANY engine
    right now - this always returns a non-None, safe, Discord-displayable
    reason. See the module-level note above for why: Phase 4A's real
    Claude execution mechanism (isolated-copy sandbox) is intentionally
    not wired into execute_approved_task() yet, and the old
    worktree-based real-execution path it replaced must never be used for
    real Claude Code execution again. Codex remains entirely unimplemented.
    Neither engine ever falls back to the other, and neither falls back to
    Phase 3's dummy execution.
    """
    if engine == DevTaskEngine.CODEX.value:
        return "このエンジンでの実行には対応していません（Codexは未実装です）。"
    if engine == DevTaskEngine.CLAUDE.value:
        return (
            "Claude Codeの実行は安全な隔離コピー（isolated copy）方式へ移行中のため、"
            "現時点では/dev_taskからの自動実行には対応していません。"
        )
    return "未対応のエンジンが指定されています。"


async def execute_approved_task(
    task_id: str,
    *,
    base_branch: str,
    timeout_seconds: int,
    branch_name: str,
    engine: str,
    prompt_text: str,
    engine_override: str | None = None,
    routing_reason_override: str | None = None,
) -> DevTaskExecutionOutcome:
    """Approval entry point for the live /dev_task pipeline.

    In Phase 4A-v2 this always safely rejects (see
    _engine_gate_failure_reason()) - claimed=False, no slot ever reserved,
    no worktree/sandbox ever created, nothing else touched beyond
    recording the rejection via store.fail_pending_approval(). This is
    entirely independent of discord.py (no Discord objects are touched
    anywhere in this module) so it can be exercised and unit-tested
    without discord.py installed.

    base_branch/timeout_seconds/branch_name/engine_override/
    routing_reason_override are accepted (and currently unused) so the
    call site in views/dev_task.py does not need to change when a future
    phase re-enables real execution here.
    """
    gate_reason = _engine_gate_failure_reason(engine)
    if gate_reason is None:
        # Should never happen - _engine_gate_failure_reason() always
        # rejects in Phase 4A-v2. Fail safely rather than silently
        # proceeding through unreviewed/removed execution machinery if
        # this ever changes without this function being updated too.
        logger.error("_engine_gate_failure_reason() unexpectedly returned None for engine=%r", engine)
        gate_reason = "内部エラーのため実行できませんでした。"

    fail_result = store.fail_pending_approval(task_id, reason=gate_reason)
    return DevTaskExecutionOutcome(
        task_id=task_id,
        claimed=False,
        worktree_created=False,
        branch_name=None,
        execution_result=None,
        final_task=fail_result.task if fail_result.success else None,
        failure_reason=gate_reason,
    )
