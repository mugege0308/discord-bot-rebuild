"""Sequential worker for persisted Batch image jobs.

This module is intentionally service-only.  It consumes the lifecycle API and
the 3C-11b single-image adapter, but does not register commands, start from
Bot lifecycle hooks, enqueue Gallery work, or run Selection AI.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import logging
import threading
import time
from typing import Any

from services.batch_generation import (
    JOB_CANCELLED,
    JOB_COMPLETED,
    JOB_FAILED,
    BatchSpec,
    manager as batch_manager,
)
from services.batch_generation_adapter import (
    generate_one_batch_image,
)
from services.sd_control import is_generation_running


logger = logging.getLogger("secretary-bot.batch_generation_worker")


class BatchWorkerError(RuntimeError):
    """Worker-level failure that is persisted against the current job."""


class BatchGenerationWorker:
    """Run claimed Batch jobs one image at a time in variation order."""

    def __init__(
        self,
        *,
        manager: Any = batch_manager,
        generate_one: Callable[..., object] = generate_one_batch_image,
        generation_lock_is_active: Callable[[], bool] = is_generation_running,
        sleep: Callable[[float], None] = time.sleep,
        lock_poll_seconds: float = 0.1,
    ) -> None:
        if lock_poll_seconds < 0:
            raise ValueError("lock_poll_secondsは0以上で指定してください。")
        self._manager = manager
        self._generate_one = generate_one
        self._generation_lock_is_active = generation_lock_is_active
        self._sleep = sleep
        self._lock_poll_seconds = lock_poll_seconds
        self._execution_lock = threading.Lock()
        self._events_lock = threading.RLock()
        self._job_events: dict[str, threading.Event] = {}
        self._shutdown_event = threading.Event()

    @property
    def active_job_ids(self) -> tuple[str, ...]:
        with self._events_lock:
            return tuple(self._job_events)

    def request_stop(self, job_id: str) -> None:
        """Ask an active job to stop before its next image."""

        with self._events_lock:
            event = self._job_events.get(job_id.strip())
        if event is not None:
            event.set()

    def shutdown(self) -> None:
        """Stop active jobs after their current image returns."""

        self._shutdown_event.set()
        with self._events_lock:
            events = tuple(self._job_events.values())
        for event in events:
            event.set()

    close = shutdown

    def run_jobs(self, job_ids: Iterable[str], requested_by: int) -> list[dict[str, Any]]:
        """Run a set of jobs sequentially, preserving the supplied job order."""

        results = []
        for job_id in job_ids:
            results.append(self.run_job(str(job_id), requested_by))
        return results

    def run_job(self, job_id: str, requested_by: int) -> dict[str, Any]:
        """Claim and run one job; duplicate starts become a no-op snapshot."""

        normalized_job_id = str(job_id or "").strip()
        if not normalized_job_id:
            raise BatchWorkerError("job_idが空です。")

        # One worker instance never runs two jobs concurrently.  The database
        # claim below also protects against another worker instance/process.
        with self._execution_lock:
            claimed = self._manager.try_claim_job(normalized_job_id, requested_by)
            if not claimed:
                return self._manager.get_job_for_user(normalized_job_id, requested_by)

            event = threading.Event()
            with self._events_lock:
                self._job_events[normalized_job_id] = event
            try:
                return self._run_claimed_job(normalized_job_id, requested_by, event)
            except BaseException as error:
                # An external task/thread cancellation must not leave a
                # claimed job looking completed or running forever.
                status = JOB_CANCELLED if self._stop_requested(normalized_job_id, requested_by, event) else JOB_FAILED
                self._finish_safely(
                    normalized_job_id,
                    requested_by,
                    status=status,
                    error=str(error),
                )
                logger.exception("Batch worker aborted: job=%s status=%s", normalized_job_id, status)
                raise
            finally:
                with self._events_lock:
                    self._job_events.pop(normalized_job_id, None)

    def _run_claimed_job(
        self,
        job_id: str,
        requested_by: int,
        event: threading.Event,
    ) -> dict[str, Any]:
        request = self._manager.get_request_for_user(job_id, requested_by)
        specs = self._manager.get_batch_specs_for_user(job_id, requested_by)
        if not specs:
            return self._finish_safely(
                job_id,
                requested_by,
                status=JOB_FAILED,
                error="Batchに実行対象画像がありません。",
            )

        completed = 0
        failed = 0
        image_serial = 0
        last_error = ""
        for spec in specs:
            batch_completed = 0
            batch_failed = 0
            batch_error = ""
            for image_index in range(1, spec.requested_count + 1):
                if self._stop_requested(job_id, requested_by, event):
                    return self._finish_safely(
                        job_id,
                        requested_by,
                        status=JOB_CANCELLED,
                        error=last_error,
                    )
                if not self._wait_for_generation_slot(job_id, requested_by, event):
                    return self._finish_safely(
                        job_id,
                        requested_by,
                        status=JOB_CANCELLED,
                        error=last_error,
                    )

                attempted_serial = image_serial
                busy_retry = False
                try:
                    self._generate_one(
                        request,
                        spec,
                        image_index=image_index,
                        image_serial=attempted_serial,
                    )
                except Exception as error:
                    if self._is_generation_lock_error(error):
                        # The shared sd_control lock can be acquired by a
                        # normal generation between the pre-check and call.
                        busy_retry = True
                        if not self._wait_for_generation_slot(job_id, requested_by, event):
                            return self._finish_safely(
                                job_id,
                                requested_by,
                                status=JOB_CANCELLED,
                                error=last_error,
                            )
                        continue
                    failed += 1
                    batch_failed += 1
                    batch_error = str(error) or "Batch画像の生成に失敗しました。"
                    last_error = batch_error
                    logger.warning(
                        "Batch image failed; continuing: job=%s batch=%s image=%s error=%s",
                        job_id,
                        spec.batch_id,
                        image_index,
                        batch_error,
                    )
                else:
                    completed += 1
                    batch_completed += 1
                finally:
                    # Busy-lock retries do not reach this branch as a new
                    # image attempt; successful or failed image attempts do.
                    if not busy_retry:
                        image_serial += 1
                        self._persist_progress(
                            job_id,
                            requested_by,
                            spec,
                            completed=completed,
                            failed=failed,
                            batch_completed=batch_completed,
                            batch_failed=batch_failed,
                            error=batch_error or last_error,
                        )

            if self._stop_requested(job_id, requested_by, event):
                return self._finish_safely(
                    job_id,
                    requested_by,
                    status=JOB_CANCELLED,
                    error=last_error,
                )

        final_status = JOB_FAILED if failed else JOB_COMPLETED
        return self._finish_safely(
            job_id,
            requested_by,
            status=final_status,
            error=last_error,
        )

    def _persist_progress(
        self,
        job_id: str,
        requested_by: int,
        spec: BatchSpec,
        *,
        completed: int,
        failed: int,
        batch_completed: int,
        batch_failed: int,
        error: str,
    ) -> None:
        # The worker owns one ordered image at a time, so these are the
        # complete counters for this batch at the moment of the update.
        self._manager.update_batch_progress(
            spec.batch_id,
            requested_by,
            completed_count=batch_completed,
            failed_count=batch_failed,
            error=error,
        )
        self._manager.update_progress(
            job_id,
            requested_by,
            completed_count=completed,
            failed_count=failed,
            current_batch_id=spec.batch_id,
            current_prompt=spec.prompt,
            error=error,
        )

    def _wait_for_generation_slot(
        self,
        job_id: str,
        requested_by: int,
        event: threading.Event,
    ) -> bool:
        while self._generation_lock_is_active():
            if self._stop_requested(job_id, requested_by, event):
                return False
            self._sleep(self._lock_poll_seconds)
        return not self._stop_requested(job_id, requested_by, event)

    def _stop_requested(self, job_id: str, requested_by: int, event: threading.Event) -> bool:
        if event.is_set() or self._shutdown_event.is_set():
            return True
        row = self._manager.get_job_for_user(job_id, requested_by)
        return bool(row.get("cancel_requested"))

    @staticmethod
    def _is_generation_lock_error(error: BaseException) -> bool:
        values = [str(error)]
        cause = getattr(error, "cause", None)
        if cause is not None:
            values.append(str(cause))
        text = " ".join(values)
        return "すでに画像生成中" in text or "generation already running" in text.lower()

    def _finish_safely(
        self,
        job_id: str,
        requested_by: int,
        *,
        status: str,
        error: str = "",
    ) -> dict[str, Any]:
        try:
            return self._manager.finish_job(
                job_id,
                requested_by,
                status=status,
                error=error,
            )
        except Exception as finish_error:
            raise BatchWorkerError(
                f"Batch jobの最終状態保存に失敗しました: {finish_error}"
            ) from finish_error


def run_batch_job(
    job_id: str,
    requested_by: int,
    **worker_options: Any,
) -> dict[str, Any]:
    """Small service entry point for tests and a future lifecycle hook."""

    return BatchGenerationWorker(**worker_options).run_job(job_id, requested_by)
