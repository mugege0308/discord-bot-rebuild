"""Resident, generation-friendly queue for the gallery's incremental AI review.

The gallery owns the actual scorer and its SQLite writes.  The bot only
serialises review requests, waits until generation is idle, and reports the
result back to a batch job when one exists.  This keeps /generate_ai and
/generate_batch on the same review path without loading an image model in the
Discord command coroutine.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

from services.gallery_control import (
    GalleryControlError,
    get_gallery_ai_status,
    get_gallery_job_summary,
    get_gallery_status,
    start_gallery,
    start_gallery_review,
)
from services.sd_control import is_generation_running


logger = logging.getLogger("secretary-bot.gallery_review_queue")

REVIEW_INGEST_WAIT_SECONDS = 180
REVIEW_POLL_SECONDS = 5
REVIEW_POLL_LIMIT = 720
GENERATION_IDLE_POLL_SECONDS = 1
REVIEW_INGEST_BACKOFF_INITIAL_SECONDS = 1
REVIEW_INGEST_BACKOFF_MAX_SECONDS = 30

ReviewStateCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class _ReviewRequest:
    target_id: str
    expected: int
    future: asyncio.Future[dict[str, Any]]
    on_state: ReviewStateCallback | None = None


class GalleryUnavailableError(RuntimeError):
    """Expected best-effort failure which should not produce a traceback."""


class GalleryReviewQueue:
    """One background worker that never competes with an active generation."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[_ReviewRequest] = asyncio.Queue()
        self._pending: dict[str, _ReviewRequest] = {}
        self._worker_task: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        self._stopping = False
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        if self._stopping:
            return
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(
                self._worker(), name="gallery-review-queue"
            )

    def enqueue(
        self,
        target_id: str,
        expected: int,
        *,
        on_state: ReviewStateCallback | None = None,
    ) -> asyncio.Future[dict[str, Any]]:
        """Queue one gallery job scope, de-duplicating repeated submissions."""

        target = target_id.strip()
        if not target:
            raise ValueError("AI評価対象のjob_idが空です。")
        if expected <= 0:
            raise ValueError("AI評価対象の画像がありません。")
        existing = self._pending.get(target)
        if existing is not None:
            return existing.future

        loop = asyncio.get_running_loop()
        request = _ReviewRequest(
            target_id=target,
            expected=expected,
            future=loop.create_future(),
            on_state=on_state,
        )
        self._pending[target] = request
        self._ensure_worker()
        self._queue.put_nowait(request)
        self._emit(request, "queued")
        return request.future

    @staticmethod
    def _emit(request: _ReviewRequest, state: str, **details: Any) -> None:
        if request.on_state is None:
            return
        try:
            request.on_state(state, details)
        except Exception:
            logger.exception(
                "Gallery review state callback failed: target=%s state=%s",
                request.target_id,
                state,
            )

    async def _wait_for_generation_idle(self) -> None:
        while is_generation_running():
            await asyncio.sleep(GENERATION_IDLE_POLL_SECONDS)

    async def _ensure_gallery_running(self, request: _ReviewRequest) -> None:
        try:
            status = await asyncio.to_thread(get_gallery_status)
        except Exception as exc:
            raise GalleryUnavailableError(
                f"ギャラリー状態を確認できませんでした: {exc}"
            ) from exc
        if status.running:
            return

        logger.info("Gallery is stopped; starting it for review: target=%s", request.target_id)
        try:
            started = await asyncio.to_thread(start_gallery)
        except Exception as exc:
            raise GalleryUnavailableError(
                f"ギャラリーを起動できませんでした: {exc}"
            ) from exc
        if not started.status.running:
            raise GalleryUnavailableError("ギャラリー起動後もAPIがreadyになりませんでした。")
        logger.info(
            "Gallery started for review: target=%s port=%s pid=%s",
            request.target_id,
            started.status.port,
            started.status.pid or "-",
        )

    async def _wait_for_ingest(self, request: _ReviewRequest) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + REVIEW_INGEST_WAIT_SECONDS
        last: dict[str, Any] = {}
        last_error: Exception | None = None
        attempt = 0
        delay = REVIEW_INGEST_BACKOFF_INITIAL_SECONDS
        while asyncio.get_running_loop().time() < deadline:
            await self._wait_for_generation_idle()
            attempt += 1
            try:
                last = await asyncio.to_thread(get_gallery_job_summary, request.target_id)
                if int(last.get("total") or 0) >= request.expected:
                    return last
            except Exception as exc:
                last_error = exc
                if attempt == 1:
                    logger.warning(
                        "Gallery ingest check failed: target=%s attempt=%s",
                        request.target_id,
                        attempt,
                        exc_info=True,
                    )
                else:
                    logger.warning(
                        "Gallery ingest check failed; retrying: target=%s attempt=%s next_wait=%ss",
                        request.target_id,
                        attempt,
                        min(delay, REVIEW_INGEST_BACKOFF_MAX_SECONDS),
                    )
            wait_seconds = min(delay, REVIEW_INGEST_BACKOFF_MAX_SECONDS)
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(wait_seconds, remaining))
            delay = min(delay * 2, REVIEW_INGEST_BACKOFF_MAX_SECONDS)
        if last_error is not None:
            logger.error(
                "Gallery ingest retry limit reached: target=%s attempts=%s actual=%s",
                request.target_id,
                attempt,
                last.get("total", 0),
                exc_info=(type(last_error), last_error, last_error.__traceback__),
            )
        raise RuntimeError(
            "ギャラリーへの取り込みがタイムアウトしました。"
            f" job={request.target_id} expected={request.expected}"
            f" actual={last.get('total', 0)}"
        )

    async def _process(self, request: _ReviewRequest) -> dict[str, Any]:
        self._emit(request, "starting")
        await self._ensure_gallery_running(request)
        summary = await self._wait_for_ingest(request)
        del summary

        for _ in range(REVIEW_POLL_LIMIT):
            await self._wait_for_generation_idle()
            current = await asyncio.to_thread(get_gallery_ai_status)
            current_state = str(current.get("status") or "idle")
            if current_state == "running":
                await asyncio.sleep(REVIEW_POLL_SECONDS)
                continue

            # Run the existing gallery scorer on CPU by default.  This keeps
            # ComfyUI's RTX3060 workload as the priority while still reusing
            # the gallery's CLIP/fallback/cache implementation.
            started = await asyncio.to_thread(
                start_gallery_review,
                job_id=request.target_id,
                device="cpu",
            )
            review_job_id = str(started.job.get("job_id") or "")
            target_job_id = str(started.job.get("target_job_id") or request.target_id)
            if target_job_id != request.target_id:
                await asyncio.sleep(REVIEW_POLL_SECONDS)
                continue
            self._emit(request, "running", review_job_id=review_job_id)

            for _ in range(REVIEW_POLL_LIMIT):
                await asyncio.sleep(REVIEW_POLL_SECONDS)
                status = await asyncio.to_thread(get_gallery_ai_status)
                if review_job_id and str(status.get("job_id") or "") != review_job_id:
                    continue
                state = str(status.get("status") or "")
                if state == "completed":
                    fresh_summary = await asyncio.to_thread(
                        get_gallery_job_summary, request.target_id
                    )
                    scored = int(fresh_summary.get("ai_scored") or 0)
                    if scored >= request.expected:
                        result = {
                            "status": "completed",
                            "review_job_id": review_job_id,
                            **fresh_summary,
                        }
                        self._emit(request, "completed", **result)
                        return result
                    raise RuntimeError(
                        "AI評価完了通知後も評価結果が不足しています。"
                        f" {scored}/{request.expected}件"
                    )
                if state == "failed":
                    raise RuntimeError(
                        str(status.get("error") or "AI評価に失敗しました。")
                    )
            raise RuntimeError("AI評価の監視がタイムアウトしました。")
        raise RuntimeError("AI評価の開始待ちがタイムアウトしました。")

    async def _worker(self) -> None:
        while not self._stopping:
            request = await self._queue.get()
            try:
                result = await self._process(request)
            except asyncio.CancelledError:
                if not request.future.done():
                    request.future.cancel()
                raise
            except GalleryUnavailableError as error:
                logger.warning(
                    "Gallery review skipped: target=%s reason=%s",
                    request.target_id,
                    error,
                )
                self._emit(request, "failed", error=str(error))
                result = {"status": "failed", "error": str(error)}
            except (GalleryControlError, OSError, RuntimeError, ValueError) as error:
                logger.error(
                    "Gallery review failed: target=%s error=%s",
                    request.target_id,
                    error,
                )
                self._emit(request, "failed", error=str(error))
                result = {"status": "failed", "error": str(error)}
            except Exception as error:
                logger.exception("Unexpected gallery review failure: target=%s", request.target_id)
                self._emit(request, "failed", error=str(error))
                result = {"status": "failed", "error": str(error)}
            finally:
                self._pending.pop(request.target_id, None)
                if not request.future.done():
                    request.future.set_result(result)
                self._queue.task_done()

    async def shutdown(self) -> None:
        self._stopping = True
        task = self._worker_task
        self._worker_task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for request in self._pending.values():
            if not request.future.done():
                request.future.cancel()
        self._pending.clear()


_queue = GalleryReviewQueue()


async def start_gallery_review_queue() -> None:
    await _queue.start()


def enqueue_gallery_review(
    target_id: str,
    expected: int,
    *,
    on_state: ReviewStateCallback | None = None,
) -> asyncio.Future[dict[str, Any]]:
    return _queue.enqueue(target_id, expected, on_state=on_state)


async def shutdown_gallery_review_queue() -> None:
    await _queue.shutdown()
