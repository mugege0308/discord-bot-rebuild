import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services.gallery_review_queue import GalleryReviewQueue


class GalleryReviewQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_reuses_gallery_scorer_and_prefers_cpu(self):
        queue = GalleryReviewQueue()
        try:
            with patch(
                "services.gallery_review_queue.is_generation_running",
                return_value=False,
            ), patch(
                "services.gallery_review_queue.get_gallery_status",
                return_value=SimpleNamespace(running=True, port=8765, pid=1234),
            ), patch(
                "services.gallery_review_queue.REVIEW_POLL_SECONDS",
                0,
            ), patch(
                "services.gallery_review_queue.get_gallery_job_summary",
                side_effect=[
                    {"total": 2, "ai_scored": 0},
                    {"total": 2, "ai_scored": 2},
                ],
            ), patch(
                "services.gallery_review_queue.get_gallery_ai_status",
                side_effect=[
                    {"status": "idle"},
                    {"job_id": "review-1", "status": "completed"},
                ],
            ), patch(
                "services.gallery_review_queue.start_gallery_review",
                return_value=SimpleNamespace(
                    job={"job_id": "review-1", "target_job_id": "target-1"}
                ),
            ) as start_review:
                await queue.start()
                future = queue.enqueue("target-1", 2)
                result = await asyncio.wait_for(future, timeout=2)

            self.assertEqual("completed", result["status"])
            start_review.assert_called_once_with(job_id="target-1", device="cpu")
        finally:
            await queue.shutdown()

    async def test_stopped_gallery_is_started_before_ingest_check(self):
        queue = GalleryReviewQueue()
        try:
            with patch(
                "services.gallery_review_queue.is_generation_running",
                return_value=False,
            ), patch(
                "services.gallery_review_queue.get_gallery_status",
                return_value=SimpleNamespace(running=False, port=8765, pid=None),
            ), patch(
                "services.gallery_review_queue.start_gallery",
                return_value=SimpleNamespace(
                    status=SimpleNamespace(running=True, port=8765, pid=4321)
                ),
            ) as start_gallery, patch(
                "services.gallery_review_queue.get_gallery_job_summary",
                side_effect=[
                    {"total": 1, "ai_scored": 0},
                    {"total": 1, "ai_scored": 1},
                ],
            ), patch(
                "services.gallery_review_queue.get_gallery_ai_status",
                side_effect=[
                    {"status": "idle"},
                    {"job_id": "review-1", "status": "completed"},
                ],
            ), patch(
                "services.gallery_review_queue.REVIEW_POLL_SECONDS",
                0,
            ), patch(
                "services.gallery_review_queue.start_gallery_review",
                return_value=SimpleNamespace(
                    job={"job_id": "review-1", "target_job_id": "target-1"}
                ),
            ):
                future = queue.enqueue("target-1", 1)
                result = await asyncio.wait_for(future, timeout=2)

            self.assertEqual("completed", result["status"])
            start_gallery.assert_called_once_with()
        finally:
            await queue.shutdown()

    async def test_gallery_start_failure_is_best_effort_and_finishes(self):
        queue = GalleryReviewQueue()
        try:
            with patch(
                "services.gallery_review_queue.get_gallery_status",
                return_value=SimpleNamespace(running=False, port=8765, pid=None),
            ), patch(
                "services.gallery_review_queue.start_gallery",
                side_effect=RuntimeError("gallery unavailable"),
            ):
                future = queue.enqueue("target-1", 1)
                result = await asyncio.wait_for(future, timeout=2)

            self.assertEqual("failed", result["status"])
            self.assertIn("起動できませんでした", result["error"])
        finally:
            await queue.shutdown()

    async def test_duplicate_target_returns_same_future(self):
        queue = GalleryReviewQueue()
        try:
            first = queue.enqueue("same-target", 1)
            second = queue.enqueue("same-target", 1)
            self.assertIs(first, second)
        finally:
            await queue.shutdown()
