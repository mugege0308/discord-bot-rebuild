import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from commands import generate_batch


def _job(*, auto_review=True, completed=2, status="completed"):
    return {
        "job_id": "job-1",
        "status": status,
        "requested_by": 123,
        "total_count": 3,
        "completed_count": completed,
        "failed_count": 1 if completed else 3,
        "auto_review_enabled": auto_review,
        "review_status": "not_requested",
        "review_job_id": "",
        "review_error": "",
        "current_batch_id": "",
        "error": "",
    }


class BatchGalleryReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_auto_review_updates_review_only_and_preserves_generation_state(self):
        future = asyncio.get_running_loop().create_future()
        future.set_result({"status": "completed", "review_job_id": "review-1"})
        with (
            patch.object(generate_batch.manager, "get_job_for_user", return_value=_job()),
            patch.object(generate_batch.manager, "update_review_status") as update,
            patch.object(generate_batch, "enqueue_gallery_review", return_value=future) as enqueue,
            patch.object(generate_batch.manager, "finish_job") as finish,
        ):
            await generate_batch._post_generation_review("job-1", 123)

        enqueue.assert_called_once()
        self.assertEqual(2, enqueue.call_args.args[1])
        self.assertEqual(["queued", "completed"], [call.kwargs["status"] for call in update.call_args_list])
        finish.assert_not_called()

    async def test_gallery_review_failure_does_not_fail_batch_job(self):
        future = asyncio.get_running_loop().create_future()
        future.set_result({"status": "failed", "error": "Gallery stopped"})
        with (
            patch.object(generate_batch.manager, "get_job_for_user", return_value=_job()),
            patch.object(generate_batch.manager, "update_review_status") as update,
            patch.object(generate_batch, "enqueue_gallery_review", return_value=future),
            patch.object(generate_batch.manager, "finish_job") as finish,
        ):
            await generate_batch._post_generation_review("job-1", 123)

        self.assertEqual("failed", update.call_args.kwargs["status"])
        self.assertEqual("Gallery stopped", update.call_args.kwargs["error"])
        finish.assert_not_called()

    async def test_review_disabled_does_not_enqueue(self):
        with (
            patch.object(generate_batch.manager, "get_job_for_user", return_value=_job(auto_review=False)),
            patch.object(generate_batch, "enqueue_gallery_review") as enqueue,
        ):
            await generate_batch._post_generation_review("job-1", 123)
        enqueue.assert_not_called()


class BatchProgressMessageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        generate_batch._shutting_down = False
        generate_batch._gallery_followup_tasks.clear()

    async def asyncTearDown(self):
        tasks = list(generate_batch._gallery_followup_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        generate_batch._gallery_followup_tasks.clear()

    async def test_terminal_watcher_publishes_gallery_summary_and_link(self):
        interaction = SimpleNamespace(user=SimpleNamespace(id=123), followup=SimpleNamespace(send=AsyncMock()))
        message = Mock()
        status = SimpleNamespace(
            urls=SimpleNamespace(local="http://127.0.0.1:8765/", tailscale=()),
        )
        with (
            patch.object(generate_batch, "PROGRESS_POLL_SECONDS", 0),
            patch.object(
                generate_batch.manager,
                "get_job_for_user",
                return_value=_job(auto_review=False),
            ),
            patch.object(
                generate_batch,
                "_terminal_gallery_context",
                new=AsyncMock(return_value=(status, {"total": 2, "ai_scored": 0}, "")),
            ),
            patch.object(generate_batch, "_publish_progress", new=AsyncMock(return_value=message)) as publish,
        ):
            await generate_batch._watch_job(interaction, "job-1", message)

            followup = generate_batch._gallery_followup_tasks["job-1"]
            await followup

        publish.assert_awaited_once()
        self.assertNotIn("Gallery: 2枚", publish.call_args.args[2])
        interaction.followup.send.assert_awaited_once()
        self.assertIn("Gallery: 2枚", interaction.followup.send.call_args.args[0])
        self.assertIsNotNone(interaction.followup.send.call_args.kwargs["view"])

    async def test_completion_is_published_before_delayed_gallery_summary(self):
        interaction = SimpleNamespace(user=SimpleNamespace(id=123), followup=SimpleNamespace(send=AsyncMock()))
        message = Mock()
        release_gallery = asyncio.Event()

        async def delayed_gallery(*args, **kwargs):
            await release_gallery.wait()
            return None, {"total": 2, "ai_scored": 0}, ""

        with (
            patch.object(generate_batch, "PROGRESS_POLL_SECONDS", 0),
            patch.object(generate_batch.manager, "get_job_for_user", return_value=_job(auto_review=False)),
            patch.object(generate_batch, "_terminal_gallery_context", side_effect=delayed_gallery),
            patch.object(generate_batch, "_publish_progress", new=AsyncMock(return_value=message)) as publish,
        ):
            await generate_batch._watch_job(interaction, "job-1", message)
            publish.assert_awaited_once()
            self.assertIn("状態: **completed**", publish.call_args.args[2])
            self.assertNotIn("Gallery:", publish.call_args.args[2])
            self.assertFalse(interaction.followup.send.await_count)

            release_gallery.set()
            await generate_batch._gallery_followup_tasks["job-1"]

        interaction.followup.send.assert_awaited_once()
        self.assertIn("Gallery: 2枚", interaction.followup.send.call_args.args[0])

    async def test_gallery_timeout_does_not_change_completion_or_send_retry(self):
        interaction = SimpleNamespace(user=SimpleNamespace(id=123), followup=SimpleNamespace(send=AsyncMock()))
        message = Mock()
        with (
            patch.object(generate_batch, "PROGRESS_POLL_SECONDS", 0),
            patch.object(generate_batch.manager, "get_job_for_user", return_value=_job(auto_review=False)),
            patch.object(
                generate_batch,
                "_terminal_gallery_context",
                new=AsyncMock(return_value=(None, None, "summary timeout")),
            ),
            patch.object(generate_batch, "_publish_progress", new=AsyncMock(return_value=message)) as publish,
        ):
            await generate_batch._watch_job(interaction, "job-1", message)
            await generate_batch._gallery_followup_tasks["job-1"]

        publish.assert_awaited_once()
        self.assertIn("状態: **completed**", publish.call_args.args[2])
        interaction.followup.send.assert_not_awaited()

    async def test_gallery_followup_notification_failure_is_best_effort(self):
        interaction = SimpleNamespace(
            user=SimpleNamespace(id=123),
            followup=SimpleNamespace(send=AsyncMock(side_effect=RuntimeError("Discord unavailable"))),
        )
        message = Mock()
        status = SimpleNamespace(
            urls=SimpleNamespace(local="http://127.0.0.1:8765/", tailscale=()),
        )
        with (
            patch.object(generate_batch.manager, "get_job_for_user", return_value=_job(auto_review=False)),
            patch.object(
                generate_batch,
                "_terminal_gallery_context",
                new=AsyncMock(return_value=(status, {"total": 2, "ai_scored": 0}, "")),
            ),
        ):
            generate_batch._schedule_gallery_followup(interaction, "job-1")
            await generate_batch._gallery_followup_tasks["job-1"]

        interaction.followup.send.assert_awaited_once()

    async def test_shutdown_cancels_pending_gallery_followup(self):
        interaction = SimpleNamespace(user=SimpleNamespace(id=123), followup=SimpleNamespace(send=AsyncMock()))
        release_gallery = asyncio.Event()

        async def blocked_gallery(*args, **kwargs):
            await release_gallery.wait()
            return None, {"total": 1}, ""

        with (
            patch.object(generate_batch.manager, "get_job_for_user", return_value=_job(auto_review=False)),
            patch.object(generate_batch, "_terminal_gallery_context", side_effect=blocked_gallery),
            patch.object(generate_batch._worker, "shutdown"),
            patch.object(generate_batch, "shutdown_gallery_review_queue", new=AsyncMock()),
        ):
            generate_batch._schedule_gallery_followup(interaction, "job-1")
            await asyncio.sleep(0)
            await generate_batch.shutdown()

        self.assertFalse(generate_batch._gallery_followup_tasks)


if __name__ == "__main__":
    unittest.main()
