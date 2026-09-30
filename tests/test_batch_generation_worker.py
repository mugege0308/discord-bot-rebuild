import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

from services.batch_generation import (
    BatchGenerationManager,
    BatchGenerationRequest,
    JOB_CANCELLED,
    JOB_COMPLETED,
    JOB_FAILED,
)
from services.batch_generation_worker import BatchGenerationWorker


class BatchGenerationWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "batch.db"
        self.environment = unittest.mock.patch.dict(
            os.environ,
            {"BATCH_GENERATION_DB_PATH": str(self.db_path)},
            clear=False,
        )
        self.environment.start()
        self.manager = BatchGenerationManager()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def create_job(self, *, total_count=3, variation_enabled=True):
        request = BatchGenerationRequest(
            requested_by=7,
            character="worker-character",
            total_count=total_count,
            images_per_prompt=1,
            variation_enabled=variation_enabled,
            variation_count=total_count if variation_enabled else None,
            base_prompt="base prompt",
            rating="safe",
        )
        return self.manager.create_job(request)

    def add_batch(self, job_id, *, batch_id, variation_index, requested_count=1):
        return self.manager.create_batch(
            job_id,
            7,
            batch_id=batch_id,
            variation_index=variation_index,
            prompt=f"prompt {variation_index}",
            negative_prompt="negative",
            requested_count=requested_count,
        )

    def worker(self, generate_one, **kwargs):
        return BatchGenerationWorker(
            manager=self.manager,
            generate_one=generate_one,
            generation_lock_is_active=kwargs.pop("generation_lock_is_active", lambda: False),
            sleep=kwargs.pop("sleep", lambda _seconds: None),
            lock_poll_seconds=0,
            **kwargs,
        )

    def test_variation_and_image_order_and_progress_are_persisted(self):
        job_id = self.create_job(total_count=3)
        # Insert out of order; persisted variation_index controls execution.
        self.add_batch(job_id, batch_id="batch-2", variation_index=2, requested_count=2)
        self.add_batch(job_id, batch_id="batch-1", variation_index=1, requested_count=1)
        calls = []

        def generate(_request, spec, *, image_index, image_serial):
            calls.append((spec.variation_index, image_index, image_serial))
            return object()

        result = self.worker(generate).run_job(job_id, 7)
        self.assertEqual(JOB_COMPLETED, result["status"])
        self.assertEqual([(1, 1, 0), (2, 1, 1), (2, 2, 2)], calls)
        self.assertEqual(3, result["completed_count"])
        self.assertEqual(0, result["failed_count"])
        self.assertEqual(
            [("batch-1", 1, 0), ("batch-2", 2, 0)],
            [
                (batch["batch_id"], batch["completed_count"], batch["failed_count"])
                for batch in result["batches"]
            ],
        )

    def test_one_failure_continues_and_partial_job_is_failed(self):
        job_id = self.create_job(total_count=3, variation_enabled=False)
        self.add_batch(job_id, batch_id="batch-1", variation_index=1, requested_count=3)
        calls = []

        def generate(_request, _spec, *, image_index, image_serial):
            calls.append((image_index, image_serial))
            if image_index == 2:
                raise RuntimeError("one image failed")
            return object()

        result = self.worker(generate).run_job(job_id, 7)
        self.assertEqual(JOB_FAILED, result["status"])
        self.assertEqual([(1, 0), (2, 1), (3, 2)], calls)
        self.assertEqual((2, 1), (result["completed_count"], result["failed_count"]))
        batch = result["batches"][0]
        self.assertEqual((2, 1), (batch["completed_count"], batch["failed_count"]))
        self.assertIn("one image failed", result["error"])

    def test_all_failures_are_persisted_as_failed(self):
        job_id = self.create_job(total_count=2, variation_enabled=False)
        self.add_batch(job_id, batch_id="batch-1", variation_index=1, requested_count=2)
        result = self.worker(Mock(side_effect=RuntimeError("ComfyUI down"))).run_job(job_id, 7)
        self.assertEqual(JOB_FAILED, result["status"])
        self.assertEqual((0, 2), (result["completed_count"], result["failed_count"]))
        self.assertEqual((0, 2), (result["batches"][0]["completed_count"], result["batches"][0]["failed_count"]))

    def test_cancel_between_images_finishes_current_and_stops_remaining(self):
        job_id = self.create_job(total_count=3, variation_enabled=False)
        self.add_batch(job_id, batch_id="batch-1", variation_index=1, requested_count=3)
        calls = []

        def generate(_request, _spec, *, image_index, image_serial):
            calls.append((image_index, image_serial))
            if image_index == 1:
                accepted, _ = self.manager.cancel_job(job_id, 7)
                self.assertTrue(accepted)
            return object()

        result = self.worker(generate).run_job(job_id, 7)
        self.assertEqual(JOB_CANCELLED, result["status"])
        self.assertEqual([(1, 0)], calls)
        self.assertEqual((1, 0), (result["completed_count"], result["failed_count"]))

    def test_generation_lock_is_waited_on_without_parallel_generation(self):
        job_id = self.create_job(total_count=1, variation_enabled=False)
        self.add_batch(job_id, batch_id="batch-1", variation_index=1)
        states = iter((True, False))
        waits = []

        def lock_is_active():
            return next(states, False)

        result = self.worker(
            Mock(return_value=object()),
            generation_lock_is_active=lock_is_active,
            sleep=lambda seconds: waits.append(seconds),
        ).run_job(job_id, 7)
        self.assertEqual(JOB_COMPLETED, result["status"])
        self.assertEqual([0], waits)

    def test_multiple_registered_jobs_run_serially_in_supplied_order(self):
        first_job = self.create_job(total_count=1, variation_enabled=False)
        second_job = self.create_job(total_count=1, variation_enabled=False)
        self.add_batch(first_job, batch_id="batch-1", variation_index=1)
        self.add_batch(second_job, batch_id="batch-2", variation_index=1)
        calls = []

        def generate(_request, spec, **_kwargs):
            calls.append(spec.batch_id)
            return object()

        results = self.worker(generate).run_jobs([first_job, second_job], 7)
        self.assertEqual(["batch-1", "batch-2"], calls)
        self.assertEqual([JOB_COMPLETED, JOB_COMPLETED], [row["status"] for row in results])

    def test_duplicate_start_does_not_generate_twice(self):
        job_id = self.create_job(total_count=1, variation_enabled=False)
        self.add_batch(job_id, batch_id="batch-1", variation_index=1)
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def generate(*_args, **_kwargs):
            calls.append(1)
            entered.set()
            self.assertTrue(release.wait(2))
            return object()

        first_worker = self.worker(generate)
        second_worker = self.worker(generate)
        first_result = []
        thread = threading.Thread(
            target=lambda: first_result.append(first_worker.run_job(job_id, 7)),
            daemon=True,
        )
        thread.start()
        self.assertTrue(entered.wait(2))
        second_snapshot = second_worker.run_job(job_id, 7)
        release.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(1, len(calls))
        self.assertEqual("running", second_snapshot["status"])
        self.assertEqual(JOB_COMPLETED, first_result[0]["status"])

    def test_shutdown_after_current_image_leaves_cancelled_state(self):
        job_id = self.create_job(total_count=2, variation_enabled=False)
        self.add_batch(job_id, batch_id="batch-1", variation_index=1, requested_count=2)
        worker = None

        def generate(*_args, **_kwargs):
            worker.shutdown()
            return object()

        worker = self.worker(generate)
        result = worker.run_job(job_id, 7)
        self.assertEqual(JOB_CANCELLED, result["status"])
        self.assertEqual(1, result["completed_count"])


if __name__ == "__main__":
    unittest.main()
