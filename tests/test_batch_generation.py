import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.batch_generation import (
    BatchGenerationRequest,
    BatchJobOwnershipError,
    BatchJobNotFoundError,
    JOB_CANCELLED,
    JOB_FAILED,
    JOB_RUNNING,
    MAX_IMAGES_PER_PROMPT,
    MAX_JOB_COUNT,
    build_variation_patterns,
    manager,
)
from services.generate_ai_last_settings import (
    get_last_batch_settings,
    save_last_batch_settings,
)


class BatchGenerationLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "batch.db"
        self.environment = patch.dict(
            os.environ,
            {"BATCH_GENERATION_DB_PATH": str(self.db_path)},
            clear=False,
        )
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    @staticmethod
    def request(**overrides):
        values = {
            "requested_by": 123,
            "character": "test-character",
            "total_count": 6,
            "images_per_prompt": 1,
            "model_key": "animagine",
            "variation_enabled": True,
        }
        values.update(overrides)
        return BatchGenerationRequest(**values)

    def test_create_get_progress_finish_and_batch_state(self):
        job_id = manager.create_job(self.request(total_count=2, variation_count=2))
        queued = manager.get_job_for_user(job_id, 123)
        self.assertEqual("queued", queued["status"])
        self.assertEqual(0, queued["completed_count"])

        manager.mark_running(job_id, 123)
        batch_id = manager.create_batch(
            job_id,
            123,
            prompt="prompt",
            requested_count=2,
            variation_index=1,
        )
        manager.update_batch_progress(batch_id, 123, completed_count=1)
        running = manager.update_progress(
            job_id,
            123,
            completed_count=1,
            failed_count=0,
            current_batch_id=batch_id,
            current_prompt="prompt",
        )
        self.assertEqual(JOB_RUNNING, running["status"])
        self.assertEqual(1, running["completed_count"])
        self.assertEqual(1, running["batches"][0]["completed_count"])

        completed = manager.finish_job(job_id, 123, status="completed")
        self.assertEqual("completed", completed["status"])
        self.assertEqual("", completed["current_batch_id"])

    def test_owner_separation_applies_to_get_cancel_and_repeat(self):
        job_id = manager.create_job(self.request())
        with self.assertRaises(BatchJobOwnershipError):
            manager.get_job_for_user(job_id, 999)
        with self.assertRaises(BatchJobOwnershipError):
            manager.cancel_job(job_id, 999)
        with self.assertRaises(BatchJobOwnershipError):
            manager.repeat_job(job_id, 999)

        with self.assertRaises(BatchJobNotFoundError):
            manager.get_job_for_user("missing", 123)

    def test_cancel_queued_job_is_terminal_and_non_destructive(self):
        job_id = manager.create_job(self.request(total_count=2))
        accepted, message = manager.cancel_job(job_id, 123)
        self.assertTrue(accepted)
        self.assertIn("キュー待ち", message)
        cancelled = manager.get_job_for_user(job_id, 123)
        self.assertEqual(JOB_CANCELLED, cancelled["status"])
        self.assertTrue(cancelled["cancel_requested"])
        self.assertEqual(0, cancelled["completed_count"])

    def test_running_cancel_sets_flag_without_claiming_current_image_finished(self):
        job_id = manager.create_job(self.request(total_count=2))
        manager.mark_running(job_id, 123)
        accepted, _ = manager.cancel_job(job_id, 123)
        self.assertTrue(accepted)
        running = manager.get_job_for_user(job_id, 123)
        self.assertEqual(JOB_RUNNING, running["status"])
        self.assertTrue(running["cancel_requested"])
        self.assertEqual(0, running["completed_count"])

    def test_repeat_copies_request_and_normalizes_seed(self):
        source_id = manager.create_job(
            self.request(
                total_count=4,
                images_per_prompt=2,
                images_per_variation=2,
                variation_count=2,
                base_prompt="fixed prompt",
                base_negative_prompt="fixed negative",
                generation_config={"width": 768, "height": 1024, "seed": 1234},
            )
        )
        manager.finish_job(source_id, 123, status="completed")
        repeated_id = manager.repeat_job(source_id, 123)
        repeated = manager.get_request_for_user(repeated_id, 123)
        self.assertNotEqual(source_id, repeated_id)
        self.assertEqual("fixed prompt", repeated.base_prompt)
        self.assertEqual("fixed negative", repeated.base_negative_prompt)
        self.assertEqual(2, repeated.effective_variation_count)
        self.assertEqual(2, repeated.effective_images_per_variation)
        self.assertEqual(-1, repeated.generation_config["seed"])
        self.assertEqual(source_id, repeated.repeat_of_job_id)

    def test_restart_recovery_fails_queued_and_running_only(self):
        queued_id = manager.create_job(self.request())
        running_id = manager.create_job(self.request())
        manager.mark_running(running_id, 123)
        completed_id = manager.create_job(self.request())
        manager.finish_job(completed_id, 123, status="completed")

        recovered = manager.recover_incomplete_jobs()
        self.assertEqual(2, recovered)
        self.assertEqual(JOB_FAILED, manager.get_job_for_user(queued_id, 123)["status"])
        self.assertEqual(JOB_FAILED, manager.get_job_for_user(running_id, 123)["status"])
        self.assertEqual("completed", manager.get_job_for_user(completed_id, 123)["status"])
        self.assertIn("Bot再起動時", manager.get_job_for_user(queued_id, 123)["error"])

    def test_limits_and_progress_validation(self):
        with self.assertRaises(ValueError):
            manager.create_job(self.request(total_count=MAX_JOB_COUNT + 1))
        with self.assertRaises(ValueError):
            manager.create_job(self.request(images_per_prompt=MAX_IMAGES_PER_PROMPT + 1))
        job_id = manager.create_job(self.request(total_count=2))
        with self.assertRaises(ValueError):
            manager.update_progress(job_id, 123, completed_count=2, failed_count=1)

    def test_variation_patterns_are_deterministic_and_bounded(self):
        first = build_variation_patterns(10)
        self.assertEqual(first, build_variation_patterns(10))
        self.assertEqual(10, len(first))
        self.assertEqual((), build_variation_patterns(0))

    def test_database_failure_does_not_leave_a_partial_job(self):
        broken_path = Path(self.temporary.name) / "not-a-database-directory"
        broken_path.mkdir()
        with patch.dict(os.environ, {"BATCH_GENERATION_DB_PATH": str(broken_path)}):
            with self.assertRaises(Exception):
                manager.create_job(self.request())
        self.assertFalse(self.db_path.exists())


class BatchSettingsRegressionTests(unittest.TestCase):
    def test_existing_batch_settings_api_remains_usable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "settings.json"
            save_last_batch_settings(123, {"model_key": "animagine", "loras": []}, path)
            self.assertEqual(
                {"model_key": "animagine", "loras": []},
                {
                    key: value
                    for key, value in get_last_batch_settings(123, path).items()
                    if key != "updated_at"
                },
            )


if __name__ == "__main__":
    unittest.main()
