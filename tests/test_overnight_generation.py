from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from aiohttp.test_utils import TestClient, TestServer

from services.generation_api import GenerationApi, create_generation_app
from services.overnight_generation import (
    OVERNIGHT_ROUND_SIZE,
    OVERNIGHT_TOTALS,
    OvernightJobStore,
    overnight_response,
)


class OvernightGenerationStoreTests(unittest.TestCase):
    def test_selection_identity_uses_overnight_job_and_global_index(self) -> None:
        rows = GenerationApi._selected_rows(
            [Path("score_prefix_image.png")],
            [{"image": {"filename": "image.png"}, "global_index": 37, "variation_index": 5}],
        )
        # Selector outputs are score-prefixed copies; the durable row keeps
        # the source filename and global slot identity.
        self.assertEqual("image.png", rows[0]["filename"])
        identity = GenerationApi._add_selection_identity(rows, "overnight-1")
        self.assertEqual("overnight-1", identity[0]["overnight_job_id"])
        self.assertEqual("overnight-1:37", identity[0]["selection_identity"])

    def test_selection_identity_handles_flattened_final_candidate_rows(self) -> None:
        rows = GenerationApi._selected_rows(
            [Path("score_prefix_image.png")],
            [{"filename": "image.png", "global_index": 38, "variation_index": 6}],
        )
        identity = GenerationApi._add_selection_identity(rows, "overnight-1")
        self.assertEqual("image.png", identity[0]["filename"])
        self.assertEqual(38, identity[0]["global_index"])
        self.assertEqual("overnight-1:38", identity[0]["selection_identity"])

    def test_selector_path_resolution_uses_project_local_output_without_paths(self) -> None:
        filename = "overnight-selector-test.png"
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "data" / "sd_outputs"
            output_root.mkdir(parents=True)
            (output_root / filename).write_bytes(b"test")
            with patch("services.generation_api.OUTPUT_DIR", output_root), patch(
                "services.generation_api.get_gallery_images_root", return_value=None
            ):
                resolved = GenerationApi._overnight_path_for_filename(filename)
                self.assertEqual(output_root / filename, resolved)
                self.assertIsNone(GenerationApi._overnight_path_for_filename("../" + filename))

    def test_supported_totals_create_rounds_and_persist_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for total in sorted(OVERNIGHT_TOTALS):
                store = OvernightJobStore(Path(temporary) / f"{total}.db")
                record = store.create({"model_key": "animagine", "seed": -1}, total)
                self.assertEqual(total // OVERNIGHT_ROUND_SIZE, record["total_rounds"])
                updated = store.update(
                    record["job_id"],
                    state="paused",
                    current_round=2,
                    current_slot=7,
                    timing={"round_1_started_at": "2026-10-07T00:00:00+00:00"},
                    in_flight={"global_index": 39, "seed": 12},
                )
                loaded = store.get(record["job_id"])
                self.assertEqual("paused", updated["state"])
                self.assertEqual(2, loaded["current_round"])
                self.assertEqual(7, loaded["current_slot"])
                self.assertEqual("2026-10-07T00:00:00+00:00", loaded["timing"]["round_1_started_at"])
                self.assertEqual(39, loaded["in_flight"]["global_index"])

    def test_only_one_active_job_and_invalid_total_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = OvernightJobStore(Path(temporary) / "overnight.db")
            first = store.create({}, 128)
            with self.assertRaisesRegex(RuntimeError, "同時に1件"):
                store.create({}, 256)
            store.update(first["job_id"], state="completed", finished_at="now")
            second = store.create({}, 512)
            self.assertEqual(512, second["total_count"])
            with self.assertRaisesRegex(RuntimeError, "128、256、512"):
                store.create({}, 9)

    def test_response_contains_progress_and_timing_but_no_request_snapshot(self) -> None:
        record = {
            "job_id": "overnight-safe-id",
            "state": "completed",
            "total_count": 128,
            "round_size": 32,
            "total_rounds": 4,
            "generated_count": 128,
            "failed_count": 0,
            "rounds": [],
            "final_selected": [{"filename": "best.png", "rank": 1}],
            "timing": {"final_selection_finished_at": "2026-10-07T01:00:00+00:00"},
            "request": {"reference_image_id": "opaque-id", "secret": "must-not-return"},
        }
        response = overnight_response(record)
        self.assertEqual(1.0, response["progress"])
        self.assertEqual("2026-10-07T01:00:00+00:00", response["timing"]["final_selection_finished_at"])
        self.assertNotIn("request", response)
        self.assertNotIn("opaque-id", str(response))
        self.assertNotIn("must-not-return", str(response))

    def test_shutdown_request_is_per_job_and_safe_response_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = OvernightJobStore(Path(temporary) / "overnight.db")
            requested = store.create({}, 128, shutdown_after_completion=True)
            self.assertTrue(requested["timing"]["shutdown_after_completion"])
            self.assertEqual(requested["job_id"], requested["timing"]["shutdown_job_id"])
            self.assertEqual("", requested["timing"]["shutdown_attempted_at"])
            response = overnight_response(requested)
            self.assertTrue(response["shutdown_after_completion"])
            self.assertEqual("requested", response["shutdown"]["status"])
            self.assertEqual(requested["job_id"], response["shutdown"]["job_id"])
            self.assertEqual([], response["shutdown"]["command"])

            store.update(requested["job_id"], state="completed", finished_at="now")
            self.assertTrue(store.get(requested["job_id"])["timing"]["shutdown_after_completion"])
            ordinary = store.create({}, 128)
            self.assertFalse(ordinary["timing"]["shutdown_after_completion"])
            self.assertEqual("not_requested", ordinary["timing"]["shutdown_status"])
            self.assertEqual("", ordinary["timing"]["shutdown_scheduled_at"])


class OvernightShutdownEligibilityTests(unittest.IsolatedAsyncioTestCase):
    def _record(self, temporary: str, *, requested: bool = True) -> tuple[OvernightJobStore, dict[str, object]]:
        store = OvernightJobStore(Path(temporary) / "overnight.db")
        record = store.create({}, 128, shutdown_after_completion=requested)
        rounds = [
            {
                "selection_state": "completed",
                "generation_finished_at": "now",
                "selected": [{"image": {"filename": f"best-{index}.png"}}],
                "slots": [
                    {
                        "status": "completed",
                        "image": {"filename": f"slot-{index}-{slot}.png"},
                    }
                    for slot in range(1, 33)
                ],
            }
            for index in range(4)
        ]
        record = store.update(
            record["job_id"],
            state="completed",
            selection_state="completed",
            generated_count=128,
            failed_count=0,
            rounds=rounds,
            final_selected=[{"filename": "final.png", "global_index": 1}],
            in_flight={},
            finished_at="now",
        )
        return store, record

    async def test_shutdown_is_scheduled_only_after_strict_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store, record = self._record(temporary)
            api = object.__new__(GenerationApi)
            api.overnight_jobs = store
            with patch("services.generation_api.is_comfy_queue_empty", return_value=True), patch.object(
                GenerationApi, "_overnight_gallery_reflected", return_value=True
            ), patch(
                "services.generation_api.schedule_overnight_shutdown",
                return_value={
                    "shutdown_attempted_at": "attempted",
                    "shutdown_command": ["shutdown.exe", "/s", "/t", "60"],
                    "shutdown_return_code": 0,
                    "shutdown_stdout": "",
                    "shutdown_stderr": "",
                    "shutdown_scheduled_at": "now",
                    "shutdown_cancelled_at": "",
                    "shutdown_failed_at": "",
                    "shutdown_failure_reason": "",
                    "shutdown_delay_seconds": 60,
                    "shutdown_status": "scheduled",
                },
            ) as schedule:
                await api._maybe_schedule_overnight_shutdown(record["job_id"])
            schedule.assert_called_once()
            saved = store.get(record["job_id"])
            self.assertEqual("scheduled", saved["timing"]["shutdown_status"])
            self.assertEqual(record["job_id"], saved["timing"]["shutdown_job_id"])
            self.assertEqual(0, saved["timing"]["shutdown_return_code"])
            self.assertEqual("attempted", saved["timing"]["shutdown_attempted_at"])

    async def test_shutdown_is_not_scheduled_for_failed_or_nonempty_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store, record = self._record(temporary)
            api = object.__new__(GenerationApi)
            api.overnight_jobs = store
            with patch("services.generation_api.is_comfy_queue_empty", return_value=False), patch.object(
                GenerationApi, "_overnight_gallery_reflected", return_value=True
            ), patch("services.generation_api.schedule_overnight_shutdown") as schedule:
                await api._maybe_schedule_overnight_shutdown(record["job_id"])
            schedule.assert_not_called()
            self.assertEqual("blocked", store.get(record["job_id"])["timing"]["shutdown_status"])
            self.assertEqual("comfyui_queue_not_empty", store.get(record["job_id"])["timing"]["shutdown_failure_reason"])

            failed = store.update(record["job_id"], state="failed", selection_state="failed")
            eligible, reason = GenerationApi._overnight_shutdown_eligible(
                failed, queue_empty=True, gallery_reflected=True
            )
            self.assertFalse(eligible)
            self.assertEqual("job_state_not_completed", reason)
            for state, selection_state in (
                ("stopped", "pending"),
                ("paused_error", "paused_error"),
                ("completed_selection_failed", "failed"),
            ):
                with self.subTest(state=state):
                    candidate = dict(record, state=state, selection_state=selection_state)
                    eligible, reason = GenerationApi._overnight_shutdown_eligible(
                        candidate, queue_empty=True, gallery_reflected=True
                    )
                    self.assertFalse(eligible)
                    self.assertEqual("job_state_not_completed", reason)

    async def test_shutdown_is_not_scheduled_for_gallery_reflection_or_in_flight_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store, record = self._record(temporary)
            api = object.__new__(GenerationApi)
            api.overnight_jobs = store
            with patch("services.generation_api.is_comfy_queue_empty", return_value=True), patch.object(
                GenerationApi, "_overnight_gallery_reflected", return_value=False
            ), patch("services.generation_api.schedule_overnight_shutdown") as schedule:
                await api._maybe_schedule_overnight_shutdown(record["job_id"])
            schedule.assert_not_called()
            self.assertEqual("gallery_reflection_incomplete", store.get(record["job_id"])["timing"]["shutdown_failure_reason"])

            candidate = dict(record, in_flight={"slot": 1})
            eligible, reason = GenerationApi._overnight_shutdown_eligible(
                candidate, queue_empty=True, gallery_reflected=True
            )
            self.assertFalse(eligible)
            self.assertEqual("in_flight_slot_present", reason)

    async def test_failed_shutdown_command_metadata_is_persisted_with_job_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store, record = self._record(temporary)
            api = object.__new__(GenerationApi)
            api.overnight_jobs = store
            with patch("services.generation_api.is_comfy_queue_empty", return_value=True), patch.object(
                GenerationApi, "_overnight_gallery_reflected", return_value=True
            ), patch(
                "services.generation_api.schedule_overnight_shutdown",
                return_value={
                    "shutdown_attempted_at": "attempted",
                    "shutdown_command": ["shutdown.exe"],
                    "shutdown_return_code": 5,
                    "shutdown_stdout": "",
                    "shutdown_stderr": "denied",
                    "shutdown_scheduled_at": "",
                    "shutdown_cancelled_at": "",
                    "shutdown_failed_at": "failed",
                    "shutdown_failure_reason": "shutdown.exe returned 5",
                    "shutdown_delay_seconds": 60,
                    "shutdown_status": "failed",
                },
            ):
                await api._maybe_schedule_overnight_shutdown(record["job_id"])
            saved = store.get(record["job_id"])
            self.assertEqual(record["job_id"], saved["timing"]["shutdown_job_id"])
            self.assertEqual("failed", saved["timing"]["shutdown_status"])
            self.assertEqual(5, saved["timing"]["shutdown_return_code"])
            self.assertEqual("denied", saved["timing"]["shutdown_stderr"])

    async def test_stopping_requested_shutdown_persists_cancellation_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = OvernightJobStore(Path(temporary) / "overnight.db")
            record = store.create({}, 128, shutdown_after_completion=True)
            api = object.__new__(GenerationApi)
            api.overnight_jobs = store
            api._overnight_task = None
            request = Mock()
            request.match_info = {"overnight_id": record["job_id"]}
            response = await api.stop_overnight_job(request)
            self.assertEqual(202, response.status)
            saved = store.get(record["job_id"])
            self.assertEqual("cancelled", saved["timing"]["shutdown_status"])
            self.assertTrue(saved["timing"]["shutdown_cancelled_at"])
            self.assertEqual("job_stop_requested", saved["timing"]["shutdown_failure_reason"])


class OvernightGenerationApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temporary.name) / "overnight.db")
        self.env_patch = patch.dict(os.environ, {"BATCH_GENERATION_DB_PATH": self.db_path}, clear=False)
        self.env_patch.start()
        self.app = create_generation_app()
        self.server = TestServer(self.app)
        self.client = TestClient(self.server)
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()
        self.env_patch.stop()
        self.temporary.cleanup()

    async def test_valid_overnight_job_is_durable_and_exposes_safe_state(self) -> None:
        # Keep this test at the API boundary without starting a real generation
        # worker or touching ComfyUI.
        with patch.object(GenerationApi, "_start_overnight_task"), patch.object(
            GenerationApi, "_preflight_overnight_dependencies", new=AsyncMock()
        ):
            response = await self.client.post(
                "/api/generation/overnight-jobs",
                json={
                    "total_count": 128,
                    "freeform_character": "かわいい女の子",
                    "model_key": "animagine",
                    "rating": "safe",
                    "size": "square",
                    "count": 1,
                    "variation": "auto",
                    "seed": -1,
                    "reference_mode": "none",
                },
            )
        self.assertEqual(202, response.status)
        payload = await response.json()
        self.assertEqual(128, payload["total_requested"])
        self.assertEqual("queued", payload["state"])
        self.assertEqual(4, payload["total_rounds"])
        self.assertNotIn("freeform_character", payload)

        polled = await self.client.get(f"/api/generation/overnight-jobs/{payload['overnight_job_id']}")
        self.assertEqual(200, polled.status)
        self.assertEqual(payload["overnight_job_id"], (await polled.json())["overnight_job_id"])

        normal = await self.client.post(
            "/api/generation/jobs",
            json={"model_key": "animagine", "rating": "safe", "size": "square", "count": 1, "freeform_character": "test"},
        )
        self.assertEqual(409, normal.status, await normal.text())
        bulk = await self.client.post(
            "/api/generation/bulk-jobs",
            json={"model_key": "animagine", "rating": "safe", "size": "square", "total_count": 16, "freeform_character": "test"},
        )
        self.assertEqual(409, bulk.status)

    async def test_shutdown_flag_is_persisted_only_on_the_requested_job(self) -> None:
        with patch.object(GenerationApi, "_start_overnight_task"), patch.object(
            GenerationApi, "_preflight_overnight_dependencies", new=AsyncMock()
        ):
            response = await self.client.post(
                "/api/generation/overnight-jobs",
                json={
                    "total_count": 128,
                    "model_key": "animagine",
                    "size": "square",
                    "freeform_character": "test",
                    "shutdown_after_completion": True,
                },
            )
        self.assertEqual(202, response.status)
        payload = await response.json()
        self.assertTrue(payload["shutdown_after_completion"])
        self.assertTrue(payload["shutdown"]["requested"])
        self.assertEqual("requested", payload["shutdown"]["status"])

    async def test_shutdown_flag_requires_a_boolean(self) -> None:
        with patch.object(GenerationApi, "_start_overnight_task") as start:
            response = await self.client.post(
                "/api/generation/overnight-jobs",
                json={"total_count": 128, "model_key": "animagine", "size": "square", "shutdown_after_completion": "yes"},
            )
        self.assertEqual(400, response.status)
        self.assertIn("shutdown_after_completion", (await response.json())["error"])
        start.assert_not_called()

    async def test_offline_comfyui_rejects_before_job_creation(self) -> None:
        with patch("services.generation_api.is_api_available", return_value=False):
            response = await self.client.post(
                "/api/generation/overnight-jobs",
                json={"total_count": 128, "model_key": "animagine", "size": "square", "freeform_character": "test"},
            )
        self.assertEqual(503, response.status)
        self.assertIn("ComfyUI", (await response.json())["error"])
        self.assertEqual([], OvernightJobStore(Path(self.db_path)).list_active())

    async def test_offline_lm_studio_rejects_before_job_creation(self) -> None:
        with patch("services.generation_api.is_api_available", return_value=True), patch(
            "services.generation_api._lm_studio_ready", return_value=False
        ):
            response = await self.client.post(
                "/api/generation/overnight-jobs",
                json={"total_count": 128, "model_key": "animagine", "size": "square", "freeform_character": "test"},
            )
        self.assertEqual(503, response.status)
        self.assertIn("LM Studio", (await response.json())["error"])
        self.assertEqual([], OvernightJobStore(Path(self.db_path)).list_active())

    async def test_dependency_preflight_success_is_before_persistence(self) -> None:
        with patch("services.generation_api.is_api_available", return_value=True), patch(
            "services.generation_api._lm_studio_ready", return_value=True
        ), patch.object(GenerationApi, "_start_overnight_task"):
            response = await self.client.post(
                "/api/generation/overnight-jobs",
                json={"total_count": 128, "model_key": "animagine", "size": "square", "freeform_character": "test"},
            )
        self.assertEqual(202, response.status)
        self.assertEqual(1, len(OvernightJobStore(Path(self.db_path)).list_active()))

    async def test_lm_studio_is_not_required_for_explicit_non_lm_path(self) -> None:
        with patch("services.generation_api.is_api_available", return_value=True), patch(
            "services.generation_api._lm_studio_ready", return_value=False
        ), patch.object(GenerationApi, "_overnight_requires_lm_studio", return_value=False), patch.object(
            GenerationApi, "_start_overnight_task"
        ):
            response = await self.client.post(
                "/api/generation/overnight-jobs",
                json={"total_count": 128, "model_key": "animagine", "size": "square", "freeform_character": "test"},
            )
        self.assertEqual(202, response.status)

    async def test_invalid_total_is_rejected_before_worker_start(self) -> None:
        with patch.object(GenerationApi, "_start_overnight_task") as start:
            response = await self.client.post(
                "/api/generation/overnight-jobs",
                json={"total_count": 9, "model_key": "animagine", "size": "square"},
            )
        self.assertEqual(400, response.status)
        self.assertIn("128", (await response.json())["error"])
        start.assert_not_called()

    async def test_terminal_job_reselection_is_async_and_does_not_start_generation(self) -> None:
        store = OvernightJobStore(Path(self.db_path))
        record = store.create({"model_key": "animagine"}, 128)
        store.update(record["job_id"], state="completed", generated_count=128, finished_at="now")
        with patch.object(GenerationApi, "_start_overnight_reselection_task") as start:
            response = await self.client.post(f"/api/generation/overnight-jobs/{record['job_id']}/reselect", json={})
        self.assertEqual(202, response.status)
        self.assertEqual("selecting", (await response.json())["state"])
        start.assert_called_once_with(record["job_id"])

    async def test_fully_generated_paused_job_can_reselect_after_restart(self) -> None:
        store = OvernightJobStore(Path(self.db_path))
        record = store.create({"model_key": "animagine"}, 128)
        store.update(record["job_id"], state="paused", generated_count=128, finished_at="now")
        with patch.object(GenerationApi, "_start_overnight_reselection_task") as start:
            response = await self.client.post(f"/api/generation/overnight-jobs/{record['job_id']}/reselect", json={})
        self.assertEqual(202, response.status)
        self.assertEqual("selecting", (await response.json())["state"])
        start.assert_called_once_with(record["job_id"])

    async def test_restart_pauses_in_flight_job_without_auto_resume(self) -> None:
        store = OvernightJobStore(Path(self.db_path))
        record = store.create({"model_key": "animagine"}, 128)
        store.update(
            record["job_id"],
            state="running",
            current_round=2,
            current_slot=3,
            generated_count=34,
            in_flight={"global_index": 35, "seed": 123},
        )
        await self.client.close()

        app = create_generation_app()
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            response = await client.get(f"/api/generation/overnight-jobs/{record['job_id']}")
            self.assertEqual(200, response.status)
            payload = await response.json()
            self.assertEqual("paused_error", payload["state"])
            self.assertIn("重複生成", payload["error_summary"])
            self.assertEqual(34, payload["generated_count"])
            self.assertIsNotNone(payload["timing"].get("backend_restart_recovered_at"))
            self.assertEqual("paused_error_in_flight", payload["timing"].get("backend_restart_recovery_mode"))
        finally:
            await client.close()
