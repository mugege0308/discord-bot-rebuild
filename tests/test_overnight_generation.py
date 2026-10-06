from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer

from services.generation_api import GenerationApi, create_generation_app
from services.overnight_generation import (
    OVERNIGHT_ROUND_SIZE,
    OVERNIGHT_TOTALS,
    OvernightJobStore,
    overnight_response,
)


class OvernightGenerationStoreTests(unittest.TestCase):
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
        with patch.object(GenerationApi, "_start_overnight_task"):
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

    async def test_invalid_total_is_rejected_before_worker_start(self) -> None:
        with patch.object(GenerationApi, "_start_overnight_task") as start:
            response = await self.client.post(
                "/api/generation/overnight-jobs",
                json={"total_count": 9, "model_key": "animagine", "size": "square"},
            )
        self.assertEqual(400, response.status)
        self.assertIn("128", (await response.json())["error"])
        start.assert_not_called()

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
