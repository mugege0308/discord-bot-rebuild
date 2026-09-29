from __future__ import annotations

import asyncio
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from services import lora_job, lora_upload


class LoraJobServiceTestCase(unittest.TestCase):
    def _prepared_session(self, root: Path) -> tuple[dict, Path, Path]:
        sessions = root / "sessions"
        manager_root = root / "manager"
        jobs = root / "jobs"
        (manager_root / ".venv" / "Scripts").mkdir(parents=True)
        (manager_root / ".venv" / "Scripts" / "python.exe").write_bytes(b"python")
        (manager_root / "lora_bot_bridge.py").write_text("# bridge", encoding="utf-8")
        expected = manager_root / "outputs" / "dummy_v1.safetensors"
        expected.parent.mkdir(parents=True)
        expected.write_bytes(b"dummy output")
        info_path = manager_root / "plans" / "training_info.json"
        info_path.parent.mkdir()
        info_path.write_text(
            json.dumps(
                {
                    "dataset_name": "dummy",
                    "output_name": "dummy_v1",
                    "settings": {"total_steps": 2, "lora_type": "style"},
                    "lora_metadata": {"lora_type": "style", "trigger_tag": "", "recommended_strength": 0.5},
                    "paths": {"expected_output": str(expected), "dataset_dir": str(manager_root / "lora_datasets" / "dummy")},
                }
            ),
            encoding="utf-8",
        )
        environment = patch.dict(
            os.environ,
            {"LORA_UPLOAD_ROOT": str(sessions), "LORA_MANAGER_ROOT": str(manager_root), "LORA_JOB_ROOT": str(jobs)},
            clear=False,
        )
        environment.start()
        session = lora_upload.create_session(
            owner_id=1, guild_id=None, channel_id=1, message_id=None,
            dataset_name="dummy", lora_type="style", trigger_tag="", recommended_strength=0.5,
        )
        session["status"] = "plan_created"
        session["plan"] = {"info_path": str(info_path), "output_name": "dummy_v1", "expected_output": str(expected)}
        (sessions / f"{session['session_id']}.json").write_text(json.dumps(session), encoding="utf-8")
        return session, expected, info_path

    def _finish_environment(self) -> None:
        for name in ("LORA_UPLOAD_ROOT", "LORA_MANAGER_ROOT", "LORA_JOB_ROOT"):
            os.environ.pop(name, None)

    def test_progress_line_parsing(self) -> None:
        progress = lora_job.parse_progress_line("epoch 2/10 step 50/500 loss=0.123 1.25it/s")
        self.assertEqual(progress["epoch"], 2)
        self.assertEqual(progress["step"], 50)
        self.assertEqual(progress["total_steps"], 500)
        self.assertAlmostEqual(progress["loss"], 0.123)
        self.assertEqual(progress["speed"], 1.25)

    def test_short_bridge_forces_utf8_and_preserves_japanese_error(self) -> None:
        class Stream:
            def __init__(self, *lines: bytes) -> None:
                self.lines = list(lines)

            async def readline(self) -> bytes:
                await asyncio.sleep(0)
                return self.lines.pop(0) if self.lines else b""

        class Process:
            stdout = Stream(
                'LORA_BRIDGE_RESULT:{"error":"日本語のエラー"}\n'.encode("utf-8")
            )
            stderr = Stream("詳細な日本語エラー\n".encode("utf-8"))

            async def wait(self) -> int:
                return 1

        captured: dict[str, object] = {}

        async def fake_exec(*args, **kwargs):
            captured.update(kwargs)
            return Process()

        async def run() -> tuple[int, str, str, dict | None]:
            with patch("services.lora_job.asyncio.create_subprocess_exec", new=fake_exec):
                return await lora_job._run_short_bridge(["python", "bridge"], cwd=Path("."))

        return_code, _stdout, stderr, payload = asyncio.run(run())
        self.assertEqual(return_code, 1)
        self.assertEqual(stderr, "詳細な日本語エラー")
        self.assertEqual(payload, {"error": "日本語のエラー"})
        environment = captured["env"]
        self.assertIsInstance(environment, dict)
        self.assertEqual(environment["PYTHONUTF8"], "1")
        self.assertEqual(environment["PYTHONIOENCODING"], "utf-8")

    def test_kohya_tqdm_progress_requires_training_total_and_reads_avr_loss(self) -> None:
        line = "steps:  90%|████████▉ | 864/960 [35:57<03:59,  2.50s/it, avr_loss=0.0855]"
        progress = lora_job.parse_progress_line(line, expected_total_steps=960)
        self.assertEqual(progress["step"], 864)
        self.assertEqual(progress["total_steps"], 960)
        self.assertAlmostEqual(progress["speed"], 0.4)
        self.assertEqual(progress["speed_text"], "2.50s/it")
        self.assertAlmostEqual(progress["loss"], 0.0855)

        preparation = lora_job.parse_progress_line(
            "100%|██████████| 12/12 [00:16<00:00, 1.41s/it]",
            expected_total_steps=960,
        )
        self.assertEqual(preparation, {})

        combined = "\rsteps: 10%|█         | 100/960 [04:00<10:00, 2.40s/it, avr_loss=0.2]"
        combined += "\rsteps: 10%|█         | 101/960 [04:02<09:59, 2.40s/it, avr_loss=0.19]"
        latest = lora_job.parse_progress_line(combined, expected_total_steps=960)
        self.assertEqual(latest["step"], 101)
        self.assertAlmostEqual(latest["loss"], 0.19)

    def test_completed_progress_snapshot_reconciles_stale_saved_state(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "sutera_v2_character_v1.safetensors"
            output.write_bytes(b"completed")
            log = root / "job.log"
            log.write_text(
                "steps:  90%|████████▉ | 864/960 [35:57<03:59, 2.50s/it, avr_loss=0.0855]\n",
                encoding="utf-8",
            )
            snapshot = lora_job.progress_snapshot(
                {
                    "status": "success",
                    "expected_output": str(output),
                    "log_file": str(log),
                    "progress": {"step": 0, "total_steps": 960, "loss": None},
                }
            )
        self.assertEqual(snapshot["step"], 960)
        self.assertEqual(snapshot["percent"], 100.0)
        self.assertEqual(snapshot["eta_seconds"], 0)
        self.assertAlmostEqual(snapshot["loss"], 0.0855)

    def test_startup_configuration_validation_checks_fixed_kohya_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manager_root = root / "manager"
            kohya_root = root / "kohya"
            (manager_root / ".venv" / "Scripts").mkdir(parents=True)
            (manager_root / ".venv" / "Scripts" / "python.exe").write_bytes(b"python")
            (manager_root / "lora_bot_bridge.py").write_text("# bridge", encoding="utf-8")
            (kohya_root / ".venv" / "Scripts").mkdir(parents=True)
            (kohya_root / ".venv" / "Scripts" / "python.exe").write_bytes(b"python")
            (kohya_root / "sd-scripts").mkdir()
            (kohya_root / "sd-scripts" / "sdxl_train_network.py").write_text("# kohya", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"LORA_MANAGER_ROOT": str(manager_root), "LORA_KOHYA_ROOT": str(kohya_root)},
                clear=False,
            ):
                configuration = lora_job.validate_configuration()
            self.assertEqual(Path(configuration["train_script"]).name, "sdxl_train_network.py")

    def test_startup_configuration_validation_rejects_missing_kohya_script(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manager_root = root / "manager"
            kohya_root = root / "kohya"
            (manager_root / ".venv" / "Scripts").mkdir(parents=True)
            (manager_root / ".venv" / "Scripts" / "python.exe").write_bytes(b"python")
            (manager_root / "lora_bot_bridge.py").write_text("# bridge", encoding="utf-8")
            (kohya_root / ".venv" / "Scripts").mkdir(parents=True)
            (kohya_root / ".venv" / "Scripts" / "python.exe").write_bytes(b"python")
            (kohya_root / "sd-scripts").mkdir()
            with patch.dict(
                os.environ,
                {"LORA_MANAGER_ROOT": str(manager_root), "LORA_KOHYA_ROOT": str(kohya_root)},
                clear=False,
            ):
                with self.assertRaises(lora_job.LoraJobError):
                    lora_job.validate_configuration()

    def test_one_running_job_is_rejected_without_starting_a_process(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"LORA_JOB_ROOT": str(root / "jobs")}, clear=False):
                job_id = "0123456789abcdef0123456789abcdef"
                (root / "jobs").mkdir()
                (root / "jobs" / f"{job_id}.json").write_text(
                    json.dumps({"job_id": job_id, "owner_id": 1, "status": "running"}), encoding="utf-8"
                )
                manager = lora_job.LoraJobManager()
                self.assertIsNotNone(manager.has_running_job())
                with self.assertRaises(lora_job.LoraJobError):
                    asyncio.run(manager.start("fedcba9876543210fedcba9876543210", 1))

    def test_plan_preview_rejects_paths_outside_fixed_manager_root(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = root / "sessions"
            manager_root = root / "manager"
            sessions.mkdir()
            manager_root.mkdir()
            with patch.dict(
                os.environ,
                {"LORA_UPLOAD_ROOT": str(sessions), "LORA_MANAGER_ROOT": str(manager_root)},
                clear=False,
            ):
                session = lora_upload.create_session(
                    owner_id=1, guild_id=None, channel_id=1, message_id=None,
                    dataset_name="安全確認", lora_type="style", trigger_tag="", recommended_strength=0.5,
                )
                session_path = sessions / session["session_id"]
                state_path = sessions / f"{session['session_id']}.json"
                session["status"] = "plan_created"
                session["plan"] = {"info_path": str(root / "outside" / "training_info.json")}
                state_path.write_text(json.dumps(session), encoding="utf-8")
                with self.assertRaises(lora_job.LoraJobError):
                    lora_job.LoraJobManager().plan_preview(session["session_id"], 1)
                self.assertTrue(session_path.exists())

    def test_plan_preview_blocks_changes_to_embedded_weighted_design(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = root / "sessions"
            manager_root = root / "manager"
            train_dir = manager_root / "lora_datasets" / "weighted" / "train"
            general = train_dir / "images" / "general"
            detail = train_dir / "images" / "identity_detail"
            general.mkdir(parents=True)
            detail.mkdir(parents=True)
            (manager_root / "plans").mkdir(parents=True)
            dataset_config = manager_root / "plans" / "dataset.toml"
            training_config = manager_root / "plans" / "config.toml"
            info_path = manager_root / "plans" / "training_info.json"
            dataset_config.write_text(
                f'[[datasets]]\n  [[datasets.subsets]]\n  image_dir = "{general.as_posix()}"\n  num_repeats = 8\n\n'
                f'  [[datasets.subsets]]\n  image_dir = "{detail.as_posix()}"\n  num_repeats = 24\n',
                encoding="utf-8",
            )
            training_config.write_text("max_train_epochs = 6\n", encoding="utf-8")
            info_path.write_text(
                json.dumps(
                    {
                        "dataset_name": "weighted",
                        "output_name": "weighted_v1",
                        "settings": {
                            "epochs": 6,
                            "subsets": [
                                {"path": "images/general", "repeats": 8},
                                {"path": "images/identity_detail", "repeats": 24},
                            ],
                        },
                        "embedded_training": {"source_manifest": "manifest.json"},
                        "paths": {
                            "train_dir": str(train_dir),
                            "dataset_config": str(dataset_config),
                            "training_config": str(training_config),
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"LORA_UPLOAD_ROOT": str(sessions), "LORA_MANAGER_ROOT": str(manager_root)},
                clear=False,
            ):
                session = lora_upload.create_session(
                    owner_id=1, guild_id=None, channel_id=1, message_id=None,
                    dataset_name="weighted", lora_type="style", trigger_tag="", recommended_strength=0.5,
                )
                session["status"] = "plan_created"
                session["plan"] = {"info_path": str(info_path)}
                (sessions / f"{session['session_id']}.json").write_text(json.dumps(session), encoding="utf-8")
                preview = lora_job.LoraJobManager().plan_preview(session["session_id"], 1)
                self.assertEqual(preview["settings"]["epochs"], 6)
                dataset_config.write_text(dataset_config.read_text(encoding="utf-8").replace("num_repeats = 24", "num_repeats = 8"), encoding="utf-8")
                with self.assertRaisesRegex(lora_job.LoraJobError, "subset別repeats"):
                    lora_job.LoraJobManager().plan_preview(session["session_id"], 1)

    def test_dummy_process_success_is_persisted_with_progress(self) -> None:
        class Stream:
            def __init__(self, lines: list[bytes]) -> None:
                self.lines = lines

            async def readline(self) -> bytes:
                await asyncio.sleep(0)
                return self.lines.pop(0) if self.lines else b""

        class Process:
            pid = 4567

            def __init__(self) -> None:
                self.stdout = Stream([b"step 1/2 1.0it/s loss=0.2\n", b"step 2/2 1.0it/s loss=0.1\n"])
                self.stderr = Stream([])

            async def wait(self) -> int:
                return 0

        with TemporaryDirectory() as directory:
            root = Path(directory)
            session, _expected, _info = self._prepared_session(root)
            try:
                manager = lora_job.LoraJobManager()
                process = Process()

                async def fake_exec(*args, **kwargs):
                    return process

                async def run() -> dict:
                    with patch("services.lora_job.asyncio.create_subprocess_exec", new=fake_exec):
                        job = await manager.start(session["session_id"], 1)
                        await manager._tasks[job["job_id"]]
                    return manager.get(job["job_id"])

                job = asyncio.run(run())
                self.assertEqual(job["status"], "success")
                self.assertEqual(job["progress"]["step"], 2)
                self.assertAlmostEqual(job["progress"]["percent"], 100.0)
                self.assertEqual(job["progress"]["eta_seconds"], 0)
            finally:
                self._finish_environment()

    def test_dummy_process_failure_and_safe_stop_are_distinguished(self) -> None:
        class Stream:
            def __init__(self) -> None:
                self.read = False

            async def readline(self) -> bytes:
                await asyncio.sleep(0)
                if self.read:
                    return b""
                self.read = True
                return b"training failed\n"

        class FailedProcess:
            pid = 4568

            def __init__(self) -> None:
                self.stdout = Stream()
                self.stderr = Stream()

            async def wait(self) -> int:
                return 1

        with TemporaryDirectory() as directory:
            root = Path(directory)
            session, expected, _info = self._prepared_session(root)
            expected.unlink()
            try:
                manager = lora_job.LoraJobManager()

                async def run_failure() -> dict:
                    async def fake_exec(*args, **kwargs):
                        return FailedProcess()

                    with patch("services.lora_job.asyncio.create_subprocess_exec", new=fake_exec):
                        job = await manager.start(session["session_id"], 1)
                        await manager._tasks[job["job_id"]]
                    return manager.get(job["job_id"])

                failed = asyncio.run(run_failure())
                self.assertEqual(failed["status"], "failed")
                self.assertIn("training failed", failed["error_summary"])
            finally:
                self._finish_environment()

    def test_dummy_process_is_stopped_only_after_manager_confirmation(self) -> None:
        class EmptyStream:
            async def readline(self) -> bytes:
                await asyncio.sleep(0)
                return b""

        class WaitingProcess:
            pid = 4569

            def __init__(self) -> None:
                self.done = asyncio.Event()
                self.stdout = EmptyStream()
                self.stderr = EmptyStream()

            async def wait(self) -> int:
                await self.done.wait()
                return 130

            def terminate(self) -> None:
                self.done.set()

        class Stopper:
            def __init__(self, process: WaitingProcess) -> None:
                self.process = process

            async def wait(self) -> int:
                self.process.done.set()
                return 0

        with TemporaryDirectory() as directory:
            root = Path(directory)
            session, _expected, _info = self._prepared_session(root)
            try:
                manager = lora_job.LoraJobManager()
                process = WaitingProcess()

                async def fake_exec(*args, **kwargs):
                    if args and args[0] == "taskkill":
                        return Stopper(process)
                    return process

                async def run() -> dict:
                    with patch("services.lora_job.asyncio.create_subprocess_exec", new=fake_exec):
                        job = await manager.start(session["session_id"], 1)
                        await asyncio.sleep(0)
                        with self.assertRaises(lora_job.LoraJobError):
                            await manager.cancel(job["job_id"], 999)
                        await manager.cancel(job["job_id"], 1)
                        await manager._tasks[job["job_id"]]
                    return manager.get(job["job_id"])

                job = asyncio.run(run())
                self.assertEqual(job["status"], "stopped")
                self.assertTrue(job["cancel_requested"])
            finally:
                self._finish_environment()


if __name__ == "__main__":
    unittest.main()
