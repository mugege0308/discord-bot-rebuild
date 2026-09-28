from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from services import bot_restart_launcher


RUNTIME_MARKERS = [
    "2026-09-28 15:00:00,000 [INFO] secretary-bot: runtime_event=process_start pid=4321",
    "2026-09-28 15:00:02,000 [INFO] secretary-bot: Slash Command sync succeeded: global_count=0 guild_count=99.",
    "2026-09-28 15:00:03,000 [INFO] secretary-bot: runtime_event=discord_login_success user_id=1 pid=4321",
]


class BotRestartLauncherTestCase(unittest.TestCase):
    def test_runtime_markers_extract_process_sync_and_login(self) -> None:
        pid, startup_line, sync, login = bot_restart_launcher.parse_runtime_markers(
            RUNTIME_MARKERS,
            None,
        )

        self.assertEqual(pid, 4321)
        self.assertIn("process_start", startup_line)
        self.assertTrue(sync)
        self.assertTrue(login)

    def test_runtime_markers_accept_cached_sync_skip(self) -> None:
        markers = [
            "2026-09-28 15:00:00,000 [INFO] secretary-bot: runtime_event=process_start pid=4321",
            "2026-09-28 15:00:02,000 [INFO] secretary-bot: Skipped Slash Command sync: command fingerprint and guild target are unchanged.",
            "2026-09-28 15:00:03,000 [INFO] secretary-bot: runtime_event=discord_login_success user_id=1 pid=4321",
        ]

        pid, _startup_line, sync, login = bot_restart_launcher.parse_runtime_markers(markers, None)

        self.assertEqual(pid, 4321)
        self.assertTrue(sync)
        self.assertTrue(login)

    def test_success_records_stdout_stderr_and_final_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            restart_log = root / "bot_restart.log"
            runtime_log = root / "bot_runtime.log"
            stdout_log = root / "stdout.log"
            stderr_log = root / "stderr.log"
            start_file = root / "start_bot_restart.bat"
            start_file.write_text("@echo off\n", encoding="utf-8")
            stdout_log.write_text("launcher stdout\n", encoding="utf-8")
            stderr_log.write_text("launcher stderr\n", encoding="utf-8")
            fake_process = SimpleNamespace(pid=9876, poll=lambda: None)

            with patch.object(bot_restart_launcher, "wait_for_process_exit", return_value=True), patch.object(
                bot_restart_launcher,
                "start_restart_process",
                return_value=fake_process,
            ), patch.object(
                bot_restart_launcher,
                "read_runtime_delta",
                return_value=(RUNTIME_MARKERS, 1),
            ), patch.object(
                bot_restart_launcher,
                "process_is_alive",
                return_value=True,
            ), patch.object(bot_restart_launcher.time, "monotonic", return_value=0.0):
                result = bot_restart_launcher.run_restart(
                    old_pid=1234,
                    start_file=start_file,
                    project_dir=root,
                    restart_log=restart_log,
                    runtime_log=runtime_log,
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    startup_timeout=0,
                    monitor_timeout=0,
                    poll_interval=0,
                )

            text = restart_log.read_text(encoding="utf-8")
            self.assertEqual(result, 0)
            self.assertIn("new_process_start detected=true new_pid=4321", text)
            self.assertIn("sync_detected=True login_detected=True", text)
            self.assertIn("launcher_stdout_tail=launcher stdout", text)
            self.assertIn("launcher_stderr_tail=launcher stderr", text)
            self.assertIn("final_restart_result=success new_pid=4321", text)

    def test_launcher_failure_is_recorded_without_claiming_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            restart_log = root / "bot_restart.log"
            runtime_log = root / "bot_runtime.log"
            stdout_log = root / "stdout.log"
            stderr_log = root / "stderr.log"
            start_file = root / "start_bot_restart.bat"
            start_file.write_text("@echo off\n", encoding="utf-8")

            with patch.object(bot_restart_launcher, "wait_for_process_exit", return_value=True), patch.object(
                bot_restart_launcher,
                "start_restart_process",
                side_effect=OSError("cmd unavailable"),
            ):
                result = bot_restart_launcher.run_restart(
                    old_pid=1234,
                    start_file=start_file,
                    project_dir=root,
                    restart_log=restart_log,
                    runtime_log=runtime_log,
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    startup_timeout=0,
                    monitor_timeout=0,
                    poll_interval=0,
                )

            text = restart_log.read_text(encoding="utf-8")
            self.assertEqual(result, 1)
            self.assertIn("launcher_process_start success=false", text)
            self.assertIn("final_restart_result=failed:launcher_start", text)

    def test_new_process_exit_is_recorded_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            restart_log = root / "bot_restart.log"
            runtime_log = root / "bot_runtime.log"
            stdout_log = root / "stdout.log"
            stderr_log = root / "stderr.log"
            start_file = root / "start_bot_restart.bat"
            start_file.write_text("@echo off\n", encoding="utf-8")
            fake_process = SimpleNamespace(pid=9876, poll=lambda: 1)

            with patch.object(bot_restart_launcher, "wait_for_process_exit", return_value=True), patch.object(
                bot_restart_launcher,
                "start_restart_process",
                return_value=fake_process,
            ), patch.object(
                bot_restart_launcher,
                "read_runtime_delta",
                return_value=(RUNTIME_MARKERS, 1),
            ), patch.object(
                bot_restart_launcher,
                "process_is_alive",
                return_value=False,
            ), patch.object(bot_restart_launcher.time, "monotonic", return_value=0.0):
                result = bot_restart_launcher.run_restart(
                    old_pid=1234,
                    start_file=start_file,
                    project_dir=root,
                    restart_log=restart_log,
                    runtime_log=runtime_log,
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    startup_timeout=0,
                    monitor_timeout=0,
                    poll_interval=0,
                )

            text = restart_log.read_text(encoding="utf-8")
            self.assertEqual(result, 1)
            self.assertIn("new_process_exit detected=true new_pid=4321", text)
            self.assertIn("final_restart_result=failed:new_process_exited", text)


if __name__ == "__main__":
    unittest.main()
