from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import bot_control


class BotControlTestCase(unittest.TestCase):
    def test_instance_lock_can_be_released_and_reacquired(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bot.lock"
            first = bot_control.BotInstanceLock(path).acquire()
            first.release()
            second = bot_control.BotInstanceLock(path).acquire()
            second.release()

    def test_instance_lock_rejects_another_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bot.lock"
            script = (
                "import sys, time; "
                "from pathlib import Path; "
                "from services.bot_control import BotInstanceLock; "
                "lock=BotInstanceLock(Path(sys.argv[1])).acquire(); "
                "print('ready', flush=True); time.sleep(2)"
            )
            child = subprocess.Popen(
                [sys.executable, "-c", script, str(path)],
                cwd=str(Path(__file__).resolve().parents[1]),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(child.stdout.readline().strip(), "ready")
                with self.assertRaises(bot_control.BotAlreadyRunningError):
                    bot_control.BotInstanceLock(path).acquire()
            finally:
                child.wait(timeout=5)
                if child.stdout is not None:
                    child.stdout.close()
                if child.stderr is not None:
                    child.stderr.close()

    def test_restart_request_is_claimed_once_until_released(self) -> None:
        bot_control.release_restart_request()
        self.assertTrue(bot_control.claim_restart_request())
        self.assertFalse(bot_control.claim_restart_request())
        bot_control.release_restart_request()
        self.assertTrue(bot_control.claim_restart_request())
        bot_control.release_restart_request()

    def test_default_instance_lock_path_is_under_bot_data(self) -> None:
        with patch.dict(os.environ, {"BOT_INSTANCE_LOCK_FILE": ""}, clear=False):
            path = bot_control.get_bot_instance_lock_path()
        self.assertEqual(path.name, "bot_instance.lock")
        self.assertEqual(path.parent.name, "data")

    def test_restart_monitor_is_started_before_old_bot_exits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start_file = root / "start_bot.bat"
            restart_file = root / "start_bot_restart.bat"
            start_file.write_text("@echo off\npause\n", encoding="utf-8")
            restart_file.write_text("@echo off\nexit /b 0\n", encoding="utf-8")
            log_file = root / "bot_restart.log"
            with patch.dict(
                os.environ,
                {
                    "BOT_START_FILE": str(start_file),
                    "BOT_RESTART_LOG_FILE": str(log_file),
                },
                clear=False,
            ), patch("services.bot_control.subprocess.Popen") as popen:
                popen.return_value.pid = 4321
                result = bot_control.launch_bot_restart(wait_for_pid=1234)

            command = popen.call_args.args[0]
            self.assertEqual(result.monitor_pid, 4321)
            self.assertEqual(result.start_file, restart_file.resolve())
            self.assertIn("bot_restart_launcher.py", " ".join(command))
            self.assertIn("--old-pid", command)
            self.assertIn("1234", command)
            log_text = log_file.read_text(encoding="utf-8")
            self.assertIn("restart_request", log_text)
            self.assertIn("launcher_process_start success=true", log_text)

    def test_restart_monitor_start_failure_is_logged_and_raised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start_file = root / "start_bot.bat"
            restart_file = root / "start_bot_restart.bat"
            start_file.write_text("@echo off\npause\n", encoding="utf-8")
            restart_file.write_text("@echo off\nexit /b 0\n", encoding="utf-8")
            log_file = root / "bot_restart.log"
            with patch.dict(
                os.environ,
                {
                    "BOT_START_FILE": str(start_file),
                    "BOT_RESTART_LOG_FILE": str(log_file),
                },
                clear=False,
            ), patch(
                "services.bot_control.subprocess.Popen",
                side_effect=OSError("launcher unavailable"),
            ), self.assertRaises(OSError):
                bot_control.launch_bot_restart(wait_for_pid=1234)

            self.assertIn(
                "launcher_process_start success=false",
                log_file.read_text(encoding="utf-8"),
            )

    def test_manual_launcher_keeps_pause_but_restart_launcher_does_not(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manual = (root / "start_bot.bat").read_text(encoding="utf-8").lower()
        restart = (root / "start_bot_restart.bat").read_text(encoding="utf-8").lower()
        self.assertIn("pause", manual)
        self.assertNotIn("pause", restart)


if __name__ == "__main__":
    unittest.main()
