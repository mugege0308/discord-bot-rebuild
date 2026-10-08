from __future__ import annotations

import unittest
import subprocess
from unittest.mock import Mock, patch

from services.overnight_shutdown import (
    DEFAULT_OVERNIGHT_SHUTDOWN_DELAY_SECONDS,
    build_shutdown_command,
    schedule_overnight_shutdown,
)


class OvernightShutdownTests(unittest.TestCase):
    def test_builds_fixed_delayed_shutdown_command(self) -> None:
        self.assertEqual(
            [
                "shutdown.exe",
                "/s",
                "/t",
                "60",
                "/c",
                "Overnight generation completed",
            ],
            build_shutdown_command(DEFAULT_OVERNIGHT_SHUTDOWN_DELAY_SECONDS),
        )

    def test_rejects_unsafe_delay_values(self) -> None:
        for value in (True, 1, 3601, "60"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_shutdown_command(value)  # type: ignore[arg-type]

    def test_schedule_is_injectable_and_does_not_run_real_shutdown(self) -> None:
        runner = Mock(return_value=subprocess.CompletedProcess(
            build_shutdown_command(DEFAULT_OVERNIGHT_SHUTDOWN_DELAY_SECONDS),
            0,
            stdout="scheduled",
            stderr="",
        ))
        with patch("services.overnight_shutdown.os.name", "nt"):
            metadata = schedule_overnight_shutdown(runner=runner)
        runner.assert_called_once_with(
            build_shutdown_command(DEFAULT_OVERNIGHT_SHUTDOWN_DELAY_SECONDS),
            check=False,
            shell=False,
            timeout=10,
            capture_output=True,
            text=True,
        )
        self.assertEqual("scheduled", metadata["shutdown_status"])
        self.assertEqual(DEFAULT_OVERNIGHT_SHUTDOWN_DELAY_SECONDS, metadata["shutdown_delay_seconds"])
        self.assertEqual(0, metadata["shutdown_return_code"])
        self.assertEqual("scheduled", metadata["shutdown_stdout"])
        self.assertEqual(build_shutdown_command(DEFAULT_OVERNIGHT_SHUTDOWN_DELAY_SECONDS), metadata["shutdown_command"])
        self.assertTrue(metadata["shutdown_attempted_at"])
        self.assertIn("shutdown_scheduled_at", metadata)

    def test_nonzero_command_result_is_persistable_failure(self) -> None:
        runner = Mock(return_value=subprocess.CompletedProcess(
            build_shutdown_command(DEFAULT_OVERNIGHT_SHUTDOWN_DELAY_SECONDS),
            5,
            stdout="",
            stderr="access denied",
        ))
        with patch("services.overnight_shutdown.os.name", "nt"):
            metadata = schedule_overnight_shutdown(runner=runner)
        self.assertEqual("failed", metadata["shutdown_status"])
        self.assertEqual(5, metadata["shutdown_return_code"])
        self.assertEqual("access denied", metadata["shutdown_stderr"])
        self.assertTrue(metadata["shutdown_failed_at"])
        self.assertIn("5", metadata["shutdown_failure_reason"])

    def test_runner_exception_is_persistable_failure(self) -> None:
        runner = Mock(side_effect=OSError("shutdown.exe missing"))
        with patch("services.overnight_shutdown.os.name", "nt"):
            metadata = schedule_overnight_shutdown(runner=runner)
        self.assertEqual("failed", metadata["shutdown_status"])
        self.assertIsNone(metadata["shutdown_return_code"])
        self.assertIn("shutdown.exe missing", metadata["shutdown_failure_reason"])


if __name__ == "__main__":
    unittest.main()
