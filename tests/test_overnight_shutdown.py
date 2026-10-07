from __future__ import annotations

import unittest
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
        runner = Mock()
        with patch("services.overnight_shutdown.os.name", "nt"):
            metadata = schedule_overnight_shutdown(runner=runner)
        runner.assert_called_once_with(
            build_shutdown_command(DEFAULT_OVERNIGHT_SHUTDOWN_DELAY_SECONDS),
            check=True,
            shell=False,
            timeout=10,
        )
        self.assertEqual("scheduled", metadata["shutdown_status"])
        self.assertEqual(DEFAULT_OVERNIGHT_SHUTDOWN_DELAY_SECONDS, metadata["shutdown_delay_seconds"])
        self.assertIn("shutdown_scheduled_at", metadata)


if __name__ == "__main__":
    unittest.main()
