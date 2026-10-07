"""One-shot Windows shutdown scheduling for completed Overnight jobs.

This module deliberately has no persistent global setting.  The caller must
prove that one specific Overnight job reached its terminal success state
before calling :func:`schedule_overnight_shutdown`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
import subprocess
from typing import Callable


DEFAULT_OVERNIGHT_SHUTDOWN_DELAY_SECONDS = 60
MIN_OVERNIGHT_SHUTDOWN_DELAY_SECONDS = 10
MAX_OVERNIGHT_SHUTDOWN_DELAY_SECONDS = 3600


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_shutdown_command(delay_seconds: int) -> list[str]:
    """Return the fixed, non-shell Windows command used for this feature."""

    if isinstance(delay_seconds, bool) or not isinstance(delay_seconds, int):
        raise ValueError("delay_seconds must be an integer.")
    if not MIN_OVERNIGHT_SHUTDOWN_DELAY_SECONDS <= delay_seconds <= MAX_OVERNIGHT_SHUTDOWN_DELAY_SECONDS:
        raise ValueError("delay_seconds is outside the allowed range.")
    return [
        "shutdown.exe",
        "/s",
        "/t",
        str(delay_seconds),
        "/c",
        "Overnight generation completed",
    ]


def schedule_overnight_shutdown(
    delay_seconds: int = DEFAULT_OVERNIGHT_SHUTDOWN_DELAY_SECONDS,
    *,
    runner: Callable[..., object] | None = None,
) -> dict[str, object]:
    """Schedule one Windows shutdown and return safe timing metadata.

    ``runner`` is injectable for tests.  Production always uses
    ``subprocess.run`` with ``shell=False`` and never includes secrets or
    request data in the command line.
    """

    if os.name != "nt":
        raise RuntimeError("Overnight shutdown is supported only on Windows.")
    command = build_shutdown_command(delay_seconds)
    (runner or subprocess.run)(command, check=True, shell=False, timeout=10)
    return {
        "shutdown_scheduled_at": _now_iso(),
        "shutdown_delay_seconds": delay_seconds,
        "shutdown_status": "scheduled",
    }


__all__ = [
    "DEFAULT_OVERNIGHT_SHUTDOWN_DELAY_SECONDS",
    "MAX_OVERNIGHT_SHUTDOWN_DELAY_SECONDS",
    "MIN_OVERNIGHT_SHUTDOWN_DELAY_SECONDS",
    "build_shutdown_command",
    "schedule_overnight_shutdown",
]
