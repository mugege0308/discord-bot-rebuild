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


def _output_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


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
    attempted_at = _now_iso()
    metadata: dict[str, object] = {
        "shutdown_attempted_at": attempted_at,
        "shutdown_command": list(command),
        "shutdown_return_code": None,
        "shutdown_stdout": "",
        "shutdown_stderr": "",
        "shutdown_scheduled_at": "",
        "shutdown_cancelled_at": "",
        "shutdown_failed_at": "",
        "shutdown_failure_reason": "",
        "shutdown_delay_seconds": delay_seconds,
    }
    try:
        result = (runner or subprocess.run)(
            command,
            check=False,
            shell=False,
            timeout=10,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must survive command errors
        failed_at = _now_iso()
        metadata.update(
            {
                "shutdown_status": "failed",
                "shutdown_failed_at": failed_at,
                "shutdown_failure_reason": f"{type(exc).__name__}: {exc}",
            }
        )
        return metadata

    return_code = getattr(result, "returncode", 0)
    stdout = _output_text(getattr(result, "stdout", ""))
    stderr = _output_text(getattr(result, "stderr", ""))
    metadata.update(
        {
            "shutdown_return_code": return_code,
            "shutdown_stdout": stdout,
            "shutdown_stderr": stderr,
        }
    )
    if return_code == 0:
        metadata.update(
            {
                "shutdown_status": "scheduled",
                "shutdown_scheduled_at": _now_iso(),
            }
        )
    else:
        metadata.update(
            {
                "shutdown_status": "failed",
                "shutdown_failed_at": _now_iso(),
                "shutdown_failure_reason": f"shutdown.exe returned {return_code}",
            }
        )
    return metadata


__all__ = [
    "DEFAULT_OVERNIGHT_SHUTDOWN_DELAY_SECONDS",
    "MAX_OVERNIGHT_SHUTDOWN_DELAY_SECONDS",
    "MIN_OVERNIGHT_SHUTDOWN_DELAY_SECONDS",
    "build_shutdown_command",
    "schedule_overnight_shutdown",
]
