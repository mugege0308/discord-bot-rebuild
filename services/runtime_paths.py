"""Project-root based paths used by the standalone Generation Backend.

The Backend can be launched by the Gallery process or from an arbitrary
terminal.  Runtime files must therefore never be derived from the caller's
current working directory.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(*parts: str) -> Path:
    """Return a path below the Rebuild project root."""

    return PROJECT_ROOT.joinpath(*parts)


def resolve_project_path(value: str | Path | None, default: str | Path) -> Path:
    """Resolve an explicit path or a default relative to ``PROJECT_ROOT``.

    Absolute environment-variable values remain explicit overrides.  Relative
    values are intentionally interpreted relative to the project root rather
    than the process CWD.
    """

    candidate = Path(value).expanduser() if value is not None and str(value).strip() else Path(default)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


__all__ = ["PROJECT_ROOT", "project_path", "resolve_project_path"]
