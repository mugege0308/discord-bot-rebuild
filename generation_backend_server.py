"""Standalone launcher for the loopback Generation API.

It is intentionally separate from the Discord Gateway process.  No server is
started by importing this module.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# Keep the standalone Backend launcher consistent with the main Bot entrypoint:
# deployment-local .env values are loaded without overriding explicit process
# environment values supplied by a regular launcher.
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

from services.generation_api import (
    DEFAULT_GENERATION_API_HOST,
    DEFAULT_GENERATION_API_PORT,
    DEFAULT_MAX_BODY_BYTES,
    GenerationApiConfig,
    run_generation_api,
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}が不正です。") from exc


def main() -> None:
    config = GenerationApiConfig(
        host=os.getenv("GENERATION_API_HOST", DEFAULT_GENERATION_API_HOST),
        port=_env_int("GENERATION_API_PORT", DEFAULT_GENERATION_API_PORT),
        max_body_bytes=_env_int("GENERATION_API_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES),
    )
    run_generation_api(config=config)


if __name__ == "__main__":
    main()
