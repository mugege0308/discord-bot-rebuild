"""Serialize GPU-bound inference initiated by the bot.

LM Studio and ComfyUI can each use most of the RTX 3060's VRAM.  The bot's
normal flow is sequential, so a small process-local gate is safer than
stopping either external service or guessing at private configuration files.
This gate does not inspect or interrupt jobs started outside this process.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Iterator


logger = logging.getLogger("secretary-bot.gpu_resource_manager")

_gpu_inference_lock = threading.Lock()


@contextmanager
def gpu_inference_slot(owner: str) -> Iterator[None]:
    """Hold the bot's single GPU inference slot until the operation finishes."""

    started_at = time.monotonic()
    _gpu_inference_lock.acquire()
    waited_seconds = time.monotonic() - started_at
    if waited_seconds >= 0.05:
        logger.info("GPU inference slot acquired: owner=%s waited=%.2fs", owner, waited_seconds)
    try:
        yield
    finally:
        _gpu_inference_lock.release()
        logger.debug("GPU inference slot released: owner=%s", owner)


def is_gpu_inference_busy() -> bool:
    """Return whether a bot-owned GPU inference operation is currently running."""

    acquired = _gpu_inference_lock.acquire(blocking=False)
    if acquired:
        _gpu_inference_lock.release()
        return False
    return True
