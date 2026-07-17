from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger("secretary-bot.dev_cli_runner")

# This module is a generalization of services.dev_git_runner.run_git() for
# running an arbitrary fixed-argument-list CLI (Phase 4A: the Claude Code
# CLI) instead of only `git`. The core subprocess-safety primitives below
# (capped output draining, Windows taskkill process-tree kill, launch-error
# classification) are deliberately re-implemented here rather than imported
# from dev_git_runner, rather than refactoring that already-hardened,
# already-tested Phase 2 module to share code with a brand new Phase 4A
# caller - see the Phase 4A task notes on preferring safety/stability over a
# forced shared-module refactor.

_WINDOWS_SUBPROCESS_KWARGS: dict[str, Any] = (
    {
        "creationflags": (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    }
    if os.name == "nt"
    else {}
)
_POSIX_SUBPROCESS_KWARGS: dict[str, Any] = {} if os.name == "nt" else {"start_new_session": True}

_READER_CHUNK_SIZE = 65536
_PROCESS_TREE_KILL_TIMEOUT_SECONDS = 10.0
_READER_JOIN_TIMEOUT_SECONDS = 10.0
_WRITER_JOIN_TIMEOUT_SECONDS = 10.0
# How often the wait loop below wakes up to check the deadline/cancel token.
# Small enough that a cancel request or a timeout is noticed promptly,
# without busy-looping.
_WAIT_POLL_SECONDS = 0.2

# Stable, machine-checkable reasons, mirroring the shape of
# services.dev_git_runner's own error_code set.
ERROR_CLI_NOT_FOUND = "cli_not_found"
ERROR_CWD_NOT_FOUND = "cwd_not_found"
ERROR_LAUNCH_FAILED = "launch_failed"
ERROR_COMMAND_FAILED = "command_failed"
ERROR_TIMEOUT = "timeout"
ERROR_CANCELLED = "cancelled"
ERROR_OUTPUT_READER_STUCK = "output_reader_stuck"
ERROR_STDIN_WRITE_FAILED = "stdin_write_failed"
ERROR_STDIN_WRITER_STUCK = "stdin_writer_stuck"


@dataclass(frozen=True)
class CliRunResult:
    success: bool
    returncode: int | None
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    timed_out: bool
    cancelled: bool
    # Only meaningful when timed_out or cancelled is True: whether the
    # platform-specific tree-kill mechanism itself confirmed success
    # (taskkill's own exit code 0 on Windows; os.killpg() raising nothing on
    # POSIX). True (not applicable) whenever no kill was ever attempted.
    termination_confirmed: bool = True
    child_processes_may_remain: bool = False
    duration_seconds: float = 0.0
    error_code: str | None = None
    # False whenever a stdout/stderr reader thread was still alive after
    # being given every chance to terminate.
    reader_threads_terminated: bool = True
    # False whenever the stdin writer thread (see _StdinWriter) was still
    # alive after being given every chance to terminate - e.g. blocked
    # writing to a CLI that never reads its stdin at all. True whenever no
    # stdin was ever written (nothing to wait for).
    stdin_writer_terminated: bool = True


class CancellationToken:
    """A simple thread-safe cancellation flag: set() is called from the
    asyncio event loop thread (e.g. in response to /dev_task_stop), and
    is_set() is polled from the worker thread that actually runs run_cli()
    (via asyncio.to_thread), since that worker thread - not the asyncio task
    wrapping it - is what is blocked waiting on the subprocess.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


def _decode_for_log(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _close_quietly(stream: Any) -> None:
    if stream is None:
        return
    try:
        stream.close()
    except OSError:
        pass


class _CappedReader(threading.Thread):
    """Drains a subprocess pipe in a background thread, keeping only the
    first `cap` bytes and discarding everything past it - never buffering
    more than `cap` bytes anywhere, in memory or otherwise.

    Continuing to read (and discard) past the cap is what actually matters:
    a subprocess.PIPE has a small OS-level buffer, and if nobody keeps
    reading it once it fills, the child process blocks on its next write
    forever while this process blocks in Popen.wait() - the classic
    subprocess.PIPE deadlock. This thread's job is specifically to prevent
    that, independent of how much output the CLI produces.
    """

    def __init__(self, stream: Any, cap: int):
        super().__init__(daemon=True)
        self._stream = stream
        self._cap = max(0, cap)
        self._chunks: list[bytes] = []
        self._total = 0
        self._truncated = False

    def run(self) -> None:
        try:
            while True:
                chunk = self._stream.read(_READER_CHUNK_SIZE)
                if not chunk:
                    break
                if self._total < self._cap:
                    remaining = self._cap - self._total
                    if len(chunk) > remaining:
                        self._chunks.append(chunk[:remaining])
                        self._truncated = True
                    else:
                        self._chunks.append(chunk)
                else:
                    self._truncated = True
                self._total += len(chunk)
        except (ValueError, OSError):
            # The stream was closed out from under this thread (e.g. the
            # main thread gave up waiting and moved on) - nothing left to
            # read, and nothing more this thread can safely do.
            pass

    def result(self) -> tuple[bytes, bool]:
        return b"".join(self._chunks), self._truncated


class _StdinWriter(threading.Thread):
    """Writes `data` to the child's stdin pipe, then closes it, entirely on
    a background thread.

    This must never run on the caller's thread: if the CLI never reads its
    stdin at all (or reads slowly) and `data` is larger than the OS pipe
    buffer, a synchronous write() blocks indefinitely - which would
    previously have blocked run_cli() itself before it ever reached the
    timeout/cancel-aware wait loop below, defeating both. Running it here
    instead means the main wait loop starts polling immediately regardless
    of whether the write has completed, and a stuck write is later detected
    (not silently ignored) via is_alive() after being asked to stop - see
    _join_writer_and_check_alive() and the proactive popen.stdin.close()
    call on the timeout/cancel path, which is what actually unblocks a
    write stuck on a full, unread pipe (turning it into a normal
    BrokenPipeError this thread catches).
    """

    def __init__(self, stream: Any, data: bytes):
        super().__init__(daemon=True)
        self._stream = stream
        self._data = data
        self.write_failed = False

    def run(self) -> None:
        try:
            self._stream.write(self._data)
        except (OSError, ValueError):
            self.write_failed = True
        finally:
            _close_quietly(self._stream)


def _join_writer_and_check_alive(writer: "_StdinWriter | None") -> bool:
    if writer is None:
        return True
    writer.join(timeout=_WRITER_JOIN_TIMEOUT_SECONDS)
    if writer.is_alive():
        logger.error("Stdin writer thread did not terminate")
        return False
    return True


def _join_readers_and_check_alive(
    popen: "subprocess.Popen[bytes]",
    stdout_reader: "_CappedReader",
    stderr_reader: "_CappedReader",
    command: list[str],
) -> bool:
    stdout_reader.join(timeout=_READER_JOIN_TIMEOUT_SECONDS)
    stderr_reader.join(timeout=_READER_JOIN_TIMEOUT_SECONDS)

    if stdout_reader.is_alive() or stderr_reader.is_alive():
        _close_quietly(popen.stdout)
        _close_quietly(popen.stderr)
        stdout_reader.join(timeout=_READER_JOIN_TIMEOUT_SECONDS)
        stderr_reader.join(timeout=_READER_JOIN_TIMEOUT_SECONDS)

    stdout_alive = stdout_reader.is_alive()
    stderr_alive = stderr_reader.is_alive()
    if stdout_alive or stderr_alive:
        logger.error(
            "Output reader thread(s) did not terminate: args=%s stdout_alive=%s stderr_alive=%s",
            command,
            stdout_alive,
            stderr_alive,
        )
        return False

    _close_quietly(popen.stdout)
    _close_quietly(popen.stderr)
    return True


def _kill_process_tree(popen: "subprocess.Popen[bytes]") -> tuple[bool, bool]:
    """Attempt to terminate `popen` and its full process tree.

    Returns (termination_confirmed, child_processes_may_remain). On Windows
    this is `taskkill /PID <pid> /T /F`; taskkill's own exit code is what
    counts as confirmation, never merely that it was invoked without
    raising. On POSIX, os.killpg() targets the whole process group (the
    child was started with start_new_session=True specifically so this
    reaches its descendants too).
    """
    tree_kill_confirmed = False

    if os.name == "nt":
        try:
            taskkill_result = subprocess.run(
                ["taskkill", "/PID", str(popen.pid), "/T", "/F"],
                capture_output=True,
                timeout=_PROCESS_TREE_KILL_TIMEOUT_SECONDS,
                **_WINDOWS_SUBPROCESS_KWARGS,
            )
            tree_kill_confirmed = taskkill_result.returncode == 0
            if not tree_kill_confirmed:
                logger.error(
                    "taskkill reported failure while killing CLI process tree: pid=%s returncode=%s stderr=%s",
                    popen.pid,
                    taskkill_result.returncode,
                    _decode_for_log(taskkill_result.stderr).strip(),
                )
        except (OSError, subprocess.TimeoutExpired):
            logger.exception("taskkill failed while killing CLI process tree: pid=%s", popen.pid)
    else:
        import signal

        try:
            os.killpg(os.getpgid(popen.pid), signal.SIGKILL)
            tree_kill_confirmed = True
        except (ProcessLookupError, PermissionError, OSError):
            logger.exception("killpg failed while killing CLI process group: pid=%s", popen.pid)

    try:
        popen.kill()
    except OSError:
        pass

    return tree_kill_confirmed, not tree_kill_confirmed


def _cwd_exists_now(cwd: Path) -> bool:
    try:
        return Path(cwd).is_dir()
    except OSError:
        return False


_WINERROR_CWD_CODES = frozenset({3, 267})
_WINERROR_EXECUTABLE_NOT_FOUND = 2


def _classify_launch_exception(exc: OSError, cwd: Path) -> str:
    """Distinguish "cwd disappeared between our pre-check and the actual
    Popen() call" from "the CLI executable was not found" - see
    dev_git_runner._classify_launch_exception(), which this mirrors.
    """
    winerror = getattr(exc, "winerror", None)
    if winerror in _WINERROR_CWD_CODES:
        return ERROR_CWD_NOT_FOUND
    if winerror == _WINERROR_EXECUTABLE_NOT_FOUND:
        return ERROR_CLI_NOT_FOUND

    filename = getattr(exc, "filename", None)
    if filename is not None:
        try:
            if Path(filename) == Path(cwd):
                return ERROR_CWD_NOT_FOUND
        except (OSError, ValueError):
            pass

    if not _cwd_exists_now(cwd):
        return ERROR_CWD_NOT_FOUND

    return ERROR_CLI_NOT_FOUND


def _result(
    *,
    success: bool,
    returncode: int | None,
    stdout: bytes = b"",
    stderr: bytes = b"",
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
    timed_out: bool = False,
    cancelled: bool = False,
    termination_confirmed: bool = True,
    child_processes_may_remain: bool = False,
    duration_seconds: float,
    error_code: str | None,
    reader_threads_terminated: bool = True,
    stdin_writer_terminated: bool = True,
) -> CliRunResult:
    return CliRunResult(
        success=success,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        timed_out=timed_out,
        cancelled=cancelled,
        termination_confirmed=termination_confirmed,
        child_processes_may_remain=child_processes_may_remain,
        duration_seconds=duration_seconds,
        error_code=error_code,
        reader_threads_terminated=reader_threads_terminated,
        stdin_writer_terminated=stdin_writer_terminated,
    )


def run_cli(
    args: list[str],
    *,
    cwd: Path,
    timeout: float,
    input_bytes: bytes | None = None,
    max_output_bytes: int,
    cancel_token: CancellationToken | None = None,
    env: dict[str, str] | None = None,
) -> CliRunResult:
    """Run `args` (a fixed argument list - never shell=True, never a
    string-interpolated command line) with cwd fixed to `cwd`, optionally
    feeding `input_bytes` to the child's stdin (on its own writer thread,
    never the calling thread - see _StdinWriter) and then closing it,
    draining stdout/stderr live via capped background reader threads (never
    an unbounded buffer, never a temp file) so the child can neither exhaust
    memory nor deadlock this process by filling an OS pipe buffer nobody
    reads.

    `env`, if given, REPLACES the subprocess's environment entirely (it is
    passed straight to subprocess.Popen(env=...), which never merges it
    with this process's own environment) - callers that need isolation
    from the calling process's environment (e.g. secrets/tokens held by a
    Discord bot process) must build an explicit allowlisted dict
    themselves; `env=None` (the default) preserves the previous behavior of
    inheriting this process's environment, for callers (e.g.
    services.dev_git_runner-style trusted internal tooling) that don't need
    isolation.

    This is a blocking call, intended to be run via asyncio.to_thread() by
    callers. In addition to `timeout`, the wait loop below also polls
    `cancel_token` (if given) roughly every _WAIT_POLL_SECONDS, so a caller
    can request cooperative cancellation of a still-running CLI without
    waiting for the full timeout - both paths kill the entire process tree
    the same way, and both proactively close the child's stdin pipe first
    (see below) so a stdin writer thread blocked on a full, unread pipe is
    unblocked rather than left to leak.
    """
    start = time.monotonic()

    if not _cwd_exists_now(cwd):
        logger.error("CLI command cwd does not exist or is not a directory: cwd=%s", cwd)
        return _result(
            success=False,
            returncode=None,
            duration_seconds=time.monotonic() - start,
            error_code=ERROR_CWD_NOT_FOUND,
        )

    extra_kwargs: dict[str, Any] = dict(_WINDOWS_SUBPROCESS_KWARGS)
    extra_kwargs.update(_POSIX_SUBPROCESS_KWARGS)
    if env is not None:
        extra_kwargs["env"] = env

    try:
        popen: "subprocess.Popen[bytes]" = subprocess.Popen(
            args,
            cwd=str(cwd),
            shell=False,
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **extra_kwargs,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        error_code = _classify_launch_exception(exc, cwd)
        logger.error("CLI launch failed: cwd=%s classified_as=%s", cwd, error_code)
        return _result(
            success=False,
            returncode=None,
            duration_seconds=time.monotonic() - start,
            error_code=error_code,
        )
    except OSError:
        logger.exception("Failed to launch CLI command")
        return _result(
            success=False,
            returncode=None,
            duration_seconds=time.monotonic() - start,
            error_code=ERROR_LAUNCH_FAILED,
        )

    stdout_reader = _CappedReader(popen.stdout, max_output_bytes)
    stderr_reader = _CappedReader(popen.stderr, max_output_bytes)
    stdout_reader.start()
    stderr_reader.start()

    stdin_writer: "_StdinWriter | None" = None
    if input_bytes is not None:
        stdin_writer = _StdinWriter(popen.stdin, input_bytes)
        stdin_writer.start()

    deadline = start + max(0.0, timeout)
    returncode: int | None = None
    timed_out = False
    cancelled = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        if cancel_token is not None and cancel_token.is_cancelled():
            cancelled = True
            break
        try:
            returncode = popen.wait(timeout=min(_WAIT_POLL_SECONDS, remaining))
            break
        except subprocess.TimeoutExpired:
            continue

    if timed_out or cancelled:
        logger.info("CLI command %s: killing process tree", "timed out" if timed_out else "was cancelled")
        # Deliberately do NOT close popen.stdin from this (the main) thread
        # here: if the stdin writer thread is currently blocked inside a
        # syscall-level write() to a full, unread pipe, CPython's buffered
        # IO objects serialize read/write/close via an internal lock, so a
        # close() from another thread would itself block until that write
        # call returns - i.e. exactly the opposite of "unblock it". Instead,
        # killing the whole process tree closes the child's end of the
        # pipe, which is what actually turns the writer's blocked write()
        # into a normal BrokenPipeError (handled by _StdinWriter.run()'s own
        # except clause, which then closes the stream itself) once there are
        # no readers left anywhere.
        termination_confirmed, child_processes_may_remain = _kill_process_tree(popen)
        try:
            popen.wait(timeout=_PROCESS_TREE_KILL_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            logger.error("CLI process tree did not exit after kill: pid=%s", popen.pid)
            termination_confirmed = False
            child_processes_may_remain = True
        stdin_writer_terminated = _join_writer_and_check_alive(stdin_writer)
        reader_threads_terminated = _join_readers_and_check_alive(popen, stdout_reader, stderr_reader, args)
        return _result(
            success=False,
            returncode=None,
            timed_out=timed_out,
            cancelled=cancelled,
            termination_confirmed=termination_confirmed,
            child_processes_may_remain=child_processes_may_remain,
            duration_seconds=time.monotonic() - start,
            error_code=ERROR_TIMEOUT if timed_out else ERROR_CANCELLED,
            reader_threads_terminated=reader_threads_terminated,
            stdin_writer_terminated=stdin_writer_terminated,
        )

    stdin_writer_terminated = _join_writer_and_check_alive(stdin_writer)
    if not stdin_writer_terminated:
        # Never treat a still-writing/stuck stdin thread as a benign detail
        # - the process already exited, so a writer still blocked here
        # means the pipe write genuinely never completed cleanly.
        reader_threads_terminated = _join_readers_and_check_alive(popen, stdout_reader, stderr_reader, args)
        return _result(
            success=False,
            returncode=returncode,
            duration_seconds=time.monotonic() - start,
            error_code=ERROR_STDIN_WRITER_STUCK,
            reader_threads_terminated=reader_threads_terminated,
            stdin_writer_terminated=False,
        )
    stdin_write_failed = stdin_writer.write_failed if stdin_writer is not None else False

    reader_threads_terminated = _join_readers_and_check_alive(popen, stdout_reader, stderr_reader, args)
    if not reader_threads_terminated:
        return _result(
            success=False,
            returncode=returncode,
            duration_seconds=time.monotonic() - start,
            error_code=ERROR_OUTPUT_READER_STUCK,
            reader_threads_terminated=False,
        )

    stdout_bytes, stdout_truncated = stdout_reader.result()
    stderr_bytes, stderr_truncated = stderr_reader.result()

    if stdin_write_failed:
        return _result(
            success=False,
            returncode=returncode,
            stdout=stdout_bytes,
            stderr=stderr_bytes,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            duration_seconds=time.monotonic() - start,
            error_code=ERROR_STDIN_WRITE_FAILED,
        )

    if returncode != 0:
        return _result(
            success=False,
            returncode=returncode,
            stdout=stdout_bytes,
            stderr=stderr_bytes,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            duration_seconds=time.monotonic() - start,
            error_code=ERROR_COMMAND_FAILED,
        )

    return _result(
        success=True,
        returncode=0,
        stdout=stdout_bytes,
        stderr=stderr_bytes,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        duration_seconds=time.monotonic() - start,
        error_code=None,
    )
