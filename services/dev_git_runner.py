from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger("secretary-bot.dev_git_runner")

# subprocess.Popen() only accepts creationflags on Windows; omit them
# entirely elsewhere rather than pass a value the platform doesn't
# understand. CREATE_NEW_PROCESS_GROUP puts the child in its own process
# group so a Ctrl+C delivered to this process's console does not also reach
# git, and so a timed-out process's descendants can be targeted as a group
# by _kill_process_tree() below.
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
# On POSIX, start_new_session=True makes the child (and anything it spawns)
# a new process group leader, so os.killpg() can reach the whole tree
# instead of only the immediate git process.
_POSIX_SUBPROCESS_KWARGS: dict[str, Any] = {} if os.name == "nt" else {"start_new_session": True}

# Cap how much of stdout/stderr this module ever holds - in memory OR on
# disk - per stream. Enforced live by _CappedReader while the process is
# still running, not just after the fact: git is never allowed to write an
# unbounded amount of data anywhere this process is responsible for.
_MAX_CAPTURED_BYTES = 1_000_000  # 1 MiB

_READER_CHUNK_SIZE = 65536

_PROCESS_TREE_KILL_TIMEOUT_SECONDS = 10.0
_READER_JOIN_TIMEOUT_SECONDS = 10.0

# Stable, machine-checkable reasons so callers can branch on why a git
# invocation failed instead of parsing stderr text.
ERROR_GIT_NOT_FOUND = "git_not_found"
ERROR_GIT_TIMEOUT = "git_timeout"
ERROR_GIT_LAUNCH_FAILED = "git_launch_failed"
ERROR_GIT_COMMAND_FAILED = "git_command_failed"
ERROR_CWD_NOT_FOUND = "cwd_not_found"
ERROR_OUTPUT_READER_STUCK = "output_reader_stuck"

# Infra-level failures (git missing / hung / unlaunchable / cwd gone) are
# worth preserving as their own error_code wherever they surface, rather
# than being flattened into a generic "operation failed" code; a git-reported
# semantic failure (non-zero exit that is actually git telling us something,
# e.g. "no such ref") maps to the operation-specific default instead.
INFRA_ERROR_CODES = frozenset(
    {ERROR_GIT_NOT_FOUND, ERROR_GIT_TIMEOUT, ERROR_GIT_LAUNCH_FAILED, ERROR_CWD_NOT_FOUND}
)


def map_git_error(error_code: str | None, *, default: str) -> str:
    if error_code in INFRA_ERROR_CODES:
        return error_code
    return default


@dataclass(frozen=True)
class GitRunResult:
    success: bool
    returncode: int | None
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    error_code: str | None
    # Only meaningful when error_code == ERROR_GIT_TIMEOUT: whether the
    # platform-specific tree-kill mechanism itself confirmed success
    # (taskkill's own exit code 0 on Windows; os.killpg() raising nothing on
    # POSIX) - never merely because the top-level process's .kill() was
    # called, which never reaches descendants. True/False (not applicable)
    # for every other outcome, since no kill was ever attempted.
    termination_confirmed: bool = True
    child_processes_may_remain: bool = False
    # False whenever a stdout/stderr reader thread was still alive after
    # being given every chance to terminate (join + closing the stream to
    # help it unblock). True is the default because most outcomes never
    # even risk this; run_git() sets it explicitly for every path that
    # actually waits on the reader threads.
    reader_threads_terminated: bool = True


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
    that, independent of how much output git produces.
    """

    def __init__(self, stream: Any, cap: int):
        super().__init__(daemon=True)
        self._stream = stream
        self._cap = cap
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


def _join_readers_and_check_alive(
    popen: "subprocess.Popen[bytes]", stdout_reader: "_CappedReader", stderr_reader: "_CappedReader", command: list[str]
) -> bool:
    """Join both reader threads, closing the pipes to help an unresponsive
    reader unblock if the first join doesn't finish them, and report
    whether both are actually terminated afterward - never assumed.

    A reader thread normally exits the instant its pipe hits EOF (i.e. the
    moment the process itself exits), so the first join is expected to
    succeed immediately in the common case. If a reader is still alive
    after that, closing its stream out from under it typically raises
    inside the thread's blocking .read() call (caught by
    _CappedReader.run()'s own except clause), letting it exit its loop -
    this is attempted before giving up. The threads remain daemon=True
    regardless (never converted to non-daemon to force-block on them), but
    that is not treated as "fine" here: is_alive() after every reasonable
    attempt is the actual, honest answer this function returns.
    """
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

    Returns (termination_confirmed, child_processes_may_remain).
    termination_confirmed is True only when the platform-specific tree-kill
    mechanism itself reported success: taskkill's own exit code 0 on
    Windows (not merely that it was invoked without raising), or
    os.killpg() completing without error on POSIX. popen.kill() is always
    attempted as well, but never counts toward confirmation on its own,
    since it only ever targets the immediate git process - a git hook or
    credential helper it spawned would not be affected by it.
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
                    "taskkill reported failure while killing timed-out git process tree: pid=%s "
                    "returncode=%s stderr=%s",
                    popen.pid,
                    taskkill_result.returncode,
                    _decode_for_log(taskkill_result.stderr).strip(),
                )
        except (OSError, subprocess.TimeoutExpired):
            logger.exception("taskkill failed while killing timed-out git process tree: pid=%s", popen.pid)
    else:
        try:
            os.killpg(os.getpgid(popen.pid), signal.SIGKILL)
            tree_kill_confirmed = True
        except (ProcessLookupError, PermissionError, OSError):
            logger.exception("killpg failed while killing timed-out git process group: pid=%s", popen.pid)

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


# Windows error codes that indicate the working directory itself was the
# problem (missing, or no longer a directory), as opposed to the
# executable. ERROR_FILE_NOT_FOUND (2) is the executable-missing case;
# ERROR_PATH_NOT_FOUND (3) and ERROR_DIRECTORY_NOT_FOUND (267) are cwd
# problems.
_WINERROR_CWD_CODES = frozenset({3, 267})
_WINERROR_EXECUTABLE_NOT_FOUND = 2


def _classify_launch_exception(exc: OSError, cwd: Path) -> str:
    """Distinguish "cwd disappeared between our pre-check and the actual
    Popen() call" (a TOCTOU race, not a git problem) from "git executable
    not found" - both can surface from the same subprocess.Popen() call
    (and, on Windows, as different exception types), but must never be
    conflated into the same error_code.

    On Windows, a missing cwd raises NotADirectoryError or
    FileNotFoundError with winerror 267/3, distinct from a missing
    executable's winerror 2 - checked first since it is the least
    ambiguous. On POSIX, both typically raise FileNotFoundError, but
    .filename identifies which path was actually missing. If neither
    signal is conclusive, a final direct re-check of whether cwd currently
    exists is used as the last word.
    """
    winerror = getattr(exc, "winerror", None)
    if winerror in _WINERROR_CWD_CODES:
        return ERROR_CWD_NOT_FOUND
    if winerror == _WINERROR_EXECUTABLE_NOT_FOUND:
        return ERROR_GIT_NOT_FOUND

    filename = getattr(exc, "filename", None)
    if filename is not None:
        try:
            if Path(filename) == Path(cwd):
                return ERROR_CWD_NOT_FOUND
        except (OSError, ValueError):
            pass

    if not _cwd_exists_now(cwd):
        return ERROR_CWD_NOT_FOUND

    return ERROR_GIT_NOT_FOUND


def run_git(args: list[str], *, cwd: Path, timeout: float) -> GitRunResult:
    """Run `git <args>` as a fixed argument list (never shell=True, never
    string-interpolated) with cwd fixed to `cwd`, and translate every
    failure mode into a safe, structured result instead of letting an
    exception escape or leaking unbounded stderr to callers.

    stdout/stderr are returned as raw bytes (not decoded text): callers
    that only need to log or display output decode it themselves (UTF-8,
    errors="replace"), while callers parsing NUL-delimited `--porcelain -z`
    output need the exact, undecoded bytes so a NUL byte is never confused
    with anything a lossy decode step could have altered first.

    Output is captured via subprocess.PIPE, drained live by background
    _CappedReader threads (never an unbounded temp file, never an
    unbounded in-memory buffer) so a git command producing large output
    can neither exhaust memory/disk nor deadlock this process by filling
    the OS pipe buffer while nobody reads it.

    On timeout, the entire process tree spawned by git is killed, not just
    the immediate child, and whether that kill was actually confirmed is
    reported via termination_confirmed/child_processes_may_remain rather
    than assumed. A missing `cwd` - including one that disappears between
    the up-front check and the actual Popen() call - is reported as
    ERROR_CWD_NOT_FOUND, distinct from ERROR_GIT_NOT_FOUND, even though
    both can otherwise surface from the same exception type.
    """
    command = ["git", *args]

    if not _cwd_exists_now(cwd):
        logger.error("Git command cwd does not exist or is not a directory: cwd=%s args=%s", cwd, command)
        return GitRunResult(
            success=False,
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
            error_code=ERROR_CWD_NOT_FOUND,
        )

    extra_kwargs: dict[str, Any] = dict(_WINDOWS_SUBPROCESS_KWARGS)
    extra_kwargs.update(_POSIX_SUBPROCESS_KWARGS)

    try:
        popen: "subprocess.Popen[bytes]" = subprocess.Popen(
            command,
            cwd=str(cwd),
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **extra_kwargs,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        error_code = _classify_launch_exception(exc, cwd)
        logger.error("Git launch failed: args=%s cwd=%s classified_as=%s", command, cwd, error_code)
        return GitRunResult(
            success=False,
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
            error_code=error_code,
        )
    except OSError:
        logger.exception("Failed to launch git command: args=%s", command)
        return GitRunResult(
            success=False,
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
            error_code=ERROR_GIT_LAUNCH_FAILED,
        )

    # Deliberately NOT `with popen:` - Popen.__exit__() calls self.wait()
    # with no timeout at all when no exception is propagating, which would
    # block indefinitely in the rare case where even the tree-kill below
    # failed to actually terminate the process. Pipes are closed manually
    # instead, only after the reader threads are done with them.
    stdout_reader = _CappedReader(popen.stdout, _MAX_CAPTURED_BYTES)
    stderr_reader = _CappedReader(popen.stderr, _MAX_CAPTURED_BYTES)
    stdout_reader.start()
    stderr_reader.start()

    try:
        returncode = popen.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error("Git command timed out: args=%s timeout=%s", command, timeout)
        termination_confirmed, child_processes_may_remain = _kill_process_tree(popen)
        try:
            popen.wait(timeout=_PROCESS_TREE_KILL_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            logger.error("Git process tree did not exit after kill: pid=%s", popen.pid)
            termination_confirmed = False
            child_processes_may_remain = True
        reader_threads_terminated = _join_readers_and_check_alive(popen, stdout_reader, stderr_reader, command)
        return GitRunResult(
            success=False,
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
            error_code=ERROR_GIT_TIMEOUT,
            termination_confirmed=termination_confirmed,
            child_processes_may_remain=child_processes_may_remain,
            reader_threads_terminated=reader_threads_terminated,
        )

    reader_threads_terminated = _join_readers_and_check_alive(popen, stdout_reader, stderr_reader, command)
    if not reader_threads_terminated:
        # A reader thread that never terminates means we cannot trust that
        # stdout/stderr were fully and safely captured (it may still be
        # blocked mid-read, or - worse - still writing into
        # _CappedReader's internal buffer from another thread while this
        # function is about to hand the "final" bytes to a caller). Report
        # this as its own failure rather than returning success=True with
        # output that might still be in flux.
        return GitRunResult(
            success=False,
            returncode=returncode,
            stdout=b"",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
            error_code=ERROR_OUTPUT_READER_STUCK,
            reader_threads_terminated=False,
        )

    stdout_bytes, stdout_truncated = stdout_reader.result()
    stderr_bytes, stderr_truncated = stderr_reader.result()

    if returncode != 0:
        logger.warning(
            "Git command exited non-zero: args=%s returncode=%s stderr=%s",
            command,
            returncode,
            _decode_for_log(stderr_bytes).strip(),
        )
        return GitRunResult(
            success=False,
            returncode=returncode,
            stdout=stdout_bytes,
            stderr=stderr_bytes,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            error_code=ERROR_GIT_COMMAND_FAILED,
        )

    return GitRunResult(
        success=True,
        returncode=0,
        stdout=stdout_bytes,
        stderr=stderr_bytes,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        error_code=None,
    )


@dataclass(frozen=True)
class RefCheckResult:
    # None means "could not determine" (see error_code); otherwise the
    # actual existence of the ref.
    exists: bool | None
    error_code: str | None


def check_ref_exists(ref: str, *, cwd: Path, timeout: float) -> RefCheckResult:
    """Check whether the fully-qualified ref `ref` (e.g. "refs/heads/foo")
    exists, distinguishing git's own documented "not found" signal from any
    other failure.

    `git show-ref --verify` exits exactly 1 when the ref is simply absent -
    that is normal, expected information, not an error. Any other outcome
    (timeout, missing git, missing cwd, or a non-zero exit code other than
    1, which can indicate a corrupted repository) must never be reported as
    "does not exist"; callers get exists=None and an error_code instead so
    they can propagate the failure rather than silently treating unknown
    state as absence.
    """
    result = run_git(["show-ref", "--verify", "--quiet", ref], cwd=cwd, timeout=timeout)
    if result.success:
        return RefCheckResult(exists=True, error_code=None)
    if result.error_code == ERROR_GIT_COMMAND_FAILED and result.returncode == 1:
        return RefCheckResult(exists=False, error_code=None)
    return RefCheckResult(exists=None, error_code=result.error_code or ERROR_GIT_COMMAND_FAILED)
