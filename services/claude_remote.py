from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable


logger = logging.getLogger("secretary-bot.claude_remote")

DEFAULT_SESSION_NAME = "初号ちゃん"
DEFAULT_START_TIMEOUT_SECONDS = 30.0
DEFAULT_STOP_TIMEOUT_SECONDS = 10.0
ANSI_ESCAPE_RE = re.compile(
    r"(?:\x1B[@-_][0-?]*[ -/]*[@-~]|\x1B\][^\x07]*(?:\x07|\x1B\\))"
)
REMOTE_URL_RE = re.compile(r"https://claude\.ai/code/[^\s<>\]\[\"']+")
CMD_META_RE = re.compile(r'["&|<>^%()!\r\n]')

ERROR_DISABLED = "disabled"
ERROR_PROJECT_NOT_FOUND = "project_not_found"
ERROR_COMMAND_NOT_FOUND = "command_not_found"
ERROR_NOT_LOGGED_IN = "not_logged_in"
ERROR_WORKSPACE_UNTRUSTED = "workspace_untrusted"
ERROR_REMOTE_UNAVAILABLE = "remote_unavailable"
ERROR_START_TIMEOUT = "start_timeout"
ERROR_EARLY_EXIT = "early_exit"
ERROR_START_FAILED = "start_failed"
ERROR_STOP_FAILED = "stop_failed"

ERROR_MESSAGES = {
    ERROR_DISABLED: "Claude Code Remote Control機能は無効です。CLAUDE_REMOTE_ENABLED=true に設定してBotを再起動してください。",
    ERROR_PROJECT_NOT_FOUND: "対象プロジェクトフォルダが存在しないため起動できません。.env の CLAUDE_REMOTE_PROJECT_DIR を確認してください。",
    ERROR_COMMAND_NOT_FOUND: "Claude Codeコマンドが見つかりません。Claude Codeのインストールと CLAUDE_REMOTE_COMMAND / PATH を確認してください。",
    ERROR_NOT_LOGGED_IN: "Claude Codeが未ログインのようです。対象PCの対象フォルダで一度Claude Codeを起動し、ログインを完了してください。",
    ERROR_WORKSPACE_UNTRUSTED: "対象フォルダのワークスペース信頼が未承認のようです。対象PCで一度Claude Codeを起動し、信頼を承認してください。",
    ERROR_REMOTE_UNAVAILABLE: "このClaude Code環境ではRemote Controlを利用できないようです。Claude Codeを更新し、契約・機能提供状況を確認してください。",
    ERROR_START_TIMEOUT: "接続URLの取得がタイムアウトしました。プロセスは起動中の可能性があります。少し待って /claude_remote_status または /claude_remote_link を確認してください。",
    ERROR_EARLY_EXIT: "接続URLを取得する前にClaude Code Remote Controlが終了しました。対象PCでClaude Codeを直接起動して状態を確認してください。",
    ERROR_START_FAILED: "Claude Code Remote Controlを起動できませんでした。Botのログと対象PCのClaude Code設定を確認してください。",
    ERROR_STOP_FAILED: "Claude Code Remote Controlを停止できませんでした。対象PCでプロセス状態を確認してください。",
}


class ClaudeRemoteError(RuntimeError):
    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or ERROR_MESSAGES.get(code, ERROR_MESSAGES[ERROR_START_FAILED]))


@dataclass(frozen=True)
class ClaudeRemoteConfig:
    enabled: bool
    command: str
    project_dir: Path | None
    session_name: str
    start_timeout_seconds: float
    stop_timeout_seconds: float


@dataclass(frozen=True)
class ClaudeRemoteStartResult:
    pid: int
    url: str | None
    already_running: bool = False
    timed_out: bool = False


@dataclass(frozen=True)
class ClaudeRemoteStopResult:
    already_stopped: bool
    exit_code: int | None
    used_terminate: bool = False
    used_kill: bool = False


@dataclass(frozen=True)
class ClaudeRemoteStatus:
    running: bool
    pid: int | None
    started_at: datetime | None
    elapsed_seconds: float | None
    project_dir: Path | None
    url_available: bool
    url: str | None
    exit_code: int | None
    last_error: str | None
    stopping: bool


def _parse_positive_seconds(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s; using default.", name)
        return default
    if value <= 0 or value > 3600:
        logger.warning("Out-of-range %s; using default.", name)
        return default
    return value


def load_claude_remote_config() -> ClaudeRemoteConfig:
    enabled = os.getenv("CLAUDE_REMOTE_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    command = os.getenv("CLAUDE_REMOTE_COMMAND", "claude").strip() or "claude"
    project_dir_text = os.getenv("CLAUDE_REMOTE_PROJECT_DIR", "").strip()
    project_dir = Path(project_dir_text) if project_dir_text else None
    session_name = os.getenv("CLAUDE_REMOTE_SESSION_NAME", DEFAULT_SESSION_NAME).strip() or DEFAULT_SESSION_NAME
    return ClaudeRemoteConfig(
        enabled=enabled,
        command=command,
        project_dir=project_dir,
        session_name=session_name,
        start_timeout_seconds=_parse_positive_seconds(
            "CLAUDE_REMOTE_START_TIMEOUT_SECONDS", DEFAULT_START_TIMEOUT_SECONDS
        ),
        stop_timeout_seconds=_parse_positive_seconds(
            "CLAUDE_REMOTE_STOP_TIMEOUT_SECONDS", DEFAULT_STOP_TIMEOUT_SECONDS
        ),
    )


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def extract_remote_url(text: str) -> str | None:
    """Extract a Claude Code URL after removing terminal control sequences."""
    match = REMOTE_URL_RE.search(strip_ansi(text))
    if match is None:
        return None
    return match.group(0).rstrip(".,;:!?)。）、」』")


def summarize_output_error(text: str) -> tuple[str, str] | None:
    """Return a safe error code/message without copying terminal output."""
    normalized = strip_ansi(text).casefold()
    if any(value in normalized for value in ("not logged in", "please log in", "login required", "authentication required")):
        return ERROR_NOT_LOGGED_IN, ERROR_MESSAGES[ERROR_NOT_LOGGED_IN]
    if any(value in normalized for value in ("workspace trust", "trust this folder", "untrusted workspace", "workspace is not trusted")):
        return ERROR_WORKSPACE_UNTRUSTED, ERROR_MESSAGES[ERROR_WORKSPACE_UNTRUSTED]
    if any(
        value in normalized
        for value in (
            "remote control is not available",
            "remote-control is not available",
            "unknown command: remote-control",
            "remote control is not supported",
        )
    ):
        return ERROR_REMOTE_UNAVAILABLE, ERROR_MESSAGES[ERROR_REMOTE_UNAVAILABLE]
    return None


def resolve_claude_command(command: str) -> str | None:
    """Resolve only the administrator-configured executable or PATH command."""
    candidate = command.strip()
    if not candidate:
        return None
    if Path(candidate).is_absolute() or any(separator in candidate for separator in ("/", "\\")):
        path = Path(candidate).expanduser()
        return str(path.resolve()) if path.is_file() else None
    return shutil.which(candidate)


def build_process_command(
    resolved_command: str,
    session_name: str,
    *,
    os_name: str | None = None,
    comspec: str | None = None,
) -> tuple[list[str], int]:
    """Build a fixed argv; no Discord-supplied value is accepted here."""
    platform_name = os.name if os_name is None else os_name
    fixed_args = [resolved_command, "remote-control", "--name", session_name]
    creationflags = 0
    if platform_name == "nt":
        # A separate process group lets the Bot request CTRL_BREAK without
        # opening a new console window. CREATE_NO_WINDOW would prevent that
        # graceful signal from reaching the process group.
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if Path(resolved_command).suffix.casefold() in {".cmd", ".bat"}:
            if CMD_META_RE.search(resolved_command) or CMD_META_RE.search(session_name):
                raise ValueError("Unsafe cmd.exe metacharacter in fixed configuration.")
            command_processor = comspec or os.environ.get("COMSPEC") or shutil.which("cmd.exe") or "cmd.exe"
            fixed_command_line = subprocess.list2cmdline(fixed_args)
            return [command_processor, "/d", "/s", "/c", fixed_command_line], creationflags
    return fixed_args, creationflags


async def terminate_windows_process_tree(pid: int, *, force: bool) -> bool:
    """Terminate only the live subprocess tree owned by the in-memory manager."""
    args = ["taskkill", "/PID", str(pid), "/T"]
    if force:
        args.append("/F")
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return await asyncio.wait_for(process.wait(), timeout=5.0) == 0
    except asyncio.TimeoutError:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        logger.warning("Windows process-tree termination timed out for pid=%s.", pid)
        return False
    except OSError:
        logger.warning("Windows process-tree termination was unavailable for pid=%s.", pid)
        return False


class ClaudeRemoteManager:
    def __init__(
        self,
        *,
        config_provider: Callable[[], ClaudeRemoteConfig] = load_claude_remote_config,
        executable_resolver: Callable[[str], str | None] = resolve_claude_command,
        process_factory: Callable[..., Awaitable[asyncio.subprocess.Process]] = asyncio.create_subprocess_exec,
        windows_tree_terminator: Callable[..., Awaitable[bool]] = terminate_windows_process_tree,
    ) -> None:
        self._config_provider = config_provider
        self._executable_resolver = executable_resolver
        self._process_factory = process_factory
        self._windows_tree_terminator = windows_tree_terminator
        self._lock = asyncio.Lock()
        self._process: asyncio.subprocess.Process | None = None
        self._pid: int | None = None
        self._started_at: datetime | None = None
        self._project_dir: Path | None = None
        self._url: str | None = None
        self._last_error: str | None = None
        self._last_error_code: str | None = None
        self._exit_code: int | None = None
        self._stopping = False
        self._url_event = asyncio.Event()
        self._reader_tasks: list[asyncio.Task[None]] = []
        self._watch_task: asyncio.Task[None] | None = None

    def get_config(self) -> ClaudeRemoteConfig:
        return self._config_provider()

    async def start(self) -> ClaudeRemoteStartResult:
        config = self.get_config()
        if not config.enabled:
            raise ClaudeRemoteError(ERROR_DISABLED)
        if config.project_dir is None or not config.project_dir.is_dir():
            raise ClaudeRemoteError(ERROR_PROJECT_NOT_FOUND)

        resolved_command = self._executable_resolver(config.command)
        if resolved_command is None:
            raise ClaudeRemoteError(ERROR_COMMAND_NOT_FOUND)
        try:
            argv, creationflags = build_process_command(resolved_command, config.session_name)
        except ValueError as exc:
            self._last_error = ERROR_MESSAGES[ERROR_START_FAILED]
            raise ClaudeRemoteError(ERROR_START_FAILED) from exc

        async with self._lock:
            if self._process is not None and self._process.returncode is None:
                return ClaudeRemoteStartResult(
                    pid=self._pid or self._process.pid,
                    url=self._url,
                    already_running=True,
                )
            await self._clear_finished_process_locked()
            self._url_event = asyncio.Event()
            self._url = None
            self._last_error = None
            self._last_error_code = None
            self._exit_code = None
            try:
                process = await self._process_factory(
                    *argv,
                    cwd=str(config.project_dir.resolve()),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    creationflags=creationflags,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                logger.error("Failed to start Claude Remote Control: %s", type(exc).__name__)
                self._last_error = ERROR_MESSAGES[ERROR_START_FAILED]
                raise ClaudeRemoteError(ERROR_START_FAILED) from exc

            self._process = process
            self._pid = process.pid
            self._started_at = datetime.now(timezone.utc)
            self._project_dir = config.project_dir.resolve()
            self._stopping = False
            self._reader_tasks = [
                asyncio.create_task(self._read_stream(process.stdout), name="claude-remote-stdout"),
                asyncio.create_task(self._read_stream(process.stderr), name="claude-remote-stderr"),
            ]
            self._watch_task = asyncio.create_task(self._watch_process(process), name="claude-remote-watch")

        url_wait = asyncio.create_task(self._url_event.wait())
        exit_wait = asyncio.create_task(process.wait())
        try:
            done, _ = await asyncio.wait(
                {url_wait, exit_wait},
                timeout=config.start_timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if url_wait in done and self._url is not None:
                return ClaudeRemoteStartResult(pid=process.pid, url=self._url)
            if exit_wait in done:
                await self._collect_reader_tasks()
                if self._last_error is None:
                    self._last_error = ERROR_MESSAGES[ERROR_EARLY_EXIT]
                error_code = self._last_error_code or ERROR_EARLY_EXIT
                raise ClaudeRemoteError(error_code, self._last_error)
            self._last_error = ERROR_MESSAGES[ERROR_START_TIMEOUT]
            return ClaudeRemoteStartResult(pid=process.pid, url=None, timed_out=True)
        finally:
            for task in (url_wait, exit_wait):
                if not task.done():
                    task.cancel()
            await asyncio.gather(url_wait, exit_wait, return_exceptions=True)

    async def stop(self) -> ClaudeRemoteStopResult:
        config = self.get_config()
        async with self._lock:
            process = self._process
            if process is None or process.returncode is not None:
                if process is not None:
                    self._exit_code = process.returncode
                await self._clear_finished_process_locked(clear_url=True)
                return ClaudeRemoteStopResult(already_stopped=True, exit_code=self._exit_code)
            self._stopping = True
            watch_task = self._watch_task
            self._watch_task = None
            if watch_task is not None and not watch_task.done():
                watch_task.cancel()

        if watch_task is not None:
            await asyncio.gather(watch_task, return_exceptions=True)

        used_terminate = False
        used_kill = False
        try:
            graceful_requested = True
            try:
                self._request_graceful_exit(process)
            except (OSError, ProcessLookupError):
                graceful_requested = False
                logger.warning("Graceful Claude Remote Control stop signal was unavailable; falling back.")
            if not graceful_requested or not await self._wait_for_exit(process, config.stop_timeout_seconds):
                used_terminate = True
                await self._terminate_process(process, force=False)
                if not await self._wait_for_exit(process, max(1.0, min(3.0, config.stop_timeout_seconds))):
                    used_kill = True
                    await self._terminate_process(process, force=True)
                    if not await self._wait_for_exit(process, 2.0):
                        raise ClaudeRemoteError(ERROR_STOP_FAILED)
        except (OSError, ProcessLookupError) as exc:
            if process.returncode is None:
                self._last_error = ERROR_MESSAGES[ERROR_STOP_FAILED]
                raise ClaudeRemoteError(ERROR_STOP_FAILED) from exc
        finally:
            await self._collect_reader_tasks(cancel_pending=True)

        async with self._lock:
            self._exit_code = process.returncode
            self._process = None
            self._pid = None
            self._started_at = None
            self._project_dir = None
            self._url = None
            self._url_event = asyncio.Event()
            self._stopping = False
            self._watch_task = None
        return ClaudeRemoteStopResult(
            already_stopped=False,
            exit_code=self._exit_code,
            used_terminate=used_terminate,
            used_kill=used_kill,
        )

    async def cleanup(self) -> None:
        """Best-effort shutdown for SecretaryClient.close()."""
        try:
            await self.stop()
        except Exception:
            logger.exception("Failed to clean up Claude Remote Control during Bot shutdown.")
            await self._collect_reader_tasks(cancel_pending=True)

    async def status(self) -> ClaudeRemoteStatus:
        configured_project_dir = self.get_config().project_dir
        async with self._lock:
            process = self._process
            running = process is not None and process.returncode is None
            if process is not None and process.returncode is not None:
                self._exit_code = process.returncode
            elapsed = None
            if running and self._started_at is not None:
                elapsed = max(0.0, (datetime.now(timezone.utc) - self._started_at).total_seconds())
            return ClaudeRemoteStatus(
                running=running,
                pid=self._pid if running else None,
                started_at=self._started_at,
                elapsed_seconds=elapsed,
                project_dir=self._project_dir or configured_project_dir,
                url_available=self._url is not None,
                url=self._url,
                exit_code=self._exit_code,
                last_error=self._last_error,
                stopping=self._stopping,
            )

    async def _read_stream(self, stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        try:
            while True:
                raw_line = await stream.readline()
                if not raw_line:
                    break
                line = strip_ansi(raw_line.decode("utf-8", errors="replace")).strip()
                if not line:
                    continue
                url = extract_remote_url(line)
                if url is not None and self._url is None:
                    self._url = url
                    self._url_event.set()
                summary = summarize_output_error(line)
                if summary is not None:
                    self._last_error_code, self._last_error = summary
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Claude Remote Control output reader failed (output omitted).")

    async def _watch_process(self, process: asyncio.subprocess.Process) -> None:
        try:
            exit_code = await process.wait()
            await self._collect_reader_tasks()
            async with self._lock:
                if self._process is process:
                    self._exit_code = exit_code
                    if not self._stopping and self._url is None and self._last_error is None:
                        self._last_error = ERROR_MESSAGES[ERROR_EARLY_EXIT]
        except asyncio.CancelledError:
            raise

    async def _collect_reader_tasks(self, *, cancel_pending: bool = False) -> None:
        tasks = list(self._reader_tasks)
        if not tasks:
            return
        if cancel_pending:
            for task in tasks:
                if not task.done():
                    task.cancel()
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2.0)
        except asyncio.TimeoutError:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self._reader_tasks.clear()

    async def _clear_finished_process_locked(self, *, clear_url: bool = False) -> None:
        process = self._process
        if process is not None and process.returncode is None:
            return
        watch_task = self._watch_task
        self._watch_task = None
        if watch_task is not None and not watch_task.done():
            watch_task.cancel()
            await asyncio.gather(watch_task, return_exceptions=True)
        if process is not None:
            self._exit_code = process.returncode
        await self._collect_reader_tasks(cancel_pending=True)
        self._process = None
        self._pid = None
        self._started_at = None
        self._project_dir = None
        self._stopping = False
        if clear_url:
            self._url = None
            self._url_event = asyncio.Event()

    @staticmethod
    def _request_graceful_exit(process: asyncio.subprocess.Process) -> None:
        if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)

    @staticmethod
    async def _wait_for_exit(process: asyncio.subprocess.Process, timeout: float) -> bool:
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _terminate_process(self, process: asyncio.subprocess.Process, *, force: bool) -> None:
        if os.name == "nt" and await self._windows_tree_terminator(process.pid, force=force):
            return
        if force:
            process.kill()
        else:
            process.terminate()


claude_remote_manager = ClaudeRemoteManager()
