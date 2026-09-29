"""Safe control and URL discovery for the optional local image gallery.

The gallery owns its launch, PID validation, ComfyUI output discovery, and
Tailscale URL detection.  The bot delegates those jobs to the gallery's
existing launcher scripts and treats all review operations as best effort.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from ipaddress import IPv4Address, ip_address
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


logger = logging.getLogger("secretary-bot.gallery_control")

DEFAULT_GALLERY_PROJECT_ROOT = Path(
    r"C:\Users\PC_User\Documents\Codex\2026-08-13\lora-lora-comfyui-lora-git-commit"
)
DEFAULT_GALLERY_PORT = 8765
DEFAULT_START_TIMEOUT_SECONDS = 45
DEFAULT_STOP_TIMEOUT_SECONDS = 20
DEFAULT_READY_POLL_SECONDS = 0.5


class GalleryControlError(RuntimeError):
    """Raised when the optional gallery cannot be inspected or controlled."""


@dataclass(frozen=True)
class GalleryURLs:
    local: str
    tailscale: tuple[str, ...] = ()


@dataclass(frozen=True)
class GalleryStatus:
    running: bool
    pid: int | None
    port: int
    urls: GalleryURLs


@dataclass(frozen=True)
class GalleryStartResult:
    status: GalleryStatus
    already_running: bool


@dataclass(frozen=True)
class GalleryStopResult:
    status: GalleryStatus
    already_stopped: bool


@dataclass(frozen=True)
class GalleryReviewResult:
    status: GalleryStatus
    job: dict[str, object]


def get_gallery_project_root() -> Path:
    raw_value = os.getenv("GALLERY_PROJECT_ROOT", "").strip().strip('"')
    return Path(raw_value) if raw_value else DEFAULT_GALLERY_PROJECT_ROOT


def _gallery_directory() -> Path:
    return get_gallery_project_root() / ".gallery"


def _access_info_path() -> Path:
    return _gallery_directory() / "access_urls.txt"


def get_gallery_images_root() -> Path | None:
    """Return the output root selected by the gallery launcher, if present."""

    path = _gallery_directory() / "images_root.txt"
    try:
        value = path.read_text(encoding="utf-8-sig").strip().strip('"')
    except OSError:
        return None
    if not value:
        return None
    candidate = Path(value).expanduser()
    try:
        return candidate.resolve() if candidate.exists() and candidate.is_dir() else None
    except OSError:
        return None


def gallery_job_url(url: str, job_id: str) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode({'job_id': job_id})}"


def _pid_file_path() -> Path:
    return _gallery_directory() / "gallery_server.pid.json"


def _gallery_port() -> int:
    raw_value = os.getenv("GALLERY_PORT", str(DEFAULT_GALLERY_PORT)).strip()
    try:
        port = int(raw_value)
    except ValueError:
        logger.warning("Invalid GALLERY_PORT=%s; using %s.", raw_value, DEFAULT_GALLERY_PORT)
        return DEFAULT_GALLERY_PORT
    if not 1 <= port <= 65535:
        logger.warning("GALLERY_PORT out of range=%s; using %s.", port, DEFAULT_GALLERY_PORT)
        return DEFAULT_GALLERY_PORT
    return port


def _local_url() -> str:
    return f"http://127.0.0.1:{_gallery_port()}/"


def _read_pid() -> int | None:
    try:
        record = json.loads(_pid_file_path().read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    try:
        pid = int(record.get("pid"))
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _is_tailscale_ipv4(host: str) -> bool:
    try:
        address = ip_address(host)
    except ValueError:
        return False
    return isinstance(address, IPv4Address) and IPv4Address("100.64.0.0") <= address <= IPv4Address("100.127.255.255")


def _is_allowed_gallery_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        if parsed.scheme != "http" or parsed.path not in {"", "/"}:
            return False
        if parsed.port != _gallery_port() or parsed.hostname is None:
            return False
        return parsed.hostname == "127.0.0.1" or _is_tailscale_ipv4(parsed.hostname)
    except ValueError:
        return False


def _normalise_url(value: str) -> str | None:
    candidate = value.strip().rstrip(".,;)")
    if not _is_allowed_gallery_url(candidate):
        return None
    return candidate if candidate.endswith("/") else f"{candidate}/"


def read_gallery_urls() -> GalleryURLs:
    local = _local_url()
    tailscale: list[str] = []
    try:
        lines = _access_info_path().read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        lines = []
    for raw_line in lines:
        marker = raw_line.find("http://")
        if marker < 0:
            continue
        candidate = _normalise_url(raw_line[marker:])
        if candidate is None:
            continue
        host = urlparse(candidate).hostname
        if host == "127.0.0.1":
            local = candidate
        elif host and _is_tailscale_ipv4(host) and candidate not in tailscale:
            tailscale.append(candidate)
    return GalleryURLs(local=local, tailscale=tuple(tailscale))


def _gallery_ready() -> bool:
    request = Request(f"{_local_url()}api/config", method="GET")
    try:
        with urlopen(request, timeout=2) as response:
            return response.status == 200
    except (OSError, HTTPError, URLError):
        return False


def _gallery_api_request(
    path: str,
    method: str = "GET",
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(
        f"{_local_url().rstrip('/')}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=10) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, HTTPError, URLError, json.JSONDecodeError) as exc:
        raise GalleryControlError(f"ギャラリーAPIに接続できません: {exc}") from exc
    if not isinstance(value, dict) or not value.get("ok", False):
        detail = value.get("error") if isinstance(value, dict) else "不正な応答"
        raise GalleryControlError(f"ギャラリーAPIが失敗しました: {detail}")
    return value


def get_gallery_status() -> GalleryStatus:
    return GalleryStatus(
        running=_gallery_ready(),
        pid=_read_pid(),
        port=_gallery_port(),
        urls=read_gallery_urls(),
    )


def get_gallery_ai_status() -> dict[str, object]:
    value = _gallery_api_request("/api/ai/status")
    job = value.get("job")
    return job if isinstance(job, dict) else {"status": "idle"}


def get_gallery_job_summary(job_id: str) -> dict[str, object]:
    value = _gallery_api_request(f"/api/images/summary?{urlencode({'job_id': job_id.strip()})}")
    summary = value.get("summary")
    return summary if isinstance(summary, dict) else {}


def start_gallery_review(
    *,
    reference_name: str = "",
    limit: int = 0,
    force: bool = False,
    job_id: str = "",
    device: str | None = None,
) -> GalleryReviewResult:
    started = start_gallery()
    payload: dict[str, object] = {
        "reference_name": reference_name,
        "limit": limit,
        "force": force,
        "job_id": job_id.strip(),
    }
    if device is not None:
        payload["device"] = device
    job = _gallery_api_request("/api/ai/review", method="POST", payload=payload).get("job")
    if not isinstance(job, dict):
        raise GalleryControlError("AI評価ジョブの応答が不正です。")
    return GalleryReviewResult(status=started.status, job=job)


def _process_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if creationflags:
        kwargs["creationflags"] = creationflags
    return kwargs


def _require_project_file(filename: str) -> Path:
    path = get_gallery_project_root() / filename
    if not path.exists() or not path.is_file():
        raise GalleryControlError(f"ギャラリー用ファイルが見つかりません: {path}")
    return path


def _launch_start_file() -> None:
    start_file = _require_project_file("start_gallery.bat")
    subprocess.Popen(
        ["cmd.exe", "/d", "/c", str(start_file)],
        cwd=str(start_file.parent),
        shell=False,
        **_process_kwargs(),
    )


def _wait_until_ready(timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _gallery_ready():
            return True
        time.sleep(DEFAULT_READY_POLL_SECONDS)
    return _gallery_ready()


def _wait_until_stopped(timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _gallery_ready():
            return True
        time.sleep(DEFAULT_READY_POLL_SECONDS)
    return not _gallery_ready()


def _last_launch_error() -> str:
    try:
        return (_gallery_directory() / "last_launch_error.txt").read_text(encoding="utf-8-sig").strip()[-1000:]
    except OSError:
        return ""


def start_gallery() -> GalleryStartResult:
    """Start through the existing launcher and wait for its health check."""

    already_running = _gallery_ready()
    if already_running:
        try:
            _launch_start_file()
            time.sleep(0.5)
        except (OSError, GalleryControlError):
            logger.warning("Could not refresh gallery access_urls.txt.", exc_info=True)
        return GalleryStartResult(get_gallery_status(), already_running=True)

    try:
        _launch_start_file()
    except (OSError, GalleryControlError) as exc:
        raise GalleryControlError(f"ギャラリー起動処理を開始できませんでした: {exc}") from exc
    timeout = _env_timeout("GALLERY_START_TIMEOUT_SECONDS", DEFAULT_START_TIMEOUT_SECONDS)
    if not _wait_until_ready(timeout):
        detail = _last_launch_error()
        suffix = f"\n{detail}" if detail else ""
        raise GalleryControlError(f"ギャラリーが{timeout}秒以内に起動しませんでした。{suffix}")
    return GalleryStartResult(get_gallery_status(), already_running=False)


def _env_timeout(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except ValueError:
        return default
    return max(1, min(value, 300))


def stop_gallery() -> GalleryStopResult:
    """Use the gallery's own PID-safe PowerShell stop handler."""

    already_stopped = not _gallery_ready()
    stop_script = get_gallery_project_root() / "stop_gallery.ps1"
    stop_batch = get_gallery_project_root() / "stop_gallery.bat"
    if stop_script.exists() and stop_script.is_file():
        command = ["powershell.exe", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(stop_script)]
    elif stop_batch.exists() and stop_batch.is_file():
        command = ["cmd.exe", "/d", "/c", str(stop_batch)]
    else:
        raise GalleryControlError("ギャラリー停止用の既存スクリプトが見つかりません。")

    try:
        completed = subprocess.run(
            command,
            cwd=str(get_gallery_project_root()),
            shell=False,
            check=False,
            timeout=_env_timeout("GALLERY_STOP_TIMEOUT_SECONDS", DEFAULT_STOP_TIMEOUT_SECONDS),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            **({"creationflags": getattr(subprocess, "CREATE_NO_WINDOW")} if getattr(subprocess, "CREATE_NO_WINDOW", 0) else {}),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GalleryControlError(f"ギャラリー停止処理を実行できませんでした: {exc}") from exc

    output = (completed.stdout or "").strip()
    if completed.returncode != 0:
        detail = f"\n{output}" if output else ""
        raise GalleryControlError(f"ギャラリー停止に失敗しました。{detail}")
    timeout = _env_timeout("GALLERY_STOP_TIMEOUT_SECONDS", DEFAULT_STOP_TIMEOUT_SECONDS)
    if not _wait_until_stopped(timeout):
        detail = f"\n{output}" if output else ""
        raise GalleryControlError(f"停止処理後もギャラリーAPIの応答が残っています。{detail}")
    return GalleryStopResult(get_gallery_status(), already_stopped=already_stopped)


def format_gallery_links(status: GalleryStatus, *, started: bool = False) -> str:
    if started:
        heading = "画像ギャラリーを起動しました。"
    elif status.running:
        heading = "画像ギャラリーは起動済みです。"
    else:
        heading = "画像ギャラリーは停止中です。"
    lines = [heading]
    if status.running:
        lines.extend([f"ポート: {status.port}", f"PID: {status.pid or '不明'}", ""])
        if status.urls.tailscale:
            lines.append("Tailscale:")
            lines.extend(f"[ギャラリーを開く]({url})" for url in status.urls.tailscale)
        else:
            lines.append("Tailscale: 未接続またはURL未検出")
        lines.extend(["", f"PC内: [ギャラリーを開く]({status.urls.local})"])
    return "\n".join(lines)
