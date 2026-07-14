import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    requests = None

from services.weather import JST


BOT_STARTED_AT = datetime.now(JST)
PROJECT_ROOT = Path.cwd()
LOG_DIR = PROJECT_ROOT / "logs"
RECENT_FILE_TARGETS = [
    PROJECT_ROOT / "commands",
    PROJECT_ROOT / "services",
    PROJECT_ROOT / "views",
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / ".env.example",
]
SENSITIVE_KEYWORDS = ("TOKEN", "SECRET", "KEY", "PASSWORD", "CREDENTIAL")
LIKELY_REQUIRED_ENV_KEYS = [
    "DISCORD_BOT_TOKEN",
    "DISCORD_GUILD_ID",
]
OPTIONAL_FEATURE_ENV_KEYS = [
    "PC_CONTROL_ALLOWED_USER_IDS",
    "WEATHER_CHANNEL_ID",
    "SD_API_URL",
]


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_health_check_interval_minutes() -> int:
    try:
        value = int(os.getenv("HEALTH_CHECK_INTERVAL_MINUTES", "60"))
    except ValueError:
        return 60
    return max(1, min(value, 1440))


def get_health_check_channel_id() -> int | None:
    raw_value = os.getenv("HEALTH_CHECK_CHANNEL_ID", "").strip()
    if not raw_value:
        return None
    try:
        return int(raw_value)
    except ValueError:
        return None


def is_periodic_health_check_enabled() -> bool:
    return _bool_env("HEALTH_CHECK_ENABLED", False) and get_health_check_channel_id() is not None


def get_uptime_text(now: datetime | None = None) -> str:
    current = now or datetime.now(JST)
    uptime = current - BOT_STARTED_AT
    total_seconds = max(0, int(uptime.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}時間{minutes}分{seconds}秒"
    if minutes:
        return f"{minutes}分{seconds}秒"
    return f"{seconds}秒"


def _read_env_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()

    keys: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key = stripped.split("=", 1)[0].strip()
            if key:
                keys.add(key)
    except OSError:
        return set()
    return keys


def _redact_log_line(line: str) -> str:
    redacted = re.sub(
        r"(?i)(token|secret|api[_-]?key|password|credential)\s*[:=]\s*\S+",
        r"\1=<redacted>",
        line.strip(),
    )
    if len(redacted) > 180:
        return redacted[:177] + "..."
    return redacted


def _format_mtime(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, JST).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return "確認できません"


def _iter_recent_target_files(paths: Iterable[Path]) -> Iterable[Path]:
    ignored_dirs = {"__pycache__", ".git", ".venv", "venv", "data"}
    for path in paths:
        if path.is_file():
            yield path
            continue
        if not path.exists():
            continue
        for child in path.rglob("*"):
            if child.is_dir():
                continue
            if any(part in ignored_dirs for part in child.parts):
                continue
            yield child


def _sanitize_url_label(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return "設定URL"
    return f"{parsed.scheme}://{parsed.netloc}"


def _join_api_path(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def collect_runtime_section() -> list[str]:
    return [
        "【Bot】",
        "状態: 起動中",
        f"起動時刻: {BOT_STARTED_AT.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"稼働時間: {get_uptime_text()}",
        f"Python: {sys.version.split()[0]}",
        f"作業ディレクトリ: {Path.cwd()}",
    ]


def collect_config_section() -> list[str]:
    env_path = PROJECT_ROOT / ".env"
    example_path = PROJECT_ROOT / ".env.example"
    env_keys = _read_env_keys(env_path)
    missing_required = [key for key in LIKELY_REQUIRED_ENV_KEYS if not os.getenv(key)]
    missing_optional = [key for key in OPTIONAL_FEATURE_ENV_KEYS if key not in env_keys and not os.getenv(key)]

    lines = [
        "【設定】",
        f".env: {'あり' if env_path.exists() else 'なし'}",
        f".env.example: {'あり' if example_path.exists() else 'なし'}",
    ]
    if missing_required:
        lines.append("警告: 未設定の可能性があります: " + ", ".join(missing_required))
    if missing_optional:
        lines.append("任意設定の未設定: " + ", ".join(missing_optional))
    lines.append("秘密情報: 値は表示していません")
    return lines


def collect_log_section() -> list[str]:
    if not LOG_DIR.exists():
        return ["【ログ】", "logs/ フォルダはありません。"]

    lines = ["【ログ】"]
    target_files = [LOG_DIR / "error.log", LOG_DIR / "bot.log"]
    found = False
    for path in target_files:
        if not path.exists():
            continue
        found = True
        try:
            log_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
        except OSError:
            lines.append(f"{path.name}: 読み取りできません")
            continue

        error_lines = [line for line in log_lines if re.search(r"(?i)\b(error|exception|traceback|failed)\b", line)]
        lines.append(f"{path.name}: 最終更新 {_format_mtime(path)} / 直近エラー {len(error_lines)}件")
        if error_lines:
            lines.append("直近: " + _redact_log_line(error_lines[-1]))

    if not found:
        lines.append("error.log / bot.log はありません。")
    return lines


def collect_recent_files_section() -> list[str]:
    now = datetime.now(JST)
    since = now - timedelta(hours=24)
    entries: list[tuple[datetime, Path]] = []

    for path in _iter_recent_target_files(RECENT_FILE_TARGETS):
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, JST)
        except OSError:
            continue
        if modified >= since:
            entries.append((modified, path))

    lines = ["【最近更新されたファイル】"]
    if not entries:
        lines.append("直近24時間の変更はありません。")
        return lines

    for modified, path in sorted(entries, reverse=True)[:15]:
        try:
            relative = path.relative_to(PROJECT_ROOT)
        except ValueError:
            relative = path
        lines.append(f"- {relative} ({modified.strftime('%m/%d %H:%M')})")
    return lines


def collect_external_services_section() -> list[str]:
    lines = ["【外部サービス】"]

    if requests is None:
        return lines + ["requests が未導入のため、外部サービス疎通確認はスキップしました。"]

    sd_url = os.getenv("SD_API_URL", "").strip()
    if sd_url:
        try:
            response = requests.get(_join_api_path(sd_url, "/sdapi/v1/progress"), timeout=3)
            if response.ok:
                lines.append(f"Stable Diffusion API: OK ({_sanitize_url_label(sd_url)})")
            else:
                lines.append(f"Stable Diffusion API: 警告 HTTP {response.status_code} ({_sanitize_url_label(sd_url)})")
        except requests.RequestException:
            lines.append(f"Stable Diffusion API: 接続できません ({_sanitize_url_label(sd_url)})")
    else:
        lines.append("Stable Diffusion API: SD_API_URL 未設定のためスキップ")

    llm_url = (
        os.getenv("LMSTUDIO_API_URL", "").strip()
        or os.getenv("LM_STUDIO_API_URL", "").strip()
        or os.getenv("LOCAL_LLM_API_URL", "").strip()
        or os.getenv("OPENAI_COMPATIBLE_API_URL", "").strip()
        or os.getenv("LLM_API_URL", "").strip()
    )
    if llm_url:
        models_url = _join_api_path(llm_url, "/models" if llm_url.rstrip("/").endswith("/v1") else "/v1/models")
        try:
            response = requests.get(models_url, timeout=3)
            if response.ok:
                lines.append(f"ローカルLLM API: OK ({_sanitize_url_label(llm_url)})")
            else:
                lines.append(f"ローカルLLM API: 警告 HTTP {response.status_code} ({_sanitize_url_label(llm_url)})")
        except requests.RequestException:
            lines.append(f"ローカルLLM API: 接続できません ({_sanitize_url_label(llm_url)})")
    else:
        lines.append("ローカルLLM API: URL未設定のためスキップ")

    return lines


def build_health_check_report(max_chars: int = 1900) -> str:
    sections = [
        collect_runtime_section(),
        collect_config_section(),
        collect_log_section(),
        collect_recent_files_section(),
        collect_external_services_section(),
    ]
    report = "\n\n".join("\n".join(section) for section in sections)
    if len(report) <= max_chars:
        return report
    return report[: max_chars - 30] + "\n\n...長いので一部省略したよ。"
