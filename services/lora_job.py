"""Botが開始したLoRA planの非同期実行・進捗・安全停止を管理する。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from services import lora_upload
from services.lora_paths import get_lora_manager_outputs_dir


class LoraJobError(ValueError):
    """利用者に表示できる学習ジョブエラー。"""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _project_root() -> Path:
    configured = os.getenv("LORA_MANAGER_ROOT", "").strip().strip('"')
    if configured:
        return Path(configured).resolve()
    # Preserve the existing output override and its established default.
    return get_lora_manager_outputs_dir().resolve().parent


def _kohya_root() -> Path | None:
    configured = os.getenv("LORA_KOHYA_ROOT", "").strip().strip('"')
    if configured:
        return Path(configured).resolve()
    return None


def _utf8_subprocess_environment() -> dict[str, str]:
    """Use UTF-8 for the bridge and Kohya child processes on Windows too."""

    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def _jobs_root() -> Path:
    configured = os.getenv("LORA_JOB_ROOT", "").strip().strip('"')
    root = Path(configured) if configured else Path(__file__).resolve().parents[1] / "data" / "lora_jobs"
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_job_id(job_id: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise LoraJobError("不正な学習ジョブIDです。")
    return job_id


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = handle.name
        os.replace(temporary, path)
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def _load_job(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LoraJobError("学習ジョブの保存状態を読み込めません。") from exc
    if not isinstance(value, dict):
        raise LoraJobError("学習ジョブの保存状態の形式が不正です。")
    return value


def parse_progress_line(line: str, *, expected_total_steps: int | None = None) -> dict[str, Any]:
    """Parse one kohya/tqdm update without confusing preparation bars for training."""

    # tqdm refreshes a single line with ``\r``.  A pipe capture can therefore
    # deliver several refreshes together; the last matching refresh is the
    # current one.
    segments = re.split(r"[\r\n]+", line)
    result: dict[str, Any] = {}
    if expected_total_steps is None:
        epoch = re.search(r"\bepoch\s*[:=]?\s*(\d+)\s*/\s*(\d+)", line, re.I)
        if epoch:
            result.update(epoch=int(epoch.group(1)), total_epochs=int(epoch.group(2)))

    for segment in segments:
        if not segment.strip():
            continue

        named_step = re.search(r"\b(?:step|steps)\s*[:=]?\s*(\d+)\s*/\s*(\d+)\b", segment, re.I)
        fraction_matches = list(re.finditer(r"\b(\d+)\s*/\s*(\d+)\b", segment))
        if expected_total_steps is not None:
            # The real kohya output is:
            # ``steps: ... 864/960 [..., 2.50s/it, avr_loss=...]``.
            # Requiring the configured total prevents 12/12 image-prep bars
            # and other unrelated tqdm speeds from entering job progress.
            candidates = [
                match
                for match in fraction_matches
                if int(match.group(2)) == expected_total_steps
            ]
            if not candidates:
                continue
            selected = candidates[-1]
        elif named_step:
            selected = named_step
        elif fraction_matches and re.search(r"(?:it/s|s/it|%\|)", segment, re.I):
            selected = fraction_matches[-1]
        else:
            selected = None

        if selected is None:
            continue

        result.update(step=int(selected.group(1)), total_steps=int(selected.group(2)))
        speed = re.search(r"(\d+(?:\.\d+)?)\s*(it/s|steps?/s|s/it)", segment, re.I)
        if speed:
            value = float(speed.group(1))
            result["speed"] = (1.0 / value if speed.group(2).lower() == "s/it" and value else value)
            result["speed_text"] = speed.group(0)
        loss = re.search(
            r"(?:^|[\s,\[])\s*(?:avr_loss|loss|average_loss)\s*[:=]\s*"
            r"(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
            segment,
            re.I,
        )
        if loss:
            result["loss"] = float(loss.group(1))

    if expected_total_steps is None:
        eta = re.search(r"\bETA\s*[:=]?\s*([0-9:]+)", line, re.I)
        if eta:
            result["eta"] = eta.group(1)
    return result


def progress_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    """Return a read-only display snapshot, reconciling stale persisted state."""

    progress = dict(job.get("progress") or {})
    total = progress.get("total_steps")
    if not isinstance(total, int) or total <= 0:
        return progress

    log_file = job.get("log_file")
    if isinstance(log_file, str) and log_file.strip():
        latest: dict[str, Any] = {}
        try:
            with Path(log_file).open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    parsed = parse_progress_line(line, expected_total_steps=total)
                    if parsed:
                        latest.update(parsed)
        except OSError:
            pass
        if latest:
            progress.update(latest)
            progress["training_started"] = True

    step = progress.get("step")
    speed = progress.get("speed")
    if isinstance(step, int):
        progress["percent"] = min(100.0, max(0.0, step * 100.0 / total))
        if step >= total:
            progress["eta_seconds"] = 0
        elif step > 0 and isinstance(speed, (int, float)) and speed > 0:
            progress["eta_seconds"] = (total - step) / speed

    expected_output = job.get("expected_output")
    if (
        job.get("status") == "success"
        and isinstance(expected_output, str)
        and Path(expected_output).is_file()
    ):
        progress.update(step=total, percent=100.0, eta_seconds=0, training_started=True)
    return progress


def _safe_tail(lines: list[str], limit: int = 5) -> str:
    safe = []
    for line in lines[-limit:]:
        # Never include environment-like secrets in a Discord-facing error.
        if re.search(r"(?:token|password|secret|api[_ -]?key)\s*[=:]", line, re.I):
            safe.append("[機密情報を含む可能性がある行を省略]")
        else:
            safe.append(_safe_user_text(line.strip()[:300]))
    return "\n".join(safe)


def _safe_user_text(value: str) -> str:
    value = re.sub(r"[A-Za-z]:\\[^\r\n]+", "[PC側パス省略]", value)
    if re.search(r"(?:token|password|secret|api[_ -]?key)\s*[=:]", value, re.I):
        return "[機密情報を含む可能性がある行を省略]"
    return value


def _validate_embedded_plan(plan: dict[str, Any]) -> None:
    """Block confirmation if a preserved ZIP design was altered afterward."""
    embedded = plan.get("embedded_training")
    settings = plan.get("settings")
    paths = plan.get("paths")
    if not isinstance(embedded, dict) or not isinstance(settings, dict) or not isinstance(paths, dict):
        return
    expected_subsets = settings.get("subsets")
    if not isinstance(expected_subsets, list) or not expected_subsets:
        return
    dataset_path = Path(str(paths.get("dataset_config", "")))
    config_path = Path(str(paths.get("training_config", "")))
    try:
        dataset_value = tomllib.loads(dataset_path.read_text(encoding="utf-8"))
        config_value = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise LoraJobError(f"ZIP由来の学習設計を確認できません。学習開始を止めました: {error}") from error
    datasets = dataset_value.get("datasets")
    actual_subsets = datasets[0].get("subsets") if isinstance(datasets, list) and datasets and isinstance(datasets[0], dict) else None
    if not isinstance(actual_subsets, list) or len(actual_subsets) != len(expected_subsets):
        raise LoraJobError("ZIP由来のsubset設定がplanと一致しません。学習開始を止めました。")
    train_dir = Path(str(paths.get("train_dir", ""))).resolve()
    for expected, actual in zip(expected_subsets, actual_subsets, strict=True):
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            raise LoraJobError("ZIP由来のsubset設定が不正です。学習開始を止めました。")
        expected_dir = (train_dir / str(expected.get("path", ""))).resolve()
        actual_dir = Path(str(actual.get("image_dir", ""))).resolve()
        if actual_dir != expected_dir or int(actual.get("num_repeats", -1)) != int(expected.get("repeats", -2)):
            raise LoraJobError("ZIP由来のsubset別repeatsがplanと一致しません。学習開始を止めました。")
    if int(config_value.get("max_train_epochs", -1)) != int(settings.get("epochs", -2)):
        raise LoraJobError("ZIP由来のepochsがplanと一致しません。学習開始を止めました。")


def _progress_interval() -> float:
    try:
        return max(2.0, float(os.getenv("LORA_JOB_PROGRESS_INTERVAL_SECONDS", "5")))
    except ValueError:
        return 5.0


def _resolve_under(path_value: object, root: Path, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise LoraJobError(f"planの{label}がありません。")
    path = Path(path_value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise LoraJobError(f"planの{label}が管理対象外の場所を指しています。") from exc
    return path


def _bridge_paths() -> tuple[Path, Path, Path, Path | None]:
    root = _project_root()
    python_exe = root / ".venv" / "Scripts" / "python.exe"
    bridge = root / "lora_bot_bridge.py"
    kohya = _kohya_root()
    if not root.is_dir():
        raise LoraJobError("LoRA学習本体のフォルダが見つかりません。LORA_MANAGER_ROOTを確認してください。")
    if not python_exe.is_file() or not bridge.is_file():
        raise LoraJobError("LoRA学習本体の固定Pythonまたは連携ブリッジが見つかりません。")
    return root, python_exe, bridge, kohya


def validate_configuration() -> dict[str, str]:
    """起動時にLoRA連携の固定実行経路を読み取り検証する。"""
    root, manager_python, bridge, kohya = _bridge_paths()
    if kohya is None:
        raise LoraJobError("LORA_KOHYA_ROOTが設定されていません。")
    if not kohya.is_dir():
        raise LoraJobError("LORA_KOHYA_ROOTのフォルダが見つかりません。")

    kohya_python = kohya / ".venv" / "Scripts" / "python.exe"
    if not kohya_python.is_file():
        raise LoraJobError("kohya専用Pythonが見つかりません。")

    script = next(
        (
            candidate
            for candidate in (
                kohya / "sd-scripts" / "sdxl_train_network.py",
                kohya / "sd-scripts" / "train_network.py",
            )
            if candidate.is_file()
        ),
        None,
    )
    if script is None:
        raise LoraJobError("sd-scriptsのsdxl_train_network.pyまたはtrain_network.pyが見つかりません。")
    return {
        "manager_root": str(root),
        "manager_python": str(manager_python),
        "bridge": str(bridge),
        "kohya_root": str(kohya),
        "kohya_python": str(kohya_python),
        "train_script": str(script),
    }


async def _run_short_bridge(command: list[str], *, cwd: Path) -> tuple[int, str, str, dict[str, Any] | None]:
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        env=_utf8_subprocess_environment(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def collect(stream: asyncio.StreamReader) -> list[str]:
        lines: list[str] = []
        while True:
            raw = await stream.readline()
            if not raw:
                break
            lines.append(raw.decode("utf-8", errors="replace").rstrip())
        return lines

    stdout_lines, stderr_lines = await asyncio.gather(collect(process.stdout), collect(process.stderr))
    return_code = await process.wait()
    payload = None
    for line in reversed(stdout_lines):
        if line.startswith("LORA_BRIDGE_RESULT:"):
            try:
                value = json.loads(line.split(":", 1)[1])
                if isinstance(value, dict):
                    payload = value
            except json.JSONDecodeError:
                pass
            break
    return return_code, "\n".join(stdout_lines), "\n".join(stderr_lines), payload


class LoraJobManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._starting = False
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._active_jobs: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._notify: Callable[[dict[str, Any], str], Awaitable[None]] | None = None

    def set_notifier(self, callback: Callable[[dict[str, Any], str], Awaitable[None]] | None) -> None:
        self._notify = callback

    def _job_path(self, job_id: str) -> Path:
        return _jobs_root() / f"{_safe_job_id(job_id)}.json"

    def _save(self, job: dict[str, Any]) -> None:
        _atomic_json(self._job_path(str(job["job_id"])), job)

    def update_notification_target(self, job_id: str, *, channel_id: int, message_id: int) -> dict[str, Any]:
        job = self.get(job_id)
        job["channel_id"] = int(channel_id)
        job["message_id"] = int(message_id)
        self._save(job)
        return job

    def get(self, job_id: str) -> dict[str, Any]:
        path = self._job_path(job_id)
        if not path.is_file():
            raise LoraJobError("学習ジョブが見つかりません。")
        return _load_job(path)

    def list_for_owner(self, owner_id: int, *, include_finished: bool = True) -> list[dict[str, Any]]:
        result = []
        for path in _jobs_root().glob("*.json"):
            try:
                job = _load_job(path)
            except LoraJobError:
                continue
            if job.get("owner_id") != int(owner_id):
                continue
            if not include_finished and job.get("status") in {"success", "failed", "stopped"}:
                continue
            result.append(job)
        return sorted(result, key=lambda item: str(item.get("created_at", "")), reverse=True)

    def plans_for_owner(self, owner_id: int) -> list[dict[str, Any]]:
        result = []
        for session in lora_upload.list_sessions(owner_id, include_closed=True):
            plan = session.get("plan")
            if session.get("status") == "plan_created" and isinstance(plan, dict):
                item = dict(session)
                item["plan"] = dict(plan)
                result.append(item)
        return result

    def has_running_job(self) -> dict[str, Any] | None:
        for path in _jobs_root().glob("*.json"):
            try:
                job = _load_job(path)
            except LoraJobError:
                continue
            if job.get("status") in {"starting", "running"}:
                return job
        return None

    async def create_plan(self, session_id: str, owner_id: int) -> dict[str, Any]:
        session = lora_upload.get_session(session_id)
        if session.get("owner_id") != int(owner_id):
            raise LoraJobError("このアップロードセッションを操作する権限がありません。")
        root, python_exe, bridge, kohya = _bridge_paths()
        session, staging = lora_upload.finalize_to_staging(session_id)
        command = [
            str(python_exe), str(bridge), "prepare-dataset",
            "--project-root", str(root), "--staging-dir", str(staging),
            "--dataset-name", str(session["dataset_name"]),
            "--lora-type", str(session["lora_type"]),
            "--trigger-tag", str(session.get("trigger_tag", "")),
            "--recommended-strength", str(session["recommended_strength"]),
        ]
        if kohya is not None:
            command.extend(("--kohya-root", str(kohya)))
        return_code, _stdout, stderr, payload = await _run_short_bridge(command, cwd=root)
        if return_code != 0 or not payload or payload.get("ok") is not True:
            message = (payload or {}).get("error") if isinstance(payload, dict) else None
            raise LoraJobError(_safe_user_text(str(message or stderr or "学習planの作成に失敗しました。"))[:800])
        result = dict(payload)
        lora_upload.mark_plan_created(session_id, result)
        return result

    def plan_preview(self, session_id: str, owner_id: int) -> dict[str, Any]:
        session = lora_upload.get_session(session_id)
        if session.get("owner_id") != int(owner_id):
            raise LoraJobError("この学習planを操作する権限がありません。")
        plan_value = session.get("plan")
        if not isinstance(plan_value, dict):
            raise LoraJobError("このセッションには確定済みの学習planがありません。")
        root = _project_root()
        info_path = _resolve_under(plan_value.get("info_path"), root, "training_info.json")
        try:
            plan = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise LoraJobError("学習planを読み込めません。") from exc
        if not isinstance(plan, dict) or not isinstance(plan.get("paths"), dict):
            raise LoraJobError("学習planの形式が不正です。")
        _validate_embedded_plan(plan)
        plan["info_path"] = str(info_path)
        return plan

    async def start(self, session_id: str, owner_id: int, channel_id: int | None = None, message_id: int | None = None) -> dict[str, Any]:
        with self._lock:
            if self._starting:
                raise LoraJobError("別の学習開始処理が進行中です。少し待ってから再試行してください。")
            if self.has_running_job():
                running = self.has_running_job()
                raise LoraJobError(f"別の学習が実行中です（{running.get('dataset_name', '不明')}）。同時実行はできません。")
            self._starting = True
        try:
            return await self._start_reserved(session_id, owner_id, channel_id, message_id)
        finally:
            with self._lock:
                self._starting = False

    async def _start_reserved(self, session_id: str, owner_id: int, channel_id: int | None = None, message_id: int | None = None) -> dict[str, Any]:
        running = self.has_running_job()
        if running:
            raise LoraJobError(f"別の学習が実行中です（{running.get('dataset_name', '不明')}）。同時実行はできません。")
        plan = self.plan_preview(session_id, owner_id)
        root, python_exe, bridge, _kohya = _bridge_paths()
        info_path = _resolve_under(plan.get("info_path"), root, "training_info.json")
        expected_output = _resolve_under(plan.get("paths", {}).get("expected_output"), root, "出力ファイル")
        job_id = uuid.uuid4().hex
        job = {
            "schema_version": 1,
            "job_id": job_id,
            "owner_id": int(owner_id),
            "channel_id": channel_id,
            "message_id": message_id,
            "session_id": session_id,
            "dataset_name": plan.get("dataset_name"),
            "dataset_dir": plan.get("paths", {}).get("dataset_dir"),
            "output_name": plan.get("output_name"),
            "lora_type": plan.get("lora_metadata", {}).get("lora_type", plan.get("settings", {}).get("lora_type")),
            "plan_info": str(info_path),
            "expected_output": str(expected_output),
            "status": "starting",
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "started_by_bot": True,
            "pid": None,
            "process_created_at": None,
            "command_fingerprint": None,
            "progress": {"step": 0, "total_steps": plan.get("settings", {}).get("total_steps"), "epoch": None, "speed": None, "eta": None, "loss": None},
            "cancel_requested": False,
            "error_summary": None,
            "log_file": str(_jobs_root() / f"{job_id}.log"),
        }
        command = [str(python_exe), str(bridge), "run-plan", "--info-path", str(info_path)]
        job["command_fingerprint"] = hashlib.sha256("\0".join(command).encode("utf-8")).hexdigest()
        self._save(job)
        log_path = Path(job["log_file"])
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(root),
                env=_utf8_subprocess_environment(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, asyncio.SubprocessError) as exc:
            job["status"] = "failed"
            job["finished_at"] = _now()
            job["error_summary"] = "学習プロセスを起動できませんでした。"
            self._save(job)
            raise LoraJobError("学習プロセスを起動できませんでした。固定設定とLoRA本体を確認してください。") from exc
        job["pid"] = process.pid
        job["process_created_at"] = _now()
        job["started_at"] = _now()
        job["status"] = "running"
        self._save(job)
        self._processes[job_id] = process
        self._active_jobs[job_id] = job
        self._tasks[job_id] = asyncio.create_task(self._monitor(job, process, log_path))
        return job

    async def _monitor(self, job: dict[str, Any], process: asyncio.subprocess.Process, log_path: Path) -> None:
        job_id = str(job["job_id"])
        lines: list[str] = []
        marker: dict[str, Any] | None = None
        last_saved = 0.0
        progress_interval = _progress_interval()
        loop = asyncio.get_running_loop()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as log:
            async def consume(stream: asyncio.StreamReader, label: str) -> None:
                nonlocal marker, last_saved
                while True:
                    raw = await stream.readline()
                    if not raw:
                        break
                    text = raw.decode("utf-8", errors="replace").rstrip()
                    log.write(f"[{label}] {text}\n")
                    log.flush()
                    lines.append(text)
                    if len(lines) > 100:
                        del lines[:-100]
                    if text.startswith("LORA_BRIDGE_RESULT:"):
                        try:
                            parsed = json.loads(text.split(":", 1)[1])
                            if isinstance(parsed, dict):
                                marker = parsed
                        except json.JSONDecodeError:
                            pass
                    total_steps = job["progress"].get("total_steps")
                    expected_total_steps = total_steps if isinstance(total_steps, int) and total_steps > 0 else None
                    values = parse_progress_line(text, expected_total_steps=expected_total_steps)
                    if values:
                        progress = job["progress"]
                        progress["training_started"] = True
                        progress.update(values)
                        total = progress.get("total_steps")
                        step = progress.get("step")
                        if isinstance(total, int) and total > 0 and isinstance(step, int):
                            progress["percent"] = min(100.0, max(0.0, step * 100.0 / total))
                            speed = progress.get("speed")
                            if step >= total:
                                progress["eta_seconds"] = 0
                            elif step > 0 and isinstance(speed, (int, float)) and speed > 0:
                                progress["eta_seconds"] = (total - step) / speed
                        if loop.time() - last_saved >= progress_interval:
                            self._save(job)
                            last_saved = loop.time()
                            if self._notify is not None and job.get("message_id"):
                                try:
                                    await self._notify(job, "progress")
                                except Exception:
                                    pass
            await asyncio.gather(consume(process.stdout, "stdout"), consume(process.stderr, "stderr"))
            return_code = await process.wait()
        job["finished_at"] = _now()
        job["return_code"] = return_code
        output = Path(str(job["expected_output"]))
        if job.get("cancel_requested"):
            job["status"] = "stopped"
            job["error_summary"] = "利用者の確認により学習を停止しました。"
        elif return_code == 0 and output.is_file():
            job["status"] = "success"
            job["error_summary"] = None
            total = job["progress"].get("total_steps")
            if isinstance(total, int) and total > 0:
                job["progress"].update(
                    step=total,
                    percent=100.0,
                    eta_seconds=0,
                    training_started=True,
                )
        else:
            job["status"] = "failed"
            job["error_summary"] = _safe_tail(lines) or f"学習プロセスが終了コード {return_code} で終了しました。"
        if marker and marker.get("history_path"):
            job["history_path"] = marker["history_path"]
        self._save(job)
        self._processes.pop(job_id, None)
        self._active_jobs.pop(job_id, None)
        self._tasks.pop(job_id, None)
        if self._notify is not None:
            try:
                await self._notify(job, str(job["status"]))
            except Exception:
                # Discord通知の失敗で、学習結果の永続状態を壊さない。
                pass

    async def cancel(self, job_id: str, owner_id: int) -> dict[str, Any]:
        job = self.get(job_id)
        if job.get("owner_id") != int(owner_id):
            raise LoraJobError("この学習ジョブを停止する権限がありません。")
        if not job.get("started_by_bot") or job_id not in self._processes:
            raise LoraJobError("Botが現在管理している学習だけ停止できます。Bot再起動前の学習はPC側で確認してください。")
        process = self._processes[job_id]
        active_job = self._active_jobs.get(job_id, job)
        active_job["cancel_requested"] = True
        self._save(active_job)
        if os.name == "nt" and process.pid:
            stopper = await asyncio.create_subprocess_exec("taskkill", "/PID", str(process.pid), "/T", "/F", stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            await stopper.wait()
        else:
            process.terminate()
        return active_job

    @staticmethod
    def _process_matches(pid: int, bridge: Path, info_path: Path, recorded_created_at: object = None) -> bool:
        if pid <= 0 or os.name != "nt":
            return False
        # The PID is not trusted alone: compare the command line to both fixed
        # bridge and plan paths. The interpolated value is an integer only.
        script = (
            f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId = {int(pid)}'; "
            "if ($p) { $p | Select-Object CommandLine,CreationDate | ConvertTo-Json -Compress }"
        )
        try:
            completed = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, text=True, timeout=5, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        try:
            identity = json.loads(completed.stdout)
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(identity, dict):
            return False
        command_line = str(identity.get("CommandLine", "")).casefold()
        if str(bridge).casefold() not in command_line or str(info_path).casefold() not in command_line:
            return False
        creation_text = str(identity.get("CreationDate", ""))
        if not isinstance(recorded_created_at, str) or not creation_text:
            return False
        try:
            recorded = datetime.fromisoformat(recorded_created_at).timestamp()
            # WMI DMTF is local time with a trailing timezone offset. Comparing
            # the first 14 digits is sufficient for this short launch window;
            # failure to parse remains a safe non-match.
            created = datetime.strptime(creation_text[:14], "%Y%m%d%H%M%S").astimezone().timestamp()
        except (TypeError, ValueError, OSError):
            return False
        return abs(created - recorded) <= 30

    async def recover(self) -> list[dict[str, Any]]:
        """Bot再起動時に照合だけ行い、外部プロセスを停止しない。"""
        root = _project_root()
        bridge = root / "lora_bot_bridge.py"
        changed: list[dict[str, Any]] = []
        for path in _jobs_root().glob("*.json"):
            try:
                job = _load_job(path)
            except LoraJobError:
                continue
            if str(job.get("job_id", "")) in self._processes:
                continue
            if job.get("status") not in {"starting", "running"} or not isinstance(job.get("pid"), int):
                continue
            info = Path(str(job.get("plan_info", ""))).resolve()
            if self._process_matches(int(job["pid"]), bridge, info, job.get("process_created_at")):
                job["status"] = "running_unattached"
                job["error_summary"] = "Bot再起動後に実行中プロセスを照合しました。自動停止は行いません。"
            else:
                job["status"] = "orphaned"
                job["error_summary"] = "Bot再起動後に学習プロセスを安全に再接続できませんでした。PC側を確認してください。"
            self._save(job)
            changed.append(job)
        return changed

    async def shutdown(self) -> None:
        # Deliberately do not terminate children. A Bot shutdown must not
        # silently stop or damage a training started by the Bot.
        for job_id, task in list(self._tasks.items()):
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                job = self.get(job_id)
                if job.get("status") == "running":
                    job["status"] = "running_unattached"
                    job["error_summary"] = "Bot終了のため監視を解除しました。学習プロセスは停止していません。"
                    self._save(job)
            except Exception:
                pass


job_manager = LoraJobManager()
