import csv
import logging
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from services.weather import JST
from services.runtime_paths import project_path, resolve_project_path


logger = logging.getLogger("secretary-bot.image_selection")

DATA_DIR = project_path("data")
BATCH_DIR = DATA_DIR / "sd_selection_batches"
SELECTION_DIAGNOSTIC_DIR = project_path(".generation_backend/logs/selection")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SCORE_TIMEOUT_SECONDS = 900
DEFAULT_SELECT_GENERATE_COUNT = 5
DEFAULT_SELECT_RETURN_COUNT = 1


@dataclass(frozen=True)
class ImageSelectionResult:
    selected_paths: list[Path]
    used_selection: bool
    message: str = ""
    error: str = ""
    # Best-effort scores produced by the external selector.  The selection
    # algorithm remains owned by that tool; this data only lets delivery keep
    # the remaining generated candidates in ranking order.
    candidate_scores: dict[Path, float] = field(default_factory=dict)


def _local_ai_project_dir() -> Path | None:
    raw_value = os.getenv("LOCAL_AI_PROJECT_DIR", "").strip().strip('"')
    if not raw_value:
        return None
    return resolve_project_path(raw_value, raw_value)


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%s. Using default %s.", name, raw_value, default)
        value = default
    return max(minimum, min(value, maximum))


def get_selection_generate_count() -> int:
    return _env_int("SD_SELECT_GENERATE_COUNT", DEFAULT_SELECT_GENERATE_COUNT, 5, 20)


def get_selection_return_count() -> int:
    return _env_int("SD_SELECT_RETURN_COUNT", DEFAULT_SELECT_RETURN_COUNT, 1, 3)


def is_image_selection_available() -> bool:
    project_dir = _local_ai_project_dir()
    if project_dir is None or not project_dir.exists() or not project_dir.is_dir():
        return False
    script_path = project_dir / "scripts" / "score_batch_top5.py"
    return script_path.exists() and script_path.is_file()


def _image_paths(paths: list[Path]) -> list[Path]:
    image_paths = []
    for raw_path in paths:
        path = Path(raw_path)
        logger.info(
            "Image selection source candidate: path=%s exists=%s is_file=%s suffix=%s",
            path,
            path.exists(),
            path.is_file(),
            path.suffix.lower(),
        )
        if path.exists() and path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path.resolve())
    logger.info("Image selection source images: count=%s", len(image_paths))
    return image_paths


def _prepare_batch_dir(paths: list[Path]) -> Path:
    timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
    batch_dir = (BATCH_DIR / f"{timestamp}_{uuid.uuid4().hex[:8]}").resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Image selection batch folder created: path=%s exists=%s", batch_dir, batch_dir.exists())

    for index, path in enumerate(paths, start=1):
        destination = batch_dir / f"{index:02d}_{path.name}"
        shutil.copy2(path, destination)
        logger.info("Image selection copied image: source=%s destination=%s", path, destination)

    copied_images = _image_paths(list(batch_dir.iterdir()) if batch_dir.exists() else [])
    logger.info("Image selection batch folder image count after copy: path=%s count=%s", batch_dir, len(copied_images))
    return batch_dir


def _latest_top5_dir(project_dir: Path) -> Path:
    return project_dir / "outputs" / "auto_selected" / "latest_top5"


def _load_selected_paths(project_dir: Path, return_count: int | None = None) -> list[Path]:
    target_dir = _latest_top5_dir(project_dir)
    if not target_dir.exists() or not target_dir.is_dir():
        return []

    limit = get_selection_return_count() if return_count is None else max(1, min(int(return_count), 4))
    return sorted(
        path
        for path in target_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )[:limit]


def _read_candidate_scores(project_dir: Path, source_paths: list[Path]) -> dict[Path, float]:
    """Read the selector's existing scores.csv without making it required."""

    try:
        csv_path = _latest_top5_dir(project_dir) / "scores.csv"
        if not csv_path.exists():
            return {}
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows_by_filename = {
                row["filename"]: row
                for row in csv.DictReader(handle)
                if "filename" in row
            }
    except Exception:
        logger.warning("Could not read selection score data; continuing without it.", exc_info=True)
        return {}

    scores: dict[Path, float] = {}
    for index, source_path in enumerate(source_paths, start=1):
        row = rows_by_filename.get(f"{index:02d}_{source_path.name}")
        if row is None:
            continue
        try:
            scores[source_path] = float(row["score"])
        except (KeyError, TypeError, ValueError):
            continue
    return scores


def _selector_environment() -> dict[str, str]:
    """Run the local selector without making unattended jobs network-bound."""

    environment = os.environ.copy()
    # The CLIP model is a local runtime dependency.  In the old mode each
    # subprocess performed a Hugging Face HEAD request even when its files
    # were already cached, so a transient DNS/TLS failure could fail one
    # round.  Offline flags make cache state deterministic and fail fast with
    # a useful diagnostic if the cache is genuinely incomplete.
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["HF_DATASETS_OFFLINE"] = "1"
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return environment


def _write_selection_diagnostic(
    *,
    context: str,
    command: list[str],
    cwd: Path,
    batch_dir: Path,
    returncode: int | str,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
) -> Path | None:
    """Persist bounded subprocess evidence for unattended selection runs."""

    try:
        SELECTION_DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        path = SELECTION_DIAGNOSTIC_DIR / f"{timestamp}_{uuid.uuid4().hex[:8]}.log"
        safe_context = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(context))[:80] or "selection"
        lines = [
            f"context={safe_context}",
            f"command={command!r}",
            f"cwd={cwd}",
            f"batch_dir={batch_dir}",
            f"returncode={returncode}",
            f"timed_out={str(bool(timed_out)).lower()}",
            "--- stdout ---",
            (stdout or "")[-12000:],
            "--- stderr ---",
            (stderr or "")[-12000:],
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
    except OSError:
        logger.warning("Could not write selection diagnostic log.", exc_info=True)
        return None


def select_generated_images(
    paths: list[Path],
    *,
    return_count: int | None = None,
    diagnostic_context: str = "generation",
) -> ImageSelectionResult:
    logger.info("Image selection requested: raw_path_count=%s", len(paths))
    source_paths = _image_paths(paths)
    if not source_paths:
        logger.warning("Image selection skipped: no source images.")
        return ImageSelectionResult([], False, error="選別対象の画像がありません。")

    project_dir = _local_ai_project_dir()
    if project_dir is None:
        return ImageSelectionResult(
            source_paths,
            False,
            message="LOCAL_AI_PROJECT_DIR が未設定のため、画像選別AIはスキップしました。",
        )

    if not project_dir.exists() or not project_dir.is_dir():
        return ImageSelectionResult(
            source_paths,
            False,
            error=f"LOCAL_AI_PROJECT_DIR が見つかりません: {project_dir}",
        )

    script_path = project_dir / "scripts" / "score_batch_top5.py"
    if not script_path.exists() or not script_path.is_file():
        return ImageSelectionResult(
            source_paths,
            False,
            error=f"score_batch_top5.py が見つかりません: {script_path}",
        )

    batch_dir = _prepare_batch_dir(source_paths)
    if not batch_dir.exists() or not batch_dir.is_dir():
        logger.error("Image selection batch folder missing before score: path=%s", batch_dir)
        return ImageSelectionResult(source_paths, False, error=f"画像選別用フォルダが見つかりません: {batch_dir}")

    batch_images = _image_paths(list(batch_dir.iterdir()))
    if not batch_images:
        logger.warning("Image selection skipped: batch folder has no images. path=%s", batch_dir)
        return ImageSelectionResult(source_paths, False, error="画像選別用フォルダに画像がありません。")

    logger.info(
        "Running local image selector: project=%s script=%s batch_dir=%s batch_exists=%s batch_image_count=%s cwd=%s",
        project_dir,
        script_path,
        batch_dir,
        batch_dir.exists(),
        len(batch_images),
        project_dir,
    )

    command = [sys.executable, str(script_path), str(batch_dir)]
    try:
        completed = subprocess.run(
            command,
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=SCORE_TIMEOUT_SECONDS,
            shell=False,
            env=_selector_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        _write_selection_diagnostic(
            context=diagnostic_context,
            command=command,
            cwd=project_dir,
            batch_dir=batch_dir,
            returncode="timeout",
            stdout=str(getattr(exc, "stdout", "") or ""),
            stderr=str(getattr(exc, "stderr", "") or ""),
            timed_out=True,
        )
        logger.exception("Image selection timed out.")
        return ImageSelectionResult(source_paths, False, error="画像選別AIがタイムアウトしました。")
    except Exception as exc:
        _write_selection_diagnostic(
            context=diagnostic_context,
            command=command,
            cwd=project_dir,
            batch_dir=batch_dir,
            returncode="spawn_error",
            stderr=str(exc),
        )
        logger.exception("Failed to run image selection.")
        return ImageSelectionResult(source_paths, False, error="画像選別AIの実行に失敗しました。")

    diagnostic_path = _write_selection_diagnostic(
        context=diagnostic_context,
        command=command,
        cwd=project_dir,
        batch_dir=batch_dir,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if completed.returncode != 0:
        logger.error(
            "Image selection failed: returncode=%s stdout=%s stderr=%s",
            completed.returncode,
            completed.stdout[-2000:],
            completed.stderr[-2000:],
        )
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        reason = detail[-1] if detail else f"exit code {completed.returncode}"
        if diagnostic_path is not None:
            logger.error("Image selection diagnostic saved: path=%s", diagnostic_path)
        return ImageSelectionResult(source_paths, False, error=f"画像選別AIに失敗しました: {reason[:180]}")

    selected_paths = _load_selected_paths(project_dir, return_count=return_count)
    if not selected_paths:
        logger.error("Image selection completed but no selected images were found.")
        return ImageSelectionResult(
            source_paths,
            False,
            error="画像選別AIは完了しましたが、上位画像が見つかりませんでした。",
        )

    candidate_scores = _read_candidate_scores(project_dir, source_paths)
    return ImageSelectionResult(
        selected_paths,
        True,
        message=f"画像選別AIで上位{len(selected_paths)}枚を選びました。",
        candidate_scores=candidate_scores,
    )
