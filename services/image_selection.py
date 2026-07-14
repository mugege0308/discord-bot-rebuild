import logging
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from services.weather import JST


logger = logging.getLogger("secretary-bot.image_selection")

DATA_DIR = Path("data")
BATCH_DIR = DATA_DIR / "sd_selection_batches"
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


def _local_ai_project_dir() -> Path | None:
    raw_value = os.getenv("LOCAL_AI_PROJECT_DIR", "").strip().strip('"')
    if not raw_value:
        return None
    return Path(raw_value)


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


def _load_selected_paths(project_dir: Path) -> list[Path]:
    target_dir = _latest_top5_dir(project_dir)
    if not target_dir.exists() or not target_dir.is_dir():
        return []

    return sorted(
        path
        for path in target_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )[: get_selection_return_count()]


def select_generated_images(paths: list[Path]) -> ImageSelectionResult:
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

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path), str(batch_dir)],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=SCORE_TIMEOUT_SECONDS,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        logger.exception("Image selection timed out.")
        return ImageSelectionResult(source_paths, False, error="画像選別AIがタイムアウトしました。")
    except Exception:
        logger.exception("Failed to run image selection.")
        return ImageSelectionResult(source_paths, False, error="画像選別AIの実行に失敗しました。")

    if completed.returncode != 0:
        logger.error(
            "Image selection failed: returncode=%s stdout=%s stderr=%s",
            completed.returncode,
            completed.stdout[-2000:],
            completed.stderr[-2000:],
        )
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        reason = detail[-1] if detail else f"exit code {completed.returncode}"
        return ImageSelectionResult(source_paths, False, error=f"画像選別AIに失敗しました: {reason[:180]}")

    selected_paths = _load_selected_paths(project_dir)
    if not selected_paths:
        logger.error("Image selection completed but no selected images were found.")
        return ImageSelectionResult(
            source_paths,
            False,
            error="画像選別AIは完了しましたが、上位画像が見つかりませんでした。",
        )

    return ImageSelectionResult(
        selected_paths,
        True,
        message=f"画像選別AIで上位{len(selected_paths)}枚を選びました。",
    )
