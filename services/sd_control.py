import base64
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from services.pc_control import get_shutdown_delay_seconds, is_windows, schedule_shutdown
from services.weather import JST


logger = logging.getLogger("secretary-bot.sd_control")

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "sd_outputs"
FAVORITES_DIR = DATA_DIR / "sd_favorites"
LATEST_FILE = DATA_DIR / "sd_latest.json"

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, bad anatomy, extra fingers, missing fingers, "
    "bad hands, blurry, jpeg artifacts, watermark, text, logo"
)
DEFAULT_COMFY_WORKFLOW_FILE = Path("workflows") / "comfy_workflow.json"
REFERENCE_COMFY_WORKFLOW_FILE = Path("workflows") / "comfy_workflow_ipadapter.json"
DEFAULT_REFERENCE_STRENGTH = 0.5
REFERENCE_IMAGE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
COMFY_CHECKPOINTS = {
    "janku": "JANKUTrainedChenkinNoobai_v777.safetensors",
    "animagine": "animagineXL40Opt_clear.safetensors",
}
COMFY_POLL_INTERVAL_SECONDS = 2
COMFY_TIMEOUT_SECONDS = 600
STARTUP_STATUS_WAIT_SECONDS = 5
COMFY_PAYLOAD_LOG_LIMIT = 20000
COMFY_ERROR_TEXT_LIMIT = 800

_generate_lock = threading.Lock()
_shutdown_after_done = False


@dataclass(frozen=True)
class SdGenerationOptions:
    prompt: str
    negative_prompt: str | None = None
    model_key: str | None = None
    filename_prefix: str | None = None
    model_selection_explicit: bool = False
    preserve_prompt_text: bool = False
    count: int = 1
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg: float = 7.0
    sampler: str = "Euler a"
    seed: int = -1
    reference_image_bytes: bytes | None = None
    reference_image_content_type: str | None = None
    reference_strength: float | None = None


@dataclass(frozen=True)
class GeneratedImage:
    """One generated image plus the generation conditions actually used to
    produce it. All condition fields except path/batch_index are Optional -
    when a value cannot be safely read back from the workflow that was
    actually queued (or, for the webui backend, is simply not knowable),
    None is stored rather than a guessed or .env-derived value.
    """

    path: Path
    batch_index: int
    seed: int | None = None
    checkpoint_name: str | None = None
    sampler_name: str | None = None
    scheduler: str | None = None
    steps: int | None = None
    cfg: float | None = None
    width: int | None = None
    height: int | None = None
    workflow_file: str | None = None
    workflow_hash: str | None = None


def _now_text() -> str:
    return datetime.now(JST).strftime("%Y%m%d_%H%M%S")


def _api_url(path: str) -> str:
    base_url = os.getenv("SD_API_URL", "http://127.0.0.1:8188").strip().rstrip("/") + "/"
    return urljoin(base_url, path.lstrip("/"))


def _backend() -> str:
    return os.getenv("SD_BACKEND", "comfy").strip().lower() or "comfy"


def is_comfy_backend() -> bool:
    return _backend() == "comfy"


def _max_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s: %s. Using default %s.", name, raw, default)
        return default


def _max_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s: %s. Using default %s.", name, raw, default)
        return default


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def is_generation_running() -> bool:
    return _generate_lock.locked()


def _load_latest_paths() -> list[Path]:
    if not LATEST_FILE.exists():
        return []

    try:
        data = json.loads(LATEST_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to load latest SD image state.")
        return []

    if not isinstance(data, dict):
        return []

    raw_paths = data.get("paths", [])
    if not isinstance(raw_paths, list):
        return []

    paths = []
    for raw_path in raw_paths:
        path = Path(str(raw_path))
        if path.exists() and path.is_file():
            paths.append(path)
    return paths


def _save_latest_paths(paths: list[Path]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "updated_at": datetime.now(JST).isoformat(timespec="seconds"),
        "paths": [str(path) for path in paths],
    }
    LATEST_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_latest_image_paths() -> list[Path]:
    return _load_latest_paths()


def get_negative_prompt(value: str | None) -> str:
    if value and value.strip():
        return value.strip()
    return os.getenv("SD_DEFAULT_NEGATIVE_PROMPT", DEFAULT_NEGATIVE_PROMPT).strip() or DEFAULT_NEGATIVE_PROMPT


def resolve_reference_strength(reference_used: bool, strength: float | None) -> float | None:
    if not reference_used:
        if strength is not None:
            raise ValueError("reference_strengthを指定するにはreference_imageを添付してください。")
        return None
    resolved = DEFAULT_REFERENCE_STRENGTH if strength is None else strength
    if isinstance(resolved, bool) or not isinstance(resolved, (int, float)):
        raise ValueError("reference_strengthは0.0～1.0の数値で指定してください。")
    resolved = float(resolved)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError("reference_strengthは0.0～1.0の範囲で指定してください。")
    return resolved


def get_reference_status_text(reference_used: bool, strength: float | None) -> str:
    resolved = resolve_reference_strength(reference_used, strength)
    if resolved is None:
        return "Reference image: Not used"
    return f"Reference image: Used\nReference strength: {resolved:g}"


def validate_options(options: SdGenerationOptions) -> SdGenerationOptions:
    max_count = _max_int("SD_MAX_COUNT", 10)
    max_width = _max_int("SD_MAX_WIDTH", 1024)
    max_height = _max_int("SD_MAX_HEIGHT", 1024)
    max_steps = _max_int("SD_MAX_STEPS", 60)
    max_cfg = _max_float("SD_MAX_CFG", 20.0)

    prompt = options.prompt if options.preserve_prompt_text else options.prompt.strip()
    if not prompt.strip():
        raise ValueError("prompt is empty.")

    if options.preserve_prompt_text:
        negative_prompt = options.negative_prompt if options.negative_prompt is not None else ""
    else:
        negative_prompt = get_negative_prompt(options.negative_prompt)

    reference_used = options.reference_image_bytes is not None
    reference_strength = resolve_reference_strength(reference_used, options.reference_strength)
    reference_content_type = None
    if reference_used:
        if not isinstance(options.reference_image_bytes, bytes) or not options.reference_image_bytes:
            raise ValueError("reference_imageの画像データが空です。")
        reference_content_type = (options.reference_image_content_type or "").split(";", 1)[0].strip().lower()
        if reference_content_type not in REFERENCE_IMAGE_EXTENSIONS:
            raise ValueError("reference_imageはJPEG / PNG / WebPを指定してください。")

    return SdGenerationOptions(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_key=options.model_key,
        filename_prefix=options.filename_prefix,
        model_selection_explicit=options.model_selection_explicit,
        preserve_prompt_text=options.preserve_prompt_text,
        count=_clamp_int(options.count, 1, max_count),
        width=_clamp_int(options.width, 256, max_width),
        height=_clamp_int(options.height, 256, max_height),
        steps=_clamp_int(options.steps, 1, max_steps),
        cfg=_clamp_float(options.cfg, 1.0, max_cfg),
        sampler=options.sampler.strip() or "Euler a",
        seed=options.seed if options.seed is not None else -1,
        reference_image_bytes=options.reference_image_bytes,
        reference_image_content_type=reference_content_type,
        reference_strength=reference_strength,
    )


def is_api_available() -> bool:
    try:
        if _backend() == "comfy":
            response = requests.get(_api_url("/system_stats"), timeout=3)
        else:
            response = requests.get(_api_url("/sdapi/v1/progress"), timeout=3)
        return response.status_code < 500
    except requests.RequestException:
        return False


def _start_file_env_name() -> str:
    return "COMFYUI_START_FILE" if _backend() == "comfy" else "SD_START_FILE"


def _start_file_missing_message(env_name: str) -> str:
    if env_name == "COMFYUI_START_FILE":
        return "COMFYUI_START_FILE が未設定です。ComfyUIを起動する .bat / .cmd / .exe の固定パスを .env に設定してください。"
    return "SD_START_FILE が未設定です。起動用 .bat / .cmd / .exe の固定パスを .env に設定してください。"


def _launch_fixed_start_file(path: Path, cwd: Path) -> None:
    if path.suffix.lower() in {".bat", ".cmd"}:
        subprocess.Popen(["cmd.exe", "/c", str(path)], cwd=str(cwd), shell=False)
    else:
        subprocess.Popen([str(path)], cwd=str(cwd), shell=False)


def _start_sd_app_impl() -> str:
    if is_api_available():
        return "画像生成APIはすでに起動しています。"

    env_name = _start_file_env_name()
    start_file = os.getenv(env_name, "").strip().strip('"')
    if not start_file:
        return _start_file_missing_message(env_name)

    path = Path(start_file)
    if not path.exists() or not path.is_file():
        return f"{env_name} の起動ファイルが見つかりません: {path}"

    if path.suffix.lower() not in {".bat", ".cmd", ".exe"}:
        return f"{env_name} は .bat / .cmd / .exe の固定パスだけに対応しています。"

    cwd_text = os.getenv("SD_START_CWD", "").strip().strip('"')
    cwd = Path(cwd_text) if cwd_text else path.parent
    if not cwd.exists() or not cwd.is_dir():
        return f"起動時の作業フォルダが見つかりません: {cwd}"

    try:
        _launch_fixed_start_file(path, cwd)
    except Exception:
        logger.exception("Failed to start SD app: backend=%s env=%s path=%s cwd=%s", _backend(), env_name, path, cwd)
        return "画像生成APIの起動に失敗しました。Botのログを確認してください。"

    logger.info("Started SD app: backend=%s env=%s path=%s cwd=%s", _backend(), env_name, path, cwd)
    time.sleep(STARTUP_STATUS_WAIT_SECONDS)
    if is_api_available():
        return "画像生成APIを起動しました。疎通確認もできています。"

    return "起動ファイルを実行しました。APIはまだ応答していません。少し待ってから /sd_status を確認してください。"


def start_sd_app() -> str:
    return _start_sd_app_impl()


def _legacy_start_sd_app_unreachable() -> str:
    if is_api_available():
        return "画像生成APIは起動済みだよ。"

    start_file = os.getenv("SD_START_FILE", "").strip().strip('"')
    if not start_file:
        return "SD_START_FILE が未設定だよ。Stability MatrixからComfyUIを手動で起動してね。"

    path = Path(start_file)
    if not path.exists() or not path.is_file():
        return f"起動ファイルが見つからないよ: {path}"

    if path.suffix.lower() not in {".bat", ".cmd", ".exe"}:
        return "SD_START_FILE は .bat / .cmd / .exe の固定パスだけ対応しているよ。"

    cwd_text = os.getenv("SD_START_CWD", "").strip().strip('"')
    cwd = Path(cwd_text) if cwd_text else path.parent
    if not cwd.exists() or not cwd.is_dir():
        return f"作業フォルダが見つからないよ: {cwd}"

    try:
        if path.suffix.lower() in {".bat", ".cmd"}:
            subprocess.Popen(["cmd.exe", "/c", str(path)], cwd=str(cwd), shell=False)
        else:
            subprocess.Popen([str(path)], cwd=str(cwd), shell=False)
    except Exception:
        logger.exception("Failed to start SD app.")
        return "画像生成UIの起動に失敗したよ。Botのログを確認してね。"

    return "画像生成UIを起動したよ。APIが使えるようになったら /sd_status で確認してね。"


def get_status_message() -> str:
    backend = _backend()
    available = is_api_available()
    lines = [
        "画像生成の状態だよ。",
        f"backend: {backend}",
        f"api: {'起動中' if available else '未起動または未応答'}",
        f"bot_generation_lock: {'生成中' if is_generation_running() else '待機中'}",
    ]

    if backend == "comfy":
        workflow_file = _get_comfy_workflow_file()
        lines.append(f"workflow: {workflow_file}")
        lines.append(f"workflow_exists: {'yes' if workflow_file.exists() else 'no'}")
    elif available:
        try:
            response = requests.get(_api_url("/sdapi/v1/progress"), params={"skip_current_image": "true"}, timeout=5)
            response.raise_for_status()
            data = response.json()
            progress = float(data.get("progress", 0.0)) * 100
            eta = data.get("eta_relative")
            lines.append(f"webui_progress: {progress:.1f}%")
            if isinstance(eta, (int, float)) and eta > 0:
                lines.append(f"eta_seconds: {eta:.1f}")
        except requests.RequestException:
            logger.exception("Failed to fetch SD progress.")
            lines.append("webui_progress: 取得失敗")

    return "\n".join(lines)


def _decode_and_save_webui_images(images: list[Any]) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = _now_text()
    paths: list[Path] = []

    for index, raw_image in enumerate(images, start=1):
        if not isinstance(raw_image, str):
            continue
        image_data = raw_image.split(",", 1)[-1]
        image_bytes = base64.b64decode(image_data)
        output_path = OUTPUT_DIR / f"{timestamp}_{index:02d}.png"
        output_path.write_bytes(image_bytes)
        paths.append(output_path)

    _save_latest_paths(paths)
    return paths


def _get_comfy_workflow_file(reference_enabled: bool = False) -> Path:
    if reference_enabled:
        return REFERENCE_COMFY_WORKFLOW_FILE
    raw_value = os.getenv("COMFY_WORKFLOW_FILE", "").strip().strip('"')
    return Path(raw_value) if raw_value else DEFAULT_COMFY_WORKFLOW_FILE


def _get_node_id(name: str) -> str:
    return os.getenv(name, "").strip()


def _load_comfy_workflow(reference_enabled: bool = False) -> dict[str, Any]:
    workflow_file = _get_comfy_workflow_file(reference_enabled)
    if not workflow_file.exists() or not workflow_file.is_file():
        raise RuntimeError(f"ComfyUI workflow file が見つからないよ: {workflow_file}")

    try:
        workflow = json.loads(workflow_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ComfyUI workflow file のJSONを読み取れなかったよ: {workflow_file}") from exc

    if not isinstance(workflow, dict):
        raise RuntimeError("ComfyUI workflow はAPI形式のJSONオブジェクトにしてね。")
    _validate_comfy_workflow(workflow, workflow_file)
    return workflow


def _validate_comfy_workflow(workflow: dict[str, Any], workflow_file: Path) -> None:
    if not workflow:
        raise RuntimeError(f"ComfyUI workflow が空です: {workflow_file}")

    invalid_nodes: list[str] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            invalid_nodes.append(str(node_id))
            continue
        if not isinstance(node.get("class_type"), str) or not node.get("class_type"):
            invalid_nodes.append(str(node_id))
            continue
        if not isinstance(node.get("inputs"), dict):
            invalid_nodes.append(str(node_id))

    if invalid_nodes:
        raise RuntimeError(
            "ComfyUI workflow がAPI形式として正しくない可能性があります。"
            f" class_type / inputs を確認してください: {', '.join(invalid_nodes[:10])}"
        )

    save_nodes = _find_save_image_nodes(workflow)
    if not save_nodes:
        raise RuntimeError("ComfyUI workflow に SaveImage ノードがありません。")

    positive_id, negative_id = _find_prompt_text_nodes(workflow)
    if not positive_id or not negative_id:
        raise RuntimeError("ComfyUI workflow の Positive / Negative CLIPTextEncode を検出できません。")

    logger.info(
        "ComfyUI workflow validated: file=%s nodes=%s positive=%s negative=%s save_nodes=%s",
        workflow_file,
        len(workflow),
        positive_id,
        negative_id,
        save_nodes,
    )


def _node_inputs(workflow: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    node = workflow.get(str(node_id))
    if not isinstance(node, dict):
        return None
    inputs = node.get("inputs")
    return inputs if isinstance(inputs, dict) else None


def _node_class(workflow: dict[str, Any], node_id: str) -> str:
    node = workflow.get(str(node_id))
    if not isinstance(node, dict):
        return ""
    return str(node.get("class_type", ""))


def _linked_node_id(value: Any) -> str | None:
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, tuple) and value:
        return str(value[0])
    return None


def _is_clip_text_node(workflow: dict[str, Any], node_id: str) -> bool:
    inputs = _node_inputs(workflow, node_id)
    return _node_class(workflow, node_id) == "CLIPTextEncode" and inputs is not None and "text" in inputs


def _find_sampler_node(workflow: dict[str, Any]) -> str | None:
    preferred_classes = {
        "KSampler",
        "KSamplerAdvanced",
        "SamplerCustom",
        "SamplerCustomAdvanced",
    }
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        class_type = str(node.get("class_type", ""))
        if class_type in preferred_classes and ("positive" in inputs or "negative" in inputs):
            return str(node_id)

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict) and "positive" in inputs and "negative" in inputs:
            return str(node_id)
    return None


def _resolve_clip_text_node(workflow: dict[str, Any], start_node_id: str | None, visited: set[str] | None = None) -> str | None:
    if not start_node_id:
        return None
    visited = visited or set()
    if start_node_id in visited:
        return None
    visited.add(start_node_id)

    if _is_clip_text_node(workflow, start_node_id):
        return start_node_id

    inputs = _node_inputs(workflow, start_node_id)
    if not inputs:
        return None

    for value in inputs.values():
        linked_id = _linked_node_id(value)
        if linked_id is None:
            continue
        found = _resolve_clip_text_node(workflow, linked_id, visited)
        if found:
            return found
    return None


def _find_prompt_text_nodes(workflow: dict[str, Any]) -> tuple[str | None, str | None]:
    sampler_id = _find_sampler_node(workflow)
    if sampler_id:
        sampler_inputs = _node_inputs(workflow, sampler_id) or {}
        positive_id = _resolve_clip_text_node(workflow, _linked_node_id(sampler_inputs.get("positive")))
        negative_id = _resolve_clip_text_node(workflow, _linked_node_id(sampler_inputs.get("negative")))
        if positive_id or negative_id:
            logger.info(
                "ComfyUI prompt nodes detected from sampler: sampler=%s positive=%s negative=%s",
                sampler_id,
                positive_id,
                negative_id,
            )
            return positive_id, negative_id

    clip_nodes = [
        str(node_id)
        for node_id, node in workflow.items()
        if isinstance(node, dict) and _is_clip_text_node(workflow, str(node_id))
    ]
    logger.warning("Falling back to CLIPTextEncode order detection: nodes=%s", clip_nodes)
    positive_id = clip_nodes[0] if len(clip_nodes) >= 1 else None
    negative_id = clip_nodes[1] if len(clip_nodes) >= 2 else None
    return positive_id, negative_id


def _find_seed_input(workflow: dict[str, Any]) -> tuple[str | None, str | None]:
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if node.get("class_type") == "RandomNoise" and isinstance(inputs, dict) and "noise_seed" in inputs:
            return str(node_id), "noise_seed"

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict) and "seed" in inputs:
            return str(node_id), "seed"

    return None, None


def _find_save_image_nodes(workflow: dict[str, Any]) -> list[str]:
    return [
        str(node_id)
        for node_id, node in workflow.items()
        if isinstance(node, dict) and str(node.get("class_type", "")) == "SaveImage"
    ]


def _extract_checkpoint_name(workflow: dict[str, Any]) -> str | None:
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type", "")) != "CheckpointLoaderSimple":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        ckpt_name = inputs.get("ckpt_name")
        if isinstance(ckpt_name, str) and ckpt_name:
            return ckpt_name
    return None


def _extract_sampler_settings(
    workflow: dict[str, Any], sampler_node_id: str | None
) -> tuple[int | None, float | None, str | None, str | None]:
    if not sampler_node_id:
        return None, None, None, None
    inputs = _node_inputs(workflow, sampler_node_id) or {}

    steps = inputs.get("steps")
    steps = steps if isinstance(steps, int) else None

    cfg = inputs.get("cfg")
    cfg = float(cfg) if isinstance(cfg, (int, float)) else None

    sampler_name = inputs.get("sampler_name")
    sampler_name = sampler_name if isinstance(sampler_name, str) and sampler_name else None

    scheduler = inputs.get("scheduler")
    scheduler = scheduler if isinstance(scheduler, str) and scheduler else None

    return steps, cfg, sampler_name, scheduler


def _extract_latent_dimensions(workflow: dict[str, Any]) -> tuple[int | None, int | None]:
    # Prefer an exact EmptyLatentImage match (what this project's workflow
    # actually uses); only fall back to a looser "class name contains
    # Latent, and both width/height are present" match if no exact match is
    # found, so we don't misattribute dimensions from an unrelated node.
    fallback_match: tuple[int, int] | None = None
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        width = inputs.get("width")
        height = inputs.get("height")
        if not (isinstance(width, int) and isinstance(height, int)):
            continue
        class_type = str(node.get("class_type", ""))
        if class_type == "EmptyLatentImage":
            return width, height
        if fallback_match is None and "Latent" in class_type:
            fallback_match = (width, height)
    return fallback_match if fallback_match is not None else (None, None)


def _safe_workflow_file_hash(workflow_file: Path) -> str | None:
    try:
        data = workflow_file.read_bytes()
    except OSError:
        logger.warning("Could not read workflow file to compute its hash: %s", workflow_file)
        return None
    return hashlib.sha256(data).hexdigest()


def _set_node_input(workflow: dict[str, Any], node_id: str, key: str, value: Any) -> None:
    node = workflow.get(node_id)
    if not isinstance(node, dict):
        raise RuntimeError(f"ComfyUI workflow の node {node_id} が見つからないよ。")
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError(f"ComfyUI workflow の node {node_id} に inputs がないよ。")
    inputs[key] = value


def _resolve_comfy_checkpoint(model_key: str) -> str:
    try:
        return COMFY_CHECKPOINTS[model_key]
    except KeyError as exc:
        raise ValueError("指定された画像生成モデルには対応していません。") from exc


def _find_checkpoint_loader_node_id(workflow: dict[str, Any]) -> str:
    node_ids = [
        str(node_id)
        for node_id, node in workflow.items()
        if isinstance(node, dict) and str(node.get("class_type", "")) == "CheckpointLoaderSimple"
    ]
    if not node_ids:
        raise RuntimeError(
            "ComfyUI workflow に CheckpointLoaderSimple ノードが見つかりません。"
            "モデルを選択するにはチェックポイント読込ノードが1個必要です。"
        )
    if len(node_ids) > 1:
        raise RuntimeError(
            "ComfyUI workflow に CheckpointLoaderSimple ノードが複数あります。"
            "変更対象を特定できないためモデルを適用できません。"
        )
    return node_ids[0]


def _apply_comfy_options(
    workflow: dict[str, Any], options: SdGenerationOptions, reference_image_name: str | None = None
) -> tuple[dict[str, Any], str | None, str | None]:
    checkpoint_name = None
    checkpoint_node_id = None
    if options.model_key is not None:
        checkpoint_name = _resolve_comfy_checkpoint(options.model_key)
        checkpoint_node_id = _find_checkpoint_loader_node_id(workflow)

    save_nodes = _find_save_image_nodes(workflow)
    if not save_nodes:
        raise RuntimeError("ComfyUI workflow に SaveImage ノードが見つかりません。完成画像を保存するWorkflowを指定してください。")
    if options.filename_prefix is not None:
        if len(save_nodes) != 1:
            raise RuntimeError(
                "ComfyUI workflow に SaveImage ノードが複数あります。"
                "モデル別の保存先を安全に特定できないため filename_prefix を適用できません。"
            )
        _set_node_input(workflow, save_nodes[0], "filename_prefix", options.filename_prefix)

    env_positive_node_id = _get_node_id("COMFY_POSITIVE_NODE_ID")
    env_negative_node_id = _get_node_id("COMFY_NEGATIVE_NODE_ID")
    detected_positive_node_id, detected_negative_node_id = _find_prompt_text_nodes(workflow)
    positive_node_id = env_positive_node_id or detected_positive_node_id
    negative_node_id = env_negative_node_id or detected_negative_node_id

    if not positive_node_id or not negative_node_id:
        raise RuntimeError(
            "Positive/Negative用のCLIPTextEncodeノードを自動検出できませんでした。"
            "KSamplerのpositive/negative接続を確認してください。"
        )

    if checkpoint_node_id is not None and checkpoint_name is not None:
        _set_node_input(workflow, checkpoint_node_id, "ckpt_name", checkpoint_name)

    _set_node_input(workflow, positive_node_id, "text", options.prompt)
    _set_node_input(workflow, negative_node_id, "text", options.negative_prompt or "")

    if options.reference_image_bytes is not None:
        load_image_nodes = [
            str(node_id)
            for node_id, node in workflow.items()
            if isinstance(node, dict) and str(node.get("class_type", "")) == "LoadImage"
        ]
        adapter_nodes = [
            str(node_id)
            for node_id, node in workflow.items()
            if isinstance(node, dict) and str(node.get("class_type", "")) == "IPAdapterAdvanced"
        ]
        if len(load_image_nodes) != 1 or len(adapter_nodes) != 1:
            raise RuntimeError("参考画像workflowにはLoadImageとIPAdapterAdvancedが各1個必要です。")
        if not reference_image_name:
            raise RuntimeError("ComfyUIへアップロードした参考画像名がありません。")
        _set_node_input(workflow, load_image_nodes[0], "image", reference_image_name)
        _set_node_input(workflow, adapter_nodes[0], "weight", options.reference_strength)

    env_seed_node_id = _get_node_id("COMFY_SEED_NODE_ID")
    seed_node_id, seed_key = _find_seed_input(workflow)
    if env_seed_node_id:
        seed_node_id = env_seed_node_id
        seed_key = "noise_seed" if _node_class(workflow, seed_node_id) == "RandomNoise" else "seed"

    if seed_node_id and seed_key:
        _set_node_input(workflow, seed_node_id, seed_key, random.randint(0, 2**32 - 1))
        logger.info(
            "ComfyUI workflow inputs applied: positive=%s negative=%s seed=%s.%s save_nodes=%s",
            positive_node_id,
            negative_node_id,
            seed_node_id,
            seed_key,
            save_nodes,
        )
    else:
        logger.warning("ComfyUI seed input was not found. Workflow seed was left unchanged.")

    return workflow, seed_node_id, seed_key


def _queue_comfy_prompt(workflow: dict[str, Any], client_id: str) -> str:
    response = requests.post(_api_url("/prompt"), json={"prompt": workflow, "client_id": client_id}, timeout=30)
    response.raise_for_status()
    data = response.json()
    prompt_id = data.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise RuntimeError("ComfyUI APIから prompt_id が返らなかったよ。")
    return prompt_id


def _upload_comfy_reference_image(image_bytes: bytes, content_type: str) -> str:
    extension = REFERENCE_IMAGE_EXTENSIONS[content_type]
    filename = f"discord_reference_{hashlib.sha256(image_bytes).hexdigest()[:20]}{extension}"
    try:
        response = requests.post(
            _api_url("/upload/image"),
            files={"image": (filename, image_bytes, content_type)},
            data={"type": "input", "overwrite": "true"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.exception("Failed to upload reference image to ComfyUI.")
        raise RuntimeError("参考画像をComfyUIへアップロードできませんでした。") from exc
    uploaded_name = data.get("name")
    if not isinstance(uploaded_name, str) or not uploaded_name:
        raise RuntimeError("ComfyUIの参考画像アップロード結果にファイル名がありません。")
    return uploaded_name


def _short_error_text(text: str) -> str:
    clean_text = " ".join((text or "").split())
    if not clean_text:
        return "レスポンス本文は空でした。"
    return clean_text[:COMFY_ERROR_TEXT_LIMIT]


def _queue_comfy_prompt_with_logging(workflow: dict[str, Any], client_id: str) -> str:
    payload = {"prompt": workflow, "client_id": client_id}
    payload_text = json.dumps(payload, ensure_ascii=False)
    logger.info(
        "ComfyUI /prompt request: url=%s payload_length=%s payload=%s",
        _api_url("/prompt"),
        len(payload_text),
        payload_text[:COMFY_PAYLOAD_LOG_LIMIT],
    )

    try:
        response = requests.post(_api_url("/prompt"), json=payload, timeout=30)
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_text = response.text if response is not None else ""
        logger.error(
            "ComfyUI /prompt HTTP error: status=%s response=%s",
            getattr(response, "status_code", "unknown"),
            response_text,
        )
        raise RuntimeError(
            "ComfyUI APIがリクエストを受け付けませんでした。"
            f" HTTP {getattr(response, 'status_code', 'unknown')}: {_short_error_text(response_text)}"
        ) from exc
    except requests.RequestException as exc:
        logger.exception("ComfyUI /prompt request failed.")
        raise RuntimeError(f"ComfyUI APIへの接続に失敗しました: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        logger.error("ComfyUI /prompt returned non-JSON response: %s", response.text)
        raise RuntimeError("ComfyUI APIからJSONではない応答が返りました。") from exc

    prompt_id = data.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        logger.error("ComfyUI /prompt response did not include prompt_id: %s", data)
        raise RuntimeError("ComfyUI APIから prompt_id が返りませんでした。")
    logger.info("ComfyUI /prompt accepted: prompt_id=%s", prompt_id)
    return prompt_id


def _wait_comfy_history(prompt_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + COMFY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        response = requests.get(_api_url(f"/history/{prompt_id}"), timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and prompt_id in data:
            history = data[prompt_id]
            if isinstance(history, dict):
                return history
        time.sleep(COMFY_POLL_INTERVAL_SECONDS)
    raise RuntimeError("ComfyUIの生成完了待ちがタイムアウトしたよ。")


def _download_comfy_images(history: dict[str, Any], batch_index: int) -> list[Path]:
    outputs = history.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("ComfyUI history に outputs がなかったよ。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = _now_text()
    paths: list[Path] = []

    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        images = output.get("images")
        if not isinstance(images, list):
            continue
        for image_index, image in enumerate(images, start=1):
            if not isinstance(image, dict):
                continue
            filename = image.get("filename")
            image_type = image.get("type", "output")
            subfolder = image.get("subfolder", "")
            if not isinstance(filename, str) or not filename:
                continue

            response = requests.get(
                _api_url("/view"),
                params={"filename": filename, "type": image_type, "subfolder": subfolder},
                timeout=60,
            )
            response.raise_for_status()
            suffix = Path(filename).suffix or ".png"
            output_path = OUTPUT_DIR / f"{timestamp}_{batch_index:02d}_{len(paths) + image_index:02d}{suffix}"
            output_path.write_bytes(response.content)
            paths.append(output_path)

    if not paths:
        raise RuntimeError("ComfyUIの生成画像を取得できなかったよ。SaveImageノードがあるWorkflowか確認してね。")
    return paths


def _generate_webui_images(options: SdGenerationOptions) -> list[Path]:
    payload = {
        "prompt": options.prompt,
        "negative_prompt": options.negative_prompt,
        "batch_size": options.count,
        "n_iter": 1,
        "width": options.width,
        "height": options.height,
        "steps": options.steps,
        "cfg_scale": options.cfg,
        "sampler_name": options.sampler,
        "seed": options.seed,
    }
    response = requests.post(_api_url("/sdapi/v1/txt2img"), json=payload, timeout=None)
    response.raise_for_status()
    data = response.json()
    images = data.get("images", [])
    if not isinstance(images, list) or not images:
        raise RuntimeError("画像生成APIから画像が返りませんでした。")

    paths = _decode_and_save_webui_images(images)
    if not paths:
        raise RuntimeError("生成画像の保存に失敗しました。")
    return paths


def _generate_comfy_images_with_metadata(options: SdGenerationOptions) -> list[GeneratedImage]:
    reference_enabled = options.reference_image_bytes is not None
    workflow_file = _get_comfy_workflow_file(reference_enabled)
    workflow_file_text = str(workflow_file)
    workflow_hash = _safe_workflow_file_hash(workflow_file)

    reference_image_name = None
    if reference_enabled:
        reference_image_name = _upload_comfy_reference_image(
            options.reference_image_bytes or b"",
            options.reference_image_content_type or "image/png",
        )

    images: list[GeneratedImage] = []
    client_id = str(uuid.uuid4())
    for batch_index in range(1, options.count + 1):
        workflow, seed_node_id, seed_key = _apply_comfy_options(
            _load_comfy_workflow(reference_enabled), options, reference_image_name
        )

        seed_value: int | None = None
        if seed_node_id and seed_key:
            seed_inputs = _node_inputs(workflow, seed_node_id) or {}
            raw_seed = seed_inputs.get(seed_key)
            seed_value = raw_seed if isinstance(raw_seed, int) else None

        checkpoint_name = _extract_checkpoint_name(workflow)
        sampler_node_id = _find_sampler_node(workflow)
        steps, cfg, sampler_name, scheduler = _extract_sampler_settings(workflow, sampler_node_id)
        width, height = _extract_latent_dimensions(workflow)

        prompt_id = _queue_comfy_prompt_with_logging(workflow, client_id)
        history = _wait_comfy_history(prompt_id)
        paths = _download_comfy_images(history, batch_index)

        for path in paths:
            images.append(
                GeneratedImage(
                    path=path,
                    batch_index=batch_index,
                    seed=seed_value,
                    checkpoint_name=checkpoint_name,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    steps=steps,
                    cfg=cfg,
                    width=width,
                    height=height,
                    workflow_file=workflow_file_text,
                    workflow_hash=workflow_hash,
                )
            )
    return images


def _generate_webui_images_with_metadata(options: SdGenerationOptions) -> list[GeneratedImage]:
    # The webui backend has no ComfyUI-style workflow file/checkpoint-loader
    # node to read back, and the actual per-image seed used by the webui API
    # is not currently parsed out of its response - those fields are left as
    # None (see GeneratedImage's docstring) rather than guessed. The
    # remaining fields reflect the request options actually sent, unchanged.
    paths = _generate_webui_images(options)
    return [
        GeneratedImage(
            path=path,
            batch_index=index,
            seed=None,
            checkpoint_name=None,
            sampler_name=options.sampler or None,
            scheduler=None,
            steps=options.steps,
            cfg=options.cfg,
            width=options.width,
            height=options.height,
            workflow_file=None,
            workflow_hash=None,
        )
        for index, path in enumerate(paths, start=1)
    ]


def generate_images_with_metadata(options: SdGenerationOptions) -> list[GeneratedImage]:
    global _shutdown_after_done

    if not _generate_lock.acquire(blocking=False):
        raise RuntimeError("すでに画像生成中だよ。完了してからもう一度実行してね。")

    safe_options = validate_options(options)
    try:
        backend = _backend()
        if backend != "comfy" and safe_options.model_key is not None and safe_options.model_selection_explicit:
            raise RuntimeError("モデルを明示選択できるのはComfyUIバックエンドだけです。SD_BACKEND=comfyを指定してください。")
        if backend != "comfy" and safe_options.reference_image_bytes is not None:
            raise RuntimeError("reference_imageを指定できるのはComfyUIバックエンドだけです。SD_BACKEND=comfyを指定してください。")
        if backend == "comfy":
            images = _generate_comfy_images_with_metadata(safe_options)
        elif backend == "webui":
            images = _generate_webui_images_with_metadata(safe_options)
        else:
            raise RuntimeError("SD_BACKEND は comfy または webui を指定してね。")

        _save_latest_paths([image.path for image in images])

        if _shutdown_after_done:
            _shutdown_after_done = False
            if is_windows():
                schedule_shutdown(get_shutdown_delay_seconds(), description="画像生成完了後のシャットダウン")
            else:
                logger.warning("Shutdown-after-done was requested, but OS is not Windows.")

        return images
    finally:
        _generate_lock.release()


def generate_images(options: SdGenerationOptions) -> list[Path]:
    return [image.path for image in generate_images_with_metadata(options)]


def set_shutdown_after_done(enabled: bool = True) -> str:
    global _shutdown_after_done
    _shutdown_after_done = enabled
    if enabled:
        return "画像生成完了後にPCをシャットダウンするよう予約したよ。次の生成完了後に実行するね。"
    return "画像生成完了後シャットダウンの予約を解除したよ。"


def save_favorites(*, all_images: bool = True, index: int = 1) -> list[Path]:
    latest_paths = get_latest_image_paths()
    if not latest_paths:
        raise RuntimeError("保存できる最新画像がないよ。")

    if all_images:
        targets = latest_paths
    else:
        if index < 1 or index > len(latest_paths):
            raise RuntimeError("指定した番号の最新画像が見つからないよ。")
        targets = [latest_paths[index - 1]]

    FAVORITES_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for path in targets:
        destination = FAVORITES_DIR / path.name
        shutil.copy2(path, destination)
        saved_paths.append(destination)
    return saved_paths
