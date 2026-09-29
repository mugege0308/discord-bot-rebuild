"""安全なDiscord添付ファイル受付と一時セッション管理。"""

from __future__ import annotations

import hashlib
import json
import logging
import ntpath
import os
import re
import shutil
import tempfile
import threading
import tomllib
import uuid
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageOps

from services.lora_training import LORA_TYPES


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_MAX_IMAGES = 200
DEFAULT_MAX_ATTACHMENT_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_ZIP_BYTES = 200 * 1024 * 1024
DEFAULT_MAX_ZIP_UNCOMPRESSED_BYTES = 1 * 1024 * 1024 * 1024
DEFAULT_MAX_ZIP_ENTRIES = 500
DEFAULT_MAX_PIXELS = 50_000_000
DEFAULT_MAX_ATTACHMENTS = 50
_lock = threading.RLock()
logger = logging.getLogger(__name__)


class LoraUploadError(ValueError):
    """利用者に表示できる安全な受付エラー。"""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _int_env(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except ValueError:
        return default
    return max(minimum, value)


def upload_root() -> Path:
    configured = os.getenv("LORA_UPLOAD_ROOT", "").strip().strip('"')
    root = Path(configured) if configured else Path(__file__).resolve().parents[1] / "data" / "lora_upload_sessions"
    return root.resolve()


def state_root() -> Path:
    root = upload_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def backup_root() -> Path:
    configured = os.getenv("LORA_UPLOAD_BACKUP_ROOT", "").strip().strip('"')
    root = Path(configured) if configured else upload_root().parent / "lora_upload_session_backups"
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _state_path(session_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", session_id):
        raise LoraUploadError("不正なアップロードセッションIDです。")
    return state_root() / f"{session_id}.json"


def _safe_session_dir(session: dict[str, Any]) -> Path:
    root = upload_root()
    path = Path(str(session.get("root", ""))).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise LoraUploadError("アップロード一時領域が安全な場所ではありません。") from exc
    if path == root or path.name != str(session.get("session_id", "")):
        raise LoraUploadError("アップロード一時領域の形式が不正です。")
    return path


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LoraUploadError("アップロードセッションの保存状態を読み込めません。") from exc
    if not isinstance(value, dict):
        raise LoraUploadError("アップロードセッションの形式が不正です。")
    return value


def validate_dataset_name(value: str) -> str:
    name = value.strip()
    if not name or name in {".", ".."}:
        raise LoraUploadError("データセット名を入力してください。")
    if "/" in name or "\\" in name or any(char in name for char in '<>:"|?*'):
        raise LoraUploadError("データセット名にパス区切り文字やWindowsで使えない文字は使えません。")
    if name.endswith((" ", ".")) or any(ord(char) < 32 for char in name):
        raise LoraUploadError("データセット名の末尾空白・ピリオド・制御文字は使えません。")
    if name.split(".", 1)[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise LoraUploadError("そのデータセット名はWindowsでは使用できません。")
    return name


def create_session(
    *, owner_id: int, guild_id: int | None, channel_id: int, message_id: int | None,
    dataset_name: str, lora_type: str, trigger_tag: str, recommended_strength: float,
) -> dict[str, Any]:
    with _lock:
        dataset_name = validate_dataset_name(dataset_name)
        lora_type = str(lora_type).strip().casefold()
        if lora_type not in LORA_TYPES:
            raise LoraUploadError("LoRA種別は character / style / additional から選択してください。")
        trigger_tag = trigger_tag.strip()
        if len(trigger_tag) > 200 or any(ord(char) < 32 for char in trigger_tag):
            raise LoraUploadError("trigger_tagは200文字以内で、改行などの制御文字を含めないでください。")
        if lora_type == "character" and not trigger_tag:
            raise LoraUploadError("characterではtrigger_tagが必須です。")
        if not isinstance(recommended_strength, (int, float)) or isinstance(recommended_strength, bool) or not 0 <= float(recommended_strength) <= 1:
            raise LoraUploadError("recommended_strengthは0.0～1.0で指定してください。")
        replace_open_sessions(owner_id)
        session_id = uuid.uuid4().hex
        root = upload_root() / session_id
        root.mkdir(parents=True, exist_ok=False)
        session = {
            "schema_version": 1,
            "session_id": session_id,
            "owner_id": int(owner_id),
            "guild_id": guild_id,
            "channel_id": int(channel_id),
            "message_id": message_id,
            "dataset_name": dataset_name,
            "lora_type": lora_type,
            "trigger_tag": trigger_tag,
            "recommended_strength": float(recommended_strength),
            "status": "open",
            "created_at": _now(),
            "updated_at": _now(),
            "root": str(root),
            "items": [],
            "attachment_count": 0,
            "rejected": [],
            "duplicates": [],
        }
        try:
            _atomic_json(_state_path(session_id), session)
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise
        return session


def get_session(session_id: str) -> dict[str, Any]:
    with _lock:
        path = _state_path(session_id)
        if not path.is_file():
            raise LoraUploadError("アップロードセッションが見つかりません。")
        return _load(path)


def list_sessions(owner_id: int, *, include_closed: bool = False) -> list[dict[str, Any]]:
    root = state_root()
    result: list[dict[str, Any]] = []
    with _lock:
        for path in root.glob("*.json"):
            try:
                session = _load(path)
            except LoraUploadError:
                continue
            if session.get("owner_id") != int(owner_id):
                continue
            if not include_closed and session.get("status") != "open":
                continue
            result.append(session)
    return sorted(result, key=lambda item: str(item.get("updated_at", "")), reverse=True)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() not in {"", "0", "false", "no", "off", "none", "null"}
    return bool(value)


def is_item_included(item: dict[str, Any]) -> bool:
    """受付表示と確定処理で共有する画像採用判定。旧フィールドも読む。"""
    if "excluded" in item and _as_bool(item.get("excluded")):
        return False
    for key in ("included", "include", "accepted"):
        if key in item:
            return _as_bool(item.get(key))
    return True


def included_items(session: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in session.get("items", []) if is_item_included(item)]


def _save(session: dict[str, Any]) -> None:
    session["updated_at"] = _now()
    _atomic_json(_state_path(str(session["session_id"])), session)


def _backup_session(session: dict[str, Any]) -> Path:
    """セッションを削除せず、置換前の状態を別領域へ保存する。"""
    session_id = str(session.get("session_id", ""))
    source_state = _state_path(session_id)
    source_root = _safe_session_dir(session)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    destination = backup_root() / f"{stamp}_{session_id}"
    destination.mkdir(parents=True, exist_ok=False)
    if source_state.is_file():
        shutil.copy2(source_state, destination / "session.json")
    if source_root.is_dir():
        shutil.copytree(source_root, destination / "session_root", copy_function=shutil.copy2)
    return destination


def replace_open_sessions(owner_id: int) -> list[dict[str, Any]]:
    """同一所有者の未開始受付だけをバックアップして置換終了する。

    ``plan_created`` は /lora_train の確認待ち状態であり、学習ジョブでは
    ない。確認メッセージが失効した場合に新しい受付で置き換えられるよう
    対象に含めるが、実行中ジョブに紐づくセッションは保護する。
    """
    replaced: list[dict[str, Any]] = []
    with _lock:
        for session in list_sessions(owner_id, include_closed=True):
            if session.get("status") not in {"open", "plan_created"}:
                continue
            if session.get("status") == "plan_created" and _session_has_any_job(str(session.get("session_id", ""))):
                logger.info(
                    "LoRA upload session replacement skipped for already-used session_id=%s",
                    session.get("session_id"),
                )
                continue
            if _session_has_running_job(str(session.get("session_id", ""))):
                logger.warning(
                    "LoRA upload session replacement skipped for running session_id=%s",
                    session.get("session_id"),
                )
                continue
            backup = _backup_session(session)
            session["status"] = "cancelled"
            session["cancel_reason"] = "replaced_by_new_session"
            session["cancelled_at"] = _now()
            session["backup_root"] = str(backup)
            _save(session)
            replaced.append(session)
            logger.info(
                "LoRA upload session replaced session_id=%s status=%s backup=%s",
                session.get("session_id"),
                session.get("status"),
                backup,
            )
    return replaced


def _session_has_running_job(session_id: str) -> bool:
    """Return whether a persisted starting/running job owns this session."""
    if not session_id:
        return False
    configured = os.getenv("LORA_JOB_ROOT", "").strip().strip('"')
    jobs_root = Path(configured) if configured else Path(__file__).resolve().parents[1] / "data" / "lora_jobs"
    if not jobs_root.is_dir():
        return False
    for path in jobs_root.glob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and str(value.get("session_id", "")) == session_id and value.get("status") in {"starting", "running"}:
            return True
    return False


def _session_has_any_job(session_id: str) -> bool:
    if not session_id:
        return False
    configured = os.getenv("LORA_JOB_ROOT", "").strip().strip('"')
    jobs_root = Path(configured) if configured else Path(__file__).resolve().parents[1] / "data" / "lora_jobs"
    if not jobs_root.is_dir():
        return False
    for path in jobs_root.glob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and str(value.get("session_id", "")) == session_id:
            return True
    return False


def _normalized_zip_name(name: str) -> str:
    if "\x00" in name or "\\" in name or ntpath.splitdrive(name)[0] or name.startswith(("/", "\\")):
        raise LoraUploadError("ZIP内に絶対パスまたは安全でないパスがあります。")
    path = PurePosixPath(name)
    if any(part in {"", ".", ".."} for part in path.parts) or not path.parts:
        raise LoraUploadError("ZIP内に親ディレクトリ参照または不正なパスがあります。")
    return "/".join(path.parts)


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    return ((info.external_attr >> 16) & 0o170000) == 0o120000


def _validate_image(data: bytes, extension: str) -> tuple[int, int, bytes]:
    if len(data) > _int_env("LORA_UPLOAD_MAX_ATTACHMENT_BYTES", DEFAULT_MAX_ATTACHMENT_BYTES):
        raise LoraUploadError("画像サイズが上限を超えています。")
    try:
        with Image.open(BytesIO(data)) as probe:
            probe.verify()
        with Image.open(BytesIO(data)) as opened:
            width, height = opened.size
            if width <= 0 or height <= 0 or width * height > _int_env("LORA_UPLOAD_MAX_PIXELS", DEFAULT_MAX_PIXELS):
                raise LoraUploadError("画像の解像度が上限を超えています。")
            image = ImageOps.exif_transpose(opened)
            output = BytesIO()
            format_name = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}[extension]
            if format_name == "JPEG" and image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            image.save(output, format=format_name)
            return image.width, image.height, output.getvalue()
    except LoraUploadError:
        raise
    except Exception as exc:
        raise LoraUploadError("画像が壊れているか、対応形式ではありません。") from exc


def _store_image(session: dict[str, Any], *, original_name: str, attachment_name: str, data: bytes, caption: str | None) -> dict[str, Any]:
    max_images = _int_env("LORA_UPLOAD_MAX_IMAGES", DEFAULT_MAX_IMAGES)
    included_count = len(included_items(session))
    if included_count >= max_images:
        raise LoraUploadError(f"画像枚数が上限（{max_images}枚）を超えています。")
    extension = Path(original_name).suffix.casefold()
    if extension not in IMAGE_EXTENSIONS:
        raise LoraUploadError("対応形式はPNG/JPG/JPEG/WebPです。")
    width, height, normalized = _validate_image(data, extension)
    digest = hashlib.sha256(data).hexdigest()
    if any(item.get("sha256") == digest for item in session["items"]):
        session["duplicates"].append({"file": original_name, "reason": "同一内容の画像です。"})
        return {"duplicate": True, "original_name": original_name}
    item_id = len(session["items"]) + 1
    suffix = extension
    filename = f"item_{item_id:04d}{suffix}"
    root = _safe_session_dir(session)
    files = root / "files"
    files.mkdir(exist_ok=True)
    image_path = files / filename
    image_path.write_bytes(normalized)
    caption_path: Path | None = None
    if caption is not None:
        caption_path = image_path.with_suffix(".txt")
        caption_path.write_text(caption, encoding="utf-8", newline="\n")
    item = {
        "id": item_id,
        "filename": filename,
        "original_name": original_name,
        "attachment_name": attachment_name,
        "sha256": digest,
        "size_bytes": len(data),
        "width": width,
        "height": height,
        "has_caption": caption_path is not None and bool(caption and caption.strip()),
        "included": True,
    }
    session["items"].append(item)
    return item


def _add_rejected(session: dict[str, Any], file_name: str, reason: str) -> None:
    session["rejected"].append({"file": file_name, "reason": reason})


def ingest_attachment(session_id: str, *, filename: str, data: bytes) -> dict[str, Any]:
    """検証済みDiscord添付の内容をセッション専用領域へ追加する。"""
    with _lock:
        session = get_session(session_id)
        if session.get("status") != "open":
            raise LoraUploadError("このアップロードセッションは受付中ではありません。")
        max_attachments = _int_env("LORA_UPLOAD_MAX_ATTACHMENTS", DEFAULT_MAX_ATTACHMENTS)
        if int(session.get("attachment_count", 0)) >= max_attachments:
            raise LoraUploadError(f"添付数が上限（{max_attachments}件）を超えています。")
        session["attachment_count"] = int(session.get("attachment_count", 0)) + 1
        if len(data) > _int_env("LORA_UPLOAD_MAX_ATTACHMENT_BYTES", DEFAULT_MAX_ATTACHMENT_BYTES):
            raise LoraUploadError("添付ファイルサイズが上限を超えています。")
        extension = Path(filename).suffix.casefold()
        if extension in IMAGE_EXTENSIONS:
            try:
                item = _store_image(session, original_name=Path(filename).name, attachment_name=filename, data=data, caption=None)
            except LoraUploadError as exc:
                _add_rejected(session, filename, str(exc))
                _save(session)
                raise
            _save(session)
            return item
        if extension != ".zip":
            _add_rejected(session, filename, "対応外ファイルのため取り込みませんでした。")
            _save(session)
            raise LoraUploadError("対応形式はPNG/JPG/JPEG/WebPまたはZIPです。")
        if len(data) > _int_env("LORA_UPLOAD_MAX_ZIP_BYTES", DEFAULT_MAX_ZIP_BYTES):
            raise LoraUploadError("ZIPのサイズが上限を超えています。")
        try:
            archive = zipfile.ZipFile(BytesIO(data))
            members = archive.infolist()
        except (OSError, zipfile.BadZipFile) as exc:
            raise LoraUploadError("ZIPが壊れているか、読み込めません。") from exc
        if not members:
            raise LoraUploadError("空のZIPは取り込めません。")
        if len(members) > _int_env("LORA_UPLOAD_MAX_ZIP_ENTRIES", DEFAULT_MAX_ZIP_ENTRIES):
            raise LoraUploadError("ZIP内のファイル数が上限を超えています。")
        total_uncompressed = 0
        safe_members: list[tuple[zipfile.ZipInfo, str]] = []
        try:
            for member in members:
                if _is_zip_symlink(member):
                    raise LoraUploadError("ZIP内のシンボリックリンクは取り込めません。")
                normalized = _normalized_zip_name(member.filename)
                if normalized.startswith("__MACOSX/") or normalized.endswith("/.DS_Store") or normalized == ".DS_Store":
                    continue
                if member.is_dir():
                    continue
                total_uncompressed += int(member.file_size)
                if total_uncompressed > _int_env("LORA_UPLOAD_MAX_ZIP_UNCOMPRESSED_BYTES", DEFAULT_MAX_ZIP_UNCOMPRESSED_BYTES):
                    raise LoraUploadError("ZIP展開後のサイズが上限を超えています。")
                safe_members.append((member, normalized))
        except (KeyError, ValueError) as exc:
            raise LoraUploadError("ZIP内のパスを安全に確認できません。") from exc
        image_members = [(member, name) for member, name in safe_members if Path(name).suffix.casefold() in IMAGE_EXTENSIONS]
        if not image_members:
            raise LoraUploadError("ZIP内に対応画像がありません。")
        embedded_files: dict[str, dict[str, str]] = {}
        canonical_names = {
            "training/dataset.toml": "dataset.toml",
            "training/config.toml": "config.toml",
            "lora_manifest.json": "lora_manifest.json",
            "training/lora_manifest.json": "lora_manifest.json",
        }
        for member, name in safe_members:
            canonical_name = canonical_names.get(name.casefold())
            if canonical_name is None:
                continue
            try:
                text = archive.read(member).decode("utf-8-sig")
                if canonical_name.endswith(".toml"):
                    tomllib.loads(text)
                else:
                    parsed_manifest = json.loads(text)
                    if not isinstance(parsed_manifest, dict):
                        raise ValueError("トップレベルはオブジェクトである必要があります")
            except (UnicodeDecodeError, json.JSONDecodeError, tomllib.TOMLDecodeError, ValueError) as exc:
                raise LoraUploadError(f"ZIP内の{canonical_name}を読み込めません: {exc}") from exc
            embedded_files[canonical_name] = {"archive_path": name, "text": text}
        if any(name in embedded_files for name in ("dataset.toml", "config.toml")) and not all(
            name in embedded_files for name in ("dataset.toml", "config.toml")
        ):
            raise LoraUploadError(
                "ZIP内の学習設計が不完全です。training/dataset.tomlとtraining/config.tomlを両方含めてください。"
            )
        text_members = [(member, name) for member, name in safe_members if Path(name).suffix.casefold() == ".txt"]
        exact_text: dict[str, list[tuple[zipfile.ZipInfo, str]]] = {}
        stem_text: dict[str, list[tuple[zipfile.ZipInfo, str]]] = {}
        for member, name in text_members:
            path = PurePosixPath(name)
            key = str(path.with_suffix("")).casefold()
            exact_text.setdefault(key, []).append((member, name))
            stem_text.setdefault(path.stem.casefold(), []).append((member, name))
        results: list[dict[str, Any]] = []
        for member, name in image_members:
            try:
                image_data = archive.read(member)
                path = PurePosixPath(name)
                caption_candidates = exact_text.get(str(path.with_suffix("")).casefold(), [])
                if not caption_candidates:
                    caption_candidates = stem_text.get(path.stem.casefold(), [])
                caption: str | None = None
                if len(caption_candidates) == 1:
                    caption = archive.read(caption_candidates[0][0]).decode("utf-8-sig")
                elif len(caption_candidates) > 1:
                    _add_rejected(session, name, "対応するcaptionが複数あるためcaptionを取り込みませんでした。")
                results.append(_store_image(session, original_name=name, attachment_name=filename, data=image_data, caption=caption))
            except (UnicodeDecodeError, KeyError) as exc:
                _add_rejected(session, name, "captionをUTF-8で読み込めませんでした。")
            except LoraUploadError as exc:
                _add_rejected(session, name, str(exc))
        if not any(not result.get("duplicate") for result in results):
            _save(session)
            raise LoraUploadError("ZIPから取り込める画像がありませんでした。")
        if embedded_files:
            root = _safe_session_dir(session)
            embedded_dir = root / "embedded_training"
            embedded_dir.mkdir(parents=True, exist_ok=True)
            for canonical_name, record in embedded_files.items():
                target = embedded_dir / canonical_name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(record["text"], encoding="utf-8", newline="\n")
            session["embedded_training"] = {
                "files": {
                    canonical_name: {
                        "archive_path": record["archive_path"],
                        "path": str((embedded_dir / canonical_name).resolve()),
                    }
                    for canonical_name, record in embedded_files.items()
                }
            }
            logger.info(
                "LoRA ZIP embedded training design stored session_id=%s files=%s",
                session_id,
                sorted(embedded_files),
            )
        _save(session)
        return {"images": results, "image_count": len([result for result in results if not result.get("duplicate")])}


def toggle_item(session_id: str, item_id: int, included: bool) -> dict[str, Any]:
    with _lock:
        session = get_session(session_id)
        if session.get("status") != "open":
            raise LoraUploadError("このセッションは受付中ではありません。")
        for item in session.get("items", []):
            if item.get("id") == int(item_id):
                item["included"] = bool(included)
                _save(session)
                return item
        raise LoraUploadError("指定された画像が見つかりません。")


def finalize_to_staging(session_id: str) -> tuple[dict[str, Any], Path]:
    with _lock:
        session = get_session(session_id)
        if session.get("status") != "open":
            raise LoraUploadError("このセッションはすでに確定済みかキャンセルされています。")
        included = included_items(session)
        if not included:
            raise LoraUploadError("確定できる画像がありません。除外状態を確認してください。")
        root = _safe_session_dir(session)
        files = root / "files"
        staging = root / "staging"
        if staging.exists():
            staging_resolved = staging.resolve()
            try:
                staging_resolved.relative_to(root)
            except ValueError as exc:
                raise LoraUploadError("staging領域が安全ではありません。") from exc
            shutil.rmtree(staging_resolved)
        staging.mkdir()
        for index, item in enumerate(included, start=1):
            source = files / str(item["filename"])
            extension = source.suffix.lower()
            original_name = str(item.get("original_name", ""))
            relative_name = PurePosixPath(original_name)
            if (
                not original_name
                or any(part in {"", ".", ".."} for part in relative_name.parts)
                or relative_name.is_absolute()
            ):
                raise LoraUploadError(f"取り込み画像の相対パスが安全ではありません: {original_name}")
            target = staging.joinpath(*relative_name.parts)
            if target.exists():
                target = staging / f"upload_{index:04d}{extension}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            caption = source.with_suffix(".txt")
            if caption.is_file():
                target.with_suffix(".txt").write_text(caption.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        embedded = session.get("embedded_training")
        if isinstance(embedded, dict) and isinstance(embedded.get("files"), dict):
            for canonical_name, record in embedded["files"].items():
                if not isinstance(record, dict):
                    continue
                source = Path(str(record.get("path", ""))).resolve()
                if not source.is_file():
                    raise LoraUploadError(f"保存済みのZIP内学習設計が見つかりません: {canonical_name}")
                target = staging / "training" / canonical_name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        return session, staging


def mark_plan_created(session_id: str, plan: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        session = get_session(session_id)
        session["status"] = "plan_created"
        session["plan"] = {key: plan[key] for key in ("info_path", "plan_dir", "output_name", "expected_output") if key in plan}
        _save(session)
        return session


def cancel_session(session_id: str) -> None:
    with _lock:
        session = get_session(session_id)
        root = _safe_session_dir(session)
        state = _state_path(session_id)
        if root.exists():
            shutil.rmtree(root)
        state.unlink(missing_ok=True)


def summary(session: dict[str, Any]) -> dict[str, Any]:
    items = list(session.get("items", []))
    included = included_items(session)
    return {
        "session_id": session.get("session_id"),
        "dataset_name": session.get("dataset_name"),
        "lora_type": session.get("lora_type"),
        "trigger_tag": session.get("trigger_tag", ""),
        "recommended_strength": session.get("recommended_strength"),
        "status": session.get("status"),
        "image_count": len(included),
        "total_items": len(items),
        "caption_count": sum(1 for item in included if item.get("has_caption")),
        "excluded_count": len(items) - len(included),
        "duplicates": list(session.get("duplicates", [])),
        "rejected": list(session.get("rejected", [])),
        "items": items,
    }


def format_summary(session: dict[str, Any], *, page: int = 0, page_size: int = 20) -> str:
    data = summary(session)
    items = data["items"]
    start = max(0, page) * page_size
    lines = [
        "LoRAアップロード受付内容",
        f"データセット: {data['dataset_name']} / 種別: {data['lora_type']}",
        f"trigger_tag: {data['trigger_tag'] or 'なし'} / 推奨強度: {data['recommended_strength']:g}",
        f"画像: {data['image_count']}枚（受付済み{data['total_items']}枚、除外{data['excluded_count']}枚）",
        f"caption: {data['caption_count']}枚 / 重複: {len(data['duplicates'])}件",
    ]
    for item in items[start : start + page_size]:
        mark = "採用" if is_item_included(item) else "除外"
        caption = "captionあり" if item.get("has_caption") else "captionなし"
        lines.append(f"{item['id']}. [{mark}] {item['original_name']} ({item['width']}x{item['height']}, {caption})")
    if items[start + page_size :]:
        lines.append(f"（{page + 1}ページ目。次ページあり）")
    if data["rejected"]:
        lines.append("取り込まなかったファイル: " + "、".join(str(item["file"]) for item in data["rejected"][:5]))
    return "\n".join(lines)[:1900]
