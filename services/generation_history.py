import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from services.sd_control import GeneratedImage
from services.weather import JST


logger = logging.getLogger("secretary-bot.generation_history")

DATA_DIR = Path("data")
DB_FILE = DATA_DIR / "generation_history.db"

# Schema generation recorded in the DB itself via PRAGMA user_version, so a
# future change can tell which shape an existing file has and migrate it.
# Bump this whenever the CREATE TABLE statements below change, and add the
# matching migration step to _apply_schema_version().
SCHEMA_VERSION = 1

_HASH_CHUNK_SIZE = 1024 * 1024

# The only rating values image_ratings.rating may hold. Anything else is
# rejected before the DB is touched.
ALLOWED_RATINGS = ("like", "meh")


@dataclass(frozen=True)
class GenerationRequestRecord:
    user_id: int
    request_text: str
    reference_id: int | None
    positive: str
    negative: str
    summary: str
    assumptions: Sequence[str]
    workflow_file: str | None
    workflow_hash: str | None
    selection_used: bool
    selection_message: str


@dataclass(frozen=True)
class SavedGeneration:
    request_id: int
    # Index-aligned with the images passed to save_generation_history_detailed,
    # so image_ids[i] is the row id of images[i]. This alignment is what lets
    # sent images be tied back to their DB rows without guessing.
    image_ids: list[int]


def _now_text() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_FILE) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS generation_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                request_text TEXT NOT NULL,
                reference_id INTEGER,
                positive TEXT NOT NULL,
                negative TEXT NOT NULL,
                summary TEXT NOT NULL,
                assumptions_json TEXT NOT NULL,
                workflow_file TEXT,
                workflow_hash TEXT,
                selection_used INTEGER NOT NULL DEFAULT 0,
                selection_message TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS generation_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id INTEGER NOT NULL REFERENCES generation_requests(id) ON DELETE CASCADE,
                image_path TEXT NOT NULL,
                seed INTEGER,
                batch_index INTEGER NOT NULL,
                checkpoint_name TEXT,
                sampler_name TEXT,
                scheduler TEXT,
                steps INTEGER,
                cfg REAL,
                width INTEGER,
                height INTEGER,
                was_selected_by_ai INTEGER NOT NULL DEFAULT 0,
                discord_message_id INTEGER,
                discord_channel_id INTEGER
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS image_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL REFERENCES generation_images(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL,
                rating TEXT NOT NULL,
                rated_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(image_id, user_id)
            )
            """
        )
        _apply_schema_version(connection)
        connection.commit()


def _apply_schema_version(connection: sqlite3.Connection) -> None:
    """Stamps/checks PRAGMA user_version so future schema changes have a
    reliable version to migrate from.

    PRAGMA user_version does not accept parameter binding, so SCHEMA_VERSION
    is interpolated - it is a module-level int constant, never user input.
    """
    current = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if current == SCHEMA_VERSION:
        return
    if current == 0:
        # Either a brand-new file, or one written before versioning existed.
        # The CREATE TABLE IF NOT EXISTS statements above have just brought
        # it to the current shape either way, so stamping it is safe.
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return
    if current > SCHEMA_VERSION:
        logger.warning(
            "generation_history.db reports schema version %s, newer than this build expects (%s). "
            "Leaving the version untouched.",
            current,
            SCHEMA_VERSION,
        )
        return
    # Reachable only once SCHEMA_VERSION > 1. Deliberately does not bump the
    # version: silently relabelling an un-migrated file would hide the gap.
    logger.warning(
        "generation_history.db schema version %s is older than %s and no migration step is defined yet.",
        current,
        SCHEMA_VERSION,
    )


def _connect() -> sqlite3.Connection:
    # Mirrors daily_log.py's _connect(): init_db() is idempotent (CREATE
    # TABLE IF NOT EXISTS) and cheap, so calling it on every connect keeps
    # this module usable even if the DB file was deleted between calls.
    init_db()
    connection = sqlite3.connect(DB_FILE)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def _resolved(path: Path) -> Path | None:
    try:
        return path.resolve()
    except OSError:
        return None


def _file_digest(path: Path) -> str | None:
    """SHA-256 of the file's bytes, or None if it cannot be read."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        logger.warning("Could not read image while matching Selection AI output: path=%s", path)
        return None


def _build_selection_index(selected_paths: Sequence[Path]) -> tuple[set[Path], set[str]]:
    """Indexes Selection AI's chosen files by resolved path and by content
    digest, so each generated image can be tested against them exactly."""
    resolved_paths: set[Path] = set()
    digests: set[str] = set()
    for raw_candidate in selected_paths:
        candidate = Path(raw_candidate)
        resolved = _resolved(candidate)
        if resolved is not None:
            resolved_paths.add(resolved)
        digest = _file_digest(candidate)
        if digest is not None:
            digests.add(digest)
    return resolved_paths, digests


def _is_selected(image_path: Path, selected_paths: set[Path], selected_digests: set[str]) -> bool:
    """True only on an exact identity match against Selection AI's output.

    Two checks, both exact - no filename guessing:

    1. Resolved-path equality, which covers the "selection unavailable, fall
       back to the original paths" case where selected_paths *are* the
       originals.
    2. SHA-256 content equality. The external score_batch_top5.py script
       (outside this repo) copies its chosen images into its own folder under
       a name this repo cannot predict, but every step of that chain is a
       byte-preserving copy (shutil.copy2), so the content digest is a
       reliable identity test where the filename is not.

    A file that cannot be read is reported as not selected rather than
    guessed at; _file_digest logs when that happens.
    """
    resolved = _resolved(image_path)
    if resolved is not None and resolved in selected_paths:
        return True
    if not selected_digests:
        return False
    digest = _file_digest(image_path)
    return digest is not None and digest in selected_digests


def save_generation_history_detailed(
    request: GenerationRequestRecord,
    images: Sequence[GeneratedImage],
    *,
    selected_paths: Sequence[Path] = (),
) -> SavedGeneration | None:
    """Persists one /generate_ai request and all of its generated images in
    a single transaction. Returns the new row ids on success.

    Never raises: any failure (DB corruption, disk full, locked file, ...)
    is caught, logged, and reported back as None so callers can continue
    sending the Discord result to the user regardless of whether history
    was recorded.
    """
    try:
        # Built once per request; skipped entirely when Selection AI did not
        # run, so no image is hashed needlessly.
        selected_resolved, selected_digests = (
            _build_selection_index(selected_paths) if request.selection_used else (set(), set())
        )
        with _connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO generation_requests (
                    created_at, user_id, request_text, reference_id,
                    positive, negative, summary, assumptions_json,
                    workflow_file, workflow_hash, selection_used, selection_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now_text(),
                    request.user_id,
                    request.request_text,
                    request.reference_id,
                    request.positive,
                    request.negative,
                    request.summary,
                    json.dumps(list(request.assumptions), ensure_ascii=False),
                    request.workflow_file,
                    request.workflow_hash,
                    1 if request.selection_used else 0,
                    request.selection_message,
                ),
            )
            request_id = int(cursor.lastrowid)

            image_ids: list[int] = []
            for image in images:
                was_selected = request.selection_used and _is_selected(
                    image.path, selected_resolved, selected_digests
                )
                image_cursor = connection.execute(
                    """
                    INSERT INTO generation_images (
                        request_id, image_path, seed, batch_index,
                        checkpoint_name, sampler_name, scheduler, steps, cfg,
                        width, height, was_selected_by_ai,
                        discord_message_id, discord_channel_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        str(image.path),
                        image.seed,
                        image.batch_index,
                        image.checkpoint_name,
                        image.sampler_name,
                        image.scheduler,
                        image.steps,
                        image.cfg,
                        image.width,
                        image.height,
                        1 if was_selected else 0,
                        None,
                        None,
                    ),
                )
                image_ids.append(int(image_cursor.lastrowid))
            # sqlite3.Connection's context manager commits here on a clean
            # exit, or rolls back the whole transaction (request + images
            # together) if any of the statements above raised.
        logger.info(
            "Generation history saved: request_id=%s image_count=%s selection_used=%s",
            request_id,
            len(images),
            request.selection_used,
        )
        return SavedGeneration(request_id=request_id, image_ids=image_ids)
    except Exception:
        logger.exception("Failed to save generation history; continuing without blocking Discord delivery.")
        return None


def save_generation_history(
    request: GenerationRequestRecord,
    images: Sequence[GeneratedImage],
    *,
    selected_paths: Sequence[Path] = (),
) -> int | None:
    """Backward-compatible wrapper returning only the request id."""
    saved = save_generation_history_detailed(request, images, selected_paths=selected_paths)
    return None if saved is None else saved.request_id


def resolve_image_ids_for_paths(
    images: Sequence[GeneratedImage],
    image_ids: Sequence[int],
    paths: Sequence[Path],
) -> list[int | None]:
    """Maps each path actually sent to Discord back to its generation_images
    row id, returning None for any path that cannot be matched.

    Uses the same exact-identity rules as _is_selected - resolved-path
    equality first, then SHA-256 content equality - so a path that went
    through Selection AI's rename chain still resolves. Filenames are never
    used to guess. `image_ids` is index-aligned with `images`, which is how
    save_generation_history_detailed returns them.
    """
    by_resolved: dict[Path, int] = {}
    by_digest: dict[str, int] = {}
    for image, image_id in zip(images, image_ids):
        resolved = _resolved(image.path)
        if resolved is not None:
            by_resolved.setdefault(resolved, image_id)
        digest = _file_digest(image.path)
        if digest is not None:
            # setdefault: if two generated images are byte-identical, the
            # first one wins rather than the mapping flip-flopping.
            by_digest.setdefault(digest, image_id)

    matched: list[int | None] = []
    for raw_path in paths:
        path = Path(raw_path)
        resolved = _resolved(path)
        if resolved is not None and resolved in by_resolved:
            matched.append(by_resolved[resolved])
            continue
        digest = _file_digest(path)
        if digest is not None and digest in by_digest:
            matched.append(by_digest[digest])
            continue
        logger.warning("Sent image could not be matched to a generation_images row: path=%s", path)
        matched.append(None)
    return matched


def upsert_image_rating(image_id: int, user_id: int, rating: str) -> bool:
    """Records one user's rating for one image, replacing that user's
    previous rating if there is one. Returns True only when stored.

    rated_at keeps the first-rating timestamp; updated_at moves every time,
    including when the rating value changes. Never raises: an unknown rating,
    a missing image_id (foreign key violation) or any DB error is logged and
    reported as False so the caller can show a short message instead.
    """
    if rating not in ALLOWED_RATINGS:
        logger.warning("Rejected unknown rating value: rating=%r image_id=%s user_id=%s", rating, image_id, user_id)
        return False
    try:
        now = _now_text()
        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO image_ratings (image_id, user_id, rating, rated_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(image_id, user_id) DO UPDATE SET
                    rating = excluded.rating,
                    updated_at = excluded.updated_at
                """,
                (image_id, user_id, rating, now, now),
            )
        return True
    except Exception:
        logger.exception("Failed to save image rating: image_id=%s user_id=%s rating=%s", image_id, user_id, rating)
        return False


def set_image_discord_message(
    image_ids: Sequence[int],
    message_id: int,
    channel_id: int,
) -> bool:
    """Attaches the Discord message/channel the images were posted in.

    Best-effort: returns False on any failure instead of raising, so a
    bookkeeping problem never breaks image delivery or rating.
    """
    ids = [int(image_id) for image_id in image_ids if image_id is not None]
    if not ids:
        return False
    try:
        with _connect() as connection:
            connection.executemany(
                "UPDATE generation_images SET discord_message_id = ?, discord_channel_id = ? WHERE id = ?",
                [(int(message_id), int(channel_id), image_id) for image_id in ids],
            )
        return True
    except Exception:
        logger.exception(
            "Failed to store Discord message info: message_id=%s channel_id=%s image_count=%s",
            message_id,
            channel_id,
            len(ids),
        )
        return False
