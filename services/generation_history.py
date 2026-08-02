import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

from services.image_generation_profiles import (
    DEFAULT_IMAGE_GENERATION_MODEL_KEY,
    get_image_generation_profile,
    get_model_key_for_checkpoint,
)
from services.sd_control import GeneratedImage
from services.weather import JST


logger = logging.getLogger("secretary-bot.generation_history")

DATA_DIR = Path("data")
DB_FILE = DATA_DIR / "generation_history.db"

# Schema generation recorded in the DB itself via PRAGMA user_version, so a
# future change can tell which shape an existing file has and migrate it.
# Bump this whenever the CREATE TABLE statements below change, and add the
# matching migration step to _apply_schema_version().
SCHEMA_VERSION = 5

_HASH_CHUNK_SIZE = 1024 * 1024

# The only rating values image_ratings.rating may hold. Anything else is
# rejected before the DB is touched. Phase 1D added "favorite" alongside the
# existing three - the column is untyped TEXT, so no schema change was
# needed, and existing like/meh/off_target rows are unaffected.
ALLOWED_RATINGS = ("favorite", "like", "meh", "off_target")

# Self-improvement Phase 1: the only issue_categories_json entries
# image_generation_feedback.issue_categories_json may hold. Deliberately
# disjoint from ALLOWED_RATINGS above - image_ratings is a separate, lighter
# thumbs-up/down-style signal used by selection/prompt preference, while
# image_generation_feedback records structured, multi-select adoption
# feedback plus a snapshot of the generation conditions. Neither table reads
# from or writes to the other.
ALLOWED_ISSUE_CATEGORIES = (
    "face_mismatch",
    "hair_mismatch",
    "outfit_mismatch",
    "composition_mismatch",
    "low_quality",
    "other",
)

# The only regeneration_type values generation_requests.regeneration_type
# may hold (besides NULL, for a request that is not a regeneration at all).
REGENERATION_TYPES = ("same_prompt", "change_composition", "custom_instruction")

# Phase 2A: /generate_ai_stats. NULL (not a regeneration) is reported under
# this label rather than as a Python None, so it can be used directly as a
# dict key and a display label.
NORMAL_GENERATION_LABEL = "normal"

# Phase 2A minimum sample sizes, below which stats are still computed but
# comparative/trend commentary is withheld rather than overclaiming from a
# handful of ratings.
MIN_RATED_FOR_TREND = 10
MIN_GROUP_SIZE_FOR_TAG_COMPARISON = 3

# Phase 2A tag-trend scanning: a fixed, curated set of tags/phrases this
# repo's own Prompt Designer instruction (shogou_chan_prompt_designer_instruction.md,
# Phase 1E) is guided to use for lighting/color/framing/scene. This is a
# lightweight first pass over stored text - not image analysis - so it is
# deliberately a closed list rather than free-form keyword extraction.
# Matched case-insensitively with word boundaries so e.g. "day" does not
# match inside "today", and "night" does not match inside some longer tag.
WATCHED_TAG_PHRASES: tuple[str, ...] = (
    # color temperature
    "warm lighting", "warm daylight", "soft amber light", "golden highlights",
    "warm sunset lighting", "soft golden rim light", "neutral lighting",
    "natural lighting", "neutral to warm daylight", "balanced color palette",
    "cool lighting", "cool colors", "blue tones", "controlled cool lighting",
    # brightness
    "bright lighting", "bright daylight", "bright natural lighting",
    "low light", "moody lighting",
    # backlight / rim light
    "backlight", "rim light", "subtle backlight", "soft rim light",
    "strong backlight", "dramatic backlighting", "intense rim light",
    # indoor / outdoor
    "indoor", "outdoor", "soft warm indoor lighting",
    # time of day
    "daylight", "sunset", "night", "evening",
    # framing
    "close-up", "close up", "upper body", "full body", "cowboy shot", "portrait",
    # scene
    "school uniform", "serafuku", "classroom", "cafe", "coffee shop",
    "beach", "ocean", "bedroom", "living room",
)
_WATCHED_TAG_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = tuple(
    (phrase, re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)) for phrase in WATCHED_TAG_PHRASES
)


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
    # Phase 1C: set together when this request is a regeneration of an
    # earlier one. Both stay None for a normal (non-regenerated) request.
    parent_request_id: int | None = None
    regeneration_type: str | None = None
    # Only set when regeneration_type == "custom_instruction": the free-text
    # instruction the user typed into the modal (e.g. "髪型をポニーテールに").
    # None for every other regeneration_type and for normal requests.
    custom_instruction: str | None = None
    model_key: str = DEFAULT_IMAGE_GENERATION_MODEL_KEY


@dataclass(frozen=True)
class SavedGeneration:
    request_id: int
    # Index-aligned with the images passed to save_generation_history_detailed,
    # so image_ids[i] is the row id of images[i]. This alignment is what lets
    # sent images be tied back to their DB rows without guessing.
    image_ids: list[int]


@dataclass(frozen=True)
class GenerationRequestDetail:
    """Read-only view of one generation_requests row, e.g. for regeneration."""

    id: int
    created_at: str
    user_id: int
    request_text: str
    reference_id: int | None
    positive: str
    negative: str
    summary: str
    assumptions: tuple[str, ...]
    workflow_file: str | None
    workflow_hash: str | None
    selection_used: bool
    selection_message: str
    parent_request_id: int | None
    regeneration_type: str | None
    custom_instruction: str | None
    model_key: str = DEFAULT_IMAGE_GENERATION_MODEL_KEY


@dataclass(frozen=True)
class FeedbackImageTarget:
    """Read-only: one generation_images row plus its parent request's
    owner, joined by (channel_id, message_id) - used by the image feedback
    context menu to find which image(s) a Discord message is offering
    without requiring the caller to know any internal id. Carries every
    generation_images column the feedback snapshot needs, unlike the leaner
    GenerationImageRecord used by the "他の候補を見る" feature.
    """

    image_id: int
    request_id: int
    request_user_id: int
    image_index: int
    image_path: str
    seed: int | None
    checkpoint_name: str | None
    sampler_name: str | None
    scheduler: str | None
    steps: int | None
    cfg: float | None
    width: int | None
    height: int | None
    was_selected_by_ai: bool


@dataclass(frozen=True)
class ImageFeedbackRecord:
    """Read-only view of one image_generation_feedback row."""

    id: int
    image_id: int
    image_index: int
    image_path: str
    user_id: int
    guild_id: int | None
    channel_id: int | None
    message_id: int | None
    accepted: bool
    issue_categories: tuple[str, ...]
    comment: str | None
    generation_snapshot: dict
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class RatingCounts:
    """Phase 2A: how many of a set of rated images fall into each of the
    four rating values. "Good" (favorite+like) and "low" (meh+off_target)
    are convenience roll-ups for display - the four original categories are
    always available too, and off_target is never merged into a generic
    quality bucket, since it represents a request mismatch rather than
    low image quality.
    """

    favorite: int = 0
    like: int = 0
    meh: int = 0
    off_target: int = 0

    @property
    def total(self) -> int:
        return self.favorite + self.like + self.meh + self.off_target

    @property
    def good(self) -> int:
        return self.favorite + self.like

    @property
    def low(self) -> int:
        return self.meh + self.off_target


def _rating_counts_from_ratings(ratings: Sequence[str]) -> RatingCounts:
    counts = {"favorite": 0, "like": 0, "meh": 0, "off_target": 0}
    for rating in ratings:
        if rating in counts:
            counts[rating] += 1
    return RatingCounts(**counts)


@dataclass(frozen=True)
class RegenerationTypeStats:
    """Phase 2A: one row of the "re-generation trend" breakdown - how many
    generation_requests of this type exist, and how the caller's own ratings
    on their images are distributed. regeneration_type is NORMAL_GENERATION_LABEL
    for a request that is not a regeneration (NULL in the DB)."""

    regeneration_type: str
    request_count: int
    ratings: RatingCounts


@dataclass(frozen=True)
class TagTrendEntry:
    """Phase 2A: how often one watched tag/phrase appears in the prompt text
    of images the caller rated favorite/like ("good") versus meh/off_target
    ("low"). Counted at most once per rated image, even if the underlying
    request's text repeats the phrase, and even if the phrase shows up in
    more than one of positive/negative/summary/assumptions/request_text for
    that same image."""

    phrase: str
    good_count: int
    low_count: int


@dataclass(frozen=True)
class GenerationStats:
    """Phase 2A: /generate_ai_stats aggregate for one user. Every count here
    is scoped to that user's own generation_requests AND that same user's own
    image_ratings rows - another user's rating on this user's image is not
    counted, matching "実行者本人の評価履歴のみ集計"."""

    total_requests: int
    total_images: int
    rated_images: int
    unrated_images: int
    ratings_all_time: RatingCounts
    ratings_7d: RatingCounts
    ratings_30d: RatingCounts
    regeneration_breakdown: tuple[RegenerationTypeStats, ...]
    tag_trend: tuple[TagTrendEntry, ...]
    good_sample_count: int
    low_sample_count: int


def _extract_watched_tags(text: str) -> set[str]:
    """Returns the set of WATCHED_TAG_PHRASES present in text, each phrase
    counted at most once regardless of how many times it appears."""
    return {phrase for phrase, pattern in _WATCHED_TAG_PATTERNS if pattern.search(text)}


def _now_text() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_FILE) as connection:
        # _migrate_v1_to_v2 (via _add_column_if_missing) reads PRAGMA
        # table_info rows by column name, so this connection needs the same
        # row_factory as _connect()'s.
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS generation_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                request_text TEXT NOT NULL,
                reference_id INTEGER,
                model_key TEXT,
                positive TEXT NOT NULL,
                negative TEXT NOT NULL,
                summary TEXT NOT NULL,
                assumptions_json TEXT NOT NULL,
                workflow_file TEXT,
                workflow_hash TEXT,
                selection_used INTEGER NOT NULL DEFAULT 0,
                selection_message TEXT,
                parent_request_id INTEGER,
                regeneration_type TEXT,
                custom_instruction TEXT
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
        # Self-improvement Phase 1 (image feedback): structured, multi-select
        # adoption feedback ("採用"/problem categories/free comment) plus a
        # frozen snapshot of the generation conditions at the time feedback
        # was recorded, so a later change to prompts/dictionary/models never
        # rewrites what an earlier evaluation was actually judging. A brand
        # new table needs no ALTER-based migration - CREATE TABLE IF NOT
        # EXISTS alone brings an older DB file up to date - so
        # _migrate_v4_to_v5 below is a documented no-op.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS image_generation_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL REFERENCES generation_images(id) ON DELETE CASCADE,
                image_index INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                guild_id INTEGER,
                channel_id INTEGER,
                message_id INTEGER,
                accepted INTEGER NOT NULL DEFAULT 0,
                issue_categories_json TEXT NOT NULL DEFAULT '[]',
                comment TEXT,
                generation_snapshot_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(image_id, user_id)
            )
            """
        )
        _apply_schema_version(connection)
        connection.commit()


def _add_column_if_missing(connection: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
    existing = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        # No DEFAULT clause, so SQLite backfills every existing row with
        # NULL - exactly what a pre-Phase-1C row should have for both of
        # these columns.
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    """Adds parent_request_id/regeneration_type to generation_requests.

    Additive only (no data is rewritten or dropped), and guarded by
    PRAGMA table_info so re-running it against an already-migrated file
    (e.g. one a fresh CREATE TABLE already brought to the v2 shape) is a
    no-op rather than a "duplicate column" error.
    """
    _add_column_if_missing(connection, "generation_requests", "parent_request_id", "INTEGER")
    _add_column_if_missing(connection, "generation_requests", "regeneration_type", "TEXT")


def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
    """Adds custom_instruction to generation_requests (additive, guarded the
    same way as _migrate_v1_to_v2 - a no-op if the column already exists)."""
    _add_column_if_missing(connection, "generation_requests", "custom_instruction", "TEXT")


def _migrate_v3_to_v4(connection: sqlite3.Connection) -> None:
    """Adds the selected /generate_ai model key without rewriting old rows.

    Existing records remain NULL and are resolved from their stored image
    checkpoint when read, with Animagine as the compatibility fallback.
    """
    _add_column_if_missing(connection, "generation_requests", "model_key", "TEXT")


def _migrate_v4_to_v5(connection: sqlite3.Connection) -> None:
    """No-op: image_generation_feedback is a brand new table, already created
    by the unconditional CREATE TABLE IF NOT EXISTS in init_db() regardless
    of the file's prior version. This step exists only so SCHEMA_VERSION's
    "bump on every CREATE TABLE change" rule has a matching migration entry,
    and so PRAGMA user_version reaches 5 the same way every other version
    bump does.
    """
    return


def _apply_schema_version(connection: sqlite3.Connection) -> None:
    """Stamps/checks PRAGMA user_version so future schema changes have a
    reliable version to migrate from.

    PRAGMA user_version does not accept parameter binding, so SCHEMA_VERSION
    is interpolated - it is a module-level int constant, never user input.
    Any failure raised while migrating propagates to the caller's try/except
    (init_db() is only ever reached through _connect(), which every public
    function in this module wraps), so a broken migration is reported as a
    failure rather than silently corrupting or half-upgrading the file: the
    whole connection's transaction (including the CREATE TABLE IF NOT EXISTS
    statements above) rolls back together.
    """
    current = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if current == SCHEMA_VERSION:
        return
    if current == 0:
        # Either a brand-new file, or one written before versioning existed.
        # Apply every additive, guarded migration too: CREATE TABLE IF NOT
        # EXISTS does not add columns to an old unversioned table.
        _migrate_v1_to_v2(connection)
        _migrate_v2_to_v3(connection)
        _migrate_v3_to_v4(connection)
        _migrate_v4_to_v5(connection)
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
    if current == 1:
        _migrate_v1_to_v2(connection)
        _migrate_v2_to_v3(connection)
        _migrate_v3_to_v4(connection)
        _migrate_v4_to_v5(connection)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return
    if current == 2:
        _migrate_v2_to_v3(connection)
        _migrate_v3_to_v4(connection)
        _migrate_v4_to_v5(connection)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return
    if current == 3:
        _migrate_v3_to_v4(connection)
        _migrate_v4_to_v5(connection)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return
    if current == 4:
        _migrate_v4_to_v5(connection)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return
    # Deliberately does not bump an unknown older version: silently
    # relabelling an un-migrated file would hide the gap.
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
            resolved_parent_id = None
            if request.parent_request_id is not None:
                parent_exists = connection.execute(
                    "SELECT 1 FROM generation_requests WHERE id = ?", (request.parent_request_id,)
                ).fetchone()
                if parent_exists is not None:
                    resolved_parent_id = int(request.parent_request_id)
                else:
                    logger.warning(
                        "parent_request_id does not exist; storing NULL instead: parent_request_id=%s",
                        request.parent_request_id,
                    )

            resolved_regeneration_type = request.regeneration_type
            if resolved_regeneration_type is not None and resolved_regeneration_type not in REGENERATION_TYPES:
                logger.warning("Rejected unknown regeneration_type value: %r", resolved_regeneration_type)
                resolved_regeneration_type = None

            cursor = connection.execute(
                """
                INSERT INTO generation_requests (
                    created_at, user_id, request_text, reference_id, model_key,
                    positive, negative, summary, assumptions_json,
                    workflow_file, workflow_hash, selection_used, selection_message,
                    parent_request_id, regeneration_type, custom_instruction
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now_text(),
                    request.user_id,
                    request.request_text,
                    request.reference_id,
                    request.model_key,
                    request.positive,
                    request.negative,
                    request.summary,
                    json.dumps(list(request.assumptions), ensure_ascii=False),
                    request.workflow_file,
                    request.workflow_hash,
                    1 if request.selection_used else 0,
                    request.selection_message,
                    resolved_parent_id,
                    resolved_regeneration_type,
                    request.custom_instruction,
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


def get_generation_images_for_message(channel_id: int, message_id: int) -> list[FeedbackImageTarget]:
    """Read-only: every generation_images row attached to one Discord
    message (via the discord_message_id/discord_channel_id set by
    set_image_discord_message), ordered by batch_index - i.e. attachment
    order. Used by the image feedback context menu, which acts on a message
    rather than a slash-command argument, so it never needs the caller to
    know any internal id.

    Never raises: any DB error is logged and reported as an empty list,
    matching every other read function in this module.
    """
    try:
        with _connect() as connection:
            rows = connection.execute(
                """
                SELECT gi.id AS id, gi.request_id AS request_id, gr.user_id AS request_user_id,
                       gi.batch_index AS batch_index, gi.image_path AS image_path, gi.seed AS seed,
                       gi.checkpoint_name AS checkpoint_name, gi.sampler_name AS sampler_name,
                       gi.scheduler AS scheduler, gi.steps AS steps, gi.cfg AS cfg,
                       gi.width AS width, gi.height AS height, gi.was_selected_by_ai AS was_selected_by_ai
                FROM generation_images gi
                JOIN generation_requests gr ON gr.id = gi.request_id
                WHERE gi.discord_channel_id = ? AND gi.discord_message_id = ?
                ORDER BY gi.batch_index ASC, gi.id ASC
                """,
                (channel_id, message_id),
            ).fetchall()
        return [
            FeedbackImageTarget(
                image_id=int(row["id"]),
                request_id=int(row["request_id"]),
                request_user_id=int(row["request_user_id"]),
                image_index=int(row["batch_index"]),
                image_path=row["image_path"],
                seed=row["seed"],
                checkpoint_name=row["checkpoint_name"],
                sampler_name=row["sampler_name"],
                scheduler=row["scheduler"],
                steps=row["steps"],
                cfg=row["cfg"],
                width=row["width"],
                height=row["height"],
                was_selected_by_ai=bool(row["was_selected_by_ai"]),
            )
            for row in rows
        ]
    except Exception:
        logger.exception(
            "Failed to read generation images for message: channel_id=%s message_id=%s", channel_id, message_id
        )
        return []


def _image_feedback_record_from_row(row: sqlite3.Row) -> ImageFeedbackRecord:
    try:
        issue_categories = tuple(json.loads(row["issue_categories_json"] or "[]"))
    except (TypeError, json.JSONDecodeError):
        issue_categories = ()
    try:
        generation_snapshot = json.loads(row["generation_snapshot_json"] or "{}")
    except (TypeError, json.JSONDecodeError):
        generation_snapshot = {}
    return ImageFeedbackRecord(
        id=int(row["id"]),
        image_id=int(row["image_id"]),
        image_index=int(row["image_index"]),
        image_path=row["image_path"],
        user_id=int(row["user_id"]),
        guild_id=row["guild_id"],
        channel_id=row["channel_id"],
        message_id=row["message_id"],
        accepted=bool(row["accepted"]),
        issue_categories=issue_categories,
        comment=row["comment"],
        generation_snapshot=generation_snapshot,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def upsert_image_feedback(
    *,
    image_id: int,
    image_index: int,
    image_path: str,
    user_id: int,
    guild_id: int | None,
    channel_id: int | None,
    message_id: int | None,
    accepted: bool,
    issue_categories: Sequence[str],
    comment: str | None,
    generation_snapshot_json: str,
) -> bool:
    """Records one user's feedback for one image, replacing that user's
    previous feedback for the same image if there is one (full replace, not
    a merge - matches upsert_image_rating's semantics). Returns True only
    when stored.

    Validation (never raises - an invalid call is rejected and logged, the
    same way an unknown rating is rejected by upsert_image_rating):
    - every entry in issue_categories must be one of ALLOWED_ISSUE_CATEGORIES
    - accepted feedback and problem-category feedback are mutually
      exclusive - "採用" clears any categories, and any categories supplied
      together with accepted=True is rejected outright rather than silently
      dropped, so a caller bug is visible instead of quietly losing data
    - at least one of accepted / issue_categories must be given - a row with
      neither would record nothing meaningful

    created_at keeps the first-feedback timestamp; updated_at moves every
    time, including when only the comment or snapshot changes.
    """
    categories = list(dict.fromkeys(issue_categories))  # de-dup, preserve order
    unknown = [category for category in categories if category not in ALLOWED_ISSUE_CATEGORIES]
    if unknown:
        logger.warning("Rejected unknown issue_categories value(s): %r image_id=%s user_id=%s", unknown, image_id, user_id)
        return False
    if accepted and categories:
        logger.warning(
            "Rejected feedback with both accepted=True and issue_categories: image_id=%s user_id=%s categories=%r",
            image_id, user_id, categories,
        )
        return False
    if not accepted and not categories:
        logger.warning("Rejected feedback with neither accepted nor any issue_categories: image_id=%s user_id=%s", image_id, user_id)
        return False

    try:
        now = _now_text()
        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO image_generation_feedback (
                    image_id, image_index, image_path, user_id, guild_id, channel_id, message_id,
                    accepted, issue_categories_json, comment, generation_snapshot_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id, user_id) DO UPDATE SET
                    image_index = excluded.image_index,
                    image_path = excluded.image_path,
                    guild_id = excluded.guild_id,
                    channel_id = excluded.channel_id,
                    message_id = excluded.message_id,
                    accepted = excluded.accepted,
                    issue_categories_json = excluded.issue_categories_json,
                    comment = excluded.comment,
                    generation_snapshot_json = excluded.generation_snapshot_json,
                    updated_at = excluded.updated_at
                """,
                (
                    image_id,
                    image_index,
                    image_path,
                    user_id,
                    guild_id,
                    channel_id,
                    message_id,
                    1 if accepted else 0,
                    json.dumps(categories, ensure_ascii=False),
                    comment,
                    generation_snapshot_json,
                    now,
                    now,
                ),
            )
        return True
    except Exception:
        logger.exception("Failed to save image feedback: image_id=%s user_id=%s", image_id, user_id)
        return False


def get_feedback_for_generation(image_id: int) -> list[ImageFeedbackRecord]:
    """Read-only: every user's feedback for one generation_images row, most
    recently updated first. Never raises: a DB error is logged and reported
    as an empty list.
    """
    try:
        with _connect() as connection:
            rows = connection.execute(
                "SELECT * FROM image_generation_feedback WHERE image_id = ? ORDER BY updated_at DESC",
                (image_id,),
            ).fetchall()
        return [_image_feedback_record_from_row(row) for row in rows]
    except Exception:
        logger.exception("Failed to read image feedback: image_id=%s", image_id)
        return []


def get_feedback_by_user_and_image(image_id: int, user_id: int) -> ImageFeedbackRecord | None:
    """Read-only: one user's feedback for one image, or None if they have
    not left any (or on DB error - same "never raises" convention as
    get_generation_image)."""
    try:
        with _connect() as connection:
            row = connection.execute(
                "SELECT * FROM image_generation_feedback WHERE image_id = ? AND user_id = ?",
                (image_id, user_id),
            ).fetchone()
        return None if row is None else _image_feedback_record_from_row(row)
    except Exception:
        logger.exception("Failed to read image feedback: image_id=%s user_id=%s", image_id, user_id)
        return None


# Defensive cap on one aggregation pass over image_generation_feedback,
# mirroring REQUEST_TEXT_SCAN_ROW_LIMIT above - a personal-bot-sized history
# is expected to stay far under this, but an unbounded SELECT * must never be
# possible from a stats command.
FEEDBACK_SCAN_ROW_LIMIT = 5000


def list_all_feedback_rows(limit: int = FEEDBACK_SCAN_ROW_LIMIT) -> list[ImageFeedbackRecord]:
    """Read-only: up to `limit` most recently updated image_generation_feedback
    rows, across every user - used by get_character_feedback_summary and
    get_common_failure_categories, which aggregate across everyone's
    feedback rather than one user's own (a character's quality is not
    specific to who happened to press the button).

    Never raises: a DB error is logged and reported as an empty list, so an
    aggregation feature can never crash the bot.
    """
    try:
        with _connect() as connection:
            rows = connection.execute(
                "SELECT * FROM image_generation_feedback ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_image_feedback_record_from_row(row) for row in rows]
    except Exception:
        logger.exception("Failed to read image feedback rows for aggregation.")
        return []


def list_feedback_for_user(user_id: int, limit: int = 10) -> list[ImageFeedbackRecord]:
    """Read-only: this user's own most recently updated feedback rows, for
    /image_feedback_history. Never raises: a DB error is logged and reported
    as an empty list.
    """
    try:
        with _connect() as connection:
            rows = connection.execute(
                "SELECT * FROM image_generation_feedback WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [_image_feedback_record_from_row(row) for row in rows]
    except Exception:
        logger.exception("Failed to read image feedback for user: user_id=%s", user_id)
        return []


def _resolve_request_model_key(connection: sqlite3.Connection, row: sqlite3.Row) -> str:
    stored_key = row["model_key"]
    if stored_key:
        try:
            return get_image_generation_profile(str(stored_key)).model_key
        except ValueError:
            logger.warning("Unknown stored generation model key; falling back from checkpoint: %r", stored_key)

    checkpoint_rows = connection.execute(
        "SELECT DISTINCT checkpoint_name FROM generation_images WHERE request_id = ? AND checkpoint_name IS NOT NULL",
        (int(row["id"]),),
    ).fetchall()
    if len(checkpoint_rows) == 1:
        inferred = get_model_key_for_checkpoint(checkpoint_rows[0]["checkpoint_name"])
        if inferred is not None:
            return inferred
    return DEFAULT_IMAGE_GENERATION_MODEL_KEY


def get_generation_request(request_id: int) -> GenerationRequestDetail | None:
    """Read-only lookup of one generation_requests row (Phase 1C: used to
    look up the original request a regeneration button should build from).

    Never raises: a missing row returns None like any other lookup miss, and
    any DB error (corruption, locked file, ...) is logged and also reported
    as None rather than letting a raw sqlite3 exception reach the caller.
    """
    try:
        with _connect() as connection:
            row = connection.execute(
                "SELECT * FROM generation_requests WHERE id = ?", (request_id,)
            ).fetchone()
            if row is None:
                return None
            model_key = _resolve_request_model_key(connection, row)
        return GenerationRequestDetail(
            id=int(row["id"]),
            created_at=row["created_at"],
            user_id=int(row["user_id"]),
            request_text=row["request_text"],
            reference_id=row["reference_id"],
            positive=row["positive"],
            negative=row["negative"],
            summary=row["summary"],
            assumptions=tuple(json.loads(row["assumptions_json"])),
            workflow_file=row["workflow_file"],
            workflow_hash=row["workflow_hash"],
            selection_used=bool(row["selection_used"]),
            selection_message=row["selection_message"] or "",
            parent_request_id=row["parent_request_id"],
            regeneration_type=row["regeneration_type"],
            custom_instruction=row["custom_instruction"],
            model_key=model_key,
        )
    except Exception:
        logger.exception("Failed to read generation request: request_id=%s", request_id)
        return None


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def get_generation_stats(user_id: int) -> GenerationStats | None:
    """Phase 2A: /generate_ai_stats aggregate for one user (see
    GenerationStats' docstring for the exact scoping rule: this user's own
    generation_requests, joined to this same user's own image_ratings rows).

    Read-only and additive-only against the existing schema - no new tables
    or columns. Never raises: any DB error (corruption, locked file, ...) is
    logged and reported as None so the command can show a short message
    instead of crashing the bot.
    """
    try:
        with _connect() as connection:
            total_requests = int(
                connection.execute(
                    "SELECT COUNT(*) AS c FROM generation_requests WHERE user_id = ?", (user_id,)
                ).fetchone()["c"]
            )
            total_images = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS c
                    FROM generation_images gi
                    JOIN generation_requests gr ON gr.id = gi.request_id
                    WHERE gr.user_id = ?
                    """,
                    (user_id,),
                ).fetchone()["c"]
            )
            request_type_rows = connection.execute(
                """
                SELECT regeneration_type, COUNT(*) AS c
                FROM generation_requests
                WHERE user_id = ?
                GROUP BY regeneration_type
                """,
                (user_id,),
            ).fetchall()
            # One row per image this user has personally rated, restricted to
            # their own generation_requests - the single source every other
            # aggregate below (ratings by rating value, by recency, by
            # regeneration_type, and the tag trend) is derived from. The
            # UNIQUE(image_id, user_id) constraint on image_ratings means
            # this can never return more than one row per image.
            rated_rows = connection.execute(
                """
                SELECT
                    ir.rating AS rating,
                    ir.updated_at AS updated_at,
                    gr.regeneration_type AS regeneration_type,
                    gr.positive AS positive,
                    gr.negative AS negative,
                    gr.summary AS summary,
                    gr.assumptions_json AS assumptions_json,
                    gr.request_text AS request_text
                FROM image_ratings ir
                JOIN generation_images gi ON gi.id = ir.image_id
                JOIN generation_requests gr ON gr.id = gi.request_id
                WHERE gr.user_id = ? AND ir.user_id = ?
                """,
                (user_id, user_id),
            ).fetchall()
    except Exception:
        logger.exception("Failed to read generation stats: user_id=%s", user_id)
        return None

    request_counts_by_type: dict[str, int] = {}
    for row in request_type_rows:
        label = row["regeneration_type"] or NORMAL_GENERATION_LABEL
        request_counts_by_type[label] = request_counts_by_type.get(label, 0) + int(row["c"])

    rated_images = len(rated_rows)
    unrated_images = max(total_images - rated_images, 0)
    ratings_all_time = _rating_counts_from_ratings([row["rating"] for row in rated_rows])

    now = datetime.now(JST)
    cutoff_7d = now - timedelta(days=7)
    cutoff_30d = now - timedelta(days=30)
    ratings_7d_values: list[str] = []
    ratings_30d_values: list[str] = []
    for row in rated_rows:
        rated_at = _parse_timestamp(row["updated_at"])
        if rated_at is None:
            continue
        if rated_at >= cutoff_30d:
            ratings_30d_values.append(row["rating"])
        if rated_at >= cutoff_7d:
            ratings_7d_values.append(row["rating"])
    ratings_7d = _rating_counts_from_ratings(ratings_7d_values)
    ratings_30d = _rating_counts_from_ratings(ratings_30d_values)

    ratings_by_type: dict[str, list[str]] = {}
    for row in rated_rows:
        label = row["regeneration_type"] or NORMAL_GENERATION_LABEL
        ratings_by_type.setdefault(label, []).append(row["rating"])

    all_type_labels = set(request_counts_by_type) | set(ratings_by_type)
    regeneration_breakdown = tuple(
        RegenerationTypeStats(
            regeneration_type=label,
            request_count=request_counts_by_type.get(label, 0),
            ratings=_rating_counts_from_ratings(ratings_by_type.get(label, [])),
        )
        for label in sorted(all_type_labels, key=lambda value: (value != NORMAL_GENERATION_LABEL, value))
    )

    good_tag_counts: dict[str, int] = {}
    low_tag_counts: dict[str, int] = {}
    good_sample_count = 0
    low_sample_count = 0
    for row in rated_rows:
        rating = row["rating"]
        if rating in ("favorite", "like"):
            bucket = good_tag_counts
            good_sample_count += 1
        elif rating in ("meh", "off_target"):
            bucket = low_tag_counts
            low_sample_count += 1
        else:
            continue
        try:
            assumptions = json.loads(row["assumptions_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            assumptions = []
        combined_text = " ".join(
            [
                row["positive"] or "",
                row["negative"] or "",
                row["summary"] or "",
                " ".join(str(item) for item in assumptions),
                row["request_text"] or "",
            ]
        )
        # Deduped per image (a set, not a list/Counter over raw matches), so
        # a phrase repeated within one request's text is not over-counted.
        for phrase in _extract_watched_tags(combined_text):
            bucket[phrase] = bucket.get(phrase, 0) + 1

    tag_trend = tuple(
        TagTrendEntry(
            phrase=phrase,
            good_count=good_tag_counts.get(phrase, 0),
            low_count=low_tag_counts.get(phrase, 0),
        )
        for phrase in WATCHED_TAG_PHRASES
        if good_tag_counts.get(phrase, 0) or low_tag_counts.get(phrase, 0)
    )

    return GenerationStats(
        total_requests=total_requests,
        total_images=total_images,
        rated_images=rated_images,
        unrated_images=unrated_images,
        ratings_all_time=ratings_all_time,
        ratings_7d=ratings_7d,
        ratings_30d=ratings_30d,
        regeneration_breakdown=regeneration_breakdown,
        tag_trend=tag_trend,
        good_sample_count=good_sample_count,
        low_sample_count=low_sample_count,
    )


@dataclass(frozen=True)
class RequestTextRow:
    """Read-only: the (id, created_at, user_id, request_text) columns of one
    generation_requests row - deliberately not positive/negative/summary/
    assumptions, which are LM Studio's own output rather than the user's own
    wording. Used to cross-reference generation history against the
    character dictionary (e.g. /character_missing); this module has no
    dedicated character_name/resolved_character column to read instead - see
    services/character_missing_finder.py's module docstring for why.
    """

    id: int
    created_at: str
    user_id: int
    request_text: str


# Defensive cap on one /character_missing scan, so a very large history file
# cannot make one diagnostic command scan unboundedly many rows. Generous
# relative to expected history sizes; raise via the `limit` parameter if a
# deployment genuinely needs more scanned at once.
REQUEST_TEXT_SCAN_ROW_LIMIT = 5000


def list_request_texts(limit: int = REQUEST_TEXT_SCAN_ROW_LIMIT) -> list[RequestTextRow]:
    """Read-only: up to `limit` most recent generation_requests rows' id/
    created_at/user_id/request_text, most recent first.

    Deliberately NOT scoped to one user_id, unlike get_generation_stats/
    get_rated_prompt_rows above: a dictionary-coverage scan needs to see
    every user's requests to be useful at all. Callers must gate who is
    allowed to invoke this - it can surface other users' raw request text.

    Never raises: any DB error (corruption, locked file, ...) is logged and
    reported as an empty list, matching every other read function in this
    module - a scan feature must never be able to crash the bot.
    """
    try:
        with _connect() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, user_id, request_text
                FROM generation_requests
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            RequestTextRow(
                id=int(row["id"]),
                created_at=row["created_at"],
                user_id=int(row["user_id"]),
                request_text=row["request_text"] or "",
            )
            for row in rows
        ]
    except Exception:
        logger.exception("Failed to read request texts for character dictionary scan.")
        return []
