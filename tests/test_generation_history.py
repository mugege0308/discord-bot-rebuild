import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from services.generation_history import (
    ALLOWED_RATINGS,
    MIN_GROUP_SIZE_FOR_TAG_COMPARISON,
    MIN_RATED_FOR_TREND,
    NORMAL_GENERATION_LABEL,
    SCHEMA_VERSION,
    WATCHED_TAG_PHRASES,
    GenerationRequestRecord,
    get_generation_request,
    get_generation_stats,
    init_db,
    list_request_texts,
    resolve_image_ids_for_paths,
    save_generation_history,
    save_generation_history_detailed,
    set_image_discord_message,
    upsert_image_rating,
)
from services.sd_control import GeneratedImage
from services.weather import JST


def _make_request(**overrides) -> GenerationRequestRecord:
    defaults = dict(
        user_id=42,
        request_text="夕暮れの海辺",
        reference_id=None,
        positive="1girl, safe, masterpiece",
        negative="lowres, bad anatomy",
        summary="夕暮れの海辺に立つ少女",
        assumptions=("背景を補完", "ratingはsafe"),
        workflow_file="workflows/comfy_workflow.json",
        workflow_hash="deadbeef",
        selection_used=True,
        selection_message="画像選別AIで上位1枚を選びました。",
    )
    defaults.update(overrides)
    return GenerationRequestRecord(**defaults)


def _make_image(path: str, batch_index: int = 1, **overrides) -> GeneratedImage:
    defaults = dict(
        path=Path(path),
        batch_index=batch_index,
        seed=12345,
        checkpoint_name="animagineXL40Opt_clear.safetensors",
        sampler_name="euler_ancestral",
        scheduler="normal",
        steps=20,
        cfg=5.0,
        width=1024,
        height=1024,
        workflow_file="workflows/comfy_workflow.json",
        workflow_hash="deadbeef",
    )
    defaults.update(overrides)
    return GeneratedImage(**defaults)


class GenerationHistoryTestCase(unittest.TestCase):
    def setUp(self):
        # ignore_cleanup_errors: WAL mode leaves -wal/-shm sidecar files that
        # Windows can briefly keep locked after the last sqlite3 connection
        # object is garbage-collected; that's a teardown-only artifact of
        # this test process, not a functional issue in the module under test.
        self.tmp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp_dir.cleanup)
        self.db_file = Path(self.tmp_dir.name) / "generation_history.db"

        patcher_data_dir = mock.patch("services.generation_history.DATA_DIR", Path(self.tmp_dir.name))
        patcher_db_file = mock.patch("services.generation_history.DB_FILE", self.db_file)
        patcher_data_dir.start()
        patcher_db_file.start()
        self.addCleanup(patcher_data_dir.stop)
        self.addCleanup(patcher_db_file.stop)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_file)
        connection.row_factory = sqlite3.Row
        return connection

    def _write_file(self, relative_name: str, content: bytes) -> Path:
        """Creates a real file: selection matching compares file content, so
        these tests need bytes on disk rather than bare path strings."""
        path = Path(self.tmp_dir.name) / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _selected_flags(self, request_id: int) -> dict[str, int]:
        with self._connect() as connection:
            return {
                row["image_path"]: row["was_selected_by_ai"]
                for row in connection.execute(
                    "SELECT image_path, was_selected_by_ai FROM generation_images WHERE request_id = ?",
                    (request_id,),
                )
            }

    def test_init_db_creates_expected_tables(self):
        init_db()
        with self._connect() as connection:
            names = {
                row["name"]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
        self.assertIn("generation_requests", names)
        self.assertIn("generation_images", names)
        self.assertIn("image_ratings", names)

    def test_init_db_records_schema_version(self):
        init_db()
        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(version, SCHEMA_VERSION)

    def test_init_db_is_idempotent_and_keeps_schema_version(self):
        init_db()
        init_db()
        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(version, SCHEMA_VERSION)

    def test_schema_version_newer_than_this_build_is_left_untouched(self):
        init_db()
        with self._connect() as connection:
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
            connection.commit()

        init_db()

        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(version, SCHEMA_VERSION + 5)

    def test_save_generation_history_persists_request_and_images_in_one_transaction(self):
        request = _make_request()
        images = [
            _make_image("data/sd_outputs/img_01.png", batch_index=1),
            _make_image("data/sd_outputs/img_02.png", batch_index=2, seed=999),
        ]

        request_id = save_generation_history(request, images, selected_paths=[])

        self.assertIsNotNone(request_id)
        with self._connect() as connection:
            request_row = connection.execute(
                "SELECT * FROM generation_requests WHERE id = ?", (request_id,)
            ).fetchone()
            image_rows = connection.execute(
                "SELECT * FROM generation_images WHERE request_id = ? ORDER BY batch_index", (request_id,)
            ).fetchall()

        self.assertIsNotNone(request_row)
        self.assertEqual(request_row["user_id"], 42)
        self.assertEqual(request_row["request_text"], "夕暮れの海辺")
        self.assertEqual(request_row["positive"], request.positive)
        self.assertEqual(request_row["negative"], request.negative)
        self.assertEqual(request_row["summary"], request.summary)
        self.assertEqual(request_row["workflow_file"], request.workflow_file)
        self.assertEqual(request_row["workflow_hash"], request.workflow_hash)
        self.assertEqual(request_row["selection_used"], 1)

        self.assertEqual(len(image_rows), 2)
        self.assertEqual(image_rows[0]["image_path"], str(images[0].path))
        self.assertEqual(image_rows[1]["image_path"], str(images[1].path))
        self.assertEqual(image_rows[1]["seed"], 999)
        self.assertEqual(image_rows[0]["checkpoint_name"], "animagineXL40Opt_clear.safetensors")

    def test_assumptions_are_stored_as_round_trippable_json(self):
        request = _make_request(assumptions=("補完A", "補完B", "矛盾解決の説明"))
        images = [_make_image("data/sd_outputs/img_01.png")]

        request_id = save_generation_history(request, images, selected_paths=[])

        with self._connect() as connection:
            row = connection.execute(
                "SELECT assumptions_json FROM generation_requests WHERE id = ?", (request_id,)
            ).fetchone()
        decoded = json.loads(row["assumptions_json"])
        self.assertEqual(decoded, ["補完A", "補完B", "矛盾解決の説明"])

    def test_selected_image_is_marked_was_selected_by_ai(self):
        request = _make_request(selection_used=True)
        matched_image = _make_image("data/sd_outputs/chosen.png", batch_index=1)
        unmatched_image = _make_image("data/sd_outputs/not_chosen.png", batch_index=2)

        request_id = save_generation_history(
            request,
            [matched_image, unmatched_image],
            selected_paths=[Path("data/sd_outputs/chosen.png")],
        )

        with self._connect() as connection:
            rows = {
                row["image_path"]: row["was_selected_by_ai"]
                for row in connection.execute(
                    "SELECT image_path, was_selected_by_ai FROM generation_images WHERE request_id = ?",
                    (request_id,),
                )
            }
        self.assertEqual(rows[str(matched_image.path)], 1)
        self.assertEqual(rows[str(unmatched_image.path)], 0)

    def test_selected_image_matches_by_content_even_when_renamed(self):
        # The external score_batch_top5.py copies its chosen images into its
        # own folder under a name this repo cannot predict. Every step of
        # that chain is a byte-preserving copy, so matching is done on file
        # content rather than by guessing from the filename.
        request = _make_request(selection_used=True)
        original = self._write_file("sd_outputs/20260722_120000_01_01.png", b"PNG-BYTES-A")
        other = self._write_file("sd_outputs/20260722_120000_01_02.png", b"PNG-BYTES-B")
        renamed_copy = self._write_file("latest_top5/top1_unpredictable_name.png", b"PNG-BYTES-A")

        request_id = save_generation_history(
            request,
            [_make_image(str(original), batch_index=1), _make_image(str(other), batch_index=2)],
            selected_paths=[renamed_copy],
        )

        flags = self._selected_flags(request_id)
        self.assertEqual(flags[str(original)], 1)
        self.assertEqual(flags[str(other)], 0)

    def test_same_filename_with_different_content_is_not_marked_selected(self):
        # Regression guard for the old filename-based heuristic: an identical
        # name must not be enough to claim a match.
        request = _make_request(selection_used=True)
        original = self._write_file("sd_outputs/img_01.png", b"PNG-BYTES-A")
        same_name_different_image = self._write_file("latest_top5/img_01.png", b"PNG-BYTES-DIFFERENT")

        request_id = save_generation_history(
            request,
            [_make_image(str(original), batch_index=1)],
            selected_paths=[same_name_different_image],
        )

        self.assertEqual(self._selected_flags(request_id)[str(original)], 0)

    def test_unreadable_image_is_reported_unselected_instead_of_guessed(self):
        request = _make_request(selection_used=True)
        selected = self._write_file("latest_top5/top1.png", b"PNG-BYTES-A")
        missing = Path(self.tmp_dir.name) / "sd_outputs" / "never_written.png"

        request_id = save_generation_history(
            request,
            [_make_image(str(missing), batch_index=1)],
            selected_paths=[selected],
        )

        self.assertEqual(self._selected_flags(request_id)[str(missing)], 0)

    def test_selection_ai_failure_marks_all_images_unselected(self):
        request = _make_request(selection_used=False, selection_message="score_batch_top5.py が見つかりません")
        images = [_make_image("data/sd_outputs/img_01.png"), _make_image("data/sd_outputs/img_02.png", batch_index=2)]

        request_id = save_generation_history(request, images, selected_paths=[])

        with self._connect() as connection:
            request_row = connection.execute(
                "SELECT selection_used FROM generation_requests WHERE id = ?", (request_id,)
            ).fetchone()
            image_rows = connection.execute(
                "SELECT was_selected_by_ai FROM generation_images WHERE request_id = ?", (request_id,)
            ).fetchall()

        self.assertEqual(request_row["selection_used"], 0)
        self.assertTrue(all(row["was_selected_by_ai"] == 0 for row in image_rows))

    def test_save_failure_returns_none_without_raising(self):
        request = _make_request()
        images = [_make_image("data/sd_outputs/img_01.png")]

        with mock.patch("services.generation_history.sqlite3.connect", side_effect=sqlite3.OperationalError("disk I/O error")):
            result = save_generation_history(request, images, selected_paths=[])

        self.assertIsNone(result)


class SchemaMigrationTestCase(GenerationHistoryTestCase):
    """Phase 1C: generation_requests gains parent_request_id/regeneration_type
    via an additive v1 -> v2 migration that must not disturb existing rows
    or leave the file half-upgraded if it fails partway through."""

    def _create_v1_db(self) -> None:
        """Builds a file with the pre-Phase-1C generation_requests shape
        (no parent_request_id/regeneration_type columns) stamped at schema
        version 1, with one pre-existing row - independent of this module's
        own init_db(), which already creates the v2 shape."""
        DATA_DIR_MARKER = self.db_file.parent
        DATA_DIR_MARKER.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_file) as connection:
            connection.execute(
                """
                CREATE TABLE generation_requests (
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
                CREATE TABLE generation_images (
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
                CREATE TABLE image_ratings (
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
            connection.execute(
                """
                INSERT INTO generation_requests (
                    created_at, user_id, request_text, reference_id,
                    positive, negative, summary, assumptions_json,
                    workflow_file, workflow_hash, selection_used, selection_message
                ) VALUES ('2026-01-01T00:00:00+09:00', 42, '既存の依頼', NULL,
                    '1girl', 'lowres', '既存の要約', '[]', NULL, NULL, 0, '')
                """
            )
            connection.execute("PRAGMA user_version = 1")
            connection.commit()

    def test_migration_adds_columns_and_bumps_version(self):
        self._create_v1_db()

        init_db()

        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(generation_requests)")}
        self.assertEqual(version, SCHEMA_VERSION)
        self.assertIn("parent_request_id", columns)
        self.assertIn("regeneration_type", columns)
        self.assertIn("custom_instruction", columns)

    def test_migration_preserves_existing_row_and_backfills_null(self):
        self._create_v1_db()

        init_db()

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM generation_requests WHERE request_text = '既存の依頼'"
            ).fetchone()
        self.assertIsNotNone(row, "pre-existing row must survive the migration")
        self.assertEqual(row["positive"], "1girl")
        self.assertEqual(row["negative"], "lowres")
        self.assertIsNone(row["parent_request_id"])
        self.assertIsNone(row["regeneration_type"])
        self.assertIsNone(row["custom_instruction"])

    def test_migration_is_idempotent_when_rerun_against_an_already_migrated_file(self):
        self._create_v1_db()
        init_db()

        init_db()  # must not raise "duplicate column" or similar

        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(version, SCHEMA_VERSION)

    def test_migration_failure_leaves_the_v1_file_untouched(self):
        self._create_v1_db()

        with mock.patch(
            "services.generation_history._add_column_if_missing",
            side_effect=[None, sqlite3.OperationalError("disk full")],
        ):
            with self.assertRaises(sqlite3.OperationalError):
                init_db()

        # The whole ALTER TABLE / PRAGMA user_version transaction must have
        # rolled back together - a half-migrated file (one new column but
        # not the other, or a bumped version without both columns) would be
        # worse than an un-migrated one.
        with sqlite3.connect(self.db_file) as connection:
            connection.row_factory = sqlite3.Row
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(generation_requests)")}
        self.assertEqual(version, 1)
        self.assertNotIn("parent_request_id", columns)
        self.assertNotIn("regeneration_type", columns)

    def _create_v2_db(self) -> None:
        """Builds a file with the pre-custom_instruction generation_requests
        shape (parent_request_id/regeneration_type present, custom_instruction
        absent) stamped at schema version 2, with one pre-existing row."""
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_file) as connection:
            connection.execute(
                """
                CREATE TABLE generation_requests (
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
                    selection_message TEXT,
                    parent_request_id INTEGER,
                    regeneration_type TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE generation_images (
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
                CREATE TABLE image_ratings (
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
            connection.execute(
                """
                INSERT INTO generation_requests (
                    created_at, user_id, request_text, reference_id,
                    positive, negative, summary, assumptions_json,
                    workflow_file, workflow_hash, selection_used, selection_message,
                    parent_request_id, regeneration_type
                ) VALUES ('2026-01-01T00:00:00+09:00', 42, 'v2の依頼', NULL,
                    '1girl', 'lowres', 'v2の要約', '[]', NULL, NULL, 0, '', NULL, NULL)
                """
            )
            connection.execute("PRAGMA user_version = 2")
            connection.commit()

    def test_v2_to_v3_migration_adds_custom_instruction_column_and_bumps_version(self):
        self._create_v2_db()

        init_db()

        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(generation_requests)")}
        self.assertEqual(version, SCHEMA_VERSION)
        self.assertIn("custom_instruction", columns)

    def test_v2_to_v3_migration_preserves_existing_row_and_backfills_null(self):
        self._create_v2_db()

        init_db()

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM generation_requests WHERE request_text = 'v2の依頼'"
            ).fetchone()
        self.assertIsNotNone(row, "pre-existing v2 row must survive the migration")
        self.assertEqual(row["positive"], "1girl")
        self.assertIsNone(row["custom_instruction"])

    def test_v2_to_v3_migration_is_idempotent_when_rerun(self):
        self._create_v2_db()
        init_db()

        init_db()  # must not raise "duplicate column" or similar

        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(version, SCHEMA_VERSION)


class RegenerationFieldsTestCase(GenerationHistoryTestCase):
    """Phase 1C: parent_request_id/regeneration_type on save, and the
    read-only get_generation_request lookup used by the regenerate flow."""

    def _save_request(self, **overrides) -> int:
        path = self._write_file(f"sd_outputs/{overrides.get('request_text', 'img')}.png", b"PNG-BYTES")
        saved = save_generation_history_detailed(
            _make_request(selection_used=False, **overrides),
            [_make_image(str(path))],
            selected_paths=[],
        )
        self.assertIsNotNone(saved)
        return saved.request_id

    def test_parent_request_id_and_regeneration_type_are_stored(self):
        parent_id = self._save_request(request_text="元の依頼")

        child_id = self._save_request(
            request_text="再生成", parent_request_id=parent_id, regeneration_type="same_prompt"
        )

        with self._connect() as connection:
            row = connection.execute(
                "SELECT parent_request_id, regeneration_type FROM generation_requests WHERE id = ?",
                (child_id,),
            ).fetchone()
        self.assertEqual(row["parent_request_id"], parent_id)
        self.assertEqual(row["regeneration_type"], "same_prompt")

    def test_change_composition_regeneration_type_is_stored(self):
        parent_id = self._save_request(request_text="元の依頼2")

        child_id = self._save_request(
            request_text="構図変更再生成", parent_request_id=parent_id, regeneration_type="change_composition"
        )

        with self._connect() as connection:
            row = connection.execute(
                "SELECT regeneration_type FROM generation_requests WHERE id = ?", (child_id,)
            ).fetchone()
        self.assertEqual(row["regeneration_type"], "change_composition")

    def test_custom_instruction_regeneration_type_and_text_are_stored(self):
        parent_id = self._save_request(request_text="元の依頼3")

        child_id = self._save_request(
            request_text="指示追加再生成",
            parent_request_id=parent_id,
            regeneration_type="custom_instruction",
            custom_instruction="髪型をポニーテールに",
        )

        with self._connect() as connection:
            row = connection.execute(
                "SELECT regeneration_type, custom_instruction, parent_request_id FROM generation_requests WHERE id = ?",
                (child_id,),
            ).fetchone()
        self.assertEqual(row["regeneration_type"], "custom_instruction")
        self.assertEqual(row["custom_instruction"], "髪型をポニーテールに")
        self.assertEqual(row["parent_request_id"], parent_id)

    def test_custom_instruction_is_null_for_non_custom_instruction_requests(self):
        request_id = self._save_request(request_text="通常の依頼2", regeneration_type="same_prompt")

        with self._connect() as connection:
            row = connection.execute(
                "SELECT custom_instruction FROM generation_requests WHERE id = ?", (request_id,)
            ).fetchone()
        self.assertIsNone(row["custom_instruction"])

    def test_nonexistent_parent_request_id_is_stored_as_null(self):
        request_id = self._save_request(request_text="孤立した再生成", parent_request_id=999999, regeneration_type="same_prompt")

        with self._connect() as connection:
            row = connection.execute(
                "SELECT parent_request_id FROM generation_requests WHERE id = ?", (request_id,)
            ).fetchone()
        self.assertIsNone(row["parent_request_id"])

    def test_unknown_regeneration_type_is_rejected_and_stored_as_null(self):
        request_id = self._save_request(request_text="不正なタイプ", regeneration_type="bogus_type")

        with self._connect() as connection:
            row = connection.execute(
                "SELECT regeneration_type FROM generation_requests WHERE id = ?", (request_id,)
            ).fetchone()
        self.assertIsNone(row["regeneration_type"])

    def test_normal_non_regenerated_request_has_null_parent_fields(self):
        request_id = self._save_request(request_text="通常の依頼")

        with self._connect() as connection:
            row = connection.execute(
                "SELECT parent_request_id, regeneration_type FROM generation_requests WHERE id = ?",
                (request_id,),
            ).fetchone()
        self.assertIsNone(row["parent_request_id"])
        self.assertIsNone(row["regeneration_type"])

    def test_get_generation_request_returns_full_detail(self):
        request_id = self._save_request(
            request_text="詳細取得テスト",
            user_id=777,
            reference_id=3,
            positive="1girl, safe",
            negative="lowres",
            summary="要約テキスト",
            assumptions=("補完A",),
        )

        detail = get_generation_request(request_id)

        self.assertIsNotNone(detail)
        self.assertEqual(detail.id, request_id)
        self.assertEqual(detail.user_id, 777)
        self.assertEqual(detail.request_text, "詳細取得テスト")
        self.assertEqual(detail.reference_id, 3)
        self.assertEqual(detail.positive, "1girl, safe")
        self.assertEqual(detail.negative, "lowres")
        self.assertEqual(detail.summary, "要約テキスト")
        self.assertEqual(detail.assumptions, ("補完A",))
        self.assertIsNone(detail.parent_request_id)
        self.assertIsNone(detail.regeneration_type)
        self.assertIsNone(detail.custom_instruction)

    def test_get_generation_request_includes_custom_instruction(self):
        parent_id = self._save_request(request_text="元祖2")
        child_id = self._save_request(
            request_text="指示追加の子",
            parent_request_id=parent_id,
            regeneration_type="custom_instruction",
            custom_instruction="笑顔にして",
        )

        detail = get_generation_request(child_id)

        self.assertIsNotNone(detail)
        self.assertEqual(detail.regeneration_type, "custom_instruction")
        self.assertEqual(detail.custom_instruction, "笑顔にして")

    def test_get_generation_request_includes_parent_info(self):
        parent_id = self._save_request(request_text="元祖")
        child_id = self._save_request(
            request_text="子", parent_request_id=parent_id, regeneration_type="change_composition"
        )

        detail = get_generation_request(child_id)

        self.assertEqual(detail.parent_request_id, parent_id)
        self.assertEqual(detail.regeneration_type, "change_composition")

    def test_get_generation_request_returns_none_for_missing_id(self):
        self.assertIsNone(get_generation_request(999999))

    def test_get_generation_request_returns_none_on_db_error_without_raising(self):
        with mock.patch(
            "services.generation_history.sqlite3.connect",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            self.assertIsNone(get_generation_request(1))


class ImageRatingTestCase(GenerationHistoryTestCase):
    """Phase 1B: image_ratings upserts, id mapping, Discord message info."""

    def _save_one_image(self, image_path: Path | None = None) -> int:
        """Saves a request with a single image and returns its row id."""
        path = image_path or self._write_file("sd_outputs/img_01.png", b"PNG-BYTES-A")
        saved = save_generation_history_detailed(
            _make_request(selection_used=False),
            [_make_image(str(path), batch_index=1)],
            selected_paths=[],
        )
        self.assertIsNotNone(saved)
        self.assertEqual(len(saved.image_ids), 1)
        return saved.image_ids[0]

    def _rating_rows(self, image_id: int) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM image_ratings WHERE image_id = ? ORDER BY user_id", (image_id,)
            ).fetchall()

    def test_favorite_is_stored_on_first_rating(self):
        image_id = self._save_one_image()

        self.assertTrue(upsert_image_rating(image_id, user_id=42, rating="favorite"))

        rows = self._rating_rows(image_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rating"], "favorite")
        self.assertEqual(rows[0]["user_id"], 42)

    def test_like_is_stored_on_first_rating(self):
        image_id = self._save_one_image()

        self.assertTrue(upsert_image_rating(image_id, user_id=42, rating="like"))

        rows = self._rating_rows(image_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rating"], "like")
        self.assertEqual(rows[0]["user_id"], 42)

    def test_meh_is_stored_on_first_rating(self):
        image_id = self._save_one_image()

        self.assertTrue(upsert_image_rating(image_id, user_id=42, rating="meh"))

        rows = self._rating_rows(image_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rating"], "meh")

    def test_off_target_is_stored_on_first_rating(self):
        image_id = self._save_one_image()

        self.assertTrue(upsert_image_rating(image_id, user_id=42, rating="off_target"))

        rows = self._rating_rows(image_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rating"], "off_target")

    def test_like_rating_updated_to_off_target_keeps_rated_at(self):
        image_id = self._save_one_image()
        self.assertTrue(upsert_image_rating(image_id, user_id=42, rating="like"))
        first = self._rating_rows(image_id)[0]

        with mock.patch("services.generation_history._now_text", return_value="2026-07-23T12:00:00+09:00"):
            self.assertTrue(upsert_image_rating(image_id, user_id=42, rating="off_target"))

        rows = self._rating_rows(image_id)
        self.assertEqual(len(rows), 1, "an update must not add a second row")
        self.assertEqual(rows[0]["rating"], "off_target")
        self.assertEqual(rows[0]["rated_at"], first["rated_at"])
        self.assertEqual(rows[0]["updated_at"], "2026-07-23T12:00:00+09:00")

    def test_off_target_rating_updated_to_meh_keeps_rated_at(self):
        image_id = self._save_one_image()
        self.assertTrue(upsert_image_rating(image_id, user_id=42, rating="off_target"))
        first = self._rating_rows(image_id)[0]

        with mock.patch("services.generation_history._now_text", return_value="2026-07-23T13:00:00+09:00"):
            self.assertTrue(upsert_image_rating(image_id, user_id=42, rating="meh"))

        rows = self._rating_rows(image_id)
        self.assertEqual(len(rows), 1, "an update must not add a second row")
        self.assertEqual(rows[0]["rating"], "meh")
        self.assertEqual(rows[0]["rated_at"], first["rated_at"])
        self.assertEqual(rows[0]["updated_at"], "2026-07-23T13:00:00+09:00")

    def test_repeated_rating_updates_value_and_keeps_rated_at(self):
        image_id = self._save_one_image()
        self.assertTrue(upsert_image_rating(image_id, user_id=42, rating="like"))
        first = self._rating_rows(image_id)[0]

        with mock.patch("services.generation_history._now_text", return_value="2026-07-23T12:00:00+09:00"):
            self.assertTrue(upsert_image_rating(image_id, user_id=42, rating="meh"))

        rows = self._rating_rows(image_id)
        self.assertEqual(len(rows), 1, "an update must not add a second row")
        self.assertEqual(rows[0]["rating"], "meh")
        self.assertEqual(rows[0]["rated_at"], first["rated_at"], "rated_at keeps the first-rating time")
        self.assertEqual(rows[0]["updated_at"], "2026-07-23T12:00:00+09:00")
        self.assertNotEqual(rows[0]["updated_at"], rows[0]["rated_at"])

    def test_like_rating_updated_to_favorite_keeps_rated_at(self):
        image_id = self._save_one_image()
        self.assertTrue(upsert_image_rating(image_id, user_id=42, rating="like"))
        first = self._rating_rows(image_id)[0]

        with mock.patch("services.generation_history._now_text", return_value="2026-07-23T14:00:00+09:00"):
            self.assertTrue(upsert_image_rating(image_id, user_id=42, rating="favorite"))

        rows = self._rating_rows(image_id)
        self.assertEqual(len(rows), 1, "an update must not add a second row")
        self.assertEqual(rows[0]["rating"], "favorite")
        self.assertEqual(rows[0]["rated_at"], first["rated_at"])
        self.assertEqual(rows[0]["updated_at"], "2026-07-23T14:00:00+09:00")

    def test_favorite_rating_updated_to_meh_keeps_rated_at(self):
        image_id = self._save_one_image()
        self.assertTrue(upsert_image_rating(image_id, user_id=42, rating="favorite"))
        first = self._rating_rows(image_id)[0]

        with mock.patch("services.generation_history._now_text", return_value="2026-07-23T15:00:00+09:00"):
            self.assertTrue(upsert_image_rating(image_id, user_id=42, rating="meh"))

        rows = self._rating_rows(image_id)
        self.assertEqual(len(rows), 1, "an update must not add a second row")
        self.assertEqual(rows[0]["rating"], "meh")
        self.assertEqual(rows[0]["rated_at"], first["rated_at"])
        self.assertEqual(rows[0]["updated_at"], "2026-07-23T15:00:00+09:00")

    def test_favorite_rating_updated_to_off_target_keeps_rated_at(self):
        image_id = self._save_one_image()
        self.assertTrue(upsert_image_rating(image_id, user_id=42, rating="favorite"))
        first = self._rating_rows(image_id)[0]

        with mock.patch("services.generation_history._now_text", return_value="2026-07-23T16:00:00+09:00"):
            self.assertTrue(upsert_image_rating(image_id, user_id=42, rating="off_target"))

        rows = self._rating_rows(image_id)
        self.assertEqual(len(rows), 1, "an update must not add a second row")
        self.assertEqual(rows[0]["rating"], "off_target")
        self.assertEqual(rows[0]["rated_at"], first["rated_at"])
        self.assertEqual(rows[0]["updated_at"], "2026-07-23T16:00:00+09:00")

    def test_different_users_rate_the_same_image_independently(self):
        image_id = self._save_one_image()

        self.assertTrue(upsert_image_rating(image_id, user_id=1, rating="like"))
        self.assertTrue(upsert_image_rating(image_id, user_id=2, rating="meh"))

        rows = self._rating_rows(image_id)
        self.assertEqual([(row["user_id"], row["rating"]) for row in rows], [(1, "like"), (2, "meh")])

    def test_invalid_rating_is_rejected_without_touching_db(self):
        image_id = self._save_one_image()

        for bad in ("love", "LIKE", "", "1", "good"):
            self.assertFalse(upsert_image_rating(image_id, user_id=42, rating=bad), bad)

        self.assertEqual(self._rating_rows(image_id), [])

    def test_allowed_ratings_are_exactly_favorite_like_meh_and_off_target(self):
        self.assertEqual(set(ALLOWED_RATINGS), {"favorite", "like", "meh", "off_target"})

    def test_unknown_image_id_is_rejected_and_returns_false(self):
        self._save_one_image()

        self.assertFalse(upsert_image_rating(999999, user_id=42, rating="like"))

        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) AS c FROM image_ratings").fetchone()["c"]
        self.assertEqual(total, 0)

    def test_rating_failure_returns_false_without_raising(self):
        image_id = self._save_one_image()
        with mock.patch(
            "services.generation_history.sqlite3.connect",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            self.assertFalse(upsert_image_rating(image_id, user_id=42, rating="like"))

    def test_saved_image_ids_are_index_aligned_with_images(self):
        first = self._write_file("sd_outputs/a.png", b"AAA")
        second = self._write_file("sd_outputs/b.png", b"BBB")
        saved = save_generation_history_detailed(
            _make_request(selection_used=False),
            [_make_image(str(first), batch_index=1), _make_image(str(second), batch_index=2)],
            selected_paths=[],
        )

        with self._connect() as connection:
            rows = {
                row["id"]: row["image_path"]
                for row in connection.execute("SELECT id, image_path FROM generation_images")
            }
        self.assertEqual(rows[saved.image_ids[0]], str(first))
        self.assertEqual(rows[saved.image_ids[1]], str(second))

    def test_resolve_image_ids_matches_exact_path_and_sha256_only(self):
        original = self._write_file("sd_outputs/a.png", b"AAA")
        other = self._write_file("sd_outputs/b.png", b"BBB")
        images = [_make_image(str(original), batch_index=1), _make_image(str(other), batch_index=2)]
        image_ids = [101, 202]
        renamed_copy = self._write_file("latest_top5/unpredictable.png", b"AAA")
        same_name_other_content = self._write_file("elsewhere/a.png", b"ZZZ")

        matched = resolve_image_ids_for_paths(
            images, image_ids, [original, renamed_copy, same_name_other_content]
        )

        self.assertEqual(matched[0], 101, "exact path match")
        self.assertEqual(matched[1], 101, "sha256 match through rename")
        self.assertIsNone(matched[2], "same filename but different content must not match")

    def test_resolve_image_ids_returns_none_for_unreadable_path(self):
        original = self._write_file("sd_outputs/a.png", b"AAA")
        missing = Path(self.tmp_dir.name) / "sd_outputs" / "never_written.png"

        matched = resolve_image_ids_for_paths([_make_image(str(original))], [101], [missing])

        self.assertEqual(matched, [None])

    def test_set_image_discord_message_stores_ids(self):
        image_id = self._save_one_image()

        self.assertTrue(set_image_discord_message([image_id], message_id=555, channel_id=777))

        with self._connect() as connection:
            row = connection.execute(
                "SELECT discord_message_id, discord_channel_id FROM generation_images WHERE id = ?",
                (image_id,),
            ).fetchone()
        self.assertEqual(row["discord_message_id"], 555)
        self.assertEqual(row["discord_channel_id"], 777)

    def test_set_image_discord_message_handles_empty_and_failure(self):
        image_id = self._save_one_image()
        self.assertFalse(set_image_discord_message([], message_id=1, channel_id=2))
        with mock.patch(
            "services.generation_history.sqlite3.connect",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            self.assertFalse(set_image_discord_message([image_id], message_id=1, channel_id=2))

    def test_save_generation_history_wrapper_still_returns_request_id(self):
        path = self._write_file("sd_outputs/img_01.png", b"PNG-BYTES-A")
        request_id = save_generation_history(
            _make_request(selection_used=False), [_make_image(str(path))], selected_paths=[]
        )
        self.assertIsInstance(request_id, int)


class GenerationStatsTestCase(GenerationHistoryTestCase):
    """Phase 2A: /generate_ai_stats aggregation."""

    def _save_and_rate(
        self,
        *,
        user_id: int = 42,
        rater_id: int | None = None,
        rating: str | None = None,
        filename: str = "img",
        **request_overrides,
    ) -> tuple[int, int]:
        """Saves one request with one image, optionally rating it, and
        returns (request_id, image_id)."""
        path = self._write_file(f"sd_outputs/{filename}.png", f"PNG-{filename}".encode())
        request = _make_request(user_id=user_id, selection_used=False, **request_overrides)
        saved = save_generation_history_detailed(request, [_make_image(str(path))], selected_paths=[])
        self.assertIsNotNone(saved)
        image_id = saved.image_ids[0]
        if rating is not None:
            self.assertTrue(upsert_image_rating(image_id, rater_id if rater_id is not None else user_id, rating))
        return saved.request_id, image_id

    def _set_rating_updated_at(self, image_id: int, user_id: int, when) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE image_ratings SET updated_at = ? WHERE image_id = ? AND user_id = ?",
                (when.isoformat(timespec="seconds"), image_id, user_id),
            )

    def test_no_history_returns_all_zero_counts(self):
        stats = get_generation_stats(42)

        self.assertEqual(stats.total_requests, 0)
        self.assertEqual(stats.total_images, 0)
        self.assertEqual(stats.rated_images, 0)
        self.assertEqual(stats.unrated_images, 0)
        self.assertEqual(stats.ratings_all_time.total, 0)

    def test_unrated_images_are_counted_as_unrated_not_ignored(self):
        self._save_and_rate(user_id=42, filename="a")
        self._save_and_rate(user_id=42, filename="b")

        stats = get_generation_stats(42)

        self.assertEqual(stats.total_requests, 2)
        self.assertEqual(stats.total_images, 2)
        self.assertEqual(stats.rated_images, 0)
        self.assertEqual(stats.unrated_images, 2)

    def test_all_four_ratings_are_counted_separately(self):
        self._save_and_rate(user_id=42, rating="favorite", filename="a")
        self._save_and_rate(user_id=42, rating="like", filename="b")
        self._save_and_rate(user_id=42, rating="meh", filename="c")
        self._save_and_rate(user_id=42, rating="off_target", filename="d")

        stats = get_generation_stats(42)

        self.assertEqual(stats.rated_images, 4)
        self.assertEqual(stats.unrated_images, 0)
        self.assertEqual(stats.ratings_all_time.favorite, 1)
        self.assertEqual(stats.ratings_all_time.like, 1)
        self.assertEqual(stats.ratings_all_time.meh, 1)
        self.assertEqual(stats.ratings_all_time.off_target, 1)
        self.assertEqual(stats.ratings_all_time.good, 2)
        self.assertEqual(stats.ratings_all_time.low, 2)

    def test_rating_update_counts_only_the_latest_value(self):
        _, image_id = self._save_and_rate(user_id=42, rating="like", filename="a")
        self.assertTrue(upsert_image_rating(image_id, 42, "favorite"))

        stats = get_generation_stats(42)

        self.assertEqual(stats.rated_images, 1)
        self.assertEqual(stats.ratings_all_time.favorite, 1)
        self.assertEqual(stats.ratings_all_time.like, 0)

    def test_off_target_is_not_merged_into_meh_as_generic_low_quality(self):
        self._save_and_rate(user_id=42, rating="meh", filename="a")
        self._save_and_rate(user_id=42, rating="off_target", filename="b")

        stats = get_generation_stats(42)

        self.assertEqual(stats.ratings_all_time.meh, 1)
        self.assertEqual(stats.ratings_all_time.off_target, 1)
        self.assertEqual(stats.ratings_all_time.low, 2)

    def test_other_users_ratings_on_my_images_are_not_counted(self):
        _, image_id = self._save_and_rate(user_id=42, filename="a")
        self.assertTrue(upsert_image_rating(image_id, 999, "favorite"))

        stats = get_generation_stats(42)

        self.assertEqual(stats.rated_images, 0, "another user's rating must not count as my own evaluation history")
        self.assertEqual(stats.unrated_images, 1)

    def test_other_users_requests_do_not_leak_into_my_stats(self):
        self._save_and_rate(user_id=999, rating="favorite", filename="other")
        self._save_and_rate(user_id=42, rating="like", filename="mine")

        stats = get_generation_stats(42)

        self.assertEqual(stats.total_requests, 1)
        self.assertEqual(stats.rated_images, 1)
        self.assertEqual(stats.ratings_all_time.like, 1)
        self.assertEqual(stats.ratings_all_time.favorite, 0)

    def test_regeneration_breakdown_counts_requests_and_ratings_independently(self):
        self._save_and_rate(user_id=42, rating="like", filename="a")
        self._save_and_rate(user_id=42, filename="b", regeneration_type=None, parent_request_id=None)
        req_id, _ = self._save_and_rate(user_id=42, rating="favorite", filename="c")
        self._save_and_rate(user_id=42, rating="off_target", filename="d", regeneration_type="same_prompt", parent_request_id=req_id)
        self._save_and_rate(user_id=42, filename="e", regeneration_type="change_composition", parent_request_id=req_id)

        stats = get_generation_stats(42)

        breakdown = {entry.regeneration_type: entry for entry in stats.regeneration_breakdown}
        self.assertEqual(breakdown[NORMAL_GENERATION_LABEL].request_count, 3)
        self.assertEqual(breakdown[NORMAL_GENERATION_LABEL].ratings.total, 2)
        self.assertEqual(breakdown["same_prompt"].request_count, 1)
        self.assertEqual(breakdown["same_prompt"].ratings.off_target, 1)
        self.assertEqual(breakdown["change_composition"].request_count, 1)
        self.assertEqual(breakdown["change_composition"].ratings.total, 0, "unrated request contributes 0 ratings, not omission")

    def test_regeneration_breakdown_counts_custom_instruction_like_other_types(self):
        req_id, _ = self._save_and_rate(user_id=42, rating="favorite", filename="f")
        self._save_and_rate(
            user_id=42,
            rating="like",
            filename="g",
            regeneration_type="custom_instruction",
            parent_request_id=req_id,
            custom_instruction="髪型をポニーテールに",
        )

        stats = get_generation_stats(42)

        breakdown = {entry.regeneration_type: entry for entry in stats.regeneration_breakdown}
        self.assertEqual(breakdown["custom_instruction"].request_count, 1)
        self.assertEqual(breakdown["custom_instruction"].ratings.like, 1)

    def test_recent_windows_use_the_ratings_own_recency_not_all_time(self):
        _, recent_image = self._save_and_rate(user_id=42, rating="like", filename="recent")
        _, old_image = self._save_and_rate(user_id=42, rating="meh", filename="old")
        now = datetime.now(JST)
        self._set_rating_updated_at(recent_image, 42, now - timedelta(days=3))
        self._set_rating_updated_at(old_image, 42, now - timedelta(days=40))

        stats = get_generation_stats(42)

        self.assertEqual(stats.ratings_7d.total, 1)
        self.assertEqual(stats.ratings_7d.like, 1)
        self.assertEqual(stats.ratings_30d.total, 1)
        self.assertEqual(stats.ratings_30d.like, 1)
        self.assertEqual(stats.ratings_all_time.total, 2)

    def test_tag_trend_deduplicates_repeated_phrase_within_one_request(self):
        self._save_and_rate(
            user_id=42,
            rating="favorite",
            filename="a",
            positive="1girl, natural lighting, natural lighting, masterpiece",
        )

        stats = get_generation_stats(42)

        entry = next(item for item in stats.tag_trend if item.phrase == "natural lighting")
        self.assertEqual(entry.good_count, 1)

    def test_tag_trend_is_case_insensitive(self):
        self._save_and_rate(user_id=42, rating="favorite", filename="a", positive="1girl, Natural Lighting")

        stats = get_generation_stats(42)

        entry = next(item for item in stats.tag_trend if item.phrase == "natural lighting")
        self.assertEqual(entry.good_count, 1)

    def test_tag_trend_strips_surrounding_whitespace_around_comma_separated_tags(self):
        self._save_and_rate(
            user_id=42, rating="favorite", filename="a", positive="1girl,   natural lighting  , masterpiece"
        )

        stats = get_generation_stats(42)

        entry = next(item for item in stats.tag_trend if item.phrase == "natural lighting")
        self.assertEqual(entry.good_count, 1)

    def test_tag_trend_respects_word_boundaries_and_avoids_false_positives(self):
        # "night" must not match inside "midnight", and "cafe" must not
        # match inside "cafeteria" - both share a run of word characters
        # with no boundary at the point the shorter phrase would end.
        self._save_and_rate(
            user_id=42,
            rating="favorite",
            filename="a",
            positive="1girl, midnight snack, cafeteria scene",
        )

        stats = get_generation_stats(42)

        matched_phrases = {item.phrase for item in stats.tag_trend}
        self.assertNotIn("night", matched_phrases)
        self.assertNotIn("cafe", matched_phrases)

    def test_tag_trend_only_counts_my_own_rated_images_not_unrated_ones(self):
        self._save_and_rate(user_id=42, filename="a", positive="1girl, natural lighting")

        stats = get_generation_stats(42)

        self.assertEqual(stats.tag_trend, ())

    def test_good_and_low_sample_counts_split_by_rating(self):
        self._save_and_rate(user_id=42, rating="favorite", filename="a")
        self._save_and_rate(user_id=42, rating="like", filename="b")
        self._save_and_rate(user_id=42, rating="meh", filename="c")
        self._save_and_rate(user_id=42, rating="off_target", filename="d")

        stats = get_generation_stats(42)

        self.assertEqual(stats.good_sample_count, 2)
        self.assertEqual(stats.low_sample_count, 2)

    def test_thresholds_are_module_level_constants(self):
        self.assertIsInstance(MIN_RATED_FOR_TREND, int)
        self.assertIsInstance(MIN_GROUP_SIZE_FOR_TAG_COMPARISON, int)
        self.assertGreater(MIN_RATED_FOR_TREND, 0)
        self.assertGreater(MIN_GROUP_SIZE_FOR_TAG_COMPARISON, 0)

    def test_watched_tag_phrases_is_a_non_empty_fixed_list(self):
        self.assertGreater(len(WATCHED_TAG_PHRASES), 0)
        self.assertEqual(len(WATCHED_TAG_PHRASES), len(set(WATCHED_TAG_PHRASES)), "no duplicate phrases")

    def test_db_error_returns_none_without_raising(self):
        with mock.patch(
            "services.generation_history.sqlite3.connect",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            self.assertIsNone(get_generation_stats(42))


class RequestTextScanTestCase(GenerationHistoryTestCase):
    """list_request_texts - the read used by /character_missing to scan
    generation history for known-but-unresolved character aliases."""

    def test_returns_saved_rows_most_recent_first(self):
        save_generation_history(_make_request(user_id=1, request_text="最初の依頼"), [])
        save_generation_history(_make_request(user_id=2, request_text="二番目の依頼"), [])

        rows = list_request_texts()

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].request_text, "二番目の依頼")
        self.assertEqual(rows[1].request_text, "最初の依頼")
        self.assertEqual(rows[0].user_id, 2)

    def test_no_rows_returns_empty_list(self):
        self.assertEqual(list_request_texts(), [])

    def test_respects_limit_parameter(self):
        for i in range(5):
            save_generation_history(_make_request(user_id=1, request_text=f"依頼{i}"), [])

        rows = list_request_texts(limit=2)

        self.assertEqual(len(rows), 2)

    def test_db_error_returns_empty_list_without_raising(self):
        with mock.patch(
            "services.generation_history.sqlite3.connect",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            self.assertEqual(list_request_texts(), [])

    def test_corrupted_db_file_returns_empty_list_without_raising(self):
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self.db_file.write_bytes(b"this is not a sqlite database file")

        self.assertEqual(list_request_texts(), [])


if __name__ == "__main__":
    unittest.main()
