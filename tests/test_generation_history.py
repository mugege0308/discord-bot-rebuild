import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services.generation_history import (
    SCHEMA_VERSION,
    GenerationRequestRecord,
    init_db,
    save_generation_history,
)
from services.sd_control import GeneratedImage


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


if __name__ == "__main__":
    unittest.main()
