import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services.generation_history import (
    ALLOWED_RATINGS,
    SCHEMA_VERSION,
    GenerationRequestRecord,
    init_db,
    resolve_image_ids_for_paths,
    save_generation_history,
    save_generation_history_detailed,
    set_image_discord_message,
    upsert_image_rating,
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

    def test_allowed_ratings_are_exactly_like_meh_and_off_target(self):
        self.assertEqual(set(ALLOWED_RATINGS), {"like", "meh", "off_target"})

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


if __name__ == "__main__":
    unittest.main()
