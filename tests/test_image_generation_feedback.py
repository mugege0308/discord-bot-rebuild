import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services.generation_history import (
    ALLOWED_ISSUE_CATEGORIES,
    SCHEMA_VERSION,
    FeedbackImageTarget,
    GenerationRequestDetail,
    GenerationRequestRecord,
    get_feedback_by_user_and_image,
    get_feedback_for_generation,
    get_generation_images_for_message,
    get_generation_request,
    init_db,
    list_all_feedback_rows,
    list_feedback_for_user,
    save_generation_history_detailed,
    set_image_discord_message,
    upsert_image_feedback,
)
from services.image_generation_feedback import (
    GENERATION_SNAPSHOT_SCHEMA_VERSION,
    UNKNOWN_CHARACTER_LABEL,
    build_generation_snapshot,
    get_accepted_generation_snapshots,
    get_character_feedback_summary,
    get_common_failure_categories,
    save_image_feedback,
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
        assumptions=("背景を補完",),
        workflow_file="workflows/comfy_workflow.json",
        workflow_hash="deadbeef",
        selection_used=False,
        selection_message="",
        model_key="animagine",
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


class _DbBackedTestCase(unittest.TestCase):
    def setUp(self):
        # ignore_cleanup_errors: WAL sidecar files can stay briefly locked on
        # Windows after the last sqlite3 connection is garbage-collected -
        # matches tests/test_generation_history.py's setUp.
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
        path = Path(self.tmp_dir.name) / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _save_one_image(self, *, request_text: str = "夕暮れの海辺", model_key: str = "animagine") -> tuple[int, int]:
        """Saves a request with a single image; returns (request_id, image_id)."""
        path = self._write_file("sd_outputs/img_01.png", b"PNG-BYTES-A")
        saved = save_generation_history_detailed(
            _make_request(request_text=request_text, model_key=model_key),
            [_make_image(str(path), batch_index=1)],
            selected_paths=[],
        )
        self.assertIsNotNone(saved)
        return saved.request_id, saved.image_ids[0]

    def _feedback_target(self, request_id: int, image_id: int, *, image_index: int = 1) -> FeedbackImageTarget:
        request = get_generation_request(request_id)
        self.assertIsNotNone(request)
        return FeedbackImageTarget(
            image_id=image_id,
            request_id=request_id,
            request_user_id=request.user_id,
            image_index=image_index,
            image_path="data/sd_outputs/img_01.png",
            seed=12345,
            checkpoint_name="animagineXL40Opt_clear.safetensors",
            sampler_name="euler_ancestral",
            scheduler="normal",
            steps=20,
            cfg=5.0,
            width=1024,
            height=1024,
            was_selected_by_ai=False,
        )


class SchemaTestCase(_DbBackedTestCase):
    def test_init_db_creates_feedback_table(self):
        init_db()
        with self._connect() as connection:
            names = {row["name"] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn("image_generation_feedback", names)

    def test_schema_version_is_current_and_composition_migration_is_defined(self):
        init_db()
        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(version, SCHEMA_VERSION)
        self.assertEqual(SCHEMA_VERSION, 6)

    def test_old_db_file_gains_the_feedback_table_on_next_init(self):
        # Simulates a DB file written before this feature existed: every
        # table except image_generation_feedback, stamped at the old
        # version. init_db() must bring it up to date without touching
        # existing rows.
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE generation_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, user_id INTEGER NOT NULL,
                    request_text TEXT NOT NULL, reference_id INTEGER, model_key TEXT, positive TEXT NOT NULL,
                    negative TEXT NOT NULL, summary TEXT NOT NULL, assumptions_json TEXT NOT NULL,
                    workflow_file TEXT, workflow_hash TEXT, selection_used INTEGER NOT NULL DEFAULT 0,
                    selection_message TEXT, parent_request_id INTEGER, regeneration_type TEXT, custom_instruction TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE generation_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, request_id INTEGER NOT NULL, image_path TEXT NOT NULL,
                    seed INTEGER, batch_index INTEGER NOT NULL, checkpoint_name TEXT, sampler_name TEXT,
                    scheduler TEXT, steps INTEGER, cfg REAL, width INTEGER, height INTEGER,
                    was_selected_by_ai INTEGER NOT NULL DEFAULT 0, discord_message_id INTEGER, discord_channel_id INTEGER
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE image_ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, image_id INTEGER NOT NULL, user_id INTEGER NOT NULL,
                    rating TEXT NOT NULL, rated_at TEXT NOT NULL, updated_at TEXT NOT NULL, UNIQUE(image_id, user_id)
                )
                """
            )
            connection.execute("PRAGMA user_version = 4")
            connection.commit()

        init_db()

        with self._connect() as connection:
            names = {row["name"] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(generation_requests)")}
            version = connection.execute("PRAGMA user_version").fetchone()[0]
        self.assertIn("image_generation_feedback", names)
        self.assertIn("composition_preset", columns)
        self.assertEqual(version, SCHEMA_VERSION)


class UpsertImageFeedbackTestCase(_DbBackedTestCase):
    def _rows(self, image_id: int) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM image_generation_feedback WHERE image_id = ? ORDER BY user_id", (image_id,)
            ).fetchall()

    def test_new_accepted_feedback_is_stored(self):
        _, image_id = self._save_one_image()

        self.assertTrue(
            upsert_image_feedback(
                image_id=image_id, image_index=1, image_path="p.png", user_id=42,
                guild_id=1, channel_id=2, message_id=3,
                accepted=True, issue_categories=(), comment=None,
                generation_snapshot_json="{}",
            )
        )

        rows = self._rows(image_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["accepted"], 1)
        self.assertEqual(json.loads(rows[0]["issue_categories_json"]), [])
        self.assertEqual(rows[0]["created_at"], rows[0]["updated_at"])

    def test_new_issue_feedback_with_multiple_categories_is_stored(self):
        _, image_id = self._save_one_image()

        self.assertTrue(
            upsert_image_feedback(
                image_id=image_id, image_index=1, image_path="p.png", user_id=42,
                guild_id=None, channel_id=None, message_id=None,
                accepted=False, issue_categories=["hair_mismatch", "outfit_mismatch"], comment=None,
                generation_snapshot_json="{}",
            )
        )

        rows = self._rows(image_id)
        self.assertEqual(json.loads(rows[0]["issue_categories_json"]), ["hair_mismatch", "outfit_mismatch"])
        self.assertEqual(rows[0]["accepted"], 0)

    def test_japanese_comment_is_stored_verbatim(self):
        _, image_id = self._save_one_image()
        comment = "顔の輪郭が別人みたいで、衣装の色も指定と違います。"

        self.assertTrue(
            upsert_image_feedback(
                image_id=image_id, image_index=1, image_path="p.png", user_id=42,
                guild_id=None, channel_id=None, message_id=None,
                accepted=False, issue_categories=["other"], comment=comment,
                generation_snapshot_json="{}",
            )
        )

        rows = self._rows(image_id)
        self.assertEqual(rows[0]["comment"], comment)

    def test_same_user_same_image_second_call_updates_instead_of_inserting(self):
        _, image_id = self._save_one_image()
        upsert_image_feedback(
            image_id=image_id, image_index=1, image_path="p.png", user_id=42,
            guild_id=None, channel_id=None, message_id=None,
            accepted=False, issue_categories=["low_quality"], comment="最初のコメント",
            generation_snapshot_json="{}",
        )

        self.assertTrue(
            upsert_image_feedback(
                image_id=image_id, image_index=1, image_path="p.png", user_id=42,
                guild_id=None, channel_id=None, message_id=None,
                accepted=True, issue_categories=(), comment=None,
                generation_snapshot_json="{}",
            )
        )

        rows = self._rows(image_id)
        self.assertEqual(len(rows), 1, "re-evaluation must update, not insert a new row")
        self.assertEqual(rows[0]["accepted"], 1)
        self.assertIsNone(rows[0]["comment"], "full replace clears the previous comment")
        self.assertLessEqual(rows[0]["created_at"], rows[0]["updated_at"])

    def test_different_users_on_the_same_image_get_independent_rows(self):
        _, image_id = self._save_one_image()

        upsert_image_feedback(
            image_id=image_id, image_index=1, image_path="p.png", user_id=1,
            guild_id=None, channel_id=None, message_id=None,
            accepted=True, issue_categories=(), comment=None, generation_snapshot_json="{}",
        )
        upsert_image_feedback(
            image_id=image_id, image_index=1, image_path="p.png", user_id=2,
            guild_id=None, channel_id=None, message_id=None,
            accepted=False, issue_categories=["face_mismatch"], comment=None, generation_snapshot_json="{}",
        )

        rows = self._rows(image_id)
        self.assertEqual(len(rows), 2)

    def test_rejects_unknown_category(self):
        _, image_id = self._save_one_image()

        self.assertFalse(
            upsert_image_feedback(
                image_id=image_id, image_index=1, image_path="p.png", user_id=42,
                guild_id=None, channel_id=None, message_id=None,
                accepted=False, issue_categories=["not_a_real_category"], comment=None,
                generation_snapshot_json="{}",
            )
        )
        self.assertEqual(self._rows(image_id), [])

    def test_rejects_accepted_and_categories_together(self):
        _, image_id = self._save_one_image()

        self.assertFalse(
            upsert_image_feedback(
                image_id=image_id, image_index=1, image_path="p.png", user_id=42,
                guild_id=None, channel_id=None, message_id=None,
                accepted=True, issue_categories=["hair_mismatch"], comment=None,
                generation_snapshot_json="{}",
            )
        )
        self.assertEqual(self._rows(image_id), [])

    def test_rejects_neither_accepted_nor_categories(self):
        _, image_id = self._save_one_image()

        self.assertFalse(
            upsert_image_feedback(
                image_id=image_id, image_index=1, image_path="p.png", user_id=42,
                guild_id=None, channel_id=None, message_id=None,
                accepted=False, issue_categories=[], comment="コメントだけ",
                generation_snapshot_json="{}",
            )
        )

    def test_nonexistent_image_id_is_rejected_not_raised(self):
        self.assertFalse(
            upsert_image_feedback(
                image_id=999999, image_index=1, image_path="p.png", user_id=42,
                guild_id=None, channel_id=None, message_id=None,
                accepted=True, issue_categories=(), comment=None,
                generation_snapshot_json="{}",
            )
        )

    def test_all_allowed_categories_are_individually_accepted(self):
        _, image_id = self._save_one_image()
        for position, category in enumerate(ALLOWED_ISSUE_CATEGORIES):
            with self.subTest(category=category):
                self.assertTrue(
                    upsert_image_feedback(
                        image_id=image_id, image_index=1, image_path="p.png", user_id=position,
                        guild_id=None, channel_id=None, message_id=None,
                        accepted=False, issue_categories=[category], comment=None,
                        generation_snapshot_json="{}",
                    )
                )


class ReadFunctionsTestCase(_DbBackedTestCase):
    def test_get_generation_images_for_message_returns_matches_in_batch_order(self):
        request_id, image_id = self._save_one_image()
        set_image_discord_message([image_id], message_id=555, channel_id=777)

        targets = get_generation_images_for_message(777, 555)

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].image_id, image_id)
        self.assertEqual(targets[0].request_id, request_id)
        self.assertEqual(targets[0].request_user_id, 42)

    def test_get_generation_images_for_message_returns_empty_when_not_found(self):
        self.assertEqual(get_generation_images_for_message(1, 2), [])

    def test_get_feedback_for_generation_returns_every_users_rows(self):
        _, image_id = self._save_one_image()
        upsert_image_feedback(
            image_id=image_id, image_index=1, image_path="p.png", user_id=1,
            guild_id=None, channel_id=None, message_id=None,
            accepted=True, issue_categories=(), comment=None, generation_snapshot_json="{}",
        )
        upsert_image_feedback(
            image_id=image_id, image_index=1, image_path="p.png", user_id=2,
            guild_id=None, channel_id=None, message_id=None,
            accepted=False, issue_categories=["low_quality"], comment=None, generation_snapshot_json="{}",
        )

        records = get_feedback_for_generation(image_id)
        self.assertEqual({record.user_id for record in records}, {1, 2})

    def test_get_feedback_by_user_and_image_missing_returns_none(self):
        _, image_id = self._save_one_image()
        self.assertIsNone(get_feedback_by_user_and_image(image_id, 42))

    def test_get_feedback_by_user_and_image_found(self):
        _, image_id = self._save_one_image()
        upsert_image_feedback(
            image_id=image_id, image_index=1, image_path="p.png", user_id=42,
            guild_id=None, channel_id=None, message_id=None,
            accepted=True, issue_categories=(), comment=None, generation_snapshot_json="{}",
        )
        record = get_feedback_by_user_and_image(image_id, 42)
        self.assertIsNotNone(record)
        self.assertTrue(record.accepted)

    def test_list_feedback_for_user_respects_limit_and_scoping(self):
        _, image_id = self._save_one_image()
        upsert_image_feedback(
            image_id=image_id, image_index=1, image_path="p.png", user_id=42,
            guild_id=None, channel_id=None, message_id=None,
            accepted=True, issue_categories=(), comment=None, generation_snapshot_json="{}",
        )
        upsert_image_feedback(
            image_id=image_id, image_index=1, image_path="p.png", user_id=999,
            guild_id=None, channel_id=None, message_id=None,
            accepted=True, issue_categories=(), comment=None, generation_snapshot_json="{}",
        )

        records = list_feedback_for_user(42, limit=10)
        self.assertEqual([record.user_id for record in records], [42])


class BuildGenerationSnapshotTestCase(_DbBackedTestCase):
    def test_snapshot_contains_prompt_and_image_fields_without_fabricating_untracked_ones(self):
        request_id, image_id = self._save_one_image(request_text="東方の古明地こいしが笑っている")
        request = get_generation_request(request_id)
        target = self._feedback_target(request_id, image_id)

        snapshot = build_generation_snapshot(request, target)

        self.assertEqual(snapshot["schema_version"], GENERATION_SNAPSHOT_SCHEMA_VERSION)
        self.assertEqual(snapshot["prompt"]["positive"], request.positive)
        self.assertEqual(snapshot["prompt"]["negative"], request.negative)
        self.assertEqual(snapshot["image"]["image_id"], image_id)
        self.assertEqual(snapshot["image"]["seed"], 12345)
        # Not tracked anywhere in this codebase - must stay null, never guessed.
        self.assertIsNone(snapshot["image"]["denoise"])
        self.assertIsNone(snapshot["lora"])
        self.assertIsNone(snapshot["ip_adapter"])
        self.assertEqual(snapshot["model_profile"]["model_key"], "animagine")

    def test_snapshot_resolves_known_character_from_request_text(self):
        request_id, image_id = self._save_one_image(request_text="東方の古明地こいしが笑っている")
        request = get_generation_request(request_id)
        target = self._feedback_target(request_id, image_id)

        snapshot = build_generation_snapshot(request, target)

        tags = [entry["character_tag"] for entry in snapshot["character_tags"]["resolutions"]]
        self.assertTrue(tags, "known character in request_text must resolve to at least one tag")
        self.assertEqual(snapshot["character_tags"]["resolved_at"], "feedback_save_time")

    def test_snapshot_has_no_character_resolution_for_unrelated_text(self):
        request_id, image_id = self._save_one_image(request_text="猫耳の女の子が公園にいる")
        request = get_generation_request(request_id)
        target = self._feedback_target(request_id, image_id)

        snapshot = build_generation_snapshot(request, target)

        self.assertEqual(snapshot["character_tags"]["resolutions"], [])

    def test_snapshot_handles_unknown_model_key_without_raising(self):
        # get_generation_request() always resolves model_key to something
        # valid (falling back to checkpoint inference, then the default
        # profile - see _resolve_request_model_key), so an unresolvable key
        # can only reach build_generation_snapshot via a directly
        # constructed GenerationRequestDetail, as here.
        request_id, image_id = self._save_one_image()
        target = self._feedback_target(request_id, image_id)
        request = GenerationRequestDetail(
            id=request_id, created_at="2026-01-01T00:00:00+09:00", user_id=42,
            request_text="夕暮れの海辺", reference_id=None,
            positive="1girl", negative="lowres", summary="summary", assumptions=(),
            workflow_file=None, workflow_hash=None, selection_used=False, selection_message="",
            parent_request_id=None, regeneration_type=None, custom_instruction=None,
            model_key="totally_unknown_model",
        )

        snapshot = build_generation_snapshot(request, target)

        self.assertIsNone(snapshot["model_profile"])

    def test_snapshot_is_json_serializable_even_with_a_bad_character_resolution(self):
        request_id, image_id = self._save_one_image(request_text="東方の古明地こいしが笑っている")
        request = get_generation_request(request_id)
        target = self._feedback_target(request_id, image_id)

        with mock.patch(
            "services.image_generation_feedback.resolve_all_character_tags",
            side_effect=RuntimeError("dictionary exploded"),
        ):
            snapshot = build_generation_snapshot(request, target)

        self.assertEqual(snapshot["character_tags"]["resolutions"], [])
        json.dumps(snapshot, ensure_ascii=False)  # must not raise


class SaveImageFeedbackTestCase(_DbBackedTestCase):
    def test_save_image_feedback_persists_snapshot_and_row(self):
        request_id, image_id = self._save_one_image(request_text="東方の古明地こいしが笑っている")
        request = get_generation_request(request_id)
        target = self._feedback_target(request_id, image_id)

        self.assertTrue(
            save_image_feedback(
                request=request, image=target, user_id=42, guild_id=1, channel_id=2, message_id=3,
                accepted=False, issue_categories=["composition_mismatch", "other"], comment="構図が思っていたのと違う",
            )
        )

        record = get_feedback_by_user_and_image(image_id, 42)
        self.assertIsNotNone(record)
        self.assertEqual(set(record.issue_categories), {"composition_mismatch", "other"})
        self.assertEqual(record.comment, "構図が思っていたのと違う")
        self.assertEqual(record.generation_snapshot["request"]["request_id"], request_id)
        self.assertGreater(len(record.generation_snapshot["character_tags"]["resolutions"]), 0)

    def test_save_image_feedback_with_non_serializable_snapshot_value_still_saves(self):
        request_id, image_id = self._save_one_image()
        request = get_generation_request(request_id)
        target = self._feedback_target(request_id, image_id)

        class _Unserializable:
            def __repr__(self):
                return "<unserializable>"

        with mock.patch(
            "services.image_generation_feedback.build_generation_snapshot",
            return_value={"schema_version": 1, "broken": _Unserializable()},
        ):
            saved = save_image_feedback(
                request=request, image=target, user_id=42, guild_id=None, channel_id=None, message_id=None,
                accepted=True, issue_categories=(), comment=None,
            )

        self.assertTrue(saved)
        record = get_feedback_by_user_and_image(image_id, 42)
        self.assertIsNotNone(record)
        self.assertIn("broken", record.generation_snapshot)


class AggregationTestCase(_DbBackedTestCase):
    def _feedback(self, request_text: str, *, user_id: int, accepted: bool, categories=(), model_key: str = "animagine"):
        request_id, image_id = self._save_one_image(request_text=request_text, model_key=model_key)
        request = get_generation_request(request_id)
        target = self._feedback_target(request_id, image_id)
        save_image_feedback(
            request=request, image=target, user_id=user_id, guild_id=None, channel_id=None, message_id=None,
            accepted=accepted, issue_categories=categories, comment=None,
        )
        return image_id

    def test_character_feedback_summary_groups_by_resolved_character(self):
        self._feedback("東方の古明地こいしが笑っている", user_id=1, accepted=True)
        self._feedback("東方の古明地こいしが怒っている", user_id=2, accepted=False, categories=["hair_mismatch"])
        self._feedback("猫耳の女の子が公園にいる", user_id=3, accepted=True)

        summaries = get_character_feedback_summary()
        by_tag = {summary.character_tag: summary for summary in summaries}

        koishi = next(summary for summary in summaries if summary.character_tag != UNKNOWN_CHARACTER_LABEL)
        self.assertEqual(koishi.total, 2)
        self.assertEqual(koishi.accepted, 1)
        self.assertEqual(koishi.hair_mismatch, 1)
        self.assertAlmostEqual(koishi.acceptance_rate, 0.5)

        self.assertIn(UNKNOWN_CHARACTER_LABEL, by_tag)
        self.assertEqual(by_tag[UNKNOWN_CHARACTER_LABEL].total, 1)

    def test_common_failure_categories_includes_every_known_category_at_least_at_zero(self):
        self._feedback("東方の古明地こいしが笑っている", user_id=1, accepted=False, categories=["low_quality"])

        results = {entry.category: entry.count for entry in get_common_failure_categories()}

        self.assertEqual(set(results.keys()), set(ALLOWED_ISSUE_CATEGORIES))
        self.assertEqual(results["low_quality"], 1)
        self.assertEqual(results["face_mismatch"], 0)

    def test_common_failure_categories_can_filter_by_character(self):
        self._feedback("東方の古明地こいしが笑っている", user_id=1, accepted=False, categories=["low_quality"])
        self._feedback("猫耳の女の子が公園にいる", user_id=2, accepted=False, categories=["outfit_mismatch"])

        koishi_tag = get_character_feedback_summary()
        tag = next(summary.character_tag for summary in koishi_tag if summary.character_tag != UNKNOWN_CHARACTER_LABEL)

        filtered = {entry.category: entry.count for entry in get_common_failure_categories(tag)}
        self.assertEqual(filtered["low_quality"], 1)
        self.assertEqual(filtered["outfit_mismatch"], 0)

    def test_get_accepted_generation_snapshots_only_returns_accepted_rows(self):
        self._feedback("東方の古明地こいしが笑っている", user_id=1, accepted=True)
        self._feedback("東方の古明地こいしが怒っている", user_id=2, accepted=False, categories=["low_quality"])

        snapshots = get_accepted_generation_snapshots()

        self.assertEqual(len(snapshots), 1)
        self.assertIn("request", snapshots[0])

    def test_get_accepted_generation_snapshots_respects_limit(self):
        for index in range(3):
            self._feedback("東方の古明地こいしが笑っている", user_id=index, accepted=True)

        self.assertEqual(len(get_accepted_generation_snapshots(limit=2)), 2)


if __name__ == "__main__":
    unittest.main()
