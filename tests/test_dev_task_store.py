import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import services.dev_task_store as store
from services.dev_task_router import DevTaskEngine


class DevTaskStoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="dev_task_store_test_")
        self._original_data_dir = store.DATA_DIR
        store.DATA_DIR = Path(self._tmp_dir.name) / "dev_tasks"

    def tearDown(self):
        store.DATA_DIR = self._original_data_dir
        self._tmp_dir.cleanup()

    def _create_task(self, **overrides):
        params = dict(
            requested_by_user_id=111,
            guild_id=222,
            channel_id=333,
            prompt_text="バグを修正して",
            routed_engine=DevTaskEngine.CODEX.value,
            routing_reason="小規模な修正を示すキーワードを検出したため",
            engine_overridden=False,
            timeout_seconds=600,
        )
        params.update(overrides)
        return store.create_task(**params)

    def _write_raw(self, task_id: str, payload: dict) -> Path:
        store.DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = store.DATA_DIR / f"{task_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _valid_payload(self, task_id: str, **overrides) -> dict:
        payload = {
            "task_id": task_id,
            "status": store.DevTaskStatus.PENDING_APPROVAL.value,
            "requested_by_user_id": 111,
            "guild_id": 222,
            "channel_id": 333,
            "requested_at": "2026-07-15T00:00:00.000000+00:00",
            "prompt_text": "バグを修正して",
            "routed_engine": DevTaskEngine.CODEX.value,
            "routing_reason": "reason",
            "engine_overridden": False,
            "timeout_seconds": 600,
            "error_message": None,
        }
        payload.update(overrides)
        return payload

    # --- create / get / update basics ---

    def test_create_task_starts_pending_approval_with_uuid4_task_id(self):
        result = self._create_task()
        self.assertTrue(result.success)
        self.assertEqual(result.task.status, store.DevTaskStatus.PENDING_APPROVAL.value)
        self.assertEqual(uuid.UUID(result.task.task_id).version, 4)

    def test_get_task_roundtrip(self):
        task = self._create_task().task
        fetched = store.get_task(task.task_id)
        self.assertTrue(fetched.success)
        self.assertEqual(fetched.task, task)

    def test_get_task_nonexistent_id_fails(self):
        result = store.get_task(str(uuid.uuid4()))
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_TASK_NOT_FOUND)

    def test_get_latest_task_for_user_returns_most_recent(self):
        self._create_task(prompt_text="最初のタスク")
        second = self._create_task(prompt_text="次のタスク").task
        latest = store.get_latest_task_for_user(111)
        self.assertTrue(latest.success)
        self.assertEqual(latest.task.task_id, second.task_id)

    def test_get_latest_task_for_user_no_tasks_fails(self):
        result = store.get_latest_task_for_user(999999)
        self.assertFalse(result.success)

    def test_atomic_save_leaves_no_tmp_file(self):
        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)
        self.assertEqual(list(store.DATA_DIR.glob("*.tmp")), [])

    # --- update_task: allow-list / immutability ---

    def test_update_task_accepts_allowed_field(self):
        task = self._create_task().task
        result = store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)
        self.assertTrue(result.success)
        self.assertEqual(result.task.status, store.DevTaskStatus.RUNNING.value)

    def test_update_task_rejects_immutable_field(self):
        task = self._create_task().task
        result = store.update_task(task.task_id, prompt_text="改変しようとする")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_IMMUTABLE_FIELD)
        self.assertEqual(store.get_task(task.task_id).task.prompt_text, "バグを修正して")

    def test_update_task_rejects_unknown_field(self):
        task = self._create_task().task
        result = store.update_task(task.task_id, some_unknown_field="x")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_UNKNOWN_FIELD)

    def test_update_task_rejects_invalid_status_value(self):
        task = self._create_task().task
        result = store.update_task(task.task_id, status="not_a_real_status")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    def test_update_task_rejects_string_false_for_engine_overridden(self):
        task = self._create_task().task
        result = store.update_task(task.task_id, engine_overridden="false")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)
        self.assertFalse(store.get_task(task.task_id).task.engine_overridden)

    def test_update_task_rejects_negative_timeout(self):
        task = self._create_task().task
        result = store.update_task(task.task_id, timeout_seconds=-1)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    def test_update_task_rejects_bool_timeout(self):
        task = self._create_task().task
        result = store.update_task(task.task_id, timeout_seconds=True)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    def test_update_task_failure_does_not_modify_existing_json(self):
        task = self._create_task().task
        before = store.get_task(task.task_id).task
        store.update_task(task.task_id, timeout_seconds=-1)
        after = store.get_task(task.task_id).task
        self.assertEqual(before, after)

    # --- state transitions ---

    def test_valid_transition_pending_to_running(self):
        task = self._create_task().task
        result = store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)
        self.assertTrue(result.success)

    def test_valid_transition_pending_to_failed(self):
        task = self._create_task().task
        result = store.update_task(
            task.task_id, status=store.DevTaskStatus.FAILED.value, error_message="失敗理由"
        )
        self.assertTrue(result.success)

    def test_valid_transition_running_to_completed(self):
        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)
        result = store.update_task(task.task_id, status=store.DevTaskStatus.COMPLETED.value)
        self.assertTrue(result.success)

    def test_valid_transition_running_to_failed(self):
        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)
        result = store.update_task(
            task.task_id, status=store.DevTaskStatus.FAILED.value, error_message="失敗理由"
        )
        self.assertTrue(result.success)

    def test_invalid_transition_completed_to_running_rejected(self):
        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)
        store.update_task(task.task_id, status=store.DevTaskStatus.COMPLETED.value)
        result = store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_TRANSITION)

    def test_invalid_transition_failed_to_running_rejected(self):
        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.FAILED.value, error_message="失敗理由")
        result = store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_TRANSITION)

    def test_same_state_transition_is_rejected(self):
        # Explicit design decision: Phase 1's transition table only lists real
        # state changes, so requesting the current status again is treated the
        # same as any other transition absent from the table.
        task = self._create_task().task
        result = store.update_task(task.task_id, status=store.DevTaskStatus.PENDING_APPROVAL.value)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_TRANSITION)

    # --- task_id filename vs. content integrity ---

    def test_mark_running_skips_content_task_id_mismatch(self):
        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)

        path = store.DATA_DIR / f"{task.task_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["task_id"] = str(uuid.uuid4())
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        changed = store.mark_running_tasks_failed_on_startup()
        self.assertEqual(changed, 0)

        raw_after = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(raw_after["status"], store.DevTaskStatus.RUNNING.value)

    def test_mark_running_content_task_id_cannot_escape_data_dir(self):
        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)

        path = store.DATA_DIR / f"{task.task_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["task_id"] = "../../../../escaped"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        changed = store.mark_running_tasks_failed_on_startup()
        self.assertEqual(changed, 0)

        # No file was ever written outside DATA_DIR using the malicious id.
        for ancestor in [store.DATA_DIR.parent, *store.DATA_DIR.parents]:
            self.assertFalse((ancestor / "escaped.json").exists())
        self.assertEqual([p.name for p in store.DATA_DIR.glob("*.json")], [f"{task.task_id}.json"])

    # --- invalid UTF-8 handling ---

    def test_get_task_handles_invalid_utf8_safely(self):
        task_id = str(uuid.uuid4())
        store.DATA_DIR.mkdir(parents=True, exist_ok=True)
        (store.DATA_DIR / f"{task_id}.json").write_bytes(b"\xff\xfe\x00\x01not valid utf-8")

        result = store.get_task(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_READ_FAILED)

    def test_mark_running_skips_invalid_utf8_file_and_continues_with_valid_ones(self):
        broken_id = str(uuid.uuid4())
        store.DATA_DIR.mkdir(parents=True, exist_ok=True)
        (store.DATA_DIR / f"{broken_id}.json").write_bytes(b"\xff\xfe\x00\x01broken")

        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)

        changed = store.mark_running_tasks_failed_on_startup()
        self.assertEqual(changed, 1)

        result = store.get_task(task.task_id)
        self.assertEqual(result.task.status, store.DevTaskStatus.FAILED.value)

    # --- directory creation failure ---

    def test_create_task_returns_failure_when_data_dir_cannot_be_created(self):
        store.DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
        store.DATA_DIR.write_text("this occupies the path as a file, not a directory", encoding="utf-8")

        result = self._create_task()
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_SAVE_FAILED)

    # --- startup failover scope ---

    def test_mark_running_only_changes_running_tasks(self):
        pending = self._create_task().task
        running = self._create_task().task
        store.update_task(running.task_id, status=store.DevTaskStatus.RUNNING.value)

        changed = store.mark_running_tasks_failed_on_startup()
        self.assertEqual(changed, 1)

        self.assertEqual(
            store.get_task(pending.task_id).task.status, store.DevTaskStatus.PENDING_APPROVAL.value
        )
        after_running = store.get_task(running.task_id).task
        self.assertEqual(after_running.status, store.DevTaskStatus.FAILED.value)
        self.assertIsNotNone(after_running.error_message)

    def test_mark_running_second_pass_is_a_no_op(self):
        running = self._create_task().task
        store.update_task(running.task_id, status=store.DevTaskStatus.RUNNING.value)

        store.mark_running_tasks_failed_on_startup()
        second_pass = store.mark_running_tasks_failed_on_startup()
        self.assertEqual(second_pass, 0)

    # --- strict UUID4 validation ---

    def test_get_task_rejects_uuid1(self):
        result = store.get_task(str(uuid.uuid1()))
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_TASK_ID)

    def test_get_task_rejects_uuid3(self):
        result = store.get_task(str(uuid.uuid3(uuid.NAMESPACE_DNS, "example.com")))
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_TASK_ID)

    def test_get_task_rejects_uuid5(self):
        result = store.get_task(str(uuid.uuid5(uuid.NAMESPACE_DNS, "example.com")))
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_TASK_ID)

    def test_get_task_rejects_malformed_string(self):
        result = store.get_task("not-a-uuid-at-all")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_TASK_ID)

    def test_mark_running_skips_uuid1_named_file(self):
        uuid1_id = str(uuid.uuid1())
        payload = self._valid_payload(uuid1_id, status=store.DevTaskStatus.RUNNING.value)
        path = self._write_raw(uuid1_id, payload)

        changed = store.mark_running_tasks_failed_on_startup()
        self.assertEqual(changed, 0)
        raw_after = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(raw_after["status"], store.DevTaskStatus.RUNNING.value)

    # --- get_task/update_task: filename vs. content task_id consistency ---

    def test_get_task_rejects_content_id_mismatch(self):
        task_id = str(uuid.uuid4())
        other_id = str(uuid.uuid4())
        self._write_raw(task_id, self._valid_payload(other_id))

        result = store.get_task(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_CORRUPTED)

    def test_update_task_rejects_content_id_mismatch(self):
        task_id = str(uuid.uuid4())
        other_id = str(uuid.uuid4())
        path = self._write_raw(task_id, self._valid_payload(other_id))
        before_raw = path.read_text(encoding="utf-8")

        result = store.update_task(task_id, status=store.DevTaskStatus.RUNNING.value)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_CORRUPTED)
        self.assertEqual(path.read_text(encoding="utf-8"), before_raw)

    # --- strict semantic validation of stored records ---

    def test_get_task_rejects_invalid_status_in_record(self):
        task_id = str(uuid.uuid4())
        self._write_raw(task_id, self._valid_payload(task_id, status="bogus"))
        result = store.get_task(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_CORRUPTED)

    def test_get_task_rejects_invalid_engine_in_record(self):
        task_id = str(uuid.uuid4())
        self._write_raw(task_id, self._valid_payload(task_id, routed_engine="gpt"))
        result = store.get_task(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_CORRUPTED)

    def test_get_task_rejects_string_false_engine_overridden_in_record(self):
        task_id = str(uuid.uuid4())
        self._write_raw(task_id, self._valid_payload(task_id, engine_overridden="false"))
        result = store.get_task(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_CORRUPTED)

    def test_get_task_rejects_negative_timeout_in_record(self):
        task_id = str(uuid.uuid4())
        self._write_raw(task_id, self._valid_payload(task_id, timeout_seconds=-5))
        result = store.get_task(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_CORRUPTED)

    def test_get_task_rejects_bool_timeout_in_record(self):
        task_id = str(uuid.uuid4())
        self._write_raw(task_id, self._valid_payload(task_id, timeout_seconds=True))
        result = store.get_task(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_CORRUPTED)

    def test_get_task_rejects_invalid_requested_at(self):
        task_id = str(uuid.uuid4())
        self._write_raw(task_id, self._valid_payload(task_id, requested_at="not-a-timestamp"))
        result = store.get_task(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_CORRUPTED)

    def test_get_task_rejects_bool_id_fields_in_record(self):
        for field_name in ("requested_by_user_id", "guild_id", "channel_id"):
            with self.subTest(field=field_name):
                broken_id = str(uuid.uuid4())
                self._write_raw(broken_id, self._valid_payload(broken_id, **{field_name: True}))
                result = store.get_task(broken_id)
                self.assertFalse(result.success)
                self.assertEqual(result.error_code, store.ERROR_CORRUPTED)

    def test_get_latest_task_for_user_skips_invalid_requested_at(self):
        valid_task = self._create_task().task
        bad_id = str(uuid.uuid4())
        self._write_raw(bad_id, self._valid_payload(bad_id, requested_at="not-a-timestamp"))

        result = store.get_latest_task_for_user(111)
        self.assertTrue(result.success)
        self.assertEqual(result.task.task_id, valid_task.task_id)

    def test_get_latest_task_for_user_deterministic_on_tied_requested_at(self):
        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())
        tied_at = "2026-07-15T00:00:00.000000+00:00"
        self._write_raw(id_a, self._valid_payload(id_a, requested_at=tied_at))
        self._write_raw(id_b, self._valid_payload(id_b, requested_at=tied_at))

        expected_winner = min(id_a, id_b)
        first = store.get_latest_task_for_user(111)
        second = store.get_latest_task_for_user(111)
        self.assertTrue(first.success)
        self.assertEqual(first.task.task_id, expected_winner)
        self.assertEqual(first.task.task_id, second.task.task_id)

    def test_mark_running_skips_corrupted_record_and_continues(self):
        corrupted_id = str(uuid.uuid4())
        self._write_raw(
            corrupted_id,
            self._valid_payload(corrupted_id, status=store.DevTaskStatus.RUNNING.value, timeout_seconds=-1),
        )

        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)

        changed = store.mark_running_tasks_failed_on_startup()
        self.assertEqual(changed, 1)
        self.assertEqual(store.get_task(task.task_id).task.status, store.DevTaskStatus.FAILED.value)

    # --- serialization failures and temp file cleanup ---

    def test_create_task_rejects_unencodable_prompt_text(self):
        # A lone surrogate is now rejected by input validation (_is_utf8_safe_str)
        # before task_id generation or any write is attempted, not at the
        # json.dumps().encode("utf-8") stage in _write_task_atomic().
        result = self._create_task(prompt_text="\ud800")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)
        self.assertEqual(list(store.DATA_DIR.glob("*.json")), [])
        self.assertEqual(list(store.DATA_DIR.glob("*.tmp")), [])

    def test_update_task_rejects_unencodable_error_message_and_preserves_file(self):
        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)
        path = store.DATA_DIR / f"{task.task_id}.json"
        before_raw = path.read_text(encoding="utf-8")

        result = store.update_task(task.task_id, status=store.DevTaskStatus.FAILED.value, error_message="\ud800")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)
        self.assertEqual(path.read_text(encoding="utf-8"), before_raw)
        self.assertEqual(list(store.DATA_DIR.glob("*.tmp")), [])

    def test_write_failure_during_os_replace_cleans_up_tmp_file(self):
        task = self._create_task().task
        with mock.patch("services.dev_task_store.os.replace", side_effect=OSError("boom")):
            result = store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_SAVE_FAILED)
        self.assertEqual(list(store.DATA_DIR.glob("*.tmp")), [])
        self.assertEqual(store.get_task(task.task_id).task.status, store.DevTaskStatus.PENDING_APPROVAL.value)

    def test_write_does_not_remove_unrelated_tmp_file(self):
        task = self._create_task().task
        stray_path = store.DATA_DIR / "some_other_process.tmp"
        stray_path.write_text("keep me", encoding="utf-8")

        result = store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)
        self.assertTrue(result.success)
        self.assertTrue(stray_path.exists())
        self.assertEqual(stray_path.read_text(encoding="utf-8"), "keep me")

    def test_repeated_saves_do_not_collide_on_tmp_filenames(self):
        for i in range(20):
            result = self._create_task(prompt_text=f"タスク{i}")
            self.assertTrue(result.success)

        self.assertEqual(len(list(store.DATA_DIR.glob("*.json"))), 20)
        self.assertEqual(list(store.DATA_DIR.glob("*.tmp")), [])

    # --- create_task: strict input validation ---

    def test_create_task_rejects_invalid_engine(self):
        result = self._create_task(routed_engine="gpt")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)
        self.assertEqual(list(store.DATA_DIR.glob("*.json")), [])

    def test_create_task_rejects_string_false_engine_overridden(self):
        result = self._create_task(engine_overridden="false")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)
        self.assertEqual(list(store.DATA_DIR.glob("*.json")), [])

    def test_create_task_rejects_negative_timeout(self):
        result = self._create_task(timeout_seconds=-1)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)
        self.assertEqual(list(store.DATA_DIR.glob("*.json")), [])

    def test_create_task_rejects_bool_timeout(self):
        result = self._create_task(timeout_seconds=True)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)
        self.assertEqual(list(store.DATA_DIR.glob("*.json")), [])

    def test_create_task_rejects_bool_snowflake_id(self):
        result = self._create_task(requested_by_user_id=True)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)
        self.assertEqual(list(store.DATA_DIR.glob("*.json")), [])

    # --- field-level update permissions (FIELD_ALLOWED_STATUSES) ---

    def test_update_task_allows_timeout_change_while_pending_approval(self):
        task = self._create_task().task
        result = store.update_task(task.task_id, timeout_seconds=1200)
        self.assertTrue(result.success)
        self.assertEqual(result.task.timeout_seconds, 1200)

    def test_update_task_rejects_timeout_change_after_running(self):
        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)

        result = store.update_task(task.task_id, timeout_seconds=1200)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_FIELD_LOCKED)
        self.assertEqual(store.get_task(task.task_id).task.timeout_seconds, 600)

    def test_update_task_rejects_routed_engine_change_after_running(self):
        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)

        result = store.update_task(task.task_id, routed_engine=DevTaskEngine.CLAUDE.value)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_FIELD_LOCKED)

    def test_update_task_rejects_field_change_after_completed(self):
        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)
        store.update_task(task.task_id, status=store.DevTaskStatus.COMPLETED.value)

        result = store.update_task(task.task_id, timeout_seconds=1200)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_FIELD_LOCKED)

        result2 = store.update_task(task.task_id, error_message="後から書き換え")
        self.assertFalse(result2.success)
        self.assertEqual(result2.error_code, store.ERROR_FIELD_LOCKED)

    def test_update_task_multiple_fields_all_or_nothing(self):
        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)
        before = store.get_task(task.task_id).task

        # error_message is editable while running, but timeout_seconds is
        # locked once the task has left pending_approval; the whole call
        # must be rejected, including the otherwise-valid error_message change.
        result = store.update_task(task.task_id, error_message="進行中のメモ", timeout_seconds=1200)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_FIELD_LOCKED)

        after = store.get_task(task.task_id).task
        self.assertEqual(before, after)

    # --- startup failover: partial write failure continuation ---

    def test_mark_running_continues_after_one_write_failure(self):
        task_a = self._create_task(prompt_text="A").task
        task_b = self._create_task(prompt_text="B").task
        store.update_task(task_a.task_id, status=store.DevTaskStatus.RUNNING.value)
        store.update_task(task_b.task_id, status=store.DevTaskStatus.RUNNING.value)

        real_write = store._write_task_atomic

        def flaky_write(task_id, payload):
            if task_id == task_a.task_id:
                return False
            return real_write(task_id, payload)

        with mock.patch("services.dev_task_store._write_task_atomic", side_effect=flaky_write):
            changed = store.mark_running_tasks_failed_on_startup()

        self.assertEqual(changed, 1)
        self.assertEqual(store.get_task(task_a.task_id).task.status, store.DevTaskStatus.RUNNING.value)
        self.assertEqual(store.get_task(task_b.task_id).task.status, store.DevTaskStatus.FAILED.value)

    # --- get_latest_task_for_user: datetime comparison, not string order ---

    def test_get_latest_task_for_user_uses_datetime_not_string_order(self):
        # String order would pick "20261101T00:00:00+00:00" (Nov 1, basic
        # ISO format with no hyphens) over "2026-12-01T00:00:00+00:00"
        # (Dec 1), because '-' (0x2D) sorts before '1' (0x31). Chronologically
        # Dec 1 is later, so comparison must go through datetime parsing.
        earlier_id = str(uuid.uuid4())
        later_id = str(uuid.uuid4())
        self._write_raw(earlier_id, self._valid_payload(earlier_id, requested_at="20261101T00:00:00+00:00"))
        self._write_raw(later_id, self._valid_payload(later_id, requested_at="2026-12-01T00:00:00+00:00"))

        result = store.get_latest_task_for_user(111)
        self.assertTrue(result.success)
        self.assertEqual(result.task.task_id, later_id)

    def test_get_latest_task_for_user_compares_z_and_offset_notation(self):
        earlier_id = str(uuid.uuid4())
        later_id = str(uuid.uuid4())
        self._write_raw(earlier_id, self._valid_payload(earlier_id, requested_at="2026-07-15T00:00:00Z"))
        self._write_raw(later_id, self._valid_payload(later_id, requested_at="2026-07-15T00:00:01+00:00"))

        result = store.get_latest_task_for_user(111)
        self.assertTrue(result.success)
        self.assertEqual(result.task.task_id, later_id)

    def test_get_latest_task_for_user_compares_microsecond_precision(self):
        earlier_id = str(uuid.uuid4())
        later_id = str(uuid.uuid4())
        self._write_raw(earlier_id, self._valid_payload(earlier_id, requested_at="2026-07-15T00:00:00+00:00"))
        self._write_raw(
            later_id, self._valid_payload(later_id, requested_at="2026-07-15T00:00:00.500000+00:00")
        )

        result = store.get_latest_task_for_user(111)
        self.assertTrue(result.success)
        self.assertEqual(result.task.task_id, later_id)

    def test_get_latest_task_for_user_selects_latest_valid_when_corrupted_present(self):
        older_id = str(uuid.uuid4())
        newer_id = str(uuid.uuid4())
        corrupted_id = str(uuid.uuid4())
        self._write_raw(older_id, self._valid_payload(older_id, requested_at="2026-07-15T00:00:00+00:00"))
        self._write_raw(newer_id, self._valid_payload(newer_id, requested_at="2026-07-15T00:00:01+00:00"))
        self._write_raw(corrupted_id, self._valid_payload(corrupted_id, requested_at="not-a-timestamp"))

        result = store.get_latest_task_for_user(111)
        self.assertTrue(result.success)
        self.assertEqual(result.task.task_id, newer_id)

    # --- lone surrogates in stored string fields ---

    def _write_payload_with_lone_surrogate(self, task_id: str, field_name: str, **extra) -> Path:
        # Build the record with an ASCII placeholder via the normal
        # json.dumps() path (which would itself fail to encode a real lone
        # surrogate), then substitute in a literal "\ud800" JSON escape
        # sequence as text. That escape is valid ASCII in the file on disk,
        # but json.loads() decodes it back into a python str containing an
        # unpaired surrogate - a value that reads as UTF-8 but cannot be
        # re-encoded to it, exactly like the corruption this module guards
        # against.
        payload = self._valid_payload(task_id, **{field_name: "__SURROGATE__"}, **extra)
        text = json.dumps(payload, ensure_ascii=False, indent=2).replace("__SURROGATE__", "\\ud800")
        store.DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = store.DATA_DIR / f"{task_id}.json"
        path.write_text(text, encoding="utf-8")
        return path

    def test_get_task_rejects_lone_surrogate_in_prompt_text(self):
        task_id = str(uuid.uuid4())
        self._write_payload_with_lone_surrogate(task_id, "prompt_text")
        result = store.get_task(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_CORRUPTED)

    def test_get_task_rejects_lone_surrogate_in_routing_reason(self):
        task_id = str(uuid.uuid4())
        self._write_payload_with_lone_surrogate(task_id, "routing_reason")
        result = store.get_task(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_CORRUPTED)

    def test_get_task_rejects_lone_surrogate_in_error_message(self):
        task_id = str(uuid.uuid4())
        self._write_payload_with_lone_surrogate(task_id, "error_message")
        result = store.get_task(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_CORRUPTED)

    def test_get_latest_task_for_user_skips_lone_surrogate_record(self):
        valid_task = self._create_task().task
        bad_id = str(uuid.uuid4())
        self._write_payload_with_lone_surrogate(bad_id, "prompt_text")

        result = store.get_latest_task_for_user(111)
        self.assertTrue(result.success)
        self.assertEqual(result.task.task_id, valid_task.task_id)

    def test_mark_running_skips_lone_surrogate_record_and_continues(self):
        bad_id = str(uuid.uuid4())
        self._write_payload_with_lone_surrogate(
            bad_id, "prompt_text", status=store.DevTaskStatus.RUNNING.value
        )

        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)

        changed = store.mark_running_tasks_failed_on_startup()
        self.assertEqual(changed, 1)
        self.assertEqual(store.get_task(task.task_id).task.status, store.DevTaskStatus.FAILED.value)

    # --- os.fdopen() failure: fd/tmp ownership on write (Fix 3) ---

    def test_fdopen_failure_closes_fd_removes_tmp_preserves_unrelated_and_allows_recovery(self):
        task = self._create_task().task
        stray_path = store.DATA_DIR / "unrelated_process.tmp"
        stray_path.write_text("keep me", encoding="utf-8")

        captured = {}

        def failing_fdopen(fd, mode):
            captured["fd"] = fd
            raise OSError("boom")

        with mock.patch("services.dev_task_store.os.fdopen", side_effect=failing_fdopen):
            result = store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_SAVE_FAILED)

        # The fd handed to os.fdopen() must have been closed explicitly by
        # production code; closing an already-closed fd raises OSError.
        with self.assertRaises(OSError):
            os.close(captured["fd"])

        # Only this call's own temp file is removed; an unrelated stray
        # .tmp file (e.g. from a concurrent writer) is left untouched.
        self.assertEqual(list(store.DATA_DIR.glob("*.tmp")), [stray_path])
        self.assertEqual(stray_path.read_text(encoding="utf-8"), "keep me")

        # A subsequent save is unaffected by the earlier failure.
        follow_up = store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)
        self.assertTrue(follow_up.success)

    # --- get_latest_task_for_user: user_id validation (Fix 4) ---

    def test_get_latest_task_for_user_rejects_bool_true(self):
        result = store.get_latest_task_for_user(True)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    def test_get_latest_task_for_user_rejects_bool_false(self):
        result = store.get_latest_task_for_user(False)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    def test_get_latest_task_for_user_rejects_string(self):
        result = store.get_latest_task_for_user("111")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    def test_get_latest_task_for_user_rejects_zero(self):
        result = store.get_latest_task_for_user(0)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    def test_get_latest_task_for_user_rejects_negative(self):
        result = store.get_latest_task_for_user(-1)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    def test_get_latest_task_for_user_allows_positive_int(self):
        task = self._create_task().task
        result = store.get_latest_task_for_user(111)
        self.assertTrue(result.success)
        self.assertEqual(result.task.task_id, task.task_id)

    # --- Snowflake / timeout upper bounds ---

    def test_create_task_allows_max_snowflake_id(self):
        result = self._create_task(requested_by_user_id=store.MAX_SNOWFLAKE_ID)
        self.assertTrue(result.success)

    def test_create_task_rejects_snowflake_id_above_max(self):
        result = self._create_task(requested_by_user_id=store.MAX_SNOWFLAKE_ID + 1)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    def test_get_latest_task_for_user_rejects_snowflake_id_above_max(self):
        result = store.get_latest_task_for_user(store.MAX_SNOWFLAKE_ID + 1)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    def test_create_task_allows_max_timeout_seconds(self):
        result = self._create_task(timeout_seconds=store.MAX_DEV_TASK_TIMEOUT_SECONDS)
        self.assertTrue(result.success)

    def test_create_task_rejects_timeout_seconds_above_max(self):
        result = self._create_task(timeout_seconds=store.MAX_DEV_TASK_TIMEOUT_SECONDS + 1)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    def test_update_task_rejects_timeout_seconds_above_max(self):
        task = self._create_task().task
        result = store.update_task(task.task_id, timeout_seconds=store.MAX_DEV_TASK_TIMEOUT_SECONDS + 1)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    # --- status / error_message terminal-state invariant ---

    def test_update_task_rejects_completed_with_error_message(self):
        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)
        result = store.update_task(
            task.task_id, status=store.DevTaskStatus.COMPLETED.value, error_message="矛盾したメモ"
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)
        self.assertEqual(store.get_task(task.task_id).task.status, store.DevTaskStatus.RUNNING.value)

    def test_update_task_rejects_failed_without_error_message(self):
        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)
        result = store.update_task(task.task_id, status=store.DevTaskStatus.FAILED.value)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)
        self.assertEqual(store.get_task(task.task_id).task.status, store.DevTaskStatus.RUNNING.value)

    def test_update_task_rejects_failed_with_empty_error_message(self):
        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)
        result = store.update_task(task.task_id, status=store.DevTaskStatus.FAILED.value, error_message="")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    def test_update_task_allows_failed_with_error_message(self):
        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)
        result = store.update_task(
            task.task_id, status=store.DevTaskStatus.FAILED.value, error_message="失敗理由"
        )
        self.assertTrue(result.success)

    def test_get_task_rejects_completed_record_with_error_message(self):
        task_id = str(uuid.uuid4())
        self._write_raw(
            task_id,
            self._valid_payload(task_id, status=store.DevTaskStatus.COMPLETED.value, error_message="矛盾"),
        )
        result = store.get_task(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_CORRUPTED)

    def test_get_task_rejects_failed_record_without_error_message(self):
        task_id = str(uuid.uuid4())
        self._write_raw(
            task_id,
            self._valid_payload(task_id, status=store.DevTaskStatus.FAILED.value, error_message=None),
        )
        result = store.get_task(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_CORRUPTED)

    # --- str Enum normalization ---

    def test_create_task_normalizes_engine_enum_to_plain_str(self):
        result = self._create_task(routed_engine=DevTaskEngine.CODEX)
        self.assertTrue(result.success)
        self.assertIs(type(result.task.routed_engine), str)

    def test_update_task_normalizes_status_enum_to_plain_str(self):
        task = self._create_task().task
        result = store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING)
        self.assertTrue(result.success)
        self.assertIs(type(result.task.status), str)

    def test_task_field_type_matches_immediately_after_create_and_after_reload(self):
        result = self._create_task(routed_engine=DevTaskEngine.CODEX)
        self.assertTrue(result.success)
        reloaded = store.get_task(result.task.task_id).task
        self.assertIs(type(result.task.routed_engine), type(reloaded.routed_engine))
        self.assertEqual(result.task.routed_engine, reloaded.routed_engine)
        self.assertIs(type(result.task.status), type(reloaded.status))


if __name__ == "__main__":
    unittest.main()
