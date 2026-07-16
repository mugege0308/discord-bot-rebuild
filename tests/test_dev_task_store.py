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
            base_branch="master",
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
            "base_branch": "master",
            "error_message": None,
            "branch_name": None,
            "approved_at": None,
            "started_at": None,
            "finished_at": None,
            "decision": None,
            "decided_at": None,
            "notification_method": None,
            "result_summary": None,
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

    # --- Phase 3: base_branch (create field, immutable) ---

    def test_create_task_stores_base_branch(self):
        result = self._create_task(base_branch="feature/x")
        self.assertTrue(result.success)
        self.assertEqual(result.task.base_branch, "feature/x")

    def test_create_task_rejects_empty_base_branch(self):
        result = self._create_task(base_branch="")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    def test_create_task_rejects_whitespace_only_base_branch(self):
        result = self._create_task(base_branch="   ")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    def test_update_task_rejects_base_branch_change(self):
        task = self._create_task(base_branch="master").task
        result = store.update_task(task.task_id, base_branch="other")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_IMMUTABLE_FIELD)
        self.assertEqual(store.get_task(task.task_id).task.base_branch, "master")

    # --- Phase 3 (round 2): audit fields are no longer generic update_task()
    # targets at all - only the dedicated CAS functions below may touch them.

    def test_update_task_rejects_branch_name_as_unknown_field(self):
        task = self._create_task().task
        result = store.update_task(task.task_id, branch_name="dev-task/xyz")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_UNKNOWN_FIELD)

    def test_update_task_rejects_approved_at_as_unknown_field(self):
        task = self._create_task().task
        result = store.update_task(task.task_id, approved_at="2026-07-15T00:00:00.000000+00:00")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_UNKNOWN_FIELD)

    def test_update_task_rejects_finished_at_as_unknown_field(self):
        task = self._create_task().task
        result = store.update_task(task.task_id, finished_at="2026-07-15T00:00:00.000000+00:00")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_UNKNOWN_FIELD)

    def test_update_task_rejects_notification_method_as_unknown_field(self):
        task = self._create_task().task
        result = store.update_task(task.task_id, notification_method="dm")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_UNKNOWN_FIELD)

    def test_update_task_rejects_decision_as_unknown_field(self):
        task = self._create_task().task
        result = store.update_task(task.task_id, decision="kept", decided_at="2026-07-15T00:00:00.000000+00:00")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_UNKNOWN_FIELD)

    # --- claim_task_for_execution: pending_approval -> running, atomically ---

    def test_claim_task_for_execution_succeeds_from_pending_approval(self):
        task = self._create_task().task
        result = store.claim_task_for_execution(task.task_id, branch_name="dev-task/abc")
        self.assertTrue(result.success)
        self.assertEqual(result.task.status, store.DevTaskStatus.RUNNING.value)
        self.assertEqual(result.task.branch_name, "dev-task/abc")
        self.assertIsNotNone(result.task.approved_at)
        self.assertIsNotNone(result.task.started_at)

    def test_claim_task_for_execution_applies_engine_override(self):
        task = self._create_task(routed_engine="codex").task
        result = store.claim_task_for_execution(
            task.task_id, branch_name="dev-task/abc", engine_override="claude"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.task.routed_engine, "claude")
        self.assertTrue(result.task.engine_overridden)

    def test_claim_task_for_execution_rejects_invalid_engine_override(self):
        task = self._create_task().task
        result = store.claim_task_for_execution(
            task.task_id, branch_name="dev-task/abc", engine_override="gpt"
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)
        self.assertEqual(store.get_task(task.task_id).task.status, store.DevTaskStatus.PENDING_APPROVAL.value)

    def test_claim_task_for_execution_rejects_when_not_pending_approval(self):
        task = self._create_task().task
        store.claim_task_for_execution(task.task_id, branch_name="dev-task/abc")

        result = store.claim_task_for_execution(task.task_id, branch_name="dev-task/abc-again")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_TRANSITION)

    def test_claim_task_for_execution_is_all_or_nothing_race_safe(self):
        # Simulates two concurrent approvals of the SAME task_id: only the
        # first claim may succeed; the second must be rejected outright,
        # without touching anything the first one wrote.
        task = self._create_task().task
        first = store.claim_task_for_execution(task.task_id, branch_name="dev-task/first")
        second = store.claim_task_for_execution(task.task_id, branch_name="dev-task/second")

        self.assertTrue(first.success)
        self.assertFalse(second.success)
        self.assertEqual(second.error_code, store.ERROR_INVALID_TRANSITION)
        self.assertEqual(store.get_task(task.task_id).task.branch_name, "dev-task/first")

    def test_claim_task_for_execution_rejects_empty_branch_name(self):
        task = self._create_task().task
        result = store.claim_task_for_execution(task.task_id, branch_name="")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    def test_claim_task_for_execution_rejects_invalid_task_id(self):
        result = store.claim_task_for_execution("not-a-uuid", branch_name="dev-task/abc")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_TASK_ID)

    # --- fail_pending_approval: timeout/cancel/pre-claim rejection ---

    def test_fail_pending_approval_succeeds_from_pending_approval(self):
        task = self._create_task().task
        result = store.fail_pending_approval(task.task_id, reason="承認が時間内に行われませんでした。")
        self.assertTrue(result.success)
        self.assertEqual(result.task.status, store.DevTaskStatus.FAILED.value)
        self.assertEqual(result.task.error_message, "承認が時間内に行われませんでした。")
        self.assertIsNotNone(result.task.finished_at)

    def test_fail_pending_approval_never_fails_an_already_claimed_task(self):
        # This is the exact race Codex flagged: on_timeout()/cancel firing
        # after a concurrent approval has already claimed (and started
        # running) the same task must never flip it to failed.
        task = self._create_task().task
        claim_result = store.claim_task_for_execution(task.task_id, branch_name="dev-task/abc")
        self.assertTrue(claim_result.success)

        result = store.fail_pending_approval(task.task_id, reason="遅すぎたタイムアウト処理")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_TRANSITION)
        self.assertEqual(store.get_task(task.task_id).task.status, store.DevTaskStatus.RUNNING.value)

    def test_fail_pending_approval_rejects_empty_reason(self):
        task = self._create_task().task
        result = store.fail_pending_approval(task.task_id, reason="")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    # --- finish_task: running -> completed/failed, atomically ---

    def test_finish_task_success_sets_completed_and_result_summary(self):
        task = self._create_task().task
        store.claim_task_for_execution(task.task_id, branch_name="dev-task/abc")

        result = store.finish_task(task.task_id, success=True, result_summary="0 files changed")
        self.assertTrue(result.success)
        self.assertEqual(result.task.status, store.DevTaskStatus.COMPLETED.value)
        self.assertEqual(result.task.result_summary, "0 files changed")
        self.assertIsNotNone(result.task.finished_at)
        self.assertIsNone(result.task.error_message)

    def test_finish_task_failure_sets_failed_and_error_message(self):
        task = self._create_task().task
        store.claim_task_for_execution(task.task_id, branch_name="dev-task/abc")

        result = store.finish_task(task.task_id, success=False, error_message="ダミー実行に失敗しました。")
        self.assertTrue(result.success)
        self.assertEqual(result.task.status, store.DevTaskStatus.FAILED.value)
        self.assertEqual(result.task.error_message, "ダミー実行に失敗しました。")
        self.assertIsNone(result.task.result_summary)

    def test_finish_task_rejects_when_not_running(self):
        task = self._create_task().task  # still pending_approval
        result = store.finish_task(task.task_id, success=True)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_TRANSITION)

    def test_finish_task_rejects_success_with_error_message(self):
        task = self._create_task().task
        store.claim_task_for_execution(task.task_id, branch_name="dev-task/abc")
        result = store.finish_task(task.task_id, success=True, error_message="matters not")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    def test_finish_task_rejects_failure_without_error_message(self):
        task = self._create_task().task
        store.claim_task_for_execution(task.task_id, branch_name="dev-task/abc")
        result = store.finish_task(task.task_id, success=False)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    def test_finish_task_cannot_double_finish(self):
        task = self._create_task().task
        store.claim_task_for_execution(task.task_id, branch_name="dev-task/abc")
        first = store.finish_task(task.task_id, success=True, result_summary="ok")
        self.assertTrue(first.success)

        second = store.finish_task(task.task_id, success=False, error_message="too late")
        self.assertFalse(second.success)
        self.assertEqual(second.error_code, store.ERROR_INVALID_TRANSITION)
        self.assertEqual(store.get_task(task.task_id).task.status, store.DevTaskStatus.COMPLETED.value)

    # --- set_notification_method_once: decoupled from finish_task() ---

    def test_set_notification_method_once_succeeds_after_finish(self):
        task = self._create_task().task
        store.claim_task_for_execution(task.task_id, branch_name="dev-task/abc")
        store.finish_task(task.task_id, success=True, result_summary="ok")

        result = store.set_notification_method_once(task.task_id, method="dm")
        self.assertTrue(result.success)
        self.assertEqual(result.task.notification_method, "dm")

    def test_set_notification_method_once_rejects_second_call(self):
        task = self._create_task().task
        store.claim_task_for_execution(task.task_id, branch_name="dev-task/abc")
        store.finish_task(task.task_id, success=True, result_summary="ok")
        store.set_notification_method_once(task.task_id, method="dm")

        result = store.set_notification_method_once(task.task_id, method="channel")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_FIELD_LOCKED)
        self.assertEqual(store.get_task(task.task_id).task.notification_method, "dm")

    def test_set_notification_method_once_rejects_invalid_value(self):
        task = self._create_task().task
        store.claim_task_for_execution(task.task_id, branch_name="dev-task/abc")
        store.finish_task(task.task_id, success=True, result_summary="ok")

        result = store.set_notification_method_once(task.task_id, method="carrier_pigeon")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    def test_finish_task_does_not_set_notification_method(self):
        # finish_task() must never itself claim a delivery method - that is
        # only known after the notification is actually attempted.
        task = self._create_task().task
        store.claim_task_for_execution(task.task_id, branch_name="dev-task/abc")
        result = store.finish_task(task.task_id, success=True, result_summary="ok")
        self.assertIsNone(result.task.notification_method)

    # --- finalize_decision: terminal-only, once ---

    def test_finalize_decision_succeeds_when_completed(self):
        task = self._create_task().task
        store.claim_task_for_execution(task.task_id, branch_name="dev-task/abc")
        store.finish_task(task.task_id, success=True, result_summary="ok")

        result = store.finalize_decision(task.task_id, decision="kept")
        self.assertTrue(result.success)
        self.assertEqual(result.task.decision, "kept")
        self.assertIsNotNone(result.task.decided_at)

    def test_finalize_decision_succeeds_when_failed(self):
        task = self._create_task().task
        store.claim_task_for_execution(task.task_id, branch_name="dev-task/abc")
        store.finish_task(task.task_id, success=False, error_message="失敗理由")

        result = store.finalize_decision(task.task_id, decision="discarded")
        self.assertTrue(result.success)

    def test_finalize_decision_rejects_while_pending_approval(self):
        task = self._create_task().task
        result = store.finalize_decision(task.task_id, decision="kept")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_FIELD_LOCKED)

    def test_finalize_decision_rejects_while_running(self):
        task = self._create_task().task
        store.claim_task_for_execution(task.task_id, branch_name="dev-task/abc")
        result = store.finalize_decision(task.task_id, decision="kept")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_FIELD_LOCKED)

    def test_finalize_decision_rejects_invalid_value(self):
        task = self._create_task().task
        store.claim_task_for_execution(task.task_id, branch_name="dev-task/abc")
        store.finish_task(task.task_id, success=True, result_summary="ok")

        result = store.finalize_decision(task.task_id, decision="deleted_forever")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_INVALID_VALUE)

    def test_finalize_decision_rejects_second_call(self):
        task = self._create_task().task
        store.claim_task_for_execution(task.task_id, branch_name="dev-task/abc")
        store.finish_task(task.task_id, success=True, result_summary="ok")
        store.finalize_decision(task.task_id, decision="kept")

        result = store.finalize_decision(task.task_id, decision="discarded")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_DECISION_ALREADY_MADE)
        self.assertEqual(store.get_task(task.task_id).task.decision, "kept")

    def test_finalize_decision_concurrent_keep_and_discard_only_one_wins(self):
        task = self._create_task().task
        store.claim_task_for_execution(task.task_id, branch_name="dev-task/abc")
        store.finish_task(task.task_id, success=True, result_summary="ok")

        keep_result = store.finalize_decision(task.task_id, decision="kept")
        discard_result = store.finalize_decision(task.task_id, decision="discarded")

        self.assertTrue(keep_result.success)
        self.assertFalse(discard_result.success)
        self.assertEqual(store.get_task(task.task_id).task.decision, "kept")

    def test_get_task_rejects_stored_decision_without_decided_at(self):
        task_id = str(uuid.uuid4())
        self._write_raw(
            task_id,
            self._valid_payload(
                task_id, status=store.DevTaskStatus.COMPLETED.value, decision="kept", decided_at=None
            ),
        )
        result = store.get_task(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_CORRUPTED)

    def test_get_task_rejects_stored_decided_at_without_decision(self):
        task_id = str(uuid.uuid4())
        self._write_raw(
            task_id,
            self._valid_payload(
                task_id,
                status=store.DevTaskStatus.COMPLETED.value,
                decision=None,
                decided_at="2026-07-15T01:00:00.000000+00:00",
            ),
        )
        result = store.get_task(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_CORRUPTED)

    def test_get_task_rejects_invalid_notification_method_in_record(self):
        task_id = str(uuid.uuid4())
        self._write_raw(task_id, self._valid_payload(task_id, notification_method="carrier_pigeon"))
        result = store.get_task(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_CORRUPTED)

    def test_get_task_accepts_completed_record_without_result_summary(self):
        # result_summary is optional even for a completed record (e.g. an
        # older record from before this field existed) - only forbidden
        # outright on non-completed statuses.
        task_id = str(uuid.uuid4())
        self._write_raw(
            task_id,
            self._valid_payload(task_id, status=store.DevTaskStatus.COMPLETED.value, result_summary=None),
        )
        result = store.get_task(task_id)
        self.assertTrue(result.success)

    def test_get_task_rejects_pending_record_with_result_summary(self):
        task_id = str(uuid.uuid4())
        self._write_raw(
            task_id,
            self._valid_payload(
                task_id,
                status=store.DevTaskStatus.PENDING_APPROVAL.value,
                result_summary="not allowed yet",
            ),
        )
        result = store.get_task(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_CORRUPTED)

    def test_get_task_rejects_missing_base_branch_field(self):
        task_id = str(uuid.uuid4())
        payload = self._valid_payload(task_id)
        del payload["base_branch"]
        self._write_raw(task_id, payload)
        result = store.get_task(task_id)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, store.ERROR_CORRUPTED)

    # --- Phase 3: startup recovery also records finished_at ---

    def test_mark_running_failed_on_startup_sets_finished_at(self):
        task = self._create_task().task
        store.update_task(task.task_id, status=store.DevTaskStatus.RUNNING.value)

        store.mark_running_tasks_failed_on_startup()

        after = store.get_task(task.task_id).task
        self.assertEqual(after.status, store.DevTaskStatus.FAILED.value)
        self.assertIsNotNone(after.finished_at)

    # --- Phase 3: full happy-path lifecycle through the new fields ---

    def test_full_lifecycle_with_phase3_fields(self):
        task = self._create_task(base_branch="master").task

        claim_result = store.claim_task_for_execution(
            task.task_id, branch_name=f"dev-task/{task.task_id}"
        )
        self.assertTrue(claim_result.success)

        finish_result = store.finish_task(task.task_id, success=True, result_summary="0 files changed")
        self.assertTrue(finish_result.success)

        notify_result = store.set_notification_method_once(task.task_id, method="dm")
        self.assertTrue(notify_result.success)

        decide_result = store.finalize_decision(task.task_id, decision="discarded")
        self.assertTrue(decide_result.success)

        final = store.get_task(task.task_id).task
        self.assertEqual(final.status, store.DevTaskStatus.COMPLETED.value)
        self.assertEqual(final.branch_name, f"dev-task/{task.task_id}")
        self.assertEqual(final.result_summary, "0 files changed")
        self.assertEqual(final.decision, "discarded")
        self.assertEqual(final.notification_method, "dm")


if __name__ == "__main__":
    unittest.main()
