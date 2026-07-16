import unittest
from unittest import mock

from services.dev_task_config import (
    get_dev_task_allowed_user_ids,
    get_dev_task_result_channel_id,
)
from services.dev_task_store import MAX_SNOWFLAKE_ID


def _env(**values: str):
    return mock.patch.dict("os.environ", values, clear=False)


class GetDevTaskAllowedUserIdsTestCase(unittest.TestCase):
    def test_parses_valid_comma_separated_ids(self):
        with _env(DEV_TASK_ALLOWED_USER_IDS="111, 222,333"):
            self.assertEqual(get_dev_task_allowed_user_ids(), {111, 222, 333})

    def test_empty_env_var_yields_empty_set(self):
        with _env(DEV_TASK_ALLOWED_USER_IDS=""):
            self.assertEqual(get_dev_task_allowed_user_ids(), set())

    def test_rejects_zero(self):
        with _env(DEV_TASK_ALLOWED_USER_IDS="0"):
            self.assertEqual(get_dev_task_allowed_user_ids(), set())

    def test_rejects_negative(self):
        with _env(DEV_TASK_ALLOWED_USER_IDS="-5"):
            self.assertEqual(get_dev_task_allowed_user_ids(), set())

    def test_rejects_value_above_max_snowflake(self):
        with _env(DEV_TASK_ALLOWED_USER_IDS=str(MAX_SNOWFLAKE_ID + 1)):
            self.assertEqual(get_dev_task_allowed_user_ids(), set())

    def test_accepts_max_snowflake_boundary(self):
        with _env(DEV_TASK_ALLOWED_USER_IDS=str(MAX_SNOWFLAKE_ID)):
            self.assertEqual(get_dev_task_allowed_user_ids(), {MAX_SNOWFLAKE_ID})

    def test_rejects_non_numeric_entry_but_keeps_valid_ones(self):
        with _env(DEV_TASK_ALLOWED_USER_IDS="111,not-a-number,222"):
            self.assertEqual(get_dev_task_allowed_user_ids(), {111, 222})


class GetDevTaskResultChannelIdTestCase(unittest.TestCase):
    def test_returns_none_when_unset(self):
        with _env(DEV_TASK_RESULT_CHANNEL_ID=""):
            self.assertIsNone(get_dev_task_result_channel_id())

    def test_parses_valid_id(self):
        with _env(DEV_TASK_RESULT_CHANNEL_ID="999"):
            self.assertEqual(get_dev_task_result_channel_id(), 999)

    def test_rejects_zero(self):
        with _env(DEV_TASK_RESULT_CHANNEL_ID="0"):
            self.assertIsNone(get_dev_task_result_channel_id())

    def test_rejects_negative(self):
        with _env(DEV_TASK_RESULT_CHANNEL_ID="-1"):
            self.assertIsNone(get_dev_task_result_channel_id())

    def test_rejects_value_above_max_snowflake(self):
        with _env(DEV_TASK_RESULT_CHANNEL_ID=str(MAX_SNOWFLAKE_ID + 1)):
            self.assertIsNone(get_dev_task_result_channel_id())

    def test_rejects_non_numeric(self):
        with _env(DEV_TASK_RESULT_CHANNEL_ID="not-a-number"):
            self.assertIsNone(get_dev_task_result_channel_id())


if __name__ == "__main__":
    unittest.main()
