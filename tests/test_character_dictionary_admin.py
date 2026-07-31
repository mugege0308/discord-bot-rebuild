import unittest
from unittest import mock

from services.character_dictionary_admin import (
    get_character_admin_allowed_user_ids,
    is_character_admin_allowed,
)


class CharacterAdminAllowedUserIdsTestCase(unittest.TestCase):
    def test_empty_env_var_yields_no_allowed_users(self):
        with mock.patch.dict("os.environ", {"CHARACTER_ADMIN_ALLOWED_USER_IDS": ""}, clear=False):
            self.assertEqual(get_character_admin_allowed_user_ids(), set())

    def test_comma_separated_ids_are_parsed(self):
        with mock.patch.dict("os.environ", {"CHARACTER_ADMIN_ALLOWED_USER_IDS": "111, 222,333"}, clear=False):
            self.assertEqual(get_character_admin_allowed_user_ids(), {111, 222, 333})

    def test_invalid_entry_is_skipped_not_raised(self):
        with mock.patch.dict("os.environ", {"CHARACTER_ADMIN_ALLOWED_USER_IDS": "111,not-a-number,222"}, clear=False):
            self.assertEqual(get_character_admin_allowed_user_ids(), {111, 222})

    def test_is_character_admin_allowed_true_for_listed_user(self):
        with mock.patch.dict("os.environ", {"CHARACTER_ADMIN_ALLOWED_USER_IDS": "111"}, clear=False):
            self.assertTrue(is_character_admin_allowed(111))
            self.assertFalse(is_character_admin_allowed(999))


if __name__ == "__main__":
    unittest.main()
