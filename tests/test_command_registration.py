import unittest

import main


class CommandRegistrationTestCase(unittest.TestCase):
    def test_restored_commands_are_registered_without_deferred_legacy_commands(self):
        names = {command.name for command in main.client.tree.get_commands()}
        self.assertTrue(
            {
                "character_list",
                "character_info",
                "character_search",
                "character_missing",
                "character_validate",
                "generate_ai_debug",
                "generate_ai_debug_detail",
                "generate_ai_settings_reset",
            }.issubset(names)
        )
        self.assertEqual(len(names), 95)
        self.assertTrue(
            {
                "character_edit",
                "character_register_from_images",
                "character_update",
                "image_improvement_suggestions",
            }.isdisjoint(names)
        )
