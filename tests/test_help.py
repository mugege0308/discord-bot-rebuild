import unittest

from commands.help import handle_help, split_discord_message


class HelpMessageSplitTestCase(unittest.TestCase):
    def test_short_message_is_unchanged(self):
        content = "短いヘルプ"
        self.assertEqual(split_discord_message(content), [content])

    def test_exact_limit_is_one_chunk(self):
        content = "x" * 2000
        self.assertEqual(split_discord_message(content), [content])

    def test_long_help_splits_at_line_boundaries(self):
        content = "\n".join(f"/command_{index}" for index in range(300))
        chunks = split_discord_message(content)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 2000 for chunk in chunks))
        self.assertEqual("".join(chunks), content)
        self.assertEqual(sum(chunk.count("/command_") for chunk in chunks), 300)

    def test_current_help_content_is_safe_to_send(self):
        content = handle_help("@bot")
        chunks = split_discord_message(content)
        self.assertGreater(len(content), 2000)
        self.assertTrue(all(len(chunk) <= 2000 for chunk in chunks))
        self.assertEqual("".join(chunks), content)

    def test_long_line_uses_safe_boundary_without_losing_content(self):
        content = "command " + ("説明 " * 800)
        chunks = split_discord_message(content)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 2000 for chunk in chunks))
        self.assertEqual("".join(chunks), content)

    def test_invalid_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            split_discord_message("content", 2001)
