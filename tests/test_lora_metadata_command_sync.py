import ast
import unittest
from pathlib import Path
from unittest.mock import patch

from commands import lora


REPO_ROOT = Path(__file__).resolve().parents[1]


class LoraMetadataCommandSyncBoundaryTestCase(unittest.TestCase):
    def test_lora_commands_do_not_sync_the_tree_directly(self):
        source_path = REPO_ROOT / "commands" / "lora.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        sync_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "sync"
        ]
        self.assertEqual(sync_calls, [])
        self.assertIn("ENABLE_LORA_SLASH_COMMAND_CHOICE_REFRESH = False", source_path.read_text(encoding="utf-8"))

    def test_live_choice_refresh_is_disabled_without_importing_generation_wiring(self):
        with patch.object(lora, "ENABLE_LORA_SLASH_COMMAND_CHOICE_REFRESH", False):
            lora._refresh_lora_choice_cache(object())

    def test_main_keeps_sync_as_startup_responsibility(self):
        source = (REPO_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("await self.tree.sync()", source)
        self.assertIn("await self.tree.sync(guild=guild)", source)

    def test_startup_recovery_failure_is_logged_and_does_not_escape_setup_hook(self):
        source = (REPO_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("LoRA job recovery failed; continuing Bot startup.", source)
        self.assertIn("await lora_job.job_manager.recover()", source)


if __name__ == "__main__":
    unittest.main()
