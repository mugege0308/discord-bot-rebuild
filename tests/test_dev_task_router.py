import unittest

from services.dev_task_router import (
    LARGE_PROMPT_LENGTH_THRESHOLD,
    DevTaskEngine,
    DevTaskRoutingError,
    route_dev_task,
)


class DevTaskRouterTests(unittest.TestCase):
    # --- 10 designed concrete examples ---

    def test_small_keyword_routes_to_codex(self):
        decision = route_dev_task("ちょっとしたタイポを直して", "auto")
        self.assertEqual(decision.engine, DevTaskEngine.CODEX)
        self.assertFalse(decision.overridden)

    def test_large_keyword_routes_to_claude(self):
        decision = route_dev_task("アーキテクチャを全体的にrefactorしたい", "auto")
        self.assertEqual(decision.engine, DevTaskEngine.CLAUDE)
        self.assertFalse(decision.overridden)

    def test_large_and_small_keywords_both_present_routes_to_claude(self):
        decision = route_dev_task("バグを修正して、ついでに複数ファイルのアーキテクチャも再設計して", "auto")
        self.assertEqual(decision.engine, DevTaskEngine.CLAUDE)
        self.assertIn("両方", decision.reason)

    def test_small_keyword_only_routes_to_codex(self):
        decision = route_dev_task("軽微なbugです", "auto")
        self.assertEqual(decision.engine, DevTaskEngine.CODEX)

    def test_english_small_keyword_uppercase_is_case_insensitive(self):
        decision = route_dev_task("FIX the TYPO please", "auto")
        self.assertEqual(decision.engine, DevTaskEngine.CODEX)

    def test_english_large_keyword_uppercase_is_case_insensitive(self):
        decision = route_dev_task("REWRITE the whole MODULE", "auto")
        self.assertEqual(decision.engine, DevTaskEngine.CLAUDE)

    def test_no_keyword_short_prompt_defaults_to_codex(self):
        decision = route_dev_task("今日の天気を教えて", "auto")
        self.assertEqual(decision.engine, DevTaskEngine.CODEX)
        self.assertIn("デフォルト", decision.reason)

    def test_manual_override_codex(self):
        decision = route_dev_task("何かプロンプト", "codex")
        self.assertEqual(decision.engine, DevTaskEngine.CODEX)
        self.assertTrue(decision.overridden)
        self.assertEqual(decision.reason, "手動指定のため")

    def test_manual_override_claude_wins_over_small_keyword(self):
        decision = route_dev_task("修正して", "claude")
        self.assertEqual(decision.engine, DevTaskEngine.CLAUDE)
        self.assertTrue(decision.overridden)

    def test_engine_option_is_case_insensitive_for_auto(self):
        decision = route_dev_task("migrateして", "AUTO")
        self.assertEqual(decision.engine, DevTaskEngine.CLAUDE)
        self.assertFalse(decision.overridden)

    # --- boundary / edge cases ---

    def test_empty_prompt_defaults_to_codex(self):
        decision = route_dev_task("", "auto")
        self.assertEqual(decision.engine, DevTaskEngine.CODEX)
        self.assertFalse(decision.overridden)

    def test_prompt_length_boundary_200_routes_to_claude(self):
        decision = route_dev_task("a" * LARGE_PROMPT_LENGTH_THRESHOLD, "auto")
        self.assertEqual(decision.engine, DevTaskEngine.CLAUDE)

    def test_prompt_length_boundary_199_routes_to_codex_default(self):
        decision = route_dev_task("a" * (LARGE_PROMPT_LENGTH_THRESHOLD - 1), "auto")
        self.assertEqual(decision.engine, DevTaskEngine.CODEX)

    def test_large_and_small_keyword_conflict_routes_to_claude(self):
        decision = route_dev_task("refactor this typo", "auto")
        self.assertEqual(decision.engine, DevTaskEngine.CLAUDE)

    def test_manual_override_takes_priority_over_auto_keywords(self):
        # "refactor" would normally route to claude on its own, but an explicit
        # codex override must always win regardless of keyword content.
        decision = route_dev_task("refactorして", "codex")
        self.assertEqual(decision.engine, DevTaskEngine.CODEX)
        self.assertTrue(decision.overridden)

    def test_invalid_engine_option_raises(self):
        with self.assertRaises(DevTaskRoutingError):
            route_dev_task("何か", "invalid_engine")

    def test_empty_engine_option_raises(self):
        with self.assertRaises(DevTaskRoutingError):
            route_dev_task("何か", "")

    def test_deterministic_same_input_same_output(self):
        first = route_dev_task("バグを修正して", "auto")
        second = route_dev_task("バグを修正して", "auto")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
