import unittest

from services.prompt_context_conflicts import analyze_explicit_visual_constraints


class PromptContextConflictBaseTests(unittest.TestCase):
    def test_explicit_composition_is_request_owned(self):
        result = analyze_explicit_visual_constraints("full body, looking at viewer")
        self.assertTrue(result.controls_composition)

    def test_unrelated_request_has_no_composition_owner(self):
        result = analyze_explicit_visual_constraints("夕暮れの海辺")
        self.assertFalse(result.controls_composition)


if __name__ == "__main__":
    unittest.main()
