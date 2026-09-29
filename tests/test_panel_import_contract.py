import unittest

from commands import prompt as prompt_commands
from views.image_generation import ImageGenerationPanelState


class PanelImportContractTests(unittest.TestCase):
    def test_panel_module_import_contract_defaults(self):
        self.assertEqual(prompt_commands.REFERENCE_MODE_NONE, "none")
        self.assertTrue(prompt_commands.GENERATE_AI_DEFAULT_SELECTION_AI)
        self.assertEqual(prompt_commands.DEFAULT_SIZE_KEY, "square")
        self.assertEqual(prompt_commands.DEFAULT_COMPOSITION_PRESET, "none")

    def test_panel_state_constructs_with_expected_defaults(self):
        state = ImageGenerationPanelState(user_id=1)
        self.assertEqual(state.input_mode, "ja_ai")
        self.assertEqual(state.rating, "auto")
        self.assertEqual(state.reference_mode, "none")
        self.assertTrue(state.selection_ai)
        self.assertEqual(state.size_key, "square")
        self.assertEqual(state.composition_preset, "none")


if __name__ == "__main__":
    unittest.main()
