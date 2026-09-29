import unittest

from services.image_composition import (
    DEFAULT_COMPOSITION_PRESET,
    DEFAULT_SIZE_KEY,
    build_composition_preset_instruction,
    get_image_size_spec,
    validate_composition_preset,
)


class ImageCompositionTests(unittest.TestCase):
    def test_defaults_and_known_size(self):
        self.assertEqual(DEFAULT_SIZE_KEY, get_image_size_spec(None).key)
        self.assertEqual("portrait", get_image_size_spec("portrait").key)
        self.assertEqual(DEFAULT_COMPOSITION_PRESET, validate_composition_preset(None))

    def test_composition_validation_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            validate_composition_preset("diagonal")

    def test_explicit_request_owns_composition(self):
        instruction = build_composition_preset_instruction("全身で描いて", "upper_body")
        self.assertEqual("", instruction)

    def test_preset_produces_weak_helper_when_request_is_unspecified(self):
        instruction = build_composition_preset_instruction("夕暮れの海辺", "full_body")
        self.assertIn("head to toe", instruction)


if __name__ == "__main__":
    unittest.main()
