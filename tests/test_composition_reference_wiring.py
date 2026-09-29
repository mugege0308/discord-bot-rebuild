import asyncio
import unittest

from discord import app_commands

from commands import prompt as prompt_commands


class CompositionReferenceWiringTests(unittest.TestCase):
    def tearDown(self) -> None:
        prompt_commands._last_reference_images.clear()

    def test_public_reference_modes_are_none_composition_and_pose(self):
        self.assertEqual(
            [(choice.value, choice.name) for choice in prompt_commands.GENERATE_AI_REFERENCE_MODE_CHOICES],
            [("none", "参照なし"), ("composition", "構図参照"), ("pose", "ポーズ参照")],
        )

    def test_resolver_clears_strength_for_explicit_none(self):
        settings = prompt_commands._resolve_generate_ai_settings(
            1,
            reference_mode=app_commands.Choice(name="参照なし", value="none"),
            reference_strength=0.9,
            inherit_previous=False,
        )

        self.assertEqual(settings.reference_mode, "none")
        self.assertIsNone(settings.reference_strength)

    def test_resolver_accepts_composition_and_strength(self):
        settings = prompt_commands._resolve_generate_ai_settings(
            1,
            reference_mode=app_commands.Choice(name="構図参照", value="composition"),
            reference_strength=0.8,
            inherit_previous=False,
        )

        self.assertEqual(settings.reference_mode, "composition")
        self.assertEqual(settings.reference_strength, 0.8)

    def test_panel_forwards_composition_reuse_and_strength(self):
        captured = []

        async def callback(interaction, request, **kwargs):
            captured.append((interaction, request, kwargs))

        previous_callback = prompt_commands._generate_ai_panel_callback
        prompt_commands._generate_ai_panel_callback = callback
        try:
            asyncio.run(
                prompt_commands.run_generate_ai_from_panel(
                    "interaction",
                    request="構図参照",
                    model_key="animagine",
                    input_mode="ja_ai",
                    rating="auto",
                    reference_mode="composition",
                    use_last_reference=True,
                    reference_strength=0.8,
                    selection_ai=True,
                    candidate_count=1,
                    size_key="square",
                    composition_preset="none",
                )
            )
        finally:
            prompt_commands._generate_ai_panel_callback = previous_callback

        payload = captured[0][2]
        self.assertEqual(payload["reference_mode"].value, "composition")
        self.assertTrue(payload["use_last_reference"])
        self.assertEqual(payload["reference_strength"], 0.8)

    def test_last_reference_info_is_available_without_exposing_bytes(self):
        prompt_commands._last_reference_images[7] = (b"reference", "image/png")

        self.assertTrue(prompt_commands.has_last_reference_image(7))
        info = prompt_commands.get_last_reference_image_info(7)
        self.assertIsNotNone(info)
        self.assertEqual(info["content_type"], "image/png")
        self.assertNotIn("bytes", info)
        self.assertFalse(prompt_commands.has_last_reference_image(8))


if __name__ == "__main__":
    unittest.main()
