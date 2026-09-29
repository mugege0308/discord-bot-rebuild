import asyncio
import unittest
from unittest import mock

from discord import app_commands

import commands.prompt as prompt_module
import commands.references as references_module
from services.shogou_prompt_designer import ShogouPromptResult
from views.image_generation import ImageGenerationPanelState, ReferencePanel


class _FakeTree:
    def __init__(self):
        self.commands = {}

    def command(self, *, name, description):
        del description

        def decorator(callback):
            self.commands[name] = callback
            return callback

        return decorator


class _FakeClient:
    def __init__(self):
        self.tree = _FakeTree()


class SavedReferenceWiringTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        client = _FakeClient()
        prompt_module.setup(client)
        self.command = prompt_module._generate_ai_panel_callback

    def _interaction(self, user_id=10):
        interaction = mock.MagicMock()
        interaction.user.id = user_id
        interaction.response.defer = mock.AsyncMock()
        interaction.followup.send = mock.AsyncMock()
        return interaction

    async def test_panel_bridge_keeps_saved_reference_id_separate_from_memory_id(self):
        captured = []

        async def callback(interaction, request, **kwargs):
            captured.append(kwargs)

        old = prompt_module._generate_ai_panel_callback
        prompt_module._generate_ai_panel_callback = callback
        try:
            await prompt_module.run_generate_ai_from_panel(
                self._interaction(),
                request="test",
                reference_mode="composition",
                saved_reference_id=17,
            )
        finally:
            prompt_module._generate_ai_panel_callback = old
        self.assertEqual(captured[0]["saved_reference_id"], 17)
        self.assertIsNone(captured[0]["reference_id"])

    async def test_saved_reference_routes_to_composition_without_changing_memory_id(self):
        result = ShogouPromptResult(positive="positive", negative="negative", summary="summary")
        item = mock.Mock(kind="reference")
        interaction = self._interaction()
        with (
            mock.patch("commands.prompt.is_pc_control_allowed", return_value=True),
            mock.patch("commands.prompt.is_comfy_backend", return_value=True),
            mock.patch("commands.prompt.get_reference", return_value=item),
            mock.patch("commands.prompt.read_reference_bytes", return_value=b"image"),
            mock.patch("commands.prompt.detect_image_content_type", return_value="image/png"),
            mock.patch("commands.prompt.validate_image_bytes"),
            mock.patch("commands.prompt._build_shogou_prompt_result", new=mock.AsyncMock(return_value=result)),
            mock.patch("commands.prompt._safe_followup_send", new=mock.AsyncMock()),
            mock.patch("commands.prompt._generate_images_from_shogou_result", new=mock.AsyncMock()) as generate,
        ):
            await self.command(
                interaction,
                "request",
                reference_mode=app_commands.Choice(name="構図参照", value="composition"),
                saved_reference_id=17,
                reference_id=99,
            )
        kwargs = generate.await_args.kwargs
        self.assertEqual(kwargs["reference_image_bytes"], b"image")
        self.assertEqual(kwargs["reference_mode"], "composition")
        self.assertEqual(kwargs["reference_id"], 99)

    def test_panel_reference_state_defaults_to_none_and_has_saved_picker(self):
        state = ImageGenerationPanelState.from_user(10, inherit=False)
        self.assertIsNone(state.saved_reference_id)
        panel = ReferencePanel(10, "@bot", state)
        labels = {getattr(child, "label", None) for child in panel.children}
        self.assertIn("保存済みを選ぶ", labels)

    def test_reference_commands_register_without_using_image_memory_reference_id(self):
        client = _FakeClient()
        references_module.setup(client)
        self.assertEqual(
            {"ref_save", "ref_list", "ref_delete", "ref_random"},
            set(client.tree.commands),
        )
        self.assertTrue(all(choice.value in references_module.REFERENCE_KINDS for choice in references_module.REF_KIND_CHOICES))

    async def test_saved_reference_requires_non_none_mode(self):
        interaction = self._interaction()
        with (
            mock.patch("commands.prompt.is_pc_control_allowed", return_value=True),
            mock.patch("commands.prompt._send_error", new=mock.AsyncMock()) as send_error,
        ):
            await self.command(interaction, "request", saved_reference_id=17)
        self.assertEqual(send_error.await_args.kwargs["label"], "generate_ai.saved_reference_mode_none")


if __name__ == "__main__":
    unittest.main()
