import unittest
from unittest import mock

from discord import app_commands

import commands.prompt as prompt_module
from services.shogou_prompt_designer import ShogouPromptResult


class FakeCommandTree:
    def __init__(self) -> None:
        self.commands: dict[str, object] = {}

    def command(self, *, name: str, description: str):
        del description

        def decorator(callback):
            self.commands[name] = callback
            return callback

        return decorator


class FakeClient:
    def __init__(self) -> None:
        self.tree = FakeCommandTree()


class GenerateAiLastReferenceTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        prompt_module._last_reference_images.clear()
        client = FakeClient()
        prompt_module.setup(client)
        # use_last_reference is intentionally not a public Slash option. Test
        # the internal callback used by the Panel instead of the public wrapper.
        self.command = prompt_module._generate_ai_panel_callback

    def _interaction(self, user_id: int):
        interaction = mock.MagicMock()
        interaction.user.id = user_id
        interaction.response.defer = mock.AsyncMock()
        interaction.followup.send = mock.AsyncMock()
        return interaction

    def _attachment(self, image_bytes: bytes, content_type: str = "image/png"):
        attachment = mock.MagicMock()
        attachment.size = len(image_bytes)
        attachment.content_type = content_type
        attachment.read = mock.AsyncMock(return_value=image_bytes)
        return attachment

    async def _invoke(self, interaction, **kwargs):
        result = ShogouPromptResult(positive="positive", negative="negative", summary="summary")
        with mock.patch("commands.prompt.is_pc_control_allowed", return_value=True), mock.patch(
            "commands.prompt.is_comfy_backend", return_value=True
        ), mock.patch(
            "commands.prompt.detect_image_content_type", return_value="image/png"
        ), mock.patch(
            "commands.prompt.validate_image_bytes"
        ), mock.patch(
            "commands.prompt._build_shogou_prompt_result", new=mock.AsyncMock(return_value=result)
        ), mock.patch(
            "commands.prompt._safe_followup_send", new=mock.AsyncMock()
        ) as send_mock, mock.patch(
            "commands.prompt._generate_images_from_shogou_result", new=mock.AsyncMock()
        ) as generate_mock:
            await self.command(interaction, "request", **kwargs)
        return generate_mock, send_mock

    async def test_explicit_reference_wins_and_updates_the_last_reference(self):
        prompt_module._last_reference_images[101] = (b"old", "image/jpeg")
        attachment = self._attachment(b"new")

        generate_mock, _ = await self._invoke(
            self._interaction(101),
            reference_image=attachment,
            use_last_reference=True,
        )

        self.assertEqual(generate_mock.await_args.kwargs["reference_image_bytes"], b"new")
        self.assertEqual(generate_mock.await_args.kwargs["reference_image_content_type"], "image/png")
        self.assertEqual(prompt_module._last_reference_images[101], (b"new", "image/png"))

    async def test_last_reference_is_reused_only_for_the_same_user(self):
        await self._invoke(self._interaction(101), reference_image=self._attachment(b"user-101"))
        await self._invoke(self._interaction(202), reference_image=self._attachment(b"user-202"))

        user_101_generate, _ = await self._invoke(
            self._interaction(101), use_last_reference=True, reference_strength=0.8
        )
        user_202_generate, _ = await self._invoke(
            self._interaction(202), use_last_reference=True
        )

        self.assertEqual(user_101_generate.await_args.kwargs["reference_image_bytes"], b"user-101")
        self.assertEqual(user_101_generate.await_args.kwargs["reference_strength"], 0.8)
        self.assertEqual(user_202_generate.await_args.kwargs["reference_image_bytes"], b"user-202")

    async def test_omitting_both_reference_options_keeps_reference_disabled(self):
        prompt_module._last_reference_images[101] = (b"cached", "image/png")

        generate_mock, _ = await self._invoke(self._interaction(101))

        self.assertIsNone(generate_mock.await_args.kwargs["reference_image_bytes"])
        self.assertIsNone(generate_mock.await_args.kwargs["reference_image_content_type"])

    async def test_explicit_none_clears_attachment_reuse_and_strength(self):
        prompt_module._last_reference_images[101] = (b"cached", "image/png")

        generate_mock, _ = await self._invoke(
            self._interaction(101),
            reference_mode=app_commands.Choice(name="参照なし", value="none"),
            reference_image=self._attachment(b"ignored"),
            use_last_reference=True,
            reference_strength=0.9,
        )

        self.assertIsNone(generate_mock.await_args.kwargs["reference_image_bytes"])
        self.assertIsNone(generate_mock.await_args.kwargs["reference_image_content_type"])
        self.assertIsNone(generate_mock.await_args.kwargs["reference_strength"])

    async def test_explicit_pose_routes_image_to_pose_fields(self):
        generate_mock, _ = await self._invoke(
            self._interaction(101),
            reference_mode=app_commands.Choice(name="ポーズ参照", value="pose"),
            reference_image=self._attachment(b"pose"),
            reference_strength=0.7,
        )

        kwargs = generate_mock.await_args.kwargs
        self.assertEqual(kwargs["reference_mode"], "pose")
        self.assertIsNone(kwargs["reference_image_bytes"])
        self.assertIsNone(kwargs["reference_image_content_type"])
        self.assertEqual(kwargs["pose_reference_bytes"], b"pose")
        self.assertEqual(kwargs["pose_reference_content_type"], "image/png")
        self.assertEqual(kwargs["pose_strength"], 0.7)

    async def test_pose_reuses_the_cached_image_as_pose_input(self):
        prompt_module._last_reference_images[101] = (b"cached-pose", "image/jpeg")

        generate_mock, _ = await self._invoke(
            self._interaction(101),
            reference_mode=app_commands.Choice(name="ポーズ参照", value="pose"),
            use_last_reference=True,
            reference_strength=0.65,
        )

        kwargs = generate_mock.await_args.kwargs
        self.assertEqual(kwargs["reference_mode"], "pose")
        self.assertIsNone(kwargs["reference_image_bytes"])
        self.assertEqual(kwargs["pose_reference_bytes"], b"cached-pose")
        self.assertEqual(kwargs["pose_reference_content_type"], "image/jpeg")
        self.assertEqual(kwargs["pose_strength"], 0.65)

    async def test_reuse_without_a_previous_reference_reports_an_error(self):
        generate_mock, send_mock = await self._invoke(
            self._interaction(101), use_last_reference=True
        )

        generate_mock.assert_not_awaited()
        self.assertEqual(
            send_mock.await_args.kwargs["content"],
            prompt_module.LAST_REFERENCE_MISSING_MESSAGE,
        )


if __name__ == "__main__":
    unittest.main()
