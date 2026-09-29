import unittest
from unittest import mock

import discord
from discord import app_commands

from commands import gallery
from services.gallery_control import (
    GalleryReviewResult,
    GalleryStartResult,
    GalleryStatus,
    GalleryURLs,
    GalleryControlError,
)


def _status(*, running: bool) -> GalleryStatus:
    return GalleryStatus(
        running=running,
        pid=4321 if running else None,
        port=8765,
        urls=GalleryURLs(
            local="http://127.0.0.1:8765/",
            tailscale=("http://100.64.0.10:8765/",) if running else (),
        ),
    )


class GalleryCommandRegistrationTests(unittest.TestCase):
    def test_setup_registers_expected_subcommands_once(self):
        client = discord.Client(intents=discord.Intents.none())
        client.tree = app_commands.CommandTree(client)

        gallery.setup(client)
        group = client.tree.get_command("gallery")

        self.assertIsNotNone(group)
        self.assertEqual(
            {command.name for command in group.commands},
            {"open", "status", "stop", "review"},
        )

    def test_main_registers_gallery_and_requires_it_for_sync(self):
        import main

        self.assertIn("gallery", main.REQUIRED_SLASH_COMMANDS)
        self.assertIsNotNone(main.client.tree.get_command("gallery"))


class GalleryCommandCallbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        client = discord.Client(intents=discord.Intents.none())
        client.tree = app_commands.CommandTree(client)
        gallery.setup(client)
        self.group = client.tree.get_command("gallery")

    @staticmethod
    def _interaction(user_id: int = 1):
        interaction = mock.MagicMock()
        interaction.user.id = user_id
        interaction.response.defer = mock.AsyncMock()
        interaction.response.send_message = mock.AsyncMock()
        interaction.followup.send = mock.AsyncMock()
        return interaction

    def _command(self, name: str):
        command = self.group.get_command(name)
        self.assertIsNotNone(command)
        return command

    async def test_denied_user_gets_ephemeral_response_without_control_call(self):
        interaction = self._interaction(999)
        command = self._command("open")
        with mock.patch.object(gallery, "is_pc_control_allowed", return_value=False), mock.patch.object(
            gallery, "start_gallery"
        ) as start:
            await command.callback(interaction)

        interaction.response.send_message.assert_awaited_once_with(gallery.DENIED_MESSAGE, ephemeral=True)
        interaction.response.defer.assert_not_awaited()
        start.assert_not_called()

    async def test_open_starts_gallery_and_returns_link_view(self):
        interaction = self._interaction()
        command = self._command("open")
        result = GalleryStartResult(_status(running=True), already_running=False)
        with mock.patch.object(gallery, "is_pc_control_allowed", return_value=True), mock.patch.object(
            gallery, "start_gallery", return_value=result
        ) as start:
            await command.callback(interaction)

        start.assert_called_once_with()
        interaction.response.defer.assert_awaited_once_with(ephemeral=True)
        sent = interaction.followup.send.await_args
        self.assertTrue(sent.kwargs["ephemeral"])
        self.assertIsNotNone(sent.kwargs["view"])
        self.assertIn("起動しました", sent.args[0])

    async def test_status_reports_stopped_gallery(self):
        interaction = self._interaction()
        command = self._command("status")
        with mock.patch.object(gallery, "is_pc_control_allowed", return_value=True), mock.patch.object(
            gallery, "get_gallery_status", return_value=_status(running=False)
        ) as status:
            await command.callback(interaction)

        status.assert_called_once_with()
        sent = interaction.followup.send.await_args
        self.assertIn("停止中", sent.args[0])
        self.assertTrue(sent.kwargs["ephemeral"])

    async def test_stop_failure_is_reported_as_short_gallery_error(self):
        interaction = self._interaction()
        command = self._command("stop")
        with mock.patch.object(gallery, "is_pc_control_allowed", return_value=True), mock.patch.object(
            gallery, "stop_gallery", side_effect=GalleryControlError("stop failed")
        ):
            await command.callback(interaction)

        interaction.followup.send.assert_awaited_once()
        self.assertIn("停止できませんでした", interaction.followup.send.await_args.args[0])
        self.assertIn("stop failed", interaction.followup.send.await_args.args[0])

    async def test_review_uses_existing_gallery_review_api_and_reports_completed_job(self):
        interaction = self._interaction()
        command = self._command("review")
        result = GalleryReviewResult(
            status=_status(running=True),
            job={"job_id": "review-1", "status": "completed", "total": 3},
        )
        with mock.patch.object(gallery, "is_pc_control_allowed", return_value=True), mock.patch.object(
            gallery, "start_gallery_review", return_value=result
        ) as review:
            await command.callback(interaction, reference_name="  Hu Tao  ")

        review.assert_called_once_with(reference_name="Hu Tao")
        sent = interaction.followup.send.await_args
        self.assertIn("完了しました", sent.args[0])
        self.assertIsNotNone(sent.kwargs["view"])

    async def test_open_unexpected_failure_keeps_control_error_user_safe(self):
        interaction = self._interaction()
        command = self._command("open")
        with mock.patch.object(gallery, "is_pc_control_allowed", return_value=True), mock.patch.object(
            gallery, "start_gallery", side_effect=RuntimeError("secret detail")
        ):
            await command.callback(interaction)

        message = interaction.followup.send.await_args.args[0]
        self.assertIn("予期しないエラー", message)
        self.assertNotIn("secret detail", message)


if __name__ == "__main__":
    unittest.main()
