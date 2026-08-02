import unittest
from unittest import mock

import discord

from commands.image_feedback import (
    FEEDBACK_ISSUE_ORDER,
    FeedbackAcceptButton,
    FeedbackCommentModal,
    FeedbackIssueSelect,
    ImageFeedbackView,
    _handle_image_feedback_context_menu,
)
from services.generation_history import FeedbackImageTarget


def _make_target(image_id: int = 101, request_id: int = 1, request_user_id: int = 42) -> FeedbackImageTarget:
    return FeedbackImageTarget(
        image_id=image_id,
        request_id=request_id,
        request_user_id=request_user_id,
        image_index=1,
        image_path="data/sd_outputs/img_01.png",
        seed=123,
        checkpoint_name="animagineXL40Opt_clear.safetensors",
        sampler_name="euler_ancestral",
        scheduler="normal",
        steps=20,
        cfg=5.0,
        width=1024,
        height=1024,
        was_selected_by_ai=False,
    )


class ImageFeedbackViewLayoutTestCase(unittest.TestCase):
    def test_single_image_has_no_picker(self):
        view = ImageFeedbackView([_make_target()], owner_id=42, guild_id=1, channel_id=2, message_id=3)

        custom_ids = [child.custom_id for child in view.children]
        self.assertEqual(custom_ids, ["genai_fb_accept", "genai_fb_issues"])

    def test_multiple_images_get_a_picker_defaulting_to_the_first(self):
        targets = [_make_target(image_id=101), _make_target(image_id=202)]
        view = ImageFeedbackView(targets, owner_id=42, guild_id=1, channel_id=2, message_id=3)

        custom_ids = [child.custom_id for child in view.children]
        self.assertEqual(custom_ids, ["genai_fb_pick", "genai_fb_accept", "genai_fb_issues"])
        self.assertEqual(view.selected_image_id, 101)

    def test_all_custom_ids_are_unique_and_within_discord_limit(self):
        targets = [_make_target(image_id=101), _make_target(image_id=202), _make_target(image_id=303)]
        view = ImageFeedbackView(targets, owner_id=42, guild_id=1, channel_id=2, message_id=3)

        custom_ids = [child.custom_id for child in view.children]
        self.assertEqual(len(custom_ids), len(set(custom_ids)))
        for custom_id in custom_ids:
            self.assertLessEqual(len(custom_id), 100)

    def test_issue_select_offers_every_allowed_category_and_allows_multi_select(self):
        view = ImageFeedbackView([_make_target()], owner_id=42, guild_id=1, channel_id=2, message_id=3)
        issue_select = next(child for child in view.children if isinstance(child, FeedbackIssueSelect))

        self.assertEqual([option.value for option in issue_select.options], list(FEEDBACK_ISSUE_ORDER))
        self.assertEqual(issue_select.min_values, 1)
        self.assertEqual(issue_select.max_values, len(FEEDBACK_ISSUE_ORDER))

    def test_stays_within_discord_component_limits(self):
        targets = [_make_target(image_id=index) for index in range(1, 6)]
        view = ImageFeedbackView(targets, owner_id=42, guild_id=1, channel_id=2, message_id=3)

        rows: dict[int, list] = {}
        for child in view.children:
            rows.setdefault(child.row, []).append(child)
        self.assertLessEqual(len(rows), 5)
        for children in rows.values():
            self.assertLessEqual(len(children), 5)

    def test_timeout_is_well_under_the_ephemeral_webhook_token_expiry(self):
        # Discord's ephemeral interaction token is valid for 15 minutes
        # (900s); this view must time out safely before that.
        view = ImageFeedbackView([_make_target()], owner_id=42, guild_id=1, channel_id=2, message_id=3)
        self.assertLess(view.timeout, 900)


class InteractionCheckTestCase(unittest.IsolatedAsyncioTestCase):
    def _interaction(self, user_id: int):
        interaction = mock.MagicMock()
        interaction.user.id = user_id
        interaction.response.send_message = mock.AsyncMock()
        return interaction

    async def test_owner_is_allowed(self):
        view = ImageFeedbackView([_make_target()], owner_id=42, guild_id=1, channel_id=2, message_id=3)
        with mock.patch("commands.image_feedback.is_pc_control_allowed", return_value=False):
            allowed = await view.interaction_check(self._interaction(42))
        self.assertTrue(allowed)

    async def test_other_user_is_denied_with_a_single_ephemeral_reply(self):
        view = ImageFeedbackView([_make_target()], owner_id=42, guild_id=1, channel_id=2, message_id=3)
        interaction = self._interaction(999)
        with mock.patch("commands.image_feedback.is_pc_control_allowed", return_value=False):
            allowed = await view.interaction_check(interaction)

        self.assertFalse(allowed)
        interaction.response.send_message.assert_awaited_once()
        args, kwargs = interaction.response.send_message.call_args
        self.assertTrue(kwargs.get("ephemeral"))

    async def test_pc_control_allowed_admin_may_act_for_another_owner(self):
        view = ImageFeedbackView([_make_target()], owner_id=42, guild_id=1, channel_id=2, message_id=3)
        with mock.patch("commands.image_feedback.is_pc_control_allowed", return_value=True):
            allowed = await view.interaction_check(self._interaction(999))
        self.assertTrue(allowed)


class FeedbackAcceptButtonTestCase(unittest.IsolatedAsyncioTestCase):
    def _interaction(self, user_id: int = 42):
        interaction = mock.MagicMock()
        interaction.user.id = user_id
        interaction.response.send_message = mock.AsyncMock()
        return interaction

    async def test_accept_saves_with_no_categories_and_replies_ephemerally(self):
        view = ImageFeedbackView([_make_target()], owner_id=42, guild_id=1, channel_id=2, message_id=3)
        view.save = mock.AsyncMock(return_value=True)
        button = next(child for child in view.children if isinstance(child, FeedbackAcceptButton))

        await button.callback(self._interaction())

        view.save.assert_awaited_once()
        _, kwargs = view.save.call_args
        self.assertTrue(kwargs["accepted"])
        self.assertEqual(list(kwargs["issue_categories"]), [])
        self.assertIsNone(kwargs["comment"])

    async def test_save_failure_reports_briefly(self):
        view = ImageFeedbackView([_make_target()], owner_id=42, guild_id=1, channel_id=2, message_id=3)
        view.save = mock.AsyncMock(return_value=False)
        button = next(child for child in view.children if isinstance(child, FeedbackAcceptButton))
        interaction = self._interaction()

        await button.callback(interaction)

        text = interaction.response.send_message.call_args.args[0]
        self.assertIn("失敗", text)


class FeedbackIssueSelectTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_selecting_categories_opens_the_comment_modal(self):
        view = ImageFeedbackView([_make_target()], owner_id=42, guild_id=1, channel_id=2, message_id=3)
        select = next(child for child in view.children if isinstance(child, FeedbackIssueSelect))
        select._values = ["hair_mismatch", "outfit_mismatch"]
        interaction = mock.MagicMock()
        interaction.response.send_modal = mock.AsyncMock()

        await select.callback(interaction)

        self.assertEqual(view.pending_categories, ["hair_mismatch", "outfit_mismatch"])
        interaction.response.send_modal.assert_awaited_once()
        modal = interaction.response.send_modal.call_args.args[0]
        self.assertIsInstance(modal, FeedbackCommentModal)


class FeedbackCommentModalTestCase(unittest.IsolatedAsyncioTestCase):
    def _interaction(self, user_id: int = 42):
        interaction = mock.MagicMock()
        interaction.user.id = user_id
        interaction.response.send_message = mock.AsyncMock()
        return interaction

    async def test_submit_saves_pending_categories_with_comment(self):
        view = ImageFeedbackView([_make_target()], owner_id=42, guild_id=1, channel_id=2, message_id=3)
        view.pending_categories = ["low_quality"]
        view.save = mock.AsyncMock(return_value=True)
        modal = FeedbackCommentModal(view=view)
        modal.comment._value = "画質が粗い気がします"

        await modal.on_submit(self._interaction())

        view.save.assert_awaited_once()
        _, kwargs = view.save.call_args
        self.assertFalse(kwargs["accepted"])
        self.assertEqual(kwargs["issue_categories"], ["low_quality"])
        self.assertEqual(kwargs["comment"], "画質が粗い気がします")

    async def test_blank_comment_is_saved_as_none(self):
        view = ImageFeedbackView([_make_target()], owner_id=42, guild_id=1, channel_id=2, message_id=3)
        view.pending_categories = ["other"]
        view.save = mock.AsyncMock(return_value=True)
        modal = FeedbackCommentModal(view=view)
        modal.comment._value = "   "

        await modal.on_submit(self._interaction())

        self.assertIsNone(view.save.call_args.kwargs["comment"])

    async def test_other_user_submitting_the_modal_is_denied(self):
        view = ImageFeedbackView([_make_target()], owner_id=42, guild_id=1, channel_id=2, message_id=3)
        view.save = mock.AsyncMock(return_value=True)
        modal = FeedbackCommentModal(view=view)
        modal.comment._value = "コメント"
        interaction = self._interaction(user_id=999)

        with mock.patch("commands.image_feedback.is_pc_control_allowed", return_value=False):
            await modal.on_submit(interaction)

        view.save.assert_not_called()
        interaction.response.send_message.assert_awaited_once()


class ContextMenuHandlerTestCase(unittest.IsolatedAsyncioTestCase):
    def _interaction(self, user_id: int):
        interaction = mock.MagicMock()
        interaction.user.id = user_id
        interaction.guild_id = 1
        interaction.response.send_message = mock.AsyncMock()
        return interaction

    def _message(self):
        message = mock.MagicMock()
        message.id = 555
        message.channel.id = 777
        return message

    async def test_no_history_found_reports_briefly(self):
        interaction = self._interaction(42)
        with mock.patch("commands.image_feedback.get_generation_images_for_message", return_value=[]):
            await _handle_image_feedback_context_menu(interaction, self._message())

        text = interaction.response.send_message.call_args.args[0]
        self.assertIn("見つかりませんでした", text)

    async def test_non_owner_non_admin_is_denied(self):
        interaction = self._interaction(999)
        target = _make_target(request_user_id=42)
        with mock.patch(
            "commands.image_feedback.get_generation_images_for_message", return_value=[target]
        ), mock.patch("commands.image_feedback.is_pc_control_allowed", return_value=False):
            await _handle_image_feedback_context_menu(interaction, self._message())

        text = interaction.response.send_message.call_args.args[0]
        self.assertIn("本人だけ", text)

    async def test_owner_gets_the_feedback_view(self):
        interaction = self._interaction(42)
        target = _make_target(request_user_id=42)
        with mock.patch("commands.image_feedback.get_generation_images_for_message", return_value=[target]):
            await _handle_image_feedback_context_menu(interaction, self._message())

        interaction.response.send_message.assert_awaited_once()
        _, kwargs = interaction.response.send_message.call_args
        self.assertIsInstance(kwargs["view"], ImageFeedbackView)
        self.assertTrue(kwargs["ephemeral"])

    async def test_lookup_failure_reports_briefly_without_raising(self):
        interaction = self._interaction(42)
        with mock.patch(
            "commands.image_feedback.get_generation_images_for_message", side_effect=RuntimeError("db down")
        ):
            await _handle_image_feedback_context_menu(interaction, self._message())  # must not raise

        text = interaction.response.send_message.call_args.args[0]
        self.assertIn("失敗", text)


if __name__ == "__main__":
    unittest.main()
