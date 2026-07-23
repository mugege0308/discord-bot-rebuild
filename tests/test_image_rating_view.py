import unittest
from pathlib import Path
from unittest import mock

import commands.prompt as prompt_module
from commands.prompt import (
    MAX_RATEABLE_IMAGES,
    REGENERATE_ROW,
    ImageRatingButton,
    ImageRatingView,
    RegenerateButton,
)
from services.generation_history import SavedGeneration
from services.image_selection import ImageSelectionResult
from services.sd_control import GeneratedImage
from services.shogou_prompt_designer import ShogouPromptResult


def _make_result() -> ShogouPromptResult:
    return ShogouPromptResult(
        positive="1girl, safe, masterpiece, high score, great score, absurdres",
        negative="lowres, bad anatomy",
        summary="テスト用の要約",
        assumptions=("背景を補完",),
    )


def _make_generated_images(count: int = 1) -> list[GeneratedImage]:
    return [
        GeneratedImage(path=Path(f"data/sd_outputs/test_{index}.png"), batch_index=index, seed=100 + index)
        for index in range(1, count + 1)
    ]


class ImageRatingViewLayoutTestCase(unittest.TestCase):
    def test_each_image_gets_like_meh_and_off_target_buttons(self):
        view = ImageRatingView([(1, 101), (2, 202)])

        ratings = [(child.image_id, child.rating) for child in view.children]
        self.assertEqual(
            ratings,
            [
                (101, "like"),
                (101, "meh"),
                (101, "off_target"),
                (202, "like"),
                (202, "meh"),
                (202, "off_target"),
            ],
        )

    def test_custom_id_contains_the_db_image_id(self):
        view = ImageRatingView([(1, 4242)])

        custom_ids = [child.custom_id for child in view.children]
        self.assertEqual(
            custom_ids,
            ["genai_rate:4242:like", "genai_rate:4242:meh", "genai_rate:4242:off_target"],
        )
        for custom_id in custom_ids:
            self.assertLessEqual(len(custom_id), 100, "Discord custom_id limit")

    def test_labels_follow_attachment_order_even_with_gaps(self):
        # Image 2 of 3 failed to map, so only display 1 and 3 are rateable -
        # the labels must still match what the user sees attached.
        view = ImageRatingView([(1, 101), (3, 303)])

        labels = [child.label for child in view.children]
        self.assertEqual(
            labels,
            ["👍 好き 1", "👎 微妙 1", "🎯 想定と違う 1", "👍 好き 3", "👎 微妙 3", "🎯 想定と違う 3"],
        )

    def test_ten_images_stay_within_discord_component_limits(self):
        # 3 buttons/image means only the first MAX_RATEABLE_IMAGES (4, in
        # attachment order - Phase 1C reserves the 5th row for regenerate
        # buttons) get buttons - 10 images would otherwise need far more
        # button slots than Discord's 5-row x 5-per-row (25) cap allows.
        view = ImageRatingView([(index, index * 10) for index in range(1, 11)])

        self.assertEqual(len(view.children), MAX_RATEABLE_IMAGES * 3)
        self.assertEqual(
            sorted({child.image_id for child in view.children}),
            [index * 10 for index in range(1, MAX_RATEABLE_IMAGES + 1)],
            "only the first MAX_RATEABLE_IMAGES images (by attachment order) become rateable",
        )
        rows = {}
        for child in view.children:
            rows.setdefault(child.row, []).append(child)
        self.assertLessEqual(len(rows), 5, "Discord allows at most 5 action rows")
        for row, children in rows.items():
            self.assertLessEqual(len(children), 5, f"row {row} exceeds 5 components")

    def test_max_rateable_images_all_get_buttons_within_limits(self):
        view = ImageRatingView(
            [(index, index * 10) for index in range(1, MAX_RATEABLE_IMAGES + 1)]
        )

        self.assertEqual(len(view.children), MAX_RATEABLE_IMAGES * 3)
        self.assertEqual(
            sorted({child.image_id for child in view.children}),
            [index * 10 for index in range(1, MAX_RATEABLE_IMAGES + 1)],
        )

    def test_timeout_is_one_hour(self):
        self.assertEqual(ImageRatingView([(1, 1)]).timeout, 3600)

    def test_view_is_not_persistent(self):
        # A persistent view requires timeout=None; Phase 1B deliberately is not one.
        self.assertIsNotNone(ImageRatingView([(1, 1)]).timeout)


class ImageRatingButtonCallbackTestCase(unittest.IsolatedAsyncioTestCase):
    def _interaction(self, user_id: int = 42):
        interaction = mock.MagicMock()
        interaction.user.id = user_id
        interaction.response.send_message = mock.AsyncMock()
        return interaction

    async def test_like_press_saves_and_replies_ephemerally(self):
        button = ImageRatingButton(image_id=101, rating="like", display_index=1, row=0)
        interaction = self._interaction()

        with mock.patch("commands.prompt.upsert_image_rating", return_value=True) as upsert:
            await button.callback(interaction)

        upsert.assert_called_once_with(101, 42, "like")
        interaction.response.send_message.assert_awaited_once()
        args, kwargs = interaction.response.send_message.call_args
        self.assertIn("好き", args[0])
        self.assertTrue(kwargs["ephemeral"])

    async def test_meh_press_saves_and_replies_ephemerally(self):
        button = ImageRatingButton(image_id=101, rating="meh", display_index=1, row=0)
        interaction = self._interaction()

        with mock.patch("commands.prompt.upsert_image_rating", return_value=True):
            await button.callback(interaction)

        args, kwargs = interaction.response.send_message.call_args
        self.assertIn("微妙", args[0])
        self.assertTrue(kwargs["ephemeral"])

    async def test_off_target_press_saves_and_replies_ephemerally(self):
        button = ImageRatingButton(image_id=101, rating="off_target", display_index=1, row=0)
        interaction = self._interaction()

        with mock.patch("commands.prompt.upsert_image_rating", return_value=True) as upsert:
            await button.callback(interaction)

        upsert.assert_called_once_with(101, 42, "off_target")
        args, kwargs = interaction.response.send_message.call_args
        self.assertIn("想定と違う", args[0])
        self.assertTrue(kwargs["ephemeral"])

    async def test_other_user_press_is_saved_as_that_users_rating(self):
        button = ImageRatingButton(image_id=101, rating="like", display_index=1, row=0)

        with mock.patch("commands.prompt.upsert_image_rating", return_value=True) as upsert:
            await button.callback(self._interaction(user_id=1))
            await button.callback(self._interaction(user_id=2))

        self.assertEqual(
            [call.args for call in upsert.call_args_list],
            [(101, 1, "like"), (101, 2, "like")],
        )

    async def test_failed_save_reports_briefly_without_leaking_db_error(self):
        button = ImageRatingButton(image_id=101, rating="like", display_index=1, row=0)
        interaction = self._interaction()

        with mock.patch("commands.prompt.upsert_image_rating", return_value=False):
            await button.callback(interaction)

        text = interaction.response.send_message.call_args.args[0]
        self.assertIn("保存できませんでした", text)
        for leaked in ("Traceback", "sqlite3", "FOREIGN KEY", "OperationalError"):
            self.assertNotIn(leaked, text)

    async def test_unexpected_exception_does_not_propagate(self):
        button = ImageRatingButton(image_id=101, rating="like", display_index=1, row=0)
        interaction = self._interaction()

        with mock.patch("commands.prompt.upsert_image_rating", side_effect=RuntimeError("db exploded")):
            await button.callback(interaction)

        text = interaction.response.send_message.call_args.args[0]
        self.assertIn("保存できませんでした", text)

    async def test_reply_failure_does_not_propagate(self):
        button = ImageRatingButton(image_id=101, rating="like", display_index=1, row=0)
        interaction = self._interaction()
        interaction.response.send_message = mock.AsyncMock(side_effect=RuntimeError("discord down"))

        with mock.patch("commands.prompt.upsert_image_rating", return_value=True):
            await button.callback(interaction)  # must not raise


class ImageRatingViewTimeoutTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_disables_buttons_and_edits_message(self):
        view = ImageRatingView([(1, 101)])
        view.message = mock.MagicMock()
        view.message.edit = mock.AsyncMock()

        await view.on_timeout()

        self.assertTrue(all(child.disabled for child in view.children))
        view.message.edit.assert_awaited_once()

    async def test_timeout_without_message_is_a_no_op(self):
        view = ImageRatingView([(1, 101)])
        view.message = None

        await view.on_timeout()  # must not raise

        self.assertTrue(all(child.disabled for child in view.children))

    async def test_timeout_edit_failure_is_swallowed(self):
        view = ImageRatingView([(1, 101)])
        view.message = mock.MagicMock()
        view.message.edit = mock.AsyncMock(side_effect=RuntimeError("message gone"))

        await view.on_timeout()  # must not raise


class RatingViewBuildTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_only_sent_images_become_rateable(self):
        images = _make_generated_images(3)
        saved = SavedGeneration(request_id=1, image_ids=[11, 22, 33])
        # Selection AI sent only the 2nd image.
        send_paths = [images[1].path]

        with mock.patch(
            "commands.prompt.resolve_image_ids_for_paths", return_value=[22]
        ) as resolve_mock:
            view, rated_ids = await prompt_module._build_rating_view(saved, images, send_paths)

        self.assertEqual(rated_ids, [22])
        self.assertEqual([child.image_id for child in view.children], [22, 22, 22])
        # The unsent images must never be offered for rating.
        self.assertNotIn(11, rated_ids)
        self.assertNotIn(33, rated_ids)
        self.assertEqual(resolve_mock.call_args.args[2], send_paths)

    async def test_no_view_when_history_save_failed(self):
        view, rated_ids = await prompt_module._build_rating_view(None, _make_generated_images(1), [Path("a.png")])
        self.assertIsNone(view)
        self.assertEqual(rated_ids, [])

    async def test_no_view_when_nothing_can_be_mapped(self):
        images = _make_generated_images(1)
        saved = SavedGeneration(request_id=1, image_ids=[11])

        with mock.patch("commands.prompt.resolve_image_ids_for_paths", return_value=[None]):
            view, rated_ids = await prompt_module._build_rating_view(saved, images, [images[0].path])

        self.assertIsNone(view)
        self.assertEqual(rated_ids, [])

    async def test_unmapped_image_is_skipped_but_others_remain_rateable(self):
        images = _make_generated_images(2)
        saved = SavedGeneration(request_id=1, image_ids=[11, 22])

        with mock.patch("commands.prompt.resolve_image_ids_for_paths", return_value=[None, 22]):
            view, rated_ids = await prompt_module._build_rating_view(
                saved, images, [images[0].path, images[1].path]
            )

        self.assertEqual(rated_ids, [22])
        self.assertEqual(
            [child.label for child in view.children],
            ["👍 好き 2", "👎 微妙 2", "🎯 想定と違う 2"],
        )

    async def test_mapping_failure_skips_buttons_without_raising(self):
        images = _make_generated_images(1)
        saved = SavedGeneration(request_id=1, image_ids=[11])

        with mock.patch(
            "commands.prompt.resolve_image_ids_for_paths", side_effect=RuntimeError("hash failed")
        ):
            view, rated_ids = await prompt_module._build_rating_view(saved, images, [images[0].path])

        self.assertIsNone(view)
        self.assertEqual(rated_ids, [])


class ResultViewBuildTestCase(unittest.IsolatedAsyncioTestCase):
    """Phase 1C: _build_result_view combines rating buttons with the
    per-request regenerate buttons into one View."""

    async def test_regenerate_buttons_are_added_even_at_max_rateable_images(self):
        images = _make_generated_images(MAX_RATEABLE_IMAGES)
        saved = SavedGeneration(request_id=7, image_ids=list(range(11, 11 + MAX_RATEABLE_IMAGES)))
        send_paths = [image.path for image in images]

        with mock.patch(
            "commands.prompt.resolve_image_ids_for_paths",
            return_value=list(range(11, 11 + MAX_RATEABLE_IMAGES)),
        ):
            view, rated_ids = await prompt_module._build_result_view(saved, images, send_paths, owner_id=42)

        self.assertEqual(len(rated_ids), MAX_RATEABLE_IMAGES)
        rows = {}
        for child in view.children:
            rows.setdefault(child.row, []).append(child)
        self.assertLessEqual(len(rows), 5, "Discord allows at most 5 action rows")
        for row, children in rows.items():
            self.assertLessEqual(len(children), 5, f"row {row} exceeds 5 components")
        regenerate_children = [child for child in view.children if isinstance(child, RegenerateButton)]
        self.assertEqual(len(regenerate_children), 2, "one regenerate button per REGENERATION_TYPES entry")
        self.assertTrue(all(child.row == REGENERATE_ROW for child in regenerate_children))

    async def test_regenerate_buttons_are_added_even_when_no_image_is_rateable(self):
        # _build_rating_view returns None here (nothing could be mapped), but
        # the request itself still saved successfully, so the regenerate
        # buttons must still be offered.
        saved = SavedGeneration(request_id=9, image_ids=[11])
        images = _make_generated_images(1)

        with mock.patch("commands.prompt.resolve_image_ids_for_paths", return_value=[None]):
            view, rated_ids = await prompt_module._build_result_view(
                saved, images, [images[0].path], owner_id=42
            )

        self.assertEqual(rated_ids, [])
        self.assertIsNotNone(view)
        regenerate_children = [child for child in view.children if isinstance(child, RegenerateButton)]
        self.assertEqual(len(regenerate_children), 2)

    async def test_no_view_at_all_when_history_save_failed(self):
        view, rated_ids = await prompt_module._build_result_view(
            None, _make_generated_images(1), [Path("a.png")], owner_id=42
        )

        self.assertIsNone(view)
        self.assertEqual(rated_ids, [])


class DiscordMessageBookkeepingTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_message_and_channel_ids_are_stored(self):
        sent_message = mock.MagicMock()
        sent_message.id = 555
        sent_message.channel.id = 777

        with mock.patch("commands.prompt.set_image_discord_message", return_value=True) as store:
            await prompt_module._store_discord_message_safely([11, 22], sent_message)

        store.assert_called_once_with([11, 22], 555, 777)

    async def test_missing_message_is_a_no_op(self):
        with mock.patch("commands.prompt.set_image_discord_message") as store:
            await prompt_module._store_discord_message_safely([11], None)
        store.assert_not_called()

    async def test_store_failure_does_not_propagate(self):
        sent_message = mock.MagicMock()
        sent_message.id = 555
        sent_message.channel.id = 777

        with mock.patch(
            "commands.prompt.set_image_discord_message", side_effect=RuntimeError("db down")
        ):
            await prompt_module._store_discord_message_safely([11], sent_message)  # must not raise


class GenerateAiEndToEndWiringTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_generate_ai_attaches_view_and_stores_message_ids(self):
        interaction = mock.MagicMock()
        interaction.user.id = 555
        images = _make_generated_images(2)
        selection = ImageSelectionResult(
            selected_paths=[images[0].path], used_selection=True, message="上位1枚", error=""
        )
        sent_message = mock.MagicMock()
        sent_message.id = 999
        sent_message.channel.id = 888

        with mock.patch(
            "commands.prompt.generate_images_with_metadata", return_value=images
        ), mock.patch(
            "commands.prompt.select_generated_images", return_value=selection
        ), mock.patch(
            "commands.prompt.save_generation_history_detailed",
            return_value=SavedGeneration(request_id=1, image_ids=[11, 22]),
        ), mock.patch(
            "commands.prompt.resolve_image_ids_for_paths", return_value=[11]
        ), mock.patch(
            "commands.prompt._send_image_files", return_value=sent_message
        ) as send_mock, mock.patch(
            "commands.prompt.set_image_discord_message", return_value=True
        ) as store_mock:
            await prompt_module._generate_images_from_shogou_result(
                interaction,
                _make_result(),
                label_prefix="generate_ai",
                request_text="テスト依頼",
                reference_id=None,
            )

        send_mock.assert_called_once()
        view = send_mock.call_args.kwargs["view"]
        self.assertIsInstance(view, ImageRatingView)
        self.assertTrue(send_mock.call_args.kwargs["wait"])
        rating_children = [child for child in view.children if isinstance(child, ImageRatingButton)]
        self.assertEqual([child.image_id for child in rating_children], [11, 11, 11])
        # Phase 1C: the same view also carries the two regenerate buttons for
        # this generation_request, on the row after the rating buttons.
        regenerate_children = [child for child in view.children if isinstance(child, RegenerateButton)]
        self.assertEqual(len(regenerate_children), 2)
        self.assertEqual({child.regeneration_type for child in regenerate_children}, {"same_prompt", "change_composition"})
        self.assertTrue(all(child.request_id == 1 for child in regenerate_children))
        self.assertTrue(all(child.owner_id == 555 for child in regenerate_children))
        self.assertTrue(all(child.row == REGENERATE_ROW for child in regenerate_children))
        self.assertIs(view.message, sent_message)
        store_mock.assert_called_once_with([11], 999, 888)

    async def test_images_still_sent_when_history_save_fails(self):
        interaction = mock.MagicMock()
        interaction.user.id = 555
        images = _make_generated_images(1)
        selection = ImageSelectionResult(selected_paths=[], used_selection=False, message="", error="")

        with mock.patch(
            "commands.prompt.generate_images_with_metadata", return_value=images
        ), mock.patch(
            "commands.prompt.select_generated_images", return_value=selection
        ), mock.patch(
            "commands.prompt.save_generation_history_detailed", return_value=None
        ), mock.patch(
            "commands.prompt._send_image_files"
        ) as send_mock, mock.patch(
            "commands.prompt._send_error"
        ) as error_mock:
            await prompt_module._generate_images_from_shogou_result(
                interaction,
                _make_result(),
                label_prefix="generate_ai",
                request_text="テスト依頼",
                reference_id=None,
            )

        send_mock.assert_called_once()
        self.assertIsNone(send_mock.call_args.kwargs["view"], "no history rows means no rating buttons")
        error_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
