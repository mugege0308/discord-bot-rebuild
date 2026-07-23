import asyncio
import unittest
from unittest import mock

import commands.prompt as prompt_module
from commands.prompt import (
    REGENERATE_ALREADY_RUNNING_MESSAGE,
    REGENERATE_FAILED_MESSAGE,
    REGENERATE_ONLY_OWNER_MESSAGE,
    REGENERATE_PARENT_MISSING_MESSAGE,
    REGENERATE_STARTED_MESSAGE,
    RegenerateButton,
)
from services.generation_history import GenerationRequestDetail
from services.shogou_prompt_designer import ShogouPromptResult


def _make_parent(**overrides) -> GenerationRequestDetail:
    defaults = dict(
        id=1,
        created_at="2026-01-01T00:00:00+09:00",
        user_id=555,
        request_text="雨の日の喫茶店で読書する女の子",
        reference_id=None,
        positive="1girl, cafe, reading, rain",
        negative="lowres, bad anatomy",
        summary="喫茶店で読書する猫耳の女の子",
        assumptions=("背景を補完",),
        workflow_file="workflows/comfy_workflow.json",
        workflow_hash="deadbeef",
        selection_used=True,
        selection_message="画像選別AIで上位1枚を選びました。",
        parent_request_id=None,
        regeneration_type=None,
    )
    defaults.update(overrides)
    return GenerationRequestDetail(**defaults)


class RegenerateButtonCallbackTestCase(unittest.IsolatedAsyncioTestCase):
    def _interaction(self, user_id: int = 555):
        interaction = mock.MagicMock()
        interaction.user.id = user_id
        interaction.response.send_message = mock.AsyncMock()
        return interaction

    def _button(self, *, request_id: int = 1, regeneration_type: str = "same_prompt", owner_id: int = 555) -> RegenerateButton:
        return RegenerateButton(request_id=request_id, regeneration_type=regeneration_type, owner_id=owner_id, row=4)

    async def asyncTearDown(self):
        prompt_module._active_regenerations.clear()

    async def test_other_user_press_is_rejected_ephemerally_without_starting_work(self):
        button = self._button(owner_id=555)
        interaction = self._interaction(user_id=999)

        with mock.patch("commands.prompt._run_regeneration") as run_mock:
            await button.callback(interaction)

        interaction.response.send_message.assert_awaited_once()
        args, kwargs = interaction.response.send_message.call_args
        self.assertEqual(args[0], REGENERATE_ONLY_OWNER_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])
        run_mock.assert_not_called()
        self.assertNotIn(1, prompt_module._active_regenerations)

    async def test_owner_press_acknowledges_and_schedules_regeneration(self):
        button = self._button(request_id=42)
        interaction = self._interaction(user_id=555)

        with mock.patch("commands.prompt._run_regeneration", new_callable=mock.AsyncMock) as run_mock:
            before = asyncio.all_tasks()
            await button.callback(interaction)
            scheduled = asyncio.all_tasks() - before
            for task in scheduled:
                await task

        args, kwargs = interaction.response.send_message.call_args
        self.assertEqual(args[0], REGENERATE_STARTED_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])
        run_mock.assert_called_once_with(interaction, 42, "same_prompt")

    async def test_double_click_while_active_is_rejected_and_does_not_schedule_again(self):
        button = self._button(request_id=7)
        prompt_module._active_regenerations.add(7)
        interaction = self._interaction(user_id=555)

        with mock.patch("commands.prompt._run_regeneration") as run_mock:
            await button.callback(interaction)

        args, kwargs = interaction.response.send_message.call_args
        self.assertEqual(args[0], REGENERATE_ALREADY_RUNNING_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])
        run_mock.assert_not_called()

    async def test_ack_failure_clears_active_flag_and_does_not_schedule(self):
        button = self._button(request_id=3)
        interaction = self._interaction(user_id=555)
        interaction.response.send_message = mock.AsyncMock(side_effect=RuntimeError("discord down"))

        with mock.patch("commands.prompt._run_regeneration") as run_mock:
            await button.callback(interaction)  # must not raise

        run_mock.assert_not_called()
        self.assertNotIn(3, prompt_module._active_regenerations)


class RunRegenerationTestCase(unittest.IsolatedAsyncioTestCase):
    def _interaction(self, user_id: int = 555):
        interaction = mock.MagicMock()
        interaction.user.id = user_id
        interaction.followup.send = mock.AsyncMock()
        return interaction

    async def asyncTearDown(self):
        prompt_module._active_regenerations.clear()

    async def test_missing_parent_reports_error_and_does_nothing_else(self):
        interaction = self._interaction()
        prompt_module._active_regenerations.add(1)

        with mock.patch("commands.prompt.get_generation_request", return_value=None), mock.patch(
            "commands.prompt.generate_shogou_composition_variation"
        ) as variation_mock, mock.patch(
            "commands.prompt._generate_images_from_shogou_result"
        ) as generate_mock:
            await prompt_module._run_regeneration(interaction, 1, "same_prompt")

        interaction.followup.send.assert_awaited_once()
        _, kwargs = interaction.followup.send.call_args
        self.assertEqual(kwargs["content"], REGENERATE_PARENT_MISSING_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])
        variation_mock.assert_not_called()
        generate_mock.assert_not_called()
        self.assertNotIn(1, prompt_module._active_regenerations, "must clear the anti-spam flag even on failure")

    async def test_same_prompt_does_not_call_prompt_designer_and_reuses_positive_negative(self):
        parent = _make_parent(id=10, positive="1girl, cafe", negative="lowres", request_text="元の依頼")
        interaction = self._interaction()

        with mock.patch("commands.prompt.get_generation_request", return_value=parent), mock.patch(
            "commands.prompt.generate_shogou_composition_variation"
        ) as variation_mock, mock.patch(
            "commands.prompt._generate_images_from_shogou_result"
        ) as generate_mock:
            await prompt_module._run_regeneration(interaction, 10, "same_prompt")

        variation_mock.assert_not_called()
        generate_mock.assert_called_once()
        args, kwargs = generate_mock.call_args
        result = args[1]
        self.assertEqual(result.positive, parent.positive)
        self.assertEqual(result.negative, parent.negative)
        self.assertEqual(result.summary, parent.summary)
        self.assertEqual(result.assumptions, parent.assumptions)
        self.assertFalse(result.blocked)
        self.assertEqual(kwargs["request_text"], parent.request_text)
        self.assertEqual(kwargs["reference_id"], parent.reference_id)
        self.assertEqual(kwargs["parent_request_id"], 10)
        self.assertEqual(kwargs["regeneration_type"], "same_prompt")

    async def test_change_composition_calls_prompt_designer_with_parent_fields(self):
        parent = _make_parent(id=11, reference_id=7)
        new_result = ShogouPromptResult(
            positive="1girl, cafe, standing, wide shot",
            negative="lowres",
            summary="新しい構図の要約",
            assumptions=("構図を変更",),
        )
        interaction = self._interaction()

        with mock.patch("commands.prompt.get_generation_request", return_value=parent), mock.patch(
            "commands.prompt.generate_shogou_composition_variation", return_value=new_result
        ) as variation_mock, mock.patch(
            "commands.prompt._generate_images_from_shogou_result"
        ) as generate_mock:
            await prompt_module._run_regeneration(interaction, 11, "change_composition")

        variation_mock.assert_called_once_with(
            parent.request_text,
            parent.positive,
            parent.negative,
            parent.summary,
            parent.assumptions,
            interaction.user.id,
            parent.reference_id,
        )
        generate_mock.assert_called_once()
        args, kwargs = generate_mock.call_args
        self.assertIs(args[1], new_result)
        self.assertEqual(kwargs["parent_request_id"], 11)
        self.assertEqual(kwargs["regeneration_type"], "change_composition")

    async def test_change_composition_blocked_result_is_reported_and_stops(self):
        parent = _make_parent(id=12)
        blocked_result = ShogouPromptResult(blocked_reason="NGワードを検出しました。")
        interaction = self._interaction()

        with mock.patch("commands.prompt.get_generation_request", return_value=parent), mock.patch(
            "commands.prompt.generate_shogou_composition_variation", return_value=blocked_result
        ), mock.patch("commands.prompt._generate_images_from_shogou_result") as generate_mock:
            await prompt_module._run_regeneration(interaction, 12, "change_composition")

        generate_mock.assert_not_called()
        interaction.followup.send.assert_awaited_once()
        _, kwargs = interaction.followup.send.call_args
        self.assertEqual(kwargs["content"], "NGワードを検出しました。")
        self.assertTrue(kwargs["ephemeral"])

    async def test_unknown_regeneration_type_is_a_safe_no_op(self):
        parent = _make_parent(id=13)
        interaction = self._interaction()

        with mock.patch("commands.prompt.get_generation_request", return_value=parent), mock.patch(
            "commands.prompt.generate_shogou_composition_variation"
        ) as variation_mock, mock.patch(
            "commands.prompt._generate_images_from_shogou_result"
        ) as generate_mock:
            await prompt_module._run_regeneration(interaction, 13, "unknown_type")  # must not raise

        variation_mock.assert_not_called()
        generate_mock.assert_not_called()

    async def test_unexpected_exception_is_reported_generically_without_leaking_detail(self):
        interaction = self._interaction()
        prompt_module._active_regenerations.add(14)

        with mock.patch(
            "commands.prompt.get_generation_request", side_effect=RuntimeError("sqlite3.OperationalError: disk I/O error")
        ):
            await prompt_module._run_regeneration(interaction, 14, "same_prompt")  # must not raise

        interaction.followup.send.assert_awaited_once()
        _, kwargs = interaction.followup.send.call_args
        self.assertEqual(kwargs["content"], REGENERATE_FAILED_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])
        for leaked in ("Traceback", "sqlite3", "OperationalError", "disk I/O"):
            self.assertNotIn(leaked, kwargs["content"])
        self.assertNotIn(14, prompt_module._active_regenerations)


if __name__ == "__main__":
    unittest.main()
