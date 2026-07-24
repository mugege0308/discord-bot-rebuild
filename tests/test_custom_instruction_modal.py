import asyncio
import unittest
from unittest import mock

import commands.prompt as prompt_module
from commands.prompt import (
    CUSTOM_INSTRUCTION_EMPTY_MESSAGE,
    CUSTOM_INSTRUCTION_MAX_LENGTH,
    REGENERATE_ALREADY_RUNNING_MESSAGE,
    REGENERATE_ONLY_OWNER_MESSAGE,
    REGENERATE_STARTED_MESSAGE,
    CustomInstructionModal,
    CustomInstructionRegenerateButton,
)


def _interaction(user_id: int = 555):
    interaction = mock.MagicMock()
    interaction.user.id = user_id
    interaction.response.send_message = mock.AsyncMock()
    interaction.response.send_modal = mock.AsyncMock()
    return interaction


def _modal_with_value(request_id: int, owner_id: int, value: str) -> CustomInstructionModal:
    modal = CustomInstructionModal(request_id, owner_id)
    # discord.ui.TextInput.value has no public setter (it is populated from
    # interaction payload data on a real submit); setting the backing
    # attribute directly is the standard way to simulate a filled-in modal.
    modal.instruction._value = value
    return modal


class CustomInstructionRegenerateButtonTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        prompt_module._active_regenerations.clear()

    async def test_other_user_press_is_rejected_without_opening_modal(self):
        button = CustomInstructionRegenerateButton(request_id=1, owner_id=555, row=4)
        interaction = _interaction(user_id=999)

        await button.callback(interaction)

        interaction.response.send_message.assert_awaited_once()
        args, kwargs = interaction.response.send_message.call_args
        self.assertEqual(args[0], REGENERATE_ONLY_OWNER_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])
        interaction.response.send_modal.assert_not_awaited()

    async def test_double_click_while_active_is_rejected_without_opening_modal(self):
        button = CustomInstructionRegenerateButton(request_id=7, owner_id=555, row=4)
        prompt_module._active_regenerations.add(7)
        interaction = _interaction(user_id=555)

        await button.callback(interaction)

        args, kwargs = interaction.response.send_message.call_args
        self.assertEqual(args[0], REGENERATE_ALREADY_RUNNING_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])
        interaction.response.send_modal.assert_not_awaited()

    async def test_owner_press_opens_the_modal(self):
        button = CustomInstructionRegenerateButton(request_id=42, owner_id=555, row=4)
        interaction = _interaction(user_id=555)

        await button.callback(interaction)

        interaction.response.send_modal.assert_awaited_once()
        (modal,), _ = interaction.response.send_modal.call_args
        self.assertIsInstance(modal, CustomInstructionModal)
        self.assertEqual(modal.request_id, 42)
        self.assertEqual(modal.owner_id, 555)


class CustomInstructionModalSubmitTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        prompt_module._active_regenerations.clear()

    async def test_input_field_enforces_max_length(self):
        modal = CustomInstructionModal(1, 555)
        self.assertEqual(modal.instruction.max_length, CUSTOM_INSTRUCTION_MAX_LENGTH)
        self.assertEqual(CUSTOM_INSTRUCTION_MAX_LENGTH, 1000)

    async def test_other_user_submit_is_rejected_ephemerally_without_starting_work(self):
        modal = _modal_with_value(1, 555, "笑顔にして")
        interaction = _interaction(user_id=999)

        with mock.patch("commands.prompt._run_regeneration") as run_mock:
            await modal.on_submit(interaction)

        args, kwargs = interaction.response.send_message.call_args
        self.assertEqual(args[0], REGENERATE_ONLY_OWNER_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])
        run_mock.assert_not_called()
        self.assertNotIn(1, prompt_module._active_regenerations)

    async def test_empty_instruction_is_rejected_and_does_not_start_work(self):
        modal = _modal_with_value(2, 555, "   ")
        interaction = _interaction(user_id=555)

        with mock.patch("commands.prompt._run_regeneration") as run_mock:
            await modal.on_submit(interaction)

        args, kwargs = interaction.response.send_message.call_args
        self.assertEqual(args[0], CUSTOM_INSTRUCTION_EMPTY_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])
        run_mock.assert_not_called()
        self.assertNotIn(2, prompt_module._active_regenerations)

    async def test_max_length_instruction_is_accepted_and_starts_work(self):
        long_instruction = "a" * CUSTOM_INSTRUCTION_MAX_LENGTH
        modal = _modal_with_value(3, 555, long_instruction)
        interaction = _interaction(user_id=555)

        with mock.patch("commands.prompt._run_regeneration", new_callable=mock.AsyncMock) as run_mock:
            before = asyncio.all_tasks()
            await modal.on_submit(interaction)
            scheduled = asyncio.all_tasks() - before
            for task in scheduled:
                await task

        args, kwargs = interaction.response.send_message.call_args
        self.assertEqual(args[0], REGENERATE_STARTED_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])
        run_mock.assert_called_once_with(
            interaction, 3, "custom_instruction", custom_instruction=long_instruction
        )

    async def test_surrounding_whitespace_is_stripped_before_starting_work(self):
        modal = _modal_with_value(4, 555, "  背景を夕方に  ")
        interaction = _interaction(user_id=555)

        with mock.patch("commands.prompt._run_regeneration", new_callable=mock.AsyncMock) as run_mock:
            before = asyncio.all_tasks()
            await modal.on_submit(interaction)
            scheduled = asyncio.all_tasks() - before
            for task in scheduled:
                await task

        run_mock.assert_called_once_with(
            interaction, 4, "custom_instruction", custom_instruction="背景を夕方に"
        )

    async def test_double_click_while_active_is_rejected_and_does_not_schedule_again(self):
        modal = _modal_with_value(5, 555, "笑顔にして")
        prompt_module._active_regenerations.add(5)
        interaction = _interaction(user_id=555)

        with mock.patch("commands.prompt._run_regeneration") as run_mock:
            await modal.on_submit(interaction)

        args, kwargs = interaction.response.send_message.call_args
        self.assertEqual(args[0], REGENERATE_ALREADY_RUNNING_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])
        run_mock.assert_not_called()

    async def test_ack_failure_clears_active_flag_and_does_not_schedule(self):
        modal = _modal_with_value(6, 555, "笑顔にして")
        interaction = _interaction(user_id=555)
        interaction.response.send_message = mock.AsyncMock(side_effect=RuntimeError("discord down"))

        with mock.patch("commands.prompt._run_regeneration") as run_mock:
            await modal.on_submit(interaction)  # must not raise

        run_mock.assert_not_called()
        self.assertNotIn(6, prompt_module._active_regenerations)


if __name__ == "__main__":
    unittest.main()
