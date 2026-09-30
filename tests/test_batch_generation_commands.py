import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import discord

from commands import generate_batch
from services.batch_generation import BatchGenerationRequest, BatchJobOwnershipError
from services.sd_control import LoraSelection


def _interaction(user_id: int = 123):
    interaction = SimpleNamespace()
    interaction.user = SimpleNamespace(id=user_id)
    interaction.response = SimpleNamespace(
        defer=AsyncMock(),
        send_message=AsyncMock(),
    )
    interaction.followup = SimpleNamespace(send=AsyncMock())
    return interaction


class BatchCommandSchemaTests(unittest.TestCase):
    def test_group_exposes_only_the_requested_batch_operations(self):
        client = discord.Client(intents=discord.Intents.none())
        client.tree = discord.app_commands.CommandTree(client)
        generate_batch.setup(client)
        group = next(command for command in client.tree.get_commands() if command.name == "generate_batch")
        self.assertEqual(
            {"create", "defaults", "repeat", "status", "cancel"},
            {command.name for command in group.commands},
        )

    def test_setup_is_idempotent(self):
        client = discord.Client(intents=discord.Intents.none())
        client.tree = discord.app_commands.CommandTree(client)
        generate_batch.setup(client)
        generate_batch.setup(client)
        self.assertEqual(
            1,
            sum(command.name == "generate_batch" for command in client.tree.get_commands()),
        )


class BatchCommandCallbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.request = BatchGenerationRequest(
            requested_by=123,
            character="test-character",
            total_count=2,
            images_per_prompt=1,
            model_key="animagine",
            request_text="standing",
            rating="safe",
        )

    @staticmethod
    def _callback(name):
        client = discord.Client(intents=discord.Intents.none())
        client.tree = discord.app_commands.CommandTree(client)
        generate_batch.setup(client)
        group = next(command for command in client.tree.get_commands() if command.name == "generate_batch")
        return next(command.callback for command in group.commands if command.name == name)

    async def test_create_persists_specs_and_schedules_worker(self):
        callback = self._callback("create")
        interaction = _interaction()
        job = {
            "job_id": "job-1",
            "status": "queued",
            "total_count": 2,
            "completed_count": 0,
            "failed_count": 0,
            "current_batch_id": "",
            "error": "",
        }
        with (
            patch.object(generate_batch, "is_pc_control_allowed", return_value=True),
            patch.object(generate_batch, "get_last_batch_settings", return_value={}),
            patch.object(generate_batch, "_resolve_model_and_loras", return_value=("animagine", ())),
            patch.object(generate_batch, "_resolve_rating", return_value="safe"),
            patch.object(generate_batch.manager, "create_job", return_value="job-1") as create_job,
            patch.object(generate_batch, "_build_specs") as build_specs,
            patch.object(generate_batch, "start_job", return_value=True) as start_job,
            patch.object(generate_batch.manager, "get_job_for_user", return_value=job),
            patch.object(generate_batch, "_save_last_batch_request"),
        ):
            await callback(interaction, "test-character", total_count=2)

        create_job.assert_called_once()
        build_specs.assert_called_once()
        start_job.assert_called_once_with("job-1", 123)
        interaction.response.defer.assert_awaited_once()
        self.assertIn("job-1", interaction.followup.send.await_args.args[0])

    async def test_status_and_cancel_use_owner_checked_manager_apis(self):
        status_callback = self._callback("status")
        cancel_callback = self._callback("cancel")
        job = {
            "job_id": "job-1",
            "status": "running",
            "total_count": 2,
            "completed_count": 1,
            "failed_count": 0,
            "current_batch_id": "batch-1",
            "error": "",
        }
        interaction = _interaction()
        with patch.object(generate_batch, "is_pc_control_allowed", return_value=True), patch.object(
            generate_batch.manager, "get_job_for_user", return_value=job
        ) as get_job:
            await status_callback(interaction, "job-1")
        get_job.assert_called_once_with("job-1", 123)
        self.assertIn("1 / 2", interaction.followup.send.await_args.args[0])

        interaction = _interaction()
        with (
            patch.object(generate_batch, "is_pc_control_allowed", return_value=True),
            patch.object(generate_batch.manager, "cancel_job", return_value=(True, "cancel accepted")) as cancel,
            patch.object(generate_batch._worker, "request_stop") as request_stop,
        ):
            await cancel_callback(interaction, "job-1")
        cancel.assert_called_once_with("job-1", 123)
        request_stop.assert_called_once_with("job-1")
        self.assertIn("cancel accepted", interaction.followup.send.await_args.args[0])

    async def test_foreign_job_is_rejected_by_owner_checked_status_api(self):
        callback = self._callback("status")
        interaction = _interaction(999)
        with (
            patch.object(generate_batch, "is_pc_control_allowed", return_value=True),
            patch.object(
                generate_batch.manager,
                "get_job_for_user",
                side_effect=BatchJobOwnershipError("他のユーザーのjobは閲覧できません。"),
            ),
        ):
            await callback(interaction, "foreign-job")
        self.assertIn("他のユーザー", interaction.followup.send.await_args.args[0])

    async def test_repeat_copies_request_into_new_specs_and_starts_worker(self):
        callback = self._callback("repeat")
        interaction = _interaction()
        with (
            patch.object(generate_batch, "is_pc_control_allowed", return_value=True),
            patch.object(generate_batch.manager, "get_request_for_user", side_effect=[self.request, self.request]),
            patch.object(generate_batch.manager, "repeat_job", return_value="job-2") as repeat_job,
            patch.object(generate_batch, "_build_specs") as build_specs,
            patch.object(generate_batch, "start_job", return_value=True) as start_job,
            patch.object(generate_batch, "_save_last_batch_request"),
            patch.object(
                generate_batch.manager,
                "get_job_for_user",
                return_value={
                    "job_id": "job-2",
                    "status": "queued",
                    "total_count": 2,
                    "completed_count": 0,
                    "failed_count": 0,
                    "current_batch_id": "",
                    "error": "",
                },
            ),
        ):
            await callback(interaction, "job-1")
        repeat_job.assert_called_once_with("job-1", 123)
        build_specs.assert_called_once()
        start_job.assert_called_once_with("job-2", 123)

    async def test_defaults_persist_for_next_batch(self):
        callback = self._callback("defaults")
        interaction = _interaction()
        with (
            patch.object(generate_batch, "is_pc_control_allowed", return_value=True),
            patch.object(generate_batch, "get_last_batch_settings", return_value={"total_count": 4}),
            patch.object(generate_batch, "_resolve_model_and_loras", return_value=("animagine", ())),
            patch.object(generate_batch, "save_last_batch_settings") as save,
        ):
            await callback(interaction, "saved-character")
        save.assert_called_once()
        self.assertEqual(123, save.call_args.args[0])
        self.assertEqual("saved-character", save.call_args.args[1]["character"])
        self.assertEqual(4, save.call_args.args[1]["total_count"])


class BatchWorkerLifecycleHookTests(unittest.IsolatedAsyncioTestCase):
    def test_startup_recovery_delegates_to_persisted_job_api(self):
        with patch.object(generate_batch, "recover_persisted_jobs", return_value=3) as recover:
            self.assertEqual(3, generate_batch.recover_incomplete_jobs())
        recover.assert_called_once_with()

    async def test_shutdown_requests_worker_stop_and_waits_for_tasks(self):
        task = asyncio.create_task(asyncio.sleep(0))
        with patch.object(generate_batch._worker, "shutdown") as stop:
            generate_batch._worker_tasks["job-1"] = task
            await generate_batch.shutdown()
        stop.assert_called_once()
        self.assertEqual({}, generate_batch._worker_tasks)


if __name__ == "__main__":
    unittest.main()
