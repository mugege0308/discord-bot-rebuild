import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from commands import prompt as prompt_commands
from services import sd_control
from services.gallery_review_queue import GalleryReviewQueue
from services.image_selection import ImageSelectionResult
from services.sd_control import GeneratedImage
from services.shogou_prompt_designer import ShogouPromptResult


def _result() -> ShogouPromptResult:
    return ShogouPromptResult(
        positive="1girl, safe, masterpiece, high score, great score, absurdres",
        negative="lowres, bad anatomy",
        summary="gallery test",
        assumptions=(),
    )


class GalleryQueueLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_is_idempotent_and_shutdown_cancels_worker(self):
        queue = GalleryReviewQueue()
        await queue.start()
        first = queue._worker_task
        await queue.start()
        self.assertIs(first, queue._worker_task)
        await queue.shutdown()
        self.assertIsNone(queue._worker_task)

    async def test_bot_setup_and_close_manage_gallery_queue_best_effort(self):
        import main

        client = main.SecretaryClient()
        with (
            mock.patch.dict("os.environ", {"DISCORD_GUILD_ID": ""}, clear=False),
            mock.patch.object(main.lora_job.job_manager, "recover", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(main, "start_gallery_review_queue", new=mock.AsyncMock()) as start_queue,
            mock.patch.object(main, "shutdown_gallery_review_queue", new=mock.AsyncMock()) as stop_queue,
            mock.patch.object(client.tree, "sync", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(main.claude_remote_manager, "cleanup", new=mock.AsyncMock()),
            mock.patch.object(main.voice, "cleanup", new=mock.AsyncMock()),
            mock.patch.object(main.lora_job.job_manager, "shutdown", new=mock.AsyncMock()),
            mock.patch("discord.Client.close", new=mock.AsyncMock()),
        ):
            await client.setup_hook()
            await client.close()

        start_queue.assert_awaited_once()
        stop_queue.assert_awaited_once()


class GalleryGenerationHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_candidates_enqueue_one_scope_and_preserve_delivery_order(self):
        interaction = mock.MagicMock()
        interaction.user.id = 321
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"candidate_{index}.png" for index in range(1, 4)]
            for index, path in enumerate(paths, start=1):
                path.write_bytes(f"image-{index}".encode())
            images = [GeneratedImage(path=path, batch_index=index) for index, path in enumerate(paths, 1)]
            selection = ImageSelectionResult(
                selected_paths=[paths[1]],
                used_selection=True,
                candidate_scores={paths[0]: 0.1, paths[1]: 0.9, paths[2]: 0.5},
            )
            with (
                mock.patch.object(prompt_commands, "generate_images_with_metadata", return_value=images),
                mock.patch.object(prompt_commands, "select_generated_images", return_value=selection),
                mock.patch.object(prompt_commands, "save_generation_history_detailed", return_value=None),
                mock.patch.object(prompt_commands, "_send_image_files", new_callable=mock.AsyncMock) as send,
                mock.patch.object(prompt_commands, "get_gallery_images_root", return_value=None),
                mock.patch("services.gallery_review_queue.enqueue_gallery_review") as enqueue,
            ):
                await prompt_commands._generate_images_from_shogou_result(
                    interaction,
                    _result(),
                    label_prefix="gallery-test",
                    selection_enabled=True,
                    selection_generate_count=3,
                    auto_review=True,
                )

        self.assertEqual(send.call_args.args[2], [paths[1], paths[2], paths[0]])
        enqueue.assert_called_once()
        self.assertEqual(3, enqueue.call_args.args[1])
        self.assertTrue(enqueue.call_args.args[0])

    async def test_reference_metadata_is_scoped_without_changing_selection_policy(self):
        interaction = mock.MagicMock()
        interaction.user.id = 321
        image = GeneratedImage(path=Path("pose-result.png"), batch_index=1)
        with (
            mock.patch.object(prompt_commands, "generate_images_with_metadata", return_value=[image]) as generate,
            mock.patch.object(prompt_commands, "save_generation_history_detailed", return_value=None),
            mock.patch.object(prompt_commands, "_send_image_files", new_callable=mock.AsyncMock),
            mock.patch.object(prompt_commands, "get_gallery_images_root", return_value=None),
            mock.patch("services.gallery_review_queue.enqueue_gallery_review"),
        ):
            await prompt_commands._generate_images_from_shogou_result(
                interaction,
                _result(),
                label_prefix="gallery-pose-test",
                pose_reference_bytes=b"pose",
                reference_mode="pose",
                pose_strength=0.8,
                selection_enabled=True,
                auto_review=True,
            )

        options = generate.call_args.args[0]
        self.assertEqual("pose", options.gallery_metadata["reference_mode"])
        self.assertTrue(options.gallery_metadata["reference_used"])
        self.assertEqual(0.8, options.gallery_metadata["pose_strength"])


class GallerySidecarTests(unittest.TestCase):
    def test_comfy_download_writes_local_and_authoritative_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "local"
            authoritative = root / "authoritative"
            local.mkdir()
            authoritative.mkdir()
            (authoritative / "Comfy_00001.png").write_bytes(b"existing")
            response = SimpleNamespace(content=b"generated", raise_for_status=lambda: None)
            history = {"outputs": {"19": {"images": [{"filename": "Comfy_00001.png", "type": "output", "subfolder": ""}]}}}
            with (
                mock.patch.object(sd_control, "OUTPUT_DIR", local),
                mock.patch.object(sd_control.requests, "get", return_value=response),
            ):
                paths = sd_control._download_comfy_images(
                    history,
                    1,
                    gallery_output_root=authoritative,
                    gallery_metadata={"job_id": "job-1", "reference_mode": "pose"},
                    prompt_id="prompt-1",
                )

            self.assertEqual(1, len(paths))
            sidecar = paths[0].with_name(paths[0].name + ".metadata.json")
            self.assertEqual("job-1", json.loads(sidecar.read_text(encoding="utf-8"))["job_id"])
            self.assertTrue((authoritative / "Comfy_00001.png.metadata.json").is_file())


if __name__ == "__main__":
    unittest.main()
