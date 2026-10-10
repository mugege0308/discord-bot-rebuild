import json
import hashlib
from io import BytesIO
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

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


def _png_bytes(width: int, height: int, color: tuple[int, int, int] = (40, 80, 120)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


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
    def test_shared_root_reuses_matching_source_and_writes_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "local"
            shared = root / "shared"
            local.mkdir()
            shared.mkdir()
            source = shared / "Comfy_00001.png"
            source_bytes = _png_bytes(768, 1024, (20, 40, 60))
            source.write_bytes(source_bytes)
            source_hash = hashlib.sha256(source_bytes).hexdigest()
            response = SimpleNamespace(content=source_bytes, raise_for_status=lambda: None)
            history = {"outputs": {"19": {"images": [{"filename": source.name, "type": "output", "subfolder": ""}]}}}
            with (
                mock.patch.object(sd_control, "OUTPUT_DIR", local),
                mock.patch.object(sd_control.requests, "get", return_value=response),
            ):
                sd_control._download_comfy_images(
                    history,
                    1,
                    gallery_output_root=shared,
                    gallery_metadata={"job_id": "job-shared", "width": 768, "height": 1024},
                    prompt_id="prompt-shared",
                )

            self.assertEqual([source], list(shared.glob("*.png")))
            self.assertEqual(source_hash, hashlib.sha256(source.read_bytes()).hexdigest())
            metadata = json.loads(source.with_name(source.name + ".metadata.json").read_text(encoding="utf-8"))
            self.assertEqual("job-shared", metadata["job_id"])
            self.assertEqual("prompt-shared", metadata["prompt_id"])
            self.assertEqual(source.name, metadata["gallery_filename"])
            self.assertEqual({"width": 768, "height": 1024}, metadata["gallery_dimensions"])

    def test_same_root_hash_mismatch_uses_collision_safe_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "local"
            shared = root / "shared"
            local.mkdir()
            shared.mkdir()
            source = shared / "Comfy_00001.png"
            old_bytes = _png_bytes(768, 1024, (1, 2, 3))
            new_bytes = _png_bytes(768, 1024, (4, 5, 6))
            source.write_bytes(old_bytes)
            response = SimpleNamespace(content=new_bytes, raise_for_status=lambda: None)
            history = {"outputs": {"19": {"images": [{"filename": source.name, "type": "output", "subfolder": ""}]}}}
            with (
                mock.patch.object(sd_control, "OUTPUT_DIR", local),
                mock.patch.object(sd_control.requests, "get", return_value=response),
            ):
                sd_control._download_comfy_images(
                    history,
                    1,
                    gallery_output_root=shared,
                    gallery_metadata={"job_id": "job-mismatch", "width": 768, "height": 1024},
                    prompt_id="prompt-mismatch",
                )

            imported = [path for path in shared.glob("*.png") if path != source]
            self.assertEqual(1, len(imported))
            self.assertEqual(old_bytes, source.read_bytes())
            self.assertEqual(new_bytes, imported[0].read_bytes())

    def test_external_source_root_uses_collision_safe_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "local"
            source_root = root / "comfy"
            gallery = root / "gallery"
            local.mkdir()
            source_root.mkdir()
            gallery.mkdir()
            source = source_root / "Comfy_00001.png"
            source_bytes = _png_bytes(768, 1024)
            source.write_bytes(source_bytes)
            response = SimpleNamespace(content=source_bytes, raise_for_status=lambda: None)
            history = {"outputs": {"19": {"images": [{"filename": source.name, "type": "output", "subfolder": ""}]}}}
            with (
                mock.patch.object(sd_control, "OUTPUT_DIR", local),
                mock.patch.object(sd_control.requests, "get", return_value=response),
            ):
                sd_control._download_comfy_images(
                    history,
                    1,
                    gallery_output_root=gallery,
                    gallery_metadata={"job_id": "job-external", "width": 768, "height": 1024},
                    prompt_id="prompt-external",
                )

            imported = list(gallery.glob("*.png"))
            self.assertEqual(1, len(imported))
            self.assertEqual(source_bytes, imported[0].read_bytes())

    def test_shared_root_retry_does_not_create_duplicate_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "local"
            shared = root / "shared"
            local.mkdir()
            shared.mkdir()
            source = shared / "Comfy_00001.png"
            source_bytes = _png_bytes(768, 1024)
            source.write_bytes(source_bytes)
            response = SimpleNamespace(content=source_bytes, raise_for_status=lambda: None)
            history = {"outputs": {"19": {"images": [{"filename": source.name, "type": "output", "subfolder": ""}]}}}
            with (
                mock.patch.object(sd_control, "OUTPUT_DIR", local),
                mock.patch.object(sd_control.requests, "get", return_value=response),
            ):
                for _ in range(2):
                    sd_control._download_comfy_images(
                        history,
                        1,
                        gallery_output_root=shared,
                        gallery_metadata={"job_id": "job-retry", "width": 768, "height": 1024},
                        prompt_id="prompt-retry",
                    )

            self.assertEqual([source], list(shared.glob("*.png")))
            self.assertEqual(1, len(list(shared.glob("*.metadata.json"))))

    def test_comfy_download_imports_collision_safe_authoritative_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "local"
            authoritative = root / "authoritative"
            local.mkdir()
            authoritative.mkdir()
            old = authoritative / "Comfy_00001.png"
            old_bytes = _png_bytes(1024, 1024, (1, 2, 3))
            old.write_bytes(old_bytes)
            response = SimpleNamespace(content=_png_bytes(768, 1024), raise_for_status=lambda: None)
            history = {"outputs": {"19": {"images": [{"filename": "Comfy_00001.png", "type": "output", "subfolder": ""}]}}}
            with (
                mock.patch.object(sd_control, "OUTPUT_DIR", local),
                mock.patch.object(sd_control.requests, "get", return_value=response),
            ):
                paths = sd_control._download_comfy_images(
                    history,
                    1,
                    gallery_output_root=authoritative,
                    gallery_metadata={
                        "job_id": "job-1",
                        "width": 768,
                        "height": 1024,
                        "reference_mode": "pose",
                        "requested_seed": -1,
                    },
                    prompt_id="prompt-1",
                    actual_seed=12345,
                )

            self.assertEqual(1, len(paths))
            imported = [
                path
                for path in authoritative.glob("*.png")
                if path.name != old.name
            ]
            self.assertEqual(1, len(imported))
            imported_path = imported[0]
            sidecar = imported_path.with_name(imported_path.name + ".metadata.json")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual("job-1", metadata["job_id"])
            self.assertEqual(-1, metadata["requested_seed"])
            self.assertEqual(12345, metadata["actual_seed"])
            self.assertEqual(12345, metadata["seed"])
            self.assertEqual(imported_path.name, metadata["gallery_filename"])
            self.assertEqual({"width": 768, "height": 1024}, metadata["gallery_dimensions"])
            self.assertEqual(old_bytes, old.read_bytes())

    def test_dimension_mismatch_does_not_import_authoritative_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "local"
            authoritative = root / "authoritative"
            local.mkdir()
            authoritative.mkdir()
            response = SimpleNamespace(content=b"not-a-png", raise_for_status=lambda: None)
            history = {"outputs": {"19": {"images": [{"filename": "Comfy_00001.png", "type": "output", "subfolder": ""}]}}}
            with (
                mock.patch.object(sd_control, "OUTPUT_DIR", local),
                mock.patch.object(sd_control.requests, "get", return_value=response),
            ):
                with self.assertRaises(RuntimeError):
                    sd_control._download_comfy_images(
                        history,
                        1,
                        gallery_output_root=authoritative,
                        gallery_metadata={"job_id": "job-2", "width": 768, "height": 1024},
                        prompt_id="prompt-2",
                    )
            self.assertEqual([], list(authoritative.iterdir()))

    def test_retry_never_overwrites_first_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "local"
            authoritative = root / "authoritative"
            local.mkdir()
            authoritative.mkdir()
            history = {"outputs": {"19": {"images": [{"filename": "Comfy_00001.png", "type": "output", "subfolder": ""}]}}}
            responses = iter([
                SimpleNamespace(content=b"first", raise_for_status=lambda: None),
                SimpleNamespace(content=b"second", raise_for_status=lambda: None),
            ])
            with (
                mock.patch.object(sd_control, "OUTPUT_DIR", local),
                mock.patch.object(sd_control.requests, "get", side_effect=lambda *args, **kwargs: next(responses)),
            ):
                sd_control._download_comfy_images(
                    history,
                    1,
                    gallery_output_root=authoritative,
                    gallery_metadata={"job_id": "job-3"},
                    prompt_id="prompt-3",
                )
                sd_control._download_comfy_images(
                    history,
                    1,
                    gallery_output_root=authoritative,
                    gallery_metadata={"job_id": "job-3"},
                    prompt_id="prompt-3",
                )
            imported = sorted(authoritative.glob("*.png"))
            self.assertEqual(2, len(imported))
            self.assertEqual(b"first", imported[0].read_bytes())
            self.assertEqual(b"second", imported[1].read_bytes())
            self.assertNotEqual(hashlib.sha256(imported[0].read_bytes()).digest(), hashlib.sha256(imported[1].read_bytes()).digest())


if __name__ == "__main__":
    unittest.main()
