import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services import sd_control
from services.sd_control import (
    GeneratedImage,
    SdGenerationOptions,
    _extract_checkpoint_name,
    _extract_latent_dimensions,
    _extract_sampler_settings,
    _safe_workflow_file_hash,
    generate_images,
    generate_images_with_metadata,
)


def _sample_workflow() -> dict:
    return {
        "4": {
            "inputs": {"ckpt_name": "animagineXL40Opt_clear.safetensors"},
            "class_type": "CheckpointLoaderSimple",
        },
        "5": {
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
            "class_type": "EmptyLatentImage",
        },
        "6": {
            "inputs": {"text": "placeholder positive", "clip": ["4", 1]},
            "class_type": "CLIPTextEncode",
        },
        "7": {
            "inputs": {"text": "placeholder negative", "clip": ["4", 1]},
            "class_type": "CLIPTextEncode",
        },
        "10": {
            "inputs": {
                "seed": 111,
                "steps": 20,
                "cfg": 5.0,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
            "class_type": "KSampler",
        },
        "17": {
            "inputs": {"samples": ["10", 0], "vae": ["4", 2]},
            "class_type": "VAEDecode",
        },
        "19": {
            "inputs": {"filename_prefix": "test", "images": ["17", 0]},
            "class_type": "SaveImage",
        },
    }


class ExtractionHelperTestCase(unittest.TestCase):
    def test_extract_checkpoint_name_found(self):
        self.assertEqual(
            _extract_checkpoint_name(_sample_workflow()),
            "animagineXL40Opt_clear.safetensors",
        )

    def test_extract_checkpoint_name_missing_returns_none(self):
        workflow = _sample_workflow()
        del workflow["4"]
        self.assertIsNone(_extract_checkpoint_name(workflow))

    def test_extract_latent_dimensions_found(self):
        self.assertEqual(_extract_latent_dimensions(_sample_workflow()), (1024, 1024))

    def test_extract_latent_dimensions_missing_returns_none_pair(self):
        workflow = _sample_workflow()
        del workflow["5"]
        self.assertEqual(_extract_latent_dimensions(workflow), (None, None))

    def test_extract_sampler_settings_found(self):
        steps, cfg, sampler_name, scheduler = _extract_sampler_settings(_sample_workflow(), "10")
        self.assertEqual(steps, 20)
        self.assertEqual(cfg, 5.0)
        self.assertEqual(sampler_name, "euler_ancestral")
        self.assertEqual(scheduler, "normal")

    def test_extract_sampler_settings_no_node_id_returns_all_none(self):
        self.assertEqual(
            _extract_sampler_settings(_sample_workflow(), None),
            (None, None, None, None),
        )

    def test_extract_sampler_settings_missing_fields_return_none(self):
        workflow = _sample_workflow()
        del workflow["10"]["inputs"]["cfg"]
        del workflow["10"]["inputs"]["scheduler"]
        steps, cfg, sampler_name, scheduler = _extract_sampler_settings(workflow, "10")
        self.assertEqual(steps, 20)
        self.assertIsNone(cfg)
        self.assertEqual(sampler_name, "euler_ancestral")
        self.assertIsNone(scheduler)

    def test_safe_workflow_file_hash_matches_sha256_of_file_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wf.json"
            path.write_text('{"a": 1}', encoding="utf-8")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(_safe_workflow_file_hash(path), expected)

    def test_safe_workflow_file_hash_missing_file_returns_none(self):
        self.assertIsNone(_safe_workflow_file_hash(Path("does_not_exist_workflow.json")))


class GenerateImagesWithMetadataTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        tmp_path = Path(self.tmp_dir.name)

        self.workflow_file = tmp_path / "workflow.json"
        self.workflow_file.write_text(json.dumps(_sample_workflow()), encoding="utf-8")
        self.output_dir = tmp_path / "outputs"

        self._patchers = [
            mock.patch("services.sd_control._get_comfy_workflow_file", return_value=self.workflow_file),
            mock.patch("services.sd_control.OUTPUT_DIR", self.output_dir),
            mock.patch("services.sd_control._save_latest_paths"),
            mock.patch("services.sd_control._backend", return_value="comfy"),
        ]
        for patcher in self._patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        # Defensive: never let a previous test's failure leave the module
        # generation lock held, which would make every test here fail with
        # "already generating".
        if sd_control._generate_lock.locked():
            sd_control._generate_lock.release()
        self.addCleanup(self._release_lock_if_needed)

    def _release_lock_if_needed(self):
        if sd_control._generate_lock.locked():
            sd_control._generate_lock.release()

    @staticmethod
    def _mock_requests(prompt_id: str = "prompt-123"):
        history_outputs = {
            "19": {"images": [{"filename": "result.png", "type": "output", "subfolder": ""}]}
        }

        def fake_post(url, json=None, timeout=None):
            response = mock.Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {"prompt_id": prompt_id}
            return response

        def fake_get(url, params=None, timeout=None):
            response = mock.Mock()
            response.raise_for_status.return_value = None
            if url.endswith(f"/history/{prompt_id}"):
                response.json.return_value = {prompt_id: {"outputs": history_outputs}}
            elif url.endswith("/view"):
                response.content = b"\x89PNG\r\n\x1a\nfakeimagebytes"
            else:  # pragma: no cover - defensive, should not happen
                raise AssertionError(f"unexpected GET url in test: {url}")
            return response

        return fake_post, fake_get

    def test_generate_images_returns_list_of_path_unchanged(self):
        fake_post, fake_get = self._mock_requests()
        with mock.patch("services.sd_control.requests.post", side_effect=fake_post), \
             mock.patch("services.sd_control.requests.get", side_effect=fake_get), \
             mock.patch("services.sd_control.random.randint", return_value=42):
            options = SdGenerationOptions(
                prompt="1girl", negative_prompt="lowres", preserve_prompt_text=True, count=1
            )
            paths = generate_images(options)

        self.assertIsInstance(paths, list)
        self.assertEqual(len(paths), 1)
        self.assertIsInstance(paths[0], Path)
        self.assertTrue(paths[0].exists())

    def test_generate_images_with_metadata_returns_generation_conditions(self):
        fake_post, fake_get = self._mock_requests()
        with mock.patch("services.sd_control.requests.post", side_effect=fake_post), \
             mock.patch("services.sd_control.requests.get", side_effect=fake_get), \
             mock.patch("services.sd_control.random.randint", return_value=999):
            options = SdGenerationOptions(
                prompt="1girl", negative_prompt="lowres", preserve_prompt_text=True, count=1
            )
            images = generate_images_with_metadata(options)

        self.assertEqual(len(images), 1)
        image = images[0]
        self.assertIsInstance(image, GeneratedImage)
        self.assertTrue(image.path.exists())
        self.assertEqual(image.batch_index, 1)
        self.assertEqual(image.seed, 999)
        self.assertEqual(image.checkpoint_name, "animagineXL40Opt_clear.safetensors")
        self.assertEqual(image.sampler_name, "euler_ancestral")
        self.assertEqual(image.scheduler, "normal")
        self.assertEqual(image.steps, 20)
        self.assertEqual(image.cfg, 5.0)
        self.assertEqual(image.width, 1024)
        self.assertEqual(image.height, 1024)
        self.assertEqual(image.workflow_file, str(self.workflow_file))
        self.assertEqual(image.workflow_hash, hashlib.sha256(self.workflow_file.read_bytes()).hexdigest())

    def test_multiple_images_get_distinct_seeds_and_batch_indexes(self):
        fake_post, fake_get = self._mock_requests()
        seeds = iter([111, 222])
        with mock.patch("services.sd_control.requests.post", side_effect=fake_post), \
             mock.patch("services.sd_control.requests.get", side_effect=fake_get), \
             mock.patch("services.sd_control.random.randint", side_effect=lambda a, b: next(seeds)):
            options = SdGenerationOptions(
                prompt="1girl", negative_prompt="lowres", preserve_prompt_text=True, count=2
            )
            images = generate_images_with_metadata(options)

        self.assertEqual(len(images), 2)
        self.assertEqual([image.seed for image in images], [111, 222])
        self.assertEqual([image.batch_index for image in images], [1, 2])

    def test_condition_missing_from_workflow_is_stored_as_none(self):
        # No CheckpointLoaderSimple node at all in this workflow variant -
        # checkpoint_name must come back None rather than an error or guess.
        workflow = _sample_workflow()
        del workflow["4"]
        # Drop the checkpoint's outbound clip/model links so the workflow
        # still validates (validation only requires SaveImage + prompt text
        # nodes, not the checkpoint node itself).
        workflow["6"]["inputs"]["clip"] = None
        workflow["7"]["inputs"]["clip"] = None
        workflow["10"]["inputs"]["model"] = None
        self.workflow_file.write_text(json.dumps(workflow), encoding="utf-8")

        fake_post, fake_get = self._mock_requests()
        with mock.patch("services.sd_control.requests.post", side_effect=fake_post), \
             mock.patch("services.sd_control.requests.get", side_effect=fake_get), \
             mock.patch("services.sd_control.random.randint", return_value=1):
            options = SdGenerationOptions(
                prompt="1girl", negative_prompt="lowres", preserve_prompt_text=True, count=1
            )
            images = generate_images_with_metadata(options)

        self.assertIsNone(images[0].checkpoint_name)

    def test_comfy_failure_raises_and_lock_is_released(self):
        def failing_post(url, json=None, timeout=None):
            raise sd_control.requests.RequestException("boom")

        with mock.patch("services.sd_control.requests.post", side_effect=failing_post):
            options = SdGenerationOptions(
                prompt="1girl", negative_prompt="lowres", preserve_prompt_text=True, count=1
            )
            with self.assertRaises(RuntimeError):
                generate_images_with_metadata(options)

        # The generation lock must be released even on failure, so the next
        # /generate_ai call is not permanently blocked.
        self.assertFalse(sd_control._generate_lock.locked())


if __name__ == "__main__":
    unittest.main()
