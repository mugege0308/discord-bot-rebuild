import math
import unittest
from unittest.mock import Mock, patch

from services.batch_generation import BatchGenerationRequest, BatchSpec
from services.batch_generation_adapter import (
    BatchImageConfigurationError,
    BatchImageGenerationError,
    build_batch_image_options,
    generate_one_batch_image,
)
from services.sd_control import LoraSelection


class BatchGenerationAdapterTests(unittest.TestCase):
    @staticmethod
    def request(**overrides):
        values = {
            "requested_by": 123,
            "character": "test-character",
            "total_count": 3,
            "images_per_prompt": 1,
            "model_key": "animagine",
            "request_text": "a test batch",
            "rating": "safe",
            "base_prompt": "fallback prompt",
            "base_negative_prompt": "fallback negative",
            "generation_config": {
                "width": 768,
                "height": 1024,
                "steps": 24,
                "cfg": 5.5,
                "sampler": "euler_ancestral",
                "seed": 100,
            },
        }
        values.update(overrides)
        return BatchGenerationRequest(**values)

    @staticmethod
    def spec(**overrides):
        values = {
            "batch_id": "batch-1",
            "job_id": "job-1",
            "prompt": "1girl, test-character",
            "negative_prompt": "bad anatomy",
            "character": "test-character",
            "requested_count": 3,
            "variation_index": 2,
        }
        values.update(overrides)
        return BatchSpec(**values)

    @staticmethod
    def loras():
        return (
            LoraSelection("char", "char.safetensors", "character", "Character", "char trigger", 0.7, "sdxl"),
            LoraSelection("style", "style.safetensors", "style", "Style", "style trigger", 0.8, "sdxl"),
            LoraSelection("extra", "extra.safetensors", "additional", "Additional", "extra trigger", 0.4, "sdxl"),
        )

    def test_no_lora_builds_one_image_without_reference_or_pose(self):
        options = build_batch_image_options(self.request(), self.spec())
        self.assertEqual(1, options.count)
        self.assertEqual(101, options.seed)
        self.assertEqual((), options.loras)
        self.assertEqual("none", options.reference_mode)
        self.assertIsNone(options.reference_image_bytes)
        self.assertIsNone(options.pose_reference_bytes)
        self.assertFalse(options.gallery_metadata["selection_ai"])
        self.assertEqual(
            frozenset({"width", "height", "steps", "cfg", "sampler"}),
            options.explicit_generation_fields,
        )

    def test_three_lora_slots_are_resolved_and_triggers_are_deduplicated(self):
        loras = self.loras()
        request = self.request(loras=loras)
        spec = self.spec(loras=loras, prompt="style trigger, existing text")
        with patch(
            "services.batch_generation_adapter.resolve_saved_lora_selections",
            return_value=loras,
        ):
            options = build_batch_image_options(request, spec)
        self.assertEqual(loras, options.loras)
        self.assertEqual(
            "char trigger, extra trigger, solo, safe, style trigger, existing text, masterpiece, high score, great score, absurdres",
            options.prompt,
        )
        self.assertEqual(3, len(options.gallery_metadata["loras"]))

    def test_model_family_mismatch_is_a_configuration_error(self):
        with patch(
            "services.batch_generation_adapter.resolve_saved_lora_selections",
            side_effect=ValueError("モデル系統が不一致です"),
        ):
            with self.assertRaises(BatchImageConfigurationError) as raised:
                build_batch_image_options(self.request(loras=self.loras()), self.spec())
        self.assertEqual("job-1", raised.exception.job_id)
        self.assertIn("不一致", str(raised.exception))

    def test_normal_and_adult_rating_validation(self):
        build_batch_image_options(self.request(rating="safe"), self.spec())
        build_batch_image_options(
            self.request(model_key="animagine_adult", rating="explicit"),
            self.spec(),
        )
        with self.assertRaises(BatchImageConfigurationError):
            build_batch_image_options(self.request(rating="explicit"), self.spec())

    def test_seed_minus_one_remains_backend_random_and_fixed_seed_advances(self):
        random_request = self.request(generation_config={"seed": -1})
        first = build_batch_image_options(random_request, self.spec(), image_serial=0)
        second = build_batch_image_options(random_request, self.spec(), image_serial=1)
        self.assertEqual(-1, first.seed)
        self.assertEqual(-1, second.seed)
        self.assertNotEqual(first.gallery_metadata["image_serial"], second.gallery_metadata["image_serial"])

        fixed = build_batch_image_options(self.request(), self.spec(), image_serial=7)
        self.assertEqual(107, fixed.seed)
        self.assertEqual(107, fixed.gallery_metadata["seed"])

    def test_metadata_contains_job_batch_variation_and_image_identity(self):
        options = build_batch_image_options(self.request(), self.spec(), image_index=2, image_serial=8)
        self.assertEqual(
            {
                "job_id": "job-1",
                "batch_id": "batch-1",
                "variation_index": 2,
                "image_index": 2,
                "image_index_in_batch": 2,
                "image_serial": 8,
            },
            {key: options.gallery_metadata[key] for key in (
                "job_id", "batch_id", "variation_index", "image_index",
                "image_index_in_batch", "image_serial",
            )},
        )

    def test_character_and_request_text_are_explicit_prompt_fallback(self):
        options = build_batch_image_options(
            self.request(base_prompt="", request_text="standing outdoors"),
            self.spec(prompt=""),
        )
        self.assertEqual(
            "solo, safe, test-character, standing outdoors, masterpiece, high score, great score, absurdres",
            options.prompt,
        )

    def test_generation_calls_existing_backend_once_with_count_one(self):
        generated = object()
        generator = Mock(return_value=[generated])
        result = generate_one_batch_image(
            self.request(),
            self.spec(),
            generator=generator,
        )
        self.assertIs(generated, result)
        generator.assert_called_once()
        self.assertEqual(1, generator.call_args.args[0].count)
        self.assertEqual("none", generator.call_args.args[0].reference_mode)
        self.assertIsNone(generator.call_args.args[0].pose_reference_bytes)

    def test_generation_error_is_scoped_to_one_image(self):
        generator = Mock(side_effect=RuntimeError("ComfyUI unavailable"))
        with self.assertRaises(BatchImageGenerationError) as raised:
            generate_one_batch_image(self.request(), self.spec(), generator=generator)
        self.assertEqual("job-1", raised.exception.job_id)
        self.assertEqual("batch-1", raised.exception.batch_id)
        self.assertEqual(2, raised.exception.variation_index)
        self.assertEqual(1, raised.exception.image_index)
        self.assertIn("ComfyUI", str(raised.exception))

    def test_invalid_generation_setting_is_not_silently_clamped(self):
        with self.assertRaises(BatchImageConfigurationError):
            build_batch_image_options(
                self.request(generation_config={"cfg": math.inf}),
                self.spec(),
            )


if __name__ == "__main__":
    unittest.main()
