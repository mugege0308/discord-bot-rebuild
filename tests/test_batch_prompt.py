import unittest
from unittest.mock import Mock, patch

from commands import generate_batch
from services.batch_generation import BatchGenerationRequest, BatchSpec
from services.batch_prompt import prepare_batch_prompt
from services.shogou_prompt_designer import ShogouPromptResult


def _request(**overrides):
    values = {
        "requested_by": 123,
        "character": "かわいい女の子",
        "total_count": 1,
        "images_per_prompt": 1,
        "model_key": "animagine",
        "request_text": "simple safe portrait, plain background",
        "rating": "safe",
    }
    values.update(overrides)
    return BatchGenerationRequest(**values)


class BatchPromptPreparationTests(unittest.TestCase):
    @patch("services.batch_prompt.generate_shogou_prompt_result")
    def test_standalone_batch_uses_prompt_designer_for_unregistered_text(self, generate):
        generate.return_value = ShogouPromptResult(
            positive="1girl, solo, portrait, simple background",
            negative="low quality, bad anatomy",
            summary="portrait",
        )

        result = prepare_batch_prompt(_request())

        self.assertIn("portrait", result.positive)
        self.assertNotEqual("かわいい女の子, simple safe portrait, plain background", result.positive)
        self.assertTrue(result.negative)
        generate.assert_called_once_with(
            "かわいい女の子, simple safe portrait, plain background",
            123,
            None,
            "animagine",
            "safe",
            enable_generic_danbooru=True,
        )

    @patch("services.batch_prompt.generate_shogou_prompt_result")
    def test_unregistered_character_resolution_falls_back_to_free_text(self, generate):
        generate.side_effect = [
            ShogouPromptResult(
                positive="",
                negative="",
                character_unresolved_reason="no_search_intent",
            ),
            ShogouPromptResult(
                positive="1girl, cute, portrait, simple background",
                negative="low quality, bad anatomy",
            ),
        ]

        prepared = prepare_batch_prompt(_request())

        self.assertEqual("1girl, cute, portrait, simple background", prepared.positive)
        self.assertEqual("low quality, bad anatomy", prepared.negative)
        self.assertEqual(2, generate.call_count)
        self.assertTrue(generate.call_args_list[0].kwargs["enable_generic_danbooru"])
        self.assertFalse(generate.call_args_list[1].kwargs["enable_generic_danbooru"])

    @patch("services.batch_prompt.generate_shogou_prompt_result")
    def test_character_only_free_text_is_allowed(self, generate):
        generate.return_value = ShogouPromptResult(
            positive="1girl, cute",
            negative="low quality",
        )

        prepared = prepare_batch_prompt(_request(character="かわいい女の子", request_text=""))

        self.assertEqual("1girl, cute", prepared.positive)
        generate.assert_called_once()

    @patch("services.batch_prompt.generate_shogou_prompt_result")
    def test_request_without_character_is_allowed(self, generate):
        generate.return_value = ShogouPromptResult(
            positive="portrait, simple background",
            negative="low quality",
        )

        prepared = prepare_batch_prompt(_request(character="", request_text="simple safe portrait"))

        self.assertEqual("portrait, simple background", prepared.positive)
        generate.assert_called_once()

    @patch("services.batch_prompt.generate_shogou_prompt_result")
    def test_registered_character_result_keeps_confirmed_tags_and_negative(self, generate):
        result = ShogouPromptResult(
            positive="castorice, long hair, purple eyes, portrait",
            negative="low quality, malformed hands",
            summary="registered character",
        )
        generate.return_value = result

        prepared = prepare_batch_prompt(_request(character="Castorice"))

        self.assertIn("long hair", prepared.positive)
        self.assertIn("purple eyes", prepared.positive)
        self.assertIn("malformed hands", prepared.negative)

    @patch("services.batch_prompt.generate_shogou_prompt_result")
    def test_existing_resolved_prompt_is_not_generated_twice(self, generate):
        prepared = prepare_batch_prompt(
            _request(
                base_prompt="1girl, masterpiece, portrait",
                base_negative_prompt="low quality, bad anatomy",
            )
        )

        self.assertEqual("1girl, masterpiece, portrait", prepared.positive)
        self.assertEqual("low quality, bad anatomy", prepared.negative)
        generate.assert_not_called()

    @patch("services.batch_prompt.generate_shogou_prompt_result")
    def test_empty_negative_uses_backend_default(self, generate):
        generate.return_value = ShogouPromptResult(positive="1girl, portrait", negative="")

        prepared = prepare_batch_prompt(_request())

        self.assertTrue(prepared.negative)


class BatchSpecPromptIntegrationTests(unittest.TestCase):
    @patch.object(generate_batch, "prepare_batch_prompt")
    def test_build_specs_persists_shared_prompt_result(self, prepare):
        prepare.return_value = ShogouPromptResult(
            positive="1girl, confirmed appearance, portrait",
            negative="low quality, bad anatomy",
        )
        create_batch = Mock(return_value="batch-1")
        request = _request(total_count=2, images_per_prompt=1)

        with patch.object(generate_batch.manager, "create_batch", create_batch):
            generate_batch._build_specs("job-1", request, checkpoint="checkpoint.safetensors")

        self.assertEqual(2, create_batch.call_count)
        for call in create_batch.call_args_list:
            self.assertIn("confirmed appearance", call.kwargs["prompt"])
            self.assertEqual("low quality, bad anatomy", call.kwargs["negative_prompt"])
        prepare.assert_called_once_with(request)


if __name__ == "__main__":
    unittest.main()
