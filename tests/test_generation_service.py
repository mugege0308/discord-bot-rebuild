import unittest
from pathlib import Path
from unittest.mock import Mock

from services.generation_service import (
    AppearanceResolution,
    GenerationRequest,
    GenerationService,
)
from services.sd_control import GeneratedImage, LoraSelection
from services.shogou_prompt_designer import ShogouPromptResult


def _prompt_result(
    positive: str = "1girl, portrait",
    negative: str = "low quality, bad anatomy",
    **overrides,
) -> ShogouPromptResult:
    return ShogouPromptResult(
        positive=positive,
        negative=negative,
        summary="characterized",
        **overrides,
    )


class GenerationServiceCharacterizationTests(unittest.TestCase):
    def setUp(self):
        self.prompt_resolver = Mock(return_value=_prompt_result())
        self.service = GenerationService(prompt_resolver=self.prompt_resolver)

    def test_freeform_request_keeps_existing_prompt_designer_boundary(self):
        request = GenerationRequest(
            freeform_character="かわいい女の子",
            request_text="simple safe portrait, plain background",
            model_key="animagine",
            rating="safe",
            requester_id="gallery-test-user",
            source="test",
        )

        resolved = self.service.resolve_request(request)

        self.assertEqual(
            "かわいい女の子, simple safe portrait, plain background",
            self.prompt_resolver.call_args.args[0],
        )
        self.assertIsNone(self.prompt_resolver.call_args.args[1])
        self.assertEqual("safe", self.prompt_resolver.call_args.args[4])
        self.assertEqual("1girl, portrait", resolved.prompt_result.positive)

    def test_registered_character_preset_preserves_confirmed_prompt_and_negative(self):
        confirmed = _prompt_result(
            positive="1girl, kochiya sanae, green hair",
            negative="low quality, bad anatomy, monochrome",
        )
        service = GenerationService(prompt_resolver=Mock(return_value=confirmed))
        request = GenerationRequest(
            character_preset="東風谷早苗",
            request_text="smile",
            model_key="animagine",
            rating="safe",
            source="discord",
        )

        resolved = service.resolve_request(request)

        self.assertIn("kochiya sanae", resolved.prompt_result.positive)
        self.assertIn("green hair", resolved.prompt_result.positive)
        self.assertEqual(confirmed.negative, resolved.prompt_result.negative)
        self.assertEqual("東風谷早苗, smile", service._prompt_resolver.call_args.args[0])

    def test_explicit_tags_appearance_and_request_have_separate_provenance(self):
        appearance = Mock(
            return_value=AppearanceResolution(
                input_text="黒髪ロング、赤目",
                tags=("black hair", "red eyes"),
            )
        )
        prompt_resolver = Mock(return_value=_prompt_result("1girl, front view, smile"))
        service = GenerationService(
            prompt_resolver=prompt_resolver,
            appearance_resolver=appearance,
        )
        request = GenerationRequest(
            explicit_tags=("from side", "upper body", "closed mouth"),
            appearance_text="黒髪ロング、赤目",
            request_text="白背景、少し微笑む",
            source="gallery",
        )

        resolved = service.resolve_request(request)
        positive = resolved.prompt_result.positive
        provenance = {(item.tag, item.source) for item in resolved.provenance}

        self.assertIn("from side", positive)
        self.assertIn("upper body", positive)
        self.assertIn("closed mouth", positive)
        self.assertIn("black hair", positive)
        self.assertIn("red eyes", positive)
        self.assertNotIn("front view", positive)
        self.assertIn(("from side", "user_explicit"), provenance)
        self.assertIn(("black hair", "appearance_resolver"), provenance)
        self.assertEqual("白背景、少し微笑む", prompt_resolver.call_args.args[0])
        appearance.assert_called_once_with("黒髪ロング、赤目")

    def test_unconfigured_appearance_is_not_promoted_from_free_text(self):
        unconfigured = lambda text: AppearanceResolution(
            input_text=text,
            unresolved=(text,),
            resolver="unconfigured",
        )
        service = GenerationService(appearance_resolver=unconfigured)
        request = GenerationRequest(
            freeform_character="かわいい女の子",
            appearance_text="銀髪のボブ、青い目、色白",
            request_text="portrait",
            source="gallery",
        )

        resolved = service.resolve_request(request)

        self.assertNotIn("銀髪のボブ", resolved.prompt_result.positive)
        self.assertEqual((), resolved.appearance_tags)

    def test_confirmed_character_appearance_beats_appearance_text(self):
        appearance = Mock(return_value=AppearanceResolution("青い目", ("blue eyes",)))
        service = GenerationService(
            prompt_resolver=Mock(
                return_value=_prompt_result(
                    "1girl, purple eyes",
                    confirmed_character_appearance_tags=("purple eyes",),
                )
            ),
            appearance_resolver=appearance,
        )
        request = GenerationRequest(
            freeform_character="registered character",
            appearance_text="青い目",
            request_text="portrait",
        )

        resolved = service.resolve_request(request)

        self.assertIn("purple eyes", resolved.prompt_result.positive)
        self.assertNotIn("blue eyes", resolved.prompt_result.positive)
        self.assertEqual((), resolved.appearance_tags)

    def test_explicit_appearance_beats_confirmed_and_resolved_appearance(self):
        appearance = Mock(return_value=AppearanceResolution("青い目", ("blue eyes",)))
        service = GenerationService(
            prompt_resolver=Mock(
                return_value=_prompt_result(
                    "1girl, purple eyes",
                    confirmed_character_appearance_tags=("purple eyes",),
                )
            ),
            appearance_resolver=appearance,
        )
        request = GenerationRequest(
            freeform_character="registered character",
            explicit_tags=("red eyes",),
            appearance_text="青い目",
            request_text="portrait",
        )

        resolved = service.resolve_request(request)

        self.assertIn("red eyes", resolved.prompt_result.positive)
        self.assertNotIn("purple eyes", resolved.prompt_result.positive)
        self.assertNotIn("blue eyes", resolved.prompt_result.positive)


class GenerationServicePreparationTests(unittest.TestCase):
    def test_finalization_restores_required_animagine_tags_and_default_negative(self):
        appearance = Mock(
            return_value=AppearanceResolution(
                input_text="黒髪、青い目",
                tags=("black hair", "blue eyes"),
            )
        )
        service = GenerationService(
            prompt_resolver=Mock(return_value=_prompt_result(positive="portrait", negative="  ")),
            appearance_resolver=appearance,
        )
        request = GenerationRequest(
            freeform_character="かわいい女の子",
            explicit_tags=("portrait",),
            appearance_text="黒髪、青い目",
            request_text="simple safe portrait, plain background",
            model_key="animagine",
            rating="safe",
            source="generation_api",
        )

        prepared = service.prepare_generation(request)
        positive = {token.strip() for token in prepared.options.prompt.split(",")}

        self.assertTrue(
            {
                "1girl",
                "solo",
                "safe",
                "portrait",
                "masterpiece",
                "high score",
                "great score",
                "absurdres",
                "black hair",
                "blue eyes",
            }.issubset(positive)
        )
        self.assertTrue(prepared.options.negative_prompt.strip())
        self.assertEqual(prepared.options.negative_prompt, prepared.resolved.prompt_result.negative)
        provenance = {(item.tag, item.source) for item in prepared.resolved.provenance}
        for tag in ("1girl", "solo", "safe", "masterpiece", "high score", "great score", "absurdres"):
            self.assertIn((tag, "system"), provenance)
        self.assertIn(("black hair", "appearance_resolver"), provenance)

    def test_finalization_deduplicates_existing_system_tags_and_keeps_nonempty_negative(self):
        service = GenerationService(
            prompt_resolver=Mock(
                return_value=_prompt_result(
                    positive="1girl, solo, safe, portrait, masterpiece, masterpiece, high score",
                    negative="custom negative",
                )
            )
        )
        request = GenerationRequest(
            freeform_character="かわいい女の子",
            request_text="portrait",
            model_key="animagine",
            rating="safe",
        )

        prepared = service.prepare_generation(request)
        tokens = [token.strip() for token in prepared.options.prompt.split(",")]

        self.assertEqual(len(tokens), len({token.casefold() for token in tokens}))
        self.assertEqual("custom negative", prepared.options.negative_prompt)

    def test_no_lora_safe_and_explicit_seed_are_preserved(self):
        request = GenerationRequest(
            freeform_character="かわいい女の子",
            request_text="portrait",
            model_key="animagine",
            rating="safe",
            width=768,
            height=1024,
            count=1,
            seed=925002939,
            source="test",
        )
        service = GenerationService(prompt_resolver=Mock(return_value=_prompt_result()))

        prepared = service.prepare_generation(request)

        self.assertEqual((), prepared.options.loras)
        self.assertEqual("safe", request.rating)
        self.assertEqual(925002939, prepared.options.seed)
        self.assertEqual(768, prepared.options.width)
        self.assertEqual(1024, prepared.options.height)
        provenance = {(item.tag, item.source) for item in prepared.resolved.provenance}
        self.assertIn(("1girl", "system"), provenance)

    def test_lora_selection_and_trigger_are_resolved_by_existing_backend_helpers(self):
        lora = LoraSelection(
            "komagome_style_v1",
            "こまごめピペット\\komagome_style_v1.safetensors",
            "style",
            "komagome_style_v1",
            "komagome_style",
            0.5,
            "sdxl",
        )
        resolver = Mock(return_value=(lora,))
        request = GenerationRequest(
            freeform_character="かわいい女の子",
            request_text="portrait",
            loras=(
                {"key": "komagome_style_v1", "lora_type": "style", "strength": 0.5},
            ),
            seed=-1,
            source="test",
        )
        service = GenerationService(prompt_resolver=Mock(return_value=_prompt_result()),)

        prepared = service.prepare_generation(request, lora_resolver=resolver)

        self.assertEqual((lora,), prepared.options.loras)
        self.assertTrue(prepared.options.prompt.startswith("komagome_style"))
        provenance = {(item.tag, item.source) for item in prepared.resolved.provenance}
        self.assertIn(("komagome_style", "lora_trigger"), provenance)
        resolver.assert_called_once()

    def test_submit_separates_requested_and_actual_seed(self):
        request = GenerationRequest(
            freeform_character="かわいい女の子",
            request_text="portrait",
            seed=-1,
            source="test",
        )
        image = GeneratedImage(path=Path("generated.png"), batch_index=1, seed=925002939)
        generator = Mock(return_value=[image])
        service = GenerationService(prompt_resolver=Mock(return_value=_prompt_result()))

        result = service.submit_generation(request, generator=generator)

        self.assertEqual(-1, result.requested_seed)
        self.assertEqual((925002939,), result.actual_seeds)
        self.assertEqual(925002939, result.actual_seed)
        generator.assert_called_once()


if __name__ == "__main__":
    unittest.main()
