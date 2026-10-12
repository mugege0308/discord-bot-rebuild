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

    def test_multi_person_framing_is_forwarded_to_prompt_designer(self):
        request = GenerationRequest(
            explicit_tags=("1girl", "1boy", "hetero"),
            request_text="hetero",
            multi_person_framing="both_faces",
            model_key="animagine",
            rating="safe",
            source="gallery",
        )

        self.service.resolve_request(request)

        self.assertEqual(
            "both_faces",
            self.prompt_resolver.call_args.kwargs["multi_person_framing"],
        )

    def test_resolve_request_applies_multi_person_conflicts_for_discord_path(self):
        auto_framing = (
            "1girl, 1boy, hetero, standing, full body, feet visible, "
            "portrait, upper body, close-up, face focus, solo"
        )
        service = GenerationService(prompt_resolver=Mock(return_value=_prompt_result(auto_framing)))
        for interaction in ("hetero", "hetero, clothed_sex", "hetero, clothed_sex, leg_lock"):
            with self.subTest(interaction=interaction):
                request = GenerationRequest(
                    request_text=f"1girl, 1boy, {interaction}, standing, full body, feet visible",
                    model_key="animagine_adult",
                    rating="explicit",
                    source="discord.generate_ai",
                    seed=424242,
                )
                resolved = service.resolve_request(request)
                positive = {token.strip() for token in resolved.prompt_result.positive.split(",")}
                self.assertTrue({"1girl", "1boy", "hetero", "full body", "feet visible"}.issubset(positive))
                self.assertNotIn("portrait", positive)
                self.assertNotIn("upper body", positive)
                self.assertNotIn("close-up", positive)
                self.assertNotIn("face focus", positive)
                self.assertNotIn("solo", positive)
                suppressed = {item.tag for item in resolved.suppressed_tags}
                self.assertTrue({"portrait", "upper body", "close-up", "face focus", "solo"}.issubset(suppressed))
                self.assertTrue(resolved.people_semantics.requires_multiple_people)
                self.assertEqual("full body", resolved.framing_semantics.effective[-1])

    def test_resolve_request_preserves_user_explicit_upper_body(self):
        service = GenerationService(
            prompt_resolver=Mock(
                return_value=_prompt_result("1girl, 1boy, hetero, upper body, full body")
            )
        )
        resolved = service.resolve_request(
            GenerationRequest(
                request_text="1girl, 1boy, hetero, upper body",
                model_key="animagine_adult",
                rating="explicit",
                source="discord.generate_ai",
            )
        )
        positive = {token.strip() for token in resolved.prompt_result.positive.split(",")}
        self.assertIn("upper body", positive)
        self.assertNotIn("full body", positive)
        self.assertNotIn("solo", positive)

    def test_resolve_request_multi_person_selector_resolves_cowboy_and_full_body(self):
        for mode, expected in (("cowboy", "cowboy shot"), ("full_body", "full body")):
            with self.subTest(mode=mode):
                service = GenerationService(
                    prompt_resolver=Mock(
                        return_value=_prompt_result(
                            "1girl, 1boy, hetero, portrait, upper body, face focus, solo"
                        )
                    )
                )
                resolved = service.resolve_request(
                    GenerationRequest(
                        request_text="1girl, 1boy, hetero",
                        multi_person_framing=mode,
                        model_key="animagine_adult",
                        rating="explicit",
                        source="discord.generate_ai",
                    )
                )
                positive = {token.strip() for token in resolved.prompt_result.positive.split(",")}
                self.assertIn(expected, positive)
                self.assertNotIn("portrait", positive)
                self.assertNotIn("upper body", positive)
                self.assertNotIn("face focus", positive)
                self.assertNotIn("solo", positive)

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

    def test_conflicts_keep_explicit_constraints_and_record_suppressed_tags(self):
        service = GenerationService(
            prompt_resolver=Mock(
                return_value=_prompt_result(
                    positive=(
                        "1girl, solo, 2girls, hugging, smile, serious, "
                        "running, sitting, full body, upper body, from side, three-quarter view"
                    )
                )
            )
        )
        request = GenerationRequest(
            explicit_tags=("2girls", "hugging", "smile", "running", "full body", "from side"),
            request_text="safe interaction portrait",
            model_key="animagine",
            rating="safe",
        )

        prepared = service.prepare_generation(request)
        positive = {token.strip() for token in prepared.options.prompt.split(",")}
        self.assertIn("2girls", positive)
        self.assertIn("hugging", positive)
        self.assertIn("smile", positive)
        self.assertIn("running", positive)
        self.assertIn("full body", positive)
        self.assertIn("from side", positive)
        self.assertNotIn("solo", positive)
        self.assertNotIn("serious", positive)
        self.assertNotIn("sitting", positive)
        self.assertNotIn("upper body", positive)
        self.assertNotIn("three-quarter view", positive)
        self.assertTrue(any(item.tag == "solo" for item in prepared.resolved.suppressed_tags))
        provenance = {item.tag: item.source for item in prepared.resolved.provenance}
        self.assertEqual("user_explicit", provenance["smile"])
        self.assertEqual("system", provenance["safe"])

    def test_complementary_explicit_people_keep_user_provenance_without_solo(self):
        service = GenerationService(
            prompt_resolver=Mock(
                return_value=_prompt_result(positive="1girl, solo, 1boy, holding hands")
            )
        )
        request = GenerationRequest(
            explicit_tags=("1girl", "1boy", "holding hands"),
            request_text="two people standing together",
            model_key="animagine",
            rating="safe",
        )

        prepared = service.prepare_generation(request)
        positive = {token.strip() for token in prepared.options.prompt.split(",")}
        self.assertNotIn("solo", positive)
        self.assertTrue({"1girl", "1boy", "holding hands"}.issubset(positive))
        provenance = {item.tag: item.source for item in prepared.resolved.provenance}
        self.assertEqual("user_explicit", provenance["1girl"])
        self.assertEqual("user_explicit", provenance["1boy"])
        self.assertFalse(any(item.category == "people_count" for item in prepared.resolved.prompt_conflicts))

    def test_clothed_sex_does_not_receive_late_default_solo(self):
        service = GenerationService(
            prompt_resolver=Mock(
                return_value=_prompt_result(
                    positive="1girl, clothed_sex, upper body",
                )
            )
        )
        request = GenerationRequest(
            explicit_tags=("1girl", "clothed_sex", "upper body"),
            request_text="clothed_sex",
            model_key="animagine",
            rating="safe",
        )

        prepared = service.prepare_generation(request)
        positive = {token.strip() for token in prepared.options.prompt.split(",")}
        self.assertNotIn("solo", positive)
        self.assertIn("1girl", positive)
        self.assertIn("clothed_sex", positive)
        self.assertTrue(prepared.resolved.people_semantics.requires_multiple_people)
        self.assertEqual(("1girl",), prepared.resolved.people_semantics.explicit_people_count)

    def test_clothed_sex_regression_fixture_removes_only_auto_solo(self):
        service = GenerationService(
            prompt_resolver=Mock(
                return_value=_prompt_result(
                    positive=(
                        "1girl, solo, explicit, cafe_stella_to_shinigami_no_chou, blazer, "
                        "clothed_sex, upper body, from side, blush, standing, masterpiece, "
                        "high score, great score, absurdres"
                    )
                )
            )
        )
        request = GenerationRequest(
            explicit_tags=("1girl", "clothed_sex", "upper body", "from side", "blush", "standing"),
            request_text="clothed_sex",
            model_key="animagine",
            rating="safe",
        )

        prepared = service.prepare_generation(request)
        positive = {token.strip() for token in prepared.options.prompt.split(",")}
        self.assertNotIn("solo", positive)
        self.assertIn("1girl", positive)
        self.assertIn("clothed_sex", positive)
        self.assertIn("upper body", positive)
        suppressed = [item for item in prepared.resolved.suppressed_tags if item.tag == "solo"]
        self.assertTrue(suppressed)
        self.assertEqual("clothed_sex", suppressed[-1].source_interaction)

    def test_prompt_designer_solo_is_suppressed_for_hetero_and_mixed_sex_is_recorded(self):
        service = GenerationService(
            prompt_resolver=Mock(
                return_value=_prompt_result(
                    positive="1girl, hetero, solo, upper body",
                )
            )
        )
        request = GenerationRequest(
            explicit_tags=("1girl", "hetero", "upper body"),
            request_text="hetero",
            model_key="animagine",
            rating="safe",
        )

        prepared = service.prepare_generation(request)
        positive = {token.strip() for token in prepared.options.prompt.split(",")}
        self.assertNotIn("solo", positive)
        self.assertTrue(prepared.resolved.people_semantics.requires_multiple_people)
        self.assertTrue(prepared.resolved.people_semantics.implies_mixed_sex)
        self.assertTrue(any(item.tag == "solo" for item in prepared.resolved.suppressed_tags))

    def test_prompt_designer_portrait_is_suppressed_for_multi_person_request(self):
        service = GenerationService(
            prompt_resolver=Mock(
                return_value=_prompt_result(
                    positive="1girl, 1boy, hetero, clothed_sex, portrait",
                )
            )
        )
        request = GenerationRequest(
            explicit_tags=("1girl", "1boy", "hetero", "clothed_sex"),
            request_text="hetero clothed_sex",
            model_key="animagine",
            rating="safe",
        )

        prepared = service.prepare_generation(request)
        positive = {token.strip() for token in prepared.options.prompt.split(",")}
        self.assertTrue({"1girl", "1boy", "hetero", "clothed_sex"}.issubset(positive))
        self.assertNotIn("portrait", positive)
        suppressed = next(item for item in prepared.resolved.suppressed_tags if item.tag == "portrait")
        self.assertEqual("prompt_designer", suppressed.source)
        self.assertEqual("conflicts_with_multi_person_framing", suppressed.reason)
        self.assertEqual(("portrait",), prepared.resolved.framing_semantics.suppressed)

    def test_multi_person_auto_framing_suppresses_auto_upper_body(self):
        service = GenerationService(
            prompt_resolver=Mock(
                return_value=_prompt_result(
                    positive="1girl, 1boy, hetero, upper body",
                )
            )
        )
        request = GenerationRequest(
            explicit_tags=("1girl", "1boy", "hetero"),
            request_text="hetero",
            model_key="animagine",
            rating="safe",
        )

        prepared = service.prepare_generation(request)
        self.assertNotIn("upper body", prepared.options.prompt)
        self.assertIn("upper body", prepared.resolved.framing_semantics.suppressed)

    def test_multi_person_ui_full_body_adds_verified_tag_and_suppresses_auto_framing(self):
        service = GenerationService(
            prompt_resolver=Mock(
                return_value=_prompt_result(
                    positive="1girl, 1boy, hetero, portrait",
                )
            )
        )
        request = GenerationRequest(
            explicit_tags=("1girl", "1boy", "hetero"),
            request_text="hetero",
            multi_person_framing="full_body",
            model_key="animagine",
            rating="safe",
        )

        prepared = service.prepare_generation(request)

        positive = {token.strip() for token in prepared.options.prompt.split(",")}
        self.assertIn("full body", positive)
        self.assertNotIn("portrait", positive)
        self.assertEqual("full_body", prepared.resolved.multi_person_framing_effective)
        self.assertIn(("full body", "user_ui"), {(item.tag, item.source) for item in prepared.resolved.provenance})

    def test_adult_multi_person_full_body_adds_auto_wide_shot(self):
        for rating in ("explicit", "nsfw"):
            with self.subTest(rating=rating):
                service = GenerationService(
                    prompt_resolver=Mock(
                        return_value=_prompt_result(
                            "1girl, 1boy, hetero, portrait, upper body, close-up, face focus, solo"
                        )
                    )
                )
                request = GenerationRequest(
                    explicit_tags=("1girl", "1boy", "hetero"),
                    request_text="hetero",
                    multi_person_framing="full_body",
                    model_key="animagine_adult",
                    rating=rating,
                )

                prepared = service.prepare_generation(request)
                tokens = {token.strip() for token in prepared.options.prompt.split(",")}
                self.assertIn("full body", tokens)
                self.assertIn("wide shot", tokens)
                self.assertFalse(
                    {"portrait", "upper body", "close-up", "face focus", "solo"}.intersection(tokens)
                )
                self.assertIn(
                    ("wide shot", "framing_auto"),
                    {(item.tag, item.source) for item in prepared.resolved.provenance},
                )

    def test_adult_wide_shot_injection_respects_scope_and_user_framing(self):
        cases = (
            ("safe", ("1girl", "1boy", "hetero"), "full_body", False),
            ("explicit", ("1girl",), "full_body", False),
            ("explicit", ("1girl", "1boy", "hetero"), "cowboy", False),
            ("explicit", ("1girl", "1boy", "hetero", "close-up"), "full_body", False),
        )
        for rating, explicit_tags, mode, expect_wide in cases:
            with self.subTest(rating=rating, explicit_tags=explicit_tags, mode=mode):
                prompt = (
                    "1girl, portrait, solo"
                    if len(explicit_tags) == 1
                    else "1girl, 1boy, hetero, close-up, portrait, solo"
                )
                service = GenerationService(
                    prompt_resolver=Mock(
                        return_value=_prompt_result(prompt)
                    )
                )
                prepared = service.prepare_generation(
                    GenerationRequest(
                        explicit_tags=explicit_tags,
                        request_text="hetero" if len(explicit_tags) > 1 else "standing",
                        multi_person_framing=mode,
                        model_key="animagine_adult",
                        rating=rating,
                    )
                )
                tokens = {token.strip() for token in prepared.options.prompt.split(",")}
                self.assertEqual(expect_wide, "wide shot" in tokens)
                if "close-up" in explicit_tags:
                    self.assertIn("close-up", tokens)

    def test_adult_wide_shot_does_not_duplicate_existing_tag(self):
        service = GenerationService(
            prompt_resolver=Mock(
                return_value=_prompt_result("1girl, 1boy, hetero, full body, wide shot")
            )
        )
        prepared = service.prepare_generation(
            GenerationRequest(
                explicit_tags=("1girl", "1boy", "hetero", "wide shot"),
                request_text="hetero",
                multi_person_framing="full_body",
                model_key="animagine_adult",
                rating="explicit",
            )
        )
        self.assertEqual(
            1,
            sum(token.strip() == "wide shot" for token in prepared.options.prompt.split(",")),
        )

    def test_multi_person_explicit_framing_wins_over_ui_mode_and_is_recorded(self):
        service = GenerationService(
            prompt_resolver=Mock(return_value=_prompt_result("1girl, 1boy, hetero, portrait"))
        )
        request = GenerationRequest(
            explicit_tags=("1girl", "1boy", "hetero", "portrait"),
            request_text="hetero",
            multi_person_framing="full_body",
            model_key="animagine",
            rating="safe",
        )

        prepared = service.prepare_generation(request)

        self.assertIn("portrait", prepared.options.prompt)
        self.assertNotIn("full body", prepared.options.prompt)
        self.assertEqual("explicit", prepared.resolved.multi_person_framing_effective)
        self.assertTrue(any(item.reason == "explicit_framing_overrides_multi_person_selector" for item in prepared.resolved.prompt_conflicts))

    def test_semantic_multi_person_modes_do_not_invent_positive_tags(self):
        service = GenerationService(
            prompt_resolver=Mock(return_value=_prompt_result("1girl, 1boy, hetero"))
        )
        for mode in ("both_faces", "partial_partner_ok"):
            with self.subTest(mode=mode):
                request = GenerationRequest(
                    explicit_tags=("1girl", "1boy", "hetero"),
                    request_text="hetero",
                    multi_person_framing=mode,
                    model_key="animagine",
                    rating="safe",
                )
                prepared = service.prepare_generation(request)
                self.assertNotIn(mode, prepared.options.prompt)
                self.assertEqual(mode, prepared.resolved.multi_person_framing_effective)

    def test_single_person_ignores_multi_person_framing_mode(self):
        service = GenerationService(prompt_resolver=Mock(return_value=_prompt_result("1girl, portrait")))
        request = GenerationRequest(
            explicit_tags=("1girl",),
            request_text="standing",
            multi_person_framing="full_body",
            model_key="animagine",
            rating="safe",
        )

        prepared = service.prepare_generation(request)

        self.assertNotIn("full body", prepared.options.prompt)
        self.assertEqual("auto", prepared.resolved.multi_person_framing_effective)

    def test_variation_added_portrait_is_suppressed_for_multi_person_request(self):
        service = GenerationService(
            prompt_resolver=Mock(
                return_value=_prompt_result(
                    positive="1girl, 1boy, hetero, clothed_sex",
                )
            )
        )
        request = GenerationRequest(
            explicit_tags=("1girl", "1boy", "hetero", "clothed_sex"),
            request_text="hetero clothed_sex",
            model_key="animagine",
            rating="safe",
        )

        prepared = service.prepare_generation(
            request,
            positive_override="1girl, 1boy, hetero, clothed_sex, portrait",
            variation_tags=("portrait",),
        )
        self.assertNotIn("portrait", prepared.options.prompt)
        suppressed = next(item for item in prepared.resolved.suppressed_tags if item.tag == "portrait")
        self.assertEqual("variation_auto", suppressed.source)

    def test_variation_solo_is_suppressed_for_multi_person_request(self):
        service = GenerationService(
            prompt_resolver=Mock(
                return_value=_prompt_result(
                    positive="1girl, clothed_sex, upper body",
                )
            )
        )
        request = GenerationRequest(
            explicit_tags=("1girl", "clothed_sex", "upper body"),
            request_text="clothed_sex",
            model_key="animagine",
            rating="safe",
        )

        prepared = service.prepare_generation(
            request,
            positive_override="1girl, clothed_sex, upper body, solo",
            variation_tags=("solo",),
        )
        positive = {token.strip() for token in prepared.options.prompt.split(",")}
        self.assertNotIn("solo", positive)
        self.assertTrue(any(item.tag == "solo" for item in prepared.resolved.suppressed_tags))

    def test_explicit_people_counts_keep_onegirl_oneboy_and_two_girls_without_solo(self):
        for explicit_tags, prompt in (
            (("1girl", "1boy", "hugging"), "1girl, 1boy, hugging, solo"),
            (("2girls", "hugging"), "2girls, hugging, solo"),
        ):
            with self.subTest(explicit_tags=explicit_tags):
                service = GenerationService(prompt_resolver=Mock(return_value=_prompt_result(prompt)))
                request = GenerationRequest(
                    explicit_tags=explicit_tags,
                    request_text="hugging",
                    model_key="animagine",
                    rating="safe",
                )
                prepared = service.prepare_generation(request)
                positive = {token.strip() for token in prepared.options.prompt.split(",")}
                self.assertNotIn("solo", positive)

    def test_explicit_solo_conflict_is_not_silently_rewritten(self):
        service = GenerationService(
            prompt_resolver=Mock(return_value=_prompt_result("2girls, solo, hugging"))
        )
        request = GenerationRequest(
            explicit_tags=("2girls", "solo", "hugging"),
            request_text="hugging",
            model_key="animagine",
            rating="safe",
        )

        prepared = service.prepare_generation(request)
        positive = {token.strip() for token in prepared.options.prompt.split(",")}
        self.assertIn("solo", positive)
        self.assertTrue(prepared.resolved.prompt_conflicts)

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

    def test_single_person_auto_portrait_remains_unchanged(self):
        service = GenerationService(
            prompt_resolver=Mock(return_value=_prompt_result("1girl, portrait, standing"))
        )
        request = GenerationRequest(
            explicit_tags=("1girl",),
            request_text="standing",
            model_key="animagine",
            rating="safe",
        )

        prepared = service.prepare_generation(request)
        self.assertIn("portrait", prepared.options.prompt)
        self.assertEqual(("portrait",), prepared.resolved.framing_semantics.effective)

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
