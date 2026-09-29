import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from commands import prompt as prompt_commands
from services.generate_ai_last_settings import GenerateAiLastSettings
from services.image_selection import ImageSelectionResult
from services.sd_control import GeneratedImage, SdGenerationOptions
from services.shogou_prompt_designer import ShogouPromptResult


def _result() -> ShogouPromptResult:
    return ShogouPromptResult(
        positive="1girl, safe, masterpiece, high score, great score, absurdres",
        negative="lowres, bad anatomy",
        summary="selection test",
        assumptions=(),
    )


class SelectionSettingsTests(unittest.TestCase):
    def test_generate_ai_schema_stays_within_discord_limit_without_last_reference_option(self):
        import main

        command = next(command for command in main.client.tree.get_commands() if command.name == "generate_ai")
        options = command.to_dict(main.client.tree).get("options", [])
        self.assertLessEqual(len(options), 25)
        self.assertNotIn("use_last_reference", {option["name"] for option in options})

    def test_explicit_selection_settings_win(self):
        previous = GenerateAiLastSettings(
            style_key=None,
            lora_strength=None,
            rating="safe",
            reference_strength=0.5,
            selection_ai=False,
            candidate_count=3,
        )
        with (
            mock.patch.object(prompt_commands, "get_last_model_key", return_value=None),
            mock.patch.object(prompt_commands, "get_last_settings", return_value=previous),
        ):
            settings = prompt_commands._resolve_generate_ai_settings(
                10, selection_ai=True, candidate_count=7
            )
        self.assertTrue(settings.selection_ai)
        self.assertEqual(settings.candidate_count, 7)
        self.assertEqual(settings.sources["selection_ai"], "explicit")
        self.assertEqual(settings.sources["candidate_count"], "explicit")

    def test_previous_settings_are_model_scoped_and_inherited(self):
        previous = GenerateAiLastSettings(
            style_key=None,
            lora_strength=None,
            rating="safe",
            reference_strength=0.5,
            selection_ai=False,
            candidate_count=3,
        )
        with (
            mock.patch.object(prompt_commands, "get_last_model_key", return_value=None),
            mock.patch.object(prompt_commands, "get_last_settings", return_value=previous) as get_mock,
        ):
            settings = prompt_commands._resolve_generate_ai_settings(10)
        self.assertFalse(settings.selection_ai)
        self.assertEqual(settings.candidate_count, 3)
        self.assertEqual(settings.sources["selection_ai"], "inherited")
        self.assertEqual(settings.sources["candidate_count"], "inherited")
        get_mock.assert_called_once()

    def test_defaults_use_five_candidates(self):
        with (
            mock.patch.object(prompt_commands, "get_last_model_key", return_value=None),
            mock.patch.object(prompt_commands, "get_last_settings", return_value=None),
            mock.patch.object(prompt_commands, "get_selection_generate_count", return_value=5),
        ):
            settings = prompt_commands._resolve_generate_ai_settings(10)
        self.assertTrue(settings.selection_ai)
        self.assertEqual(settings.candidate_count, 5)
        self.assertEqual(settings.sources["candidate_count"], "default")

    def test_generation_options_honor_selection_and_reference_policy(self):
        with mock.patch.object(prompt_commands, "is_image_selection_available", return_value=True):
            options = prompt_commands._build_generation_options(
                "positive",
                "negative",
                selection_enabled=True,
                selection_generate_count=3,
            )
            reference_options = prompt_commands._build_generation_options(
                "positive",
                "negative",
                selection_enabled=True,
                selection_generate_count=5,
                reference_image_bytes=b"reference",
            )
        self.assertEqual(options.count, 3)
        self.assertEqual(reference_options.count, 1)

    def test_disabled_selection_keeps_single_generation(self):
        with mock.patch.object(prompt_commands, "is_image_selection_available", return_value=True):
            options = prompt_commands._build_generation_options(
                "positive",
                "negative",
                selection_enabled=False,
                selection_generate_count=5,
            )
        self.assertEqual(options.count, 1)


class SelectionDeliveryTests(unittest.TestCase):
    def test_winner_is_first_and_remaining_candidates_follow_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"candidate_{index}.png" for index in range(1, 4)]
            for index, path in enumerate(paths, start=1):
                path.write_bytes(f"image-{index}".encode())
            selected_copy = root / "selected_copy.png"
            selected_copy.write_bytes(paths[1].read_bytes())
            selection = ImageSelectionResult(
                selected_paths=[selected_copy],
                used_selection=True,
                candidate_scores={paths[0]: 0.1, paths[1]: 0.9, paths[2]: 0.5},
            )
            delivered = prompt_commands.build_selection_delivery_paths(paths, selection)
        self.assertEqual(delivered, [selected_copy, paths[2], paths[0]])

    def test_selection_unavailable_returns_all_successful_images(self):
        paths = [Path("candidate_1.png"), Path("candidate_2.png")]
        selection = ImageSelectionResult(paths, used_selection=False, error="selector unavailable")
        self.assertEqual(prompt_commands.build_selection_delivery_paths(paths, selection), paths)


class SelectionGenerationFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_multiple_candidates_are_selected_and_all_successes_are_sent(self):
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
                message="selected",
                candidate_scores={paths[0]: 0.1, paths[1]: 0.9, paths[2]: 0.5},
            )
            with (
                mock.patch.object(prompt_commands, "is_image_selection_available", return_value=True),
                mock.patch.object(prompt_commands, "generate_images_with_metadata", return_value=images) as generate_mock,
                mock.patch.object(prompt_commands, "select_generated_images", return_value=selection) as select_mock,
                mock.patch.object(prompt_commands, "save_generation_history_detailed", return_value=None),
                mock.patch.object(prompt_commands, "_send_image_files", new_callable=mock.AsyncMock) as send_mock,
            ):
                await prompt_commands._generate_images_from_shogou_result(
                    interaction,
                    _result(),
                    label_prefix="selection-test",
                    selection_enabled=True,
                    selection_generate_count=3,
                )
            self.assertEqual(generate_mock.call_args.args[0].count, 3)
            select_mock.assert_called_once_with(paths)
            self.assertEqual(send_mock.call_args.args[2], [paths[1], paths[2], paths[0]])

    async def test_reference_bytes_skip_selection_and_send_generated_image(self):
        interaction = mock.MagicMock()
        interaction.user.id = 321
        images = [GeneratedImage(path=Path("reference-result.png"), batch_index=1)]
        with (
            mock.patch.object(prompt_commands, "generate_images_with_metadata", return_value=images),
            mock.patch.object(prompt_commands, "select_generated_images") as select_mock,
            mock.patch.object(prompt_commands, "save_generation_history_detailed", return_value=None),
            mock.patch.object(prompt_commands, "_send_image_files", new_callable=mock.AsyncMock) as send_mock,
        ):
            await prompt_commands._generate_images_from_shogou_result(
                interaction,
                _result(),
                label_prefix="reference-test",
                reference_image_bytes=b"composition-reference",
                reference_mode="composition",
                selection_enabled=True,
                selection_generate_count=5,
            )
        select_mock.assert_not_called()
        self.assertEqual(send_mock.call_args.args[2], [Path("reference-result.png")])

    async def test_selector_exception_falls_back_to_all_generated_images(self):
        interaction = mock.MagicMock()
        interaction.user.id = 321
        paths = [Path("candidate_1.png"), Path("candidate_2.png")]
        images = [GeneratedImage(path=path, batch_index=index) for index, path in enumerate(paths, 1)]
        with (
            mock.patch.object(prompt_commands, "is_image_selection_available", return_value=True),
            mock.patch.object(prompt_commands, "generate_images_with_metadata", return_value=images),
            mock.patch.object(prompt_commands, "select_generated_images", side_effect=RuntimeError("selector down")),
            mock.patch.object(prompt_commands, "save_generation_history_detailed", return_value=None),
            mock.patch.object(prompt_commands, "_send_image_files", new_callable=mock.AsyncMock) as send_mock,
        ):
            await prompt_commands._generate_images_from_shogou_result(
                interaction,
                _result(),
                label_prefix="selector-failure-test",
                selection_enabled=True,
                selection_generate_count=2,
            )
        self.assertEqual(send_mock.call_args.args[2], paths)


class PartialCandidateGenerationTests(unittest.TestCase):
    def test_failed_candidate_does_not_discard_successful_sibling(self):
        import services.sd_control as sd_control

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate_2.png"
            output.write_bytes(b"candidate")
            options = SdGenerationOptions(prompt="positive", negative_prompt="negative", count=2)
            with (
                mock.patch.object(sd_control, "_get_comfy_workflow_file", return_value=Path("workflow.json")),
                mock.patch.object(sd_control, "_safe_workflow_file_hash", return_value="hash"),
                mock.patch.object(sd_control, "_load_comfy_workflow", return_value={}),
                mock.patch.object(sd_control, "_apply_comfy_options", return_value=({}, None, None)),
                mock.patch.object(sd_control, "_extract_checkpoint_name", return_value="checkpoint"),
                mock.patch.object(sd_control, "_find_sampler_node", return_value=None),
                mock.patch.object(sd_control, "_extract_sampler_settings", return_value=(20, 7.0, "Euler a", "normal")),
                mock.patch.object(sd_control, "_extract_latent_dimensions", return_value=(512, 512)),
                mock.patch.object(
                    sd_control,
                    "_queue_comfy_prompt_with_logging",
                    side_effect=[RuntimeError("candidate one failed"), "prompt-two"],
                ),
                mock.patch.object(sd_control, "_wait_comfy_history", return_value={}),
                mock.patch.object(sd_control, "_download_comfy_images", return_value=[output]),
            ):
                generated = sd_control._generate_comfy_images_with_metadata(options)
        self.assertEqual([image.batch_index for image in generated], [2])
        self.assertEqual(generated[0].path, output)


if __name__ == "__main__":
    unittest.main()
