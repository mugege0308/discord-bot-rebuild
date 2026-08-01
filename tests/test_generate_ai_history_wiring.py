import ast
import unittest
from pathlib import Path
from unittest import mock

import commands.prompt as prompt_module
from services.generation_history import SavedGeneration
from services.image_selection import ImageSelectionResult
from services.sd_control import GeneratedImage
from services.shogou_prompt_designer import ShogouPromptResult


PROMPT_SOURCE = Path(__file__).resolve().parents[1] / "commands" / "prompt.py"


def _find_function(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function not found: {name}")


def _find_calls(function, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


def _references_name(function, name: str) -> bool:
    # generate_images / generate_images_with_metadata are passed as bare
    # references to asyncio.to_thread(...), not called directly, so they
    # show up as ast.Name nodes rather than ast.Call nodes.
    return any(
        isinstance(node, ast.Name) and node.id == name
        for node in ast.walk(function)
    )


def _make_result() -> ShogouPromptResult:
    return ShogouPromptResult(
        positive="1girl, safe, masterpiece, high score, great score, absurdres",
        negative="lowres, bad anatomy",
        summary="テスト用の要約",
        assumptions=("背景を補完",),
    )


def _make_generated_images(count: int = 1) -> list[GeneratedImage]:
    return [
        GeneratedImage(path=Path(f"data/sd_outputs/test_{index}.png"), batch_index=index, seed=100 + index)
        for index in range(1, count + 1)
    ]


class GenerateAiModelDisplayTestCase(unittest.TestCase):
    def _message(self, model_key=None, *, selection=None, send_count=1, generated_count=1) -> str:
        if selection is None:
            selection = ImageSelectionResult(selected_paths=[], used_selection=False)
        kwargs = {} if model_key is None else {"model_key": model_key}
        return prompt_module._build_shogou_generated_image_message(
            _make_result(), selection, send_count, generated_count, **kwargs
        )

    def test_janku_model_display_is_deterministic(self):
        self.assertIn("Model: JANKU v7.77", self._message("janku"))

    def test_animagine_model_display_is_deterministic(self):
        self.assertIn("Model: Animagine XL 4.0 Clear", self._message("animagine"))

    def test_unspecified_model_display_defaults_to_animagine(self):
        self.assertIn("Model: Animagine XL 4.0 Clear", self._message())

    def test_model_display_survives_selection_success_and_failure(self):
        selected = ImageSelectionResult(
            selected_paths=[Path("data/sd_outputs/test_1.png")],
            used_selection=True,
            message="画像選別AIで上位1枚を選びました。",
        )
        failed = ImageSelectionResult(
            selected_paths=[],
            used_selection=False,
            error="画像選別AIの実行に失敗しました。",
        )
        selected_message = self._message("janku", selection=selected, generated_count=2)
        failed_message = self._message("janku", selection=failed, generated_count=2)
        self.assertIn("画像選別AIで上位1枚を選びました。", selected_message)
        self.assertIn("画像選別AIの実行に失敗しました。", failed_message)
        self.assertIn("Model: JANKU v7.77", selected_message)
        self.assertIn("Model: JANKU v7.77", failed_message)

    def test_generate_and_generate_raw_message_builder_is_unchanged(self):
        message = prompt_module._build_generated_image_message(
            "1girl", "lowres", ImageSelectionResult([], False), 1, 1
        )
        self.assertNotIn("Model:", message)


class HistorySaveDoesNotBlockDiscordDeliveryTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_db_save_returning_none_still_sends_images(self):
        interaction = mock.MagicMock()
        interaction.user.id = 555
        images = _make_generated_images(1)
        selection = ImageSelectionResult(selected_paths=[], used_selection=False, message="", error="")

        with mock.patch(
            "commands.prompt.generate_images_with_metadata", return_value=images
        ) as generate_mock, mock.patch(
            "commands.prompt.select_generated_images", return_value=selection
        ), mock.patch(
            "commands.prompt.save_generation_history_detailed", return_value=None
        ) as save_mock, mock.patch(
            "commands.prompt._send_image_files"
        ) as send_mock, mock.patch(
            "commands.prompt._send_error"
        ) as error_mock:
            await prompt_module._generate_images_from_shogou_result(
                interaction,
                _make_result(),
                label_prefix="generate_ai",
                request_text="テスト依頼",
                reference_id=None,
            )

        generate_mock.assert_called_once()
        save_mock.assert_called_once()
        send_mock.assert_called_once()
        error_mock.assert_not_called()

    async def test_db_save_raising_exception_still_sends_images(self):
        interaction = mock.MagicMock()
        interaction.user.id = 555
        images = _make_generated_images(1)
        selection = ImageSelectionResult(selected_paths=[], used_selection=False, message="", error="")

        with mock.patch(
            "commands.prompt.generate_images_with_metadata", return_value=images
        ), mock.patch(
            "commands.prompt.select_generated_images", return_value=selection
        ), mock.patch(
            "commands.prompt.save_generation_history_detailed", side_effect=RuntimeError("db exploded")
        ), mock.patch(
            "commands.prompt._send_image_files"
        ) as send_mock, mock.patch(
            "commands.prompt._send_error"
        ) as error_mock:
            await prompt_module._generate_images_from_shogou_result(
                interaction,
                _make_result(),
                label_prefix="generate_ai",
                request_text="テスト依頼",
                reference_id=None,
            )

        send_mock.assert_called_once()
        error_mock.assert_not_called()


class ComfyFailureSkipsHistorySaveTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_comfy_failure_never_calls_save_generation_history(self):
        interaction = mock.MagicMock()
        interaction.user.id = 555

        with mock.patch(
            "commands.prompt.generate_images_with_metadata", side_effect=RuntimeError("comfy down")
        ), mock.patch(
            "commands.prompt.save_generation_history_detailed"
        ) as save_mock, mock.patch(
            "commands.prompt._send_error"
        ) as error_mock, mock.patch(
            "commands.prompt._send_image_files"
        ) as send_mock:
            await prompt_module._generate_images_from_shogou_result(
                interaction,
                _make_result(),
                label_prefix="generate_ai",
                request_text="テスト依頼",
                reference_id=None,
            )

        save_mock.assert_not_called()
        error_mock.assert_called_once()
        send_mock.assert_not_called()


class SelectionAiWasSelectedWiringTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_selected_paths_and_selection_flags_are_forwarded_to_history(self):
        interaction = mock.MagicMock()
        interaction.user.id = 777
        images = _make_generated_images(2)
        selection = ImageSelectionResult(
            selected_paths=[Path("data/sd_outputs/test_1.png")],
            used_selection=True,
            message="画像選別AIで上位1枚を選びました。",
            error="",
        )

        with mock.patch(
            "commands.prompt.generate_images_with_metadata", return_value=images
        ), mock.patch(
            "commands.prompt.select_generated_images", return_value=selection
        ), mock.patch(
            "commands.prompt.save_generation_history_detailed",
            return_value=SavedGeneration(request_id=1, image_ids=[11, 22]),
        ) as save_mock, mock.patch(
            "commands.prompt._send_image_files"
        ), mock.patch(
            # Keep the Phase 1B bookkeeping out of this test: without it the
            # real set_image_discord_message would create data/generation_history.db
            # in the repo.
            "commands.prompt.set_image_discord_message",
            return_value=True,
        ):
            await prompt_module._generate_images_from_shogou_result(
                interaction,
                _make_result(),
                label_prefix="generate_ai",
                request_text="猫耳の女の子",
                reference_id=3,
            )

        save_mock.assert_called_once()
        _, kwargs = save_mock.call_args
        args = save_mock.call_args.args
        request_record = args[0]
        saved_images = args[1]
        self.assertEqual(request_record.user_id, 777)
        self.assertEqual(request_record.request_text, "猫耳の女の子")
        self.assertEqual(request_record.reference_id, 3)
        self.assertTrue(request_record.selection_used)
        self.assertEqual(saved_images, images)
        self.assertEqual(kwargs["selected_paths"], selection.selected_paths)

    async def test_janku_model_and_prefix_reach_generation_and_history(self):
        interaction = mock.MagicMock()
        interaction.user.id = 777
        images = _make_generated_images(1)
        selection = ImageSelectionResult(selected_paths=[], used_selection=False)

        with mock.patch(
            "commands.prompt.generate_images_with_metadata", return_value=images
        ) as generate_mock, mock.patch(
            "commands.prompt.select_generated_images", return_value=selection
        ), mock.patch(
            "commands.prompt.save_generation_history_detailed", return_value=None
        ) as save_mock, mock.patch(
            "commands.prompt._send_image_files"
        ) as send_mock:
            await prompt_module._generate_images_from_shogou_result(
                interaction,
                _make_result(),
                label_prefix="generate_ai",
                request_text="JANKUテスト",
                model_key="janku",
                model_selection_explicit=True,
            )

        options = generate_mock.call_args.args[0]
        self.assertEqual(options.model_key, "janku")
        self.assertEqual(options.filename_prefix, "Shogou_JANKU")
        self.assertTrue(options.model_selection_explicit)
        request_record = save_mock.call_args.args[0]
        self.assertEqual(request_record.model_key, "janku")
        self.assertIn("Model: JANKU v7.77", send_mock.call_args.args[1])

    async def test_default_animagine_model_is_displayed(self):
        interaction = mock.MagicMock()
        interaction.user.id = 777
        images = _make_generated_images(1)
        selection = ImageSelectionResult(selected_paths=[], used_selection=False)

        with mock.patch(
            "commands.prompt.generate_images_with_metadata", return_value=images
        ), mock.patch(
            "commands.prompt.select_generated_images", return_value=selection
        ), mock.patch(
            "commands.prompt.save_generation_history_detailed", return_value=None
        ), mock.patch(
            "commands.prompt._send_image_files"
        ) as send_mock:
            await prompt_module._generate_images_from_shogou_result(
                interaction,
                _make_result(),
                label_prefix="generate_ai",
                request_text="既定モデルテスト",
            )

        self.assertIn("Model: Animagine XL 4.0 Clear", send_mock.call_args.args[1])


class ExistingCommandsUnaffectedByHistoryIntegrationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(PROMPT_SOURCE.read_text(encoding="utf-8"), filename=str(PROMPT_SOURCE))

    def test_generate_images_from_prompts_still_uses_plain_generate_images(self):
        function = _find_function(self.tree, "_generate_images_from_prompts")

        self.assertTrue(_references_name(function, "generate_images"))
        self.assertFalse(_references_name(function, "generate_images_with_metadata"))
        self.assertFalse(_references_name(function, "save_generation_history_detailed"))

    def test_slash_generate_raw_is_unaffected(self):
        function = _find_function(self.tree, "slash_generate_raw")
        arg_names = [arg.arg for arg in function.args.args]

        self.assertEqual(arg_names, ["interaction", "positive", "negative", "use_default_negative", "public"])
        self.assertTrue(_find_calls(function, "_generate_images_from_prompts"))
        self.assertFalse(_find_calls(function, "save_generation_history_detailed"))

    def test_only_generate_ai_path_uses_the_detailed_generation_api(self):
        shogou_function = _find_function(self.tree, "_generate_images_from_shogou_result")

        self.assertTrue(_references_name(shogou_function, "generate_images_with_metadata"))
        self.assertFalse(_references_name(shogou_function, "generate_images"))


if __name__ == "__main__":
    unittest.main()
