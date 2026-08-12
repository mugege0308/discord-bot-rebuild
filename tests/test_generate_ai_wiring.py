import ast
import unittest
from pathlib import Path

from commands.prompt import GENERATE_AI_MODEL_CHOICES

PROMPT_SOURCE = Path(__file__).resolve().parents[1] / "commands" / "prompt.py"
MAIN_SOURCE = Path(__file__).resolve().parents[1] / "main.py"


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


class GenerateAiWiringTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(PROMPT_SOURCE.read_text(encoding="utf-8"), filename=str(PROMPT_SOURCE))
        cls.main_tree = ast.parse(MAIN_SOURCE.read_text(encoding="utf-8-sig"), filename=str(MAIN_SOURCE))

    def test_generate_ai_command_keeps_request_and_reference_id_and_adds_model(self):
        function = _find_function(self.tree, "slash_generate_ai")
        arg_names = [arg.arg for arg in function.args.args]

        self.assertEqual(
            arg_names,
            [
                "interaction",
                "request",
                "model",
                "reference_id",
                "reference_image",
                "reference_strength",
                "use_last_reference",
            ],
        )

    def test_generate_ai_forwards_reference_image_data_and_strength(self):
        function = _find_function(self.tree, "slash_generate_ai")
        call = _find_calls(function, "_generate_images_from_shogou_result")[0]
        keyword_names = {keyword.arg for keyword in call.keywords}

        self.assertIn("reference_image_bytes", keyword_names)
        self.assertIn("reference_image_content_type", keyword_names)
        self.assertIn("reference_strength", keyword_names)

    def test_generate_ai_model_choices_are_exactly_animagine_and_janku(self):
        self.assertEqual(
            [(choice.name, choice.value) for choice in GENERATE_AI_MODEL_CHOICES],
            [
                ("Animagine XL 4.0 Clear", "animagine"),
                ("JANKU v7.77", "janku"),
            ],
        )

    def test_generate_ai_uses_shogou_result_generator_not_legacy_pipeline(self):
        function = _find_function(self.tree, "slash_generate_ai")

        self.assertTrue(_find_calls(function, "_build_shogou_prompt_result"))
        self.assertFalse(_find_calls(function, "_build_prompt_result"))
        self.assertFalse(_find_calls(function, "generate_sd_prompt_result"))

    def test_generate_ai_does_not_generate_images_when_blocked(self):
        function = _find_function(self.tree, "slash_generate_ai")

        # Structurally: the "if result.blocked:" branch must return before any
        # call to the image-generation helper reaches ComfyUI, and that call
        # must live in a later top-level statement, not inside the guard.
        blocked_if = None
        blocked_if_index = None
        for index, node in enumerate(function.body):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute) and node.test.attr == "blocked":
                blocked_if = node
                blocked_if_index = index
                break
        self.assertIsNotNone(blocked_if, "expected an `if result.blocked:` guard")
        self.assertTrue(any(isinstance(stmt, ast.Return) for stmt in blocked_if.body))
        self.assertFalse(_find_calls(blocked_if, "_generate_images_from_shogou_result"))

        generate_call_statement_index = None
        for index, node in enumerate(function.body):
            if _find_calls(node, "_generate_images_from_shogou_result"):
                generate_call_statement_index = index
                break
        self.assertIsNotNone(generate_call_statement_index, "expected a later call to _generate_images_from_shogou_result")
        self.assertGreater(generate_call_statement_index, blocked_if_index)

    def test_generate_ai_reuses_preserve_prompt_text_generation_path(self):
        function = _find_function(self.tree, "_generate_images_from_shogou_result")
        call = _find_calls(function, "_build_generation_options")[0]
        preserve_kw = next(kw for kw in call.keywords if kw.arg == "preserve_prompt_text")

        self.assertIsInstance(preserve_kw.value, ast.Constant)
        self.assertIs(preserve_kw.value.value, True)

    def test_generate_ai_uses_positive_and_negative_directly_without_legacy_postprocessing(self):
        function = _find_function(self.tree, "_generate_images_from_shogou_result")
        call = _find_calls(function, "_build_generation_options")[0]

        first_arg = call.args[0]
        second_arg = call.args[1]
        self.assertIsInstance(first_arg, ast.Attribute)
        self.assertEqual(first_arg.attr, "positive")
        self.assertIsInstance(second_arg, ast.Attribute)
        self.assertEqual(second_arg.attr, "negative")

    def test_generate_ai_calls_selection_ai_like_existing_commands(self):
        function = _find_function(self.tree, "_generate_images_from_shogou_result")

        # select_generated_images is passed to asyncio.to_thread(...), so it
        # shows up as a Name reference rather than a direct Call node.
        referenced_names = {
            node.id for node in ast.walk(function) if isinstance(node, ast.Name)
        }
        self.assertIn("select_generated_images", referenced_names)

    def test_generate_ai_success_visibility_matches_generate_raw_default(self):
        function = _find_function(self.tree, "_generate_images_from_shogou_result")
        defaults = dict(zip((arg.arg for arg in function.args.kwonlyargs), function.args.kw_defaults))

        # /generate_raw defaults to public=True -> result_ephemeral=False.
        # /generate_ai has no public toggle input, so it fixes the same default.
        self.assertIsInstance(defaults["result_ephemeral"], ast.Constant)
        self.assertIs(defaults["result_ephemeral"].value, False)

    def test_generate_raw_function_is_untouched_by_generate_ai_addition(self):
        function = _find_function(self.tree, "slash_generate_raw")
        arg_names = [arg.arg for arg in function.args.args]

        self.assertEqual(arg_names, ["interaction", "positive", "negative", "use_default_negative", "public"])
        self.assertTrue(_find_calls(function, "_generate_images_from_prompts"))
        self.assertFalse(_find_calls(function, "_generate_images_from_shogou_result"))

    def test_sd_generate_choices_and_omitted_model_behavior_are_unchanged(self):
        function = _find_function(self.main_tree, "slash_sd_generate")
        choices_decorator = next(
            decorator
            for decorator in function.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "choices"
        )
        model_list = next(keyword.value for keyword in choices_decorator.keywords if keyword.arg == "model")
        pairs = []
        for choice_call in model_list.elts:
            values = {keyword.arg: keyword.value.value for keyword in choice_call.keywords}
            pairs.append((values["name"], values["value"]))
        self.assertEqual(
            pairs,
            [
                ("JANKU v7.77", "janku"),
                ("Animagine XL 4.0 Clear", "animagine"),
            ],
        )
        defaults = dict(zip((arg.arg for arg in function.args.args[-len(function.args.defaults):]), function.args.defaults))
        self.assertIsNone(defaults["model"].value)


if __name__ == "__main__":
    unittest.main()
