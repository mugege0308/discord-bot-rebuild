import ast
import unittest
from pathlib import Path


PROMPT_SOURCE = Path(__file__).resolve().parents[1] / "commands" / "prompt.py"


def _find_async_function(tree: ast.AST, name: str) -> ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
            return node
    raise AssertionError(f"async function not found: {name}")


def _find_call(function: ast.AsyncFunctionDef, name: str) -> ast.Call:
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name:
            return node
    raise AssertionError(f"call not found in {function.name}: {name}")


class GenerateRawVisibilityTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(PROMPT_SOURCE.read_text(encoding="utf-8"), filename=str(PROMPT_SOURCE))

    def test_shared_generation_keeps_ephemeral_default_for_existing_commands(self):
        function = _find_async_function(self.tree, "_generate_images_from_prompts")
        defaults = dict(zip((arg.arg for arg in function.args.kwonlyargs), function.args.kw_defaults))

        self.assertIsInstance(defaults["result_ephemeral"], ast.Constant)
        self.assertIs(defaults["result_ephemeral"].value, True)

        send_call = _find_call(function, "_send_image_files")
        ephemeral_keyword = next(keyword for keyword in send_call.keywords if keyword.arg == "ephemeral")
        self.assertIsInstance(ephemeral_keyword.value, ast.Name)
        self.assertEqual(ephemeral_keyword.value.id, "result_ephemeral")

    def test_generate_raw_public_defaults_true_and_controls_final_visibility(self):
        function = _find_async_function(self.tree, "slash_generate_raw")
        defaults = dict(
            zip(
                (arg.arg for arg in function.args.args[-len(function.args.defaults) :]),
                function.args.defaults,
            )
        )

        self.assertIsInstance(defaults["public"], ast.Constant)
        self.assertIs(defaults["public"].value, True)

        generate_call = _find_call(function, "_generate_images_from_prompts")
        visibility = next(keyword for keyword in generate_call.keywords if keyword.arg == "result_ephemeral")
        self.assertIsInstance(visibility.value, ast.UnaryOp)
        self.assertIsInstance(visibility.value.op, ast.Not)
        self.assertIsInstance(visibility.value.operand, ast.Name)
        self.assertEqual(visibility.value.operand.id, "public")


if __name__ == "__main__":
    unittest.main()
