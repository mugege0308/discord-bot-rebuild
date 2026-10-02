import unittest

import commands.generate_ai_diagnostics as diagnostics


class FakeCommandTree:
    def __init__(self) -> None:
        self.commands: dict[str, object] = {}

    def command(self, *, name: str, description: str):
        del description

        def decorator(callback):
            self.commands[name] = callback
            return callback

        return decorator


class FakeClient:
    def __init__(self) -> None:
        self.tree = FakeCommandTree()


class GenerateAiDiagnosticsCommandTestCase(unittest.TestCase):
    def test_setup_registers_existing_service_adapters(self):
        client = FakeClient()
        diagnostics.setup(client)
        self.assertEqual(
            set(client.tree.commands),
            {"generate_ai_debug", "generate_ai_debug_detail", "generate_ai_settings_reset"},
        )
