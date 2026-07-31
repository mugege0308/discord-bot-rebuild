import unittest
from unittest import mock

import discord

import commands.blue_archive as blue_archive_command
from commands.blue_archive import (
    TIMELINE_CAUTION_MESSAGE,
    build_timeline_embed,
    build_timeline_embed_data,
)
from services.blue_archive_timeline import (
    MAX_TIMELINE_ACTIONS,
    TimelineConfig,
    parse_timeline,
    simulate_timeline,
)


class FakeCommandTree:
    def __init__(self) -> None:
        self.commands: dict[str, object] = {}
        self.descriptions: dict[str, str] = {}

    def command(self, *, name: str, description: str):
        self.descriptions[name] = description

        def decorator(callback):
            self.commands[name] = callback
            return callback

        return decorator


class FakeClient:
    def __init__(self) -> None:
        self.tree = FakeCommandTree()


def _example():
    actions = parse_timeline("3:55 ヒマリ 3\n3:51 アコ 3\n3:47 ドレスヒナ 6")
    config = TimelineConfig.from_values(
        battle_time=240,
        start_cost=0,
        max_cost=10,
        cost_per_second="0.7",
        recovery_multiplier=1,
    )
    return config, simulate_timeline(actions, config)


class BlueArchiveTimelineEmbedTestCase(unittest.TestCase):
    def test_display_data_contains_settings_results_and_grouped_actions(self):
        config, result = _example()
        data = build_timeline_embed_data(config, result)

        self.assertEqual(data.title, "ブルーアーカイブ TLコスト検査")
        self.assertIn("strict", data.fields[0].value)
        self.assertIn("0.7/秒", data.fields[0].value)
        self.assertIn("不成立", data.fields[1].name)
        self.assertTrue(any("ヒマリ" in field.name for field in data.fields))
        self.assertTrue(any("使用前" in field.value for field in data.fields))
        self.assertTrue(any("最短" in field.value for field in data.fields))
        self.assertEqual(data.footer, TIMELINE_CAUTION_MESSAGE)

    def test_embed_stays_within_discord_limits_at_maximum_rows(self):
        timeline = "\n".join(
            f"59:59 {'長い表示名' * 10}{index} 1000/999 | {'メモ' * 30}"
            for index in range(MAX_TIMELINE_ACTIONS)
        )
        actions = parse_timeline(timeline)
        config = TimelineConfig.from_values(
            battle_time=3600,
            start_cost=1000,
            max_cost=1000,
            cost_per_second=1000,
            recovery_multiplier=1000,
        )
        result = simulate_timeline(actions, config)
        embed = build_timeline_embed(build_timeline_embed_data(config, result))

        self.assertIsInstance(embed, discord.Embed)
        self.assertLessEqual(len(embed.title or ""), 256)
        self.assertLessEqual(len(embed.description or ""), 4096)
        self.assertLessEqual(len(embed.fields), 25)
        for field in embed.fields:
            self.assertLessEqual(len(field.name), 256)
            self.assertLessEqual(len(field.value), 1024)
        self.assertLessEqual(len(embed), 6000)


class BlueArchiveTimelineCommandTestCase(unittest.IsolatedAsyncioTestCase):
    def _client_and_command(self):
        client = FakeClient()
        blue_archive_command.setup(client)
        return client, client.tree.commands["ba_tl_cost"]

    @staticmethod
    def _interaction():
        interaction = mock.MagicMock()
        interaction.response.send_message = mock.AsyncMock()
        return interaction

    async def test_setup_registers_both_blue_archive_commands_once(self):
        client, command = self._client_and_command()
        interaction = self._interaction()

        self.assertEqual(set(client.tree.commands), {"ba_damage", "ba_tl_cost"})
        self.assertLessEqual(len(client.tree.descriptions["ba_tl_cost"]), 100)
        parameter_descriptions = getattr(command, "__discord_app_commands_param_description__")
        self.assertTrue(parameter_descriptions)
        self.assertTrue(all(len(description) <= 100 for description in parameter_descriptions.values()))
        await command(
            interaction,
            timeline="3:55 | ヒマリ | 3",
            cost_per_second=1.0,
        )

        interaction.response.send_message.assert_awaited_once()
        _, kwargs = interaction.response.send_message.call_args
        self.assertIsInstance(kwargs["embed"], discord.Embed)

    async def test_invalid_timeline_returns_japanese_ephemeral_error_once(self):
        _, command = self._client_and_command()
        interaction = self._interaction()

        await command(
            interaction,
            timeline="3:50 ヒマリ 3\n3:55 アコ 3",
            cost_per_second=0.7,
        )

        interaction.response.send_message.assert_awaited_once()
        args, kwargs = interaction.response.send_message.call_args
        self.assertIn("入力エラー", args[0])
        self.assertIn("大きい順", args[0])
        self.assertTrue(kwargs["ephemeral"])

    async def test_non_finite_value_is_safely_rejected_by_engine(self):
        _, command = self._client_and_command()
        interaction = self._interaction()

        await command(
            interaction,
            timeline="3:55 ヒマリ 3",
            cost_per_second=float("nan"),
        )

        interaction.response.send_message.assert_awaited_once()
        args, kwargs = interaction.response.send_message.call_args
        self.assertIn("NaNや無限大", args[0])
        self.assertTrue(kwargs["ephemeral"])

    async def test_unexpected_failure_hides_internal_detail_and_responds_once(self):
        _, command = self._client_and_command()
        interaction = self._interaction()

        with mock.patch(
            "commands.blue_archive.simulate_timeline",
            side_effect=RuntimeError("secret internal detail"),
        ):
            await command(
                interaction,
                timeline="3:55 ヒマリ 3",
                cost_per_second=0.7,
            )

        interaction.response.send_message.assert_awaited_once()
        args, kwargs = interaction.response.send_message.call_args
        self.assertEqual(args[0], blue_archive_command.TIMELINE_FAILED_MESSAGE)
        self.assertNotIn("secret internal detail", args[0])
        self.assertTrue(kwargs["ephemeral"])


if __name__ == "__main__":
    unittest.main()
