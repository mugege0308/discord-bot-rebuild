import unittest
from unittest import mock

import discord

import commands.blue_archive as blue_archive_command
from commands.blue_archive import (
    CAUTION_MESSAGE,
    build_damage_embed,
    build_damage_embed_data,
)
from services.blue_archive_damage import DamageCalculationInput, calculate_damage


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


class BlueArchiveDamageEmbedTestCase(unittest.TestCase):
    def setUp(self):
        self.values = DamageCalculationInput.from_percentages(
            attack=1234567,
            skill_percent=800,
            enemy_defense=1000,
            attack_buff_percent=50,
            damage_buff_percent=25,
            defense_debuff_percent=30,
            attribute_multiplier=2,
            terrain_multiplier=1.2,
            critical_damage_percent=200,
            hit_count=3,
            random_min_percent=90,
            random_max_percent=110,
            is_critical=True,
        )
        self.result = calculate_damage(self.values)

    def test_display_data_contains_required_sections_and_grouped_numbers(self):
        data = build_damage_embed_data(self.values, self.result)
        fields = {field.name: field.value for field in data.fields}

        self.assertEqual(data.title, "ブルーアーカイブ ダメージ計算")
        self.assertIn("1,234,567", fields["入力条件"])
        self.assertIn("実効攻撃力", fields["計算途中の補正値"])
        self.assertIn("実効防御力", fields["計算途中の補正値"])
        self.assertIn("防御倍率", fields["計算途中の補正値"])
        self.assertIn("非会心ダメージ（1ヒット）", fields)
        self.assertIn("会心ダメージ（1ヒット）", fields)
        self.assertIn("合計ダメージ（3ヒット）", fields)
        self.assertEqual(data.footer, CAUTION_MESSAGE)

    def test_embed_stays_within_discord_limits(self):
        embed = build_damage_embed(build_damage_embed_data(self.values, self.result))

        self.assertIsInstance(embed, discord.Embed)
        self.assertLessEqual(len(embed.title or ""), 256)
        self.assertLessEqual(len(embed.description or ""), 4096)
        self.assertLessEqual(len(embed.fields), 25)
        for field in embed.fields:
            self.assertLessEqual(len(field.name), 256)
            self.assertLessEqual(len(field.value), 1024)
        self.assertLessEqual(len(embed), 6000)


class BlueArchiveDamageCommandTestCase(unittest.IsolatedAsyncioTestCase):
    def _command(self):
        client = FakeClient()
        blue_archive_command.setup(client)
        return client.tree.commands["ba_damage"]

    @staticmethod
    def _interaction():
        interaction = mock.MagicMock()
        interaction.response.send_message = mock.AsyncMock()
        return interaction

    async def test_setup_registers_command_and_valid_input_returns_embed(self):
        command = self._command()
        interaction = self._interaction()

        await command(interaction, attack=1000, skill_percent=200, enemy_defense=0)

        interaction.response.send_message.assert_awaited_once()
        _, kwargs = interaction.response.send_message.call_args
        self.assertIsInstance(kwargs["embed"], discord.Embed)

    async def test_invalid_input_returns_japanese_ephemeral_error(self):
        command = self._command()
        interaction = self._interaction()

        await command(
            interaction,
            attack=1000,
            skill_percent=200,
            enemy_defense=0,
            random_min_percent=110,
            random_max_percent=90,
        )

        interaction.response.send_message.assert_awaited_once()
        args, kwargs = interaction.response.send_message.call_args
        self.assertIn("入力エラー", args[0])
        self.assertIn("最小値は最大値以下", args[0])
        self.assertTrue(kwargs["ephemeral"])

    async def test_unexpected_failure_returns_short_error_without_detail(self):
        command = self._command()
        interaction = self._interaction()

        with mock.patch(
            "commands.blue_archive.calculate_damage",
            side_effect=RuntimeError("secret internal detail"),
        ):
            await command(interaction, attack=1000, skill_percent=200, enemy_defense=0)

        args, kwargs = interaction.response.send_message.call_args
        self.assertEqual(args[0], blue_archive_command.CALCULATION_FAILED_MESSAGE)
        self.assertNotIn("secret internal detail", args[0])
        self.assertTrue(kwargs["ephemeral"])


if __name__ == "__main__":
    unittest.main()
