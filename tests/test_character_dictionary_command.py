import unittest
from unittest import mock

import discord

import commands.character_dictionary as character_dictionary_command


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


class CharacterDictionaryCommandTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = FakeClient()
        character_dictionary_command.setup(self.client)

    @staticmethod
    def _interaction(user_id: int = 1):
        interaction = mock.MagicMock()
        interaction.user.id = user_id
        interaction.response.send_message = mock.AsyncMock()
        interaction.response.defer = mock.AsyncMock()
        interaction.followup.send = mock.AsyncMock()
        return interaction

    def test_setup_registers_all_five_commands(self):
        self.assertEqual(
            set(self.client.tree.commands.keys()),
            {"character_list", "character_info", "character_search", "character_missing", "character_validate"},
        )

    async def test_character_list_returns_embed_grouped_by_series(self):
        command = self.client.tree.commands["character_list"]
        interaction = self._interaction()

        await command(interaction, page=1)

        interaction.response.send_message.assert_awaited_once()
        _, kwargs = interaction.response.send_message.call_args
        embed = kwargs["embed"]
        self.assertIsInstance(embed, discord.Embed)
        self.assertIn("総登録数", embed.description)
        field_names = [field.name for field in embed.fields]
        self.assertTrue(any("ブルーアーカイブ" in name for name in field_names))
        self.assertTrue(kwargs["ephemeral"])

    async def test_character_list_out_of_range_page_clamps_to_last_page(self):
        command = self.client.tree.commands["character_list"]
        interaction = self._interaction()

        await command(interaction, page=9999)

        _, kwargs = interaction.response.send_message.call_args
        embed = kwargs["embed"]
        self.assertIn("ページ", embed.footer.text)

    async def test_character_info_unique_partial_match_returns_detail_embed(self):
        command = self.client.tree.commands["character_info"]
        interaction = self._interaction()

        await command(interaction, character="こいし")

        _, kwargs = interaction.response.send_message.call_args
        embed = kwargs["embed"]
        self.assertEqual(embed.title, "古明地こいし")
        field_names = {field.name for field in embed.fields}
        self.assertIn("character_tag", field_names)
        self.assertIn("series_tag", field_names)

    async def test_character_info_no_match_returns_not_found_message(self):
        command = self.client.tree.commands["character_info"]
        interaction = self._interaction()

        await command(interaction, character="該当なしのクエリ")

        args, kwargs = interaction.response.send_message.call_args
        self.assertEqual(args[0], character_dictionary_command.NOT_FOUND_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])

    async def test_character_info_ambiguous_match_lists_candidates(self):
        command = self.client.tree.commands["character_info"]
        interaction = self._interaction()

        # "chi" partially matches both "nanami chiaki" and "shikano chinatsu".
        await command(interaction, character="chi")

        args, kwargs = interaction.response.send_message.call_args
        self.assertIn("複数のキャラクター", args[0])
        self.assertTrue(kwargs["ephemeral"])
        self.assertNotIn("embed", kwargs)

    async def test_character_search_by_series_alias_groups_all_matches(self):
        command = self.client.tree.commands["character_search"]
        interaction = self._interaction()

        await command(interaction, query="ブルアカ", page=1)

        _, kwargs = interaction.response.send_message.call_args
        embed = kwargs["embed"]
        self.assertIn("ブルアカ", embed.title)
        field_names = [field.name for field in embed.fields]
        self.assertTrue(any("ブルーアーカイブ" in name for name in field_names))

    async def test_character_search_no_match_returns_not_found_message(self):
        command = self.client.tree.commands["character_search"]
        interaction = self._interaction()

        await command(interaction, query="該当なしのクエリ", page=1)

        args, kwargs = interaction.response.send_message.call_args
        self.assertEqual(args[0], character_dictionary_command.NOT_FOUND_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])


class BuildCharacterDetailEmbedFutureFieldTestCase(unittest.TestCase):
    """extra_fields内の未知キーも表示できることを確認する（appearance_tags等が
    将来追加された場合に表示コードの変更が不要であることの保証）。"""

    def test_unknown_extra_field_key_falls_back_to_raw_name(self):
        from services.character_tag_dictionary import CharacterDetail

        detail = CharacterDetail(
            character_tag="dummy",
            display_name="ダミー",
            series_tag="dummy series",
            series_name="ダミー作品",
            extra_fields={"appearance_tags": ("long hair",), "brand_new_field": "some value"},
        )

        embed = character_dictionary_command.build_character_detail_embed(detail)

        field_by_name = {field.name: field.value for field in embed.fields}
        self.assertEqual(field_by_name["外見タグ"], "long hair")
        self.assertEqual(field_by_name["brand_new_field"], "some value")

    def test_unset_extra_field_value_displays_as_placeholder(self):
        from services.character_tag_dictionary import CharacterDetail

        detail = CharacterDetail(
            character_tag="dummy",
            display_name="ダミー",
            series_tag="dummy series",
            series_name="ダミー作品",
            extra_fields={"identity_hints": (), "notes": ""},
        )

        embed = character_dictionary_command.build_character_detail_embed(detail)

        field_by_name = {field.name: field.value for field in embed.fields}
        self.assertEqual(field_by_name["識別ヒント"], "未設定")
        self.assertEqual(field_by_name["備考"], "未設定")


class CharacterMissingCommandTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = FakeClient()
        character_dictionary_command.setup(self.client)
        self.command = self.client.tree.commands["character_missing"]

    @staticmethod
    def _interaction(user_id: int = 1):
        interaction = mock.MagicMock()
        interaction.user.id = user_id
        interaction.response.send_message = mock.AsyncMock()
        interaction.response.defer = mock.AsyncMock()
        interaction.followup.send = mock.AsyncMock()
        return interaction

    async def test_denied_for_non_allowed_user(self):
        interaction = self._interaction(user_id=999)
        with mock.patch("commands.character_dictionary.is_character_admin_allowed", return_value=False):
            await self.command(interaction)

        interaction.response.send_message.assert_awaited_once()
        args, kwargs = interaction.response.send_message.call_args
        self.assertEqual(args[0], character_dictionary_command.CHARACTER_ADMIN_DENIED_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])
        interaction.response.defer.assert_not_awaited()

    async def test_allowed_user_with_no_candidates_gets_clear_zero_result_message(self):
        interaction = self._interaction(user_id=1)
        with (
            mock.patch("commands.character_dictionary.is_character_admin_allowed", return_value=True),
            mock.patch("commands.character_dictionary.find_missing_character_candidates", return_value=[]),
        ):
            await self.command(interaction)

        interaction.followup.send.assert_awaited_once()
        args, kwargs = interaction.followup.send.call_args
        self.assertEqual(args[0], character_dictionary_command.NO_MISSING_CANDIDATES_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])

    async def test_allowed_user_sees_candidate_embed(self):
        from services.character_missing_finder import MissingCandidate

        candidate = MissingCandidate(
            character_tag="suzuna",
            display_name="スズナ",
            series_tag="pokemon",
            series_name="ポケットモンスター",
            matched_alias="スズナ",
            occurrence_count=4,
            last_used_at="2026-01-01T10:00:00+09:00",
            sample_request_text="スズナが公園にいる",
            reason="作品名「ポケットモンスター」が同時に指定されていないため、現在の辞書では解決できません。",
        )
        interaction = self._interaction(user_id=1)
        with (
            mock.patch("commands.character_dictionary.is_character_admin_allowed", return_value=True),
            mock.patch(
                "commands.character_dictionary.find_missing_character_candidates", return_value=[candidate]
            ),
        ):
            await self.command(interaction)

        interaction.followup.send.assert_awaited_once()
        _, kwargs = interaction.followup.send.call_args
        embed = kwargs["embed"]
        self.assertIsInstance(embed, discord.Embed)
        self.assertIn("スズナ — 4回", [field.name for field in embed.fields])

    async def test_scan_failure_returns_short_error_without_raising(self):
        interaction = self._interaction(user_id=1)
        with (
            mock.patch("commands.character_dictionary.is_character_admin_allowed", return_value=True),
            mock.patch(
                "commands.character_dictionary.find_missing_character_candidates",
                side_effect=RuntimeError("db exploded"),
            ),
        ):
            await self.command(interaction)

        interaction.followup.send.assert_awaited_once()
        args, kwargs = interaction.followup.send.call_args
        self.assertEqual(args[0], character_dictionary_command.MISSING_SCAN_FAILED_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])

    async def test_candidate_embed_never_exposes_full_raw_request_text(self):
        # sample_request_text is already truncated by
        # services/character_missing_finder.py before it ever reaches this
        # command layer - this asserts the command doesn't widen it back out.
        from services.character_missing_finder import SAMPLE_REQUEST_TEXT_MAX_LENGTH, MissingCandidate

        long_original = "スズナが" + ("あ" * 500)
        candidate = MissingCandidate(
            character_tag="suzuna",
            display_name="スズナ",
            series_tag="pokemon",
            series_name="ポケットモンスター",
            matched_alias="スズナ",
            occurrence_count=1,
            last_used_at="2026-01-01T10:00:00+09:00",
            sample_request_text=long_original[: SAMPLE_REQUEST_TEXT_MAX_LENGTH] + "…",
            reason="作品名「ポケットモンスター」が同時に指定されていないため、現在の辞書では解決できません。",
        )
        interaction = self._interaction(user_id=1)
        with (
            mock.patch("commands.character_dictionary.is_character_admin_allowed", return_value=True),
            mock.patch(
                "commands.character_dictionary.find_missing_character_candidates", return_value=[candidate]
            ),
        ):
            await self.command(interaction)

        _, kwargs = interaction.followup.send.call_args
        embed = kwargs["embed"]
        combined = "\n".join(field.value for field in embed.fields)
        self.assertNotIn(long_original, combined)


class CharacterValidateCommandTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = FakeClient()
        character_dictionary_command.setup(self.client)
        self.command = self.client.tree.commands["character_validate"]

    @staticmethod
    def _interaction():
        interaction = mock.MagicMock()
        interaction.response.send_message = mock.AsyncMock()
        return interaction

    async def test_no_permission_gate_general_use(self):
        # /character_validate is intentionally general-use (read-only over
        # the dictionary itself, no per-user history) - it must not check
        # is_character_admin_allowed at all.
        interaction = self._interaction()
        with mock.patch("commands.character_dictionary.is_character_admin_allowed") as gate:
            await self.command(interaction)
        gate.assert_not_called()

    async def test_real_dictionary_reports_zero_errors_inline(self):
        interaction = self._interaction()

        await self.command(interaction)

        interaction.response.send_message.assert_awaited_once()
        _, kwargs = interaction.response.send_message.call_args
        embed = kwargs["embed"]
        field_by_name = {field.name: field.value for field in embed.fields}
        self.assertEqual(field_by_name["ERROR"], "0")
        self.assertEqual(field_by_name["結果"], "使用可能")
        self.assertNotIn("file", kwargs)

    async def test_long_detail_report_is_attached_as_a_file_not_inlined(self):
        from services.character_dictionary_validator import SEVERITY_ERROR, ValidationIssue, ValidationReport

        many_issues = tuple(
            ValidationIssue(
                severity=SEVERITY_ERROR,
                code="duplicate_character_tag",
                message="x" * 200,
                subject=f"char-{i}",
            )
            for i in range(30)
        )
        fake_report = ValidationReport(character_count=30, series_count=1, issues=many_issues)

        interaction = self._interaction()
        with mock.patch("commands.character_dictionary.validate_character_dictionary", return_value=fake_report):
            await self.command(interaction)

        interaction.response.send_message.assert_awaited_once()
        _, kwargs = interaction.response.send_message.call_args
        self.assertIn("file", kwargs)
        self.assertIsInstance(kwargs["file"], discord.File)
        self.assertLessEqual(len(kwargs["content"]), character_dictionary_command.VALIDATION_DETAIL_INLINE_LIMIT)

    async def test_validation_failure_returns_short_error_without_raising(self):
        interaction = self._interaction()
        with mock.patch(
            "commands.character_dictionary.validate_character_dictionary",
            side_effect=RuntimeError("boom"),
        ):
            await self.command(interaction)

        interaction.response.send_message.assert_awaited_once()
        args, kwargs = interaction.response.send_message.call_args
        self.assertEqual(args[0], character_dictionary_command.VALIDATION_FAILED_MESSAGE)
        self.assertTrue(kwargs["ephemeral"])


if __name__ == "__main__":
    unittest.main()
