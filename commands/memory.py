import asyncio
import logging

import discord
from discord import app_commands

from services.memory import add_memory, delete_memory, list_memories
from services.prompt_generator import PromptGeneratorError, PromptGeneratorTimeout, generate_chat_message


logger = logging.getLogger("secretary-bot.memory")


def setup(client: discord.Client) -> None:
    @client.tree.command(name="remember", description="好みや設定を簡易メモリに保存します。")
    @app_commands.describe(text="覚えておく内容")
    async def slash_remember(interaction: discord.Interaction, text: str) -> None:
        response = await asyncio.to_thread(add_memory, interaction.user.id, text)
        await interaction.response.send_message(response, ephemeral=True)

    @client.tree.command(name="memory_list", description="保存済みメモリを表示します。")
    async def slash_memory_list(interaction: discord.Interaction) -> None:
        response = await asyncio.to_thread(list_memories, interaction.user.id)
        await interaction.response.send_message(response, ephemeral=True)

    @client.tree.command(name="memory_delete", description="指定IDのメモリを削除します。")
    @app_commands.describe(id="削除するメモリID")
    async def slash_memory_delete(interaction: discord.Interaction, id: int) -> None:
        response = await asyncio.to_thread(delete_memory, interaction.user.id, id)
        await interaction.response.send_message(response, ephemeral=True)

    @client.tree.command(name="chat", description="LM Studioと簡単に会話します。保存済みメモリも参照します。")
    @app_commands.describe(text="話したい内容")
    async def slash_chat(interaction: discord.Interaction, text: str) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            response = await asyncio.to_thread(generate_chat_message, text, interaction.user.id)
        except PromptGeneratorTimeout:
            logger.exception("/chat timed out.")
            await interaction.followup.send("LM Studioから応答がありませんでした", ephemeral=True)
            return
        except PromptGeneratorError as exc:
            logger.exception("/chat failed.")
            await interaction.followup.send(str(exc), ephemeral=True)
            return
        except Exception:
            logger.exception("Unexpected /chat error.")
            await interaction.followup.send("会話に失敗しました。LM Studioの起動状態を確認してください。", ephemeral=True)
            return

        await interaction.followup.send(response[:1900], ephemeral=True)
