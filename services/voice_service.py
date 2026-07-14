from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable

import discord

from services.tts_service import remove_temp_audio


logger = logging.getLogger("secretary-bot.voice")
_STOP = object()
WORKER_CONNECTION_CHECK_SECONDS = 5.0
WORKER_SHUTDOWN_TIMEOUT_SECONDS = 5.0


class VoiceServiceError(RuntimeError):
    """Base error for Discord voice operations."""


class VoiceConnectionError(VoiceServiceError):
    """Raised when joining or moving a Discord voice connection fails."""


class VoiceNotConnectedError(VoiceServiceError):
    """Raised when audio cannot be queued because the Bot is disconnected."""


class VoicePlaybackError(VoiceServiceError):
    """Raised internally when FFmpeg or Discord playback fails."""


class VoiceJoinResult(str, Enum):
    CONNECTED = "connected"
    MOVED = "moved"
    ALREADY_CONNECTED = "already_connected"


@dataclass(frozen=True)
class QueuedAudio:
    path: Path


@dataclass
class GuildVoiceState:
    queue: asyncio.Queue[QueuedAudio | object] = field(default_factory=asyncio.Queue)
    worker: asyncio.Task[None] | None = None
    temp_paths: set[Path] = field(default_factory=set)
    stopping: bool = False


def _default_audio_source(path: Path) -> discord.AudioSource:
    return discord.FFmpegPCMAudio(str(path))


class VoiceService:
    def __init__(self, audio_source_factory: Callable[[Path], object] | None = None) -> None:
        self._states: dict[int, GuildVoiceState] = {}
        self._lock = asyncio.Lock()
        self._audio_source_factory = audio_source_factory or _default_audio_source

    def has_state(self, guild_id: int) -> bool:
        return guild_id in self._states

    def worker_is_running(self, guild_id: int) -> bool:
        state = self._states.get(guild_id)
        return bool(state and state.worker and not state.worker.done())

    @staticmethod
    def is_connected(guild: object) -> bool:
        voice_client = getattr(guild, "voice_client", None)
        return bool(voice_client and voice_client.is_connected())

    async def join_voice_channel(self, guild: discord.Guild, channel: discord.abc.Connectable) -> VoiceJoinResult:
        voice_client = guild.voice_client
        try:
            if voice_client is not None and voice_client.is_connected():
                current_channel = getattr(voice_client, "channel", None)
                if getattr(current_channel, "id", None) == getattr(channel, "id", None):
                    return VoiceJoinResult.ALREADY_CONNECTED
                await voice_client.move_to(channel)
                logger.info(
                    "Moved voice connection: guild_id=%s channel_id=%s",
                    guild.id,
                    getattr(channel, "id", None),
                )
                return VoiceJoinResult.MOVED

            if voice_client is not None:
                try:
                    await voice_client.disconnect(force=True)
                except Exception:
                    logger.debug("Failed to discard stale voice client for guild_id=%s", guild.id, exc_info=True)

            await channel.connect()
            logger.info(
                "Connected to voice channel: guild_id=%s channel_id=%s",
                guild.id,
                getattr(channel, "id", None),
            )
            return VoiceJoinResult.CONNECTED
        except Exception as exc:
            logger.exception("Could not connect or move Discord voice client: guild_id=%s", guild.id)
            raise VoiceConnectionError("ボイスチャンネルへ接続できませんでした。権限と音声用依存関係を確認してね。") from exc

    async def enqueue(self, guild: discord.Guild, path: Path) -> int:
        voice_client = guild.voice_client
        if voice_client is None or not voice_client.is_connected():
            remove_temp_audio(path)
            raise VoiceNotConnectedError("Botがボイスチャンネルへ接続していません。")

        async with self._lock:
            state = self._states.get(guild.id)
            if state is None or state.stopping:
                state = GuildVoiceState()
                self._states[guild.id] = state
            state.temp_paths.add(path)
            await state.queue.put(QueuedAudio(path))
            if state.worker is None or state.worker.done():
                state.worker = asyncio.create_task(
                    self._worker(guild, state),
                    name=f"voice-worker-{guild.id}",
                )
            queue_size = state.queue.qsize()

        logger.info("Queued voice audio: guild_id=%s queue_size=%s path=%s", guild.id, queue_size, path)
        return queue_size

    async def leave_guild(self, guild: discord.Guild) -> bool:
        voice_client = guild.voice_client
        had_connection = bool(voice_client and voice_client.is_connected())
        had_state = self.has_state(guild.id)
        await self._stop_guild(guild.id, voice_client=voice_client, disconnect=True)
        return had_connection or had_state

    async def handle_manual_disconnect(self, guild_id: int) -> None:
        logger.info("Cleaning voice state after disconnect: guild_id=%s", guild_id)
        await self._stop_guild(guild_id, voice_client=None, disconnect=False)

    async def close_all(self, voice_clients: Iterable[discord.VoiceClient] = ()) -> None:
        clients_by_guild = {
            voice_client.guild.id: voice_client
            for voice_client in voice_clients
            if getattr(voice_client, "guild", None) is not None
        }
        guild_ids = set(self._states) | set(clients_by_guild)
        if not guild_ids:
            return
        await asyncio.gather(
            *(
                self._stop_guild(
                    guild_id,
                    voice_client=clients_by_guild.get(guild_id),
                    disconnect=True,
                )
                for guild_id in guild_ids
            ),
            return_exceptions=True,
        )

    async def _stop_guild(self, guild_id: int, voice_client: object, disconnect: bool) -> None:
        async with self._lock:
            state = self._states.pop(guild_id, None)
            if state is not None:
                state.stopping = True
                self._drain_queue(state)
                await state.queue.put(_STOP)
                worker = state.worker
            else:
                worker = None

        if voice_client is not None:
            try:
                if voice_client.is_playing() or voice_client.is_paused():
                    voice_client.stop()
            except Exception:
                logger.warning("Could not stop voice playback: guild_id=%s", guild_id, exc_info=True)

        if worker is not None and worker is not asyncio.current_task() and not worker.done():
            try:
                await asyncio.wait_for(asyncio.shield(worker), timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.warning("Voice worker did not stop in time; cancelling: guild_id=%s", guild_id)
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

        if disconnect and voice_client is not None:
            try:
                await voice_client.disconnect(force=True)
            except Exception:
                logger.warning("Could not disconnect voice client: guild_id=%s", guild_id, exc_info=True)

        if state is not None:
            self._cleanup_state_files(state)
        logger.info("Voice state stopped: guild_id=%s", guild_id)

    async def _worker(self, guild: discord.Guild, state: GuildVoiceState) -> None:
        logger.info("Voice worker started: guild_id=%s", guild.id)
        try:
            while not state.stopping:
                try:
                    item = await asyncio.wait_for(
                        state.queue.get(),
                        timeout=WORKER_CONNECTION_CHECK_SECONDS,
                    )
                except asyncio.TimeoutError:
                    if not self.is_connected(guild):
                        logger.info("Voice worker found disconnected client: guild_id=%s", guild.id)
                        break
                    continue

                if item is _STOP:
                    state.queue.task_done()
                    break
                if not isinstance(item, QueuedAudio):
                    state.queue.task_done()
                    continue

                try:
                    await self._play_one(guild, item.path)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Voice playback failed; stopping worker: guild_id=%s", guild.id)
                    break
                finally:
                    remove_temp_audio(item.path)
                    state.temp_paths.discard(item.path)
                    state.queue.task_done()
        except asyncio.CancelledError:
            logger.info("Voice worker cancelled: guild_id=%s", guild.id)
            raise
        finally:
            self._drain_queue(state)
            self._cleanup_state_files(state)
            async with self._lock:
                if self._states.get(guild.id) is state:
                    self._states.pop(guild.id, None)
            logger.info("Voice worker stopped: guild_id=%s", guild.id)

    async def _play_one(self, guild: discord.Guild, path: Path) -> None:
        voice_client = guild.voice_client
        if voice_client is None or not voice_client.is_connected():
            raise VoiceNotConnectedError("Bot was disconnected before playback.")

        source = None
        loop = asyncio.get_running_loop()
        completed: asyncio.Future[None] = loop.create_future()

        def after_playback(error: Exception | None) -> None:
            loop.call_soon_threadsafe(self._complete_playback, completed, error)

        try:
            source = self._audio_source_factory(path)
            voice_client.play(source, after=after_playback)
            await completed
        except FileNotFoundError as exc:
            raise VoicePlaybackError("FFmpegが見つかりません。PATHとインストール状態を確認してください。") from exc
        except discord.ClientException as exc:
            raise VoicePlaybackError(f"Discord音声再生を開始できませんでした: {exc}") from exc
        finally:
            cleanup = getattr(source, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    logger.debug("Audio source cleanup failed: path=%s", path, exc_info=True)

    @staticmethod
    def _complete_playback(future: asyncio.Future[None], error: Exception | None) -> None:
        if future.done():
            return
        if error is None:
            future.set_result(None)
        else:
            future.set_exception(VoicePlaybackError(str(error)))

    @staticmethod
    def _drain_queue(state: GuildVoiceState) -> None:
        while True:
            try:
                item = state.queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if isinstance(item, QueuedAudio):
                remove_temp_audio(item.path)
                state.temp_paths.discard(item.path)
            state.queue.task_done()

    @staticmethod
    def _cleanup_state_files(state: GuildVoiceState) -> None:
        for path in list(state.temp_paths):
            remove_temp_audio(path)
            state.temp_paths.discard(path)
