import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from commands import claude_remote as claude_remote_commands
from services.claude_remote import (
    ERROR_COMMAND_NOT_FOUND,
    ERROR_DISABLED,
    ERROR_EARLY_EXIT,
    ERROR_NOT_LOGGED_IN,
    ERROR_PROJECT_NOT_FOUND,
    ERROR_REMOTE_UNAVAILABLE,
    ERROR_WORKSPACE_UNTRUSTED,
    ClaudeRemoteConfig,
    ClaudeRemoteError,
    ClaudeRemoteManager,
    build_process_command,
    extract_remote_url,
    summarize_output_error,
)
from services.pc_control import is_pc_control_allowed


URL = "https://claude.ai/code/session-abc123"


class FakeStream:
    def __init__(self, *lines: bytes) -> None:
        self.lines = list(lines)

    async def readline(self) -> bytes:
        await asyncio.sleep(0)
        if self.lines:
            return self.lines.pop(0)
        return b""


class FakeProcess:
    _next_pid = 9000

    def __init__(
        self,
        *,
        stdout: tuple[bytes, ...] = (),
        stderr: tuple[bytes, ...] = (),
        returncode: int | None = None,
        graceful_stops: bool = True,
        terminate_stops: bool = True,
    ) -> None:
        FakeProcess._next_pid += 1
        self.pid = FakeProcess._next_pid
        self.stdout = FakeStream(*stdout)
        self.stderr = FakeStream(*stderr)
        self.returncode = returncode
        self.graceful_stops = graceful_stops
        self.terminate_stops = terminate_stops
        self.signals: list[object] = []
        self.terminate_calls = 0
        self.kill_calls = 0
        self._exit_event = asyncio.Event()
        if returncode is not None:
            self._exit_event.set()

    async def wait(self) -> int:
        await self._exit_event.wait()
        assert self.returncode is not None
        return self.returncode

    def send_signal(self, value: object) -> None:
        self.signals.append(value)
        if self.graceful_stops:
            self.finish(0)

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.terminate_stops:
            self.finish(-15)

    def kill(self) -> None:
        self.kill_calls += 1
        self.finish(-9)

    def finish(self, returncode: int) -> None:
        self.returncode = returncode
        self._exit_event.set()


class ProcessFactory:
    def __init__(self, *processes: FakeProcess) -> None:
        self.processes = list(processes)
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def __call__(self, *args: object, **kwargs: object) -> FakeProcess:
        self.calls.append((args, kwargs))
        return self.processes.pop(0)


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


class FakeResponse:
    def __init__(self) -> None:
        self.sent: list[tuple[str, bool]] = []

    async def send_message(self, content: str, *, ephemeral: bool = False) -> None:
        self.sent.append((content, ephemeral))


class FakeInteraction:
    def __init__(self, user_id: int) -> None:
        self.user = type("FakeUser", (), {"id": user_id})()
        self.response = FakeResponse()


def make_config(project_dir: Path | None, **overrides: object) -> ClaudeRemoteConfig:
    values: dict[str, object] = {
        "enabled": True,
        "command": "claude",
        "project_dir": project_dir,
        "session_name": "初号ちゃん",
        "start_timeout_seconds": 0.1,
        "stop_timeout_seconds": 0.01,
    }
    values.update(overrides)
    return ClaudeRemoteConfig(**values)  # type: ignore[arg-type]


class ClaudeRemoteParsingTestCase(unittest.TestCase):
    def test_extract_remote_url(self):
        self.assertEqual(extract_remote_url(f"Open {URL} now"), URL)

    def test_extract_remote_url_with_ansi_control_sequences(self):
        line = f"\x1b[32mConnect: {URL}\x1b[0m"
        self.assertEqual(extract_remote_url(line), URL)

    def test_pc_control_permission_is_reused(self):
        with mock.patch.dict(os.environ, {"PC_CONTROL_ALLOWED_USER_IDS": "111, 222"}, clear=False):
            self.assertTrue(is_pc_control_allowed(222))
            self.assertFalse(is_pc_control_allowed(333))

    def test_workspace_trust_error_is_safely_classified(self):
        summary = summarize_output_error("Workspace is not trusted: raw details")
        self.assertEqual(summary[0], ERROR_WORKSPACE_UNTRUSTED)
        self.assertNotIn("raw details", summary[1])

    def test_remote_unavailable_error_is_safely_classified(self):
        summary = summarize_output_error("Remote control is not available for this account")
        self.assertEqual(summary[0], ERROR_REMOTE_UNAVAILABLE)
        self.assertNotIn("account", summary[1])

    def test_windows_exe_command_is_an_argument_array(self):
        argv, flags = build_process_command(
            r"C:\Tools\claude.exe", "初号ちゃん", os_name="nt", comspec=r"C:\Windows\System32\cmd.exe"
        )
        self.assertEqual(argv, [r"C:\Tools\claude.exe", "remote-control", "--name", "初号ちゃん"])
        self.assertIsInstance(flags, int)

    def test_windows_cmd_wrapper_uses_only_fixed_values(self):
        discord_user_input = "& whoami & echo injected"
        argv, _ = build_process_command(
            r"C:\Program Files\Claude\claude.cmd",
            "初号ちゃん",
            os_name="nt",
            comspec=r"C:\Windows\System32\cmd.exe",
        )
        self.assertEqual(argv[:4], [r"C:\Windows\System32\cmd.exe", "/d", "/s", "/c"])
        self.assertIn("claude.cmd", argv[4])
        self.assertIn("remote-control", argv[4])
        self.assertIn("初号ちゃん", argv[4])
        self.assertNotIn(discord_user_input, " ".join(argv))

    def test_windows_cmd_wrapper_rejects_shell_metacharacters_in_fixed_session(self):
        with self.assertRaises(ValueError):
            build_process_command(
                r"C:\Tools\claude.cmd",
                "初号ちゃん & whoami",
                os_name="nt",
                comspec=r"C:\Windows\System32\cmd.exe",
            )


class ClaudeRemoteCommandPermissionTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_permission_denial_is_ephemeral_and_precedes_disabled_state(self):
        config = make_config(None, enabled=False)
        manager = ClaudeRemoteManager(config_provider=lambda: config)
        client = FakeClient()
        claude_remote_commands.setup(client, manager)
        interaction = FakeInteraction(999)

        with mock.patch("commands.claude_remote.is_pc_control_allowed", return_value=False):
            await client.tree.commands["claude_remote_status"](interaction)

        self.assertEqual(len(interaction.response.sent), 1)
        self.assertTrue(interaction.response.sent[0][1])
        self.assertIn("許可されたユーザー", interaction.response.sent[0][0])
        self.assertNotIn("無効", interaction.response.sent[0][0])

    async def test_disabled_message_is_ephemeral_for_allowed_user(self):
        config = make_config(None, enabled=False)
        manager = ClaudeRemoteManager(config_provider=lambda: config)
        client = FakeClient()
        claude_remote_commands.setup(client, manager)
        interaction = FakeInteraction(111)

        with mock.patch("commands.claude_remote.is_pc_control_allowed", return_value=True):
            await client.tree.commands["claude_remote_status"](interaction)

        self.assertEqual(len(interaction.response.sent), 1)
        self.assertTrue(interaction.response.sent[0][1])
        self.assertIn("無効", interaction.response.sent[0][0])


class ClaudeRemoteManagerTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="claude_remote_test ")
        self.project_dir = Path(self.temp_dir.name)
        self.managers: list[ClaudeRemoteManager] = []

    async def asyncTearDown(self):
        for manager in self.managers:
            await manager.cleanup()
        self.temp_dir.cleanup()

    def manager(
        self,
        process: FakeProcess | None = None,
        *,
        config: ClaudeRemoteConfig | None = None,
        resolver=lambda _command: r"C:\Tools\claude.exe",
    ) -> tuple[ClaudeRemoteManager, ProcessFactory]:
        processes = () if process is None else (process,)
        factory = ProcessFactory(*processes)

        async def fake_windows_tree_terminator(_pid: int, *, force: bool) -> bool:
            assert process is not None
            if force:
                process.kill()
            else:
                process.terminate()
            return True

        manager = ClaudeRemoteManager(
            config_provider=lambda: config or make_config(self.project_dir),
            executable_resolver=resolver,
            process_factory=factory,
            windows_tree_terminator=fake_windows_tree_terminator,
        )
        self.managers.append(manager)
        return manager, factory

    async def test_feature_disabled(self):
        manager, factory = self.manager(config=make_config(self.project_dir, enabled=False))
        with self.assertRaises(ClaudeRemoteError) as raised:
            await manager.start()
        self.assertEqual(raised.exception.code, ERROR_DISABLED)
        self.assertEqual(factory.calls, [])

    async def test_project_directory_not_found(self):
        missing = self.project_dir / "missing"
        manager, factory = self.manager(config=make_config(missing))
        with self.assertRaises(ClaudeRemoteError) as raised:
            await manager.start()
        self.assertEqual(raised.exception.code, ERROR_PROJECT_NOT_FOUND)
        self.assertEqual(factory.calls, [])

    async def test_command_not_found(self):
        manager, factory = self.manager(resolver=lambda _command: None)
        with self.assertRaises(ClaudeRemoteError) as raised:
            await manager.start()
        self.assertEqual(raised.exception.code, ERROR_COMMAND_NOT_FOUND)
        self.assertEqual(factory.calls, [])

    async def test_normal_start_reads_url_without_blocking(self):
        process = FakeProcess(stdout=(f"\x1b[36m{URL}\x1b[0m\n".encode(),))
        manager, factory = self.manager(process)

        result = await manager.start()

        self.assertEqual(result.url, URL)
        self.assertFalse(result.already_running)
        self.assertEqual(len(factory.calls), 1)
        args, kwargs = factory.calls[0]
        self.assertEqual(args[1:], ("remote-control", "--name", "初号ちゃん"))
        self.assertEqual(Path(str(kwargs["cwd"])), self.project_dir.resolve())

    async def test_double_start_returns_existing_process(self):
        process = FakeProcess(stdout=(f"{URL}\n".encode(),))
        manager, factory = self.manager(process)
        first = await manager.start()
        second = await manager.start()
        self.assertEqual(first.pid, second.pid)
        self.assertTrue(second.already_running)
        self.assertEqual(len(factory.calls), 1)

    async def test_exit_before_url_returns_safe_login_error(self):
        process = FakeProcess(stderr=(b"Error: not logged in. Please log in.\n",), returncode=2)
        manager, _ = self.manager(process)
        with self.assertRaises(ClaudeRemoteError) as raised:
            await manager.start()
        self.assertEqual(raised.exception.code, ERROR_NOT_LOGGED_IN)
        self.assertNotIn("Error:", str(raised.exception))

    async def test_exit_before_url_without_known_output(self):
        process = FakeProcess(stderr=(b"unexpected raw secret value\n",), returncode=3)
        manager, _ = self.manager(process)
        with self.assertRaises(ClaudeRemoteError) as raised:
            await manager.start()
        self.assertEqual(raised.exception.code, ERROR_EARLY_EXIT)
        self.assertNotIn("secret value", str(raised.exception))

    async def test_start_timeout_leaves_process_available_for_status(self):
        process = FakeProcess()
        manager, _ = self.manager(process, config=make_config(self.project_dir, start_timeout_seconds=0.01))
        result = await manager.start()
        self.assertTrue(result.timed_out)
        status = await manager.status()
        self.assertTrue(status.running)
        self.assertFalse(status.url_available)

    async def test_normal_stop_and_state_cleanup(self):
        process = FakeProcess(stdout=(f"{URL}\n".encode(),))
        manager, _ = self.manager(process)
        await manager.start()

        result = await manager.stop()

        self.assertFalse(result.already_stopped)
        self.assertFalse(result.used_terminate)
        self.assertEqual(len(process.signals), 1)
        status = await manager.status()
        self.assertFalse(status.running)
        self.assertIsNone(status.pid)
        self.assertFalse(status.url_available)
        self.assertEqual(status.project_dir, self.project_dir)
        self.assertIsNone(manager._process)
        self.assertIsNone(manager._project_dir)
        self.assertEqual(manager._reader_tasks, [])

    async def test_stop_falls_back_to_terminate(self):
        process = FakeProcess(stdout=(f"{URL}\n".encode(),), graceful_stops=False, terminate_stops=True)
        manager, _ = self.manager(process)
        await manager.start()

        result = await manager.stop()

        self.assertTrue(result.used_terminate)
        self.assertFalse(result.used_kill)
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 0)

    async def test_stop_falls_back_to_kill(self):
        process = FakeProcess(stdout=(f"{URL}\n".encode(),), graceful_stops=False, terminate_stops=False)
        manager, _ = self.manager(process)
        await manager.start()

        result = await manager.stop()

        self.assertTrue(result.used_terminate)
        self.assertTrue(result.used_kill)
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 1)

    async def test_cleanup_is_idempotent(self):
        process = FakeProcess(stdout=(f"{URL}\n".encode(),))
        manager, _ = self.manager(process)
        await manager.start()
        await manager.cleanup()
        await manager.cleanup()
        self.assertIsNone(manager._process)
        self.assertEqual(manager._reader_tasks, [])


if __name__ == "__main__":
    unittest.main()
