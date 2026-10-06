from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GenerationBackendLauncherContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = (ROOT / "launch_generation_backend.ps1").read_text(encoding="utf-8")
        cls.wrapper = (ROOT / "start_generation_backend.bat").read_text(encoding="utf-8")
        cls.gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    def test_launcher_fixes_project_root_and_loopback_port(self) -> None:
        self.assertIn("$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path", self.launcher)
        self.assertIn('$BackendUrl = "http://127.0.0.1:$Port"', self.launcher)
        self.assertIn('host = "127.0.0.1"', self.launcher)
        self.assertIn('fixed to the loopback Generation Backend port 8787', self.launcher)

    def test_health_gate_requires_configured_dependencies(self) -> None:
        self.assertIn("Test-BackendReady", self.launcher)
        self.assertIn("service_ready -eq $true", self.launcher)
        self.assertIn("lm_studio_configured -eq $true", self.launcher)
        self.assertIn("comfyui_url_configured -eq $true", self.launcher)
        self.assertIn("Get-NetTCPConnection -LocalPort $Port -State Listen", self.launcher)

    def test_launcher_is_detached_and_logs_both_streams(self) -> None:
        self.assertIn('start "" /b powershell.exe', self.launcher)
        self.assertIn("-WindowStyle Hidden", self.launcher)
        self.assertIn("-StdoutLog", self.launcher)
        self.assertIn("-StderrLog", self.launcher)
        helper = (ROOT / "run_generation_backend.ps1").read_text(encoding="utf-8")
        self.assertIn("1>> $StdoutLog", helper)
        self.assertIn("2>> $StderrLog", helper)
        self.assertIn("generation_backend", self.launcher)

    def test_launcher_has_duplicate_and_unhealthy_port_guards(self) -> None:
        self.assertIn('event = "already_healthy"', self.launcher)
        self.assertIn('event = "start_rejected_port_occupied"', self.launcher)
        self.assertIn("No duplicate process was started", self.launcher)
        self.assertIn('Local\\CodexGenerationBackendLauncher', self.launcher)

    def test_lifecycle_metadata_is_non_secret_and_python_is_not_a_secret_argument(self) -> None:
        self.assertIn("lifecycle.json", self.launcher)
        self.assertIn("backend.pid.json", self.launcher)
        self.assertIn("process_path = $python", self.launcher)
        self.assertIn("working_directory = $ProjectRoot", self.launcher)
        self.assertIn("launch_method", self.launcher)
        self.assertIn("run_generation_backend.ps1", self.launcher)
        self.assertNotIn("GALLERY_GENERATION_AUTH_TOKEN", self.launcher)
        self.assertNotIn("GALLERY_GENERATION_SESSION_SECRET", self.launcher)

    def test_python_runtime_can_be_configured_without_command_line_secrets(self) -> None:
        self.assertIn('GENERATION_BACKEND_PYTHON', self.launcher)
        self.assertIn('import aiohttp, dotenv', self.launcher)
        self.assertNotIn(".env", self.wrapper)

    def test_runtime_state_is_ignored(self) -> None:
        self.assertIn(".generation_backend/", self.gitignore)

    def test_batch_wrapper_calls_only_the_dedicated_launcher(self) -> None:
        self.assertIn("launch_generation_backend.ps1", self.wrapper)
        self.assertNotIn("generation_backend_server.py", self.wrapper)


if __name__ == "__main__":
    unittest.main()
