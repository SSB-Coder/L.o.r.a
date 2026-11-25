import json
import os
import socket
import stat
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.request import urlopen

import pytest

from lora.main import (
    GatewayStartupError,
    _private_mobile_token,
    build_argument_parser,
    start_gateway_thread,
)
from lora.safety.policy import PolicyEngine
from lora.ui.hud import LoraDesktopHUD
from lora.voice.enroll import EnrollmentSession
from lora.voice.wakeword import SpeakerVerifier


def test_start_preserves_options_given_before_subcommand(tmp_path):
    args = build_argument_parser().parse_args(["--no-audio", "--config-dir", str(tmp_path), "start", "--port", "0"])
    assert args.no_audio and args.config_dir == tmp_path and args.port == 0


def test_gateway_checks_real_listener_readiness_and_port_collision():
    with socket.socket() as unrelated:
        unrelated.bind(("127.0.0.1", 0))
        unrelated.listen()
        with pytest.raises(GatewayStartupError, match="Could not bind"):
            start_gateway_thread("127.0.0.1", unrelated.getsockname()[1])

    gateway = start_gateway_thread("127.0.0.1", 0)
    try:
        with urlopen(f"http://127.0.0.1:{gateway.config.port}/healthz", timeout=5) as response:
            assert json.load(response) == {"status": "ok"}
    finally:
        gateway.close()
    assert not gateway.thread.is_alive()


def test_mobile_credential_is_reused_without_public_file_permissions(tmp_path, monkeypatch):
    monkeypatch.delenv("LORA_MOBILE_TOKEN", raising=False)
    path = tmp_path / "mobile.txt"
    token = _private_mobile_token(path)
    assert len(token) >= 32 and _private_mobile_token(path) == token
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_failed_enrollment_preserves_existing_profile(tmp_path):
    path = tmp_path / "voice.json"
    verifier = SpeakerVerifier(profile_path=path)
    verifier.save_profile([[1.0] + [0.0] * 31])
    original = path.read_bytes()
    capture = Mock()
    capture.start.return_value = False
    capture.status_text.return_value = "Microphone permission denied"
    with patch("lora.voice.enroll.AudioCapture", return_value=capture), patch("builtins.input"):
        with pytest.raises(ValueError, match="Microphone permission denied"):
            EnrollmentSession(verifier, samples_needed=1, clip_duration_seconds=0).run_interactive()
    assert path.read_bytes() == original


def test_hud_only_hands_off_registered_safe_files(tmp_path):
    note = tmp_path / "note.md"
    other = tmp_path / "other.md"
    note.touch()
    other.touch()
    policy = PolicyEngine(allowed_roots=[str(tmp_path)])
    hud = SimpleNamespace(_artifact_paths={}, append_message=Mock())
    with patch("lora.ui.hud.PolicyEngine", return_value=policy), patch("lora.ui.hud.open_path", return_value={"status": "success"}) as launch:
        LoraDesktopHUD._prepare_artifacts(hud, {"artifacts": [{"path": str(note)}]})
        LoraDesktopHUD._open_local_artifact(hud, str(other))
        launch.assert_not_called()
        LoraDesktopHUD._open_local_artifact(hud, str(note))
        launch.assert_called_once_with(note.resolve(), policy_engine=policy, wait=False)
