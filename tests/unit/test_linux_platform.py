"""Platform selection must preserve Windows mappings and Wayland restrictions."""
from unittest.mock import Mock, patch

from lora.tools.control.apps import AppLauncher
from lora.tools.control.typing import foreground_title
from lora.tools.control.windows import WindowManager
from lora.voice.hotkey import PushToTalkListener


def test_linux_app_uses_fedora_mapping(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    with patch("subprocess.Popen") as launch:
        assert AppLauncher().open_app("calculator")["status"] == "success"
    launch.assert_called_once_with(["gnome-calculator"], shell=False)


def test_windows_app_mapping_is_preserved(monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    with patch("subprocess.Popen") as launch:
        assert AppLauncher().open_app("calculator")["status"] == "success"
    launch.assert_called_once_with(["calc.exe"], shell=False)


def test_wayland_does_not_inspect_or_focus_other_windows(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    with patch("subprocess.run") as run:
        wm = WindowManager()
        assert wm.list_windows() == []
        assert wm.focus_window("Editor")["status"] == "error"
        assert foreground_title() == ""
    run.assert_not_called()


def test_x11_window_enumeration_and_focus(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    with patch("subprocess.run", return_value=Mock(stdout="0x000000ab 0 fedora Text Editor\n", returncode=0)) as run:
        result = WindowManager().focus_window("Text Editor")
    assert result["status"] == "success"
    assert result["handle"] == 171
    assert run.call_args.args[0] == ["wmctrl", "-ia", "0xab"]


def test_wayland_hotkeys_do_not_start_partial_xwayland_listener(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    listener = PushToTalkListener()
    listener.start()
    assert listener._listener is None
