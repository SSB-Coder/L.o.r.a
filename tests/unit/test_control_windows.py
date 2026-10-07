"""Unit tests for L.O.R.A Window Management tools (src/lora/tools/control/windows.py)."""

from unittest.mock import patch

import pytest

from lora.tools.control.windows import WindowManager


@pytest.fixture
def window_manager():
    return WindowManager()


def test_list_windows_structure(window_manager: WindowManager):
    mock_windows = [
        {"handle": 1001, "title": "Visual Studio Code - LORA"},
        {"handle": 1002, "title": "Untitled - Notepad"},
        {"handle": 1003, "title": ""},  # empty title should be filtered
    ]
    with patch.object(window_manager, "_enumerate_raw_windows", return_value=mock_windows):
        windows = window_manager.list_windows()
        assert len(windows) == 2
        assert windows[0]["title"] == "Visual Studio Code - LORA"
        assert windows[1]["title"] == "Untitled - Notepad"


def test_focus_window_match_success(window_manager: WindowManager):
    mock_windows = [
        {"handle": 2001, "title": "Google Chrome - Wikipedia"},
        {"handle": 2002, "title": "Calculator"},
    ]
    with patch.object(window_manager, "_enumerate_raw_windows", return_value=mock_windows), \
         patch.object(window_manager, "_bring_to_foreground", return_value=True):
        res = window_manager.focus_window("Chrome")
        assert res["status"] == "success"
        assert "Google Chrome - Wikipedia" in res["matched_window"]


def test_focus_window_no_match(window_manager: WindowManager):
    mock_windows = [
        {"handle": 3001, "title": "Notepad"},
    ]
    with patch.object(window_manager, "_enumerate_raw_windows", return_value=mock_windows):
        res = window_manager.focus_window("Spotify")
        assert res["status"] == "error"
        assert "No open window found" in res["message"]


def test_focus_window_invalid_regex_escaped(window_manager: WindowManager):
    mock_windows = [
        {"handle": 4001, "title": "[Notepad] - report.txt"},
    ]
    with patch.object(window_manager, "_enumerate_raw_windows", return_value=mock_windows), \
         patch.object(window_manager, "_bring_to_foreground", return_value=True):
        # "[Notepad" has unmatched bracket, should not raise re.error
        res = window_manager.focus_window("[Notepad")
        assert res["status"] == "success"
        assert "[Notepad]" in res["matched_window"]
