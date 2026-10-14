"""Runs the real HUD page (droplet.html) in jsdom: key handling, streaming API, escaping, WebGL loss."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PAGE = HERE.parents[1] / "src" / "lora" / "ui" / "droplet.html"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@pytest.fixture(scope="module")
def result():
    probe = subprocess.run(["node", "-e", "require('jsdom')"], capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("jsdom is not installed (npm i jsdom)")
    res = subprocess.run(
        ["node", str(HERE / "hud_page_check.js"), str(PAGE)], capture_output=True, text=True, timeout=90
    )
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def test_page_loads_without_errors_and_renders(result):
    assert result["errors"] == []
    assert result["render_loop_ran"] is True and result["starts_empty"] is True


def test_holding_space_sends_one_press_and_one_release(result):
    assert result["ptt_events"] == ["on_ptt_down", "on_ptt_up"]  # the old page sent ~27 presses
    assert result["space_while_typing_events"] == 0


def test_typing_shows_bubble_at_once_and_locks_input_while_thinking(result):
    assert result["submit_events"] == [{"text": "what is **up**?"}]
    assert result["user_bubbles_after_submit"] == 1 and result["pending_shown"] and result["chat_mode_on"]
    assert result["state_after_submit"] == "think" and result["send_disabled_while_thinking"]
    assert result["input_value_cleared"]
    # Python's beginStream for the same question must not add a second bubble / reply
    assert result["user_bubbles_after_begin"] == 1 and result["bot_messages_after_begin"] == 1


def test_streamed_markdown_is_rendered_and_never_parsed_as_html(result):
    assert result["status_text"] == "Searching the web"  # the animated dots replace the ellipsis
    for key in ("md_bold", "md_italic", "md_code", "md_list", "md_heading", "md_good_link", "md_no_js_link", "pre_has_text_not_html"):
        assert result[key] is True, key
    assert result["no_injected_img"] is True


def test_sources_dropdown_links_and_python_owned_state(result):
    assert result["sources_summary"] == "Sources (1)"
    assert result["open_url_events"] == ["https://example.org/p"] * 2  # citation and source link both open the page
    assert result["state_after_finish"] == "think"  # JS does not force a state; Python decides
    assert result["state_after_python_idle"] == "idle" and result["send_enabled_again"]
    assert result["late_update_ignored"] and result["notice_added"] and result["voice_turn_bubble"]


def test_mic_status_and_small_menus(result):
    assert result["mic_ok"] == "ok||Microphone: Realtek Mic"  # healthy mic: just a quiet dot
    assert result["mic_bad"] == "bad|Microphone off: no input device found"
    assert result["more_menu_open"] and result["more_menu_closes_after_pick"]
    assert result["voice_toggle"] == [{"enabled": False}] and result["voice_label"] == "Voice off"
    assert result["same_state_noop"] is True


def test_add_sources_is_its_own_dropdown_without_model_wording(result):
    assert result["menu_hidden_initially"] and result["menu_open"] and result["menu_closes_on_outside_click"]
    assert result["menu_events"] == ["on_browse_media", "on_remove_media_file"]
    assert result["srcbtn_label"] == "Sources (1)" and result["srcbtn_label_empty"] == "Add sources"
    assert result["no_model_words"] is True and result["no_remote_resources"] is True


def test_webgl_context_loss_pauses_then_recovers(result):
    assert result["draws_while_lost"] == 0
    assert result["draws_after_restore"] is True


def test_verification_line_is_quiet_and_expands_only_with_notes(result):
    assert result["check_line"] == "Partly checked · 2 of 3 claims supported · second opinion suggested changes · improved"
    assert result["check_notes"] == ["Not confirmed: Mamba was invented in 1962", "Reviewer: add the date"]
    assert result["clean_line"] == "Checked · 4 of 4 claims supported" and result["clean_is_plain"] is True
    assert result["no_meta_no_line"] is True
