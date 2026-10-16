"""L.O.R.A Desktop HUD - Clean Ethereal Water Droplet Holographic Interface."""

import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, QUrl, Signal, Slot
from PySide6.QtWidgets import QMainWindow, QWidget

try:
    from PySide6.QtWebChannel import QWebChannel
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
    from PySide6.QtWebEngineWidgets import QWebEngineView

    WEB_ENGINE_AVAILABLE = True
except Exception:
    WEB_ENGINE_AVAILABLE = False

from lora.grounding.media_vector_db import MediaVectorDB
from lora.tools.files.known import known_folder

logger = logging.getLogger(__name__)


class LoraWebEnginePage(QWebEnginePage if WEB_ENGINE_AVAILABLE else object):  # type: ignore
    """Custom WebEnginePage intercepting lora:// navigation actions from droplet.html."""

    def __init__(self, hud: "LoraDesktopHUD", parent: QWidget | None = None) -> None:
        if WEB_ENGINE_AVAILABLE:
            super().__init__(parent)
        self.hud = hud

    def acceptNavigationRequest(self, url: QUrl, nav_type: Any, is_main_frame: bool) -> bool:  # noqa: N802
        if url.scheme() == "lora":
            action = url.host()
            query_str = url.query()
            payload: dict[str, Any] = {}
            if query_str.startswith("payload="):
                try:
                    raw = urllib.parse.unquote(query_str[8:])
                    payload = json.loads(raw)
                except Exception as e:
                    logger.warning("Error parsing lora url payload: %s", e)
            self.hud._handle_web_action(action, payload)
            return False
        if WEB_ENGINE_AVAILABLE:
            return super().acceptNavigationRequest(url, nav_type, is_main_frame)
        return True


class HudBridge(QObject):
    """Direct bi-directional QWebChannel IPC bridge for droplet.html."""

    def __init__(self, hud: "LoraDesktopHUD") -> None:
        super().__init__()
        self.hud = hud

    @Slot(str, str)
    def send_action(self, action: str, payload_json: str) -> None:
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except Exception as e:
            logger.warning("Error parsing HudBridge action payload: %s", e)
            payload = {}
        self.hud._handle_web_action(action, payload)


class _DummyLabel:
    """Null-safe label adapter for background status text updates."""

    def setText(self, text: str) -> None:
        pass


class LoraDesktopHUD(QMainWindow):
    """Clean, Ethereal Water Droplet Holographic HUD for L.O.R.A."""

    query_submitted = Signal(str, str)  # query, depth or voice
    voice_changed = Signal(str)  # new voice name (default: af_sky)
    model_changed = Signal(str)  # "auto" or specific Ollama model tag
    skill_changed = Signal(str)  # "auto" or specific skill name
    ptt_pressed = Signal()
    ptt_released = Signal()
    stop_audio_requested = Signal()
    sleep_toggled = Signal(bool)
    voice_toggled = Signal(bool)  # spoken replies on/off
    _media_notice = Signal(str, bool)  # (message, refresh the list) from a worker thread
    _media_report = Signal(str)  # report text from a worker thread
    exit_requested = Signal()
    rewrite_requested = Signal(int, str)
    media_updated = Signal()

    def __init__(
        self,
        media_db: MediaVectorDB | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("L.O.R.A")
        self.resize(1160, 780)
        self.setMinimumSize(860, 580)

        self.media_db = media_db or MediaVectorDB()
        self._depth_mode = "deep"
        self._current_voice_state = "idle"
        self.status_label = _DummyLabel()
        self._last_level = 0.0
        self._media_notice.connect(self._on_media_notice)
        self._media_report.connect(self._on_media_report)
        self._last_level_time = 0.0
        self.web_view: Any = None
        self._web_loading = False
        self._pending_js: list[str] = []
        self._last_ipc_action: tuple[str, str, float] = ("", "", 0.0)
        self.pending_attachments: list[dict[str, Any]] = []

        self._init_ui()
        self._init_shortcuts()

    def _init_ui(self) -> None:
        html_path = Path(__file__).parent / "droplet.html"
        if WEB_ENGINE_AVAILABLE and html_path.exists():
            try:
                self.web_view = QWebEngineView(self)
                self.web_page = LoraWebEnginePage(self, self.web_view)
                self.web_view.setPage(self.web_page)
                self.web_view.titleChanged.connect(self._on_web_title_changed)

                settings = self.web_page.settings()
                settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
                settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
                settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)

                self.web_channel = QWebChannel(self.web_view)
                self.bridge = HudBridge(self)
                self.web_channel.registerObject("pyBridge", self.bridge)
                self.web_view.page().setWebChannel(self.web_channel)

                self._web_loading = True
                self.web_view.loadFinished.connect(self._on_web_load_finished)
                self.web_view.load(QUrl.fromLocalFile(str(html_path.resolve())))
                self.setCentralWidget(self.web_view)
                return
            except Exception as e:
                logger.error("Could not initialize clean droplet WebEngineView: %s", e)

        placeholder = QWidget(self)
        placeholder.setStyleSheet("background-color: #04111b;")
        self.setCentralWidget(placeholder)

    def _init_shortcuts(self) -> None:
        pass

    def _handle_space_ptt(self) -> None:
        """Programmatic press (kept for callers/tests). Release must be sent separately."""
        if self._current_voice_state in ("thinking", "think", "verify", "verifying", "acting", "sleep", "sleeping", "off"):
            return
        self._on_ptt_pressed()

    def _on_web_title_changed(self, title: str) -> None:
        """Receives IPC messages from droplet.html via document.title fallback."""
        if title.startswith("lora://"):
            try:
                raw = title[7:]
                data = json.loads(urllib.parse.unquote(raw))
                action = data.get("action", "")
                payload = data.get("payload", {})
                self._handle_web_action(action, payload)
            except Exception as e:
                logger.warning("Error parsing lora:// web title message: %s", e)
        elif title.startswith("LORA_MSG:"):
            try:
                data = json.loads(title[9:])
                action = data.get("action", "")
                payload = data.get("payload", {})
                self._handle_web_action(action, payload)
            except Exception as e:
                logger.warning("Error parsing LORA_MSG: %s", e)

    def _on_web_load_finished(self, ok: bool = True) -> None:
        """Flush queued JS calls once droplet.html is ready in QWebEngineView."""
        self._web_loading = False
        pending = list(self._pending_js)
        self._pending_js.clear()
        for script in pending:
            self._run_js(script)
        self.sync_media_files()

    def _handle_web_action(self, action: str, payload: dict[str, Any]) -> None:
        """Handles actions dispatched from the clean droplet.html interface."""
        now = time.monotonic()
        payload_key = json.dumps(payload, sort_keys=True) if isinstance(payload, dict) else str(payload)
        last_act, last_key, last_t = self._last_ipc_action
        if action == last_act and payload_key == last_key and (now - last_t) < 0.08:
            return
        self._last_ipc_action = (action, payload_key, now)
        if action == "on_user_submit":
            text = payload.get("text", "").strip()
            depth = payload.get("depth", self._depth_mode)
            atts = payload.get("attachments")
            self.pending_attachments = list(atts) if isinstance(atts, list) else []
            if text or self.pending_attachments:
                self.set_voice_state("thinking")
                self.query_submitted.emit(text or "Analyze the attached file(s).", depth)
        elif action == "on_ptt_down":
            self._on_ptt_pressed()
        elif action == "on_ptt_up":
            self._on_ptt_released()
        elif action == "on_voice_changed":
            voice = payload.get("voice", "af_sky")
            self.voice_changed.emit(voice)
        elif action == "on_model_changed":
            model = str(payload.get("model", "auto")).strip()
            self.model_changed.emit(model)
        elif action == "on_skill_changed":
            skill = str(payload.get("skill", "auto")).strip()
            self.skill_changed.emit(skill)
        elif action == "on_sleep_toggle":
            is_sleeping = payload.get("is_sleeping", False)
            self.sleep_toggled.emit(is_sleeping)
        elif action == "on_open_url":
            self._open_external_url(str(payload.get("url", "")))
        elif action == "on_open_path":
            self._open_local_artifact(str(payload.get("path", "")))
        elif action == "on_voice_toggle":
            self.voice_toggled.emit(bool(payload.get("enabled", True)))
        elif action == "on_turn_off":
            self.exit_requested.emit()
        elif action == "on_rewrite":
            direction = payload.get("direction", 1)
            depth = payload.get("depth", self._depth_mode)
            self.rewrite_requested.emit(direction, depth)
        elif action == "on_add_media_file":
            self._add_media_paths([str(payload.get("path", "")).strip()])
        elif action == "on_remove_media_file":
            path_or_id = str(payload.get("path_or_id", "")).strip()
            if path_or_id:
                self._media_task(
                    lambda: self.media_db.remove_file(path_or_id),
                    lambda res: (f"Removed {res.get('filename', 'the file')}.", True),
                    "Removing…",
                )
                self.media_updated.emit()
        elif action == "on_browse_media":
            QTimer.singleShot(0, self._browse_media)
        elif action == "on_report_media":
            q = str(payload.get("query", "")).strip()
            self._media_task(
                lambda: self.media_db.report_on_media(query=q),
                lambda rep: (None, False),
                "Reading your sources…",
                report=True,
            )

    @staticmethod
    def _open_local_artifact(path_str: str) -> None:
        if not path_str:
            return
        p = Path(path_str)
        if not p.exists():
            return
        try:
            if sys.platform == "win32":
                os.startfile(str(p))  # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as exc:
            logger.warning("Could not open artifact path %s: %s", p, exc)

    # ------------------------------------------------------------------ media (background work)
    def _browse_media(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        chosen, _ = QFileDialog.getOpenFileNames(
            self,
            "Add sources",
            str(known_folder("documents")),
            "Documents (*.pdf *.md *.txt *.docx *.csv *.png *.jpg *.jpeg);;All files (*)",
        )
        self._add_media_paths(list(chosen))

    def _add_media_paths(self, paths: list[str]) -> None:
        paths = [p for p in paths if p]
        if not paths:
            return
        for path in paths:
            name = Path(path).name or path
            self._media_task(
                lambda path=path: self.media_db.add_file(path),
                lambda res, name=name: self._describe_add(res, name),
                f"Adding {name}… large files can take a minute.",
            )

    @staticmethod
    def _describe_add(res: dict[str, Any], name: str) -> tuple[str, bool]:
        status = res.get("status")
        if status == "exists":
            return f"{name} is already in your sources.", False
        if status == "error":
            return f"I couldn't add {name}: {res.get('message', 'unknown problem')}", False
        return f"Added {res.get('filename', name)} to your sources.", True

    def _media_task(self, work, describe, start_text: str, report: bool = False) -> None:
        """Run indexing work on a background thread; results come back through queued signals."""
        self._media_notice.emit(start_text, False)

        def _run() -> None:
            try:
                res = work()
                if report:
                    body = "Here's what's in your sources:\n" + str(res.get("summary", ""))
                    findings = res.get("findings") or []
                    if findings:
                        body += "\n\n" + "\n".join(f"- {f}" for f in findings)
                    self._media_report.emit(body)
                    return
                text, refresh = describe(res)
                self._media_notice.emit(text, refresh)
            except Exception as exc:
                logger.exception("Media task failed")
                self._media_notice.emit(f"That didn't work: {type(exc).__name__}: {exc}", False)

        threading.Thread(target=_run, daemon=True).start()

    @Slot(str, bool)
    def _on_media_notice(self, text: str, refresh: bool) -> None:
        if text:
            self.append_message("System", text)
        if refresh:
            self.sync_media_files()

    @Slot(str)
    def _on_media_report(self, text: str) -> None:
        self.present_answer("Summarize my sources", text)

    def _run_js(self, script: str) -> None:
        """Executes JavaScript safely in the droplet page (queued if page is still loading)."""
        if self.web_view is not None:
            if getattr(self, "_web_loading", False) and WEB_ENGINE_AVAILABLE and isinstance(self.web_view, QWebEngineView):
                if len(self._pending_js) < 30:
                    self._pending_js.append(script)
                return
            try:
                self.web_view.page().runJavaScript("(function(){\n" + script + "\n})();")
            except Exception:
                pass

    def set_sources(self, sources: list[Any]) -> None:
        """Attach clickable sources to the currently streaming answer bubble immediately."""
        self._run_js(f"window.setSources && window.setSources({json.dumps(sources or [])});")

    def update_model_list(self, models: list[str], active_model: str = "auto") -> None:
        """Populate the top-bar Model Selector dropdown with installed Ollama models."""
        self._run_js(f"window.updateModelList && window.updateModelList({json.dumps(models or [])}, {json.dumps(active_model or 'auto')});")

    def set_active_skill(self, skill_name: str = "auto") -> None:
        """Sync the active skill selector in droplet.html."""
        self._run_js(f"window.setActiveSkill && window.setActiveSkill({json.dumps(skill_name or 'auto')});")

    def set_active_voice(self, voice_name: str = "af_sky") -> None:
        """Sync the active Kokoro voice selector in droplet.html."""
        self._run_js(f"window.setActiveVoice && window.setActiveVoice({json.dumps(voice_name or 'af_sky')});")

    def sync_media_files(self) -> None:
        """Synchronizes active indexed media files to droplet drawer."""
        files = self.media_db.list_files()
        js_data = json.dumps(files)
        self._run_js(f"window.updateMediaList && window.updateMediaList({js_data});")

    def present_answer(
        self,
        query: str,
        answer: str,
        sources: list[Any] | None = None,
        confidence: float = 0.95,
        label: str = "",
        streamed: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Presents an answer. ``streamed=True`` means the text is already on screen."""
        js_q = json.dumps(query)
        js_ans = json.dumps(answer)
        js_src = json.dumps(sources or [])
        js_label = json.dumps(label)
        self._run_js(
            "window.finishTurn && window.finishTurn("
            f"{js_q}, {js_ans}, {js_src}, {confidence}, {js_label}, {str(streamed).lower()}, {json.dumps(meta or {})});"
        )

    @staticmethod
    def _open_external_url(url: str) -> None:
        """Open a link from a reply in the default browser (http/https only)."""
        parsed = urllib.parse.urlparse(url.strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            logger.warning("Blocked non-http link from the page: %r", url[:80])
            return
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl(url.strip()))

    def set_status(self, text: str) -> None:
        """Short progress line while LORA works ("Searching the web…")."""
        self._run_js(f"window.setStatus && window.setStatus({json.dumps(text)});")

    def stream_begin(self, query: str, label: str = "") -> None:
        """Show the question and a 'working' line while the model loads / starts."""
        self._run_js(f"window.beginStream && window.beginStream({json.dumps(query)}, {json.dumps(label)});")

    def stream_update(self, text: str) -> None:
        """Replace the caption with the text generated so far."""
        self._run_js(f"window.streamUpdate && window.streamUpdate({json.dumps(text)});")

    def set_mic_status(self, ok: bool, text: str) -> None:
        """Show whether a microphone is really open (name) or why not."""
        self._run_js(f"window.setMicStatus && window.setMicStatus({str(bool(ok)).lower()}, {json.dumps(text)});")

    def _on_ptt_pressed(self) -> None:
        if self._current_voice_state in ("thinking", "think", "verify", "verifying", "acting", "sleep", "sleeping", "off"):
            return
        self.set_voice_state("listening")
        self.ptt_pressed.emit()

    def _on_ptt_released(self) -> None:
        self.ptt_released.emit()

    def set_voice_state(self, state: str) -> None:
        """Updates the clean droplet state (idle, listen, think, speak, sleep)."""
        st = state.lower()
        self._current_voice_state = st
        web_mode = "idle"
        if st in ("listening", "listen"):
            web_mode = "listen"
        elif st in ("thinking", "think", "acting", "act"):
            web_mode = "think"
        elif st in ("verify", "verifying"):
            web_mode = "verify"
        elif st in ("speaking", "speak"):
            web_mode = "speak"
        elif st in ("sleeping", "sleep"):
            web_mode = "sleep"
        else:
            web_mode = "idle"

        self._run_js(f"window.setDropletState && window.setDropletState('{web_mode}');")

    def set_audio_level(self, level: float) -> None:
        """Updates droplet surface capillary ripples in real time."""
        clamped = max(0.0, min(1.0, level))
        now = time.monotonic()
        if clamped != 0.0 and abs(clamped - self._last_level) < 0.04 and now - self._last_level_time < 0.06:
            return
        self._last_level = clamped
        self._last_level_time = now
        self._run_js(f"window.setAudioLevel && window.setAudioLevel({clamped:.3f});")

    def pulse_droplet(self, amount: float = 0.6) -> None:
        """Triggers dynamic capillary ripple pulse on the droplet surface."""
        self._run_js(f"window.pulseDroplet && window.pulseDroplet({amount:.3f});")

    def set_truthfulness_score(self, score: float, verified: int = 8, total: int = 8) -> None:
        pass

    def add_evidence_card(self, title: str, source: str, snippet: str, confidence: float = 0.95, url: str = "") -> None:
        pass

    def clear_evidence_cards(self) -> None:
        pass

    def append_message(self, sender: str, text: str) -> None:
        """Streams message into the clean droplet dialog box."""
        js_sender = json.dumps(sender)
        js_text = json.dumps(text)
        self._run_js(f"window.addMessage && window.addMessage({js_sender}, {js_text});")

    def set_input_text(self, text: str) -> None:
        """Sets input field text in droplet interface."""
        js_text = json.dumps(text)
        self._run_js(f"const el = document.getElementById('q'); if (el) {{ el.value = {js_text}; }}")

    def toggle_window(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()

    def closeEvent(self, event: Any) -> None:
        """Emit exit_requested on window close to trigger clean microphone release."""
        self.exit_requested.emit()
        super().closeEvent(event)
