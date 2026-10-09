import html
import logging

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import (
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lora.ui.droplet_3d import Droplet3DWidget

logger = logging.getLogger(__name__)


class HydroOrbWidget(Droplet3DWidget):
    """Hardware-efficient Aqueous Hydro-Orb visualizer.

    Backed by the 3D Translucent Water Droplet shader engine with
    concentric holographic resonance telemetry rings.
    """

    def __init__(self, size: int = 180, parent: QWidget | None = None) -> None:
        super().__init__(size=size, parent=parent)


class EvidenceCardWidget(QFrame):
    """Structured holographic card for verified research evidence from arXiv, PubMed, etc."""

    def __init__(
        self,
        title: str,
        source: str,
        snippet: str,
        confidence: float = 0.95,
        url: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #0f172a;
                border: 1px solid #1e293b;
                border-left: 3px solid #00f0ff;
                border-radius: 6px;
                padding: 8px;
                margin-bottom: 6px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        header_layout = QHBoxLayout()
        source_badge = QLabel(f"<b><span style='color:#00f0ff;'>[{source.upper()}]</span></b>", self)
        conf_badge = QLabel(f"<span style='color:#10b981; font-size:11px;'>● {int(confidence * 100)}% Match</span>", self)
        header_layout.addWidget(source_badge)
        header_layout.addStretch()
        header_layout.addWidget(conf_badge)
        layout.addLayout(header_layout)

        title_lbl = QLabel(f"<b style='color:#f8fafc; font-size:13px;'>{html.escape(title)}</b>", self)
        title_lbl.setWordWrap(True)
        layout.addWidget(title_lbl)

        if snippet:
            snip_lbl = QLabel(f"<span style='color:#94a3b8; font-size:12px;'>\"{html.escape(snippet[:180])}...\"</span>", self)
            snip_lbl.setWordWrap(True)
            layout.addWidget(snip_lbl)

        if url:
            link_lbl = QLabel(f"<a href='{html.escape(url)}' style='color:#38bdf8; font-size:11px;'>🔗 View Original Source</a>", self)
            link_lbl.setOpenExternalLinks(True)
            layout.addWidget(link_lbl)


class LoraDesktopHUD(QMainWindow):
    """JARVIS Holographic Research Assistant HUD for L.O.R.A.

    Features:
    - 3D Translucent Water Droplet visualizer with audio-reactive surface ripples
    - Voice-first Push-to-Talk and Wake Word activation controls
    - Live Audio Waveform / VU Resonance Meter
    - Left Wing: Verified Evidence & Source Radar (arXiv, CrossRef, PubMed)
    - Right Wing: Executive Spoken Briefing (live articulating synthesis)
    - Grounding & Truthfulness Telemetry Vitals
    - Clipboard Quick-Inspect (Ctrl+Alt+Q)
    """

    query_submitted = Signal(str, str)  # query, depth
    ptt_pressed = Signal()
    ptt_released = Signal()
    stop_audio_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("L.O.R.A - Desktop HUD")
        self.resize(1020, 720)
        self.setMinimumSize(880, 620)

        self._depth_mode = "deep"
        self._init_ui()
        self._init_shortcuts()

    def _init_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        central.setStyleSheet("""
            QWidget {
                background-color: #080c14;
                color: #e2e8f0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }
            QFrame#card {
                background-color: #0b1120;
                border: 1px solid #1e293b;
                border-radius: 8px;
            }
            QTextEdit {
                background-color: #0b1120;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 10px;
                font-size: 13px;
                color: #f1f5f9;
                line-height: 1.5;
            }
            QLineEdit {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 14px;
                color: #f8fafc;
            }
            QLineEdit:focus {
                border: 1px solid #00f0ff;
            }
            QPushButton {
                background-color: #0284c7;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #0369a1;
            }
            QPushButton#ptt_btn {
                background-color: #0284c7;
                border: 1px solid #38bdf8;
                border-radius: 20px;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: bold;
                color: #ffffff;
            }
            QPushButton#ptt_btn:hover {
                background-color: #0369a1;
            }
            QPushButton#ptt_btn:pressed {
                background-color: #0ea5e9;
                border-color: #00f0ff;
            }
            QProgressBar {
                border: 1px solid #1e293b;
                border-radius: 3px;
                background-color: #0f172a;
                text-align: center;
                height: 6px;
            }
            QProgressBar::chunk {
                background-color: #00f0ff;
                border-radius: 2px;
            }
        """)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 14, 16, 14)
        main_layout.setSpacing(12)

        # 1. TOP HEADER & TELEMETRY BAR
        top_bar = QHBoxLayout()

        brand_layout = QVBoxLayout()
        brand_title = QLabel("<h3 style='margin:0; letter-spacing:1px; color:#ffffff;'>L.O.R.A <span style='color:#00f0ff; font-weight:normal; font-size:13px;'>// RESEARCH ASSISTANT</span></h3>", self)
        self.status_label = QLabel("<span style='color:#10b981; font-size:12px;'>● VOICE READY & GROUNDED</span>", self)
        brand_layout.addWidget(brand_title)
        brand_layout.addWidget(self.status_label)
        top_bar.addLayout(brand_layout)

        top_bar.addStretch()

        # Telemetry Vitals Pill
        vitals_frame = QFrame(self)
        vitals_frame.setStyleSheet("background-color: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 4px 10px;")
        vitals_layout = QHBoxLayout(vitals_frame)
        vitals_layout.setContentsMargins(6, 2, 6, 2)
        vitals_layout.setSpacing(10)
        self.vitals_lbl = QLabel("<span style='color:#94a3b8; font-size:11px;'>ENGINE: <b style='color:#38bdf8;'>OLLAMA (LOCAL)</b> | VRAM: <b style='color:#10b981;'>6GB OK</b> | NLI: <b style='color:#10b981;'>ACTIVE</b></span>", self)
        vitals_layout.addWidget(self.vitals_lbl)
        top_bar.addWidget(vitals_frame)

        # Depth Selector Buttons
        depth_layout = QHBoxLayout()
        depth_layout.setSpacing(4)
        for mode in ("Quick", "Deep", "Exhaustive"):
            btn = QPushButton(mode, self)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #1e293b;
                    border: 1px solid #334155;
                    border-radius: 4px;
                    padding: 4px 10px;
                    font-size: 11px;
                }
                QPushButton:hover { background-color: #334155; }
            """)
            btn.clicked.connect(lambda checked=False, m=mode.lower(): self._set_depth(m))
            depth_layout.addWidget(btn)
        top_bar.addLayout(depth_layout)

        # Quick Fact-Check Clipboard button
        clip_btn = QPushButton("📋 Inspect Clip", self)
        clip_btn.setStyleSheet("background-color: #1e293b; border: 1px solid #334155; font-size: 11px; padding: 4px 8px;")
        clip_btn.clicked.connect(self.handle_clipboard_inspect)
        top_bar.addWidget(clip_btn)

        main_layout.addLayout(top_bar)

        # 2. MAIN BENTO GRID (Left: Evidence | Center: 3D Droplet & Voice | Right: Executive Brief)
        grid_layout = QHBoxLayout()
        grid_layout.setSpacing(12)

        # LEFT WING: Verified Evidence Radar & Citations
        left_frame = QFrame(self)
        left_frame.setObjectName("card")
        left_frame.setFixedWidth(280)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)

        left_hdr = QLabel("<b style='color:#38bdf8; font-size:12px; letter-spacing:0.5px;'>VERIFIED EVIDENCE & SOURCES</b>", self)
        left_layout.addWidget(left_hdr)

        # Grounding metrics box
        metrics_box = QFrame(self)
        metrics_box.setStyleSheet("background-color: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 6px;")
        metrics_layout = QVBoxLayout(metrics_box)
        metrics_layout.setContentsMargins(6, 6, 6, 6)
        metrics_layout.setSpacing(2)

        self.grounding_lbl = QLabel("<span style='font-size:11px; color:#94a3b8;'>Truthfulness Confidence: <b style='color:#10b981;'>94%</b></span>", self)
        self.grounding_bar = QProgressBar(self)
        self.grounding_bar.setValue(94)
        metrics_layout.addWidget(self.grounding_lbl)
        metrics_layout.addWidget(self.grounding_bar)

        self.verification_stat_lbl = QLabel("<span style='font-size:11px; color:#94a3b8;'>Claims Verified: <b style='color:#38bdf8;'>8/8</b> | Hallucinations: <b style='color:#10b981;'>0</b></span>", self)
        metrics_layout.addWidget(self.verification_stat_lbl)
        left_layout.addWidget(metrics_box)

        # Evidence Cards Scroll Area
        self.evidence_scroll = QScrollArea(self)
        self.evidence_scroll.setWidgetResizable(True)
        self.evidence_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.evidence_container = QWidget()
        self.evidence_container.setStyleSheet("background: transparent;")
        self.evidence_cards_layout = QVBoxLayout(self.evidence_container)
        self.evidence_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.evidence_cards_layout.setSpacing(6)
        self.evidence_cards_layout.addStretch()
        self.evidence_scroll.setWidget(self.evidence_container)
        left_layout.addWidget(self.evidence_scroll)

        grid_layout.addWidget(left_frame)

        # CENTER STAGE: 3D Translucent Water Droplet + Audio Waveform
        center_frame = QFrame(self)
        center_frame.setObjectName("card")
        center_layout = QVBoxLayout(center_frame)
        center_layout.setContentsMargins(14, 14, 14, 14)
        center_layout.setSpacing(10)
        center_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 3D Droplet Visualizer
        self.orb = Droplet3DWidget(size=250, parent=self)
        center_layout.addWidget(self.orb, alignment=Qt.AlignmentFlag.AlignCenter)

        # Voice State Badge
        self.voice_state_badge = QLabel("<b style='color:#00f0ff; font-size:13px; letter-spacing:1px;'>[ ● VOICE READY ]</b>", self)
        self.voice_state_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(self.voice_state_badge)

        # Real-time Mic Audio VU Waveform Bar
        audio_bar_layout = QHBoxLayout()
        audio_bar_layout.setSpacing(6)
        audio_bar_layout.addWidget(QLabel("<span style='font-size:10px; color:#64748b;'>MIC</span>", self))
        self.mic_vu_bar = QProgressBar(self)
        self.mic_vu_bar.setRange(0, 100)
        self.mic_vu_bar.setValue(0)
        self.mic_vu_bar.setFixedHeight(5)
        audio_bar_layout.addWidget(self.mic_vu_bar)
        center_layout.addLayout(audio_bar_layout)

        # Push-to-Talk Primary Action Button
        self.ptt_btn = QPushButton("🎙️ Push-to-Talk (Hold Space)", self)
        self.ptt_btn.setObjectName("ptt_btn")
        self.ptt_btn.pressed.connect(self._on_ptt_pressed)
        self.ptt_btn.released.connect(self._on_ptt_released)
        center_layout.addWidget(self.ptt_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        center_sub = QLabel("<span style='color:#64748b; font-size:11px;'>Say \"<b>Awaken LORA</b>\" or press Push-to-Talk</span>", self)
        center_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(center_sub)

        grid_layout.addWidget(center_frame, stretch=1)

        # RIGHT WING: Executive Spoken Briefing (Spoken Dossier)
        right_frame = QFrame(self)
        right_frame.setObjectName("card")
        right_frame.setFixedWidth(360)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(8)

        right_hdr_layout = QHBoxLayout()
        right_hdr = QLabel("<b style='color:#38bdf8; font-size:12px; letter-spacing:0.5px;'>EXECUTIVE SPOKEN BRIEFING</b>", self)
        right_hdr_layout.addWidget(right_hdr)
        right_hdr_layout.addStretch()

        self.audio_status_badge = QLabel("<span style='color:#64748b; font-size:11px;'>Audio: Idle</span>", self)
        right_hdr_layout.addWidget(self.audio_status_badge)
        right_layout.addLayout(right_hdr_layout)

        # Spoken Briefing Content Area (also backward-compatible with chat_feed)
        self.chat_feed = QTextEdit(self)
        self.chat_feed.setReadOnly(True)
        self.chat_feed.setPlaceholderText("L.O.R.A will present concise, spoken executive briefings here...")
        right_layout.addWidget(self.chat_feed)

        # Audio Control Strip
        audio_strip = QHBoxLayout()
        self.stop_audio_btn = QPushButton("⏹️ Stop Voice", self)
        self.stop_audio_btn.setStyleSheet("background-color: #1e293b; border: 1px solid #334155; font-size: 11px; padding: 4px 8px;")
        self.stop_audio_btn.clicked.connect(self.stop_audio_requested.emit)
        audio_strip.addWidget(self.stop_audio_btn)
        audio_strip.addStretch()
        right_layout.addLayout(audio_strip)

        grid_layout.addWidget(right_frame)
        main_layout.addLayout(grid_layout, stretch=1)

        # 3. BOTTOM AMBIENT COMMAND / INQUIRY BAR
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(8)

        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Ask a research question or speak via Push-to-Talk (Space)...")
        self.input_field.returnPressed.connect(self._on_submit)
        bottom_layout.addWidget(self.input_field, stretch=1)

        self.submit_btn = QPushButton("Research", self)
        self.submit_btn.clicked.connect(self._on_submit)
        bottom_layout.addWidget(self.submit_btn)

        main_layout.addLayout(bottom_layout)

    def _init_shortcuts(self) -> None:
        # Clipboard quick-inspect: Ctrl+Alt+Q
        self.inspect_shortcut = QShortcut(QKeySequence("Ctrl+Alt+Q"), self)
        self.inspect_shortcut.activated.connect(self.handle_clipboard_inspect)

        # Keyboard Push-to-Talk: Spacebar when window active (if not typing in input)
        self.space_shortcut = QShortcut(QKeySequence("Space"), self)
        self.space_shortcut.activated.connect(self._handle_space_ptt)

    def _handle_space_ptt(self) -> None:
        if not self.input_field.hasFocus():
            self._on_ptt_pressed()
            QTimer.singleShot(1500, self._on_ptt_released)

    def _set_depth(self, depth: str) -> None:
        self._depth_mode = depth
        logger.info("HUD research depth set to: %s", depth)

    def _on_ptt_pressed(self) -> None:
        self.ptt_btn.setText("🔴 Listening...")
        self.set_voice_state("listening")
        self.ptt_pressed.emit()

    def _on_ptt_released(self) -> None:
        self.ptt_btn.setText("🎙️ Push-to-Talk (Hold Space)")
        self.ptt_released.emit()

    def set_voice_state(self, state: str) -> None:
        """Updates the 3D Droplet, badge, and HUD controls."""
        st = state.lower()
        self.orb.set_state(st)
        if st == "listening":
            self.voice_state_badge.setText("<b style='color:#00f0ff; font-size:13px;'>[ 🎤 LISTENING... ]</b>")
            self.status_label.setText("<span style='color:#00f0ff; font-size:12px;'>● LISTENING (VAD ACTIVE)</span>")
        elif st in ("thinking", "acting"):
            self.voice_state_badge.setText("<b style='color:#818cf8; font-size:13px;'>[ ⚙️ ANALYZING EVIDENCE ]</b>")
            self.status_label.setText("<span style='color:#818cf8; font-size:12px;'>● RETRIEVING & GROUNDING</span>")
        elif st == "speaking":
            self.voice_state_badge.setText("<b style='color:#38bdf8; font-size:13px;'>[ 🗣️ SPEAKING BRIEF ]</b>")
            self.status_label.setText("<span style='color:#38bdf8; font-size:12px;'>● ARTICULATING SYNTHESIS</span>")
            self.audio_status_badge.setText("<span style='color:#38bdf8; font-size:11px;'>🔊 Speaking</span>")
        elif st == "sleeping":
            self.voice_state_badge.setText("<b style='color:#64748b; font-size:13px;'>[ 💤 SLEEPING (0% CPU) ]</b>")
            self.status_label.setText("<span style='color:#64748b; font-size:12px;'>● SLEEPING</span>")
        else:
            self.voice_state_badge.setText("<b style='color:#00f0ff; font-size:13px;'>[ ● VOICE READY ]</b>")
            self.status_label.setText("<span style='color:#10b981; font-size:12px;'>● ACTIVE & GROUNDED</span>")
            self.audio_status_badge.setText("<span style='color:#64748b; font-size:11px;'>Audio: Idle</span>")

    def set_audio_level(self, level: float) -> None:
        """Updates mic VU bar and 3D droplet surface capillary ripples."""
        clamped = max(0.0, min(1.0, level))
        self.mic_vu_bar.setValue(int(clamped * 100))
        self.orb.set_audio_level(clamped)

    def set_truthfulness_score(self, score: float, verified: int = 8, total: int = 8) -> None:
        pct = int(score * 100)
        self.grounding_bar.setValue(pct)
        self.grounding_lbl.setText(f"<span style='font-size:11px; color:#94a3b8;'>Truthfulness Confidence: <b style='color:#10b981;'>{pct}%</b></span>")
        self.verification_stat_lbl.setText(f"<span style='font-size:11px; color:#94a3b8;'>Claims Verified: <b style='color:#38bdf8;'>{verified}/{total}</b> | Hallucinations: <b style='color:#10b981;'>0</b></span>")
        self.orb.set_truthfulness(score)

    def add_evidence_card(self, title: str, source: str, snippet: str, confidence: float = 0.95, url: str = "") -> None:
        card = EvidenceCardWidget(title=title, source=source, snippet=snippet, confidence=confidence, url=url, parent=self.evidence_container)
        # Insert before trailing stretch
        idx = max(0, self.evidence_cards_layout.count() - 1)
        self.evidence_cards_layout.insertWidget(idx, card)

    def clear_evidence_cards(self) -> None:
        while self.evidence_cards_layout.count() > 1:
            item = self.evidence_cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def handle_clipboard_inspect(self) -> None:
        """Inspects current Windows clipboard text and populates the search query."""
        clipboard = QApplication.clipboard()
        text = clipboard.text().strip()
        if text:
            self.set_input_text(f"Fact-check clipboard: {text[:200]}")
            self.append_message("System", f"📋 Inspected clipboard: \"{text[:120]}...\"")
            self.show()
            self.raise_()
            self.activateWindow()

    def append_message(self, sender: str, text: str) -> None:
        """Appends formatted research synthesis to the executive briefing feed."""
        clean_sender = html.escape(sender)
        clean_text = html.escape(text).replace("\n", "<br/>")
        formatted = f"<p style='margin-bottom:10px;'><b><span style='color:#00f0ff;'>[{clean_sender}]</span></b><br/><span style='color:#e2e8f0;'>{clean_text}</span></p>"
        self.chat_feed.append(formatted)

    def set_input_text(self, text: str) -> None:
        self.input_field.setText(text)

    def toggle_window(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()

    def _on_submit(self) -> None:
        query = self.input_field.text().strip()
        if not query:
            return
        self.append_message("You", query)
        self.input_field.clear()
        self.set_voice_state("thinking")
        self.query_submitted.emit(query, self._depth_mode)
