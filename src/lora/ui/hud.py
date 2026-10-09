import html
import logging
import math
from typing import override

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QKeySequence,
    QPainter,
    QRadialGradient,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class HydroOrbWidget(QWidget):
    """Hardware-efficient Aqueous Hydro-Orb visualizer.

    Draws concentric rippling water waves using QPainter with 0 MB VRAM impact.
    Capped at 30 FPS when active, 0 FPS when idle.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(140, 140)
        self._is_active = False
        self._phase = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 FPS
        self._timer.timeout.connect(self._animate_step)

    @property
    def is_active(self) -> bool:
        return self._is_active

    def set_active(self, active: bool) -> None:
        """Starts or stops animation loop to conserve 100% CPU when idle."""
        self._is_active = active
        if active:
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()
            self._phase = 0.0
            self.update()

    def _animate_step(self) -> None:
        self._phase += 0.08
        if self._phase > 2 * math.pi:
            self._phase -= 2 * math.pi
        self.update()

    @override
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        center = QPointF(w / 2.0, h / 2.0)
        base_radius = min(w, h) / 2.0 - 10.0

        # Base sphere gradient
        sphere_grad = QRadialGradient(center.x() - base_radius * 0.3, center.y() - base_radius * 0.3, base_radius * 1.3)
        sphere_grad.setColorAt(0.0, QColor(224, 242, 254, 240))  # Specular cyan light
        sphere_grad.setColorAt(0.3, QColor(14, 165, 233, 210))   # Aqueous electric cyan
        sphere_grad.setColorAt(0.8, QColor(2, 132, 199, 180))    # Deep ocean
        sphere_grad.setColorAt(1.0, QColor(15, 23, 42, 220))     # Midnight base
        painter.setBrush(sphere_grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, base_radius, base_radius)

        # Concentric ripple waves when active
        if self._is_active:
            for i in range(3):
                wave_offset = (self._phase + i * 1.5) % (2 * math.pi)
                wave_radius = base_radius * (0.4 + 0.55 * (wave_offset / (2 * math.pi)))
                alpha = int(180 * (1.0 - (wave_offset / (2 * math.pi))))
                wave_color = QColor(56, 189, 248, alpha)
                painter.setPen(wave_color)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(center, wave_radius, wave_radius)


class LoraDesktopHUD(QMainWindow):
    """Native PySide6 Desktop HUD for L.O.R.A with Aqueous Hydro-Orb and Clipboard Inspector."""

    query_submitted = Signal(str, str)  # query, depth

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("L.O.R.A - Desktop HUD")
        self.resize(720, 680)

        self._init_ui()
        self._init_shortcuts()

    def _init_ui(self) -> None:
        # Central widget and styling
        central = QWidget(self)
        self.setCentralWidget(central)
        central.setStyleSheet("""
            QWidget {
                background-color: #0b0f17;
                color: #e5e7eb;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }
            QTextEdit {
                background-color: #111827;
                border: 1px solid #1f2937;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                line-height: 1.5;
            }
            QLineEdit {
                background-color: #1f2937;
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 14px;
                color: #f3f4f6;
            }
            QPushButton {
                background-color: #0284c7;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #0369a1;
            }
        """)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # Header area: Hydro-Orb Visualizer + Title
        header_layout = QHBoxLayout()
        self.orb = HydroOrbWidget(self)
        header_layout.addWidget(self.orb)

        title_layout = QVBoxLayout()
        title_label = QLabel("<h2>L.O.R.A</h2><p style='color:#38bdf8; margin:0;'>Layer Oriented Research Assistant</p>", self)
        self.status_label = QLabel("<span style='color:#10b981;'>● Active & Grounded</span>", self)
        title_layout.addWidget(title_label)
        title_layout.addWidget(self.status_label)
        title_layout.addStretch()
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Chat / Research Feed
        self.chat_feed = QTextEdit(self)
        self.chat_feed.setReadOnly(True)
        layout.addWidget(self.chat_feed)

        # Input Area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Ask a research question or press Ctrl+Alt+Q to inspect clipboard...")
        self.input_field.returnPressed.connect(self._on_submit)
        input_layout.addWidget(self.input_field)

        self.submit_btn = QPushButton("Research", self)
        self.submit_btn.clicked.connect(self._on_submit)
        input_layout.addWidget(self.submit_btn)

        layout.addLayout(input_layout)

    def _init_shortcuts(self) -> None:
        # Global Quick-Inspect: Ctrl+Alt+Q
        self.inspect_shortcut = QShortcut(QKeySequence("Ctrl+Alt+Q"), self)
        self.inspect_shortcut.activated.connect(self.handle_clipboard_inspect)

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
        """Appends a message to the chat feed safely escaping HTML content."""
        clean_sender = html.escape(sender)
        clean_text = html.escape(text).replace("\n", "<br/>")
        formatted = f"<p><b><span style='color:#38bdf8;'>[{clean_sender}]</span></b><br/>{clean_text}</p>"
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
        self.orb.set_active(True)
        self.query_submitted.emit(query, "deep")
