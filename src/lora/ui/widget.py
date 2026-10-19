import logging
from enum import StrEnum
try:
    from typing import override
except ImportError:  # pragma: no cover - Python <3.12 fallback
    def override(fn):  # type: ignore[misc]
        return fn

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QContextMenuEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QRadialGradient,
)
from PySide6.QtWidgets import QMenu, QWidget

logger = logging.getLogger(__name__)


class WidgetState(StrEnum):
    IDLE = "idle"
    THINKING = "thinking"
    SPEAKING = "speaking"
    SLEEPING = "sleeping"
    ERROR = "error"


class AmbientDropletWidget(QWidget):
    """Ambient floating screen-corner droplet micro-widget with Ghost Mode and magnetic edge-snapping."""

    hud_requested = Signal()
    sleep_toggled = Signal(bool)
    state_changed = Signal(str)
    emergency_stop_requested = Signal()

    def __init__(
        self,
        idle_opacity: float = 0.35,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._idle_opacity = idle_opacity
        self._state = WidgetState.IDLE
        self._audio_level = 0.0
        self._drag_pos = QPoint()
        self._is_dragging = False

        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.timeout.connect(self._on_single_click_timeout)

        # Configure frameless ambient floating window
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.resize(56, 56)
        self.setWindowOpacity(self._idle_opacity)

    @property
    def state(self) -> WidgetState:
        return self._state

    def set_state(self, state: WidgetState) -> None:
        """Updates visual state and adjusts opacity."""
        self._state = state
        if state == WidgetState.SLEEPING:
            self.setWindowOpacity(0.30)
        elif state in (WidgetState.THINKING, WidgetState.SPEAKING, WidgetState.ERROR):
            self.setWindowOpacity(0.95)
        else:
            self.setWindowOpacity(self._idle_opacity)

        self.update()
        self.state_changed.emit(self._state.value)

    def set_audio_level(self, level: float) -> None:
        """Sets mic audio level to trigger liquid ripples."""
        self._audio_level = max(0.0, min(1.0, level))
        self.update()

    def toggle_sleep(self) -> None:
        """Toggles between SLEEPING and IDLE state."""
        if self._state == WidgetState.SLEEPING:
            self.set_state(WidgetState.IDLE)
            self.sleep_toggled.emit(False)
        else:
            self.set_state(WidgetState.SLEEPING)
            self.sleep_toggled.emit(True)

    def calculate_snap_position(
        self,
        target_x: int,
        target_y: int,
        screen_width: int,
        screen_height: int,
        screen_left: int = 0,
        screen_top: int = 0,
        margin: int = 24,
        snap_threshold: int = 60,
    ) -> tuple[int, int]:
        """Calculates snapped screen coordinates respecting monitor origin and dimensions."""
        w = self.width()
        h = self.height()
        x, y = target_x, target_y

        # Snap Left or Right
        if target_x <= screen_left + margin + snap_threshold:
            x = screen_left + margin
        elif target_x + w >= screen_left + screen_width - margin - snap_threshold:
            x = screen_left + screen_width - margin - w

        # Snap Top or Bottom
        if target_y <= screen_top + margin + snap_threshold:
            y = screen_top + margin
        elif target_y + h >= screen_top + screen_height - margin - snap_threshold:
            y = screen_top + screen_height - margin - h

        return x, y

    @override
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        center_x = w / 2
        center_y = h / 2
        radius = min(w, h) / 2 - 4

        # Choose palette based on state
        if self._state == WidgetState.SLEEPING:
            base_col = QColor(75, 85, 99, 180)     # Dim slate
            glow_col = QColor(107, 114, 128, 120)
        elif self._state == WidgetState.THINKING:
            base_col = QColor(14, 165, 233, 230)   # Vivid cyan/sky
            glow_col = QColor(56, 189, 248, 180)
        elif self._state == WidgetState.SPEAKING:
            base_col = QColor(6, 182, 212, 240)    # Aqueous electric cyan
            glow_col = QColor(34, 211, 238, 200)
        elif self._state == WidgetState.ERROR:
            base_col = QColor(239, 68, 68, 230)    # Crimson warning
            glow_col = QColor(248, 113, 113, 180)
        else:  # IDLE
            base_col = QColor(2, 132, 199, 190)    # Deep ocean blue
            glow_col = QColor(56, 189, 248, 140)

        # Outer luminous glow
        glow_gradient = QRadialGradient(center_x, center_y, radius + 4)
        glow_gradient.setColorAt(0.0, glow_col)
        glow_gradient.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(glow_gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(int(center_x - radius - 3), int(center_y - radius - 3), int((radius + 3) * 2), int((radius + 3) * 2))

        # Core droplet body
        drop_gradient = QRadialGradient(center_x - radius * 0.3, center_y - radius * 0.3, radius * 1.4)
        drop_gradient.setColorAt(0.0, QColor(255, 255, 255, 220))  # specular gleam
        drop_gradient.setColorAt(0.3, base_col)
        drop_gradient.setColorAt(1.0, QColor(8, 15, 30, 240))      # deep base
        painter.setBrush(drop_gradient)
        painter.drawEllipse(int(center_x - radius), int(center_y - radius), int(radius * 2), int(radius * 2))

        # Glass highlight crescent
        highlight_path = QPainterPath()
        highlight_path.addEllipse(int(center_x - radius * 0.5), int(center_y - radius * 0.7), int(radius * 1.0), int(radius * 0.5))
        painter.setBrush(QColor(255, 255, 255, 70))
        painter.drawPath(highlight_path)

        # Concentric liquid ripples when audio is active
        if self._audio_level > 0.05:
            rip_rad = radius * (0.5 + 0.5 * self._audio_level)
            painter.setPen(QColor(56, 189, 248, int(180 * self._audio_level)))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(int(center_x - rip_rad), int(center_y - rip_rad), int(rip_rad * 2), int(rip_rad * 2))

    @override
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._is_dragging = False

    @override
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._is_dragging = True
            new_pos = event.globalPosition().toPoint() - self._drag_pos
            self.move(new_pos)

    def _on_single_click_timeout(self) -> None:
        self.toggle_sleep()

    @override
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if not self._is_dragging:
                # Delay single click to avoid triggering on double click
                self._click_timer.start(250)
            else:
                # Snap to edge on release
                screen = self.screen().geometry()
                cur_pos = self.pos()
                snapped_x, snapped_y = self.calculate_snap_position(
                    cur_pos.x(),
                    cur_pos.y(),
                    screen.width(),
                    screen.height(),
                    screen_left=screen.x(),
                    screen_top=screen.y(),
                )
                self.move(snapped_x, snapped_y)
            self._is_dragging = False

    @override
    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            # Cancel single-click timer so sleep is not toggled
            self._click_timer.stop()
            # Double-click opens full HUD
            self.hud_requested.emit()

    @override
    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #111827;
                color: #e5e7eb;
                border: 1px solid #1f2937;
                border-radius: 8px;
                padding: 6px;
            }
            QMenu::item:selected {
                background-color: #0284c7;
                border-radius: 4px;
            }
        """)
        open_hud_action = menu.addAction("Open HUD")
        toggle_sleep_action = menu.addAction("Toggle Sleep / Wake")
        stop_action = menu.addAction("Emergency Stop (Halt)")
        menu.addSeparator()
        exit_action = menu.addAction("Exit L.O.R.A")

        action = menu.exec(event.globalPos())
        if action == open_hud_action:
            self.hud_requested.emit()
        elif action == toggle_sleep_action:
            self.toggle_sleep()
        elif action == stop_action:
            self.emergency_stop_requested.emit()
        elif action == exit_action:
            if self.parent():
                self.parent().close()
            else:
                self.close()

    @override
    def enterEvent(self, event) -> None:
        # Hover reveals droplet clearly
        self.setWindowOpacity(0.95)

    @override
    def leaveEvent(self, event) -> None:
        # Revert to state opacity
        if self._state == WidgetState.SLEEPING:
            self.setWindowOpacity(0.30)
        elif self._state == WidgetState.IDLE:
            self.setWindowOpacity(self._idle_opacity)
