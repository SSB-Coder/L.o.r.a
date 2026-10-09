import logging

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon, QWidget

from lora.ui.widget import WidgetState

logger = logging.getLogger(__name__)


class LoraSystemTray(QSystemTrayIcon):
    """Native Windows System Tray icon with status indicators and Dual-Tier power control."""

    open_hud_requested = Signal()
    toggle_widget_requested = Signal()
    sleep_toggled = Signal()
    exit_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = WidgetState.IDLE

        self.setIcon(self._create_tray_icon(QColor(14, 165, 233)))
        self.setToolTip("L.O.R.A - Active & Grounded")

        self._init_menu()
        self.activated.connect(self._on_tray_activated)

    def _init_menu(self) -> None:
        menu = QMenu()
        menu.setStyleSheet("""
            QMenu {
                background-color: #111827;
                color: #f3f4f6;
                border: 1px solid #1f2937;
                padding: 6px;
            }
            QMenu::item:selected {
                background-color: #0284c7;
            }
        """)

        self.action_hud = QAction("Open Desktop HUD", self)
        self.action_hud.triggered.connect(self.open_hud_requested.emit)
        menu.addAction(self.action_hud)

        self.action_widget = QAction("Toggle Droplet Widget", self)
        self.action_widget.triggered.connect(self.toggle_widget_requested.emit)
        menu.addAction(self.action_widget)

        self.action_sleep = QAction("Toggle Sleep / Wake", self)
        self.action_sleep.triggered.connect(self.sleep_toggled.emit)
        menu.addAction(self.action_sleep)

        menu.addSeparator()

        self.action_exit = QAction("Exit L.O.R.A", self)
        self.action_exit.triggered.connect(self.exit_requested.emit)
        menu.addAction(self.action_exit)

        self.setContextMenu(menu)

    def update_state(self, state: WidgetState) -> None:
        """Updates tray icon color and tooltip based on engine state."""
        self._state = state
        if state == WidgetState.SLEEPING:
            self.setIcon(self._create_tray_icon(QColor(107, 114, 128)))
            self.setToolTip("L.O.R.A - Sleeping (0% CPU)")
        elif state in (WidgetState.THINKING, WidgetState.SPEAKING):
            self.setIcon(self._create_tray_icon(QColor(6, 182, 212)))
            self.setToolTip("L.O.R.A - Processing...")
        elif state == WidgetState.ERROR:
            self.setIcon(self._create_tray_icon(QColor(239, 68, 68)))
            self.setToolTip("L.O.R.A - Alert / Attention Required")
        else:
            self.setIcon(self._create_tray_icon(QColor(14, 165, 233)))
            self.setToolTip("L.O.R.A - Active & Grounded")

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.open_hud_requested.emit()

    def _create_tray_icon(self, color: QColor) -> QIcon:
        """Generates a crisp 32x32 aqueous droplet pixmap for the system tray."""
        pixmap = QPixmap(32, 32)
        pixmap.fill(QColor(0, 0, 0, 0))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(color)
        painter.setPen(QColor(255, 255, 255, 180))
        painter.drawEllipse(4, 4, 24, 24)
        painter.end()

        return QIcon(pixmap)
