import argparse
import logging
import sys
import threading
from pathlib import Path

import uvicorn

from lora.config.manager import ConfigManager, LoraUserConfig
from lora.gateway.server import create_gateway_app
from lora.ui.audio_cues import AudioCueManager, SoundCue
from lora.ui.hud import LoraDesktopHUD
from lora.ui.tray import LoraSystemTray
from lora.ui.widget import AmbientDropletWidget, WidgetState

logger = logging.getLogger("lora")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="L.O.R.A - Layer Oriented Research Assistant (Native Desktop Platform)"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Path to custom LORA config directory (default: ~/.lora)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Gateway HTTP/WebSocket port (default: 8765)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Gateway bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without desktop UI (Gateway daemon only)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable microphone capture and TTS voice pipeline",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force air-gapped offline research mode",
    )
    return parser


def init_platform(base_dir: Path | None = None) -> tuple[ConfigManager, LoraUserConfig]:
    """Initializes the ~/.lora directory hierarchy, locks NTFS ACLs, and loads configuration."""
    cfg_mgr = ConfigManager(base_dir=base_dir)
    cfg_mgr.ensure_ecosystem(apply_acls=True)
    user_cfg = cfg_mgr.load_config()
    return cfg_mgr, user_cfg


def start_gateway_thread(host: str, port: int) -> uvicorn.Server:
    """Launches the FastAPI Gateway inside a dedicated background thread."""
    app = create_gateway_app()
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    logger.info("Gateway daemon running at http://%s:%d", host, port)
    return server


def run_desktop_app(
    cfg_mgr: ConfigManager,
    user_cfg: LoraUserConfig,
    args: argparse.Namespace,
) -> int:
    """Launches the PySide6 Ambient Droplet Widget, HUD, System Tray, and Audio Cues."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # Initialize components
    audio = AudioCueManager(enabled=user_cfg.voice.sound_effects_enabled and not args.no_audio)
    droplet = AmbientDropletWidget(idle_opacity=user_cfg.ui.ghost_mode_opacity)
    hud = LoraDesktopHUD()
    tray = LoraSystemTray()

    # Initial position: bottom right corner of primary screen
    screen = app.primaryScreen().geometry()
    margin = user_cfg.ui.snap_margin_px
    init_x = screen.width() - droplet.width() - margin
    init_y = screen.height() - droplet.height() - margin - 40  # Above Windows taskbar
    droplet.move(init_x, init_y)

    # Connect interactions
    droplet.hud_requested.connect(hud.toggle_window)
    tray.open_hud_requested.connect(hud.toggle_window)

    def on_toggle_droplet() -> None:
        if droplet.isVisible():
            droplet.hide()
        else:
            droplet.show()

    tray.toggle_widget_requested.connect(on_toggle_droplet)

    def on_sleep_toggled(is_sleeping: bool) -> None:
        if is_sleeping:
            audio.play_cue(SoundCue.SLEEP)
            tray.update_state(WidgetState.SLEEPING)
            hud.status_label.setText("<span style='color:#6b7280;'>● Sleeping (0% CPU)</span>")
        else:
            audio.play_cue(SoundCue.WAKE)
            tray.update_state(WidgetState.IDLE)
            hud.status_label.setText("<span style='color:#10b981;'>● Active & Grounded</span>")

    droplet.sleep_toggled.connect(on_sleep_toggled)

    def on_tray_sleep_toggle() -> None:
        droplet.toggle_sleep()

    tray.sleep_toggled.connect(on_tray_sleep_toggle)

    def on_exit() -> None:
        audio.play_cue(SoundCue.SLEEP)
        droplet.close()
        hud.close()
        tray.hide()
        app.quit()

    tray.exit_requested.connect(on_exit)

    def on_query_submitted(query: str, depth: str) -> None:
        if droplet.state == WidgetState.SLEEPING:
            droplet.toggle_sleep()

        droplet.set_state(WidgetState.THINKING)
        hud.status_label.setText("<span style='color:#38bdf8;'>● Researching & Grounding...</span>")
        hud.orb.set_active(True)

        def _worker() -> None:
            try:
                import asyncio

                from lora.models.ollama_client import OllamaClient
                from lora.storage.memory import MemoryStore

                ollama = OllamaClient(base_url=user_cfg.models.ollama_url)
                is_online = asyncio.run(ollama.is_online())
                if is_online:
                    models_list = asyncio.run(ollama.list_models())
                    available_tags = [m.get("name", "") for m in models_list]
                    model_to_use = user_cfg.models.default_model
                    if model_to_use not in available_tags:
                        for candidate in (
                            "spark-x25-4b-8k:latest",
                            "spark-x25-4b-32k:latest",
                            "granite4.2:8b",
                            "gemma4:e4b",
                            "ornith-1.5:9b",
                        ):
                            if candidate in available_tags:
                                model_to_use = candidate
                                break
                        else:
                            if available_tags:
                                model_to_use = available_tags[0]

                    memory_store = MemoryStore(profile_path=cfg_mgr.paths.user_profile_file)
                    profile_ctx = memory_store.get_profile_markdown()

                    sys_prompt = (
                        "You are L.O.R.A (Layer Oriented Research Assistant), a sovereign, "
                        "grounded local AI peer running entirely on the user's laptop. "
                        "Provide concise, precise, evidence-based answers."
                    )
                    if profile_ctx:
                        sys_prompt += f"\nUser Context & Preferences:\n{profile_ctx[:600]}"

                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": query},
                    ]
                    response_text = asyncio.run(
                        ollama.generate_chat(model=model_to_use, messages=messages)
                    )
                else:
                    response_text = (
                        f"⚠️ Local Ollama service is offline at {user_cfg.models.ollama_url}. "
                        "Please make sure Ollama is running (`ollama serve`) to enable local model generation."
                    )
            except Exception as e:
                response_text = f"Error processing query: {e}"

            from PySide6.QtCore import QTimer

            def _post() -> None:
                hud.append_message("L.O.R.A", response_text)
                hud.orb.set_active(False)
                droplet.set_state(WidgetState.IDLE)
                hud.status_label.setText("<span style='color:#10b981;'>● Active & Grounded</span>")
                audio.play_cue(SoundCue.COMPLETE)

            QTimer.singleShot(0, _post)

        worker_thread = threading.Thread(target=_worker, daemon=True)
        worker_thread.start()

    hud.query_submitted.connect(on_query_submitted)

    # Start Gateway background server
    gateway_server = start_gateway_thread(host=args.host, port=args.port)

    # Show ambient UI
    droplet.show()
    tray.show()
    audio.play_cue(SoundCue.WAKE)
    logger.info("L.O.R.A Desktop Platform initialized successfully.")

    exit_code = app.exec()

    # Shutdown gateway server
    gateway_server.should_exit = True
    return exit_code


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg_mgr, user_cfg = init_platform(base_dir=args.config_dir)

    if args.headless:
        logger.info("Starting L.O.R.A in headless daemon mode...")
        server = start_gateway_thread(host=args.host, port=args.port)
        try:
            import time

            while not server.should_exit:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Shutting down L.O.R.A headless daemon...")
            server.should_exit = True
        return 0

    return run_desktop_app(cfg_mgr, user_cfg, args)


if __name__ == "__main__":
    sys.exit(main())
