"""Prepare the optional browser and model assets for an installed Lora checkout."""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from lora.config.manager import ConfigManager
from lora.models.profiles import PREFERRED_MODEL
from lora.models.selection import candidate_names, choose_model, load_role

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def prepare_chat(base_url: str, model: str) -> None:
    base_url = base_url.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=10) as response:
            models = json.load(response)["models"]
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(
            "Ollama is not reachable. Install it from https://ollama.com/download, start it, "
            "then rerun Lora's installer. For a configured remote server, check ~/.lora/config.yaml."
        ) from exc

    installed = [name for entry in models if (name := entry.get("name") or entry.get("model"))]
    ready, _ = choose_model(candidate_names(model, load_role()), installed)
    if ready:
        logger.info("Ollama model already installed: %s", ready)
        return
    model = os.environ.get("LORA_MODEL") or model or PREFERRED_MODEL
    if urlsplit(base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError(f"Model {model} is not installed on the configured Ollama server. Pull it on that server, then retry.")

    logger.info("Downloading %s through Ollama (this can take a while).", model)
    request = urllib.request.Request(
        f"{base_url}/api/pull",
        data=json.dumps({"model": model, "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    status = None
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            for line in response:
                progress = json.loads(line)
                if error := progress.get("error"):
                    raise RuntimeError(f"Ollama could not pull {model}: {error}")
                if status != progress.get("status"):
                    status = progress.get("status")
                    logger.info("Ollama: %s", status)
        if status != "success":
            raise RuntimeError(f"Ollama did not finish pulling {model}. Rerun setup to retry the download.")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not download {model} through Ollama. Retry with `ollama pull {model}`.") from exc


def prepare_optional_models() -> None:
    commands = [
        ("Kokoro", ["scripts/download_models.py", "--kokoro-only"]),
        ("Needle and Mirage", ["scripts/download_cactus_models.py"]),
        ("Whisper", ["-c", 'from faster_whisper import WhisperModel; WhisperModel("base.en", device="cpu", compute_type="int8")']),
    ]
    if not (sys.platform == "darwin" and platform.machine() == "x86_64"):
        commands.insert(1, ("EmbeddingGemma", ["scripts/download_models.py", "--embedding-only"]))
    for label, command in commands:
        try:
            subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
        except (OSError, subprocess.CalledProcessError):
            logger.warning("%s setup did not finish. Text chat is still available; rerun setup to retry.", label)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-models", action="store_true", help="Do not contact Ollama or download model assets")
    parser.add_argument("--skip-browser", action="store_true", help="Do not install the Playwright Chromium browser")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.skip_models:
        config = ConfigManager().load_config().models
        try:
            prepare_chat(config.ollama_url, config.default_model)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
    if not args.skip_browser:
        try:
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], cwd=ROOT, check=True)
        except (OSError, subprocess.CalledProcessError):
            logger.error("Chromium installation failed. Rerun setup or use the --skip-browser option for desktop chat only.")
            return 1
    if not args.skip_models:
        prepare_optional_models()
    else:
        logger.info("Model setup skipped. Connect to an Ollama server with a model before starting a chat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
