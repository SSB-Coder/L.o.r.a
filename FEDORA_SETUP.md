# Lora on Fedora

Lora uses this project's `.venv` with Python 3.12. On Linux, PyTorch is the CPU
build plus CPU TorchVision for the EmbeddingGemma processor. Kokoro and Whisper
also run on CPU. Ollama owns GPU language-model inference. SimedWorld's environment and CUDA wheels are separate.

## Prerequisites and setup

Install `uv` in your user account, then install Fedora runtime libraries:

```bash
sudo dnf install gcc kernel-headers portaudio espeak-ng libxcb xcb-util-cursor xcb-util-image \
  xcb-util-keysyms xcb-util-renderutil xcb-util-wm libXcomposite libXdamage \
  libXfixes libXtst nss nspr atk at-spi2-atk cups-libs libdrm pango gtk3 \
  libXScrnSaver wmctrl xdotool gnome-text-editor gnome-calculator nodejs npm
```

Start Ollama before running `./scripts/install_linux.sh`. The installer uses the
committed uv lock, checks the configured Ollama model, and prepares Chromium
and optional local model assets. `--skip-models` skips all model provisioning;
`--skip-browser` skips Chromium. Some Hugging Face assets require account
access; check the setup output for optional features that were not provisioned.

Ollama 0.40.16 is available as `/usr/local/bin/ollama` and runs through the
system `ollama.service` on loopback port 11434. Models use the system service's
model store (normally `/usr/share/ollama/.ollama/models`).

```bash
systemctl status ollama.service
ollama list
# Install Lora and prepare the default chat model:
./scripts/install_linux.sh
# Optional controller model:
ollama pull granite4.2:3b
./run_lora.sh
./run_lora.sh start --no-audio
./run_lora.sh start --headless
./run_lora.sh stop
```

The default writer/reader/planner tag is
`hf.co/XHToken/Spark-X2.6-4B-GGUF:Q4_K_M`. Optional specialist models in the
configuration are not all required for basic conversation/research. Local assets
live in `models/`; Ollama's model store is separate.

## Desktop behavior

Fedora's GNOME Wayland session supports the Qt interface, chat, file opening,
application launching, voice, browser tools and gateway. Wayland does not expose
general window enumeration/focus or foreground-checked typing through the
existing APIs. Those actions stay unavailable; use the target application
manually. Global pynput shortcuts are disabled on Wayland; use the HUD controls
and addressed voice wake. The small floating droplet is hidden on Wayland because
the compositor can place it over the HUD instead of in the screen corner. The
large animated droplet in the HUD is unaffected.

An X11 session supports `wmctrl`/`xdotool` window tools
and the existing pynput shortcuts. No compositor security setting is relaxed.

`config/apps.yaml` contains Linux executable mappings alongside the Windows
commands. Optional applications such as Chrome, Spotify, Obsidian, Pinta,
Xournal++ and OpenBoard need their own installation if you want to launch them.
Desktop tray placement depends on GNOME's tray support; use the visible HUD.

## Verification

```bash
npm ci --no-audit --no-fund
uv run --locked --extra dev pytest
uv pip check --python .venv/bin/python
./run_lora.sh --help
```

Model inference, microphone quality and optional applications depend on available
assets/devices. Do not interpret the historical Windows results in README.md as
Linux verification.

## Verified on this machine (2026-11-17)

Python 3.12.16, CPU PyTorch 2.15.0+cpu, NumPy 2.5.3.
The full suite passed: 899 passed, 5 skipped (opt-in evaluation/integration
checks). The installed packages passed uv's dependency consistency check.
Kokoro synthesized speech without playback, Whisper ran cached CPU INT8
inference on silence, and EmbeddingGemma produced real 768-dimensional vectors
without the fallback embedder. PyAV is constrained to the compatible 16.x API
for faster-whisper file decoding. Qt/WebEngine, audio-device enumeration,
Chromium rendering, headless gateway health, and desktop startup/clean shutdown were exercised successfully.
GNOME's missing legacy tray service logs a warning; the visible HUD works.

Repeat runtime checks with `.venv/bin/python scripts/verify_linux.py`. Add
`--models` to synthesize speech without playing it, load cached Whisper, compute
real embedding vectors, and request a local Ollama completion. Results are
written to `logs/fedora-verification.json`.

The complete model verification also passed: CPU speech synthesis and file
transcription, real EmbeddingGemma inference, a writer completion at 8,192 tokens
of context, and a real Lora chat turn. The writer was fully placed on the GPU.
See `logs/fedora-verification.json` and `logs/fedora-setup/` for local results.

GPU memory is shared with other applications. Before a large SimedWorld CUDA
run, stop Lora and release its resident writer with:

```bash
./run_lora.sh stop
ollama stop hf.co/XHToken/Spark-X2.6-4B-GGUF:Q4_K_M
```

This releases model VRAM without stopping the Ollama service or deleting weights.

The configured controller, `granite4.2:3b`, is also installed and passed a local
completion check at 6,144 tokens of context. Its result is recorded in
`logs/fedora-setup/controller-verification.json`. Optional specialist model
alternatives remain optional; the default writer and controller are ready.

## App opening and document scope

“Open Zed”, “launch Zed editor” and “open Zed IDE” use the Zed allowlist entry.
App-specific executable alternatives are configured in `config/apps.yaml` under
`commands` (platform to list of executable names). Approved `desktop_ids` provide
a Linux/Flatpak fallback through `gio launch`; arbitrary requested commands are
never executed. Windows installation paths are configuration too. Restart Lora
after changing its app configuration.

Added sources are consulted when you refer to them, for example “according to my
sources, explain the training method”. A new question such as “Do deep research
on redshift” keeps its own topic and uses web research.

Local retrieval candidates now pass a relevance check with the resident writer
before answer generation or speech. An invalid or off-topic result declines the
passages; this is a relevance guard, not a guarantee of factual correctness.
Personal questions such as “my model” and named indexed files stay local. New
standalone research topics do not inherit prior document discussion.

After the Zed/source-scope changes, the full suite passed: 916 passed, 5 skipped.
The actual installed writer was also exercised with synthetic retrieval fixtures:
it answered the new astronomy question and declined an unrelated neural-network
passage. The record is `logs/fedora-setup/source-topic-writer-check.json`.

## Cactus vision and tool proposals

Needle and Mirage are now enabled for CPU tool proposals and image perception.
See [CACTUS_SETUP.md](CACTUS_SETUP.md) for installation, verification, limits and
rollback. Whisper, Kokoro and the Ollama writer remain in place. Cactus Whistle
was not installed.
