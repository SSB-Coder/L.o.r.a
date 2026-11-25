# Needle tool calling and Mirage vision

Needle and Mirage are enabled in `config/lora.yaml`. Both run on CPU in separate,
lazily started workers, with two CPU threads each. They do not replace the writer,
Whisper, Kokoro or EmbeddingGemma. SimedWorld has no Cactus dependency or changes.

## Installed assets

The Python package is pinned to `cactus-compute==4.2.3`. Its native engine is
4.1.1 and its model revisions are pinned by the SDK. Only Needle and Mirage were
downloaded; Cactus Whistle was not downloaded.

The verified assets are about 59 MB total (56.2 MiB): native library 1,935,944
bytes, Needle 35,335,380 bytes, and Mirage 21,611,043 bytes. The local
`models/cactus/manifest.json` records versions, sizes, cache paths and SHA-256
checksums. Files under `models/cactus/` use the SDK's verified user cache, with
a local copy when links are unavailable; rerun the downloader if that cache is cleaned.

The machine had approximately 358 GB of free disk and 4.9 GB available RAM at
setup. In native smoke checks, Needle's worker peaked near 126 MiB and Mirage's
near 173 MiB (about 186 MiB on the terminal screenshot). These are worker-process
measurements, not the memory requirement of all Lora, Ollama and other apps.
The system was already using swap. No additional GPU allocation is needed for
these two models; the writer continues to use Ollama and the NVIDIA GPU.

## Setup and use

From the Lora directory:

```bash
uv sync --locked --python 3.12 --extra dev
.venv/bin/python scripts/download_cactus_models.py
./run_lora.sh
```

Both installers attempt to prepare Cactus assets when model provisioning is
enabled. Native availability depends on the SDK's platform builds. Restart an
already-running Lora to load the new configuration.

Try “Open Zed” or “Open Wikipedia”. Exact app aliases and website shortcuts
are resolved from `config/apps.yaml` before Needle proposes a schema-constrained
call; a completed literal opening request does not trigger another planning turn.

To test Mirage in the desktop chat:

1. Click **+ beside the message box** and select a PNG/JPEG, or paste an image.
2. Check that an attachment thumbnail appears. **Add sources** is for indexed
   documents and is separate from chat image attachments.
3. Send “Read the large text in this image” or “Describe this image”.
4. During processing, the status says “Analyzing image … with Mirage”; logs
   include `Cactus mirage`, and the turn trace includes Mirage Vision.

Start with large, clear text or a simple photograph rather than dense terminal
logs. The writer model name in the top bar can remain unchanged: Mirage supplies
observations and the selected writer composes the answer.

The image path feeds Mirage observations to the text writer;
raw images are not sent to a text-only model. CPU image interpretation is also
available in the existing runtime/gateway composition where images are allowed;
mobile upload restrictions remain enforced.

Needle proposes registered tools: `open_app`, `open_url`, `open_path`, `list_dir`,
`read_file`, `search_files`, `web_search`, `fetch_page` and `create_note`.
Clock, stop/sleep and existing dedicated artifact/research flows keep their
normal implementation. External MCP tools and unrestricted shell/desktop input
are not exposed to Needle.

Every proposal batch is schema-validated before execution and then goes through
Lora's existing origin, path, policy and confirmation checks. Low-confidence
free-form actions require confirmation. A literal app or website request can be independently
matched to the configured catalogue, but only the matching proposed call can use that
evidence, and it still passes through the policy gate. Repeated calls are not
executed twice. Untrusted tool/image observations cannot authorize further acting
tools. No proposals are executed through the SDK's automatic `run()` loop.

## Limits and failures

Mirage is a small perception model. Tiny terminal text and dense documents can
be misread, and image descriptions can contain incorrect observations. Paste
raw log text when exact spelling matters. Its generated reasoning trace is
excluded from the evidence sent to the writer. It does not add screen capture,
Wayland window control, or automatic clicking.

Native model calls have a bounded timeout. Cancellation discards late results
and retires the worker; a future request can start a fresh worker. Workers are
closed on application shutdown. Missing assets or model errors are reported
explicitly rather than silently passing proposed actions to another executor.
SDK telemetry is disabled in the workers, which load assets offline.

## Configuration and rollback

```yaml
cactus:
  needle_enabled: true
  mirage_enabled: true
  models_dir: models/cactus
  threads: 2
  timeout_seconds: 60
  min_confidence: 0.7
```

Set either enabled flag to `false` to restore that previous path. For a temporary
rollback of both capabilities:

```bash
LORA_DISABLE_CACTUS=1 ./run_lora.sh
```

## Verification

```bash
.venv/bin/python scripts/verify_cactus.py
.venv/bin/python scripts/verify_cactus_integration.py
.venv/bin/python -m pytest -q
```

The native checks exercise a real Needle proposal, negation and independent-turn
state, and OCR of generated text with Mirage. The integration check uses real
models and the real policy gate, but replaces app launch with an in-memory
recorder; it also sends Mirage's text observations through the actual writer.
It never approves uncertain/privileged proposals. Local reports are under
`logs/fedora-setup/cactus-*.json` and `logs/fedora-setup/cactus-*.log`.

The full suite passed 934 tests with five opt-in tests skipped. The focused
Cactus tests cover unknown tools, extra/invalid arguments, low confidence,
repeated calls, cancellation, untrusted observations, text-only image synthesis,
and the emergency rollback switch. Desktop and headless startup and clean
shutdown were exercised with the enabled configuration.
