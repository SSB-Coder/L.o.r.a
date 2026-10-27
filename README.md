# L.O.R.A — Layer Oriented Research Assistant

L.O.R.A is a local-first Windows companion that combines conversation, voice,
document retrieval, web research, and bounded desktop actions. It is designed
to behave like a casual friend and dependable butler: answer naturally, notice
the local time, handle practical requests, and keep the user in control.

The reference hardware is an **RTX 3050 with 6 GB VRAM and 16 GB system RAM**.
Ollama runs the language models; Kokoro speech synthesis, Whisper speech
recognition, and embeddings run on CPU. Everyday replies use a resident writer,
while specialized models are available for tasks that need them.

Windows is the primary desktop target. The project also includes a paired
mobile web companion, a headless gateway, automated tests, and experimental
research components. This README describes the current runtime, rather than
treating every retained experiment as an active product feature.

## Contents

- [Features](#features)
- [Requirements and hardware profile](#requirements-and-hardware-profile)
- [Installation and models](#installation-and-models)
- [Running LORA](#running-lora)
- [Voice and interruption](#voice-and-interruption)
- [Documents and research](#documents-and-research)
- [Mobile companion](#mobile-companion)
- [Configuration and local data](#configuration-and-local-data)
- [Gateway API](#gateway-api)
- [Architecture](#architecture)
- [Development and validation](#development-and-validation)
- [Troubleshooting](#troubleshooting)
- [Current limits and next steps](#current-limits-and-next-steps)
- [Third-party components](#third-party-components)

## Features

| Area | Current behaviour |
|---|---|
| Conversation | Casual conversation, short-term history, local clock/date awareness, and deeper explanations when requested. |
| Desktop UI | Chat HUD, animated droplet, system tray, model/skill selectors, sources, and voice-state feedback. |
| Hands-free voice | Addressed phrases such as “Hey LORA” and “Hey Laura,” including a wake phrase followed immediately by a question. |
| Speech | CPU Kokoro-ONNX, `af_sky` by default; sentence streaming, interruption, and bounded spoken previews of long replies. |
| Documents | PDF, DOCX, Markdown, text, CSV, JSON, and supported code files; extraction, chunking, retrieval, and summaries. |
| Web research | Search, page reading, CPU page embeddings where available, numbered sources, and cited synthesis. |
| Source checks | Lightweight checks for missing citations and unknown source IDs, without an automatic truth percentage. |
| Desktop actions | Allowlisted app/site/folder opening, supported window actions, confirmed typing, notes, files, exports, and generated visuals. |
| Notebooks | Save a research session and rewrite cached evidence at another explanation depth. |
| Mobile | Paired browser chat, quick research, answer styles, citations, visible notebooks, cancellation, and optional browser readout. |
| Recovery | Isolated speech workers, bounded queues, cancellation-aware turns, restart after unexpected speech-worker failure, and ordered shutdown. |

Mobile cannot launch PC applications, type into desktop windows, upload files,
or query the desktop document index. Its browser readout uses the browser/phone
speech facility, separately from desktop Kokoro.

## Requirements and hardware profile

Use **Python 3.12** for the documented setup. Package metadata permits Python
3.11+, but recorded development and regression runs used 3.12.

You need Windows, `uv`, Git, a running Ollama installation, and compatible local
models. Voice requires microphone permission and working audio devices. Allow
space for Python dependencies, weights, and caches; model weights are not
included in the repository. Installation, uncached model downloads, and web
research need Internet access.

Playwright Chromium is needed for browser automation. Node.js and jsdom are
optional development dependencies for additional JavaScript checks.

| Component | Reference configuration |
|---|---|
| Writer | Spark-X2.6 4B Q4_K_M; tag `hf.co/XHToken/Spark-X2.6-4B-GGUF:Q4_K_M`. |
| Writer context | 8,192 tokens, enforced by model-selection code. |
| Reader and planner | Same Spark-X2.6 Q4_K_M tag and 8,192-token budget as the writer, avoiding context-profile changes between these roles. |
| Control | Granite 3B where installed, with a 6,144-token profile. |
| Specialized models | Configured complex/vision alternatives; installing every alternative is unnecessary. |
| STT | Faster-Whisper `base.en`, CPU INT8, four CPU threads. |
| TTS | Kokoro-ONNX on CPU, default voice `af_sky`. No Qwen TTS is required. |
| Embeddings | Compatible local EmbeddingGemma weights, CPU, batch size eight, bounded caching. |
| Vector search | NumPy exact inner-product search; FAISS is optional. |
| Rendering | Bounded desktop WebGL; independent mobile 2D pearl with a 30 fps cap and reduced-motion support. |

Set `OLLAMA_MAX_LOADED_MODELS=1` in the environment of the **Ollama server**.
Restart an already-running Ollama application after changing its environment.
Setting it only in a LORA client process does not reconfigure the server.
See [Ollama environment configuration](https://docs.ollama.com/faq).

The reference profile is a resource budget, not a guarantee that every model
fits entirely in GPU memory. Inspect `ollama ps` for actual CPU/GPU placement.
Context allocation is centralized in
[selection.py](src/lora/models/selection.py); changing `num_ctx` in YAML alone
may not override a fixed profile. Larger context and longer-output tuning remain
separate work.

## Installation and models

Run PowerShell commands from the repository root.

### 1. Install the Python environment

```powershell
uv sync --locked --python 3.12
```

For tests and linting, include the development dependencies:

```powershell
uv sync --locked --python 3.12 --extra dev
```

For browser automation:

```powershell
uv run playwright install chromium
```

The alternative [Windows installer](scripts/install.ps1) can create shortcuts
and initialize user configuration. Its default model setup includes optional
large models; use `-SkipModels` for a selective setup.
[run_lora.bat](run_lora.bat) starts the installed environment from the project root.

### 2. Prepare the language model

Start Ollama and inspect installed tags:

```powershell
ollama list
```

The tags in [config/lora.yaml](config/lora.yaml) describe the development
machine's choices. Writer, reader, and planner use the quantized Spark-X2.6 tag
`hf.co/XHToken/Spark-X2.6-4B-GGUF:Q4_K_M`. Install that tag with Ollama, or
import a compatible quantized GGUF and match its tag in configuration.
Spark-X2.5 is no longer required. Keep thinking disabled for the default
interactive profile; context expansion remains separate work.

For example, after obtaining a compatible, already-quantized Spark GGUF and
placing it at `models/writer.gguf`, a local import can use:

```powershell
Set-Content -Path .\models\Modelfile.spark -Value 'FROM ./writer.gguf' -Encoding ascii
ollama create spark-x26-4b -f .\models\Modelfile.spark
```

Use an Ollama build that supports the model architecture and a GGUF with correct
tokenizer/chat-template metadata. Importing a GGUF does not itself quantize its
weights. See [Ollama model importing](https://docs.ollama.com/import).

LORA's asset directory and Ollama's model store are separate. `models_dir`
does not relocate Ollama's blobs; `OLLAMA_MODELS` controls the Ollama store.

### 3. Prepare CPU speech and embeddings

```powershell
uv run python scripts/download_models.py --kokoro-only
uv run python scripts/download_models.py --embedding-only
```

| Asset | Default location |
|---|---|
| Kokoro model | `models/kokoro/kokoro-v1.0.onnx` |
| Kokoro voices | `models/kokoro/voices-v1.0.bin` |
| Embedding model | `models/embeddinggemma2/`; existing `models/embeddinggemma-2/` is also discovered. |
| Optional neural router | `models/laya-lora-ft/` |
| Optional reranker | `models/bge-reranker-base/` |
| Optional wake weights | Configured under `models/voice/`; unnecessary for the default transcript-based wake path. |

The [downloader](scripts/download_models.py) uses the repository root, reuses
recognized assets, and provides `--laya-only` and `--reranker-only` options.
Without a selector it attempts all asset groups. Hugging Face downloads may
require access approval or authentication. Read the setup summary: optional
assets can be unavailable even when the script returns success.

Whisper first tries cached models and may download its configured model when
needed. Cache the required assets before expecting offline voice. Default fast
routing does not need a neural Laya checkpoint. Core DOCX extraction uses ZIP/XML
parsing and does not require `python-docx`.

[install_tts.ps1](scripts/install_tts.ps1) provides dedicated Kokoro setup and a
synthesis check. Additional quality tools are listed separately in
[requirements-conformance.txt](requirements-conformance.txt).

## Running LORA

```powershell
uv run lora start
```

Useful variants:

```powershell
uv run lora start --no-audio
uv run lora start --headless
uv run lora start --config-dir C:\LORA-user
uv run lora net-check
uv run lora --help
```

- `--no-audio` disables microphone capture and desktop speech.
- `--headless` runs the gateway/job worker without the desktop UI or voice pipeline.
- `--config-dir` selects the user ecosystem directory, not the repository model-role YAML.
- `--offline` sets application offline flags; it is not an operating-system network sandbox. Prepare assets beforehand and account for external integrations.

For clean shutdown, use the same user directory as the running instance:

```powershell
uv run lora stop
uv run lora stop --config-dir C:\LORA-user
```

`lora kill` is an alias for the same clean shutdown request. Task interruption
through `Ctrl+Esc` keeps the app running.

Examples to type or speak:

| Goal | Example |
|---|---|
| Conversation | “How are you?” or “I've had a long day.” |
| Clock | “What time is it?” |
| App or folder | “Open VS Code.” or “Open my Downloads folder.” |
| Site search | “Open Wikipedia on Alan Turing.” |
| Web research | “Search the web for redshift.” |
| Expanded research | “Conduct deep research on redshift.” |
| Document overview | “Summarize my sources.” |
| Cached rewrite | “Explain that simpler.” or “Explain like I'm 5.” |
| Save research | “Save this to my notebook.” |
| Export | “Export this report as Markdown.” |
| Sleep | “Go to sleep.” |

Desktop actions depend on installed apps, the allowlist, and policy. File creation
and other content-producing actions may require confirmation. Supported editor
typing shows the exact text and requires confirmation plus a foreground-window
check. Selecting a larger model can replace the resident writer and increase
latency or memory use.

## Voice and interruption

The microphone normally remains open while LORA runs. VAD segments speech;
local transcription recognizes addressed phrases such as “Hey LORA” and
“Hey Laura.” A phrase alone arms a follow-up utterance; a phrase plus a question
can submit both together.

While desktop execution or spoken output is active, captured utterances can be
buffered for later transcription. Queues and utterance duration are bounded;
continuous capture does not create an unlimited recording backlog.

| Control | Effect |
|---|---|
| `Ctrl+Alt+Space` | Global push-to-talk while held. |
| Hold `Space` in the HUD | Push-to-talk outside a text input. |
| `Ctrl+Esc` | Interrupt the task and speech without exiting. |
| `Ctrl+Alt+Shift+Esc` | Existing interrupt alias. |
| `Ctrl+Alt+S` | Toggle sleep. |
| Droplet/HUD controls | Wake/sleep, voice, and interruption controls. |

Sleep changes the visual state and stops current speech. With the default
`voice.close_mic_on_sleep: false`, addressed wake listening remains available.
Sleep does not disable the microphone; use `--no-audio` or microphone permissions
when capture must be off.

For speaker enrollment:

```powershell
uv run lora enroll --speaker owner --samples 5
```

Speaker matching is a convenience filter, not secure authentication or perfect
voice separation. Without a valid profile it permits speech rather than proving
identity. VAD and filtering reduce some background noise, but fan noise,
competing voices, device gain, and playback echo still affect recognition.

Speech runs in recoverable child processes by default. Exit stops queued speech
before closing the worker and waits a bounded time for speech threads. An
unexpected worker crash can be retried on a later request. Kokoro remains the
active desktop TTS engine.

## Documents and research

### Added sources and summaries

Add documents through the desktop **Sources** controls. PDF and DOCX extraction
use bounded parser subprocesses. DOCX paragraphs, headings, and table cells are
read in document order. Markdown sections and long text form bounded chunks.
PDF page references are preserved; DOCX/Markdown do not gain invented pages.

Ordinary questions consult added desktop sources first. A missing local passage
does not silently trigger online search. Explicit web, latest, or news requests
can use web research. Web research does not automatically append unrelated
local-file matches, including files added while the search is running.

“Summarize my sources” asks the writer to explain actual ideas and conclusions,
rather than displaying index statistics or two sample excerpts. Large
collections use ordered batches and combined section notes. The current limit
is 24 batches of approximately 10,000 characters per turn; partial coverage is
reported when the limit is reached.

Image entries in the media index provide metadata, not automatic OCR or visual
understanding. Chat image attachments require an installed vision-capable model.
Unreadable or unsupported source formats report an error instead of indexing
binary content as prose.

### Embeddings and vector search

EmbeddingGemma converts text into vectors; the index retrieves similar passages;
the writer answers using their text. FAISS is an indexing library and needs no
special FAISS language model.

NumPy is the default backend, avoiding the optional native FAISS runtime.
Desktop and phone web research share CPU embedding weights, serialized inference,
and a bounded cache. Page indexing reports chunk progress. The web path reports
lexical retrieval when neural retrieval is unavailable; fallback embeddings do
not provide equivalent semantic quality.

Cold loading and indexing can be slow on the reference CPU. Chunk counts depend
on content and estimated size. Re-index originals if an older index contains
incorrect extraction. Legacy rebuilding needs usable originals or saved source
text, depending on the stored ingestion version.

### Citation checks and depth

Normal research retrieves sources and writes a cited answer with the resident
writer. Automatic checks identify missing citations and unknown source IDs.
**A valid citation ID does not prove that the source supports the claim.**
No truthfulness percentage is assigned to this structural check.

Normal `quick`, `standard`, `deep`, and compatibility `deep_with_refinement`
paths do not run NLI, an independent reviewer, iterative repairs, or prompt
mutation loops. Older implementations remain for standalone/experimental use.

ELI5, Simple, Standard, Detailed, and Expert describe explanation style, not
independent verification. Cached rewrites can reuse earlier evidence without
re-browsing. Sources can be incomplete, stale, or contradictory.

## Mobile companion

Desktop startup attempts a separate mobile listener on the detected LAN address,
normally port **8766**. Open the pairing control in the HUD and scan the QR code
with a phone on the same network.

To choose the address explicitly:

```powershell
uv run lora start --mobile-bind 192.168.1.20:8766
```

Replace the example IP with an address assigned to the PC. The desktop gateway
normally uses `127.0.0.1:8765`. Startup code determines the active mobile address;
the repository YAML's example bind is not automatically your network address.

Pairing uses a separate mobile token in the URL fragment. The browser stores
it locally and authenticates API and socket connections. The gateway uses plain
HTTP and is intended for a trusted LAN, not direct public Internet exposure.
Keep pairing links, QR codes, and tokens private. Keep the desktop listener on
loopback for ordinary use.

Mobile supports chat, quick research, answer styles, source excerpts, allowed
notebooks, cancellation, and browser readout. Its independent pearl uses outward
thinking ripples and moving speaking waves, with dark mode and reduced-motion
support. Speaking motion is a state animation, not a microphone waveform.

Mobile STT, streaming desktop Kokoro to the phone, uploads, and phone-driven PC
control are not implemented. Desktop-only notebooks are filtered. A rejected
“open VS Code on my PC” request is the current capability boundary.

For firewall setup, allow the actual mobile port only on the intended private
adapter/subnet. [net_setup.ps1](scripts/net_setup.ps1) is an administrator-run
helper for an intentionally configured isolated LAN. Its defaults are port
8765 and subnet `192.168.50.0/24`; supply values matching your actual setup.

## Configuration and local data

| Location | Purpose |
|---|---|
| [config/lora.yaml](config/lora.yaml) | Model roles, voice settings, grounding defaults, and resource budgets. |
| [config/apps.yaml](config/apps.yaml) | Allowed apps, aliases, and web shortcuts. |
| [config/policy.yaml](config/policy.yaml) | Effective action restrictions, roots, and origin rules. |
| [config/sources.yaml](config/sources.yaml) | Settings for connector components that consume them. |
| `~/.lora/config.yaml` | Per-user configuration, including Ollama URL and user defaults. |
| `~/.lora/agents/persona.md` | Optional personal tone/persona override. |
| `~/.lora/skills/` | User-defined skills. |
| `~/.lora/mcp.json` | MCP declarations, disabled by default. |
| `~/.lora/memory/user_profile.md` | Local profile context used by desktop. |
| `~/.lora/voice/user_verifier.json` | Speaker enrollment; existing legacy profiles can also be discovered. |
| `~/.lora/notebooks/notebooks.db` | Saved notebooks. |
| `~/LORA/outputs/` | Default notes and exports. |
| `data/media_db/` | Repository-local media index when persisted. |
| `cache/browser_profile/` | Browser session data in a source checkout. |
| `logs/lora.log` | Rotating runtime log. |
| `logs/native-fault-<pid>.log` | Native worker fault reports. |

`models_dir: models` resolves from the project root. Launching from another
directory does not relocate the weights. Some historical model/backend settings
remain for experiments; their YAML presence does not make them active.

| Environment variable | Purpose |
|---|---|
| `LORA_MODEL` | Override the Ollama model name. |
| `LORA_EMBEDDING_MODEL_PATH` | Select a local embedding-model directory. |
| `LORA_VECTOR_BACKEND` | `numpy` by default; `faiss` after native-runtime validation. |
| `LORA_ENABLE_CPU_PAGE_RAG` | Set to `0` for lexical page retrieval instead of neural page indexing. |
| `LORA_NEURAL_ROUTING` | Set to `1` to opt into local neural routing; default routing is rule based. |
| `LORA_ENABLE_MCP` | Set to `1` for configured desktop MCP startup. |
| `LORA_DESKTOP_TOKEN` | Explicit desktop gateway token. |
| `LORA_MOBILE_TOKEN` | Explicit mobile token. |
| `LORA_OUTPUTS_DIR` | Custom output location; policy must also allow the target. |
| `OLLAMA_MAX_LOADED_MODELS` | Ollama-server setting; reference value `1`. |
| `OLLAMA_MODELS` | Ollama model-store location, independent of LORA assets. |

The supplied filesystem MCP server is disabled. Review executables, arguments,
and permitted roots before enabling a server; an example root is not permission
to expose a whole user directory.

The supplied policy allows Documents, Downloads, Desktop, and `~/LORA/outputs`.
Destructive tools, arbitrary shell execution, and dangerous file-opening
extensions are blocked. Ingesting CSV/DOCX is distinct from opening those
formats in external applications. Parser processes have resource limits and
disabled network access, but are not a complete OS sandbox. Web content is
untrusted and inspected for embedded instructions; detection is not universal.

Local weights, logs, caches, data, private notes, and development reports are
excluded by ignore rules. Credentials do not belong in tracked configuration.
Ignoring a directory does not remove files that were already tracked.

## Gateway API

Set an explicit `LORA_DESKTOP_TOKEN` before starting headless mode when another
client needs a known token. The caller must supply that same token in its own
environment.

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Public health response. |
| `POST /v1/ask` | Queue a request; HTTP 202 with job ID. |
| `GET /v1/jobs/{job_id}` | Job status and completed result. |
| `POST /v1/stop` | Cancel a queued/running job. |
| `GET /v1/notebooks` | Notebooks visible to the origin. |
| `POST /v1/notebooks/{notebook_id}/rewrite` | Queue a rewrite. |
| `WS /v1/events` | Authenticated state, queue, partial-text, answer, and error events. |

Example client request with a valid desktop token already set:

```powershell
$headers = @{ Authorization = "Bearer $env:LORA_DESKTOP_TOKEN" }
$body = @{ text = 'Search the web for redshift'; run_mode = 'quick'; depth = 'standard' } | ConvertTo-Json
$job = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8765/v1/ask' -Headers $headers -ContentType 'application/json' -Body $body
Invoke-RestMethod -Uri "http://127.0.0.1:8765/v1/jobs/$($job.job_id)" -Headers $headers
```

Poll until a terminal status. HTTP 202 acknowledges a queued request, not a final
answer. Socket clients send an initial `{"type":"auth","token":"..."}` message.
Mobile accepts its own token and reduced capability set; deep execution modes
are rejected independently of explanation style.

Legacy `/api/hud/chat` also queues work. HTTP media routes return 410; source
selection belongs to native controls. The public `/app/` shell contains no user
data. API data access and listener Host/Origin rules are checked separately.

## Architecture

| Package | Responsibility |
|---|---|
| [main.py](src/lora/main.py) | Startup, UI dispatch, voice wiring, gateway lifecycle, shutdown. |
| [agent/](src/lora/agent/) | Turn ownership, routing, chat, research, summaries, tool execution, job workers. |
| [models/](src/lora/models/) | Ollama client, model selection, context profiles, routing backends. |
| [voice/](src/lora/voice/) | Capture, VAD, STT, wake/speaker matching, speech streaming, playback, isolation. |
| [grounding/](src/lora/grounding/) | Chunks, embeddings, indexes, retrieval, evidence formatting. |
| [tools/](src/lora/tools/) | Connectors, browser adapters, files, app/window control, MCP. |
| [gateway/](src/lora/gateway/) | Authentication, origin boundaries, queues, HTTP/WebSocket, mobile assets. |
| [ui/](src/lora/ui/) | HUD, droplet, tray, thread-safe UI delivery. |
| [storage/](src/lora/storage/) | Notebooks, memory helpers, exports, migrations. |
| [safety/](src/lora/safety/) | Policy, provenance, egress, parsing isolation, cancellation, shutdown IPC. |
| [verify/](src/lora/verify/), [supervisor/](src/lora/supervisor/), [outer_loop/](src/lora/outer_loop/) | Retained verification, supervision, and optimization components outside default research. |

Runtime routing and model calls stay in `agent/runtime.py`; answer/research
workflows live in `agent/answers.py`, commands and artifacts in `agent/actions.py`,
and shared turn data in `agent/contracts.py`. Existing runtime imports remain
compatible. Desktop microphone setup and speech callbacks live in
`voice/desktop_input.py` and `voice/desktop_output.py`; the desktop still owns
their handles, GUI dispatch, cancellation, and shutdown. Python model defaults
are shared through the lightweight `models/profiles.py` module.

A desktop request is classified, resolved against commands or sources, and
answered through the selected model or permitted tool. Research planning and
writing reuse the writer. Qt receives UI updates through its dispatcher; speech
uses separate queues and recoverable CPU processes. Phone requests enter the
authenticated queue with desktop priority and fewer capabilities.

The workflow is coordinated and resource bounded; it does not require multiple
language models to run concurrently.

## Development and validation

Install development dependencies, then run:

```powershell
uv run python -m pytest tests -q
uv run python -m ruff check src tests/unit/test_plan_conformance.py
```

For focused product checks:

```powershell
uv run python -m pytest tests/unit/test_simple_research.py tests/unit/test_document_restoration.py tests/unit/test_speech_stream.py tests/unit/test_prime_workflow.py tests/integration/test_desktop_voice_flow.py -q
```

[tests/run_sandbox_tests.py](tests/run_sandbox_tests.py) is an opt-in adapter for
restricted Windows test hosts that cannot create normal asyncio wakeup sockets.
It is not production startup and is unnecessary on a normal PC.

Live model evaluations are explicit:

```powershell
uv run python -m pytest tests/evals --run-model-evals -q
```

They need the intended local assets. Some evaluate the retained experimental
verifier rather than normal research. Additional HUD JavaScript tests need Node
and jsdom. Optional type/security tools are in `requirements-conformance.txt`.

The tracked [CI workflow](.github/workflows/ci.yml) defines Windows/Linux tests,
coverage artifacts, and lint/type/security jobs. This is not a claim that every
remote job currently passes.

Recent local validation recorded **854 passing tests and 13 skips** in a full
run before the final shutdown repair, then **57 passing targeted checks** for
speech/shutdown work. These separate runs overlap; their counts are not additive.
Controlled integration uses real Qt/controller wiring with scripted model/audio
backends, rather than certifying live microphone, GPU, or phone-network behaviour.

| Development helper | Purpose and prerequisites |
|---|---|
| [benchmark_audio.py](scripts/benchmark_audio.py) | Audio-buffer and synthetic-noise smoke measurements, not speaker-recognition accuracy. |
| [benchmark_retrieval.py](scripts/benchmark_retrieval.py) | Generated MD/DOCX retrieval checks against installed local weights. |
| [check_mobile_ui.py](scripts/check_mobile_ui.py) | Actual mobile assets in Chrome with scripted gateway responses. |
| [bench_models.py](scripts/bench_models.py) | Experimental model/swap/embedding measurements; inspect local paths and extra imports. |
| [review_inventory.py](scripts/review_inventory.py) | Developer source-inventory helper. |

Helpers have local output-directory assumptions and some machine-specific
defaults. Inspect them before running in a fresh checkout. Generated reports,
screenshots, and benchmark artifacts are not prerequisites for using the app.

## Troubleshooting

| Symptom | What to check |
|---|---|
| Engine unavailable | Start Ollama, run `ollama list`, verify its URL and an installed compatible writer tag. |
| Slow first reply | Wait for warm-up; inspect `ollama ps` for cold loading, CPU offloading, or a model switch. |
| Research waits before writing | Inspect page-reading/embedding progress; cold CPU indexing can dominate elapsed time. |
| No Kokoro voice | Run Kokoro-only setup and synthesis check; confirm both model/voices files and inspect logs. |
| Wake phrase missed | Check input device, levels, VAD/STT logs, addressed wording, and speaker profile; try push-to-talk. |
| Fan/other voices interfere | Adjust placement/gain, reduce playback echo, and check enrollment. Filtering is not perfect voice isolation. |
| Document answer misses content | Confirm readable indexing, ask a specific question, or re-index old extraction. |
| Only a few passages displayed | Samples/retrieved passages are not the whole index; use “Summarize my sources” for synthesis. |
| Phone cannot connect | Check actual LAN address/port, same-network access, firewall, and current token. Phone loopback refers to the phone. |
| Phone cannot open a PC app | Desktop control is unavailable to the mobile origin. |
| Stale UI after an update | Restart LORA and refresh the paired phone page. |
| Native speech failure | Preserve runtime/native-fault logs; a worker can restart on a later request. |
| Unexpected exit | Preserve the final logs, fault report, model tag, and reproduction steps; distinguish an exit request from a crash. |

Logs scrub recognized secrets and paths but can still contain prompts, source
titles, and personal text. Review them before sharing an issue.

## Current limits and next steps

- Windows is the primary desktop target; component tests on another OS do not
  establish equivalent desktop-control or global-hotkey behaviour.
- The writer retains its 8K profile. Answer length also depends on generation
  limits and prompts; context alone does not guarantee longer or better output.
- Semantic retrieval needs usable local embedding weights, and cold CPU
  indexing remains a significant performance cost.
- Rule-based routing is the default; uncalibrated routing values are not measured
  confidence probabilities.
- Image indexing is metadata-oriented, not general OCR or autonomous computer use.
- Browser adapters depend on external sites, login state, and changing layouts.
  Explicit delegation to another AI sends the requested content to that service.
- Experimental verification, supervision, and prompt-evolution components remain
  outside the standard conversation/research workflow.
- Mobile voice capture, desktop Kokoro streaming, uploads, and PC control are future work.

Priorities are smaller lifecycle/runtime components, isolation or removal of
obsolete paths, measured retrieval improvements, and repeatable long-session
hardware tests. Context/output tuning should be measured against memory use and
responsiveness.

## Third-party components

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency notices and
architectural attributions. Model weights have separate access conditions and
licenses and are not distributed here. This checkout currently has no separate
top-level project `LICENSE` file.
