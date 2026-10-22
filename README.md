# L.O.R.A — Layer Oriented Research Assistant

Local-first research and bounded desktop assistance for Windows, targeting an
RTX 3050 with 6 GB VRAM and 16 GB RAM.

The design is [LORA_PLAN.md](docs/LORA_PLAN.md), with the user-approved
[UPDATED_PLAN.md](docs/UPDATED_PLAN.md) overriding its voice/personality choices. The supplied audit
is preserved in [CONFORMANCE_AUDIT_SOURCE.txt](docs/CONFORMANCE_AUDIT_SOURCE.txt).
Read [REMEDIATION_STATUS.md](docs/REMEDIATION_STATUS.md) for implemented changes,
remaining gaps and verification limits. This snapshot is a remediation build;
full v1.5 acceptance has not been established.

The latest user-approved workflow is described in
[SIMPLE_RESEARCH_UPDATE.md](docs/SIMPLE_RESEARCH_UPDATE.md). Everyday research now
uses the warm writer and lightweight citation-reference checks. It does not run
the former NLI, second-opinion, or iterative verification pipeline.

## Install and run

Use Python 3.12, uv, and an installed Ollama daemon.

```powershell
uv sync --locked --extra dev
uv run playwright install chromium
uv run lora start
```

To add optional DOCX and quality-tool support after syncing:

```powershell
uv pip install -r requirements-conformance.txt
```

Model weights are not included in source archives. Existing local model
directories are reused. Configure `models_dir` and role assignments in
`config/lora.yaml`. Local Whisper wake-phrase listening needs no custom wake weights; see
[VOICE_SETUP.md](docs/VOICE_SETUP.md). Source-reference checks do not establish
factual accuracy; the interface does not present them as truth scores.

`run_lora.bat` remains available. Start without audio with
`uv run lora start --no-audio`; `--headless` starts the gateway worker.
`lora stop` and `lora kill` request clean shutdown through local, ACL-protected
application IPC. The emergency hotkey cancels jobs while keeping the app open.

## Hardware profile

- Keep `OLLAMA_MAX_LOADED_MODELS=1` in the environment of the **Ollama daemon**;
  setting it only in the client cannot reconfigure an already-running daemon.
- Spark writer: quantized 4B, 8192-token context. Long-reader tags: 32768.
- Granite controller/supervisor retains its 6144-token profile. Routine research
  does not switch to a verifier model.
- Embeddings, wake detection, and INT8 STT use CPU. Phone and desktop research
  share one embedding engine; optional verification modules are outside normal turns.
- Kokoro is the only active speech engine, explicitly on CPU with two
  inference threads. Speech does not allocate GPU VRAM or switch engines.
- Desktop WebGL uses half-resolution rendering and a 30 fps limit, and pauses
  while hidden/asleep. The original shader and droplet animations are retained.
- No RAM, latency, VRAM-fit, or truthfulness percentage is promised without
  measurements. [BENCHMARKS.md](docs/BENCHMARKS.md) records the current evidence.

## Safety boundaries

`config/policy.yaml` supplies effective allowed roots and action permissions.
Default roots are Documents, Downloads, Desktop, and `~/LORA/outputs`; the
entire Users tree is not allowed.

Editor typing requires exact-text confirmation and foreground checks. MCP startup is explicit desktop opt-in. The registry has no
media-removal tool. Opening files uses a strict viewer extension allowlist;
CSV/DOCX external opening is disabled until a safe viewer is supplied. Exports
use exclusive creation and suffix collisions instead of overwriting files.

Listener origin determines capabilities. Mobile cannot consult the desktop
media index or access local-file tools. Desktop data APIs require a desktop
Bearer token; mobile requires its distinct token. Host/Origin validation wraps
every HTTP route. Pairing puts tokens in a URL fragment, then the client sends
them in Authorization headers or the first WebSocket auth message.

The companion's static `/app/` shell is public and contains no user data;
all data APIs are authenticated. Legacy HTTP media routes are retired.
`/api/hud/chat` is a queued compatibility endpoint returning HTTP 202, not an
unverified direct-model response.

## Research and verification

The latest document extraction/chunking fixes and the distinct mobile pearl UI
are described in [DOCUMENT_MOBILE_REPAIR.md](docs/DOCUMENT_MOBILE_REPAIR.md),
including validation, performance measurements and reindex instructions.

Browsing defaults to **quick**: retrieve pages, answer with citations, and label
the result as not independently verified. Request **verified research** for
standard verification, or **deep research** for independent review/refinement.
Conversational replies do not inherit the previous research mode.

Added desktop sources take priority for ordinary questions. Explicit web,
latest or news requests can still search online. Page embedding runs on CPU
with visible chunk progress and a disclosed lexical fallback. See
[RESEARCH_REPAIR.md](docs/RESEARCH_REPAIR.md) for fixes and validation limits.

Retrieved instructions are quarantined before indexing; a bounded parser
subprocess handles HTML/PDF/DOCX on the wired extraction paths. Resource limits
and subprocess timeouts apply. A complete restricted-token parser sandbox,
all ingestion-path consolidation, and a neural injection classifier still
require work.

Heuristic NLI diagnostics are explicitly unreliable. Word overlap and failed
LLM checks cannot certify a claim. Cross-family V4 checking requires an exact
supporting quote. Final answers in verified/deep modes use repaired/verified text, and the
composite is the supported-claim fraction. The repository's synthetic gold
fixture is not evidence of a human-labelled G3 result.

## Checks

```powershell
uv run python -m ruff check src tests/unit/test_plan_conformance.py
uv run python -m pytest tests -q --basetemp scratch/pytest-run
```

GitHub Actions contains Windows/Linux tests and separate lint/type/security
jobs. They have not been run remotely in this repair session. Regression tests follow the updated product contract.
Live model evaluations are explicit (`--run-model-evals`); install the intended
local assets first. Page-level jsdom checks require Node and jsdom.


## Reliability review

See [DEEP_REVIEW.md](docs/DEEP_REVIEW.md) for the 22 October review, measured
audio-ring improvements, validation limits and engineering rating.
[FEATURE_HISTORY.md](docs/FEATURE_HISTORY.md) records preserved/restored features.
Ctrl+Esc interruption is restored. Research timeouts cancel actual tasks and
retain ownership through cleanup; skipped verification stages no longer crash
deep-research presentation. Existing speaker enrollment is discovered in place.

The default CPU vector backend is NumPy exact inner-product search, avoiding
the reproduced Windows FAISS DLL failure. Set grounding.vector_store to faiss
or LORA_VECTOR_BACKEND=faiss only with a validated native runtime. Legacy media
indexes rebuild in RAM from saved text without rewriting the original index.

Runtime logs rotate under logs/lora.log. Native faults are recorded in
logs/native-fault-<pid>.log. Keep these local files when diagnosing unexpected
exits; model availability alone is not evidence of stable hardware operation.
The current regression run recorded 810 passed, 13 skipped, zero failures/errors; raw results are in docs/validation.
Live hardware/model acceptance remains separate. All changes are in this checkout; no ZIP download is required.

## Current desktop workflow

See [WORKFLOW_RESTORATION.md](docs/WORKFLOW_RESTORATION.md) for the latest user-requested workflow: continuous addressed listening (including sleep by default), deferred recognition while busy, writer-first startup/residency, isolated CPU speech, confirmed editor typing and optional MCP. This supersedes older sleep/deferral descriptions. Final validation: **810 passed, 13 skipped, no failures/errors**; live hardware acceptance remains separate.
