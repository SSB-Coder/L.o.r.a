> Earlier repair snapshot. Current behavior and validation: [WORKFLOW_RESTORATION.md](WORKFLOW_RESTORATION.md).

# Benchmark and validation status

Target hardware: RTX 3050 laptop, 6 GB VRAM, 16 GB RAM.
Review date: 2026-10-21.

The previous table contained performance percentages and PASS claims without
raw run artifacts, used inconsistent hardware/model descriptions, and called
a synthetic fixture human-labelled. Those numbers are withdrawn as acceptance
evidence. Git history retains the previous document.

## Checks actually executed in this repair

- Focused policy/auth/export/ingestion regression selection: 160 passed.
- Expanded synchronous selection before the last additions: 306 passed,
  2 asynchronous tests excluded by an explicit marker selection.
- Final targeted security/QR/deferred-typing selection: 155 passed (146 conformance cases).
- Isolated HTML extraction smoke check under Windows Job Objects: passed.
- Mobile app/orb JavaScript syntax: passed; 9 jsdom HUD tests skipped (jsdom absent).
- Ruff over source and the new conformance module: passed.
- Python compileall over source: passed.

The final expanded result is recorded in REMEDIATION_STATUS.md. These are
code checks, not measurements of model quality, hardware fit, or LAN isolation.
No full-suite green claim or 80% coverage claim is made.

## Environment limits reproduced

An empty `asyncio.run(asyncio.sleep(0))` hangs while Windows creates its
loopback socket pair. A five-second faulthandler trace locates the block in
`socket._fallback_socketpair → accept`, called from the Proactor event loop.
Gateway TestClient, aiosqlite, Ollama/network and async integration tests
therefore cannot complete in this restricted execution environment.
The selected synchronous tests exercise middleware and queue functions
directly; they do not substitute for ASGI transport validation.

Pytest's default temporary directory is also unwritable here. Workspace
`--basetemp` resolves that separate problem.

`nvidia-smi` returned: “Failed to initialize NVML: GPU access blocked by the
operating system”. GPU inference, audio, firewall setup and orb-on/off inference
throughput were not measured.

Dependency lock regeneration could not reach PyPI (DNS denied). Runtime
dependency metadata and the existing lock are kept consistent; optional
backends and quality tools are listed separately in requirements-conformance.txt.

## Measurements still required on the unrestricted PC

| Measurement | Acceptance / required record | Current evidence |
|---|---|---|
| Wake to Kokoro acknowledgement | G1 below 2.5 s; raw timestamps | Not measured |
| Benign/forbidden control | G2 success and zero forbidden actions | Deterministic policy coverage only |
| Grounding precision | G3 at least 95%, independent human-labelled claims | Not measured; existing fixture is synthetic |
| Supervisor/round table | Real model adversarial cases, family isolation | Not validated |
| Prompt refinement | Three baselines, noise margin, improvement and rollback | Core unit tests exist; live run pending |
| Peak resident RAM | Per-component and combined peak below host ceiling | Not measured |
| Peak VRAM and swaps | Model/tag/context, load/unload latency, CPU offload | Not measured |
| STT and quality speech | RTF, voice accuracy, full GPU worker release | Not measured |
| Desktop visual overhead | Ollama tok/s with orb on/off, identical prompts | Not measured |
| Mobile isolation | Actual adapter, firewall, primary-WiFi external probe | Not validated here |
| Laya | Held-out accuracy, temperatures, confusion matrix | Existing local checkpoint warns about invalid temperature |

## Upstream corrections

Laya English and typed-decision checkpoints are ModernBERT-large, approximately
421M parameters, Apache-2.0; the former MIT/20M description was wrong.
Source: <https://pypi.org/project/laya/>.

HHEM-2.1's documented interface uses
`AutoModelForSequenceClassification.from_pretrained(...).predict(pairs)`.
An embedding-similarity blend is not semantic entailment.
Source: <https://huggingface.co/vectara/hallucination_evaluation_model>.

Active speech uses CPU Kokoro only; UPDATED_PLAN.md supersedes the earlier
Piper and quality-Qwen3 design choices.


## 22 October 2026 CPU Kokoro revision

197 targeted voice/dialogue/config/conformance tests passed. Real CPU Kokoro
synthesis produced 58,368 samples at 24 kHz: 2.432 seconds of speech in 5.73
seconds including cold load, with no speech-engine fallback. The smoke check
used a writable temp directory because the sandbox denied espeak DLL access
under its default temp path. This does not measure wake latency, warm synthesis
latency, speaker output, microphone accuracy, or total application RAM/VRAM.
See validation/voice-update-tests.xml and validation/kokoro-cpu-smoke.json.


## 22 October reliability follow-up

See DEEP_REVIEW.md and FEATURE_HISTORY.md for task/cancellation repair,
Ctrl+Esc restoration, stateful VAD and enrollment recovery, shared lazy router,
safe exact NumPy vector retrieval, WebSocket owner-loop dispatch and known
limitations. The NumPy backend supersedes FAISS-only default selection because
the local native dependency failed during validation. Original source files
remain preserved. Raw current results: validation/review-summary.json,
reliability-tests.xml, broad-review-tests.xml and audio-smoke.json. That earlier broad
snapshot is superseded by the final green regression run in completion-tests.xml.
Live model/hardware acceptance is still outstanding. Older overlapping
test counts above must not be summed into a current pass count.

## Final regression closure - 22 October 2026

The full regression run recorded **797 passed, 13 skipped, zero failures/errors**. See
DEEP_REVIEW.md and validation/review-summary.json for exact scope and skip reasons.
Controlled desktop voice/chat and HUD controller checks are included. Cache and
event-loop shutdown cleanup is explicit; obsolete test contracts are migrated.
This closes the reported regression repair/testing pass, not the entire v1.5
roadmap or the outstanding real-room/GPU/model-quality acceptance work.

## Superseding workflow revision

The later user-directed restoration is documented in WORKFLOW_RESTORATION.md. Its continuous sleep listening option, enrolled threshold, writer-first policy, isolated speech processes, editor confirmation and MCP opt-in supersede the earlier contracts above. Current test output is validation/prime-tests.xml.
