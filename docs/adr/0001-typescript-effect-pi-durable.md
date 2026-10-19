# ADR 0001: Rebuild on TypeScript, Bun, Effect, and Pi Durable

Status: accepted (2026-10-19)

## Context

LORA 0.1 was about 26,000 lines of Python plus 10,000 lines of tests: a PySide6 HUD, a FastAPI gateway,
faster-whisper, Kokoro, openWakeWord, a Laya router, FAISS, NLI verifiers, a supervisor graph with its own
checkpointing, and a prompt-evolution loop. Most of that code was orchestration: queues, retries, checkpoints,
job state, and stitching streams between threads. It was hard to change without breaking something.

Earendil released Pi 1.0 and [Pi Durable](https://www.npmjs.com/package/@earendil-works/pi-durable) in October
2026. Pi Durable commits every model turn, tool call, and piece of app state to storage before anything is shown,
and resumes unfinished work after a crash. That covers the orchestration LORA had built by hand.

We considered two stacks for the rebuild:

1. **TypeScript on Bun, with Effect and Pi Durable**, and a web UI used by both the desktop window and the phone.
2. **Rust with GPUI** (Zed's UI framework) for a native desktop app.

## Decision

TypeScript on Bun, with Effect v4 for services, errors, and HTTP, Pi Durable for the agent, and a React UI whose
state lives in Effect Atom. One `lora` binary serves the UI to an app-mode browser window on the PC and to the
phone.

## Why

| | TypeScript + Bun + Pi Durable | Rust + GPUI |
|---|---|---|
| Durable agent runtime | Pi Durable runs in-process | None in Rust. Pi Durable would need a Node sidecar and an IPC protocol |
| Languages to maintain | 1 | 2 (Rust UI + TypeScript agent) |
| Phone client | Same UI | Needs a separate web UI anyway |
| Library maturity | Effect 4 stable; Pi Durable experimental (pinned) | GPUI pre-1.0, published as `gpui-pre-*` crates that move together and break often |
| Distribution | `bun build --compile`: one 88 MB file per OS, UI embedded, cross-compiles | Native binary per OS, plus the Node sidecar |
| Agent-friendliness | Huge training corpus; one type system end to end; tests run in under a second | Smaller corpus for GPUI; borrow-checker iteration loops; two toolchains |

Measured on the built binary (Linux x64): 213 ms from launch to first HTTP response, 40 MB resident while idle.
GPUI would render faster than a browser window, but LORA's UI is a chat transcript, and its latency is dominated
by the local model, not the UI.

The deciding factor is that Pi Durable exists only for TypeScript. Choosing Rust would mean re-implementing it, or
running Node anyway.

## What was dropped, and why

- **Wake word, local Whisper/Kokoro bundles, global hotkey.** The UI uses the browser's speech recognition and
  voices, or any OpenAI-compatible STT/TTS server you point it at (for example speaches or Kokoro-FastAPI). No
  ML runtimes ship inside LORA.
- **Laya router, supervisor graph, round-table critics, RSI outer loop, prompt genome.** The model chooses tools
  itself, and Pi Durable provides durability. Grounding is one hook: an answer that links to a page no tool
  returned is sent back once for a rewrite.
- **FAISS and embeddings, NLI verifiers.** `fetch_page` keeps the passages that match the question instead.
- **Browser automation of other chatbots.** Fragile, and it needed Playwright. `open` can still open those sites.
- **PySide6 HUD and droplet widget.** Replaced by the web UI and a CSS orb.

## Consequences

- Pi Durable is experimental and its API changes between releases. `@earendil-works/*` versions are pinned exactly
  and upgraded together, by hand, after reading the changelog.
- Effect packages must stay on matching versions (`effect`, `@effect/*`). Dependabot groups them.
- Conversation storage is a single SQLite file (`~/.lora/lora.sqlite`) owned by one process. A second `lora`
  notices the port is taken and just opens the window.
