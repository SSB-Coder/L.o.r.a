# Working on LORA

LORA is one Bun process: an Effect program that opens a Pi Durable harness over SQLite, serves a React UI and a
small JSON/SSE API, and talks to Ollama through pi-ai's OpenAI-compatible client.

## Commands

```sh
bun install            # respects bunfig.toml: nothing younger than 14 days is resolved
bun run check          # biome check, tsc --noEmit, bun test: run before every commit
bun run fix            # biome autofix and format
bun run dev            # LORA_DEV=1, hot reload; set LORA_HOME to a temp dir to keep your data safe
bun run build [target] # dist/ archives; targets: linux-x64 linux-arm64 darwin-arm64 darwin-x64 windows-x64
```

## Layout

| File | What it owns |
| --- | --- |
| `src/main.ts` | CLI (`lora`, `lora pair`) with `effect/cli`; builds the app from Layers; opens the window |
| `src/config.ts` | `lora.yaml` schema (Effect Schema; every field defaulted) |
| `src/agent.ts` | Pi Durable: the `lora` extension (prompt sections, policy and grounding hooks), the chat index document, harness lifecycle (`Agent` service) |
| `src/tools.ts` | Every tool the model can call, plus the `Sources` and `Memory` documents |
| `src/policy.ts` | Tool tiers per origin, path containment, private-address checks. Pure functions |
| `src/web.ts`, `src/html.ts` | SSRF-guarded fetches, search, papers, HTML-to-text (Bun's `HTMLRewriter`) |
| `src/models.ts` | Ollama model discovery and the pi-ai provider |
| `src/server.ts`, `src/auth.ts` | HTTP API, SSE event stream, tokens, Host/Origin checks, login rate limit |
| `src/ui/` | React UI. `chat.ts` folds Pi Durable agent events into display turns (pure, tested); `state.ts` holds Effect Atoms |
| `test/` | `bun test`. `helpers.ts` builds an agent on in-memory storage with pi-ai's faux provider |

## How a message flows

`POST /api/chats/:id/messages` → `agent.send()` → `conversation.submit()` (durably admitted, idempotent by
`requestId`) → Pi Durable's generation task calls Ollama → tool calls run as durable tool tasks (our
`beforeTool` hook checks the chat's origin) → `onYield` hook checks citations → entries commit. The UI watches
`GET /api/chats/:id/events`, which streams `watchEvents()` batches, and `src/ui/chat.ts` reduces them.

## Conventions

- Effect code: `Effect.fn("name")` for reusable effects, `Schema` for anything parsed from outside (config,
  request bodies, API responses), tagged errors. `node_modules/effect/AGENTS.md` is the v4 reference; v3 examples
  online are often wrong (`Context.Service`, not `Context.Tag`).
- Pi Durable: read the types in `node_modules/@earendil-works/pi-durable/dist`, not newer READMEs online; the
  pinned version's API differs. Conversation ids are numbers inside Pi Durable and strings everywhere in LORA.
- A tool is a `defineTool` in `src/tools.ts` plus a tier in `TOOL_TIERS` (`src/policy.ts`). An untiered tool is
  refused. Tools that only read are `replay: "safe"`; tools with side effects are not.
- Anything a phone can reach must stay read-only and public-web only. Add a test when you touch policy.
- Match the surrounding style: short comments that say why, no dead code, no new dependencies without a reason.
- Tests use the faux provider (`fauxAssistantMessage`, `fauxToolCall`); never a real model or the network,
  except `src/web.ts` behaviour, which is tested through its pure parsers.

## Dependencies and releases

- [docs/dependencies.md](docs/dependencies.md): 14-day rule, security auto-merge, override ledger, packages
  that must move together.
- [docs/releases.md](docs/releases.md): nightlies from `master`, stables from `release/X.Y`; `package.json` is the
  only place the version lives.
