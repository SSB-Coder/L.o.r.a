# L.O.R.A

A local research assistant. LORA runs on your own computer against [Ollama](https://ollama.com), researches
the web, reads your files when you ask, and cites what it actually read. You can talk to it, and you can reach
it from your phone over a private network.

Every conversation is durable. LORA is built on [Pi Durable](https://www.npmjs.com/package/@earendil-works/pi-durable),
which saves each model turn and tool call before showing it. If LORA crashes or the PC restarts mid-answer, the
answer finishes when LORA starts again.

## Install

You need Ollama running with at least one chat model, for example `ollama pull qwen3:8b`.

```sh
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/SSB-Coder/L.o.r.a/master/scripts/install.sh | sh
```

```powershell
# Windows
irm https://raw.githubusercontent.com/SSB-Coder/L.o.r.a/master/scripts/install.ps1 | iex
```

Or download an archive from [Releases](https://github.com/SSB-Coder/L.o.r.a/releases) and check it against
`SHA256SUMS`. Nightly builds are prereleases; pass `nightly` to either installer. See [docs/releases.md](docs/releases.md).

## Use

```sh
lora            # starts LORA and opens its window
lora --no-open  # starts it in the background; running `lora` again opens the window
lora pair       # prints the link that pairs a phone
lora --version
```

Ask questions in the window, or press the orb to talk. LORA can:

- search the web, read pages, and search papers on arXiv and Crossref, then answer with numbered citations
- list, find, and read files in your allowed folders (Documents, Downloads, and Desktop by default)
- save Markdown notes, and remember facts about you for later chats
- open apps and websites you've allowed, such as "open wikipedia and search for durable execution"

If an answer links to a page no tool returned, LORA is sent back once to fetch it or drop it.

## Configure

Settings live in `~/.lora/lora.yaml` (Windows: `%USERPROFILE%\.lora\lora.yaml`). Every setting is optional;
[lora.example.yaml](lora.example.yaml) lists them all with their defaults. The most common ones:

```yaml
ollama:
  model: qwen3:8b       # default model for new chats (each chat can switch)
  contextWindow: 8192   # match OLLAMA_CONTEXT_LENGTH
files:
  roots: ["~/Documents", "~/Projects"]
voice:
  tts: http://127.0.0.1:8880/v1/audio/speech   # e.g. Kokoro-FastAPI, for a better voice than the browser's
```

`~/.lora` also holds the conversation database (`lora.sqlite`) and access tokens (`tokens.json`).

## Phone access

Off by default. Bind it to an isolated network interface, never to every interface:

```yaml
server:
  mobile: { host: 192.168.50.10, port: 8765 }
```

Then scan the QR code under **Pair a phone** in the sidebar. Phones can chat and research, and nothing else: no
files, notes, memory, apps, or desktop chats. [docs/NETWORK_SETUP.md](docs/NETWORK_SETUP.md) covers the network.

## Security model

- The app window is served on `127.0.0.1` only. Every API request needs that listener's token (exchanged for an
  HttpOnly cookie), a matching `Host` header (against DNS rebinding), and a same-site `Origin` (against CSRF).
  Five bad tokens in a minute ban the address for 15 minutes.
- Tools have tiers. Phone chats are offered only public-web tools, and a Pi Durable `beforeTool` hook refuses
  anything else even if a model asks.
- File tools resolve symlinks and stay inside the allowed folders. Network paths, alternate data streams, and
  URLs are refused. `open` never launches programs or scripts, and on Windows a link never passes through `cmd`.
- Web fetches refuse private, loopback, link-local, and cloud-metadata addresses, on every redirect hop.
- Nothing deletes files. `save_note` never overwrites.

## Develop

```sh
bun install
bun run dev        # LORA with hot reload (LORA_HOME=/tmp/lora to keep it apart from your real data)
bun run check      # lint, types, tests
bun run build      # binaries for every platform in dist/
```

[AGENTS.md](AGENTS.md) explains the layout and conventions. [docs/adr/0001-typescript-effect-pi-durable.md](docs/adr/0001-typescript-effect-pi-durable.md)
explains why LORA moved from Python to TypeScript, Effect, and Pi Durable, and what was dropped.
