# Third-Party Notices

LORA release binaries bundle the Bun runtime and the packages below. Each keeps its own license; full license
texts are in each package's directory under `node_modules/` after `bun install`, and in the packages' repositories.

## Runtime

| Component | License | Used for |
| --- | --- | --- |
| [Bun](https://bun.sh) | MIT (bundles JavaScriptCore, LGPL-2.1, dynamically linked) | Runtime, HTTP server, SQLite, `HTMLRewriter`, single-file builds |
| `@earendil-works/pi-durable`, `pi-ai`, `chord` | MIT | Durable agent harness, model access, document state |
| `effect`, `@effect/platform-bun`, `@effect/platform-node-shared`, `@effect/atom-react` | MIT | Services, schemas, errors, CLI, UI state |
| `react`, `react-dom`, `scheduler` | MIT | UI |
| `marked` | MIT | Markdown rendering |
| `dompurify` | MPL-2.0 OR Apache-2.0 (used under Apache-2.0) | Sanitizing rendered Markdown |
| `qrcode-generator` | MIT | Phone pairing QR code |
| `typebox` | MIT | Tool parameter schemas (via pi-ai) |
| `diff` | BSD-3-Clause | Via pi-durable |
| `openai` | Apache-2.0 | OpenAI-compatible client used to reach Ollama (via pi-ai) |
| `partial-json` | MIT | Streaming tool-call arguments (via pi-ai) |

pi-ai also depends on SDKs for providers LORA doesn't use (Anthropic, Google, AWS Bedrock). They are bundled
but never loaded.

## External services and programs (not bundled)

| Component | License | Notes |
| --- | --- | --- |
| [Ollama](https://ollama.com) | MIT | Local model server LORA talks to |
| Models you pull into Ollama | Their own licenses | Check each model's license before use |
| DuckDuckGo, Wikipedia, arXiv, Crossref | Their terms of use | Queried for research; arXiv results are marked as preprints |
| Optional STT/TTS servers (e.g. speaches, Kokoro-FastAPI) | Their own licenses | Only if you configure them |

LORA 0.1 (the Python version) drew architectural ideas from laya, storm, langgraph, dspy, and others; none of that
code is part of this version.
