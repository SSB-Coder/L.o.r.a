# Third-Party Notices & License Attributions

L.O.R.A (Layer Oriented Research Assistant) is a clean-room implementation written for local-first Windows execution. Where architectural patterns or open-source runtime libraries are used, their licenses and notices are documented below.

---

## 1. Architectural Pattern Attributions (Plan §3.1)

No GPL/AGPL source code is copied into `src/lora/`. The following open-source projects served as architectural inspiration:

| Project | License | Architectural Pattern Adapted |
|---|---|---|
| `convaiinnovations/laya` | MIT | Multi-head 20M parameter CPU decision routing (`src/lora/models/laya_router.py`) |
| `stanford-oval/storm` | MIT | Multi-perspective query decomposition and cited synthesis (`src/lora/agent/research.py`) |
| `langchain-ai/langgraph` | MIT | Typed state graph and SQLite checkpointing concepts (`src/lora/supervisor/`) |
| `stanfordnlp/dspy` | MIT | Declarative prompt genome compilation and signature scoring (`src/lora/outer_loop/`) |
| `Paths-Design/Local-Computer-Use` | MIT | Deterministic allowlisted Windows app and window control (`src/lora/tools/control/`) |
| `BrowserMCP/mcp` | MIT | Site-specific DOM selector adapter pattern (`src/lora/tools/browser/adapters/`) |
| `Silero VAD` / `faster-whisper` | MIT | Streaming pre-roll VAD segmentation and CTranslate2 INT8 CPU STT (`src/lora/voice/`) |
| `Kokoro-ONNX` | Apache-2.0 | Zero-VRAM CPU neural speech synthesis (`src/lora/voice/tts/`) |

---

## 2. Runtime Dependency Licenses (Plan §8)

| Package | License | Usage in L.O.R.A |
|---|---|---|
| `ollama` (external daemon) | MIT | Local LLM inference server (`127.0.0.1:11434`) |
| `fastapi`, `starlette`, `uvicorn`, `pydantic` | MIT / BSD-3-Clause | Local & LAN Gateway API, WebSocket streaming, schema validation |
| `httpx` | BSD-3-Clause | Async HTTP client for Ollama and knowledge connectors |
| `faiss-cpu`, `numpy`, `scipy` | MIT / BSD-3-Clause | In-memory session vector index and local media vector database |
| `sentence-transformers`, `transformers`, `onnxruntime` | Apache-2.0 / MIT | CPU embeddings (`EmbeddingGemma-2`), reranking (`bge-reranker-base`), NLI |
| `faster-whisper`, `ctranslate2` | MIT | CPU Speech-to-Text |
| `kokoro-onnx` | Apache-2.0 | CPU Text-to-Speech |
| `PySide6` (Qt for Python) | LGPL-3.0 (dynamic linking) | Desktop holographic HUD, Ambient Droplet Widget, and System Tray |
| `playwright` | Apache-2.0 | Headless web extraction and user-confirmed chatbot DOM automation |
| `trafilatura`, `beautifulsoup4` | Apache-2.0 / MIT | Boilerplate removal and article text extraction |
| `pypdf` / `PyMuPDF` (`fitz`) | BSD-3-Clause / AGPL-3.0 | Local PDF text extraction (see AGPL-3.0 notice below) |

---

## 3. PyMuPDF (AGPL-3.0) Optional Dependency Notice

`src/lora/tools/files/read.py` and `src/lora/tools/browser/pdf_fetcher.py` support `PyMuPDF` (`pymupdf` / `fitz`, licensed under **GNU AGPL-3.0**) with an automatic BSD-licensed `pypdf` fallback when `PyMuPDF` is not installed. If distributing a binary build of L.O.R.A that bundles `PyMuPDF`, the distribution must comply with the GNU Affero General Public License v3.0 (or omit `PyMuPDF` in favor of the built-in `pypdf` backend for permissive-only distribution).
