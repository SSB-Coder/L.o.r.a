# Third-Party Notices & License Attributions

L.O.R.A (Layer Oriented Research Assistant) is a clean-room implementation written for local-first Windows execution. Where architectural patterns or open-source runtime libraries are used, their licenses and notices are documented below.

---

## 1. Architectural Pattern Attributions (Plan §3.1)

No GPL/AGPL source code is copied into `src/lora/`. The following open-source projects served as architectural inspiration:

| Project | License | Architectural Pattern Adapted |
|---|---|---|
| `convaiinnovations/laya` | Apache-2.0 | ModernBERT-large 421M parameter CPU decision routing (`src/lora/models/laya_router.py`) |
| `stanford-oval/storm` | MIT | Multi-perspective query decomposition and cited synthesis (`src/lora/agent/research.py`) |
| `langchain-ai/langgraph` | MIT | Typed state graph and SQLite checkpointing concepts (`src/lora/experimental/supervisor/`) |
| `stanfordnlp/dspy` | MIT | Declarative prompt genome compilation and signature scoring (`src/lora/experimental/outer_loop/`) |
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
| `faiss-cpu`, `numpy`, `scipy` | MIT / BSD-3-Clause | NumPy CPU retrieval by default; FAISS supplied through the optional `faiss` extra |
| `sentence-transformers`, `transformers`, `onnxruntime` | Apache-2.0 / MIT | CPU embeddings (`EmbeddingGemma-2`), reranking (`bge-reranker-base`), NLI |
| `faster-whisper`, `ctranslate2` | MIT | CPU Speech-to-Text |
| `kokoro-onnx` | Apache-2.0 | CPU Text-to-Speech |
| `PySide6` (Qt for Python) | LGPL-3.0 (dynamic linking) | Desktop holographic HUD, Ambient Droplet Widget, and System Tray |
| `playwright` | Apache-2.0 | Headless web extraction and user-confirmed chatbot DOM automation |
| `trafilatura`, `beautifulsoup4` | Apache-2.0 / MIT | Boilerplate removal and article text extraction |
| `PyMuPDF` (`pymupdf`) | AGPL-3.0 | Local PDF text extraction (see AGPL-3.0 notice below) |

---

## 3. PyMuPDF dependency notice

PDF extraction in `src/lora/safety/parser_worker.py` uses `pymupdf`, which is
currently a required runtime dependency. There is no automatic `pypdf` fallback
in this implementation. The callers in `tools/files/read.py` and
`tools/browser/pdf_fetcher.py` use that isolated parser. The existing AGPL-3.0 attribution applies to this dependency; model weights
have their own terms. The previous notice describing PyMuPDF as an optional
backend with an automatic pypdf replacement did not match the implementation.
