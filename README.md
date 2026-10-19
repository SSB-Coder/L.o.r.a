# L.O.R.A — Layer Oriented Research Assistant

A voice-controlled, local-first agentic research and desktop assistant system running on Windows.

## Overview
L.O.R.A listens (STT), speaks (TTS via Kokoro-ONNX), operates allowlisted applications and files within a strict safety envelope, browses and reads sources from the web and local documents, and returns verified, source-grounded answers.

## Architecture
- **Voice Pipeline**: Faster-Whisper on CPU for speech-to-text; Kokoro-ONNX on CPU for fast, lightweight neural text-to-speech.
- **Decision Engine**: Convai Laya ModernBERT on CPU for intent, risk, and depth classification.
- **Models**: Local Ollama generative models (`spark-x25-4b-8k` base writer; `granite4.2:8b` / `Qwythos-9B` for complex queries).
- **Grounding & Session RAG**: Ephemeral session vector indexing via `EmbeddingGemma-2` and `bge-reranker-base`.
- **Desktop UI**: Ambient screen-corner droplet widget and holographic HUD via PySide6.
- **Mobile Companion**: Mobile web companion on local isolated LAN.

## Getting Started
Run `run_lora.bat` or launch via:
```bash
python -m lora.main
```
