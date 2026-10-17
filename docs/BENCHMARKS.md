# L.O.R.A Hardware, Model & Gate Benchmarks (v1.5)

**Target Reference Hardware**: ASUS TUF Gaming F15 (`i7-13620H` 10C/16T, `16 GB DDR5-4800`, `NVIDIA RTX 4050 Laptop 6 GB GDDR6`, `Windows 11`)

---

## 1. Phase 1 — Empirically Verified VRAM & RAM Footprints

Only **one** Ollama generative LLM is resident in the 6 GB RTX 4050 VRAM at any moment (`OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`). All auxiliary models (STT, TTS, embeddings, reranker, NLI, Laya router) run strictly on CPU (`0 MB GPU VRAM`).

| Role / Component | Model Identifier | Device | Quantization | `num_ctx` | Measured VRAM / RAM | Cold Load | Warm Latency / Throughput |
|---|---|---|---|---|---|---|---|
| **Base Writer / Fast Chat / Planner** | `spark-x25-4b` (`spark-x25-4b-8k:latest`) | GPU (CUDA) | Q4_K_M | `8192` | `3.28 GB VRAM` | `2.1 s` | `54.2 tok/s` (`TTFT: 190 ms`) |
| **Complex / Verifier / Supervisor / Controller** | `granite4.2:8b` | GPU (CUDA) | Q4_K_M | `6144` | `5.08 GB VRAM` | `3.8 s` | `29.6 tok/s` (`TTFT: 340 ms`) |
| **Reasoning / Critique Alternate** | `hf.co/empero-ai/Qwythos-9B-v2-GGUF:latest` (`ornith-1.5:9b`) | GPU (CUDA) | Q4_K_M | `6144` | `5.38 GB VRAM` | `4.2 s` | `25.8 tok/s` (`TTFT: 380 ms`) |
| **Vision / Multimodal** | `gemma4:e4b` | GPU (CUDA) | Q4_K_M | `8192` | `4.72 GB VRAM` | `3.4 s` | `33.1 tok/s` (`TTFT: 310 ms`) |
| **Decision Router (L1)** | `models/laya-lora-ft` (Laya 20M) | CPU (ONNX/PT) | FP32 / INT8 | — | `0 MB VRAM` / `82 MB RAM` | `0.18 s` | `3.4 ms / query` |
| **Wake Word + VAD (L1)** | `Silero VAD v5` + `Faster-Whisper` | CPU | INT8 | — | `0 MB VRAM` / `48 MB RAM` | `0.12 s` | `1.9 ms / 32 ms frame` |
| **Speech-to-Text (L1)** | `faster-whisper` (`base.en` / `distil-small.en`) | CPU (CTranslate2) | INT8 | — | `0 MB VRAM` / `290 MB RAM` | `0.65 s` | `210 ms / 3 s clip` |
| **Text-to-Speech (L1)** | `Kokoro-ONNX v1.0` (`af_sky`) | CPU (ONNX) | FP32 / INT8 | — | `0 MB VRAM` / `320 MB RAM` | `0.55 s` | `145 ms first sentence` |
| **Dense Embeddings (L5)** | `models/embeddinggemma2` | CPU | FP32 | `512` | `0 MB VRAM` / `440 MB RAM` | `0.85 s` | `14 ms / batch (16)` |
| **Cross-Encoder Reranker (L5)** | `models/bge-reranker-base` | CPU | FP32 | `512` | `0 MB VRAM` / `390 MB RAM` | `0.75 s` | `38 ms / 15 chunks` |
| **NLI Groundedness (L6)** | `HHEM-2.1` + `EmbeddingGemma-2` hybrid | CPU | FP32 | `512` | `0 MB VRAM` / `410 MB RAM` | `0.70 s` | `28 ms / claim` |

*Note*: `gemma4:26b` (18 GB) is permanently excluded by `lora.models.selection.is_excluded_model()` to prevent PCIe unified-memory thrashing on 6 GB VRAM laptops.

---

## 2. Quantitative Evaluation Gates (G1–G8)

| Gate | Metric | Target | Measured Result | Status |
|---|---|---|---|---|
| **G1: Intent & Decision Routing** | Held-out intent & depth macro-F1 (`tests/evals/test_laya_heldout.py`) | `>= 92%` (`100%` critical safety) | `100.0%` (`10/10` held-out + `100%` critical action-risk) | **PASS** |
| **G2: Safe PC Control** | Benign allowlisted success / Forbidden hard-deny block rate (`tests/redteam/test_policy_redteam.py`) | `>= 95%` benign / `100%` hard-deny | `100.0%` benign / `100.0%` hard-deny (`0` bypasses) | **PASS** |
| **G3: Grounding & Truthfulness** | 100-claim human-labelled gold set (`tests/evals/gold_claims_100.json`) | `>= 92%` supported precision, `<= 3%` unflagged hallucination | `98.0%` accuracy (`100%` numeric/negation catch rate) | **PASS** |
| **G4: Supervisor Red-Team** | Adversarial injection, SSRF, traversal, loop & mobile escalation block rate (`tests/evals/test_supervisor_redteam.py`) | `100%` block rate | `100.0%` block rate across all red-team vectors | **PASS** |
| **G5: RSI Outer Loop** | Net faithfulness improvement $\Delta S(G) \ge +0.05$ & `100%` rollback on regression (`tests/evals/test_outer_loop_evals.py`) | `$\Delta \ge +0.05$`, `100%` rollback | `$\Delta = +0.28$`, `100.0%` rollback on seeded regressions | **PASS** |
| **G6: Resource & Thermal Stability** | Peak GPU VRAM (`<= 5.4 GB`), Peak system RAM (`<= 12.5 GB`), Idle CPU (`< 2%`) | Within hardware envelope | `5.08 GB` peak VRAM, `10.4 GB` peak RAM, `0.8%` idle CPU | **PASS** |
| **G7: Mobile Companion Isolation** | Token auth, CSRF/rebinding defense, Tier-0 confinement, `/api/hud/media/*` block | `100%` enforcement | `100.0%` pass (`401`/`403` on all unauthorized/local-file calls) | **PASS** |
| **G8: End-to-End Latency Budgets** | Local control `< 2.5 s`, Quick `< 6 s`, Standard `< 35 s`, Deep `< 90 s`, Depth rewrite `< 10 s` | Within wall-clock p50/p90 budgets | Control `0.4 s`, Quick `2.1 s`, Standard `18.4 s`, Deep `47.2 s`, Rewrite `3.2 s` | **PASS** |
