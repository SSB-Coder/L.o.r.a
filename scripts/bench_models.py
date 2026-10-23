"""Phase 1 Model Benchmarking Harness for L.O.R.A.

Measures:
1. Ollama LLM throughput (tok/s, prompt eval tok/s, first-token latency).
2. JSON schema adherence and structured validation.
3. Ollama model swap latency under OLLAMA_MAX_LOADED_MODELS=1.
4. Local CPU embedding throughput and FAISS search latency using C:\\Models\\embeddinggemma-2.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure src/ is accessible
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import faiss
import httpx
import numpy as np
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer

from lora.models.ollama_client import OllamaClient


class BenchmarkStructuredResult(BaseModel):
    summary: str = Field(description="One-sentence summary of the topic")
    key_points: list[str] = Field(description="Two to three key bullet points")
    confidence: float = Field(description="Confidence between 0.0 and 1.0")


async def benchmark_llm(
    client: OllamaClient,
    model: str,
    test_prompt: str = "Explain quantum entanglement in two concise paragraphs.",
) -> dict[str, Any]:
    """Measure generation speed, duration, and structured output adherence for an Ollama model."""
    console = Console()
    console.print(f"[bold cyan]Benchmarking {model}...[/bold cyan]")

    # 1. Warm-up / Load Latency + Free-form generation
    raw_payload = {
        "model": model,
        "messages": [{"role": "user", "content": test_prompt}],
        "stream": False,
        "options": {"temperature": 0.2},
    }

    async with httpx.AsyncClient(timeout=120.0) as http:
        t0 = time.perf_counter()
        resp = await http.post(f"{client.base_url}/api/chat", json=raw_payload)
        t1 = time.perf_counter()

    if resp.status_code != 200:
        return {"model": model, "error": f"HTTP {resp.status_code}: {resp.text}"}

    data = resp.json()
    total_time = t1 - t0
    prompt_eval_count = data.get("prompt_eval_count", 0)
    prompt_eval_duration_ns = data.get("prompt_eval_duration", 1) or 1
    eval_count = data.get("eval_count", 0)
    eval_duration_ns = data.get("eval_duration", 1) or 1

    prompt_tok_per_sec = prompt_eval_count / (prompt_eval_duration_ns / 1e9)
    eval_tok_per_sec = eval_count / (eval_duration_ns / 1e9)

    # 2. Structured JSON generation test
    structured_prompt = (
        "Analyze the concept of 'Superconductivity'. Provide a 1-sentence summary, "
        "2 key points, and a confidence score."
    )
    t_struct_0 = time.perf_counter()
    structured_ok = False
    try:
        struct_res = await client.generate_structured(
            model=model,
            prompt=structured_prompt,
            response_model=BenchmarkStructuredResult,
            temperature=0.0,
        )
        structured_ok = isinstance(struct_res, BenchmarkStructuredResult)
    except Exception as e:
        console.print(f"[yellow]Structured generation error for {model}: {e}[/yellow]")
    t_struct_1 = time.perf_counter()

    return {
        "model": model,
        "total_latency_s": round(total_time, 2),
        "prompt_tokens": prompt_eval_count,
        "generated_tokens": eval_count,
        "prompt_eval_tok_s": round(prompt_tok_per_sec, 1),
        "eval_tok_s": round(eval_tok_per_sec, 1),
        "structured_json_pass": structured_ok,
        "structured_latency_s": round(t_struct_1 - t_struct_0, 2),
    }


async def measure_model_swap(
    client: OllamaClient,
    model_a: str,
    model_b: str,
) -> float:
    """Measure the time in seconds to swap from model_a to model_b."""
    # Ensure model_a is loaded
    await client.generate_chat(model=model_a, messages=[{"role": "user", "content": "hi"}])
    # Swap to model_b
    t0 = time.perf_counter()
    await client.generate_chat(model=model_b, messages=[{"role": "user", "content": "hi"}])
    t1 = time.perf_counter()
    return round(t1 - t0, 2)


def benchmark_embeddings_and_faiss(
    model_path: str | None = None,
) -> dict[str, Any]:
    """Benchmark local CPU embedding generation and FAISS similarity indexing."""
    default_models_dir = Path("models").resolve() if Path("models").exists() else Path(__file__).resolve().parent.parent / "models"
    resolved_path = model_path or os.environ.get(
        "LORA_EMBEDDING_MODEL_PATH",
        str(Path(os.environ.get("LORA_MODELS_DIR", str(default_models_dir))) / "embeddinggemma-2"),
    )
    console = Console()
    console.print(f"[bold green]Benchmarking Embeddings & FAISS with {resolved_path}...[/bold green]")

    t0_load = time.perf_counter()
    model = SentenceTransformer(resolved_path, device="cpu")
    t1_load = time.perf_counter()
    load_time_s = round(t1_load - t0_load, 2)

    # Synthetic corpus of 200 research passages
    test_corpus = [
        f"Research passage {i}: Investigation of deep learning architectures, attention mechanisms, "
        f"and efficient transformer inference on consumer hardware with bounded memory budgets."
        for i in range(200)
    ]

    t0_encode = time.perf_counter()
    embeddings = model.encode(test_corpus, normalize_embeddings=True, show_progress_bar=False)
    t1_encode = time.perf_counter()
    encode_time_s = t1_encode - t0_encode
    tok_per_sec = len(test_corpus) / encode_time_s

    dim = embeddings.shape[1]

    # FAISS IndexFlatIP (Cosine similarity since embeddings are normalized)
    t0_faiss = time.perf_counter()
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(embeddings))
    t1_faiss = time.perf_counter()
    index_build_ms = round((t1_faiss - t0_faiss) * 1000, 2)

    # 50 query lookups
    queries = [
        "How do attention mechanisms optimize transformer inference?",
        "What are bounded memory limits on consumer GPUs?",
        "Explain deep learning architectures for research.",
    ] * 10

    query_embs = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)

    t0_search = time.perf_counter()
    dists, indices = index.search(np.ascontiguousarray(query_embs), k=5)
    t1_search = time.perf_counter()
    avg_search_latency_ms = round(((t1_search - t0_search) / len(queries)) * 1000, 3)

    return {
        "model": "embeddinggemma-2 (740M)",
        "dim": dim,
        "load_time_s": load_time_s,
        "corpus_size": len(test_corpus),
        "encode_throughput_docs_s": round(tok_per_sec, 1),
        "faiss_index_build_ms": index_build_ms,
        "queries_tested": len(queries),
        "avg_search_latency_ms": avg_search_latency_ms,
    }


async def run_phase1_benchmark() -> None:
    """Execute the full Phase 1 benchmark suite."""
    console = Console()
    console.print("[bold yellow]===========================================================[/bold yellow]")
    console.print("[bold yellow]          L.O.R.A PHASE 1 MODEL BENCHMARK SUITE             [/bold yellow]")
    console.print("[bold yellow]===========================================================[/bold yellow]\n")

    client = OllamaClient()
    if not await client.is_online():
        console.print("[bold red]Ollama daemon is offline at http://127.0.0.1:11434. Aborting.[/bold red]")
        return

    installed = await client.installed_models()
    from lora.models.selection import PREFERRED_MODEL, is_excluded_model

    preferred_order = [
        PREFERRED_MODEL,
        "granite4.2:3b",
        "granite4.2:8b",
        "gemma4:e4b",
    ]
    candidate_models = [m for m in preferred_order if m in installed]
    if not candidate_models:
        candidate_models = [m for m in installed if not is_excluded_model(m)][:4]

    llm_results = []
    for m in candidate_models:
        try:
            res = await benchmark_llm(client, m)
            llm_results.append(res)
        except Exception as e:
            console.print(f"[red]Failed to benchmark {m}: {e}[/red]")

    # Measure swap latency between primary writer (Spark) and primary verifier (Granite)
    console.print("\n[bold cyan]Measuring model swap latency (Spark <-> Granite 3B)...[/bold cyan]")
    try:
        swap_time_s = await measure_model_swap(client, PREFERRED_MODEL, "granite4.2:3b")
        console.print(f"Swap latency: [bold green]{swap_time_s} s[/bold green]")
    except Exception as e:
        swap_time_s = None
        console.print(f"[yellow]Swap latency test skipped: {e}[/yellow]")

    # Embedding & FAISS benchmark
    emb_results = benchmark_embeddings_and_faiss()

    # Display LLM Table
    table = Table(title="Ollama Models Benchmark Results (RTX 3050 Laptop 6GB VRAM)")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Prompt tok/s", style="green")
    table.add_column("Gen tok/s", style="magenta")
    table.add_column("Latency (s)", style="white")
    table.add_column("JSON Schema Pass", style="yellow")

    for r in llm_results:
        if "error" in r:
            table.add_row(r["model"], "ERROR", "ERROR", "-", r["error"])
        else:
            table.add_row(
                r["model"],
                str(r["prompt_eval_tok_s"]),
                str(r["eval_tok_s"]),
                str(r["total_latency_s"]),
                "PASS" if r["structured_json_pass"] else "FAIL",
            )
    console.print(table)

    # Display Embedding Table
    emb_table = Table(title="Local CPU Embedding & FAISS Benchmark (embeddinggemma-2)")
    emb_table.add_column("Metric", style="cyan")
    emb_table.add_column("Value", style="green")
    emb_table.add_row("Embedding Dimension", str(emb_results["dim"]))
    emb_table.add_row("Model Load Time", f"{emb_results['load_time_s']} s")
    emb_table.add_row("Encoding Throughput", f"{emb_results['encode_throughput_docs_s']} docs/s")
    emb_table.add_row("FAISS Index Build (200 docs)", f"{emb_results['faiss_index_build_ms']} ms")
    emb_table.add_row("Avg Search Latency (k=5)", f"{emb_results['avg_search_latency_ms']} ms")
    console.print(emb_table)

    # Save output report
    output_dir = Path("outputs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "phase1_benchmark_report.json"

    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "llm_benchmarks": llm_results,
        "model_swap_latency_s": swap_time_s,
        "embeddings_and_faiss": emb_results,
    }

    report_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    console.print(f"\n[bold green]Report saved to {report_file}[/bold green]")


if __name__ == "__main__":
    asyncio.run(run_phase1_benchmark())
