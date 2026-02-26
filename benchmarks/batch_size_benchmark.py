"""Batch-size benchmark — measures throughput at different concurrency levels.

First validates correctness (single vs batch=1 logit match), then
benchmarks batch sizes [1, 2, 4, 6, 8] at 20 decode steps each.
"""

from __future__ import annotations

import logging
import time

import torch
from rich.console import Console
from rich.table import Table

from vllm_mps.core.sequence import SamplingParams
from vllm_mps.engine.llm_engine import LLMEngine

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
console = Console()

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPTS = [
    "The meaning of life is",
    "In a galaxy far far away",
    "The best programming language is",
    "Artificial intelligence will",
    "Once upon a time in",
    "The future of computing is",
    "Deep learning works by",
    "The most important invention is",
]
NUM_TOKENS = 20


def validate_correctness(engine: LLMEngine) -> bool:
    """Compare single-token vs batch=1 logits for correctness."""
    console.rule("[bold cyan]Correctness Validation: single vs batch=1")

    prompt = "The meaning of life is"
    token_ids = engine.tokenizer.encode(prompt)

    # We need a fresh sequence for single-token path.
    # Allocate seq, run first few tokens via single-token to build KV history.
    seq_id_single = 900
    engine.kv_cache_manager.allocate(seq_id_single)

    # Process prompt tokens sequentially (building KV cache).
    for i, tok in enumerate(token_ids):
        engine.kv_cache_manager.append_slot(seq_id_single)
        engine.adapter.forward_single_token(tok, i, seq_id_single)

    # Now get logits for next decode step via single-token path.
    engine.kv_cache_manager.append_slot(seq_id_single)
    pos_single = len(token_ids)
    # Use a dummy decode token (the last prompt token).
    decode_token = token_ids[-1]
    logits_single = engine.adapter.forward_single_token(
        decode_token, pos_single, seq_id_single
    )

    # Free single sequence.
    engine.kv_cache_manager.free(seq_id_single)
    for pl in engine.adapter.paged_layers:
        pl.evict_sequence(seq_id_single)

    # Now do the same via batch=1.
    seq_id_batch = 901
    engine.kv_cache_manager.allocate(seq_id_batch)

    for i, tok in enumerate(token_ids):
        engine.kv_cache_manager.append_slot(seq_id_batch)
        engine.adapter.forward_single_token(tok, i, seq_id_batch)

    engine.kv_cache_manager.append_slot(seq_id_batch)
    pos_batch = len(token_ids)
    logits_batch = engine.adapter.forward_batch(
        [decode_token], [seq_id_batch], [pos_batch], [pos_batch + 1]
    )

    engine.kv_cache_manager.free(seq_id_batch)
    for pl in engine.adapter.paged_layers:
        pl.evict_sequence(seq_id_batch)

    diff = (logits_single - logits_batch[0]).abs().max().item()
    console.print(f"Max logit difference (single vs batch=1): [bold]{diff:.6f}[/]")

    if diff < 0.01:
        console.print("[bold green]✅ Correctness check PASSED[/]")
        return True
    else:
        console.print(f"[bold red]❌ Correctness check FAILED: diff={diff}[/]")
        return False


def benchmark_batch_size(engine: LLMEngine, batch_size: int) -> dict:
    """Run NUM_TOKENS decode steps with `batch_size` concurrent prompts."""
    params = SamplingParams(max_tokens=NUM_TOKENS, temperature=0.8)

    # Submit `batch_size` prompts.
    req_ids = []
    for i in range(batch_size):
        req_id = engine.add_request(PROMPTS[i % len(PROMPTS)], params)
        req_ids.append(req_id)

    # Step until all finished.
    t0 = time.perf_counter()
    while any(not engine._outputs[rid].finished for rid in req_ids):
        engine.step()
    dt = time.perf_counter() - t0

    total_tokens = batch_size * NUM_TOKENS
    tok_sec = total_tokens / dt
    ms_per_step = (dt / NUM_TOKENS) * 1000

    mem_mb = 0.0
    if torch.backends.mps.is_available():
        mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)

    return {
        "batch_size": batch_size,
        "tok_sec": tok_sec,
        "ms_per_step": ms_per_step,
        "total_time": dt,
        "mem_mb": mem_mb,
    }


def main() -> None:
    torch.manual_seed(42)

    console.rule("[bold cyan]Batch Size Benchmark — TinyLlama 1.1B")
    console.print(f"Model: {MODEL_NAME}")
    console.print(f"Tokens per prompt: {NUM_TOKENS}")
    console.print()

    # Load engine (includes warmup).
    console.print("[yellow]Loading model...[/]")
    t_load = time.perf_counter()
    engine = LLMEngine.from_pretrained(MODEL_NAME)
    load_time = time.perf_counter() - t_load
    console.print(f"[green]Loaded in {load_time:.2f}s[/]")
    console.print()

    # Step 1: Correctness validation.
    with torch.no_grad():
        passed = validate_correctness(engine)
    if not passed:
        console.print("[bold red]Stopping — correctness check failed![/]")
        return

    console.print()

    # Step 2: Benchmark at different batch sizes.
    console.rule("[bold cyan]Throughput Benchmark")
    results = []
    for bs in [1, 2, 4, 6, 8]:
        console.print(f"[yellow]Running batch_size={bs}...[/]")
        r = benchmark_batch_size(engine, bs)
        results.append(r)
        console.print(
            f"  → {r['tok_sec']:.1f} tok/sec, "
            f"{r['ms_per_step']:.1f} ms/step, "
            f"{r['total_time']:.2f}s total"
        )

    console.print()

    # Print results table.
    table = Table(title="Batch Size Benchmark", show_lines=True)
    table.add_column("Batch Size", justify="center")
    table.add_column("tok/sec", justify="right")
    table.add_column("ms/step", justify="right")
    table.add_column("Total (s)", justify="right")
    table.add_column("MPS mem (MB)", justify="right")

    for r in results:
        table.add_row(
            str(r["batch_size"]),
            f"{r['tok_sec']:.1f}",
            f"{r['ms_per_step']:.1f}",
            f"{r['total_time']:.2f}",
            f"{r['mem_mb']:.0f}",
        )

    console.print(table)

    # Find peak efficiency.
    best = max(results, key=lambda r: r["tok_sec"])
    console.print(
        f"\nPeak efficiency: batch_size={best['batch_size']} "
        f"at {best['tok_sec']:.1f} tok/sec"
    )


if __name__ == "__main__":
    main()
