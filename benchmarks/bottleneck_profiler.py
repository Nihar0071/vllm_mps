"""Bottleneck profiler — measures per-operation cost inside PagedAttentionLayer.

Loads TinyLlama, warms up MPS shaders, injects an OperationTimer,
generates 20 tokens, and prints a breakdown report.
"""

from __future__ import annotations

import logging
import time

import torch
from rich.console import Console
from rich.table import Table

from vllm_mps.core.sequence import SamplingParams
from vllm_mps.engine.llm_engine import LLMEngine
from vllm_mps.profiler.operation_timer import OperationTimer

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
console = Console()

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "The meaning of life is"
NUM_TOKENS = 20


def main() -> None:
    """Run the bottleneck profiler."""
    torch.manual_seed(42)

    console.rule("[bold cyan]Bottleneck Profiler — TinyLlama 1.1B")
    console.print(f"Model: {MODEL_NAME}")
    console.print(f"Prompt: \"{PROMPT}\"")
    console.print(f"Tokens: {NUM_TOKENS}")
    console.print()

    # ── 1. Load engine (includes warmup) ──────────────────────────────────
    console.print("[yellow]Loading model + MPS warmup...[/]")
    t_load = time.perf_counter()
    engine = LLMEngine.from_pretrained(MODEL_NAME)
    load_time = time.perf_counter() - t_load
    console.print(f"[green]Loaded in {load_time:.2f}s (includes warmup)[/]")
    console.print()

    # ── 2. Create timer and inject into all layers ────────────────────────
    timer = OperationTimer(enabled=True, sync_mps=True)
    for layer in engine.adapter.paged_layers:
        layer.timer = timer

    # ── 3. First run (cold after warmup — shaders compiled, but first real request) ──
    console.print("[yellow]Run 1: profiled generation (sync_mps=True)...[/]")
    params = SamplingParams(max_tokens=NUM_TOKENS, temperature=0.8)
    req_id = engine.add_request(PROMPT, params)

    t_cold = time.perf_counter()
    while not engine._outputs[req_id].finished:
        engine.step()
    cold_time = time.perf_counter() - t_cold

    console.print(f"[green]Run 1 complete: {cold_time:.2f}s[/]")
    console.print()

    # ── 4. Print operation breakdown ──────────────────────────────────────
    timer.print_report(title=f"PagedAttentionLayer Breakdown — {NUM_TOKENS} tokens × 22 layers")

    # ── 5. Second run (warm — reset timer) ────────────────────────────────
    timer.reset()
    for layer in engine.adapter.paged_layers:
        layer.timer = None  # Disable timer for clean measurement

    console.print()
    console.print("[yellow]Run 2: warm generation (no timer overhead)...[/]")
    req_id2 = engine.add_request(PROMPT, params)

    t_warm = time.perf_counter()
    while not engine._outputs[req_id2].finished:
        engine.step()
    warm_time = time.perf_counter() - t_warm

    console.print(f"[green]Run 2 complete: {warm_time:.2f}s[/]")

    # ── 6. Summary ────────────────────────────────────────────────────────
    console.print()
    summary = Table(title="Summary", show_lines=True)
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")

    summary.add_row("Model load + warmup", f"{load_time:.2f}s")
    summary.add_row("Run 1 (profiled, sync)", f"{cold_time:.2f}s")
    summary.add_row("Run 2 (warm, no timer)", f"{warm_time:.2f}s")
    summary.add_row("Timer overhead", f"{cold_time - warm_time:.2f}s")
    summary.add_row(
        "Warm tok/sec",
        f"{NUM_TOKENS / warm_time:.1f}",
    )

    if torch.backends.mps.is_available():
        mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
        summary.add_row("MPS memory", f"{mem_mb:.1f} MB")

    summary.add_row("KV pool", f"{engine.memory_pool.get_memory_mb():.2f} MB")

    console.print(summary)

    # Flag biggest bottleneck.
    report = timer.report() if timer._timings else {}
    if not report:
        console.print()
        console.print("[yellow]Timer was reset — re-enable to identify bottleneck.[/]")
    console.print()


if __name__ == "__main__":
    main()
