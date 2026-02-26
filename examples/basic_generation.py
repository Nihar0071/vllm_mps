"""Basic generation example with TinyLlama.

Loads TinyLlama-1.1B-Chat via the adapter system and generates
text for several prompts using our paged attention engine.
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
MAX_TOKENS = 50

PROMPTS = [
    "The meaning of life is",
    "In a galaxy far far away",
    "The best programming language is",
]


def main() -> None:
    """Run generation with TinyLlama."""
    torch.manual_seed(42)

    console.rule("[bold cyan]vLLM-MPS × TinyLlama Demo")
    console.print(f"Model: {MODEL_NAME}")
    console.print(f"Max tokens: {MAX_TOKENS}")
    console.print()

    # 1. Load engine.
    console.print("[yellow]Loading model...[/]")
    t0 = time.perf_counter()
    engine = LLMEngine.from_pretrained(MODEL_NAME)
    load_time = time.perf_counter() - t0
    console.print(f"[green]Loaded in {load_time:.1f}s[/]")

    if torch.backends.mps.is_available():
        mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
        console.print(f"MPS memory: {mem_mb:.1f} MB")

    console.print()

    # 2. Generate.
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.8, top_p=0.9)
    results_table = Table(title="Generation Results", show_lines=True)
    results_table.add_column("Prompt", style="cyan")
    results_table.add_column("Generated", style="green")
    results_table.add_column("Tokens", justify="right")
    results_table.add_column("Time", justify="right")

    total_tokens = 0
    total_time = 0.0

    for prompt in PROMPTS:
        console.print(f'[yellow]Generating for: "{prompt}"[/]')
        req_id = engine.add_request(prompt, params)

        t0 = time.perf_counter()
        while not engine._outputs[req_id].finished:
            engine.step()
        elapsed = time.perf_counter() - t0

        out = engine._outputs[req_id]
        n_tokens = len(out.token_ids)
        total_tokens += n_tokens
        total_time += elapsed

        results_table.add_row(
            prompt,
            out.generated_text[:80] + ("..." if len(out.generated_text) > 80 else ""),
            str(n_tokens),
            f"{elapsed:.2f}s",
        )

    # 3. Print results.
    console.print()
    console.print(results_table)

    console.print()
    console.rule("[bold green]Summary")
    console.print(f"  Total tokens:  {total_tokens}")
    console.print(f"  Total time:    {total_time:.2f}s")
    console.print(f"  Avg tok/sec:   {total_tokens / total_time:.1f}")
    console.print(f"  GPU pool:      {engine.memory_pool.get_memory_mb():.2f} MB")
    console.print(f"  GPU util:      {engine.kv_cache_manager.get_gpu_utilisation():.1%}")
    console.print()


if __name__ == "__main__":
    main()
