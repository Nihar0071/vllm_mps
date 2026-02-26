"""Demo script — runs the full vllm_mps pipeline end to end.

Submits 4 prompts concurrently, runs the generation loop with a live
dashboard, and prints results.
"""

from __future__ import annotations

import time

import torch
from rich.console import Console
from rich.table import Table

from vllm_mps.config import DEVICE, DTYPE
from vllm_mps.core.sequence import SamplingParams
from vllm_mps.engine.llm_engine import LLMEngine
from vllm_mps.profiler.dashboard import LiveDashboard

console = Console()

PROMPTS = [
    "the quick brown fox",
    "once upon a time",
    "in a galaxy far far",
    "the meaning of life is",
]
MAX_TOKENS = 30


def main() -> None:
    """Run the full demo."""
    torch.manual_seed(42)

    console.rule("[bold cyan]vLLM-MPS Demo")
    console.print(f"Device: {DEVICE}  |  dtype: {DTYPE}")
    console.print(f"Prompts: {len(PROMPTS)}  |  max_tokens: {MAX_TOKENS}")
    console.print()

    # 1. Create engine.
    engine = LLMEngine()

    # 2. Start dashboard.
    dashboard = LiveDashboard()
    dashboard.start(engine)

    # 3. Submit all prompts.
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.8)
    request_ids = []
    for prompt in PROMPTS:
        rid = engine.add_request(prompt, params)
        request_ids.append(rid)
        console.print(f"  Submitted request {rid}: \"{prompt}\"")

    console.print()

    # 4. Run generation loop.
    start = time.perf_counter()
    step_count = 0
    while not engine.is_finished():
        engine.step()
        step_count += 1
    elapsed = time.perf_counter() - start

    # 5. Stop dashboard.
    time.sleep(0.6)  # Let dashboard render final state.
    dashboard.stop()

    # 6. Print final outputs.
    console.print()
    console.rule("[bold green]Generation Results")

    results_table = Table(title="Generated Text", show_lines=True)
    results_table.add_column("ID", style="bold", width=4)
    results_table.add_column("Prompt", style="cyan")
    results_table.add_column("Generated", style="green")
    results_table.add_column("Tokens", justify="right")

    for rid in request_ids:
        out = engine._outputs[rid]
        results_table.add_row(
            str(rid),
            out.prompt,
            out.generated_text[:60] + ("..." if len(out.generated_text) > 60 else ""),
            str(len(out.token_ids)),
        )

    console.print(results_table)

    # 7. Summary stats.
    console.print()
    console.rule("[bold yellow]Summary")
    total_tokens = engine.total_tokens_generated
    console.print(f"  Steps:         {step_count}")
    console.print(f"  Total tokens:  {total_tokens}")
    console.print(f"  Wall time:     {elapsed:.3f}s")
    console.print(f"  Avg tok/sec:   {total_tokens / elapsed:.1f}")
    console.print(f"  GPU util:      {engine.kv_cache_manager.get_gpu_utilisation():.1%}")
    console.print()


if __name__ == "__main__":
    main()
