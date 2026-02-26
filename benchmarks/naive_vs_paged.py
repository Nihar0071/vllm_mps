"""Benchmark: Naive Attention vs Paged Attention.

Runs 50 token generation steps for 4 concurrent sequences and compares
memory usage and throughput between the two approaches.
"""

from __future__ import annotations

import time

import torch
from rich.console import Console
from rich.table import Table

from vllm_mps.config import (
    BLOCK_SIZE,
    D_K,
    D_MODEL,
    DEVICE,
    DTYPE,
    MAX_BATCH_SIZE,
    MAX_SEQ_LEN,
    N_HEADS,
    NUM_CPU_BLOCKS,
    NUM_GPU_BLOCKS,
)
from vllm_mps.core.kv_cache_manager import KVCacheManager
from vllm_mps.memory.mps_memory_pool import MPSMemoryPool
from vllm_mps.model.attention import NaiveAttention
from vllm_mps.model.paged_attention import PagedAttention

# ── Config ────────────────────────────────────────────────────────────────────
NUM_SEQUENCES = 4
NUM_TOKENS = 50
SEED = 42

console = Console()


def _run_naive(device: torch.device, dtype: torch.dtype) -> dict:
    """Run the naive attention benchmark and return metrics."""
    torch.manual_seed(SEED)
    model = NaiveAttention(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_k=D_K,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        device=str(device),
        dtype=dtype,
    )
    model.eval()

    total_tokens = 0
    active_lens = [0] * NUM_SEQUENCES

    start = time.perf_counter()
    with torch.no_grad():
        for step in range(NUM_TOKENS):
            for seq_idx in range(NUM_SEQUENCES):
                x = torch.randn(1, 1, D_MODEL, device=device, dtype=dtype)
                model.forward(x, seq_idx=seq_idx, current_pos=step)
                active_lens[seq_idx] = step + 1
                total_tokens += 1
    elapsed = time.perf_counter() - start

    return {
        "cache_mb": model.get_cache_memory_mb(),
        "utilisation": model.get_cache_utilisation(active_lens),
        "tokens_per_sec": total_tokens / elapsed,
        "elapsed": elapsed,
    }


def _run_paged(device: torch.device, dtype: torch.dtype) -> dict:
    """Run the paged attention benchmark and return metrics."""
    torch.manual_seed(SEED)

    kv_mgr = KVCacheManager(
        num_gpu_blocks=NUM_GPU_BLOCKS,
        num_cpu_blocks=NUM_CPU_BLOCKS,
    )
    pool = MPSMemoryPool(
        num_blocks=NUM_GPU_BLOCKS,
        block_size=BLOCK_SIZE,
        n_heads=N_HEADS,
        d_k=D_K,
        dtype=dtype,
        device=device,
    )
    model = PagedAttention(
        kv_cache_manager=kv_mgr,
        memory_pool=pool,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_k=D_K,
        block_size=BLOCK_SIZE,
        device=str(device),
        dtype=dtype,
    )
    model.eval()

    # Allocate sequences and their initial prompt slots.
    for seq_id in range(NUM_SEQUENCES):
        kv_mgr.allocate(seq_id)

    total_tokens = 0
    start = time.perf_counter()
    with torch.no_grad():
        for step in range(NUM_TOKENS):
            for seq_id in range(NUM_SEQUENCES):
                kv_mgr.append_slot(seq_id)
                x = torch.randn(1, 1, D_MODEL, device=device, dtype=dtype)
                model.forward(x, seq_id=seq_id, current_pos=step)
                total_tokens += 1
    elapsed = time.perf_counter() - start

    # Compute paged memory stats.
    blocks_used = NUM_GPU_BLOCKS - kv_mgr.get_num_free_gpu_blocks()
    paged_memory_mb = (
        blocks_used * 2 * BLOCK_SIZE * N_HEADS * D_K
        * pool._pool.element_size()
        / (1024 * 1024)
    )
    total_slots = blocks_used * BLOCK_SIZE
    used_slots = NUM_SEQUENCES * NUM_TOKENS
    paged_util = used_slots / total_slots if total_slots > 0 else 0.0

    # Free sequences.
    for seq_id in range(NUM_SEQUENCES):
        kv_mgr.free(seq_id)

    return {
        "cache_mb": paged_memory_mb,
        "utilisation": paged_util,
        "tokens_per_sec": total_tokens / elapsed,
        "elapsed": elapsed,
        "pool_total_mb": pool.get_memory_mb(),
    }


def main() -> None:
    """Run benchmarks and print comparison table."""
    device = torch.device(DEVICE)
    dtype = DTYPE

    console.rule("[bold cyan]Naive vs Paged Attention Benchmark")
    console.print(f"Device: {device}  |  dtype: {dtype}")
    console.print(f"Sequences: {NUM_SEQUENCES}  |  Tokens: {NUM_TOKENS}")
    console.print(f"d_model={D_MODEL}  n_heads={N_HEADS}  d_k={D_K}")
    console.print(f"block_size={BLOCK_SIZE}  max_seq_len={MAX_SEQ_LEN}")
    console.print()

    console.print("[yellow]Running Naive Attention...[/]")
    naive = _run_naive(device, dtype)

    console.print("[yellow]Running Paged Attention...[/]")
    paged = _run_paged(device, dtype)

    # ── Results table ─────────────────────────────────────────────────────
    table = Table(title="Benchmark Results", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Naive", justify="right")
    table.add_column("Paged", justify="right")
    table.add_column("Δ", justify="right", style="green")

    table.add_row(
        "Cache Memory (MB)",
        f"{naive['cache_mb']:.4f}",
        f"{paged['cache_mb']:.4f}",
        f"{naive['cache_mb'] / max(paged['cache_mb'], 1e-9):.1f}× more (naive)",
    )
    table.add_row(
        "Cache Utilisation",
        f"{naive['utilisation']:.1%}",
        f"{paged['utilisation']:.1%}",
        f"paged {paged['utilisation'] / max(naive['utilisation'], 1e-9):.1f}× better",
    )
    table.add_row(
        "Tokens / sec",
        f"{naive['tokens_per_sec']:.1f}",
        f"{paged['tokens_per_sec']:.1f}",
        "",
    )
    table.add_row(
        "Wall time (s)",
        f"{naive['elapsed']:.3f}",
        f"{paged['elapsed']:.3f}",
        "",
    )

    console.print()
    console.print(table)

    # ── Summary ───────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold green]Key Insight")
    console.print(
        f"[bold]Naive[/] pre-allocates [red]{naive['cache_mb']:.4f} MB[/] "
        f"but only uses [red]{naive['utilisation']:.1%}[/] of it."
    )
    console.print(
        f"[bold]Paged[/] uses only [green]{paged['cache_mb']:.4f} MB[/] "
        f"with [green]{paged['utilisation']:.1%}[/] utilisation."
    )


if __name__ == "__main__":
    main()
