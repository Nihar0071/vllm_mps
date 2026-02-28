"""Metal compute kernels for vllm_mps.

Contains:
  - gather_kv_metal()          — Fix 5A: gather-only kernel (kept as reference)
  - fused_attention_metal()    — Fix 5B: fused write+gather+attention kernel

Fix 5B replaces 2×B+1 MPS dispatches per layer with 1 dispatch:
  Phase 1: Write new K,V to pool (all 64 threads)
  Phase 2: Compute Q·K scores (thread 0 serial loop)
  Phase 3: Softmax (thread 0, float32 accumulation)
  Phase 4: Weighted V sum (all 64 threads parallelized)

Usage:
    from vllm_mps.kernels import fused_attention_metal
"""

from __future__ import annotations

import torch

# ═══════════════════════════════════════════════════════════════════════════════
# Fix 5A: Gather-only kernel (kept as reference / future use at large BS)
# ═══════════════════════════════════════════════════════════════════════════════

GATHER_KV_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void gather_kv(
    device half*          out_k        [[buffer(0)]],
    device half*          out_v        [[buffer(1)]],
    device const half*    pool         [[buffer(2)]],
    device const int*     block_tables [[buffer(3)]],
    device const int*     seq_lens     [[buffer(4)]],
    device const int*     params       [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    const int block_size   = params[0];
    const int n_kv_heads   = params[1];
    const int d_k          = params[2];
    const int max_seq_len  = params[3];
    const int max_blocks   = params[4];
    const int pool_stride0 = params[5];
    const int pool_stride1 = params[6];

    const int hd_size = n_kv_heads * d_k;
    const int seq_stride = max_seq_len * hd_size;
    const int seq_idx = (int)tid / seq_stride;
    const int remainder1 = (int)tid % seq_stride;
    const int tok_pos = remainder1 / hd_size;
    const int hd_idx  = remainder1 % hd_size;
    const int head    = hd_idx / d_k;
    const int dim     = hd_idx % d_k;

    const int slen = seq_lens[seq_idx];
    if (tok_pos >= slen) {
        out_k[tid] = half(0.0);
        out_v[tid] = half(0.0);
        return;
    }

    const int block_idx = tok_pos / block_size;
    const int token_in_block = tok_pos % block_size;
    const int physical_block = block_tables[seq_idx * max_blocks + block_idx];
    const int token_stride = n_kv_heads * d_k;
    const int base = physical_block * pool_stride0
                   + token_in_block * token_stride
                   + head * d_k + dim;

    out_k[tid] = pool[base];
    out_v[tid] = pool[base + pool_stride1];
}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Fix 5B: Fused write+gather+attention kernel
# ═══════════════════════════════════════════════════════════════════════════════

FUSED_ATTENTION_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// Thread grid: [B, n_heads, d_k]   group_size: [1, 1, d_k]
// Each threadgroup = one (sequence, query_head) pair
// 64 threads per group, one per dimension
//
// Buffers:
//   pool           [flat]  half  — KV cache, read+write
//   new_k          [B, n_kv_heads, d_k]  half
//   new_v          [B, n_kv_heads, d_k]  half
//   queries        [B, n_heads, d_k]     half
//   block_tables   [B * max_blocks]      int32
//   seq_lens       [B]                   int32  (AFTER new token)
//   write_info     [B * 2]               int32  (block_id, token_pos) per seq
//   output         [B * n_heads * d_k]   half
//   params         [6]: n_heads, n_kv_heads, d_k, block_size, max_blocks,
//                       pool_stride0, pool_stride1

kernel void fused_attention(
    device half*          pool         [[buffer(0)]],
    device const half*    new_k        [[buffer(1)]],
    device const half*    new_v        [[buffer(2)]],
    device const half*    queries      [[buffer(3)]],
    device const int*     block_tables [[buffer(4)]],
    device const int*     seq_lens     [[buffer(5)]],
    device const int*     write_info   [[buffer(6)]],
    device half*          output       [[buffer(7)]],
    device const int*     params       [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    // Unpack params
    const int n_heads      = params[0];
    const int n_kv_heads   = params[1];
    const int d_k          = params[2];
    const int block_size   = params[3];
    const int max_blocks   = params[4];
    const int pool_stride0 = params[5];  // stride for block dim
    const int pool_stride1 = params[6];  // stride for kv dim (K=0, V=1)

    const uint seq_idx   = gid.x;    // which sequence
    const uint head_idx  = gid.y;    // which query head (0..31)
    const uint dim       = lid.z;    // which dimension (0..63)

    // GQA mapping: which KV head serves this query head
    const uint heads_per_kv = n_heads / n_kv_heads;
    const uint kv_head = head_idx / heads_per_kv;

    const int seq_len = seq_lens[seq_idx];

    // ── Per-threadgroup shared memory ────────────────────────────────────
    // MAX_SEQ_LEN=512 tokens × 4 bytes = 2048 bytes — within 32KB limit
    threadgroup float scores[512];

    // ── Phase 1: Write new K,V to pool ──────────────────────────────────
    // Every threadgroup writes to the pool — writes are idempotent
    // (same data to same location). This avoids cross-threadgroup sync issues
    // since threadgroup_barrier only syncs within one threadgroup.
    {
        const int w_block    = write_info[seq_idx * 2];
        const int w_tok_pos  = write_info[seq_idx * 2 + 1];

        const int kv_stride = n_kv_heads * d_k;
        const int k_offset = w_block * pool_stride0
                           + 0 * pool_stride1
                           + w_tok_pos * kv_stride
                           + kv_head * d_k
                           + dim;
        const int v_offset = w_block * pool_stride0
                           + 1 * pool_stride1
                           + w_tok_pos * kv_stride
                           + kv_head * d_k
                           + dim;

        const int new_kv_offset = seq_idx * (n_kv_heads * d_k)
                                + kv_head * d_k
                                + dim;

        pool[k_offset] = new_k[new_kv_offset];
        pool[v_offset] = new_v[new_kv_offset];
    }

    // Ensure ALL threads across ALL threadgroups see the writes
    threadgroup_barrier(mem_flags::mem_device);

    // ── Phase 2: Compute Q·K scores for all history tokens ──────────────
    // Thread 0 does the serial dot-product loop
    // (seq_len ≤ ~32 for 20-token decode — fast enough)
    if (dim == 0) {
        // Load query vector into registers
        // queries layout: [B, n_heads, d_k]
        const int q_base = seq_idx * (n_heads * d_k) + head_idx * d_k;

        const int kv_stride = n_kv_heads * d_k;
        const float scale = 1.0f / sqrt(float(d_k));

        for (int t = 0; t < seq_len; t++) {
            // Which block and position for token t
            const int blk_idx = t / block_size;
            const int tok_in_blk = t % block_size;
            const int phys_block = block_tables[seq_idx * max_blocks + blk_idx];

            // K offset in pool
            const int k_base = phys_block * pool_stride0
                             + 0 * pool_stride1
                             + tok_in_blk * kv_stride
                             + kv_head * d_k;

            float dot = 0.0f;
            for (int dd = 0; dd < d_k; dd++) {
                dot += float(queries[q_base + dd]) * float(pool[k_base + dd]);
            }
            scores[t] = dot * scale;
        }

        // ── Phase 3: Numerically stable softmax ─────────────────────────
        // Pass 1: find max
        float max_val = scores[0];
        for (int t = 1; t < seq_len; t++) {
            max_val = max(max_val, scores[t]);
        }

        // Pass 2: exp and sum
        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = exp(scores[t] - max_val);
            sum_exp += scores[t];
        }

        // Pass 3: normalize
        const float inv_sum = 1.0f / sum_exp;
        for (int t = 0; t < seq_len; t++) {
            scores[t] *= inv_sum;
        }
    }

    // All threads need to see the final scores
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Weighted sum of V values ───────────────────────────────
    // Each thread handles one dimension
    const int kv_stride = n_kv_heads * d_k;
    float result = 0.0f;

    for (int t = 0; t < seq_len; t++) {
        const int blk_idx = t / block_size;
        const int tok_in_blk = t % block_size;
        const int phys_block = block_tables[seq_idx * max_blocks + blk_idx];

        const int v_offset = phys_block * pool_stride0
                           + 1 * pool_stride1
                           + tok_in_blk * kv_stride
                           + kv_head * d_k
                           + dim;

        result += scores[t] * float(pool[v_offset]);
    }

    // Write output: [B, n_heads, d_k] (flattened)
    const int out_offset = seq_idx * (n_heads * d_k) + head_idx * d_k + dim;
    output[out_offset] = half(result);
}
"""

# ── Compiled shader caches ────────────────────────────────────────────────────

_gather_lib = None
_fused_lib = None
_cached_params: dict[tuple, torch.Tensor] = {}


def _get_gather_lib():
    global _gather_lib
    if _gather_lib is None:
        _gather_lib = torch.mps.compile_shader(GATHER_KV_SOURCE)
    return _gather_lib


def _get_fused_lib():
    global _fused_lib
    if _fused_lib is None:
        _fused_lib = torch.mps.compile_shader(FUSED_ATTENTION_SOURCE)
    return _fused_lib


def _get_params(key: tuple, device: torch.device) -> torch.Tensor:
    if key not in _cached_params:
        _cached_params[key] = torch.tensor(
            list(key), dtype=torch.int32, device=device,
        )
    return _cached_params[key]


# ── Fix 5A: Gather-only entry point ──────────────────────────────────────────

_pool_flat_cache: dict[int, torch.Tensor] = {}


def gather_kv_metal(
    pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused batched KV gather (Fix 5A). See module docstring."""
    B = block_tables.shape[0]
    max_blocks = block_tables.shape[1]
    _, _, block_size, n_kv_heads, d_k = pool.shape
    device = pool.device

    total_elems = B * max_seq_len * n_kv_heads * d_k
    out_k = torch.zeros(total_elems, dtype=torch.float16, device=device)
    out_v = torch.zeros(total_elems, dtype=torch.float16, device=device)

    params = _get_params(
        (block_size, n_kv_heads, d_k, max_seq_len, max_blocks,
         pool.stride(0), pool.stride(1)),
        device,
    )

    pool_id = id(pool)
    if pool_id not in _pool_flat_cache or _pool_flat_cache[pool_id].data_ptr() != pool.data_ptr():
        _pool_flat_cache[pool_id] = pool.contiguous().view(-1)
    pool_flat = _pool_flat_cache[pool_id]

    lib = _get_gather_lib()
    lib.gather_kv(out_k, out_v, pool_flat,
                  block_tables.contiguous().view(-1),
                  seq_lens.contiguous(), params)

    return (out_k.view(B, max_seq_len, n_kv_heads, d_k),
            out_v.view(B, max_seq_len, n_kv_heads, d_k))


# ── Fix 5B: Fused attention entry points ─────────────────────────────────────

_out_buf: torch.Tensor | None = None


def fused_attention_metal_fast(
    pool: torch.Tensor,           # [num_blocks, 2, block_size, n_kv_heads, d_k]
    new_k: torch.Tensor,          # [B, n_kv_heads, d_k]
    new_v: torch.Tensor,          # [B, n_kv_heads, d_k]
    queries: torch.Tensor,        # [B, n_heads, d_k]
    bt_tensor: torch.Tensor,      # [B, max_blocks] int32 MPS (pre-built)
    sl_tensor: torch.Tensor,      # [B] int32 MPS (pre-built)
    write_info: torch.Tensor,     # [B, 2] int32 MPS (pre-built)
    n_heads: int,
    n_kv_heads: int,
    d_k: int,
    block_size: int,
) -> torch.Tensor:
    """Fast path: pre-built MPS tensors, minimal per-call allocation."""
    global _out_buf

    B = bt_tensor.shape[0]
    max_blocks = bt_tensor.shape[1]
    device = pool.device
    total = B * n_heads * d_k

    # Re-use output buffer if large enough
    if _out_buf is None or _out_buf.numel() < total:
        _out_buf = torch.empty(total, dtype=torch.float16, device=device)
    output = _out_buf[:total]
    output.zero_()

    params = _get_params(
        (n_heads, n_kv_heads, d_k, block_size, max_blocks,
         pool.stride(0), pool.stride(1)),
        device,
    )

    lib = _get_fused_lib()
    lib.fused_attention(
        pool.view(-1),
        new_k.contiguous().view(-1),
        new_v.contiguous().view(-1),
        queries.contiguous().view(-1),
        bt_tensor.view(-1),
        sl_tensor,
        write_info.view(-1),
        output,
        params,
        threads=[B, n_heads, d_k],
        group_size=[1, 1, d_k],
    )

    return output.view(B, n_heads, d_k)


def fused_attention_metal(
    pool: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    queries: torch.Tensor,
    block_tables: list[list[int]],
    seq_lens: list[int],
    write_blocks: list[int],
    write_positions: list[int],
    n_heads: int,
    n_kv_heads: int,
    d_k: int,
    block_size: int,
) -> torch.Tensor:
    """Convenience wrapper — builds MPS tensors from Python lists."""
    B = len(seq_lens)
    max_blocks = max(len(bt) for bt in block_tables)
    device = pool.device

    bt_tensor = torch.zeros(B, max_blocks, dtype=torch.int32, device=device)
    for i, bt in enumerate(block_tables):
        bt_tensor[i, :len(bt)] = torch.tensor(bt, dtype=torch.int32)

    write_info = torch.zeros(B, 2, dtype=torch.int32, device=device)
    for i in range(B):
        write_info[i, 0] = write_blocks[i]
        write_info[i, 1] = write_positions[i]

    sl_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    return fused_attention_metal_fast(
        pool, new_k, new_v, queries,
        bt_tensor, sl_tensor, write_info,
        n_heads, n_kv_heads, d_k, block_size,
    )

