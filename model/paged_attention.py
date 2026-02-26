"""Paged attention using block tables and the memory pool.

Unlike NaiveAttention, this implementation stores KV data in the shared
:class:`MPSMemoryPool` and locates it via block IDs from the
:class:`KVCacheManager`.  No pre-allocation — memory is consumed only
as tokens arrive.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

from vllm_mps.config import BLOCK_SIZE, D_MODEL, DEVICE, DTYPE, D_K, N_HEADS
from vllm_mps.core.kv_cache_manager import KVCacheManager
from vllm_mps.memory.mps_memory_pool import MPSMemoryPool

logger = logging.getLogger(__name__)


class PagedAttention(nn.Module):
    """Multi-head attention backed by a paged KV-cache memory pool.

    Memory is managed externally by the KVCacheManager and MPSMemoryPool.
    This module only holds the linear projection weights.
    """

    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        memory_pool: MPSMemoryPool,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        d_k: int = D_K,
        block_size: int = BLOCK_SIZE,
        device: str = DEVICE,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        self.kv_cache_manager = kv_cache_manager
        self.memory_pool = memory_pool
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.block_size = block_size
        self.device = torch.device(device)
        self.dtype = dtype

        # Linear projections — same as NaiveAttention.
        self.W_Q = nn.Linear(d_model, d_model, bias=False, device=self.device, dtype=dtype)
        self.W_K = nn.Linear(d_model, d_model, bias=False, device=self.device, dtype=dtype)
        self.W_V = nn.Linear(d_model, d_model, bias=False, device=self.device, dtype=dtype)
        self.W_O = nn.Linear(d_model, d_model, bias=False, device=self.device, dtype=dtype)

        # NO pre-allocated cache — memory lives in the pool.

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        seq_id: int,
        current_pos: int,
    ) -> torch.Tensor:
        """Run one attention step for a single token using paged KV cache.

        Args:
            x:           Input tensor, shape ``(1, 1, d_model)``.
            seq_id:      Sequence ID registered with the KVCacheManager.
            current_pos: Current token position in the sequence.

        Returns:
            Output tensor, shape ``(1, 1, d_model)``.
        """
        # 1. Project → (1, 1, d_model) each.
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Reshape → (1, 1, n_heads, d_k).
        Q = Q.view(1, 1, self.n_heads, self.d_k)
        K = K.view(1, 1, self.n_heads, self.d_k)
        V = V.view(1, 1, self.n_heads, self.d_k)

        # 2. Write K, V to memory pool.
        block_ids = self.kv_cache_manager.get_block_table(seq_id)
        block_idx = current_pos // self.block_size
        token_pos = current_pos % self.block_size
        physical_id = block_ids[block_idx]

        k_vec = K[0, 0]  # (n_heads, d_k)
        v_vec = V[0, 0]
        self.memory_pool.write_kv(physical_id, token_pos, k_vec, v_vec)

        # 3. Gather full KV history from pool.
        K_full, V_full = self.memory_pool.gather_blocks(block_ids)
        # raw shape: (num_blocks * block_size, n_heads, d_k)
        # Slice to only valid tokens.
        K_full = K_full[: current_pos + 1]
        V_full = V_full[: current_pos + 1]

        # 4. Reshape for attention.
        Q = Q.permute(0, 2, 1, 3)  # (1, n_heads, 1, d_k)
        K_full = K_full.unsqueeze(0).permute(0, 2, 1, 3)  # (1, n_heads, pos+1, d_k)
        V_full = V_full.unsqueeze(0).permute(0, 2, 1, 3)

        # 5. Scaled dot-product attention.
        scores = torch.matmul(Q, K_full.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_full)  # (1, n_heads, 1, d_k)

        # 6. Merge heads → (1, 1, d_model) and project.
        out = out.permute(0, 2, 1, 3).contiguous().view(1, 1, self.d_model)
        return self.W_O(out)

    # ── Introspection ─────────────────────────────────────────────────────

    def get_memory_overhead_bytes(self) -> int:
        """Return only the projection weight memory (no cache).

        Compare with NaiveAttention.get_cache_memory_mb() to see the
        difference — PagedAttention has zero cache overhead here because
        the cache lives in the shared memory pool.
        """
        total = 0
        for param in self.parameters():
            total += param.numel() * param.element_size()
        return total
