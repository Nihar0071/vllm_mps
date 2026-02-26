"""Naive attention with contiguous pre-allocated KV cache (baseline).

This implementation demonstrates the memory waste that PagedAttention
solves — full ``(max_batch, max_seq_len)`` tensors are reserved at init
regardless of actual sequence lengths.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

from vllm_mps.config import (
    D_MODEL,
    DEVICE,
    DTYPE,
    D_K,
    MAX_BATCH_SIZE,
    MAX_SEQ_LEN,
    N_HEADS,
)

logger = logging.getLogger(__name__)


class NaiveAttention(nn.Module):
    """Baseline multi-head attention with a contiguous KV cache.

    The entire cache is allocated at construction time and never grows
    or shrinks.  Use :meth:`get_cache_utilisation` to measure waste.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        d_k: int = D_K,
        max_seq_len: int = MAX_SEQ_LEN,
        max_batch_size: int = MAX_BATCH_SIZE,
        device: str = DEVICE,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.device = torch.device(device)
        self.dtype = dtype

        # Linear projections.
        self.W_Q = nn.Linear(d_model, d_model, bias=False, device=self.device, dtype=dtype)
        self.W_K = nn.Linear(d_model, d_model, bias=False, device=self.device, dtype=dtype)
        self.W_V = nn.Linear(d_model, d_model, bias=False, device=self.device, dtype=dtype)
        self.W_O = nn.Linear(d_model, d_model, bias=False, device=self.device, dtype=dtype)

        # Pre-allocated KV cache — THIS is the waste PagedAttention removes.
        self._k_cache = torch.zeros(
            (max_batch_size, max_seq_len, n_heads, d_k),
            device=self.device,
            dtype=dtype,
        )
        self._v_cache = torch.zeros(
            (max_batch_size, max_seq_len, n_heads, d_k),
            device=self.device,
            dtype=dtype,
        )

        logger.info(
            "NaiveAttention: cache %.2f MB (always allocated)",
            self.get_cache_memory_mb(),
        )

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        seq_idx: int,
        current_pos: int,
    ) -> torch.Tensor:
        """Run one attention step for a single token.

        Args:
            x:           Input tensor, shape ``(1, 1, d_model)``.
            seq_idx:     Batch slot index for this sequence.
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

        # 2. Store K, V in cache.
        self._k_cache[seq_idx, current_pos] = K[0, 0]
        self._v_cache[seq_idx, current_pos] = V[0, 0]

        # 3. Retrieve full history → (pos+1, n_heads, d_k).
        K_full = self._k_cache[seq_idx, : current_pos + 1]
        V_full = self._v_cache[seq_idx, : current_pos + 1]

        # 4. Attention: permute to (1, n_heads, seq_len, d_k).
        Q = Q.permute(0, 2, 1, 3)               # (1, n_heads, 1, d_k)
        K_full = K_full.unsqueeze(0).permute(0, 2, 1, 3)  # (1, n_heads, pos+1, d_k)
        V_full = V_full.unsqueeze(0).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K_full.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_full)         # (1, n_heads, 1, d_k)

        # 5. Merge heads → (1, 1, d_model) and project.
        out = out.permute(0, 2, 1, 3).contiguous().view(1, 1, self.d_model)
        return self.W_O(out)

    # ── Cache management ──────────────────────────────────────────────────

    def reset_sequence(self, seq_idx: int) -> None:
        """Zero out the cache slot for *seq_idx*."""
        self._k_cache[seq_idx].zero_()
        self._v_cache[seq_idx].zero_()

    def get_cache_memory_mb(self) -> float:
        """Return total cache memory in MB (always allocated)."""
        # This is ALWAYS allocated regardless of actual usage.
        total = self._k_cache.numel() + self._v_cache.numel()
        return total * self._k_cache.element_size() / (1024 * 1024)

    def get_cache_utilisation(self, active_seq_lens: list[int]) -> float:
        """Compute true cache utilisation given actual sequence lengths.

        Returns:
            Fraction of cache that actually holds useful data (0.0–1.0).
        """
        total_capacity = self.max_batch_size * self.max_seq_len
        if total_capacity == 0:
            return 0.0
        return sum(active_seq_lens) / total_capacity
