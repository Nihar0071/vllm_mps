"""Rotary Position Embedding (RoPE) compatible with HuggingFace Llama.

Precomputes cos/sin tables at init so forward() is just
element-wise multiply — no recomputation per step.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by splitting in half and negating."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embedding to *x* given precomputed cos/sin."""
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE tables.

    Args:
        d_k:         Head dimension.
        max_seq_len: Maximum sequence length to precompute for.
        theta:       Base frequency (default 10 000).
        device:      Device for the buffers.
    """

    def __init__(
        self,
        d_k: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k)
        )
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)            # (max_seq_len, d_k//2)
        emb = torch.cat([freqs, freqs], dim=-1)      # (max_seq_len, d_k)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE at a single position.

        Args:
            q: Shape ``(1, n_heads, 1, d_k)``.
            k: Shape ``(1, n_kv_heads, 1, d_k)``.
            position_id: Scalar position index.

        Returns:
            ``(q_rot, k_rot)`` with the same shapes.
        """
        cos = self.cos_cached[position_id].to(q.dtype)
        sin = self.sin_cached[position_id].to(q.dtype)
        # Broadcast: cos/sin are (d_k,) → works against (1, heads, 1, d_k)
        q_rot = apply_rotary(q, cos, sin)
        k_rot = apply_rotary(k, cos, sin)
        return q_rot, k_rot
