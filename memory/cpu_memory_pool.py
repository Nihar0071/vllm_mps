"""CPU memory pool for KV-cache tensor storage (swap / fallback).

Identical interface to :class:`MPSMemoryPool` but always allocates on
CPU with ``float32`` dtype.  Supports ``pin_memory()`` for faster
CPU → GPU DMA transfers during swap-in.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

logger = logging.getLogger(__name__)


class CPUMemoryPool:
    """Fixed-size tensor pool for KV-cache data on CPU.

    Pool tensor shape: ``(num_blocks, 2, block_size, n_heads, d_k)``

    Always uses ``torch.float32`` and ``torch.device("cpu")``.

    Attributes:
        num_blocks: Total physical blocks in the pool.
        block_size: Token slots per block.
        n_heads:    Number of attention heads.
        d_k:        Dimension per head.
        dtype:      Always ``torch.float32``.
        device:     Always ``cpu``.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        n_heads: int,
        d_k: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Allocate the CPU pool tensor.

        *dtype* and *device* parameters are accepted for interface
        compatibility but ignored — the pool always uses float32 on CPU.
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_heads = n_heads
        self.d_k = d_k
        self.dtype = torch.float32
        self.device = torch.device("cpu")

        self._pool = torch.zeros(
            (num_blocks, 2, block_size, n_heads, d_k),
            dtype=self.dtype,
            device=self.device,
        )

        logger.info(
            "CPUMemoryPool: shape=%s, %.2f MB on cpu",
            list(self._pool.shape),
            self.get_memory_mb(),
        )

    # ── Write / Read ──────────────────────────────────────────────────────

    def write_kv(
        self,
        block_id: int,
        token_pos: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write a single token's K and V vectors into the pool.

        Args:
            block_id:  Physical block index.
            token_pos: Token slot within the block.
            k:         Key tensor, shape ``(n_heads, d_k)``.
            v:         Value tensor, shape ``(n_heads, d_k)``.

        Raises:
            IndexError: If *block_id* or *token_pos* is out of range.
        """
        self._validate_indices(block_id, token_pos)
        self._pool[block_id, 0, token_pos] = k
        self._pool[block_id, 1, token_pos] = v

    def read_kv(
        self, block_id: int, token_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read a single token's K and V vectors.

        Returns:
            ``(k, v)`` each with shape ``(n_heads, d_k)``.
        """
        self._validate_indices(block_id, token_pos)
        k = self._pool[block_id, 0, token_pos]
        v = self._pool[block_id, 1, token_pos]
        return k, v

    def read_block_kv(
        self, block_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read all KV data for an entire block.

        Returns:
            ``(k, v)`` each with shape ``(block_size, n_heads, d_k)``.
        """
        if block_id < 0 or block_id >= self.num_blocks:
            raise IndexError(
                f"CPUMemoryPool: block_id {block_id} out of range "
                f"(num_blocks={self.num_blocks})."
            )
        k = self._pool[block_id, 0]
        v = self._pool[block_id, 1]
        return k, v

    # ── Bulk operations ───────────────────────────────────────────────────

    def gather_blocks(
        self, block_ids: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather KV data from multiple blocks into contiguous tensors.

        Returns:
            ``(k, v)`` each with shape
            ``(len(block_ids) * block_size, n_heads, d_k)``.
        """
        ids = torch.tensor(block_ids, dtype=torch.long)
        k = self._pool[ids, 0].reshape(-1, self.n_heads, self.d_k)
        v = self._pool[ids, 1].reshape(-1, self.n_heads, self.d_k)
        return k, v

    def copy_block(self, src_block_id: int, dst_block_id: int) -> None:
        """Copy all KV data from *src* to *dst* in-place.

        Raises:
            IndexError: If either block_id is out of range.
        """
        for bid, label in (
            (src_block_id, "src_block_id"),
            (dst_block_id, "dst_block_id"),
        ):
            if bid < 0 or bid >= self.num_blocks:
                raise IndexError(
                    f"CPUMemoryPool: {label} {bid} out of range "
                    f"(num_blocks={self.num_blocks})."
                )
        self._pool[dst_block_id] = self._pool[src_block_id]

    # ── CPU-specific ──────────────────────────────────────────────────────

    def pin_memory(self) -> None:
        """Pin the pool tensor to page-locked memory.

        Enables faster CPU → GPU DMA transfers during swap-in.
        """
        self._pool = self._pool.pin_memory()
        logger.info("CPUMemoryPool: pool pinned to page-locked memory")

    # ── Introspection ─────────────────────────────────────────────────────

    def get_memory_mb(self) -> float:
        """Return pool memory usage in megabytes."""
        return self._pool.numel() * self._pool.element_size() / (1024 * 1024)

    def utilisation_snapshot(self) -> dict:
        """Return a diagnostic snapshot of the pool."""
        return {
            "device": str(self.device),
            "shape": list(self._pool.shape),
            "memory_mb": self.get_memory_mb(),
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "dtype": str(self.dtype),
        }

    def __repr__(self) -> str:
        """Return a human-readable summary."""
        return (
            f"CPUMemoryPool(device={self.device}, "
            f"shape={list(self._pool.shape)}, "
            f"memory={self.get_memory_mb():.2f} MB)"
        )

    # ── Internal ──────────────────────────────────────────────────────────

    def _validate_indices(self, block_id: int, token_pos: int) -> None:
        """Raise IndexError if block_id or token_pos is out of range."""
        if block_id < 0 or block_id >= self.num_blocks:
            raise IndexError(
                f"CPUMemoryPool: block_id {block_id} out of range "
                f"(num_blocks={self.num_blocks})."
            )
        if token_pos < 0 or token_pos >= self.block_size:
            raise IndexError(
                f"CPUMemoryPool: token_pos {token_pos} out of range "
                f"(block_size={self.block_size})."
            )
