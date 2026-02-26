"""MPS (GPU) memory pool for KV-cache tensor storage.

Allocates a single contiguous tensor on the MPS device to hold KV data
for all physical blocks.  A PhysicalBlock with ``block_id=N`` stores its
key/value data at ``pool[N]``.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

logger = logging.getLogger(__name__)


class MPSMemoryPool:
    """Fixed-size tensor pool for KV-cache data on MPS (or CPU fallback).

    Pool tensor shape: ``(num_blocks, 2, block_size, n_heads, d_k)``

    * Dim 0 — block index (direct lookup via ``block_id``)
    * Dim 1 — 0 = K, 1 = V
    * Dims 2–4 — ``(token_pos, head, dim)``

    Attributes:
        num_blocks: Total physical blocks in the pool.
        block_size: Token slots per block.
        n_heads:    Number of attention heads.
        d_k:        Dimension per head.
        dtype:      Tensor data type.
        device:     Torch device the pool lives on.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        n_heads: int,
        d_k: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Allocate the pool tensor.

        Args:
            num_blocks: Number of physical blocks.
            block_size: Tokens per block.
            n_heads:    Attention heads.
            d_k:        Dimension per head.
            dtype:      Element type (e.g. ``torch.float16``).
            device:     Target device (e.g. ``torch.device("mps")``).
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_heads = n_heads
        self.d_k = d_k
        self.dtype = dtype
        self.device = device

        self._pool = torch.zeros(
            (num_blocks, 2, block_size, n_heads, d_k),
            dtype=dtype,
            device=device,
        )

        mem_mb = self.get_memory_mb()
        logger.info(
            "MPSMemoryPool: shape=%s, %.2f MB on %s",
            list(self._pool.shape),
            mem_mb,
            device,
        )
        if str(device) == "cpu":
            logger.warning(
                "MPSMemoryPool: MPS not available, using CPU fallback"
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

        Raises:
            IndexError: If indices are out of range.
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
                f"MPSMemoryPool: block_id {block_id} out of range "
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

        This is what the attention kernel calls to assemble non-contiguous
        blocks into a single contiguous KV sequence.

        Args:
            block_ids: Ordered list of physical block IDs.

        Returns:
            ``(k, v)`` each with shape
            ``(len(block_ids) * block_size, n_heads, d_k)``.
        """
        ids = torch.tensor(block_ids, dtype=torch.long)
        k = self._pool[ids, 0].reshape(-1, self.n_heads, self.d_k)
        v = self._pool[ids, 1].reshape(-1, self.n_heads, self.d_k)
        return k, v

    def gather_blocks_tensor(
        self, ids_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Like :meth:`gather_blocks` but takes a pre-built index tensor.

        This avoids a ``torch.tensor()`` allocation per call, which has
        ~0.5 ms overhead on MPS.

        Args:
            ids_tensor: 1-D ``torch.long`` tensor of physical block IDs.

        Returns:
            ``(k, v)`` each with shape
            ``(len(ids_tensor) * block_size, n_heads, d_k)``.
        """
        k = self._pool[ids_tensor, 0].reshape(-1, self.n_heads, self.d_k)
        v = self._pool[ids_tensor, 1].reshape(-1, self.n_heads, self.d_k)
        return k, v

    def copy_block(self, src_block_id: int, dst_block_id: int) -> None:
        """Copy all KV data from *src* to *dst* in-place.

        Used during swap_in / swap_out tensor data transfer.

        Raises:
            IndexError: If either block_id is out of range.
        """
        for bid, label in (
            (src_block_id, "src_block_id"),
            (dst_block_id, "dst_block_id"),
        ):
            if bid < 0 or bid >= self.num_blocks:
                raise IndexError(
                    f"MPSMemoryPool: {label} {bid} out of range "
                    f"(num_blocks={self.num_blocks})."
                )
        self._pool[dst_block_id] = self._pool[src_block_id]

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
            f"MPSMemoryPool(device={self.device}, "
            f"shape={list(self._pool.shape)}, "
            f"memory={self.get_memory_mb():.2f} MB)"
        )

    # ── Internal ──────────────────────────────────────────────────────────

    def _validate_indices(self, block_id: int, token_pos: int) -> None:
        """Raise IndexError if block_id or token_pos is out of range."""
        if block_id < 0 or block_id >= self.num_blocks:
            raise IndexError(
                f"MPSMemoryPool: block_id {block_id} out of range "
                f"(num_blocks={self.num_blocks})."
            )
        if token_pos < 0 or token_pos >= self.block_size:
            raise IndexError(
                f"MPSMemoryPool: token_pos {token_pos} out of range "
                f"(block_size={self.block_size})."
            )
