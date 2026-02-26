"""Per-sequence page table for KV-cache block management.

A BlockTable maps a sequence's logical token positions to physical blocks
managed by a BlockAllocator.  It is the per-sequence "page table" — each
sequence gets its own BlockTable instance that shares a common allocator.
"""

from __future__ import annotations

import logging

from vllm_mps.config import BLOCK_SIZE
from vllm_mps.core.block import PhysicalBlock
from vllm_mps.core.block_allocator import BlockAllocator

logger = logging.getLogger(__name__)


class BlockTable:
    """Page table that maps one sequence's tokens to physical KV-cache blocks.

    Blocks are allocated lazily — the first block is not requested from the
    allocator until the first :meth:`append_slot` call.

    Attributes:
        seq_id:     Identifier of the owning sequence.
        block_size: Tokens per block (from config).
    """

    def __init__(self, seq_id: int, allocator: BlockAllocator) -> None:
        """Create an empty block table for *seq_id*.

        Args:
            seq_id:    Unique sequence identifier.
            allocator: Shared block allocator (not owned by this table).
        """
        self.seq_id = seq_id
        self.block_size = BLOCK_SIZE
        self._allocator = allocator
        self._blocks: list[PhysicalBlock] = []
        self._num_tokens: int = 0

    # ── Core operations ───────────────────────────────────────────────────

    def append_slot(self) -> PhysicalBlock:
        """Append a single token slot and return the block it landed in.

        If the table is empty or the last block is full, a new block is
        allocated from the shared allocator.

        Returns:
            The PhysicalBlock that now holds the new token slot.

        Raises:
            MemoryError: Propagated from the allocator if no blocks remain.
        """
        if not self._blocks or self._blocks[-1].is_full:
            new_block = self._allocator.allocate()
            self._blocks.append(new_block)

        last = self._blocks[-1]
        last.add_tokens(1)
        self._num_tokens += 1
        return last

    def free_all(self) -> None:
        """Free every block back to the allocator and reset state."""
        n = len(self._blocks)
        for block in self._blocks:
            self._allocator.free(block)
        self._blocks.clear()
        self._num_tokens = 0
        logger.info(
            "BlockTable [seq=%d]: freed %d blocks", self.seq_id, n
        )

    # ── Lookups ───────────────────────────────────────────────────────────

    def get_block(self, logical_idx: int) -> PhysicalBlock:
        """Return the physical block at *logical_idx*.

        Args:
            logical_idx: Zero-based logical block index.

        Raises:
            IndexError: If *logical_idx* is out of range.
        """
        if logical_idx < 0 or logical_idx >= len(self._blocks):
            raise IndexError(
                f"BlockTable [seq={self.seq_id}]: logical index "
                f"{logical_idx} out of range (num_blocks={len(self._blocks)})."
            )
        return self._blocks[logical_idx]

    def get_all_physical_block_ids(self) -> list[int]:
        """Return ordered list of physical block IDs for attention kernels."""
        return [block.block_id for block in self._blocks]

    def get_last_block(self) -> PhysicalBlock:
        """Return the most recently allocated block.

        Raises:
            IndexError: If the table is empty.
        """
        if not self._blocks:
            raise IndexError(
                f"BlockTable [seq={self.seq_id}]: table is empty."
            )
        return self._blocks[-1]

    # ── Introspection ─────────────────────────────────────────────────────

    def num_blocks(self) -> int:
        """Return the number of logical blocks in the table."""
        return len(self._blocks)

    def num_tokens(self) -> int:
        """Return the total number of tokens stored."""
        return self._num_tokens

    def is_empty(self) -> bool:
        """Return True if no blocks have been allocated."""
        return len(self._blocks) == 0

    def __repr__(self) -> str:
        """Return a human-readable summary of the block table."""
        ids = self.get_all_physical_block_ids()
        return (
            f"BlockTable(seq_id={self.seq_id}, "
            f"num_blocks={len(self._blocks)}, "
            f"num_tokens={self._num_tokens}, "
            f"block_ids={ids})"
        )
