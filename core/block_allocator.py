"""Block allocator for KV-cache memory management.

Manages a pool of PhysicalBlock descriptors using a FIFO free list.
Supports allocation, freeing, and copy-on-write forking of blocks.
No tensors live here — only integer indices and metadata.
"""

from __future__ import annotations

import logging
from collections import deque
from enum import Enum

from vllm_mps.config import BLOCK_SIZE
from vllm_mps.core.block import PhysicalBlock

logger = logging.getLogger(__name__)


class BlockAllocatorType(Enum):
    """Type of memory pool backing this allocator."""

    GPU = "gpu"
    CPU = "cpu"


class BlockAllocator:
    """Manages a fixed-size pool of PhysicalBlock descriptors.

    Blocks are handed out via :meth:`allocate` and returned via :meth:`free`.
    Copy-on-write sharing is supported through :meth:`fork`, which increments
    the reference count without consuming a new block.

    Attributes:
        allocator_type: Whether this pool targets GPU (MPS) or CPU memory.
        total_blocks:   Total number of blocks in the pool.
        block_size:     Number of token slots per block (from config).
    """

    def __init__(
        self,
        allocator_type: BlockAllocatorType,
        total_blocks: int,
    ) -> None:
        """Initialise the allocator with *total_blocks* empty blocks.

        Args:
            allocator_type: GPU or CPU pool identifier.
            total_blocks:   Number of blocks to create.
        """
        self.allocator_type = allocator_type
        self.total_blocks = total_blocks
        self.block_size = BLOCK_SIZE

        self._all_blocks: dict[int, PhysicalBlock] = {
            i: PhysicalBlock(block_id=i) for i in range(total_blocks)
        }
        self._free_list: deque[int] = deque(range(total_blocks))

        logger.info(
            "BlockAllocator [%s] initialised with %d blocks of size %d",
            self.allocator_type.value,
            self.total_blocks,
            self.block_size,
        )

    # ── Core operations ───────────────────────────────────────────────────

    def allocate(self) -> PhysicalBlock:
        """Allocate and return a free block from the pool.

        The block's reference count is incremented to 1.

        Returns:
            The allocated PhysicalBlock descriptor.

        Raises:
            MemoryError: If no free blocks remain.
        """
        if not self._free_list:
            raise MemoryError(
                f"BlockAllocator [{self.allocator_type.value}]: "
                f"out of memory. 0 free blocks remaining."
            )

        block_id = self._free_list.popleft()
        block = self._all_blocks[block_id]
        block.increment_ref()
        return block

    def free(self, block: PhysicalBlock) -> None:
        """Release a reference to *block*.

        If the reference count drops to zero the block is reset and returned
        to the free list.  If other sequences still reference the block
        (copy-on-write), it remains allocated.

        Args:
            block: The block to release.
        """
        block.decrement_ref()

        if block.is_free:
            block.reset()
            self._free_list.append(block.block_id)

    def fork(self, block: PhysicalBlock) -> PhysicalBlock:
        """Create a copy-on-write reference to *block*.

        No new physical block is consumed; instead the existing block's
        reference count is incremented.

        Args:
            block: The block to share.

        Returns:
            The same PhysicalBlock with an incremented ref_count.

        Raises:
            ValueError: If *block* does not belong to this allocator.
        """
        if block.block_id not in self._all_blocks:
            raise ValueError(
                f"BlockAllocator [{self.allocator_type.value}]: "
                f"block {block.block_id} is not managed by this allocator."
            )
        block.increment_ref()
        return block

    # ── Introspection ─────────────────────────────────────────────────────

    def get_num_free_blocks(self) -> int:
        """Return the number of blocks currently available for allocation."""
        return len(self._free_list)

    def get_num_used_blocks(self) -> int:
        """Return the number of blocks currently in use."""
        return self.total_blocks - len(self._free_list)

    def get_utilisation(self) -> float:
        """Return pool utilisation as a float between 0.0 and 1.0."""
        return self.get_num_used_blocks() / self.total_blocks

    def get_all_free_block_ids(self) -> list[int]:
        """Return a snapshot of the free list for profiler introspection."""
        return list(self._free_list)

    def __repr__(self) -> str:
        """Return a human-readable summary of the allocator state."""
        return (
            f"BlockAllocator(type={self.allocator_type.value}, "
            f"total={self.total_blocks}, "
            f"free={self.get_num_free_blocks()}, "
            f"used={self.get_num_used_blocks()}, "
            f"utilisation={self.get_utilisation():.1%})"
        )
