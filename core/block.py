"""Physical block descriptor for KV-cache memory management.

A PhysicalBlock is the atomic unit of KV cache memory. It is a *descriptor*
of a memory slot — it contains no tensors. The actual tensor data lives in
the memory pool (implemented separately).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm_mps.config import BLOCK_SIZE


@dataclass
class PhysicalBlock:
    """Descriptor for a single physical block in the KV-cache memory pool.

    Attributes:
        block_id:   Index into the physical memory pool.
        ref_count:  How many sequences currently point to this block.
        num_tokens: How many token slots are currently filled.
        block_size: Maximum number of tokens this block can hold.
    """

    block_id: int
    ref_count: int = 0
    num_tokens: int = 0
    block_size: int = field(default=BLOCK_SIZE)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_full(self) -> bool:
        """Return True when all token slots in the block are filled."""
        return self.num_tokens == self.block_size

    @property
    def is_free(self) -> bool:
        """Return True when no sequence references this block."""
        return self.ref_count == 0

    @property
    def num_empty_slots(self) -> int:
        """Return the number of unfilled token slots."""
        return self.block_size - self.num_tokens

    # ── Methods ───────────────────────────────────────────────────────────

    def increment_ref(self) -> None:
        """Increment the reference count by one."""
        self.ref_count += 1

    def decrement_ref(self) -> None:
        """Decrement the reference count by one.

        Raises:
            ValueError: If the reference count would drop below zero.
        """
        if self.ref_count <= 0:
            raise ValueError(
                f"Cannot decrement ref_count of block {self.block_id}: "
                f"ref_count is already {self.ref_count}."
            )
        self.ref_count -= 1

    def add_tokens(self, n: int) -> None:
        """Mark *n* additional token slots as filled.

        Args:
            n: Number of tokens to add.

        Raises:
            ValueError: If adding *n* tokens would exceed the block size.
        """
        if self.num_tokens + n > self.block_size:
            raise ValueError(
                f"Cannot add {n} tokens to block {self.block_id}: "
                f"would have {self.num_tokens + n} tokens but block_size "
                f"is {self.block_size}."
            )
        self.num_tokens += n

    def reset(self) -> None:
        """Reset the block to its initial empty state.

        Used when the block is returned to the free list.
        """
        self.num_tokens = 0
        self.ref_count = 0

    def __repr__(self) -> str:
        """Return a human-readable representation of the block."""
        return (
            f"PhysicalBlock(block_id={self.block_id}, "
            f"ref_count={self.ref_count}, "
            f"num_tokens={self.num_tokens}, "
            f"block_size={self.block_size})"
        )
