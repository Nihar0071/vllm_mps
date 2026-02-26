"""KV-cache manager — single entry point for all memory operations.

KVCacheManager is the ONLY component the rest of the system (scheduler,
model runner) talks to for memory operations.  Nothing outside ``core/``
should ever touch BlockAllocator or BlockTable directly.
"""

from __future__ import annotations

import logging
import math

from vllm_mps.config import BLOCK_SIZE
from vllm_mps.core.block import PhysicalBlock
from vllm_mps.core.block_allocator import BlockAllocator, BlockAllocatorType
from vllm_mps.core.block_table import BlockTable

logger = logging.getLogger(__name__)


class KVCacheManager:
    """Facade that coordinates GPU/CPU block allocators and per-sequence tables.

    Attributes:
        block_size: Tokens per physical block (from config).
    """

    def __init__(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """Create allocators and an empty sequence registry.

        Args:
            num_gpu_blocks: Size of the GPU (MPS) block pool.
            num_cpu_blocks: Size of the CPU fallback block pool.
        """
        self.block_size = BLOCK_SIZE
        self._gpu_allocator = BlockAllocator(
            BlockAllocatorType.GPU, num_gpu_blocks
        )
        self._cpu_allocator = BlockAllocator(
            BlockAllocatorType.CPU, num_cpu_blocks
        )
        self._block_tables: dict[int, BlockTable] = {}

        logger.info(
            "KVCacheManager: GPU=%d blocks, CPU=%d blocks, block_size=%d",
            num_gpu_blocks,
            num_cpu_blocks,
            self.block_size,
        )

    # ── Lookahead ─────────────────────────────────────────────────────────

    def can_allocate(self, seq_id: int, num_tokens: int) -> bool:
        """Check whether *num_tokens* can be appended without OOM.

        This is a **read-only** lookahead — no blocks are allocated.

        Args:
            seq_id:     Sequence to check (may or may not exist yet).
            num_tokens: Number of tokens to add.

        Returns:
            True if the GPU allocator has enough free blocks.
        """
        if seq_id in self._block_tables:
            table = self._block_tables[seq_id]
            empty_slots = (
                table.get_last_block().num_empty_slots
                if not table.is_empty()
                else 0
            )
            tokens_needing_new_blocks = max(0, num_tokens - empty_slots)
        else:
            tokens_needing_new_blocks = num_tokens

        blocks_needed = math.ceil(tokens_needing_new_blocks / self.block_size)
        return blocks_needed <= self._gpu_allocator.get_num_free_blocks()

    # ── Sequence lifecycle ────────────────────────────────────────────────

    def allocate(self, seq_id: int) -> None:
        """Register a new sequence with an empty block table.

        Args:
            seq_id: Unique sequence identifier.

        Raises:
            ValueError: If *seq_id* is already registered.
        """
        if seq_id in self._block_tables:
            raise ValueError(
                f"KVCacheManager: seq_id {seq_id} already exists."
            )
        self._block_tables[seq_id] = BlockTable(seq_id, self._gpu_allocator)
        logger.info("KVCacheManager: allocated table for seq=%d", seq_id)

    def append_slot(self, seq_id: int) -> PhysicalBlock:
        """Append one token slot to the sequence's block table.

        Args:
            seq_id: Target sequence identifier.

        Returns:
            The PhysicalBlock the token landed in.

        Raises:
            KeyError: If *seq_id* is not registered.
        """
        if seq_id not in self._block_tables:
            raise KeyError(
                f"KVCacheManager: seq_id {seq_id} not found."
            )
        return self._block_tables[seq_id].append_slot()

    def free(self, seq_id: int) -> None:
        """Free all blocks for *seq_id* and remove it from the registry.

        Args:
            seq_id: Sequence to free.

        Raises:
            KeyError: If *seq_id* is not registered.
        """
        if seq_id not in self._block_tables:
            raise KeyError(
                f"KVCacheManager: seq_id {seq_id} not found."
            )
        self._block_tables[seq_id].free_all()
        del self._block_tables[seq_id]
        logger.info("KVCacheManager: freed table for seq=%d", seq_id)

    # ── Copy-on-write fork ────────────────────────────────────────────────

    def fork(self, parent_seq_id: int, child_seq_id: int) -> None:
        """Fork *parent* into *child* using copy-on-write block sharing.

        Each block in the parent's table gets its ref_count incremented;
        the child receives references to the **same** physical blocks.

        Args:
            parent_seq_id: Existing sequence to fork from.
            child_seq_id:  New sequence to create.

        Raises:
            KeyError:   If *parent_seq_id* is not registered.
            ValueError: If *child_seq_id* already exists.
        """
        if parent_seq_id not in self._block_tables:
            raise KeyError(
                f"KVCacheManager: parent seq_id {parent_seq_id} not found."
            )
        if child_seq_id in self._block_tables:
            raise ValueError(
                f"KVCacheManager: child seq_id {child_seq_id} already exists."
            )

        parent_table = self._block_tables[parent_seq_id]
        child_table = BlockTable(child_seq_id, self._gpu_allocator)

        for block in parent_table._blocks:
            self._gpu_allocator.fork(block)
            child_table._blocks.append(block)

        child_table._num_tokens = parent_table._num_tokens
        self._block_tables[child_seq_id] = child_table

        logger.info(
            "KVCacheManager: forked seq=%d → seq=%d, %d blocks shared",
            parent_seq_id,
            child_seq_id,
            len(parent_table._blocks),
        )

    # ── Swap out / in (GPU ↔ CPU) ─────────────────────────────────────────

    def swap_out(self, seq_id: int) -> None:
        """Move a sequence's blocks from GPU to CPU (preemption).

        Only block **metadata** (num_tokens) is copied between descriptors.
        Actual tensor data copying is handled by the memory pool layer.

        Args:
            seq_id: Sequence to swap out.

        Raises:
            KeyError: If *seq_id* is not registered.
        """
        # TODO: actual tensor data copy will be done by the memory pool layer.
        if seq_id not in self._block_tables:
            raise KeyError(
                f"KVCacheManager: seq_id {seq_id} not found."
            )

        table = self._block_tables[seq_id]
        n = len(table._blocks)

        new_blocks: list[PhysicalBlock] = []
        for gpu_block in table._blocks:
            cpu_block = self._cpu_allocator.allocate()
            cpu_block.add_tokens(gpu_block.num_tokens)
            self._gpu_allocator.free(gpu_block)
            new_blocks.append(cpu_block)

        table._blocks = new_blocks
        table._allocator = self._cpu_allocator

        logger.info(
            "KVCacheManager: swapped out seq=%d, %d blocks to CPU",
            seq_id,
            n,
        )

    def swap_in(self, seq_id: int) -> None:
        """Move a sequence's blocks from CPU back to GPU.

        Only block **metadata** (num_tokens) is copied between descriptors.
        Actual tensor data copying is handled by the memory pool layer.

        Args:
            seq_id: Sequence to swap in.

        Raises:
            KeyError: If *seq_id* is not registered.
        """
        # TODO: actual tensor data copy will be done by the memory pool layer.
        if seq_id not in self._block_tables:
            raise KeyError(
                f"KVCacheManager: seq_id {seq_id} not found."
            )

        table = self._block_tables[seq_id]
        n = len(table._blocks)

        new_blocks: list[PhysicalBlock] = []
        for cpu_block in table._blocks:
            gpu_block = self._gpu_allocator.allocate()
            gpu_block.add_tokens(cpu_block.num_tokens)
            self._cpu_allocator.free(cpu_block)
            new_blocks.append(gpu_block)

        table._blocks = new_blocks
        table._allocator = self._gpu_allocator

        logger.info(
            "KVCacheManager: swapped in seq=%d, %d blocks to GPU",
            seq_id,
            n,
        )

    # ── Introspection ─────────────────────────────────────────────────────

    def get_block_table(self, seq_id: int) -> list[int]:
        """Return ordered physical block IDs for the attention kernel.

        Args:
            seq_id: Sequence to query.

        Raises:
            KeyError: If *seq_id* is not registered.
        """
        if seq_id not in self._block_tables:
            raise KeyError(
                f"KVCacheManager: seq_id {seq_id} not found."
            )
        return self._block_tables[seq_id].get_all_physical_block_ids()

    def get_num_free_gpu_blocks(self) -> int:
        """Return free GPU block count."""
        return self._gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        """Return free CPU block count."""
        return self._cpu_allocator.get_num_free_blocks()

    def get_gpu_utilisation(self) -> float:
        """Return GPU pool utilisation (0.0–1.0)."""
        return self._gpu_allocator.get_utilisation()

    def get_cpu_utilisation(self) -> float:
        """Return CPU pool utilisation (0.0–1.0)."""
        return self._cpu_allocator.get_utilisation()

    def get_all_seq_ids(self) -> list[int]:
        """Return list of all registered sequence IDs."""
        return list(self._block_tables.keys())

    def __repr__(self) -> str:
        """Return a human-readable summary."""
        return (
            f"KVCacheManager("
            f"seqs={len(self._block_tables)}, "
            f"GPU free={self.get_num_free_gpu_blocks()}/"
            f"{self._gpu_allocator.total_blocks}, "
            f"CPU free={self.get_num_free_cpu_blocks()}/"
            f"{self._cpu_allocator.total_blocks}, "
            f"GPU util={self.get_gpu_utilisation():.1%}, "
            f"CPU util={self.get_cpu_utilisation():.1%})"
        )
