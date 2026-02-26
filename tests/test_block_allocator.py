"""Unit tests for vllm_mps.core.block_allocator.BlockAllocator."""

import unittest

from vllm_mps.core.block_allocator import BlockAllocator, BlockAllocatorType
from vllm_mps.config import NUM_GPU_BLOCKS, NUM_CPU_BLOCKS


class TestBlockAllocator(unittest.TestCase):
    """Tests for BlockAllocator allocation, freeing, and forking."""

    # Use small pools so tests are fast and deterministic.
    GPU_POOL = 4
    CPU_POOL = 4

    def _gpu_allocator(self, n: int | None = None) -> BlockAllocator:
        return BlockAllocator(BlockAllocatorType.GPU, n or self.GPU_POOL)

    def _cpu_allocator(self, n: int | None = None) -> BlockAllocator:
        return BlockAllocator(BlockAllocatorType.CPU, n or self.CPU_POOL)

    # ── 1. Initialisation ─────────────────────────────────────────────────

    def test_initialisation_gpu(self):
        """GPU allocator has correct total and all blocks free."""
        alloc = self._gpu_allocator()
        self.assertEqual(alloc.total_blocks, self.GPU_POOL)
        self.assertEqual(alloc.get_num_free_blocks(), self.GPU_POOL)
        self.assertEqual(alloc.get_num_used_blocks(), 0)

    def test_initialisation_cpu(self):
        """CPU allocator has correct total and all blocks free."""
        alloc = self._cpu_allocator()
        self.assertEqual(alloc.total_blocks, self.CPU_POOL)
        self.assertEqual(alloc.get_num_free_blocks(), self.CPU_POOL)
        self.assertEqual(alloc.allocator_type, BlockAllocatorType.CPU)

    # ── 2. Allocate ───────────────────────────────────────────────────────

    def test_allocate_returns_block(self):
        """Allocated block has ref_count == 1."""
        alloc = self._gpu_allocator()
        block = alloc.allocate()
        self.assertEqual(block.ref_count, 1)

    def test_allocate_reduces_free_count(self):
        """Free count decreases by 1 per allocate."""
        alloc = self._gpu_allocator()
        alloc.allocate()
        self.assertEqual(alloc.get_num_free_blocks(), self.GPU_POOL - 1)
        alloc.allocate()
        self.assertEqual(alloc.get_num_free_blocks(), self.GPU_POOL - 2)

    def test_allocate_out_of_memory(self):
        """MemoryError raised when all blocks allocated."""
        alloc = self._gpu_allocator(2)
        alloc.allocate()
        alloc.allocate()
        with self.assertRaises(MemoryError):
            alloc.allocate()

    # ── 3. Free ───────────────────────────────────────────────────────────

    def test_free_returns_block_to_pool(self):
        """After free, block is back in free list."""
        alloc = self._gpu_allocator()
        block = alloc.allocate()
        self.assertEqual(alloc.get_num_free_blocks(), self.GPU_POOL - 1)
        alloc.free(block)
        self.assertEqual(alloc.get_num_free_blocks(), self.GPU_POOL)
        self.assertEqual(block.ref_count, 0)
        self.assertEqual(block.num_tokens, 0)

    def test_free_shared_block(self):
        """Freeing a shared block (ref_count > 1) does NOT return it to free list."""
        alloc = self._gpu_allocator()
        block = alloc.allocate()          # ref=1
        alloc.fork(block)                 # ref=2
        alloc.free(block)                 # ref=1 — still in use
        self.assertEqual(block.ref_count, 1)
        self.assertEqual(alloc.get_num_free_blocks(), self.GPU_POOL - 1)

    # ── 4. Fork ───────────────────────────────────────────────────────────

    def test_fork_increments_ref(self):
        """Fork returns same block with ref_count == 2."""
        alloc = self._gpu_allocator()
        block = alloc.allocate()          # ref=1
        forked = alloc.fork(block)        # ref=2
        self.assertIs(forked, block)
        self.assertEqual(block.ref_count, 2)
        # No extra block consumed.
        self.assertEqual(alloc.get_num_free_blocks(), self.GPU_POOL - 1)

    def test_fork_then_free_both(self):
        """Fork, free once (ref=1), free again (ref=0, back to pool)."""
        alloc = self._gpu_allocator()
        block = alloc.allocate()          # ref=1
        alloc.fork(block)                 # ref=2
        alloc.free(block)                 # ref=1
        self.assertEqual(alloc.get_num_free_blocks(), self.GPU_POOL - 1)
        alloc.free(block)                 # ref=0 → reset → back to pool
        self.assertEqual(alloc.get_num_free_blocks(), self.GPU_POOL)
        self.assertTrue(block.is_free)

    # ── 5. Utilisation ────────────────────────────────────────────────────

    def test_utilisation(self):
        """Utilisation is 0.0 at start, 1.0 when all allocated."""
        alloc = self._gpu_allocator(4)
        self.assertAlmostEqual(alloc.get_utilisation(), 0.0)
        for _ in range(4):
            alloc.allocate()
        self.assertAlmostEqual(alloc.get_utilisation(), 1.0)

    # ── 6. Allocate after free ────────────────────────────────────────────

    def test_allocate_after_free(self):
        """Allocate, free, allocate again succeeds."""
        alloc = self._gpu_allocator(1)
        block = alloc.allocate()
        alloc.free(block)
        block2 = alloc.allocate()
        self.assertEqual(block2.ref_count, 1)
        self.assertEqual(alloc.get_num_free_blocks(), 0)


if __name__ == "__main__":
    unittest.main()
