"""Unit tests for vllm_mps.core.block_table.BlockTable."""

import unittest

from vllm_mps.config import BLOCK_SIZE
from vllm_mps.core.block_allocator import BlockAllocator, BlockAllocatorType
from vllm_mps.core.block_table import BlockTable


class TestBlockTable(unittest.TestCase):
    """Tests for BlockTable allocation, lookup, and freeing."""

    POOL_SIZE = 10

    def _make_table(self, seq_id: int = 0) -> tuple[BlockTable, BlockAllocator]:
        alloc = BlockAllocator(BlockAllocatorType.GPU, self.POOL_SIZE)
        return BlockTable(seq_id, alloc), alloc

    # ── 1. Initial state ──────────────────────────────────────────────────

    def test_initial_state(self):
        """Empty table, 0 tokens, 0 blocks."""
        table, _ = self._make_table()
        self.assertTrue(table.is_empty())
        self.assertEqual(table.num_tokens(), 0)
        self.assertEqual(table.num_blocks(), 0)

    # ── 2. First append ───────────────────────────────────────────────────

    def test_first_append_allocates_block(self):
        """After 1 append, num_blocks == 1."""
        table, _ = self._make_table()
        table.append_slot()
        self.assertEqual(table.num_blocks(), 1)
        self.assertFalse(table.is_empty())

    # ── 3. Fill a block ───────────────────────────────────────────────────

    def test_append_fills_block(self):
        """After block_size appends, last block is_full."""
        table, _ = self._make_table()
        for _ in range(BLOCK_SIZE):
            table.append_slot()
        self.assertTrue(table.get_last_block().is_full)
        self.assertEqual(table.num_blocks(), 1)

    # ── 4. Spill to new block ────────────────────────────────────────────

    def test_append_spills_to_new_block(self):
        """After block_size + 1 appends, num_blocks == 2."""
        table, _ = self._make_table()
        for _ in range(BLOCK_SIZE + 1):
            table.append_slot()
        self.assertEqual(table.num_blocks(), 2)

    # ── 5. Token tracking ────────────────────────────────────────────────

    def test_num_tokens_tracks_correctly(self):
        """num_tokens matches number of appends."""
        table, _ = self._make_table()
        n = BLOCK_SIZE * 2 + 3
        for _ in range(n):
            table.append_slot()
        self.assertEqual(table.num_tokens(), n)

    # ── 6. get_block valid ────────────────────────────────────────────────

    def test_get_block_valid(self):
        """get_block(0) returns correct PhysicalBlock."""
        table, _ = self._make_table()
        returned = table.append_slot()
        self.assertIs(table.get_block(0), returned)

    # ── 7. get_block out of range ─────────────────────────────────────────

    def test_get_block_out_of_range(self):
        """get_block(99) raises IndexError."""
        table, _ = self._make_table()
        with self.assertRaises(IndexError):
            table.get_block(99)

    # ── 8. Physical block IDs ─────────────────────────────────────────────

    def test_get_all_physical_block_ids(self):
        """Returns correct list of block_ids."""
        table, _ = self._make_table()
        for _ in range(BLOCK_SIZE + 1):
            table.append_slot()
        ids = table.get_all_physical_block_ids()
        self.assertEqual(len(ids), 2)
        # IDs come from the allocator's pool — both should be valid ints.
        self.assertIsInstance(ids[0], int)
        self.assertIsInstance(ids[1], int)
        self.assertNotEqual(ids[0], ids[1])

    # ── 9. free_all returns blocks ────────────────────────────────────────

    def test_free_all_returns_blocks(self):
        """After free_all, allocator free count is restored."""
        table, alloc = self._make_table()
        for _ in range(BLOCK_SIZE * 3):
            table.append_slot()
        self.assertEqual(alloc.get_num_free_blocks(), self.POOL_SIZE - 3)
        table.free_all()
        self.assertEqual(alloc.get_num_free_blocks(), self.POOL_SIZE)

    # ── 10. free_all resets state ─────────────────────────────────────────

    def test_free_all_resets_state(self):
        """After free_all, table is empty and num_tokens == 0."""
        table, _ = self._make_table()
        for _ in range(5):
            table.append_slot()
        table.free_all()
        self.assertTrue(table.is_empty())
        self.assertEqual(table.num_tokens(), 0)
        self.assertEqual(table.num_blocks(), 0)

    # ── 11. Multiple sequences independent ────────────────────────────────

    def test_multiple_sequences_independent(self):
        """Two BlockTables on same allocator don't interfere."""
        alloc = BlockAllocator(BlockAllocatorType.GPU, self.POOL_SIZE)
        t1 = BlockTable(seq_id=0, allocator=alloc)
        t2 = BlockTable(seq_id=1, allocator=alloc)

        for _ in range(BLOCK_SIZE):
            t1.append_slot()
        for _ in range(BLOCK_SIZE + 1):
            t2.append_slot()

        self.assertEqual(t1.num_blocks(), 1)
        self.assertEqual(t2.num_blocks(), 2)
        # 3 blocks used total out of 10.
        self.assertEqual(alloc.get_num_free_blocks(), self.POOL_SIZE - 3)

        # Block IDs must be disjoint.
        ids1 = set(t1.get_all_physical_block_ids())
        ids2 = set(t2.get_all_physical_block_ids())
        self.assertTrue(ids1.isdisjoint(ids2))

    # ── 12. OOM propagates ────────────────────────────────────────────────

    def test_oom_propagates(self):
        """Allocate all blocks externally, then append_slot raises MemoryError."""
        alloc = BlockAllocator(BlockAllocatorType.GPU, 2)
        # Exhaust the pool externally.
        alloc.allocate()
        alloc.allocate()

        table = BlockTable(seq_id=0, allocator=alloc)
        with self.assertRaises(MemoryError):
            table.append_slot()


if __name__ == "__main__":
    unittest.main()
