"""Unit tests for vllm_mps.core.kv_cache_manager.KVCacheManager."""

import unittest

from vllm_mps.config import BLOCK_SIZE
from vllm_mps.core.kv_cache_manager import KVCacheManager


class TestKVCacheManager(unittest.TestCase):
    """Tests for KVCacheManager lifecycle, fork, swap, and introspection."""

    GPU = 16
    CPU = 8

    def _mgr(self) -> KVCacheManager:
        return KVCacheManager(num_gpu_blocks=self.GPU, num_cpu_blocks=self.CPU)

    # ── 1. Initial state ──────────────────────────────────────────────────

    def test_initial_state(self):
        """0 sequences, all GPU blocks free."""
        m = self._mgr()
        self.assertEqual(len(m.get_all_seq_ids()), 0)
        self.assertEqual(m.get_num_free_gpu_blocks(), self.GPU)
        self.assertEqual(m.get_num_free_cpu_blocks(), self.CPU)

    # ── 2. Allocate ───────────────────────────────────────────────────────

    def test_allocate_new_seq(self):
        """Allocate seq 0, appears in block tables."""
        m = self._mgr()
        m.allocate(0)
        self.assertIn(0, m.get_all_seq_ids())

    def test_allocate_duplicate_raises(self):
        """Allocating same seq_id twice raises ValueError."""
        m = self._mgr()
        m.allocate(0)
        with self.assertRaises(ValueError):
            m.allocate(0)

    # ── 3. Append ─────────────────────────────────────────────────────────

    def test_append_slot_basic(self):
        """Append 1 token, GPU free count decreases by 1."""
        m = self._mgr()
        m.allocate(0)
        m.append_slot(0)
        self.assertEqual(m.get_num_free_gpu_blocks(), self.GPU - 1)

    def test_append_multiple_tokens(self):
        """Append block_size + 1 tokens, 2 GPU blocks used."""
        m = self._mgr()
        m.allocate(0)
        for _ in range(BLOCK_SIZE + 1):
            m.append_slot(0)
        self.assertEqual(m.get_num_free_gpu_blocks(), self.GPU - 2)

    # ── 4. Free ───────────────────────────────────────────────────────────

    def test_free_seq(self):
        """After free, GPU blocks returned, seq removed."""
        m = self._mgr()
        m.allocate(0)
        for _ in range(BLOCK_SIZE * 2):
            m.append_slot(0)
        m.free(0)
        self.assertEqual(m.get_num_free_gpu_blocks(), self.GPU)
        self.assertNotIn(0, m.get_all_seq_ids())

    def test_free_unknown_seq_raises(self):
        """Free on unknown seq_id raises KeyError."""
        m = self._mgr()
        with self.assertRaises(KeyError):
            m.free(99)

    # ── 5. can_allocate ───────────────────────────────────────────────────

    def test_can_allocate_new_seq(self):
        """True when enough GPU blocks free."""
        m = self._mgr()
        self.assertTrue(m.can_allocate(0, BLOCK_SIZE))

    def test_can_allocate_false_when_full(self):
        """False when not enough GPU blocks."""
        m = self._mgr()
        # Need more blocks than available.
        need = (self.GPU + 1) * BLOCK_SIZE
        self.assertFalse(m.can_allocate(0, need))

    # ── 6. Fork ───────────────────────────────────────────────────────────

    def test_fork_shares_blocks(self):
        """Child has same block_ids as parent."""
        m = self._mgr()
        m.allocate(0)
        for _ in range(BLOCK_SIZE + 1):
            m.append_slot(0)
        m.fork(0, 1)
        self.assertEqual(m.get_block_table(0), m.get_block_table(1))

    def test_fork_ref_counts(self):
        """Each shared block has ref_count == 2."""
        m = self._mgr()
        m.allocate(0)
        for _ in range(BLOCK_SIZE):
            m.append_slot(0)
        m.fork(0, 1)
        # Access internal table to check ref counts.
        for block in m._block_tables[0]._blocks:
            self.assertEqual(block.ref_count, 2)

    def test_fork_then_free_parent(self):
        """After freeing parent, child blocks still intact."""
        m = self._mgr()
        m.allocate(0)
        for _ in range(BLOCK_SIZE):
            m.append_slot(0)
        m.fork(0, 1)
        child_ids = m.get_block_table(1)[:]
        m.free(0)
        # Child still has its blocks.
        self.assertEqual(m.get_block_table(1), child_ids)
        self.assertIn(1, m.get_all_seq_ids())

    # ── 7. Swap ───────────────────────────────────────────────────────────

    def test_swap_out(self):
        """After swap_out, GPU blocks freed, CPU blocks used."""
        m = self._mgr()
        m.allocate(0)
        for _ in range(BLOCK_SIZE * 2):
            m.append_slot(0)
        gpu_before = m.get_num_free_gpu_blocks()
        cpu_before = m.get_num_free_cpu_blocks()
        m.swap_out(0)
        # GPU blocks freed (2 returned), CPU blocks consumed (2 taken).
        self.assertEqual(m.get_num_free_gpu_blocks(), gpu_before + 2)
        self.assertEqual(m.get_num_free_cpu_blocks(), cpu_before - 2)

    def test_swap_in(self):
        """After swap_out then swap_in, seq back on GPU."""
        m = self._mgr()
        m.allocate(0)
        for _ in range(BLOCK_SIZE):
            m.append_slot(0)
        m.swap_out(0)
        gpu_after_out = m.get_num_free_gpu_blocks()
        cpu_after_out = m.get_num_free_cpu_blocks()
        m.swap_in(0)
        # GPU blocks consumed again, CPU blocks freed.
        self.assertEqual(m.get_num_free_gpu_blocks(), gpu_after_out - 1)
        self.assertEqual(m.get_num_free_cpu_blocks(), cpu_after_out + 1)

    # ── 8. get_block_table ────────────────────────────────────────────────

    def test_get_block_table(self):
        """Returns correct ordered block id list."""
        m = self._mgr()
        m.allocate(0)
        for _ in range(BLOCK_SIZE + 1):
            m.append_slot(0)
        ids = m.get_block_table(0)
        self.assertEqual(len(ids), 2)
        self.assertNotEqual(ids[0], ids[1])

    # ── 9. Full lifecycle ─────────────────────────────────────────────────

    def test_full_lifecycle(self):
        """allocate → append 20 → swap_out → swap_in → append 4 → free."""
        m = self._mgr()
        m.allocate(0)

        # Append 20 tokens (ceil(20/16)=2 blocks).
        for _ in range(20):
            m.append_slot(0)
        self.assertEqual(m.get_num_free_gpu_blocks(), self.GPU - 2)

        # Swap out to CPU.
        m.swap_out(0)
        self.assertEqual(m.get_num_free_gpu_blocks(), self.GPU)

        # Swap back in to GPU.
        m.swap_in(0)
        self.assertEqual(m.get_num_free_gpu_blocks(), self.GPU - 2)

        # Append 4 more tokens (still fits in 2nd block which had 4 of 16).
        for _ in range(4):
            m.append_slot(0)
        self.assertEqual(m.get_num_free_gpu_blocks(), self.GPU - 2)

        # Free everything.
        m.free(0)
        self.assertAlmostEqual(m.get_gpu_utilisation(), 0.0)
        self.assertAlmostEqual(m.get_cpu_utilisation(), 0.0)


if __name__ == "__main__":
    unittest.main()
