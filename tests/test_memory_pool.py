"""Unit tests for vllm_mps.memory.mps_memory_pool and cpu_memory_pool."""

import unittest

import torch

from vllm_mps.config import DEVICE
from vllm_mps.memory.mps_memory_pool import MPSMemoryPool
from vllm_mps.memory.cpu_memory_pool import CPUMemoryPool

# Small dimensions for fast, deterministic tests.
NUM_BLOCKS = 8
BLOCK_SIZE = 4
N_HEADS = 2
D_K = 8


class TestMPSMemoryPool(unittest.TestCase):
    """Tests for MPSMemoryPool tensor operations."""

    def _pool(self) -> MPSMemoryPool:
        return MPSMemoryPool(
            NUM_BLOCKS, BLOCK_SIZE, N_HEADS, D_K,
            dtype=torch.float32,
            device=torch.device(DEVICE),
        )

    # ── 1. Shape ──────────────────────────────────────────────────────────

    def test_pool_shape(self):
        """Pool tensor has correct shape."""
        pool = self._pool()
        self.assertEqual(
            list(pool._pool.shape),
            [NUM_BLOCKS, 2, BLOCK_SIZE, N_HEADS, D_K],
        )

    # ── 2. Device ─────────────────────────────────────────────────────────

    def test_pool_device(self):
        """Pool is on MPS (or CPU if MPS unavailable)."""
        pool = self._pool()
        self.assertEqual(pool._pool.device.type, DEVICE.split(':')[0])

    # ── 3. Write / Read ──────────────────────────────────────────────────

    def test_write_read_kv(self):
        """Write k,v at position, read back matches."""
        pool = self._pool()
        k = torch.randn(N_HEADS, D_K, device=torch.device(DEVICE))
        v = torch.randn(N_HEADS, D_K, device=torch.device(DEVICE))
        pool.write_kv(0, 0, k, v)
        k_out, v_out = pool.read_kv(0, 0)
        self.assertTrue(torch.allclose(k_out, k))
        self.assertTrue(torch.allclose(v_out, v))

    # ── 4. Multiple positions ─────────────────────────────────────────────

    def test_write_read_multiple_positions(self):
        """Write to all token positions, read all back."""
        pool = self._pool()
        device = torch.device(DEVICE)
        ks = [torch.randn(N_HEADS, D_K, device=device) for _ in range(BLOCK_SIZE)]
        vs = [torch.randn(N_HEADS, D_K, device=device) for _ in range(BLOCK_SIZE)]
        for pos in range(BLOCK_SIZE):
            pool.write_kv(0, pos, ks[pos], vs[pos])
        for pos in range(BLOCK_SIZE):
            k_out, v_out = pool.read_kv(0, pos)
            self.assertTrue(torch.allclose(k_out, ks[pos]))
            self.assertTrue(torch.allclose(v_out, vs[pos]))

    # ── 5. read_block_kv shape ────────────────────────────────────────────

    def test_read_block_kv_shape(self):
        """read_block_kv returns correct shape."""
        pool = self._pool()
        k, v = pool.read_block_kv(0)
        self.assertEqual(list(k.shape), [BLOCK_SIZE, N_HEADS, D_K])
        self.assertEqual(list(v.shape), [BLOCK_SIZE, N_HEADS, D_K])

    # ── 6. gather_blocks shape ────────────────────────────────────────────

    def test_gather_blocks_shape(self):
        """Gather 3 blocks → shape (3*block_size, n_heads, d_k)."""
        pool = self._pool()
        k, v = pool.gather_blocks([0, 1, 2])
        expected = 3 * BLOCK_SIZE
        self.assertEqual(list(k.shape), [expected, N_HEADS, D_K])
        self.assertEqual(list(v.shape), [expected, N_HEADS, D_K])

    # ── 7. gather_blocks content ──────────────────────────────────────────

    def test_gather_blocks_content(self):
        """Gathered content matches individual reads."""
        pool = self._pool()
        device = torch.device(DEVICE)
        # Write known data into blocks 0 and 1.
        for bid in (0, 1):
            for pos in range(BLOCK_SIZE):
                k = torch.full((N_HEADS, D_K), float(bid * BLOCK_SIZE + pos), device=device)
                v = torch.full((N_HEADS, D_K), float(bid * BLOCK_SIZE + pos + 100), device=device)
                pool.write_kv(bid, pos, k, v)
        k_all, v_all = pool.gather_blocks([0, 1])
        for bid in (0, 1):
            for pos in range(BLOCK_SIZE):
                idx = bid * BLOCK_SIZE + pos
                expected_k = torch.full((N_HEADS, D_K), float(bid * BLOCK_SIZE + pos), device=device)
                expected_v = torch.full((N_HEADS, D_K), float(bid * BLOCK_SIZE + pos + 100), device=device)
                self.assertTrue(torch.allclose(k_all[idx], expected_k))
                self.assertTrue(torch.allclose(v_all[idx], expected_v))

    # ── 8. copy_block ─────────────────────────────────────────────────────

    def test_copy_block(self):
        """Copy block 0 to block 1, contents match."""
        pool = self._pool()
        device = torch.device(DEVICE)
        k = torch.randn(N_HEADS, D_K, device=device)
        v = torch.randn(N_HEADS, D_K, device=device)
        pool.write_kv(0, 0, k, v)
        pool.copy_block(0, 1)
        k0, v0 = pool.read_kv(0, 0)
        k1, v1 = pool.read_kv(1, 0)
        self.assertTrue(torch.allclose(k0, k1))
        self.assertTrue(torch.allclose(v0, v1))

    # ── 9. Invalid block_id ───────────────────────────────────────────────

    def test_invalid_block_id_raises(self):
        """IndexError on block_id >= num_blocks."""
        pool = self._pool()
        device = torch.device(DEVICE)
        k = torch.randn(N_HEADS, D_K, device=device)
        v = torch.randn(N_HEADS, D_K, device=device)
        with self.assertRaises(IndexError):
            pool.write_kv(NUM_BLOCKS, 0, k, v)

    # ── 10. Invalid token_pos ─────────────────────────────────────────────

    def test_invalid_token_pos_raises(self):
        """IndexError on token_pos >= block_size."""
        pool = self._pool()
        device = torch.device(DEVICE)
        k = torch.randn(N_HEADS, D_K, device=device)
        v = torch.randn(N_HEADS, D_K, device=device)
        with self.assertRaises(IndexError):
            pool.write_kv(0, BLOCK_SIZE, k, v)

    # ── 11. Memory MB ─────────────────────────────────────────────────────

    def test_get_memory_mb(self):
        """Returns positive float."""
        pool = self._pool()
        self.assertGreater(pool.get_memory_mb(), 0.0)


class TestCPUMemoryPool(unittest.TestCase):
    """Tests for CPUMemoryPool."""

    def _pool(self) -> CPUMemoryPool:
        return CPUMemoryPool(NUM_BLOCKS, BLOCK_SIZE, N_HEADS, D_K)

    # ── 12. CPU device ────────────────────────────────────────────────────

    def test_cpu_pool_device(self):
        """Always on CPU."""
        pool = self._pool()
        self.assertEqual(str(pool._pool.device), "cpu")
        self.assertEqual(pool.dtype, torch.float32)

    # ── 13. CPU write/read ────────────────────────────────────────────────

    def test_cpu_write_read(self):
        """Basic write/read works."""
        pool = self._pool()
        k = torch.randn(N_HEADS, D_K)
        v = torch.randn(N_HEADS, D_K)
        pool.write_kv(0, 0, k, v)
        k_out, v_out = pool.read_kv(0, 0)
        self.assertTrue(torch.allclose(k_out, k))
        self.assertTrue(torch.allclose(v_out, v))

    # ── 14. Pin memory ────────────────────────────────────────────────────

    def test_pin_memory(self):
        """pin_memory() succeeds without error."""
        pool = self._pool()
        try:
            pool.pin_memory()
        except RuntimeError:
            # pin_memory may fail on some backends (e.g. MPS).
            self.skipTest("pin_memory not supported on this backend")
        # Pinned memory should still be on CPU and readable.
        k = torch.randn(N_HEADS, D_K)
        v = torch.randn(N_HEADS, D_K)
        pool.write_kv(0, 0, k, v)
        k_out, v_out = pool.read_kv(0, 0)
        self.assertTrue(torch.allclose(k_out, k))


if __name__ == "__main__":
    unittest.main()
