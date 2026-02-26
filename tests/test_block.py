"""Unit tests for vllm_mps.core.block.PhysicalBlock."""

import unittest

from vllm_mps.core.block import PhysicalBlock
from vllm_mps.config import BLOCK_SIZE


class TestPhysicalBlock(unittest.TestCase):
    """Tests for PhysicalBlock dataclass."""

    def test_defaults(self):
        """Block initialises with correct defaults."""
        block = PhysicalBlock(block_id=0)
        self.assertEqual(block.block_id, 0)
        self.assertEqual(block.ref_count, 0)
        self.assertEqual(block.num_tokens, 0)
        self.assertEqual(block.block_size, BLOCK_SIZE)

    def test_is_full(self):
        """is_full returns True only when num_tokens == block_size."""
        block = PhysicalBlock(block_id=1)
        self.assertFalse(block.is_full)
        block.add_tokens(BLOCK_SIZE)
        self.assertTrue(block.is_full)

    def test_is_free(self):
        """is_free returns True only when ref_count == 0."""
        block = PhysicalBlock(block_id=2)
        self.assertTrue(block.is_free)
        block.increment_ref()
        self.assertFalse(block.is_free)

    def test_add_tokens_overflow(self):
        """add_tokens raises ValueError when exceeding block_size."""
        block = PhysicalBlock(block_id=3)
        block.add_tokens(BLOCK_SIZE)
        with self.assertRaises(ValueError):
            block.add_tokens(1)

    def test_decrement_ref_below_zero(self):
        """decrement_ref raises ValueError when ref_count would go below 0."""
        block = PhysicalBlock(block_id=4)
        with self.assertRaises(ValueError):
            block.decrement_ref()

    def test_reset(self):
        """reset clears num_tokens and ref_count correctly."""
        block = PhysicalBlock(block_id=5)
        block.add_tokens(10)
        block.increment_ref()
        block.increment_ref()
        block.reset()
        self.assertEqual(block.num_tokens, 0)
        self.assertEqual(block.ref_count, 0)
        self.assertTrue(block.is_free)
        self.assertFalse(block.is_full)


if __name__ == "__main__":
    unittest.main()
