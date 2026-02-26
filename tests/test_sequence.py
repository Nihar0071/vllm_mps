"""Unit tests for vllm_mps.core.sequence."""

import unittest

from vllm_mps.core.sequence import (
    SamplingParams,
    Sequence,
    SequenceGroup,
    SequenceStatus,
)


def _seq(seq_id: int = 0, prompt_ids: list[int] | None = None,
         max_tokens: int = 5) -> Sequence:
    """Helper to build a Sequence with sensible defaults."""
    ids = [1, 2, 3] if prompt_ids is None else prompt_ids
    return Sequence(seq_id, "hello", ids, SamplingParams(max_tokens=max_tokens))


class TestSequence(unittest.TestCase):
    """Tests for Sequence and related data structures."""

    # ── 1. Initial state ──────────────────────────────────────────────────

    def test_sequence_initial_state(self):
        """WAITING, correct prompt len."""
        s = _seq()
        self.assertEqual(s.status, SequenceStatus.WAITING)
        self.assertEqual(s.get_prompt_len(), 3)
        self.assertEqual(s.get_output_len(), 0)
        self.assertEqual(s.get_total_len(), 3)

    # ── 2. add_token ──────────────────────────────────────────────────────

    def test_add_token_increments_len(self):
        """total_len grows, output_len grows."""
        s = _seq()
        s.add_token(99)
        self.assertEqual(s.get_total_len(), 4)
        self.assertEqual(s.get_output_len(), 1)
        self.assertEqual(s.get_prompt_len(), 3)

    # ── 3. Max tokens auto-finish ─────────────────────────────────────────

    def test_max_tokens_finishes_sequence(self):
        """Status FINISHED after max_tokens generated."""
        s = _seq(max_tokens=3)
        for tok in [10, 11, 12]:
            s.add_token(tok)
        self.assertTrue(s.is_finished())
        self.assertEqual(s.status, SequenceStatus.FINISHED)

    # ── 4. SamplingParams validation ──────────────────────────────────────

    def test_sampling_params_validation(self):
        """ValueError on bad temperature."""
        with self.assertRaises(ValueError):
            SamplingParams(temperature=-1.0)
        with self.assertRaises(ValueError):
            SamplingParams(top_k=-1)
        with self.assertRaises(ValueError):
            SamplingParams(top_p=0.0)
        with self.assertRaises(ValueError):
            SamplingParams(max_tokens=0)

    # ── 5. SequenceGroup filter ───────────────────────────────────────────

    def test_sequence_group_filter(self):
        """get_seqs(WAITING) vs get_seqs(RUNNING)."""
        s1 = _seq(0)
        s2 = _seq(1)
        s2.status = SequenceStatus.RUNNING
        g = SequenceGroup(0, [s1, s2], SamplingParams())
        self.assertEqual(len(g.get_seqs(SequenceStatus.WAITING)), 1)
        self.assertEqual(len(g.get_seqs(SequenceStatus.RUNNING)), 1)
        self.assertEqual(len(g.get_seqs()), 2)

    # ── 6. SequenceGroup is_finished ──────────────────────────────────────

    def test_sequence_group_is_finished(self):
        """True only when all seqs finished."""
        s1 = _seq(0, max_tokens=1)
        s2 = _seq(1, max_tokens=1)
        g = SequenceGroup(0, [s1, s2], SamplingParams(max_tokens=1))
        self.assertFalse(g.is_finished())
        s1.add_token(10)
        self.assertFalse(g.is_finished())
        s2.add_token(11)
        self.assertTrue(g.is_finished())

    # ── 7. get_last_token_id ──────────────────────────────────────────────

    def test_get_last_token_id(self):
        """Returns correct last token."""
        s = _seq(prompt_ids=[1, 2, 3])
        self.assertEqual(s.get_last_token_id(), 3)
        s.add_token(42)
        self.assertEqual(s.get_last_token_id(), 42)

    # ── 8. get_last_token_id empty ────────────────────────────────────────

    def test_get_last_token_id_empty(self):
        """Raises on empty token_ids."""
        s = _seq(prompt_ids=[])
        with self.assertRaises(IndexError):
            s.get_last_token_id()


if __name__ == "__main__":
    unittest.main()
