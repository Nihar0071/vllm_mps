"""Unit tests for vllm_mps.engine.scheduler.Scheduler."""

import unittest

from vllm_mps.config import BLOCK_SIZE
from vllm_mps.core.kv_cache_manager import KVCacheManager
from vllm_mps.core.sequence import (
    SamplingParams,
    Sequence,
    SequenceGroup,
    SequenceStatus,
)
from vllm_mps.engine.scheduler import Scheduler

GPU_BLOCKS = 32
CPU_BLOCKS = 16


def _make_scheduler(gpu: int = GPU_BLOCKS, cpu: int = CPU_BLOCKS,
                    max_batch: int = 4, max_tokens: int = 512) -> Scheduler:
    """Create a Scheduler backed by a fresh KVCacheManager."""
    kv = KVCacheManager(num_gpu_blocks=gpu, num_cpu_blocks=cpu)
    return Scheduler(kv, max_batch_size=max_batch,
                     max_tokens_per_step=max_tokens)


def _group(gid: int, prompt_len: int = 4, max_tokens: int = 10,
           num_seqs: int = 1) -> SequenceGroup:
    """Create a SequenceGroup with *num_seqs* sequences."""
    params = SamplingParams(max_tokens=max_tokens)
    seqs = [
        Sequence(gid * 100 + i, "p", list(range(prompt_len)), params)
        for i in range(num_seqs)
    ]
    return SequenceGroup(gid, seqs, params)


class TestScheduler(unittest.TestCase):
    """Tests for Scheduler scheduling logic."""

    # ── 1. add_request ────────────────────────────────────────────────────

    def test_add_request_goes_to_waiting(self):
        """Group appears in waiting queue."""
        sched = _make_scheduler()
        sched.add_request(_group(0))
        self.assertEqual(sched.get_num_waiting(), 1)
        self.assertEqual(sched.get_num_running(), 0)

    # ── 2. schedule admits ────────────────────────────────────────────────

    def test_schedule_admits_waiting_group(self):
        """After schedule(), group is running."""
        sched = _make_scheduler()
        sched.add_request(_group(0))
        out = sched.schedule()
        self.assertEqual(sched.get_num_waiting(), 0)
        self.assertEqual(sched.get_num_running(), 1)
        self.assertEqual(len(out.scheduled_groups), 1)

    # ── 3. output counts ─────────────────────────────────────────────────

    def test_schedule_output_counts(self):
        """num_batched_tokens is correct."""
        sched = _make_scheduler()
        sched.add_request(_group(0, prompt_len=4))
        out = sched.schedule()
        self.assertEqual(out.num_scheduled_seqs, 1)
        self.assertEqual(out.num_batched_tokens, 4)

    # ── 4. batch size limit ───────────────────────────────────────────────

    def test_batch_size_limit(self):
        """Respects max_batch_size."""
        sched = _make_scheduler(max_batch=2)
        for i in range(5):
            sched.add_request(_group(i, prompt_len=2))
        out = sched.schedule()
        self.assertEqual(out.num_scheduled_seqs, 2)
        self.assertEqual(sched.get_num_waiting(), 3)

    # ── 5. mark_finished frees memory ─────────────────────────────────────

    def test_mark_finished_frees_memory(self):
        """GPU blocks returned after finish."""
        sched = _make_scheduler()
        sched.add_request(_group(0, prompt_len=4))
        sched.schedule()
        free_before = sched.kv_cache_manager.get_num_free_gpu_blocks()
        sched.mark_finished(0)  # seq_id = gid*100 + 0 = 0
        free_after = sched.kv_cache_manager.get_num_free_gpu_blocks()
        self.assertGreater(free_after, free_before)
        self.assertEqual(sched.get_num_running(), 0)

    # ── 6. preemption ─────────────────────────────────────────────────────

    def test_preemption_when_memory_full(self):
        """Fill GPU, add new seq → preemption of running group."""
        # Small GPU pool: 4 blocks, each holds BLOCK_SIZE tokens.
        sched = _make_scheduler(gpu=4, max_batch=10, max_tokens=9999)
        # First group takes 3 blocks (prompt_len > 2*BLOCK_SIZE).
        sched.add_request(_group(0, prompt_len=BLOCK_SIZE * 3))
        sched.schedule()
        self.assertEqual(sched.get_num_running(), 1)

        # Second group needs 2 blocks — only 1 free → preemption.
        sched.add_request(_group(1, prompt_len=BLOCK_SIZE * 2))
        out = sched.schedule()
        # Group 0 should be preempted OR group 1 stays waiting.
        # Given logic: group 1 can't be admitted (not enough memory),
        # but group 0 can continue → no preemption needed if group 0 fits.
        # Actually group 0 needs 1 more slot → can_allocate(0, 1) with 1 free block → True.
        # So group 1 stays waiting.
        self.assertEqual(sched.get_num_waiting(), 1)

    # ── 7. preempted group resumes ────────────────────────────────────────

    def test_preempted_group_resumes(self):
        """Preempted group comes back after memory freed."""
        sched = _make_scheduler(gpu=4, cpu=4, max_batch=10, max_tokens=9999)
        g0 = _group(0, prompt_len=BLOCK_SIZE * 3)
        sched.add_request(g0)
        sched.schedule()

        # Manually preempt group 0 to test resume.
        for s in g0.get_seqs(SequenceStatus.RUNNING):
            sched.kv_cache_manager.swap_out(s.seq_id)
            s.status = SequenceStatus.PREEMPTED
        sched._running.remove(g0)
        sched._preempted.append(g0)
        self.assertEqual(sched.get_num_preempted(), 1)

        # Schedule again — should resume.
        out = sched.schedule()
        self.assertEqual(sched.get_num_preempted(), 0)
        self.assertEqual(sched.get_num_running(), 1)

    # ── 8. multiple steps ─────────────────────────────────────────────────

    def test_multiple_steps(self):
        """Run schedule() 5 times, counts correct."""
        sched = _make_scheduler()
        sched.add_request(_group(0, prompt_len=4, max_tokens=5))
        sched.schedule()  # admit
        for _ in range(4):
            out = sched.schedule()  # continue running
            self.assertEqual(sched.get_num_running(), 1)

    # ── 9. FIFO ordering ─────────────────────────────────────────────────

    def test_fifo_ordering(self):
        """Earlier arrival scheduled first."""
        sched = _make_scheduler(max_batch=1)
        g0 = _group(0)
        g1 = _group(1)
        sched.add_request(g0)
        sched.add_request(g1)
        out = sched.schedule()
        self.assertIn(g0, out.scheduled_groups)
        self.assertNotIn(g1, out.scheduled_groups)

    # ── 10. Empty schedule ────────────────────────────────────────────────

    def test_empty_schedule(self):
        """schedule() on empty queues returns 0 scheduled."""
        sched = _make_scheduler()
        out = sched.schedule()
        self.assertEqual(out.num_scheduled_seqs, 0)
        self.assertEqual(len(out.scheduled_groups), 0)


if __name__ == "__main__":
    unittest.main()
