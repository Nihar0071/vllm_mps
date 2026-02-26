"""Request scheduler for the vllm_mps inference engine.

Decides which sequences run in each forward-pass step, handles memory
admission, preemption, and resumption via the KVCacheManager.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

from vllm_mps.config import MAX_BATCH_SIZE, MAX_TOKENS_PER_STEP
from vllm_mps.core.kv_cache_manager import KVCacheManager
from vllm_mps.core.sequence import Sequence, SequenceGroup, SequenceStatus

logger = logging.getLogger(__name__)


@dataclass
class SchedulerOutput:
    """Result of a single scheduling step.

    Attributes:
        scheduled_groups:  Groups selected for this forward pass.
        preempted_groups:  Groups evicted this step due to memory pressure.
        num_batched_tokens: Total tokens in the batch.
        num_scheduled_seqs: Total individual sequences scheduled.
    """

    scheduled_groups: list[SequenceGroup] = field(default_factory=list)
    preempted_groups: list[SequenceGroup] = field(default_factory=list)
    num_batched_tokens: int = 0
    num_scheduled_seqs: int = 0


class Scheduler:
    """Coordinates request admission, preemption, and batching.

    Priority order: preempted (resume) → waiting (admit) → preempt if needed.

    Attributes:
        kv_cache_manager:    Memory manager for KV blocks.
        max_batch_size:      Max sequences per forward pass.
        max_tokens_per_step: Max total tokens batched per step.
    """

    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        max_batch_size: int = MAX_BATCH_SIZE,
        max_tokens_per_step: int = MAX_TOKENS_PER_STEP,
    ) -> None:
        """Create a scheduler.

        Args:
            kv_cache_manager:    Shared KV-cache memory manager.
            max_batch_size:      Maximum sequences in one forward pass.
            max_tokens_per_step: Maximum total tokens batched per step.
        """
        self.kv_cache_manager = kv_cache_manager
        self.max_batch_size = max_batch_size
        self.max_tokens_per_step = max_tokens_per_step

        self._waiting: deque[SequenceGroup] = deque()
        self._running: list[SequenceGroup] = []
        self._preempted: list[SequenceGroup] = []

    # ── Request management ────────────────────────────────────────────────

    def add_request(self, group: SequenceGroup) -> None:
        """Enqueue a new request for scheduling.

        Args:
            group: The sequence group to add.
        """
        self._waiting.append(group)
        logger.info(
            "Scheduler: queued group=%d, waiting=%d",
            group.group_id,
            len(self._waiting),
        )

    # ── Core scheduling ───────────────────────────────────────────────────

    def schedule(self) -> SchedulerOutput:
        """Run one scheduling step.

        Returns:
            A :class:`SchedulerOutput` describing what to run and what
            was preempted.
        """
        scheduled_groups: list[SequenceGroup] = []
        preempted_groups: list[SequenceGroup] = []
        num_scheduled_seqs = 0
        num_batched_tokens = 0

        # ── Step 1: Resume preempted groups (highest priority) ────────
        resumed: list[SequenceGroup] = []
        for group in list(self._preempted):
            if num_scheduled_seqs >= self.max_batch_size:
                break

            seqs = group.get_seqs(SequenceStatus.PREEMPTED)
            total_tokens = sum(s.get_total_len() for s in seqs)

            # Check if we can fit them.
            can_fit = all(
                self.kv_cache_manager.can_allocate(s.seq_id, s.get_total_len())
                for s in seqs
            )
            if not can_fit:
                continue

            # Swap in each sequence.
            for s in seqs:
                self.kv_cache_manager.swap_in(s.seq_id)
                s.status = SequenceStatus.RUNNING

            self._preempted.remove(group)
            self._running.append(group)
            scheduled_groups.append(group)
            num_scheduled_seqs += len(seqs)
            num_batched_tokens += total_tokens
            resumed.append(group)

        # ── Step 2: Admit waiting groups ──────────────────────────────
        admitted: list[SequenceGroup] = []
        while self._waiting:
            if num_scheduled_seqs >= self.max_batch_size:
                break

            group = self._waiting[0]
            seqs = group.get_seqs(SequenceStatus.WAITING)
            total_tokens = sum(s.get_total_len() for s in seqs)

            if num_batched_tokens + total_tokens > self.max_tokens_per_step:
                break

            # Check memory.
            can_fit = all(
                self.kv_cache_manager.can_allocate(s.seq_id, s.get_total_len())
                for s in seqs
            )
            if not can_fit:
                break

            self._waiting.popleft()

            for s in seqs:
                self.kv_cache_manager.allocate(s.seq_id)
                # Append prompt token slots.
                for _ in range(s.get_prompt_len()):
                    self.kv_cache_manager.append_slot(s.seq_id)
                s.status = SequenceStatus.RUNNING

            self._running.append(group)
            scheduled_groups.append(group)
            num_scheduled_seqs += len(seqs)
            num_batched_tokens += total_tokens
            admitted.append(group)

        # ── Step 3: Preempt running groups under memory pressure ──────
        # Check if any already-running group (not scheduled this step)
        # can append 1 more token.  If not, preempt lowest priority.
        already_running = [
            g for g in self._running if g not in scheduled_groups
        ]
        for group in already_running:
            seqs = group.get_seqs(SequenceStatus.RUNNING)
            can_continue = all(
                self.kv_cache_manager.can_allocate(s.seq_id, 1)
                for s in seqs
            )
            if can_continue:
                # Still fits — add to this step's batch.
                scheduled_groups.append(group)
                num_scheduled_seqs += len(seqs)
                num_batched_tokens += sum(s.get_total_len() for s in seqs)
            else:
                # Preempt: swap out and move to preempted list.
                for s in seqs:
                    self.kv_cache_manager.swap_out(s.seq_id)
                    s.status = SequenceStatus.PREEMPTED
                self._running.remove(group)
                self._preempted.append(group)
                preempted_groups.append(group)

        # ── Step 4: Append 1 slot per running seq in scheduled groups ─
        # Only for groups that were already running before this step
        # (newly admitted groups already had their prompt slots appended).
        for group in scheduled_groups:
            if group in admitted or group in resumed:
                continue
            for s in group.get_seqs(SequenceStatus.RUNNING):
                self.kv_cache_manager.append_slot(s.seq_id)

        # ── Step 5: Build and return output ───────────────────────────
        return SchedulerOutput(
            scheduled_groups=scheduled_groups,
            preempted_groups=preempted_groups,
            num_batched_tokens=num_batched_tokens,
            num_scheduled_seqs=num_scheduled_seqs,
        )

    # ── Completion ────────────────────────────────────────────────────────

    def mark_finished(self, seq_id: int) -> None:
        """Mark a sequence as finished and free its memory if group is done.

        Args:
            seq_id: The sequence to mark finished.
        """
        for group in self._running:
            for s in group.sequences:
                if s.seq_id == seq_id:
                    s.status = SequenceStatus.FINISHED
                    if group.is_finished():
                        for seq in group.sequences:
                            self.kv_cache_manager.free(seq.seq_id)
                        self._running.remove(group)
                        logger.info(
                            "Scheduler: group=%d finished, memory freed",
                            group.group_id,
                        )
                    return

    # ── Introspection ─────────────────────────────────────────────────────

    def get_num_waiting(self) -> int:
        """Return number of groups in the waiting queue."""
        return len(self._waiting)

    def get_num_running(self) -> int:
        """Return number of groups currently running."""
        return len(self._running)

    def get_num_preempted(self) -> int:
        """Return number of preempted groups."""
        return len(self._preempted)

    def __repr__(self) -> str:
        """Return a human-readable summary."""
        return (
            f"Scheduler(waiting={len(self._waiting)}, "
            f"running={len(self._running)}, "
            f"preempted={len(self._preempted)})"
        )
