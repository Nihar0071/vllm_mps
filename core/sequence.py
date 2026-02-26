"""Sequence data structures for request tracking.

Defines the lifecycle states, sampling parameters, and per-request
metadata used by the scheduler and model runner.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class SequenceStatus(Enum):
    """Lifecycle state of a single sequence."""

    WAITING = "waiting"       # in queue, no memory allocated
    RUNNING = "running"       # currently in a forward-pass batch
    PREEMPTED = "preempted"   # evicted due to memory pressure
    FINISHED = "finished"     # generation complete (EOS or max_tokens)


@dataclass
class SamplingParams:
    """Parameters that control token sampling for a sequence.

    Attributes:
        temperature:    Softmax temperature (0 = greedy).
        top_k:          Top-K filtering (0 = disabled).
        top_p:          Nucleus filtering (1.0 = disabled).
        max_tokens:     Maximum number of generated tokens.
        stop_sequences: Strings that trigger early stopping.
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    max_tokens: int = 128
    stop_sequences: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        if self.temperature < 0.0:
            raise ValueError(
                f"SamplingParams: temperature must be >= 0.0, "
                f"got {self.temperature}."
            )
        if self.top_k < 0:
            raise ValueError(
                f"SamplingParams: top_k must be >= 0, got {self.top_k}."
            )
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(
                f"SamplingParams: top_p must be in (0.0, 1.0], "
                f"got {self.top_p}."
            )
        if self.max_tokens < 1:
            raise ValueError(
                f"SamplingParams: max_tokens must be >= 1, "
                f"got {self.max_tokens}."
            )


class Sequence:
    """Represents a single in-flight sequence (prompt + generated tokens).

    Attributes:
        seq_id:          Unique identifier.
        prompt:          Original prompt text.
        token_ids:       Full token list (prompt + generated).
        sampling_params: Sampling configuration.
        status:          Current lifecycle state.
        arrival_time:    Wall-clock time when the sequence was created.
    """

    def __init__(
        self,
        seq_id: int,
        prompt: str,
        token_ids: list[int],
        sampling_params: SamplingParams,
    ) -> None:
        """Create a new sequence in WAITING state.

        Args:
            seq_id:          Unique sequence identifier.
            prompt:          Original prompt string.
            token_ids:       Tokenised prompt IDs.
            sampling_params: Sampling configuration.
        """
        self.seq_id = seq_id
        self.prompt = prompt
        self.token_ids = list(token_ids)
        self.sampling_params = sampling_params
        self.status = SequenceStatus.WAITING
        self.arrival_time = time.time()

        self._prompt_len = len(token_ids)
        self._output_token_ids: list[int] = []

    # ── Token management ──────────────────────────────────────────────────

    def add_token(self, token_id: int) -> None:
        """Append a generated token and auto-finish if max_tokens reached.

        Args:
            token_id: The newly sampled token ID.
        """
        self.token_ids.append(token_id)
        self._output_token_ids.append(token_id)

        if len(self._output_token_ids) >= self.sampling_params.max_tokens:
            self.status = SequenceStatus.FINISHED

    # ── Length queries ────────────────────────────────────────────────────

    def get_prompt_len(self) -> int:
        """Return the number of prompt tokens."""
        return self._prompt_len

    def get_output_len(self) -> int:
        """Return the number of generated tokens so far."""
        return len(self._output_token_ids)

    def get_total_len(self) -> int:
        """Return the total number of tokens (prompt + generated)."""
        return len(self.token_ids)

    # ── Status queries ────────────────────────────────────────────────────

    def is_finished(self) -> bool:
        """Return True if the sequence has completed generation."""
        return self.status == SequenceStatus.FINISHED

    def get_last_token_id(self) -> int:
        """Return the most recent token ID.

        Raises:
            IndexError: If the token list is empty.
        """
        if not self.token_ids:
            raise IndexError(
                f"Sequence [seq={self.seq_id}]: token_ids is empty."
            )
        return self.token_ids[-1]

    def __repr__(self) -> str:
        """Return a human-readable summary."""
        return (
            f"Sequence(seq_id={self.seq_id}, "
            f"status={self.status.value}, "
            f"total_len={self.get_total_len()})"
        )


class SequenceGroup:
    """A group of related sequences (e.g. beam search candidates).

    Attributes:
        group_id:        Unique group identifier.
        sequences:       Member sequences.
        sampling_params: Shared sampling configuration.
        arrival_time:    Wall-clock time when the group was created.
    """

    def __init__(
        self,
        group_id: int,
        sequences: list[Sequence],
        sampling_params: SamplingParams,
    ) -> None:
        """Create a sequence group.

        Args:
            group_id:        Unique group identifier.
            sequences:       List of member sequences.
            sampling_params: Shared sampling configuration.
        """
        self.group_id = group_id
        self.sequences = sequences
        self.sampling_params = sampling_params
        self.arrival_time = time.time()

    def get_seqs(
        self, status: SequenceStatus | None = None
    ) -> list[Sequence]:
        """Return sequences, optionally filtered by *status*.

        Args:
            status: If provided, only sequences with this status are returned.
        """
        if status is None:
            return list(self.sequences)
        return [s for s in self.sequences if s.status == status]

    def get_num_seqs(
        self, status: SequenceStatus | None = None
    ) -> int:
        """Return the count of sequences matching *status*."""
        return len(self.get_seqs(status))

    def is_finished(self) -> bool:
        """Return True if every sequence in the group is finished."""
        return all(s.is_finished() for s in self.sequences)

    def get_max_total_len(self) -> int:
        """Return the maximum total token length across all sequences."""
        return max(s.get_total_len() for s in self.sequences)

    def __repr__(self) -> str:
        """Return a human-readable summary."""
        return (
            f"SequenceGroup(group_id={self.group_id}, "
            f"num_seqs={len(self.sequences)}, "
            f"finished={self.is_finished()})"
        )
