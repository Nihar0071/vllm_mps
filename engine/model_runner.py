"""Model runner — prepares inputs and executes the model for one step."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from vllm_mps.config import D_MODEL, DEVICE, DTYPE, VOCAB_SIZE
from vllm_mps.core.kv_cache_manager import KVCacheManager
from vllm_mps.core.sequence import SequenceGroup, SequenceStatus
from vllm_mps.engine.tokenizer import SimpleTokenizer
from vllm_mps.memory.mps_memory_pool import MPSMemoryPool
from vllm_mps.model.paged_attention import PagedAttention
from vllm_mps.model.sampler import Sampler

logger = logging.getLogger(__name__)


@dataclass
class ModelInputs:
    """Everything the model needs for one forward step.

    Attributes:
        seq_ids:      Sequence IDs (to map outputs back).
        token_ids:    Last token for each sequence (autoregressive).
        positions:    Current position for each sequence in KV cache.
        block_tables: Physical block IDs per sequence.
    """

    seq_ids: list[int] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    block_tables: list[list[int]] = field(default_factory=list)


class ModelRunner:
    """Orchestrates input preparation and model execution.

    Attributes:
        model:            PagedAttention module.
        sampler:          Token sampler.
        kv_cache_manager: KV block manager.
        memory_pool:      Tensor pool.
        tokenizer:        Tokenizer (for vocab size).
    """

    def __init__(
        self,
        model: PagedAttention,
        sampler: Sampler,
        kv_cache_manager: KVCacheManager,
        memory_pool: MPSMemoryPool,
        tokenizer: SimpleTokenizer,
        device: str = DEVICE,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        self.model = model
        self.sampler = sampler
        self.kv_cache_manager = kv_cache_manager
        self.memory_pool = memory_pool
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dtype = dtype

        # Embedding table: token_id → d_model vector.
        self.embedding = nn.Embedding(
            VOCAB_SIZE, D_MODEL, device=self.device, dtype=dtype
        )
        # Output projection: d_model → vocab_size.
        self.lm_head = nn.Linear(
            D_MODEL, VOCAB_SIZE, bias=False, device=self.device, dtype=dtype
        )

    # ── Input preparation ─────────────────────────────────────────────────

    def prepare_inputs(
        self, scheduled_groups: list[SequenceGroup]
    ) -> ModelInputs:
        """Pack scheduled sequences into model inputs.

        For each RUNNING sequence: grab its last token, position, and
        block table.
        """
        inputs = ModelInputs()
        for group in scheduled_groups:
            for seq in group.get_seqs(SequenceStatus.RUNNING):
                inputs.seq_ids.append(seq.seq_id)
                inputs.token_ids.append(seq.get_last_token_id())
                inputs.positions.append(seq.get_total_len() - 1)
                inputs.block_tables.append(
                    self.kv_cache_manager.get_block_table(seq.seq_id)
                )
        return inputs

    # ── Model execution ───────────────────────────────────────────────────

    def execute_model(self, inputs: ModelInputs) -> list[int]:
        """Run one forward step and sample next tokens.

        Returns:
            List of next token IDs, one per sequence.
        """
        next_tokens: list[int] = []

        with torch.no_grad():
            for i, seq_id in enumerate(inputs.seq_ids):
                token_id = inputs.token_ids[i]
                pos = inputs.positions[i]

                # Embed → (1, 1, d_model).
                token_tensor = torch.tensor(
                    [[token_id]], device=self.device, dtype=torch.long
                )
                x = self.embedding(token_tensor)  # (1, 1, d_model)
                x = x.to(self.dtype)

                # Forward through attention → (1, 1, d_model).
                out = self.model.forward(x, seq_id=seq_id, current_pos=pos)

                # Project to vocab → (vocab_size,).
                logits = self.lm_head(out).squeeze(0).squeeze(0)

                # Find the group this seq belongs to for sampling params.
                sampling_params = self._find_sampling_params(seq_id)
                token = self.sampler.sample(logits, sampling_params)
                next_tokens.append(token)

        return next_tokens

    def _find_sampling_params(self, seq_id: int):
        """Look up sampling params for a sequence (via its registered group)."""
        # Default fallback — the engine passes params through Sequence.
        from vllm_mps.core.sequence import SamplingParams
        return SamplingParams()
