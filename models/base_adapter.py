"""Abstract base class for HuggingFace model adapters.

Each supported architecture implements this interface so that
the engine can load and run models polymorphically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from vllm_mps.core.model_config import ModelConfig
from vllm_mps.core.kv_cache_manager import KVCacheManager
from vllm_mps.memory.mps_memory_pool import MPSMemoryPool


class BaseModelAdapter(ABC):
    """Interface for HuggingFace model adapters."""

    @abstractmethod
    def load(
        self,
        model_config: ModelConfig,
        kv_cache_manager: KVCacheManager,
        memory_pool: MPSMemoryPool,
    ) -> None:
        """Load model weights and inject PagedAttentionLayers."""

    @abstractmethod
    def forward_single_token(
        self, token_id: int, position_id: int, seq_id: int
    ) -> torch.Tensor:
        """Run one forward pass for one token.

        Returns:
            Logits tensor, shape ``(vocab_size,)``.
        """

    @abstractmethod
    def get_tokenizer(self):
        """Return the HuggingFace tokenizer for this model."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the HuggingFace model name string."""
