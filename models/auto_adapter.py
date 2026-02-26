"""Architecture router — selects the right adapter for a model name.

Currently supports Llama.  Phi and Mistral stubs raise
``NotImplementedError`` until their adapters are built.
"""

from __future__ import annotations

import logging

from vllm_mps.core.model_config import ModelConfig
from vllm_mps.core.kv_cache_manager import KVCacheManager
from vllm_mps.memory.mps_memory_pool import MPSMemoryPool
from vllm_mps.models.base_adapter import BaseModelAdapter

logger = logging.getLogger(__name__)


class AutoAdapter:
    """Factory that routes a model name to the correct adapter."""

    @staticmethod
    def from_pretrained(
        model_name: str,
        kv_cache_manager: KVCacheManager,
        memory_pool: MPSMemoryPool,
    ) -> BaseModelAdapter:
        """Load the right adapter for *model_name*.

        Args:
            model_name:       HuggingFace model identifier.
            kv_cache_manager: Shared KV block manager.
            memory_pool:      Shared tensor pool.

        Returns:
            A loaded :class:`BaseModelAdapter`.
        """
        config = ModelConfig.from_pretrained(model_name)

        if config.architecture == "llama":
            from vllm_mps.models.llama_adapter import LlamaAdapter
            adapter = LlamaAdapter(model_name)
        elif config.architecture == "phi":
            raise NotImplementedError(
                "PhiAdapter is not implemented yet. "
                "Contributions welcome!"
            )
        elif config.architecture == "mistral":
            raise NotImplementedError(
                "MistralAdapter is not implemented yet. "
                "Contributions welcome!"
            )
        else:
            raise ValueError(
                f"AutoAdapter: unsupported architecture "
                f"'{config.architecture}'. Supported: llama, phi, mistral."
            )

        logger.info("AutoAdapter: selected %s for %s", type(adapter).__name__, model_name)
        adapter.load(config, kv_cache_manager, memory_pool)
        return adapter
