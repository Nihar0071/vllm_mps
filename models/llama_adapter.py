"""LlamaForCausalLM adapter — injects PagedAttentionLayers.

Loads LlamaForCausalLM from HuggingFace, replaces each
``self_attn`` with a :class:`PagedAttentionLayer`, and exposes
a single-token forward interface.
"""

from __future__ import annotations

import logging
import threading

import torch

from vllm_mps.core.model_config import ModelConfig
from vllm_mps.core.kv_cache_manager import KVCacheManager
from vllm_mps.layers.paged_attention_layer import PagedAttentionLayer
from vllm_mps.memory.mps_memory_pool import MPSMemoryPool
from vllm_mps.models.base_adapter import BaseModelAdapter

logger = logging.getLogger(__name__)

# Thread-local context shared between LlamaAdapter and PagedAttentionLayer.
# This lets PagedAttentionLayer.forward() know whether it's in batch or single mode,
# and what seq_ids/positions to use, without changing the HF forward signature.
_batch_context = threading.local()
_batch_context.is_batch = False
_batch_context.seq_ids = []
_batch_context.positions = []
_batch_context.seq_lens = []


class LlamaAdapter(BaseModelAdapter):
    """Adapter for LlamaForCausalLM and compatible architectures."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self.model_config: ModelConfig | None = None
        self.hf_model = None
        self.tokenizer = None
        self.paged_layers: list[PagedAttentionLayer] = []
        self.kv_cache_manager: KVCacheManager | None = None
        self.memory_pool: MPSMemoryPool | None = None

    # ── BaseModelAdapter ──────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_tokenizer(self):
        return self.tokenizer

    def load(
        self,
        model_config: ModelConfig,
        kv_cache_manager: KVCacheManager,
        memory_pool: MPSMemoryPool,
    ) -> None:
        """Load weights and inject paged attention into every layer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_config = model_config
        self.kv_cache_manager = kv_cache_manager
        self.memory_pool = memory_pool

        # Load tokenizer.
        logger.info("LlamaAdapter: loading tokenizer for %s", self._model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)

        # Load model.
        logger.info("LlamaAdapter: loading model %s", self._model_name)
        device_str = "mps" if torch.backends.mps.is_available() else "cpu"
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=model_config.dtype,
            device_map=device_str,
            low_cpu_mem_usage=True,
        )
        self.hf_model.eval()

        # Log memory after load.
        if torch.backends.mps.is_available():
            mem_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
            logger.info("LlamaAdapter: MPS memory after load: %.1f MB", mem_mb)

        # Inject PagedAttentionLayers.
        self.paged_layers = []
        for i, layer in enumerate(self.hf_model.model.layers):
            paged_layer = PagedAttentionLayer.from_llama_layer(
                layer.self_attn,
                layer_idx=i,
                model_config=model_config,
                kv_cache_manager=kv_cache_manager,
                memory_pool=memory_pool,
            )
            layer.self_attn = paged_layer
            self.paged_layers.append(paged_layer)

        logger.info(
            "LlamaAdapter: injected %d PagedAttentionLayers into %s",
            len(self.paged_layers),
            self._model_name,
        )

    # ── Single-token forward (warmup + prefill) ──────────────────────────

    def forward_single_token(
        self, token_id: int, position_id: int, seq_id: int
    ) -> torch.Tensor:
        """Run one forward pass for one token.

        Returns:
            Logits tensor, shape ``(vocab_size,)``.
        """
        for pl in self.paged_layers:
            pl.set_context(seq_id, position_id)

        device = self.model_config.device
        input_ids = torch.tensor([[token_id]], device=device, dtype=torch.long)

        with torch.no_grad():
            outputs = self.hf_model(
                input_ids=input_ids,
                use_cache=False,
            )

        # (1, 1, vocab_size) → (vocab_size,)
        return outputs.logits[0, 0, :]

    # ── Batched decode forward ────────────────────────────────────────────

    def forward_batch(
        self,
        token_ids: list[int],
        seq_ids: list[int],
        positions: list[int],
        seq_lens: list[int],
    ) -> torch.Tensor:
        """Run one batched decode step for B sequences simultaneously.

        Uses ``inputs_embeds`` to bypass HF's internal embedding layer
        so we control the embedding ourselves.

        Args:
            token_ids: Last token ID for each sequence.  len = B.
            seq_ids:   Sequence IDs.  len = B.
            positions: Current position for each sequence.  len = B.
            seq_lens:  Total KV history length per sequence.  len = B.

        Returns:
            Logits tensor, shape ``(B, vocab_size)``.
        """
        device = self.model_config.device
        B = len(token_ids)

        # Embed all tokens at once: (B,) → (B, 1, d_model).
        tok_tensor = torch.tensor(
            token_ids, dtype=torch.long, device=device
        )  # (B,)
        inputs_embeds = self.hf_model.model.embed_tokens(tok_tensor)  # (B, d_model)
        inputs_embeds = inputs_embeds.unsqueeze(1)  # (B, 1, d_model)

        # Set batch context for all PagedAttentionLayers to read.
        _batch_context.is_batch = True
        _batch_context.seq_ids = seq_ids
        _batch_context.positions = positions
        _batch_context.seq_lens = seq_lens

        with torch.no_grad():
            outputs = self.hf_model(
                inputs_embeds=inputs_embeds,  # (B, 1, d_model)
                use_cache=False,
            )

        _batch_context.is_batch = False

        # outputs.logits: (B, 1, vocab_size) → (B, vocab_size)
        return outputs.logits[:, 0, :]

