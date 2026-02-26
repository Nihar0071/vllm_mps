"""Model configuration extracted from HuggingFace config.json.

Reads a pretrained model's config and maps it to the fields our
paged-attention system needs (head counts, dimensions, RoPE params).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from transformers import AutoConfig

from vllm_mps.config import BLOCK_SIZE

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Unified configuration for any supported HuggingFace model.

    Use :meth:`from_pretrained` to build from a model name.
    """

    model_name: str
    architecture: str
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_k: int
    n_layers: int
    vocab_size: int
    max_seq_len: int
    rope_theta: float
    dtype: torch.dtype
    device: torch.device

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, model_name: str) -> ModelConfig:
        """Load a HuggingFace config and extract relevant fields.

        Args:
            model_name: HuggingFace model identifier (e.g.
                ``"TinyLlama/TinyLlama-1.1B-Chat-v1.0"``).
        """
        hf = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        hidden_size = getattr(hf, "hidden_size", 2048)
        num_attention_heads = getattr(hf, "num_attention_heads", 32)
        num_kv_heads = getattr(
            hf, "num_key_value_heads", num_attention_heads
        )
        num_layers = getattr(hf, "num_hidden_layers", 22)
        vocab = getattr(hf, "vocab_size", 32000)
        max_pos = getattr(hf, "max_position_embeddings", 2048)
        rope_theta = getattr(hf, "rope_theta", 10000.0)

        # Detect architecture.
        arch = "unknown"
        if hasattr(hf, "architectures") and hf.architectures:
            arch_name = hf.architectures[0].lower()
            for tag in ("llama", "phi", "mistral"):
                if tag in arch_name:
                    arch = tag
                    break

        dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        cfg = cls(
            model_name=model_name,
            architecture=arch,
            d_model=hidden_size,
            n_heads=num_attention_heads,
            n_kv_heads=num_kv_heads,
            d_k=hidden_size // num_attention_heads,
            n_layers=num_layers,
            vocab_size=vocab,
            max_seq_len=max_pos,
            rope_theta=rope_theta,
            dtype=dtype,
            device=device,
        )

        logger.info(
            "ModelConfig: %s  arch=%s  d_model=%d  heads=%d  kv_heads=%d  "
            "d_k=%d  layers=%d  vocab=%d  max_seq=%d  rope_θ=%.0f",
            model_name, arch, cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
            cfg.d_k, cfg.n_layers, cfg.vocab_size, cfg.max_seq_len,
            cfg.rope_theta,
        )
        return cfg

    # ── Helpers ────────────────────────────────────────────────────────────

    def get_kv_cache_config(self) -> dict:
        """Return dict of parameters needed to build an MPSMemoryPool."""
        return {
            "n_heads": self.n_kv_heads,
            "d_k": self.d_k,
            "block_size": BLOCK_SIZE,
            "dtype": self.dtype,
            "device": self.device,
        }
