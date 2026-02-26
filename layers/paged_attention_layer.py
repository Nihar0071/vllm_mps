"""Drop-in replacement for HuggingFace attention modules.

Clones weights from an existing attention layer and routes KV storage
through our block-based memory pool instead of contiguous tensors.
Supports Grouped Query Attention (GQA).
"""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext

import torch
import torch.nn as nn

from vllm_mps.config import BLOCK_SIZE
from vllm_mps.core.model_config import ModelConfig
from vllm_mps.core.kv_cache_manager import KVCacheManager
from vllm_mps.layers.rotary_embedding import RotaryEmbedding
from vllm_mps.memory.mps_memory_pool import MPSMemoryPool

logger = logging.getLogger(__name__)


class PagedAttentionLayer(nn.Module):
    """Attention layer that reads/writes KV data via a paged memory pool.

    Use :meth:`from_llama_layer` to build from a HuggingFace
    ``LlamaAttention`` instance.
    """

    def __init__(
        self,
        layer_idx: int,
        model_config: ModelConfig,
        kv_cache_manager: KVCacheManager,
        memory_pool: MPSMemoryPool,
        W_Q: nn.Linear,
        W_K: nn.Linear,
        W_V: nn.Linear,
        W_O: nn.Linear,
        rotary_emb: RotaryEmbedding,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.model_config = model_config
        self.kv_cache_manager = kv_cache_manager
        self.memory_pool = memory_pool

        self.W_Q = W_Q
        self.W_K = W_K
        self.W_V = W_V
        self.W_O = W_O
        self.rotary_emb = rotary_emb

        self.n_heads = model_config.n_heads
        self.n_kv_heads = model_config.n_kv_heads
        self.d_k = model_config.d_k
        self.d_model = model_config.d_model
        self.block_size = BLOCK_SIZE

        # For intercepting seq_id / position from adapter.
        self._current_seq_id: int | None = None
        self._current_position: int | None = None

        # Optional profiler — injected externally, None = zero overhead.
        self.timer = None

        # Fix 2: Per-seq block_ids tensor cache.
        # Avoids torch.tensor() allocation on every gather_blocks call.
        # Before: 0.912ms mean × 440 calls = 401ms (20.2% of total).
        # Cache is invalidated only when block table grows (every 16 tokens).
        self._block_ids_cache: dict[int, tuple[tuple[int, ...], torch.Tensor]] = {}

        # Fix 3: Pre-built GQA head repeat index.
        # Avoids repeat_interleave recomputing the index pattern each call.
        # Before: 0.393ms mean × 440 calls = 172ms (8.7% of total).
        if self.n_kv_heads != self.n_heads:
            repeat_factor = self.n_heads // self.n_kv_heads
            self._gqa_index = torch.repeat_interleave(
                torch.arange(self.n_kv_heads, device=model_config.device),
                repeat_factor,
            )
        else:
            self._gqa_index = None

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_llama_layer(
        cls,
        layer,
        layer_idx: int,
        model_config: ModelConfig,
        kv_cache_manager: KVCacheManager,
        memory_pool: MPSMemoryPool,
    ) -> PagedAttentionLayer:
        """Build from a HuggingFace ``LlamaAttention`` (transformers ≥ 5.x).

        Weights are **cloned** — the original layer can be discarded.
        """
        device = model_config.device
        dtype = model_config.dtype

        W_Q = nn.Linear(
            model_config.d_model,
            model_config.n_heads * model_config.d_k,
            bias=False, device=device, dtype=dtype,
        )
        W_K = nn.Linear(
            model_config.d_model,
            model_config.n_kv_heads * model_config.d_k,
            bias=False, device=device, dtype=dtype,
        )
        W_V = nn.Linear(
            model_config.d_model,
            model_config.n_kv_heads * model_config.d_k,
            bias=False, device=device, dtype=dtype,
        )
        W_O = nn.Linear(
            model_config.n_heads * model_config.d_k,
            model_config.d_model,
            bias=False, device=device, dtype=dtype,
        )

        # Clone weights from original layer.
        W_Q.weight.data.copy_(layer.q_proj.weight.data.to(device=device, dtype=dtype))
        W_K.weight.data.copy_(layer.k_proj.weight.data.to(device=device, dtype=dtype))
        W_V.weight.data.copy_(layer.v_proj.weight.data.to(device=device, dtype=dtype))
        W_O.weight.data.copy_(layer.o_proj.weight.data.to(device=device, dtype=dtype))

        rotary_emb = RotaryEmbedding(
            d_k=model_config.d_k,
            max_seq_len=model_config.max_seq_len,
            theta=model_config.rope_theta,
            device=device,
        )

        logger.debug("PagedAttentionLayer: cloned layer %d weights", layer_idx)
        return cls(
            layer_idx=layer_idx,
            model_config=model_config,
            kv_cache_manager=kv_cache_manager,
            memory_pool=memory_pool,
            W_Q=W_Q, W_K=W_K, W_V=W_V, W_O=W_O,
            rotary_emb=rotary_emb,
        )

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """Dispatch to single or batch path based on _batch_context.

        Args:
            hidden_states: ``(1, 1, d_model)`` single or ``(B, 1, d_model)`` batch.
        """
        from vllm_mps.models.llama_adapter import _batch_context

        if getattr(_batch_context, "is_batch", False):
            return self._forward_batch(
                hidden_states,
                _batch_context.seq_ids,
                _batch_context.positions,
                _batch_context.seq_lens,
            )
        return self._forward_single(hidden_states)

    # ── Single-token path ─────────────────────────────────────────────────

    def _forward_single(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        """Single-token forward (warmup / sequential fallback)."""
        seq_id = self._current_seq_id
        position_id = self._current_position

        if seq_id is None or position_id is None:
            raise RuntimeError(
                "PagedAttentionLayer: seq_id/position_id not set. "
                "Call set_context() before forward."
            )

        bsz, seq_len, _ = hidden_states.shape
        _t = self.timer

        # 1. Q/K/V projections.
        with _t.measure("qkv_projection") if _t else nullcontext():
            Q = self.W_Q(hidden_states).view(bsz, seq_len, self.n_heads, self.d_k)
            K = self.W_K(hidden_states).view(bsz, seq_len, self.n_kv_heads, self.d_k)
            V = self.W_V(hidden_states).view(bsz, seq_len, self.n_kv_heads, self.d_k)
            Q = Q.permute(0, 2, 1, 3)
            K = K.permute(0, 2, 1, 3)
            V = V.permute(0, 2, 1, 3)

        # 2. RoPE.
        with _t.measure("rope") if _t else nullcontext():
            Q, K = self.rotary_emb(Q, K, position_id)

        # 3. Block addressing.
        block_ids = self.kv_cache_manager.get_block_table(seq_id)
        block_idx = position_id // self.block_size
        token_pos = position_id % self.block_size

        # 4. Write K, V to pool.
        with _t.measure("write_kv") if _t else nullcontext():
            k_vec = K[0, :, 0, :]  # (n_kv_heads, d_k)
            v_vec = V[0, :, 0, :]  # (n_kv_heads, d_k)
            self.memory_pool.write_kv(block_ids[block_idx], token_pos, k_vec, v_vec)

        # 5. Gather full KV history (Fix 2: cached block_ids tensor).
        with _t.measure("gather_blocks") if _t else nullcontext():
            ids_tensor = self._get_cached_ids_tensor(seq_id, block_ids)
            K_full, V_full = self.memory_pool.gather_blocks_tensor(ids_tensor)
            K_full = K_full[: position_id + 1]
            V_full = V_full[: position_id + 1]

        # 6. GQA repeat (Fix 3: pre-built index).
        with _t.measure("gqa_repeat") if _t else nullcontext():
            K_full = K_full.unsqueeze(0).permute(0, 2, 1, 3)
            V_full = V_full.unsqueeze(0).permute(0, 2, 1, 3)
            if self._gqa_index is not None:
                K_full = K_full[:, self._gqa_index, :, :]
                V_full = V_full[:, self._gqa_index, :, :]

        # 7. Attention computation.
        with _t.measure("attention_compute") if _t else nullcontext():
            scores = torch.matmul(Q, K_full.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, V_full)

        # 8. Output projection.
        with _t.measure("output_projection") if _t else nullcontext():
            out = out.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, self.d_model)
            out = self.W_O(out)

        return (out, None)

    # ── Batched decode path ───────────────────────────────────────────────

    def _forward_batch(
        self,
        hidden_states: torch.Tensor,  # (B, 1, d_model)
        seq_ids: list[int],
        positions: list[int],
        seq_lens: list[int],
    ) -> tuple[torch.Tensor, None]:
        """Batched decode: B sequences, each contributing 1 new token.

        Steps:
            1. Project all B tokens → Q, K, V simultaneously
            2. Apply RoPE per sequence (different positions)
            3. Write new K, V to each sequence's block in memory pool
            4. Gather KV history per sequence, pad to max_len
            5. Build attention mask, run masked attention
            6. Project output → (B, 1, d_model)
        """

        B = hidden_states.shape[0]  # batch size
        device = hidden_states.device

        # 1. Project all at once — (B, 1, d_model) → (B, heads, 1, d_k).
        Q = self.W_Q(hidden_states)  # (B, 1, n_heads*d_k)
        K_new = self.W_K(hidden_states)  # (B, 1, n_kv_heads*d_k)
        V_new = self.W_V(hidden_states)  # (B, 1, n_kv_heads*d_k)

        Q = Q.view(B, 1, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, 1, d_k)
        K_new = K_new.view(B, 1, self.n_kv_heads, self.d_k).transpose(1, 2)  # (B, n_kv_heads, 1, d_k)
        V_new = V_new.view(B, 1, self.n_kv_heads, self.d_k).transpose(1, 2)  # (B, n_kv_heads, 1, d_k)

        # 2. Apply RoPE per sequence (positions differ).
        Q_rot_list, K_rot_list = [], []
        for i in range(B):
            q_rot, k_rot = self.rotary_emb(
                Q[i : i + 1],  # (1, n_heads, 1, d_k)
                K_new[i : i + 1],  # (1, n_kv_heads, 1, d_k)
                positions[i],
            )
            Q_rot_list.append(q_rot)
            K_rot_list.append(k_rot)
        Q_rot = torch.cat(Q_rot_list, dim=0)  # (B, n_heads, 1, d_k)
        K_rot = torch.cat(K_rot_list, dim=0)  # (B, n_kv_heads, 1, d_k)

        # 3. Write new K, V to memory pool for each sequence.
        for i in range(B):
            block_ids = self.kv_cache_manager.get_block_table(seq_ids[i])
            block_idx = positions[i] // self.block_size
            token_pos = positions[i] % self.block_size
            k_vec = K_rot[i, :, 0, :]  # (n_kv_heads, d_k)
            v_vec = V_new[i, :, 0, :]  # (n_kv_heads, d_k)
            self.memory_pool.write_kv(block_ids[block_idx], token_pos, k_vec, v_vec)

        # 4. Gather KV history per sequence and pad to max_len.
        max_len = max(seq_lens)

        K_padded = torch.zeros(
            B, max_len, self.n_kv_heads, self.d_k,
            dtype=hidden_states.dtype, device=device,
        )  # (B, max_len, n_kv_heads, d_k)
        V_padded = torch.zeros_like(K_padded)

        # Attention mask: True = attend, False = mask out.
        attn_mask = torch.zeros(
            B, 1, 1, max_len, dtype=torch.bool, device=device,
        )  # (B, 1, 1, max_len)

        for i in range(B):
            block_ids = self.kv_cache_manager.get_block_table(seq_ids[i])
            ids_tensor = self._get_cached_ids_tensor(seq_ids[i], block_ids)
            K_full, V_full = self.memory_pool.gather_blocks_tensor(ids_tensor)
            slen = seq_lens[i]
            K_padded[i, :slen] = K_full[:slen]  # (slen, n_kv_heads, d_k)
            V_padded[i, :slen] = V_full[:slen]
            attn_mask[i, 0, 0, :slen] = True

        # Reshape for attention: (B, n_kv_heads, max_len, d_k).
        K_padded = K_padded.transpose(1, 2)
        V_padded = V_padded.transpose(1, 2)

        # 5. GQA expand (Fix 3: pre-built index).
        if self._gqa_index is not None:
            K_padded = K_padded[:, self._gqa_index, :, :]  # (B, n_heads, max_len, d_k)
            V_padded = V_padded[:, self._gqa_index, :, :]

        # 6. Masked scaled dot-product attention.
        # Q_rot: (B, n_heads, 1, d_k), K_padded: (B, n_heads, max_len, d_k)
        scores = torch.matmul(Q_rot, K_padded.transpose(-2, -1))  # (B, n_heads, 1, max_len)
        scores = scores / math.sqrt(self.d_k)
        scores = scores.masked_fill(~attn_mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)  # (B, n_heads, 1, max_len)
        attn_out = torch.matmul(attn_weights, V_padded)  # (B, n_heads, 1, d_k)

        # 7. Combine heads and project.
        attn_out = attn_out.transpose(1, 2).contiguous()  # (B, 1, n_heads, d_k)
        attn_out = attn_out.view(B, 1, self.n_heads * self.d_k)  # (B, 1, d_model)
        output = self.W_O(attn_out)  # (B, 1, d_model)

        return (output, None)

    # ── Context setter ────────────────────────────────────────────────────

    def set_context(self, seq_id: int, position_id: int) -> None:
        """Set the sequence context before a forward pass."""
        self._current_seq_id = seq_id
        self._current_position = position_id

    # ── Cache helpers ─────────────────────────────────────────────────────

    def _get_cached_ids_tensor(
        self, seq_id: int, block_ids: list[int]
    ) -> torch.Tensor:
        """Return a cached block_ids tensor, rebuilding only when changed."""
        t = tuple(block_ids)
        cached = self._block_ids_cache.get(seq_id)
        if cached is None or cached[0] != t:
            tensor = torch.tensor(
                block_ids, dtype=torch.long, device=self.memory_pool.device
            )
            self._block_ids_cache[seq_id] = (t, tensor)
            return tensor
        return cached[1]

    def evict_sequence(self, seq_id: int) -> None:
        """Remove cached tensors for a finished/preempted sequence."""
        self._block_ids_cache.pop(seq_id, None)
