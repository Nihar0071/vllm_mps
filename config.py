"""Configuration module for vllm_mps.

Single source of truth for all hyperparameters. Every other module imports
from here — nothing is hardcoded elsewhere.
"""

import torch

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE  = torch.float16

# ── Model dimensions — kept small for M1 8GB ─────────────────────────────────
D_MODEL = 2048
N_HEADS = 32
D_K     = D_MODEL // N_HEADS   # 64 — dims per head
N_LAYERS = 22                   # ← also fix this, TinyLlama has 22 layers not 4
VOCAB_SIZE = 32000              # ← and this, TinyLlama has 32000 not 1000

# ── Memory pool ───────────────────────────────────────────────────────────────
BLOCK_SIZE         = 16    # tokens per physical block
NUM_GPU_BLOCKS     = 256   # total blocks in MPS pool
NUM_CPU_BLOCKS     = 128   # total blocks in CPU fallback pool
GPU_MEMORY_FRACTION = 0.85  # fraction of free MPS memory to use for KV pool

# ── Scheduler ─────────────────────────────────────────────────────────────────
MAX_BATCH_SIZE      = 8    # max sequences in one forward pass
MAX_SEQ_LEN         = 512  # maximum sequence length
MAX_TOKENS_PER_STEP = 512  # max total tokens batched per step

# ── Profiler ──────────────────────────────────────────────────────────────────
PROFILER_INTERVAL_SEC = 0.5  # how often dashboard refreshes
