# vllm-mps ⚡

**PagedAttention inference engine for Apple Silicon — built from scratch**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![PyTorch 2.10+](https://img.shields.io/badge/pytorch-2.10+-orange.svg)](https://pytorch.org)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black.svg)](https://apple.com/mac)
[![Tests](https://img.shields.io/badge/tests-77%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A ground-up implementation of [vLLM](https://github.com/vllm-project/vllm)-style paged KV-cache memory management for Apple's MPS (Metal Performance Shaders) backend. Runs real LLMs on your Mac with 3.46× better throughput than the HuggingFace default — no CUDA, no cloud, no cost.

---

## The Problem

vLLM is the gold standard for LLM inference — but it's CUDA-only. Apple Silicon GPUs sit idle because:

- HuggingFace's default KV cache pre-allocates `max_seq_len` slots per sequence, wasting 90%+ of memory on short requests
- No existing system implements PagedAttention's block-based memory management on MPS
- Metal shader compilation costs 10+ seconds on the first inference request, making cold-start latency appear catastrophic

Every ML developer on a Mac — the most popular developer machine in the world — is limited to slow CPU inference or HuggingFace's unoptimised MPS path.

**vllm-mps fixes this.**

---

## What This Project Is

This is both a working inference engine and a research project. We implemented PagedAttention from scratch on MPS, profiled every component at the operation level, and documented 15 concrete findings about how MPS behaves differently from CUDA — findings that don't exist anywhere else in the literature.

The code is fully tested (77 unit tests), the benchmarks are reproducible, and the findings are honest including cases where CUDA optimisation patterns made things *worse* on MPS.

---

## Performance

Measured on **MacBook Air M1 8GB**, TinyLlama 1.1B (float16), 20 decode tokens:

| Batch Size | Throughput  | ms / step |
|------------|-------------|-----------|
| 1          | 15.9 tok/s  | 62.9      |
| 2          | 18.8 tok/s  | 106.4     |
| 4          | 20.1 tok/s  | 198.6     |
| 6          | 20.8 tok/s  | 288.0     |
| **8**      | **22.5 tok/s** | **355.2** |

**Starting point (no optimisations): 6.5 tok/s → Final: 22.5 tok/s (+246%)**

### Optimisation History

Every improvement is data-driven and documented:

| Fix | Before | After | Gain | Finding |
|-----|--------|-------|------|---------|
| MPS shader warmup at startup | 6.5 tok/s | 8.6 tok/s | +32% | #008 |
| Block-ids tensor caching | 8.6 tok/s | 15.4 tok/s | +79% | #011 |
| Batched decode (BS=8) | 15.4 tok/s | 22.5 tok/s | +46% | #015 |
| **Total** | **6.5 tok/s** | **22.5 tok/s** | **+246%** | |

> Note: baseline is HuggingFace default on same hardware with same model,
> measured after MPS warmup for a fair comparison.

---

## Key Research Findings

This project discovered non-trivial MPS-specific behaviours that are not
documented elsewhere. Full details in [FINDINGS.md](findings.md).

**Finding 007 — `torch.tensor()` allocation costs ~0.9ms on MPS**
Creating a small index tensor inside a hot loop (called 22× per token)
added 401ms per 20 tokens. Caching the tensor reduced this 56%.
On CUDA the same allocation costs ~0.01ms — 90× cheaper.

**Finding 008 — Metal shader compilation is paid per-request without warmup**
First inference request pays 4+ seconds of Metal shader JIT compilation.
Running a dummy forward pass at startup eliminates this from user latency.
Every MPS inference system needs this — almost none implement it.

**Finding 012 — `torch.mps.synchronize()` makes profiling unreliable**
MPS profiling with synchronize=True adds ~3-6 seconds of artificial
overhead across 440 sync calls. Profiler showed only 2% improvement
where the real speedup was 79%. Never report synchronised MPS times
as absolute performance numbers.

**Finding 013 — `index_select` is slower than `repeat_interleave` on MPS**
Pre-building a GQA head expansion index (a standard CUDA optimisation)
increased latency by 21% on MPS. Metal's `repeat_interleave` kernel
is more optimised than the gather operation it was meant to replace.
CUDA optimisation patterns do not transfer directly to MPS.

**Finding 015 — Batch scaling model: 55ms fixed + 37ms per sequence**
Step time grows linearly at ~37ms per additional sequence — not from
memory bandwidth saturation, but from per-sequence Python loops for
KV gather and write. Batched Q/K/V matmuls are essentially free for
additional sequences. A Metal kernel fusing these loops would unlock
sub-linear scaling.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      LLMEngine                          │
│  from_pretrained() → load → warmup → step loop          │
├──────────┬──────────────────┬───────────────────────────┤
│ Scheduler│   LlamaAdapter   │        Sampler            │
│  (FCFS + │ (HF model host + │  (greedy/top-k/top-p)    │
│ preempt) │  layer injection)│                           │
├──────────┴──────────────────┴───────────────────────────┤
│           PagedAttentionLayer × N layers                │
│   Q/K/V proj → RoPE → write KV → gather blocks         │
│   → GQA expand → masked attention → output proj        │
├─────────────────────────────────────────────────────────┤
│                    Memory Layer                         │
│  MPSMemoryPool  │  BlockTable   │  BlockAllocator       │
│  (GPU tensor)   │  (per-seq)    │  (free-list FIFO)    │
│  CPUMemoryPool  │  KVCacheManager (coordinator)        │
└─────────────────────────────────────────────────────────┘
```

### How PagedAttention Works Here

Instead of pre-allocating `max_seq_len` slots per sequence (HuggingFace default),
we maintain a single pre-allocated pool tensor:

```
pool shape: (num_blocks, 2, block_size, n_kv_heads, d_k)
             256 blocks  K/V  16 tokens   4 heads    64 dims
             = 4MB total on MPS
```

Each sequence gets a `BlockTable` — a mapping from logical token positions
to physical block IDs. Blocks are allocated on demand, 16 tokens at a time,
from a FIFO free list. When a sequence finishes, its blocks return to the pool
immediately for reuse. Memory waste is bounded to one partially-filled block
per sequence (≤15 tokens) — near-zero fragmentation.

---

## Quickstart

### Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- PyTorch 2.10+ (MPS support built in)
- ~3GB free RAM for TinyLlama 1.1B

### Install

```bash
git clone https://github.com/Nihar0071/vllm_mps.git
cd vllm_mps
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Generate text

```python
from vllm_mps.engine.llm_engine import LLMEngine
from vllm_mps.core.sequence import SamplingParams

# Loads model, injects PagedAttentionLayers, runs MPS warmup
engine = LLMEngine.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

params = SamplingParams(max_tokens=50, temperature=0.8, top_p=0.95)
request_id = engine.add_request("The meaning of life is", params)

while not engine._outputs[request_id].finished:
    engine.step()

print(engine._outputs[request_id].generated_text)
```

### Run the demo

```bash
python run_demo.py
```

### Run benchmarks

```bash
# Operation-level bottleneck profiler (shows where time goes per token)
python -m vllm_mps.benchmarks.bottleneck_profiler

# Batch size throughput sweep (finds optimal batch size for your hardware)
python -m vllm_mps.benchmarks.batch_size_benchmark

# Naive vs paged memory comparison (shows 16× memory efficiency)
python -m vllm_mps.benchmarks.naive_vs_paged
```

### Run tests

```bash
python -m pytest tests/ -v
# 77 tests, all passing
```

---

## Project Structure

```
vllm_mps/
├── core/           # Block allocator, block table, KV cache manager, sequences
├── memory/         # MPS GPU pool, CPU fallback pool (for preemption swap)
├── layers/         # PagedAttentionLayer, RoPE embedding
├── models/         # LlamaAdapter, AutoAdapter (architecture router)
├── engine/         # LLMEngine, Scheduler (FCFS + preemption), ModelRunner
├── profiler/       # OperationTimer, live dashboard
├── benchmarks/     # Bottleneck profiler, batch benchmark, memory comparison
├── tests/          # 77 unit tests across all core components
├── examples/       # basic_generation.py
├── FINDINGS.md     # 15 documented MPS-specific research findings
└── run_demo.py     # End-to-end demo with 4 concurrent sequences
```

---

## Tested Models

| Model | VRAM needed | M1 8GB | M2 16GB |
|-------|-------------|--------|---------|
| TinyLlama 1.1B | ~2.2GB | ✅ Tested | ✅ |
| Llama 3.2 1B | ~2.0GB | ✅ Should work | ✅ |
| Llama 2 7B | ~14GB | ❌ OOM | ✅ Should work |
| Llama 3 8B | ~16GB | ❌ OOM | ✅ Should work |

Any Llama-architecture HuggingFace model should work via `AutoAdapter`.
Only TinyLlama 1.1B has been fully benchmarked on M1 8GB.

---

## Roadmap

- [ ] **Fix 4B** — Prefill phase batching (reduces first-token latency)
- [ ] **Fix 5** — Metal compute kernel for fused KV gather (eliminates 37ms/seq loop overhead)
- [ ] **Phi adapter** — Support Microsoft Phi model family
- [ ] **Mistral adapter** — Native Mistral architecture support
- [ ] **Block size benchmark** — Find optimal block size for M-series chips
- [ ] **pyproject.toml** — Proper pip-installable packaging

---

## Contributing

This project is early-stage and actively developed. Contributions welcome,
especially:

- Testing on M2/M3/M4 chips (different memory bandwidth characteristics)
- Metal shader implementation for `gather_blocks`
- Additional model adapters (Phi, Gemma, Qwen)
- Prefill phase implementation

Open an issue before starting large changes.

---

## Citation

If you use this work or findings in research, please cite:

```
@software{vllm_mps,
  author = {Patel, Nihar},
  title  = {vllm-mps: PagedAttention for Apple Silicon},
  year   = {2026},
  url    = {https://github.com/Nihar0071/vllm_mps}
}
```

---

## License

MIT — use freely, attribution appreciated.
