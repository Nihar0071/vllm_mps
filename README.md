# vllm-mps ⚡

**PagedAttention inference engine for Apple Silicon (MPS)**

A from-scratch implementation of [vLLM](https://github.com/vllm-project/vllm)-style paged KV-cache management, optimised for Apple's Metal Performance Shaders backend. Run LLMs efficiently on your Mac — no CUDA required.

---

## Why vllm-mps?

vLLM revolutionised LLM serving with PagedAttention, but it's CUDA-only. Apple Silicon has powerful GPUs that sit idle because existing inference frameworks don't support paged KV-cache on MPS.

**vllm-mps bridges that gap:**

| Feature | HuggingFace (default) | vllm-mps |
|---|---|---|
| KV-cache | Contiguous per-sequence | Block-paged (16-token blocks) |
| Memory fragmentation | Grows with sequences | Near-zero (block allocator) |
| Concurrent sequences | OOM at 2–3 | 8+ with same memory |
| Throughput (TinyLlama) | ~6.5 tok/s | **22.5 tok/s** (batch=8) |
| Device | CUDA only (vLLM) | **Apple MPS** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      LLMEngine                          │
│  from_pretrained() → load → warmup → step loop         │
├──────────┬──────────────────┬───────────────────────────┤
│ Scheduler│   LlamaAdapter   │        Sampler            │
│ (FCFS)   │ (HF model host)  │   (temp/top-p/greedy)    │
├──────────┴──────────────────┴───────────────────────────┤
│               PagedAttentionLayer × N                   │
│  ┌──────────┐ ┌──────┐ ┌──────────┐ ┌───────────────┐  │
│  │ Q/K/V    │→│ RoPE │→│ Write KV │→│ Gather Blocks │  │
│  │ Project  │ │      │ │ to Pool  │ │ (cached ids)  │  │
│  └──────────┘ └──────┘ └──────────┘ └───────┬───────┘  │
│                              ┌───────────────┘          │
│                              ▼                          │
│  ┌──────────┐ ┌──────────────────┐ ┌────────────────┐  │
│  │ Output   │←│ Masked Attention │←│ GQA Expand     │  │
│  │ Project  │ │ (pad & mask)     │ │ (cached index) │  │
│  └──────────┘ └──────────────────┘ └────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                    Memory Layer                         │
│  ┌──────────────┐ ┌────────────┐ ┌───────────────────┐ │
│  │ MPSMemoryPool│ │BlockTable  │ │ BlockAllocator    │ │
│  │ (GPU tensor) │ │(per-seq)   │ │ (free-list mgmt) │ │
│  └──────────────┘ └────────────┘ └───────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Core Concepts

- **Block Allocator** — Manages a pool of fixed-size (16-token) physical blocks on GPU. Sequences request blocks on demand, return them when finished. Zero fragmentation.
- **Block Table** — Per-sequence mapping from logical blocks to physical block IDs. Grows as the sequence generates more tokens.
- **KV Cache Manager** — Coordinates allocation, slot appending, and freeing across sequences. Ensures no two sequences share blocks.
- **MPS Memory Pool** — Single pre-allocated `(num_blocks, 2, block_size, n_kv_heads, d_k)` tensor on MPS. All KV data lives here. `gather_blocks_tensor()` assembles non-contiguous blocks into contiguous attention input.
- **PagedAttentionLayer** — Drop-in replacement for HuggingFace `LlamaAttention`. Supports both single-token and batched decode paths. Includes per-seq block_ids caching and pre-built GQA head index.
- **LlamaAdapter** — Loads any Llama-architecture model from HuggingFace, injects PagedAttentionLayers, and exposes `forward_single_token()` and `forward_batch()`.

---

## Quickstart

### Installation

```bash
git clone https://github.com/Nihar0071/vllm_mps.git
cd vllm-mps
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Generate text

```python
from vllm_mps.engine.llm_engine import LLMEngine
from vllm_mps.core.sequence import SamplingParams

engine = LLMEngine.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

params = SamplingParams(max_tokens=50, temperature=0.8)
request_id = engine.add_request("The meaning of life is", params)

while not engine._outputs[request_id].finished:
    engine.step()

print(engine._outputs[request_id].text)
```

### Run benchmarks

```bash
# Operation-level profiler
python -m vllm_mps.benchmarks.bottleneck_profiler

# Batch size throughput sweep
python -m vllm_mps.benchmarks.batch_size_benchmark
```

---

## Performance

Measured on **MacBook Air M1 8GB**, TinyLlama 1.1B (float16), 20 decode tokens:

| Batch Size | Throughput | ms/step |
|---|---|---|
| 1 | 15.9 tok/s | 62.9 |
| 2 | 18.8 tok/s | 106.4 |
| 4 | 20.1 tok/s | 198.6 |
| 8 | **22.5 tok/s** | 355.2 |

### Optimisation history

| Fix | Before | After | Improvement |
|---|---|---|---|
| MPS shader warmup | 6.5 tok/s | 8.6 tok/s | +32% |
| Block-ids cache + GQA index | 8.6 tok/s | 15.4 tok/s | +79% |
| Batched decode (BS=8) | 15.4 tok/s | 22.5 tok/s | +46% |

---

## Project Structure

```
vllm_mps/
├── core/                  # Block allocator, block table, KV cache manager, sequences
├── memory/                # MPS GPU memory pool, CPU memory pool (swap)
├── layers/                # PagedAttentionLayer, RoPE embedding
├── models/                # LlamaAdapter, AutoAdapter (architecture router)
├── engine/                # LLMEngine, Scheduler, ModelRunner, Tokenizer
├── profiler/              # OperationTimer, memory & throughput profilers
├── benchmarks/            # Bottleneck profiler, batch size benchmark
├── api/                   # FastAPI server (WIP)
├── examples/              # Basic generation example
└── tests/                 # Unit tests for core components
```

---

## Key MPS-Specific Findings

1. **`torch.tensor()` allocation on MPS costs ~0.5ms** — 100× slower than equivalent CUDA allocation due to Metal command buffer overhead. Solved by caching block_id tensors.
2. **`repeat_interleave` recomputes index patterns each call** — Pre-building the GQA head expansion index once eliminates this.
3. **Metal shader compilation happens lazily** — First inference request pays 10+ seconds of shader compile. Warmup at startup eliminates this from user-facing latency.
4. **MPS benefits from batching** despite per-seq KV gather — Batched Q/K/V projections and attention provide +42% throughput at BS=8.

---

## Supported Models

Any Llama-architecture model on HuggingFace:
- TinyLlama 1.1B
- Llama 2 7B / 13B
- Llama 3 8B
- Code Llama
- Mistral 7B (Llama-compatible)

---

## Requirements

- Python 3.11+
- macOS with Apple Silicon (M1/M2/M3/M4)
- PyTorch 2.10+ with MPS support

---

## License

MIT
