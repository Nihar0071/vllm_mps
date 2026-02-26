"""LLM inference engine — ties all components together.

Provides the high-level ``generate()`` and ``step()`` API that the user
or an API server would call.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch

from vllm_mps.config import (
    BLOCK_SIZE,
    D_K,
    D_MODEL,
    DEVICE,
    DTYPE,
    MAX_BATCH_SIZE,
    MAX_TOKENS_PER_STEP,
    N_HEADS,
    NUM_CPU_BLOCKS,
    NUM_GPU_BLOCKS,
    VOCAB_SIZE,
)
from vllm_mps.core.kv_cache_manager import KVCacheManager
from vllm_mps.core.sequence import (
    SamplingParams,
    Sequence,
    SequenceGroup,
    SequenceStatus,
)
from vllm_mps.engine.model_runner import ModelRunner
from vllm_mps.engine.scheduler import Scheduler
from vllm_mps.engine.tokenizer import SimpleTokenizer
from vllm_mps.memory.mps_memory_pool import MPSMemoryPool
from vllm_mps.model.paged_attention import PagedAttention
from vllm_mps.model.sampler import Sampler

logger = logging.getLogger(__name__)


@dataclass
class RequestOutput:
    """Current state of a generation request.

    Attributes:
        request_id:     Unique request identifier.
        prompt:         Original prompt text.
        generated_text: Decoded generated text so far.
        token_ids:      Output token IDs only (no prompt tokens).
        finished:       Whether generation is complete.
        num_steps:      Number of generation steps taken.
    """

    request_id: int
    prompt: str
    generated_text: str = ""
    token_ids: list[int] = field(default_factory=list)
    finished: bool = False
    num_steps: int = 0


class LLMEngine:
    """High-level inference engine coordinating all subsystems.

    Usage::

        engine = LLMEngine()
        text = engine.generate("hello world", max_tokens=50)
    """

    def __init__(self) -> None:
        """Initialise all engine components using config values."""
        device = torch.device(DEVICE)
        dtype = DTYPE

        # Memory.
        self.memory_pool = MPSMemoryPool(
            num_blocks=NUM_GPU_BLOCKS,
            block_size=BLOCK_SIZE,
            n_heads=N_HEADS,
            d_k=D_K,
            dtype=dtype,
            device=device,
        )
        self.kv_cache_manager = KVCacheManager(
            num_gpu_blocks=NUM_GPU_BLOCKS,
            num_cpu_blocks=NUM_CPU_BLOCKS,
        )

        # Model.
        self.model = PagedAttention(
            kv_cache_manager=self.kv_cache_manager,
            memory_pool=self.memory_pool,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            d_k=D_K,
            block_size=BLOCK_SIZE,
            device=str(device),
            dtype=dtype,
        )

        # Engine components.
        self.tokenizer = SimpleTokenizer(vocab_size=VOCAB_SIZE)
        self.sampler = Sampler()
        self.scheduler = Scheduler(
            kv_cache_manager=self.kv_cache_manager,
            max_batch_size=MAX_BATCH_SIZE,
            max_tokens_per_step=MAX_TOKENS_PER_STEP,
        )
        self.model_runner = ModelRunner(
            model=self.model,
            sampler=self.sampler,
            kv_cache_manager=self.kv_cache_manager,
            memory_pool=self.memory_pool,
            tokenizer=self.tokenizer,
            device=str(device),
            dtype=dtype,
        )

        self._request_counter = 0
        self._outputs: dict[int, RequestOutput] = {}
        self._seq_to_request: dict[int, int] = {}  # seq_id → request_id
        self._seq_to_params: dict[int, SamplingParams] = {}

        # Stats.
        self.total_tokens_generated = 0
        self.start_time = time.time()

        # Adapter for real HuggingFace models (None when using toy model).
        self.adapter = None

        logger.info(
            "LLMEngine: ready, pool=%.2f MB, GPU blocks=%d, CPU blocks=%d",
            self.memory_pool.get_memory_mb(),
            NUM_GPU_BLOCKS,
            NUM_CPU_BLOCKS,
        )

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        sampling_params: SamplingParams | None = None,
    ) -> "LLMEngine":
        """Load a real HuggingFace model into the engine.

        Args:
            model_name:      HuggingFace model identifier.
            sampling_params: Default sampling parameters.

        Returns:
            A fully initialised :class:`LLMEngine` with the adapter loaded.
        """
        from vllm_mps.core.model_config import ModelConfig
        from vllm_mps.models.auto_adapter import AutoAdapter

        model_config = ModelConfig.from_pretrained(model_name)
        kv_cfg = model_config.get_kv_cache_config()

        memory_pool = MPSMemoryPool(
            num_blocks=NUM_GPU_BLOCKS,
            block_size=kv_cfg["block_size"],
            n_heads=kv_cfg["n_heads"],
            d_k=kv_cfg["d_k"],
            dtype=kv_cfg["dtype"],
            device=kv_cfg["device"],
        )
        kv_cache_manager = KVCacheManager(NUM_GPU_BLOCKS, NUM_CPU_BLOCKS)

        engine = cls.__new__(cls)
        engine.memory_pool = memory_pool
        engine.kv_cache_manager = kv_cache_manager
        engine.scheduler = Scheduler(kv_cache_manager)
        engine.sampler = Sampler()
        engine.model = None
        engine.model_runner = None
        engine._request_counter = 0
        engine._outputs = {}
        engine._seq_to_request = {}
        engine._seq_to_params = {}
        engine.total_tokens_generated = 0
        engine.start_time = time.time()

        # Load real model via adapter.
        engine.adapter = AutoAdapter.from_pretrained(
            model_name, kv_cache_manager, memory_pool
        )
        engine.tokenizer = engine.adapter.get_tokenizer()
        engine.model_config = model_config

        logger.info(
            "LLMEngine.from_pretrained: %s loaded, pool=%.2f MB",
            model_name, memory_pool.get_memory_mb(),
        )

        # Warmup MPS to compile Metal shaders before first request.
        engine._warmup_mps()

        return engine

    def _warmup_mps(self) -> None:
        """Run a dummy forward pass to trigger Metal shader compilation.

        On MPS, the first forward compiles Metal shaders for every unique
        kernel shape.  This can take 3–8 s and should not be charged to
        the first user request.
        """
        if self.adapter is None:
            return
        device_type = str(getattr(self, "model_config", None) and self.model_config.device or "cpu")
        if "mps" not in device_type:
            logger.info("_warmup_mps: skipped (device is CPU)")
            return

        import time as _time
        logger.info("_warmup_mps: starting shader warmup …")
        t0 = _time.perf_counter()

        # Allocate a temporary sequence for warmup.
        dummy_seq_id = -1
        self.kv_cache_manager.allocate(dummy_seq_id)
        self.kv_cache_manager.append_slot(dummy_seq_id)

        # Use BOS token (or 1) as a dummy input.
        dummy_token = 1
        with torch.no_grad():
            self.adapter.forward_single_token(dummy_token, 0, dummy_seq_id)

        torch.mps.synchronize()

        # Free the dummy sequence.
        self.kv_cache_manager.free(dummy_seq_id)

        dt = _time.perf_counter() - t0
        logger.info(
            "_warmup_mps: complete in %.2fs — Metal shaders compiled", dt
        )

    # ── Request submission ────────────────────────────────────────────────

    def add_request(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
    ) -> int:
        """Tokenise and enqueue a generation request.

        Args:
            prompt:          Input text.
            sampling_params: Optional sampling configuration.

        Returns:
            The request ID.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        request_id = self._request_counter
        self._request_counter += 1

        if self.adapter is not None:
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = self.tokenizer.encode(prompt)
        seq_id = request_id  # 1:1 mapping for simplicity.
        seq = Sequence(seq_id, prompt, token_ids, sampling_params)
        group = SequenceGroup(request_id, [seq], sampling_params)

        self.scheduler.add_request(group)
        self._outputs[request_id] = RequestOutput(
            request_id=request_id, prompt=prompt
        )
        self._seq_to_request[seq_id] = request_id
        self._seq_to_params[seq_id] = sampling_params

        return request_id

    # ── Generation loop ───────────────────────────────────────────────────

    def step(self) -> list[RequestOutput]:
        """Execute one generation step across all active requests.

        Returns:
            List of updated :class:`RequestOutput` objects.
        """
        # 1. Schedule.
        sched_out = self.scheduler.schedule()
        if not sched_out.scheduled_groups:
            return []

        if self.adapter is not None:
            return self._step_adapter(sched_out)
        return self._step_toy(sched_out)

    def _step_toy(self, sched_out) -> list[RequestOutput]:
        """Step using the toy ModelRunner path."""
        inputs = self.model_runner.prepare_inputs(sched_out.scheduled_groups)
        if not inputs.seq_ids:
            return []

        original_find = self.model_runner._find_sampling_params
        def _patched_find(seq_id):
            return self._seq_to_params.get(seq_id, SamplingParams())
        self.model_runner._find_sampling_params = _patched_find
        next_tokens = self.model_runner.execute_model(inputs)
        self.model_runner._find_sampling_params = original_find

        return self._update_sequences(inputs.seq_ids, next_tokens)

    def _step_adapter(self, sched_out) -> list[RequestOutput]:
        """Step using the batched HuggingFace adapter path."""
        seq_ids: list[int] = []
        token_ids: list[int] = []
        positions: list[int] = []
        seq_lens: list[int] = []

        for group in sched_out.scheduled_groups:
            for seq in group.get_seqs(SequenceStatus.RUNNING):
                seq_ids.append(seq.seq_id)
                token_ids.append(seq.get_last_token_id())
                pos = seq.get_total_len() - 1
                positions.append(pos)
                seq_lens.append(seq.get_total_len())

        if not seq_ids:
            return []

        # Single batched forward pass for all B sequences.
        logits = self.adapter.forward_batch(
            token_ids, seq_ids, positions, seq_lens,
        )  # (B, vocab_size)

        # Sample next token for each sequence.
        next_tokens: list[int] = []
        for i, seq_id in enumerate(seq_ids):
            params = self._seq_to_params.get(seq_id, SamplingParams())
            token = self.sampler.sample(logits[i], params)
            next_tokens.append(token)

        return self._update_sequences(seq_ids, next_tokens)

    def _update_sequences(
        self, seq_ids: list[int], next_tokens: list[int]
    ) -> list[RequestOutput]:
        """Update sequence state after a forward step."""
        updated: list[RequestOutput] = []
        for i, seq_id in enumerate(seq_ids):
            token_id = next_tokens[i]
            request_id = self._seq_to_request.get(seq_id)
            if request_id is None:
                continue

            seq = self._find_seq(seq_id)
            if seq is None:
                continue

            seq.add_token(token_id)
            self.total_tokens_generated += 1

            if seq.is_finished():
                self.scheduler.mark_finished(seq_id)
                # Evict cached tensors for this sequence from all layers.
                if self.adapter is not None:
                    for pl in self.adapter.paged_layers:
                        pl.evict_sequence(seq_id)

            out = self._outputs[request_id]
            out.token_ids = list(seq._output_token_ids)
            if self.adapter is not None:
                out.generated_text = self.tokenizer.decode(
                    seq._output_token_ids, skip_special_tokens=True
                )
            else:
                out.generated_text = self.tokenizer.decode(seq._output_token_ids)
            out.finished = seq.is_finished()
            out.num_steps += 1
            updated.append(out)

        return updated

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 1.0,
    ) -> str:
        """Convenience: submit one request and run until finished.

        Args:
            prompt:      Input text.
            max_tokens:  Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The generated text.
        """
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        req_id = self.add_request(prompt, params)

        step_count = 0
        while not self._outputs[req_id].finished:
            self.step()
            step_count += 1
            if step_count % 10 == 0:
                logger.info(
                    "LLMEngine: generate progress — %d steps, %d tokens",
                    step_count,
                    len(self._outputs[req_id].token_ids),
                )

        return self._outputs[req_id].generated_text

    # ── Queries ───────────────────────────────────────────────────────────

    def is_finished(self) -> bool:
        """True if no work remains."""
        return (
            self.scheduler.get_num_waiting() == 0
            and self.scheduler.get_num_running() == 0
            and self.scheduler.get_num_preempted() == 0
        )

    def get_elapsed(self) -> float:
        """Return wall-clock time since engine creation."""
        return time.time() - self.start_time

    def get_tokens_per_sec(self) -> float:
        """Return average generation throughput."""
        elapsed = self.get_elapsed()
        if elapsed == 0:
            return 0.0
        return self.total_tokens_generated / elapsed

    # ── Internal ──────────────────────────────────────────────────────────

    def _find_seq(self, seq_id: int) -> Sequence | None:
        """Find a Sequence object across all scheduler queues."""
        for group in self.scheduler._running:
            for s in group.sequences:
                if s.seq_id == seq_id:
                    return s
        for group in self.scheduler._waiting:
            for s in group.sequences:
                if s.seq_id == seq_id:
                    return s
        return None
