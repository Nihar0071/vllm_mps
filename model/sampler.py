"""Token sampler with greedy, top-k, and nucleus (top-p) strategies."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm_mps.core.sequence import SamplingParams


class Sampler:
    """Samples the next token from logits using configurable strategies.

    Supports greedy (temperature=0), top-k filtering, and nucleus (top-p)
    sampling.
    """

    def sample(self, logits: torch.Tensor, sampling_params: SamplingParams) -> int:
        """Sample a single token from *logits*.

        Args:
            logits:          Shape ``(vocab_size,)``.
            sampling_params: Sampling configuration.

        Returns:
            The sampled token ID as a Python int.
        """
        temperature = sampling_params.temperature
        top_k = sampling_params.top_k
        top_p = sampling_params.top_p

        # Greedy.
        if temperature == 0.0:
            return int(torch.argmax(logits).item())

        # Apply temperature.
        logits = logits / temperature

        # Top-K filtering.
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, top_k).values[-1]
            logits = torch.where(logits < threshold, torch.tensor(float("-inf"), device=logits.device), logits)

        # Nucleus (top-p) filtering.
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Mask tokens where cumulative prob exceeds top_p.
            mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float("-inf")

            # Scatter back to original order.
            logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)

        # Sample from distribution.
        probs = F.softmax(logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1)
        return int(token_id.item())

    def sample_batch(
        self,
        logits_batch: torch.Tensor,
        sampling_params: SamplingParams,
    ) -> list[int]:
        """Sample one token per row in *logits_batch*.

        Args:
            logits_batch: Shape ``(batch_size, vocab_size)``.
            sampling_params: Shared sampling configuration.

        Returns:
            List of sampled token IDs.
        """
        return [
            self.sample(logits_batch[i], sampling_params)
            for i in range(logits_batch.size(0))
        ]
