"""Minimal word-level tokenizer for vllm_mps experiments.

No external dependencies — splits on whitespace and maps words to IDs
using a generated pseudo-vocabulary.
"""

from __future__ import annotations

import logging

from vllm_mps.config import VOCAB_SIZE

logger = logging.getLogger(__name__)

# Special token IDs.
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class SimpleTokenizer:
    """Word-level tokenizer with a generated pseudo-vocabulary.

    Attributes:
        vocab:         Word → token ID mapping.
        reverse_vocab: Token ID → word mapping.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE) -> None:
        """Build the vocabulary.

        IDs 0–3 are reserved for PAD, UNK, BOS, EOS.  Remaining IDs
        are filled with ``word_4``, ``word_5``, etc.
        """
        self.vocab: dict[str, int] = {
            "<pad>": PAD_ID,
            "<unk>": UNK_ID,
            "<bos>": BOS_ID,
            "<eos>": EOS_ID,
        }
        for i in range(4, vocab_size):
            self.vocab[f"word_{i}"] = i

        self.reverse_vocab: dict[int, str] = {v: k for k, v in self.vocab.items()}

        logger.info("SimpleTokenizer: vocab_size=%d", vocab_size)

    # ── Encode / Decode ───────────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        """Tokenise *text* into a list of token IDs.

        Prepends BOS.  Unknown words map to UNK.
        """
        tokens = [BOS_ID]
        for word in text.strip().split():
            tokens.append(self.vocab.get(word, UNK_ID))
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back to a string.

        Skips PAD and BOS.  Stops at EOS.
        """
        words: list[str] = []
        for tid in token_ids:
            if tid == PAD_ID or tid == BOS_ID:
                continue
            if tid == EOS_ID:
                break
            words.append(self.reverse_vocab.get(tid, "<unk>"))
        return " ".join(words)

    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.vocab)
