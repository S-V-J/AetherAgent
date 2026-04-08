"""Character-level tokenizer — the foundational tokenizer."""

import string
import logging
import torch

logger = logging.getLogger(__name__)

_SPECIAL = {"<|pad|>": 0, "<|unk|>": 1, "<|bos|>": 2, "<|eos|>": 3}
_PRINTABLE = string.printable  # 100 chars: digits, letters, punctuation, whitespace
# Total vocab: 4 special + 100 printable = 104 tokens
# But we reserve 0-259 for character-level, 260-7999 for future BPE

# Build mapping for printable chars starting at index 4
_CHAR_TO_IDX = {ch: i for i, ch in enumerate(_PRINTABLE, start=len(_SPECIAL))}
_IDX_TO_CHAR = {i: ch for ch, i in _CHAR_TO_IDX.items()}

# Actual vocab size used by model
VOCAB_SIZE = 260  # Tokens 0-259 for char-level


class CharTokenizer:
    """Character-level tokenizer. Produces tokens 0-259."""

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def bos_id(self) -> int:
        return 2

    @property
    def eos_id(self) -> int:
        return 3

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs, clamping unknown chars to unk_id."""
        return [_CHAR_TO_IDX.get(ch, self.unk_id) for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text, skipping invalid IDs."""
        return "".join(_IDX_TO_CHAR.get(tid, "") for tid in token_ids if 0 <= tid < VOCAB_SIZE)

    def decode_stream(self, token_ids: list[int]):
        """Yields one decoded character at a time."""
        for tid in token_ids:
            if 0 <= tid < VOCAB_SIZE:
                yield _IDX_TO_CHAR.get(tid, "")

    def batch_encode(self, texts: list[str], max_len: int = 512) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, max_len), dtype=torch.long)
        encoded = [self.encode(t)[:max_len] for t in texts]
        max_real = max((len(e) for e in encoded) if encoded else 1)
        padded = torch.full((len(texts), max_real), self.pad_id, dtype=torch.long)
        for i, ids in enumerate(encoded):
            if len(ids) > 0:
                padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        return padded