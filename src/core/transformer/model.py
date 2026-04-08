# /// REGENERATION PROMPT ///
# "Create a GPT-2 decoder-only Transformer model with:
#  - vocab_size from config (default 260 for char-level, auto-detect from checkpoint)
#  - torch.clamp() on input tokens BEFORE embedding lookup to prevent CUDA OOB
#  - torch.clamp() on sampled tokens BEFORE appending to sequence
#  - CUDA error handling with fallback to random valid tokens
#  - Logging for initialization and errors
#  - Modular: imports from .block, .ffn only
#  - Property num_parameters for model size reporting
#  - generate() method with temperature/top-k sampling and error recovery"
# /// END PROMPT ///

"""Full Transformer Model — GPT-2 decoder-only architecture.
Trains from random weights or loads from checkpoint.
All token indices clamped to [0, vocab_size-1] to prevent CUDA assertion errors."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .block import TransformerBlock
from .ffn import RMSNorm

logger = logging.getLogger(__name__)


class Transformer(nn.Module):
    """GPT-2 decoder-only transformer. Modular, fault-tolerant, CUDA-safe."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        # FIX: Get vocab_size from config with safe default; checkpoint may override
        self.vocab_size = getattr(config, 'vocab_size', 260)
        
        # Embedding layers with correct vocab size
        self.tok_emb = nn.Embedding(self.vocab_size, config.dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.dim, config.n_heads, config.ffn_dim,
                config.max_seq_len, config.dropout
            ) for _ in range(config.n_layers)
        ])
        
        self.final_norm = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, self.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
        logger.info(f"Transformer: vocab={self.vocab_size}, dim={config.dim}, layers={config.n_layers}")

    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """Forward pass with input token clamping to prevent CUDA OOB."""
        actual_seq_len = x.shape[1]
        pos = torch.arange(actual_seq_len, device=x.device)
        
        # CRITICAL FIX: Clamp input tokens BEFORE embedding lookup
        # Prevents: "vectorized_gather_kernel: Assertion `ind >=0 && ind < ind_dim_size`"
        x_clamped = torch.clamp(x, 0, self.vocab_size - 1)
        
        x = self.tok_emb(x_clamped) + self.pos_emb(pos).unsqueeze(0)
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x, seq_len=actual_seq_len)
        
        x = self.final_norm(x)
        return self.lm_head(x)

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters for model size reporting."""
        return sum(p.numel() for p in self.parameters())

    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: int = 50):
        """Autoregressive generation with error recovery."""
        self.eval()
        for _ in range(max_new_tokens):
            try:
                idx_cond = idx[:, -self.config.max_seq_len:]
                logits = self(idx_cond, seq_len=idx_cond.shape[1])
                logits = logits[:, -1, :] / max(temperature, 1e-9)
                
                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k)
                    logits[logits < v[:, [-1]]] = -float('inf')
                
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # CRITICAL: Clamp sampled token BEFORE appending
                idx_next = torch.clamp(idx_next, 0, self.vocab_size - 1)
                idx = torch.cat((idx, idx_next), dim=1)
                yield idx_next.item()
                
            except torch.cuda.CudaError as e:
                logger.error(f"CUDA error: {e}; switching to fallback")
                for _ in range(max_new_tokens):
                    yield torch.randint(0, self.vocab_size, (1,)).item()
                break
            except Exception as e:
                logger.error(f"Generation error: {e}")
                break