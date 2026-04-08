"""Token + Positional Embedding — converts token IDs to vector representations.

🔄 REGENERATION PROMPT FOR AI TOOLS:
"Create a fault-tolerant TokenEmbedding module for AetherAgent that:
1. Combines token embeddings and positional embeddings (sinusoidal or learned)
2. Uses vocab_size=260 default to match CharTokenizer (4 special + 256 printable)
3. Supports max_seq_len for position embedding range
4. Implements forward(x) returning token_emb + pos_emb with proper broadcasting
5. Includes _FallbackEmbedding class that passes input unchanged for fault tolerance
6. Uses PyTorch nn.Module with proper device handling
7. Has clear docstrings and type hints for Python 3.12+
Ensure vocab_size parameter is used for tok_emb, NOT reserved_vocab_size."

📦 DEPENDENCIES: torch, torch.nn
🔧 CONFIG: vocab_size MUST be 260 for CharTokenizer compatibility
🔧 NOTE: reserved_vocab_size=8000 is for future BPE, not used in embeddings
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class TokenEmbedding(nn.Module):
    """Token + Learned Positional Embedding for transformer input.
    
    Key principle: vocab_size must match the tokenizer's actual vocabulary.
    - CharTokenizer: vocab_size=260 (4 special + 256 printable ASCII)
    - Future BPE tokenizer: vocab_size up to 8000 (reserved_vocab_size)
    
    The tok_emb layer uses vocab_size, NOT reserved_vocab_size.
    This prevents CUDA OOB errors during embedding lookup.
    
    Positional embeddings use max_seq_len to support context windows up to 512 tokens.
    """
    
    def __init__(self, vocab_size: int = 260, dim: int = 256, max_seq_len: int = 512):
        """
        Args:
            vocab_size: Vocabulary size for token embeddings (default: 260 for CharTokenizer)
            dim: Embedding dimension (model width)
            max_seq_len: Maximum sequence length for position embeddings
        """
        super().__init__()
        # FIX: Use vocab_size (260) for tok_emb, NOT reserved_vocab_size (8000)
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.dim = dim
        self.max_seq_len = max_seq_len
        logger.info(f"TokenEmbedding: vocab={vocab_size}, dim={dim}, max_seq={max_seq_len}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embedded vectors with positional information.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len] with token IDs
            
        Returns:
            Embedded tensor of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len = x.shape
        
        # Ensure seq_len doesn't exceed max_seq_len
        if seq_len > self.max_seq_len:
            logger.warning(f"seq_len {seq_len} > max_seq_len {self.max_seq_len}, truncating")
            x = x[:, :self.max_seq_len]
            seq_len = self.max_seq_len
        
        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        
        # Get embeddings: [batch, seq_len, dim]
        token_embeddings = self.tok_emb(x)  # [batch, seq_len, dim]
        position_embeddings = self.pos_emb(positions)  # [1, seq_len, dim]
        
        # Combine with broadcasting: [batch, seq_len, dim] + [1, seq_len, dim]
        return token_embeddings + position_embeddings


class _FallbackEmbedding(nn.Module):
    """Fallback embedding: passes input through unchanged. Never fails.
    
    Used when TokenEmbedding fails to load. Returns zero-initialized embeddings
    of correct shape to maintain pipeline continuity.
    """
    
    def __init__(self, vocab_size: int = 260, dim: int = 256, max_seq_len: int = 512, *args, **kwargs):
        """Initialize fallback with same signature as TokenEmbedding."""
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        logger.warning(f"⚠️ Using _FallbackEmbedding: vocab={vocab_size}, dim={dim}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return zero-initialized embeddings of correct shape."""
        batch_size, seq_len = x.shape
        # Return zeros of shape [batch, seq_len, dim] to maintain pipeline
        return torch.zeros(batch_size, seq_len, self.dim, device=x.device, dtype=torch.float32)