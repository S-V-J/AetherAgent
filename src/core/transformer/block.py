"""Transformer Decoder Block — pre-norm GPT-2 architecture.

🔄 REGENERATION PROMPT FOR AI TOOLS:
"Create a fault-tolerant TransformerBlock module for AetherAgent that:
1. Implements pre-norm GPT-2 decoder block: x + Attention(Norm(x)) + FFN(Norm(x))
2. Uses RMSNorm for normalization (no bias, more stable than LayerNorm)
3. Integrates MultiHeadAttention with RoPE and FeedForward with SwiGLU
4. Provides _FallbackTransformerBlock class that returns input unchanged on failure
5. Uses PyTorch nn.Module with proper device handling and type hints
6. Has clear docstrings explaining residual connections and pre-norm benefits
7. Logs warnings when fallback class is activated
Use Python 3.12+, torch>=2.0, and follow standard transformer block architecture."

📦 DEPENDENCIES: torch, torch.nn, .attention.MultiHeadAttention, .ffn.RMSNorm, .ffn.FeedForward
🔧 CONFIG: No vocab_size dependency (operates on embeddings, not tokens)
🔧 NOTE: Pre-norm (Norm before Attention/FFN) is more stable for deep networks
"""

import torch
import torch.nn as nn
import logging

from .attention import MultiHeadAttention
from .ffn import RMSNorm, FeedForward

logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """One transformer decoder block with pre-norm architecture.
    
    Architecture (GPT-2 style pre-norm):
        x₁ = x + Attention(RMSNorm(x))      # Residual + Self-Attention
        x₂ = x₁ + FFN(RMSNorm(x₁))          # Residual + Feed-Forward
        return x₂
    
    Why pre-norm (not post-norm):
    - More stable gradients for deep networks (>12 layers)
    - Faster convergence during training
    - Standard in modern LLMs (LLaMA, PaLM, Mistral)
    
    Why residual connections:
    - Enables training of very deep networks
    - Preserves gradient flow through skip connections
    - Allows model to learn identity mappings easily
    
    Args:
        dim: Model embedding dimension (input/output size)
        n_heads: Number of attention heads (must divide dim evenly)
        ffn_dim: Inner FFN dimension (typically 4x dim)
        max_seq_len: Maximum sequence length for RoPE table
        dropout: Dropout probability for regularization
    """
    
    def __init__(self, dim: int, n_heads: int, ffn_dim: int,
                 max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        
        # Pre-norm for attention branch
        self.attn_norm = RMSNorm(dim)
        self.attention = MultiHeadAttention(dim, n_heads, max_seq_len, dropout)
        
        # Pre-norm for FFN branch
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FeedForward(dim, ffn_dim, dropout)
        
        logger.info(f"TransformerBlock: dim={dim}, heads={n_heads}, ffn_dim={ffn_dim}")
        
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply transformer block with residual connections.
        
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            seq_len: Actual sequence length (<= max_seq_len)
            
        Returns:
            Output tensor of same shape [batch, seq_len, dim]
        """
        # Attention branch: x + Attention(Norm(x))
        x_attn = self.attention(self.attn_norm(x), seq_len=seq_len)
        x = x + x_attn  # Residual connection
        
        # FFN branch: x + FFN(Norm(x))
        x_ffn = self.ffn(self.ffn_norm(x))
        x = x + x_ffn  # Residual connection
        
        return x


class _FallbackTransformerBlock(nn.Module):
    """Fallback TransformerBlock: returns input unchanged. Never fails.
    
    Used when TransformerBlock fails to initialize. Maintains pipeline
    continuity by returning input with correct shape and dtype.
    This allows the system to keep running even if a component fails.
    """
    
    def __init__(self, dim: int, n_heads: int, ffn_dim: int,
                 max_seq_len: int = 512, dropout: float = 0.1,
                 *args, **kwargs):
        """Initialize with same signature as TransformerBlock."""
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        logger.warning(
            f"⚠️ Using _FallbackTransformerBlock: "
            f"dim={dim}, heads={n_heads}, ffn_dim={ffn_dim}"
        )
        
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Return input unchanged (identity function)."""
        return x