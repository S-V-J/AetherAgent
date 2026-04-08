"""Multi-Head Self-Attention with Rotary Position Embedding (RoPE) from scratch.

🔄 REGENERATION PROMPT FOR AI TOOLS:
"Create a fault-tolerant Multi-Head Attention module for AetherAgent that:
1. Implements RotaryEmbedding (RoPE) for modern positional encoding
2. Implements MultiHeadAttention with parallel head processing via matmul
3. Uses pre-norm architecture compatible with GPT-2 decoder style
4. Includes dropout for regularization on attention weights and output projection
5. Provides _FallbackRotaryEmbedding and _FallbackMultiHeadAttention classes that never fail
6. Uses PyTorch nn.Module with proper device handling and type hints
7. Has clear docstrings explaining RoPE formula (cos/sin rotation) and attention scaling
8. Logs warnings when fallback classes are activated
Use Python 3.12+, torch>=2.0, and ensure head_dim = dim // n_heads."

📦 DEPENDENCIES: torch, torch.nn, torch.nn.functional, math, logging
🔧 CONFIG: No vocab_size dependency (operates on embeddings, not tokens)
🔧 NOTE: RoPE applies to Q/K only; V passes through unchanged
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input for RoPE application.
    
    Formula: rotate_half([x1, x2]) = [-x2, x1]
    This enables complex multiplication via real-valued tensor ops.
    
    Args:
        x: Input tensor of shape [..., head_dim]
        
    Returns:
        Rotated tensor of same shape
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE — Rotary Position Embedding for modern transformers.
    
    Key advantages over learned/sinusoidal positions:
    - Extrapolates better to longer sequences
    - Preserves relative position information naturally
    - Computationally efficient (pre-computed cos/sin tables)
    
    Formula: 
        For token at position m, head_dim d:
        RoPE(q_m, k_n) = q_m * e^(i*m*θ) · k_n * e^(i*n*θ)
        Implemented via real-valued rotation: [cos, -sin; sin, cos]
    
    Args:
        head_dim: Dimension per attention head (must be even)
        max_seq_len: Maximum sequence length for position table
        base: RoPE frequency base (default: 10000.0, higher = slower decay)
    """
    
    def __init__(self, head_dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        # Compute inverse frequencies: 1 / (base^(2i/head_dim)) for i in [0, head_dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        logger.debug(f"RotaryEmbedding: head_dim={head_dim}, max_seq={max_seq_len}, base={base}")
        
    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cos/sin tensors for given sequence length.
        
        Args:
            x: Input tensor of shape [batch, seq_len, n_heads, head_dim]
            seq_len: Actual sequence length to use (<= max_seq_len)
            
        Returns:
            Tuple of (cos, sin) tensors of shape [1, seq_len, 1, head_dim]
        """
        # Position indices [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        
        # Compute frequencies: outer product of positions and inv_freq
        # Shape: [seq_len, head_dim//2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Duplicate for real/imag parts: [seq_len, head_dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Compute cos/sin and reshape for broadcasting: [1, seq_len, 1, head_dim]
        cos = emb.cos()[None, :, None, :]
        sin = emb.sin()[None, :, None, :]
        
        return cos, sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, 
                         cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.
    
    Formula: 
        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
    
    Args:
        q: Query tensor [batch, n_heads, seq_len, head_dim]
        k: Key tensor [batch, n_heads, seq_len, head_dim]
        cos: Pre-computed cos tensor [1, seq_len, 1, head_dim]
        sin: Pre-computed sin tensor [1, seq_len, 1, head_dim]
        
    Returns:
        Tuple of (q_rotated, k_rotated) with same shapes as inputs
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class _FallbackRotaryEmbedding(nn.Module):
    """Fallback RoPE: returns identity cos/sin (no rotation). Never fails.
    
    Used when RotaryEmbedding fails to initialize. Returns cos=1, sin=0
    which results in no positional rotation (equivalent to no position encoding).
    """
    
    def __init__(self, head_dim: int, max_seq_len: int = 512, base: float = 10000.0, 
                 *args, **kwargs):
        """Initialize with same signature as RotaryEmbedding."""
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        logger.warning(f"⚠️ Using _FallbackRotaryEmbedding: head_dim={head_dim}")
        
    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return identity cos/sin (no rotation applied)."""
        batch, seq_len_actual, n_heads, head_dim = x.shape
        # cos=1, sin=0 means no rotation: q_rot = q*1 + rotate(q)*0 = q
        cos = torch.ones(1, seq_len_actual, 1, head_dim, device=x.device, dtype=torch.float32)
        sin = torch.zeros(1, seq_len_actual, 1, head_dim, device=x.device, dtype=torch.float32)
        return cos, sin


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with RoPE — processes all heads in parallel.
    
    Architecture (pre-norm GPT-2 style):
        1. Project input to Q, K, V: Linear(dim → dim) × 3
        2. Reshape: [B, T, dim] → [B, T, n_heads, head_dim]
        3. Apply RoPE to Q and K only (V passes through unchanged)
        4. Transpose for attention: [B, T, H, D] → [B, H, T, D]
        5. Scaled dot-product: attn = softmax(Q@K^T / sqrt(D))
        6. Apply attention to V: out = attn @ V
        7. Combine heads and project: [B, H, T, D] → [B, T, dim]
    
    Why parallel matmul (not loop over heads):
    - 3-5x faster on modern GPUs via tensor cores
    - Better memory coalescing and cache utilization
    - Simpler code, easier to debug
    
    Args:
        dim: Model embedding dimension (input/output size)
        n_heads: Number of attention heads (must divide dim evenly)
        max_seq_len: Maximum sequence length for RoPE table
        dropout: Dropout probability for attention weights and output
    """
    
    def __init__(self, dim: int, n_heads: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.max_seq_len = max_seq_len
        
        # QKV projections (no bias for efficiency, following LLaMA/PaLM)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # RoPE for positional encoding
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        
        # Dropout for regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        logger.info(f"MultiHeadAttention: dim={dim}, heads={n_heads}, head_dim={self.head_dim}")
        
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply multi-head self-attention with RoPE.
        
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            seq_len: Actual sequence length (<= max_seq_len)
            
        Returns:
            Output tensor of shape [batch, seq_len, dim]
        """
        B, T, C = x.shape
        
        # QKV projections: [B, T, dim] → [B, T, dim] each
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head: [B, T, dim] → [B, T, n_heads, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)
        
        # Apply RoPE to Q and K only
        cos, sin = self.rope(x, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Transpose for attention: [B, T, H, D] → [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        # Q@K^T: [B, H, T, D] @ [B, H, D, T] → [B, H, T, T]
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) * scale
        
        # Softmax over sequence dimension + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values: [B, H, T, T] @ [B, H, T, D] → [B, H, T, D]
        out = attn_weights @ v
        
        # Transpose back and combine heads: [B, H, T, D] → [B, T, H, D] → [B, T, dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final projection + dropout
        return self.proj_dropout(self.out_proj(out))


class _FallbackMultiHeadAttention(nn.Module):
    """Fallback MultiHeadAttention: returns input unchanged. Never fails.
    
    Used when MultiHeadAttention fails to initialize. Maintains pipeline
    continuity by returning input with correct shape and dtype.
    """
    
    def __init__(self, dim: int, n_heads: int, max_seq_len: int = 512, 
                 dropout: float = 0.1, *args, **kwargs):
        """Initialize with same signature as MultiHeadAttention."""
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        logger.warning(f"⚠️ Using _FallbackMultiHeadAttention: dim={dim}, heads={n_heads}")
        
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Return input unchanged (identity function)."""
        return x