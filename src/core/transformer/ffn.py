"""FeedForward Network — SwiGLU + RMSNorm — the standard modern FFN.

🔄 REGENERATION PROMPT FOR AI TOOLS:
"Create a fault-tolerant FeedForward module for AetherAgent that:
1. Implements RMSNorm (Root Mean Square Layer Normalization) with eps parameter
2. Implements SwiGLU FeedForward: swish(W1*x) * W3*x → W2*output
3. Includes dropout for regularization
4. Provides _FallbackRMSNorm and _FallbackFeedForward classes that never fail
5. Uses PyTorch nn.Module with proper device handling and type hints
6. Has clear docstrings explaining SwiGLU formula and RMSNorm stability
7. Logs warnings when fallback classes are activated
Use Python 3.12+, torch>=2.0, and follow GPT-2 pre-norm architecture."

📦 DEPENDENCIES: torch, torch.nn, torch.nn.functional
🔧 CONFIG: No vocab_size dependency (operates on embeddings, not tokens)
🔧 NOTE: RMSNorm has no learnable bias; FeedForward uses SwiGLU (not GeLU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization — more stable than LayerNorm.
    
    Formula: output = x * rsqrt(mean(x^2) + eps) * weight
    
    Key advantages over LayerNorm:
    - No mean subtraction (faster, same performance for transformers)
    - Single learnable scale parameter (weight)
    - More numerically stable for low-precision training
    
    Args:
        dim: Dimension to normalize (typically model embedding size)
        eps: Small constant for numerical stability (default: 1e-6)
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (no bias for RMSNorm)
        self.weight = nn.Parameter(torch.ones(dim))
        logger.debug(f"RMSNorm initialized: dim={dim}, eps={eps}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to input tensor.
        
        Args:
            x: Input tensor of shape [..., dim]
            
        Returns:
            Normalized tensor of same shape, scaled by learnable weight
        """
        # Compute RMS: sqrt(mean(x^2) + eps)
        # rsqrt = 1/sqrt for numerical stability
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Apply normalization and learned scale
        return x * rms * self.weight


class _FallbackRMSNorm(nn.Module):
    """Fallback RMSNorm: returns input unchanged. Never fails.
    
    Used when RMSNorm fails to initialize. Maintains pipeline continuity
    by returning input with correct shape and dtype.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6, *args, **kwargs):
        """Initialize with same signature as RMSNorm."""
        super().__init__()
        self.dim = dim
        self.eps = eps
        logger.warning(f"⚠️ Using _FallbackRMSNorm: dim={dim}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return input unchanged (identity function)."""
        return x


class FeedForward(nn.Module):
    """SwiGLU FeedForward — the standard modern FFN.
    
    Architecture (pre-norm GPT-2 style):
        x_norm = RMSNorm(x)
        gate = Swish(W1 @ x_norm)      # W1: dim → ffn_dim
        value = W3 @ x_norm            # W3: dim → ffn_dim  
        output = W2 @ (gate * value)   # W2: ffn_dim → dim
        return dropout(output)
    
    Why SwiGLU (not GeLU/ReLU):
    - Better gradient flow through multiplicative gating
    - Empirically superior performance in LLMs (PaLM, LLaMA)
    - Minimal compute overhead vs standard FFN
    
    Args:
        dim: Model embedding dimension (input/output size)
        ffn_dim: Inner FFN dimension (typically 4x dim)
        dropout: Dropout probability for regularization
    """
    
    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        # Pre-norm: normalize before FFN
        self.norm = RMSNorm(dim)
        
        # SwiGLU projections (no bias for efficiency)
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)  # Gate projection
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)  # Value projection  
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)  # Output projection
        
        self.dropout = nn.Dropout(dropout)
        logger.info(f"FeedForward: dim={dim}→{ffn_dim}→{dim}, dropout={dropout}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU FeedForward with pre-norm.
        
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            
        Returns:
            Output tensor of same shape [batch, seq_len, dim]
        """
        # Pre-norm
        x_normed = self.norm(x)
        
        # SwiGLU: swish(gate) * value
        gate = F.silu(self.w1(x_normed))  # Swish activation on gate
        value = self.w3(x_normed)          # Linear projection for value
        activated = gate * value           # Element-wise multiplication
        
        # Output projection + dropout
        output = self.w2(activated)
        return self.dropout(output)


class _FallbackFeedForward(nn.Module):
    """Fallback FeedForward: returns input unchanged. Never fails.
    
    Used when FeedForward fails to initialize. Maintains pipeline
    continuity by returning input with correct shape and dtype.
    """
    
    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.1, 
                 *args, **kwargs):
        """Initialize with same signature as FeedForward."""
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        logger.warning(f"⚠️ Using _FallbackFeedForward: {dim}→{ffn_dim}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return input unchanged (identity function)."""
        return x