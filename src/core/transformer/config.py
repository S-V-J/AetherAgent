"""Transformer hyperparameters — change these to scale the model.

🔄 REGENERATION PROMPT FOR AI TOOLS:
"Create a TransformerConfig dataclass for AetherAgent that:
1. Defines vocab_size (260 for CharTokenizer, auto-detected from checkpoint)
2. Includes dim, n_layers, n_heads, ffn_dim, max_seq_len, dropout, rope_base
3. Has head_dim property (dim // n_heads)
4. Has total_parameters property calculating exact param count
5. Has size_estimate_mb property for FP32 model size in megabytes
6. Supports reserved_vocab_size for future BPE upgrade (not used in model)
7. Is compatible with PyTorch, dataclasses, and checkpoint introspection
Use Python 3.12+, type hints, and clear docstrings."

📦 DEPENDENCIES: dataclasses, typing
🔧 CONFIG: vocab_size MUST match tokenizer (260 for CharTokenizer)
🔧 NOTE: reserved_vocab_size=8000 is for future expansion only
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TransformerConfig:
    """Configuration for AetherAgent's transformer model.
    
    Key principle: vocab_size must match the tokenizer's actual vocabulary.
    - CharTokenizer: vocab_size=260 (4 special + 256 printable ASCII)
    - Future BPE tokenizer: vocab_size up to 8000 (reserved_vocab_size)
    
    The model's embedding layers (tok_emb, lm_head) use vocab_size,
    NOT reserved_vocab_size. This prevents CUDA OOB errors.
    """
    
    # === Core Architecture ===
    # vocab_size: MUST match tokenizer's actual vocabulary size
    # For CharTokenizer: 260 = 4 special (<|pad|>, <|unk|>, <|bos|>, <|eos|>) + 256 printable
    # Auto-detected from checkpoint if available (see model_loader.py)
    vocab_size: int = 260
    
    # Model dimensions
    dim: int = 256           # Embedding dimension (model width)
    n_layers: int = 2        # Number of transformer blocks (model depth)
    n_heads: int = 4         # Attention heads (must divide dim evenly)
    ffn_dim: int = 1024      # Feed-forward network inner dimension
    
    # Sequence handling
    max_seq_len: int = 512   # Maximum context length (tokens)
    dropout: float = 0.1     # Dropout rate for regularization
    
    # RoPE (Rotary Position Embeddings)
    rope_base: float = 10000.0  # Base frequency for RoPE
    
    # === Future Expansion (NOT used in current model layers) ===
    # reserved_vocab_size: Slots reserved for future BPE tokenizer upgrade
    # Current model ONLY uses vocab_size (0 to vocab_size-1)
    # Embedding layers: tok_emb(vocab_size, dim), lm_head(dim, vocab_size)
    reserved_vocab_size: int = 8000
    
    def __post_init__(self):
        """Validate config after initialization."""
        # Ensure dim is divisible by n_heads for multi-head attention
        if self.dim % self.n_heads != 0:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
            )
        
        # Ensure vocab_size doesn't exceed reserved space
        if self.vocab_size > self.reserved_vocab_size:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) cannot exceed "
                f"reserved_vocab_size ({self.reserved_vocab_size})"
            )
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.dim // self.n_heads
    
    @property
    def total_parameters(self) -> int:
        """Calculate total trainable parameters for model size reporting.
        
        Formula (GPT-2 decoder-only):
        - tok_emb: vocab_size * dim
        - pos_emb: max_seq_len * dim  
        - Each transformer block:
          * 4 projections (Q,K,V,O): 4 * dim * dim
          * 2 FFN layers: 2 * dim * ffn_dim
          * 2 norms: 4 * dim (RMSNorm has no params, but keeping for estimate)
        - lm_head: dim * vocab_size
        """
        vocab = self.vocab_size  # Use actual vocab_size, NOT reserved
        
        # Embeddings
        tok_emb_params = vocab * self.dim
        pos_emb_params = self.max_seq_len * self.dim
        
        # Transformer blocks
        block_params = (
            4 * self.dim * self.dim +      # Q, K, V, O projections
            2 * self.dim * self.ffn_dim +  # FFN: up_proj + down_proj
            4 * self.dim                   # Norms (RMSNorm has no params, estimate)
        )
        total_block_params = self.n_layers * block_params
        
        # Output head
        lm_head_params = self.dim * vocab
        
        return tok_emb_params + pos_emb_params + total_block_params + lm_head_params
    
    @property
    def size_estimate_mb(self) -> float:
        """Estimate model size in megabytes (FP32 precision).
        
        Formula: params * 4 bytes (FP32) / 1,000,000 = MB
        For FP16: divide by 2. For INT8: divide by 4.
        """
        return (self.total_parameters * 4) / 1_000_000  # 4 bytes per FP32 param
    
    @classmethod
    def from_checkpoint(cls, checkpoint: dict, base_config: Optional['TransformerConfig'] = None) -> 'TransformerConfig':
        """Create config from checkpoint, preserving vocab_size if detected.
        
        Args:
            checkpoint: Loaded checkpoint dict (from torch.load)
            base_config: Optional base config to override with checkpoint values
            
        Returns:
            TransformerConfig with vocab_size matching checkpoint embeddings
        """
        # Start with defaults or provided base config
        config = base_config or cls()
        
        # Try to detect vocab_size from checkpoint
        ckpt_vocab_size = None
        
        # Method 1: From saved config
        if 'config' in checkpoint and hasattr(checkpoint['config'], 'vocab_size'):
            ckpt_vocab_size = checkpoint['config'].vocab_size
        
        # Method 2: From embedding weight shape
        elif 'model_state_dict' in checkpoint:
            emb_key = 'tok_emb.weight'
            if emb_key in checkpoint['model_state_dict']:
                emb_shape = checkpoint['model_state_dict'][emb_key].shape
                if len(emb_shape) >= 1:
                    ckpt_vocab_size = emb_shape[0]
        
        # Update vocab_size if detected and valid
        if ckpt_vocab_size and 0 < ckpt_vocab_size <= config.reserved_vocab_size:
            print(f"📦 Checkpoint vocab_size detected: {ckpt_vocab_size}")
            # Create new config with detected vocab_size
            config = cls(
                vocab_size=ckpt_vocab_size,
                dim=config.dim,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                ffn_dim=config.ffn_dim,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
                rope_base=config.rope_base,
                reserved_vocab_size=config.reserved_vocab_size
            )
        
        return config