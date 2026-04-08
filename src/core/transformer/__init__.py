"""AetherAgent Transformer — Fault-tolerant modular architecture.

🔄 REGENERATION PROMPT FOR AI TOOLS:
"Create a fault-tolerant public API module for AetherAgent transformer that:
1. Implements _safe_import() utility for graceful component loading with fallbacks
2. Exports all transformer components (Config, Tokenizer, Embedding, FFN, Attention, Block, Model, Generator)
3. Provides get_model() factory function that returns Transformer or _FallbackModule on failure
4. Re-exports Tokenizer as public alias for backward compatibility
5. Exports TokenGenerator for async streaming inference
6. Logs INFO when components load successfully, WARNING when fallbacks activate
7. Uses Python 3.12+ type hints and clear docstrings
8. Ensures system keeps running even if any single component file has bugs
Follow the pattern: import with _safe_import → log result → export in __all__"

📦 DEPENDENCIES: logging, importlib, torch (for TYPE_CHECKING only)
🔧 CONFIG: No direct config dependency — imports from sibling modules
🔧 NOTE: _FallbackModule must implement same interface as real components
"""

import logging
import importlib
import torch
from typing import TYPE_CHECKING, Any, Iterator, List

if TYPE_CHECKING:
    from .config import TransformerConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class _FallbackModule:
    """Fallback that passes data through unchanged. Never fails.
    
    This class implements the minimal interface expected by the transformer pipeline:
    - __call__(): Returns input unchanged (identity function)
    - forward(): PyTorch-compatible forward pass (identity)
    - generate(): Yields random valid tokens for fallback generation
    - num_parameters: Property returning 0 for size reporting
    - encode/decode: Tokenizer interface for fallback text handling
    
    Used when any real component fails to import or initialize.
    """
    
    def __init__(self, *args, **kwargs):
        """Accept any arguments to match real component signatures."""
        logger.warning(f"⚠️ _FallbackModule activated with args={args}, kwargs={kwargs}")
        
    def __call__(self, x: Any, *args, **kwargs) -> Any:
        """Identity function: return input unchanged."""
        return x
        
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """PyTorch-compatible forward pass: identity function."""
        return x
        
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: int = 50) -> Iterator[int]:
        """Fallback generation: yield random valid tokens (0-259 for CharTokenizer)."""
        vocab_size = 260  # CharTokenizer default
        for _ in range(max_new_tokens):
            yield torch.randint(0, vocab_size, (1,)).item()
            
    @property
    def num_parameters(self) -> int:
        """Report 0 parameters for fallback modules."""
        return 0
        
    # Tokenizer interface fallbacks
    def encode(self, text: str) -> List[int]:
        """Fallback encode: return dummy token IDs."""
        return [1] * len(text)  # Return unk_id for each char
        
    def decode(self, token_ids: List[int]) -> str:
        """Fallback decode: return empty string."""
        return ""
        
    def decode_stream(self, token_ids: List[int]) -> Iterator[str]:
        """Fallback decode_stream: yield empty strings."""
        for _ in token_ids:
            yield ""
            
    def batch_encode(self, texts: List[str], max_len: int = 512) -> torch.Tensor:
        """Fallback batch_encode: return zero tensor."""
        if not texts:
            return torch.zeros((0, max_len), dtype=torch.long)
        return torch.full((len(texts), max_len), 1, dtype=torch.long)  # unk_id
        
    @property
    def vocab_size(self) -> int:
        """Report default vocab size."""
        return 260
        
    @property
    def pad_id(self) -> int:
        """Report pad token ID."""
        return 0
        
    @property
    def unk_id(self) -> int:
        """Report unknown token ID."""
        return 1
        
    @property
    def bos_id(self) -> int:
        """Report beginning-of-sequence token ID."""
        return 2
        
    @property
    def eos_id(self) -> int:
        """Report end-of-sequence token ID."""
        return 3


class _FallbackGenerator:
    """Fallback async generator: yields safe random tokens. Never fails.
    
    Used when TokenGenerator fails. Matches the async generator interface.
    """
    
    def __init__(self, vocab_size: int = 260, eos_id: int = 3, *args, **kwargs):
        """Initialize with same parameters as TokenGenerator."""
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        logger.warning(f"⚠️ _FallbackGenerator activated: vocab={vocab_size}")
        
    async def generate_stream(self, input_ids: List[int], max_tokens: int = 200,
                             temperature: float = 0.7, top_k: int = 50) -> Any:
        """Yield safe random tokens as fallback."""
        import asyncio
        # Echo first few input tokens
        for tid in input_ids[:min(10, len(input_ids), max_tokens)]:
            yield min(max(0, tid), self.vocab_size - 1)
            await asyncio.sleep(0.01)
        # Then random valid tokens
        for _ in range(max_tokens - min(10, len(input_ids))):
            yield torch.randint(0, self.vocab_size, (1,)).item()
            await asyncio.sleep(0.01)


def _safe_import(module_path: str, class_name: str, 
                 fallback: type = _FallbackModule) -> type:
    """Safely import a class from a module path, returning fallback on any error.
    
    This is the core fault-tolerance mechanism: if ANY component file has a bug,
    the system keeps running by substituting the fallback implementation.
    
    Args:
        module_path: Relative module path (e.g., '.config', '.model', '.generate')
        class_name: Name of the class to import (e.g., 'TransformerConfig')
        fallback: Fallback class to return on import failure (default: _FallbackModule)
        
    Returns:
        The imported class, or fallback class if import fails
    """
    try:
        # Import relative to this module's package
        module = importlib.import_module(module_path, package='src.core.transformer')
        cls = getattr(module, class_name)
        logger.info(f"✅ Loaded {module_path}.{class_name}")
        return cls
    except ImportError as e:
        logger.warning(f"⚠️ Fallback activated for {module_path}.{class_name}: {e}")
        return fallback
    except AttributeError as e:
        logger.warning(f"⚠️ Fallback activated for {module_path}.{class_name}: {e}")
        return fallback
    except Exception as e:
        logger.error(f"❌ Critical error importing {module_path}.{class_name}: {e}")
        return fallback


# ─── Level 0: Standalone modules (no internal imports) ──
# Config and Tokenizer are foundational
TransformerConfig = _safe_import('.config', 'TransformerConfig')
CharTokenizer = _safe_import('.tokenization', 'CharTokenizer')
Tokenizer = CharTokenizer  # Public alias for backward compatibility

# ─── Level 1: Core Components (depend on Level 0) ──
# Embeddings and normalization
TokenEmbedding = _safe_import('.embedding', 'TokenEmbedding')
RMSNorm = _safe_import('.ffn', 'RMSNorm')
FeedForward = _safe_import('.ffn', 'FeedForward')

# Attention mechanisms
RotaryEmbedding = _safe_import('.attention', 'RotaryEmbedding')
MultiHeadAttention = _safe_import('.attention', 'MultiHeadAttention')

# ─── Level 2: Composite Components (depend on Level 1) ──
TransformerBlock = _safe_import('.block', 'TransformerBlock')

# ─── Level 3: Full Model (depends on all above) ──
Transformer = _safe_import('.model', 'Transformer')

# ─── Level 4: Generation & Sampling (depend on Model) ──
TokenGenerator = _safe_import('.generate', 'TokenGenerator', fallback=_FallbackGenerator)

# Sampling utilities (standalone, no model dependency)
sample_top_p = _safe_import('.sampling', 'sample_top_p')
sample_temperature = _safe_import('.sampling', 'sample_temperature')
sample_top_k = _safe_import('.sampling', 'sample_top_k')
apply_sampling = _safe_import('.sampling', 'apply_sampling')


def get_model(config: 'TransformerConfig' = None) -> Any:
    """Get the Transformer model instance. Falls back to _FallbackModule if anything fails.
    
    This is the main entry point for obtaining a working model:
    - If config is None, creates default TransformerConfig(vocab_size=260)
    - Attempts to instantiate Transformer(config)
    - If any import or initialization fails, returns _FallbackModule instance
    - Logs parameter count and fallback status for debugging
    
    Args:
        config: Optional TransformerConfig; creates default if None
        
    Returns:
        Transformer instance or _FallbackModule instance
    """
    if config is None:
        # Use safe import to get config class, then instantiate with defaults
        ConfigCls = _safe_import('.config', 'TransformerConfig')
        config = ConfigCls()
    
    # Attempt to get and instantiate Transformer
    ModelCls = _safe_import('.model', 'Transformer')
    
    try:
        model_instance = ModelCls(config)
        is_fallback = ModelCls is _FallbackModule
        params = model_instance.num_parameters
        logger.info(f"Model loaded. Params: {params:,} (fallback={is_fallback})")
        return model_instance
    except Exception as e:
        logger.error(f"Model instantiation failed: {e}; returning fallback")
        return _FallbackModule()


# ─── Public API Exports ──
__all__ = [
    # === Configuration ===
    'TransformerConfig',
    
    # === Tokenization ===
    'Tokenizer',  # Alias for CharTokenizer
    'CharTokenizer',
    
    # === Core Components ===
    'TokenEmbedding',
    'RMSNorm',
    'FeedForward',
    'RotaryEmbedding',
    'MultiHeadAttention',
    'TransformerBlock',
    
    # === Full Model ===
    'Transformer',
    'get_model',
    
    # === Generation ===
    'TokenGenerator',
    
    # === Sampling Utilities ===
    'sample_top_p',
    'sample_temperature',
    'sample_top_k',
    'apply_sampling',
    
    # === Utilities ===
    '_safe_import',
    '_FallbackModule',
    '_FallbackGenerator',
]

# Final system status log (only if not in import-time testing)
if __name__ != '__main__':
    try:
        temp_config = TransformerConfig()
        temp_model = get_model(temp_config)
        logger.info(
            f"Transformer system ready. "
            f"Params: {temp_model.num_parameters:,} "
            f"Config: vocab={temp_config.vocab_size}, dim={temp_config.dim}"
        )
    except Exception as e:
        logger.error(f"Critical failure in transformer system initialization: {e}")