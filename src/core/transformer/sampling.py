"""Advanced sampling strategies for better generation from small models.

🔄 REGENERATION PROMPT FOR AI TOOLS:
"Create fault-tolerant sampling utilities for AetherAgent transformer that:
1. Implements top-p (nucleus) sampling: keeps smallest token set with cumulative prob >= p
2. Implements temperature scaling: logits / temp for diversity control
3. Implements top-k filtering: sample only from k most likely tokens
4. Provides _Fallback sampling functions that return safe defaults on error
5. Uses PyTorch with proper tensor operations and type hints
6. Has clear docstrings explaining the math behind each sampling method
7. Logs warnings when fallback functions are used
Use Python 3.12+, torch>=2.0, and ensure all functions handle edge cases gracefully."

📦 DEPENDENCIES: torch, torch.nn.functional, logging
🔧 CONFIG: No vocab_size dependency — operates on logits (pre-softmax)
🔧 NOTE: All sampling functions return single token ID (int) for compatibility
"""

import torch
import torch.nn.functional as F
import logging
from typing import Union

logger = logging.getLogger(__name__)


def sample_top_p(
    logits: torch.Tensor,
    p: float = 0.9,
    filter_value: float = -float('Inf'),
    min_tokens_to_keep: int = 1
) -> int:
    """Nucleus (top-p) sampling — keeps smallest set of tokens whose cumulative prob >= p.
    
    Algorithm:
    1. Sort logits descending
    2. Compute cumulative probabilities via softmax
    3. Find cutoff where cumulative prob >= p
    4. Mask out tokens below cutoff
    5. Sample from remaining distribution
    
    Args:
        logits: Unnormalized log-probabilities of shape [vocab_size] or [1, vocab_size]
        p: Cumulative probability threshold (0.0 < p <= 1.0, typical: 0.9-0.95)
        filter_value: Value to assign to filtered logits (default: -inf)
        min_tokens_to_keep: Minimum number of tokens to keep regardless of p
        
    Returns:
        int: Sampled token ID in [0, vocab_size-1]
    """
    try:
        # Ensure 1D logits
        if logits.dim() == 2:
            logits = logits[0]
        
        # Sort descending
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Find tokens to remove: cumulative prob > p (but keep min_tokens)
        sorted_indices_to_remove = cumulative_probs > p
        
        # FIX: Handle edge case where min_keep >= length
        vocab_size = sorted_indices_to_remove.size(0)
        if min_tokens_to_keep >= vocab_size:
            # Keep all tokens
            sorted_indices_to_remove[:] = False
        else:
            # Shift right by 1, keeping first min_tokens
            # Use torch.roll or manual indexing to avoid size mismatch
            sorted_indices_to_remove[min_tokens_to_keep:] = sorted_indices_to_remove[min_tokens_to_keep-1:-1].clone()
            sorted_indices_to_remove[:min_tokens_to_keep] = False
        
        # Scatter removal mask back to original order
        indices_to_remove = sorted_indices_to_remove.scatter(
            0, sorted_indices, sorted_indices_to_remove
        )
        
        # Apply mask
        filtered_logits = logits.masked_fill(indices_to_remove, filter_value)
        
        # Sample from filtered distribution
        probs = F.softmax(filtered_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
        
    except Exception as e:
        logger.warning(f"⚠️ Top-p sampling failed: {e}; using fallback")
        return _fallback_sample_top_p(logits, p)


def _fallback_sample_top_p(logits: torch.Tensor, p: float = 0.9) -> int:
    """Fallback top-p sampling: returns argmax (greedy) on error."""
    try:
        if logits.dim() == 2:
            return torch.argmax(logits[0]).item()
        return torch.argmax(logits).item()
    except:
        # Ultimate fallback: random token in typical vocab range
        return torch.randint(0, 260, (1,)).item()


def sample_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0
) -> int:
    """Temperature scaling + categorical sampling.
    
    Formula:
        probs = softmax(logits / temperature)
        token ~ Multinomial(probs)
    
    Effects:
    - temperature < 1.0: sharper distribution (more deterministic)
    - temperature = 1.0: original distribution
    - temperature > 1.0: flatter distribution (more diverse)
    
    Args:
        logits: Unnormalized log-probabilities of shape [vocab_size] or [1, vocab_size]
        temperature: Scaling factor (> 0; typical: 0.7-1.2)
        
    Returns:
        int: Sampled token ID
    """
    try:
        # Handle 2D logits
        if logits.dim() == 2:
            logits = logits[0]
        
        # Edge case: very low temp → greedy
        if temperature < 1e-8:
            return torch.argmax(logits).item()
        
        # Scale and sample
        scaled = logits / temperature
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
        
    except Exception as e:
        logger.warning(f"⚠️ Temperature sampling failed: {e}; using fallback")
        return _fallback_sample_temperature(logits, temperature)


def _fallback_sample_temperature(logits: torch.Tensor, temperature: float = 1.0) -> int:
    """Fallback temperature sampling: returns argmax on error."""
    try:
        if logits.dim() == 2:
            return torch.argmax(logits[0]).item()
        return torch.argmax(logits).item()
    except:
        return torch.randint(0, 260, (1,)).item()


def sample_top_k(
    logits: torch.Tensor,
    k: int = 50,
    filter_value: float = -float('Inf')
) -> int:
    """Top-k sampling — sample only from k most likely tokens.
    
    Algorithm:
    1. Find k largest logits
    2. Mask out all others
    3. Sample from remaining distribution
    
    Args:
        logits: Unnormalized log-probabilities of shape [vocab_size] or [1, vocab_size]
        k: Number of top tokens to consider (typical: 40-100)
        filter_value: Value for filtered logits
        
    Returns:
        int: Sampled token ID
    """
    try:
        if logits.dim() == 2:
            logits = logits[0]
        
        # Handle edge case: k >= vocab_size
        vocab_size = logits.size(-1)
        k = min(k, vocab_size)
        
        # Find top-k indices
        top_k_vals, top_k_idx = torch.topk(logits, k)
        
        # Create mask: keep top-k, filter rest
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(0, top_k_idx, False)
        
        # Apply filter
        filtered_logits = logits.masked_fill(mask, filter_value)
        
        # Sample
        probs = F.softmax(filtered_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
        
    except Exception as e:
        logger.warning(f"⚠️ Top-k sampling failed: {e}; using fallback")
        return _fallback_sample_top_k(logits, k)


def _fallback_sample_top_k(logits: torch.Tensor, k: int = 50) -> int:
    """Fallback top-k sampling: returns argmax on error."""
    try:
        if logits.dim() == 2:
            return torch.argmax(logits[0]).item()
        return torch.argmax(logits).item()
    except:
        return torch.randint(0, 260, (1,)).item()


def apply_sampling(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0
) -> int:
    """Apply combined sampling strategy in standard order:
    1. Temperature scaling
    2. Top-k filtering (if k > 0)
    3. Top-p filtering (if p > 0)
    4. Categorical sampling
    
    Args:
        logits: Unnormalized log-probabilities
        temperature: Temperature for scaling (> 0)
        top_k: Top-k filter (0 = disabled)
        top_p: Top-p filter (0.0 = disabled)
        
    Returns:
        int: Sampled token ID
    """
    try:
        if logits.dim() == 2:
            logits = logits[0]
        
        # 1. Temperature
        if temperature >= 1e-8:
            logits = logits / temperature
        
        # 2. Top-k
        if top_k > 0:
            logits = _apply_top_k_filter(logits, top_k)
        
        # 3. Top-p
        if top_p > 0:
            logits = _apply_top_p_filter(logits, top_p)
        
        # 4. Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
        
    except Exception as e:
        logger.error(f"Combined sampling failed: {e}; using greedy fallback")
        if logits.dim() == 2:
            return torch.argmax(logits[0]).item()
        return torch.argmax(logits).item()


def _apply_top_k_filter(logits: torch.Tensor, k: int, filter_value: float = -float('Inf')) -> torch.Tensor:
    """Helper: apply top-k filter without sampling."""
    vocab_size = logits.size(-1)
    k = min(k, vocab_size)
    top_k_vals, top_k_idx = torch.topk(logits, k)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(0, top_k_idx, False)
    return logits.masked_fill(mask, filter_value)


def _apply_top_p_filter(logits: torch.Tensor, p: float, filter_value: float = -float('Inf'), min_keep: int = 1) -> torch.Tensor:
    """Helper: apply top-p filter without sampling.
    
    FIX: Handle edge case where min_keep >= vocab_size to avoid tensor size mismatch.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    
    # FIX: Handle edge case to avoid size mismatch in slicing
    vocab_size = sorted_indices_to_remove.size(0)
    if min_keep >= vocab_size:
        # Keep all tokens — remove nothing
        sorted_indices_to_remove[:] = False
    else:
        # Shift right by 1, keeping first min_keep tokens
        # Use torch.cat to avoid size mismatch: [False]*min_keep + original[:-1][min_keep:]
        keep_mask = torch.zeros_like(sorted_indices_to_remove, dtype=torch.bool)
        keep_mask[:min_keep] = True  # Always keep first min_keep
        # For rest, use cumulative prob threshold
        keep_mask[min_keep:] = ~sorted_indices_to_remove[min_keep:]
        sorted_indices_to_remove = ~keep_mask
    
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, filter_value)