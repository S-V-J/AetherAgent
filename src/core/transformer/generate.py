"""Async text generation utilities for streaming output.

🔄 REGENERATION PROMPT FOR AI TOOLS:
"Create a fault-tolerant async generator module for AetherAgent transformer that:
1. Implements TokenGenerator class with async generate_stream() yielding one token ID at a time
2. Clamps ALL sampled tokens to [0, vocab_size-1] BEFORE yielding to prevent CUDA OOB errors
3. Supports temperature scaling and top-k filtering for sampling diversity control
4. Handles CUDA errors by catching torch.cuda.CudaError and falling back to CPU random tokens
5. Provides _FallbackGenerator class that yields safe tokens when main generator fails
6. Uses PyTorch with @torch.no_grad() and async/await for non-blocking SSE response flushing
7. Has clear docstrings explaining the autoregressive generation loop and error recovery
8. Logs warnings when fallback is activated and errors with full stack traces
Use Python 3.12+, torch>=2.0, asyncio, and ensure vocab_size matches tokenizer (260 for CharTokenizer)."

📦 DEPENDENCIES: torch, torch.nn.functional, asyncio, logging, typing
🔧 CONFIG: vocab_size MUST match tokenizer (260 for CharTokenizer, 8000 for future BPE)
🔧 NOTE: All token sampling must clamp to [0, vocab_size-1] BEFORE appending to sequence
"""

import torch
import torch.nn.functional as F
import asyncio
import logging
from typing import AsyncGenerator, List, Optional, Union

logger = logging.getLogger(__name__)


class TokenGenerator:
    """Generates tokens from a trained transformer model with async streaming support.
    
    Key principles:
    - Autoregressive: each token depends on all previous tokens
    - Async streaming: yields one token at a time for SSE character-by-character output
    - Fault-tolerant: CUDA errors trigger CPU fallback, never crashes the server
    - Vocab-safe: all tokens clamped to [0, vocab_size-1] to prevent embedding OOB
    
    Usage:
        generator = TokenGenerator(model, device='cuda', vocab_size=260)
        async for token_id in generator.generate_stream(input_ids, max_tokens=200):
            char = tokenizer.decode([token_id])
            yield char  # Send to SSE client
    """
    
    def __init__(self, model: torch.nn.Module, device: str, vocab_size: int = 260):
        """
        Args:
            model: Trained transformer model with forward() method returning logits
            device: 'cuda' or 'cpu' — where to run inference
            vocab_size: Vocabulary size for clamping (260 for CharTokenizer)
        """
        # Move model to device if it has a to() method
        self.model = model.to(device) if hasattr(model, 'to') else model
        self.device = device
        self.vocab_size = vocab_size  # Must match model's embedding layer size
        self.model.eval()  # Set to evaluation mode (disable dropout, etc.)
        logger.info(f"TokenGenerator: vocab={vocab_size}, device={device}")
        
    @torch.no_grad()  # Disable gradient computation for inference (saves memory)
    async def generate_stream(self, 
                              input_ids: List[int], 
                              max_tokens: int = 200,
                              temperature: float = 0.7,
                              top_k: int = 50) -> AsyncGenerator[int, None]:
        """Yield one token ID at a time for character-by-character streaming.
        
        Autoregressive generation loop:
        1. Convert input_ids to tensor on correct device
        2. For each step up to max_tokens:
           a. Forward pass: get logits for entire sequence
           b. Extract logits for last position only
           c. Apply temperature scaling: logits / temperature
           d. Optional top-k filtering: keep only k most likely tokens
           e. Sample from probability distribution via multinomial
           f. CLAMP sampled token to [0, vocab_size-1] (CRITICAL for CUDA safety)
           g. Check for EOS token (id=3) to stop early
           h. Yield the valid token ID
           i. Append clamped token to sequence for next iteration
           j. Async yield to allow SSE flushing between tokens
        3. On CUDA error: switch to CPU fallback generation
        4. On any error: fall back to echoing input tokens (clamped)
        
        Args:
            input_ids: Initial token sequence as list of integers [0, vocab_size-1]
            max_tokens: Maximum number of NEW tokens to generate (default: 200)
            temperature: Sampling temperature (higher = more diverse, >1.0)
            top_k: Only sample from top-k most likely tokens (0 = no filtering)
            
        Yields:
            int: Next token ID in [0, vocab_size-1], always clamped for safety
        """
        # FIX: Clamp input tokens BEFORE passing to model (prevents embedding OOB)
        input_ids = [min(max(0, tid), self.vocab_size - 1) for tid in input_ids]
        
        try:
            # Copy input to avoid modifying original list
            generated = input_ids.copy()
            
            # Create input tensor on model's device
            input_tensor = torch.tensor([generated], dtype=torch.long, device=self.device)
            
            for step in range(max_tokens):
                try:
                    # Forward pass through model: [batch=1, seq_len, vocab_size]
                    logits = self.model(input_tensor)
                    
                    # Handle logits shape variations (3D: [B,T,V] or 2D: [T,V])
                    if logits.dim() == 3:
                        next_logits = logits[0, -1, :]  # Extract last position: [V]
                    elif logits.dim() == 2:
                        next_logits = logits[-1, :]  # Extract last position: [V]
                    else:
                        logger.warning(f"Unexpected logits dim: {logits.dim()}, stopping generation")
                        break
                    
                    # Temperature scaling: divide by temp to control diversity
                    # Avoid division by zero with max(temperature, 1e-8)
                    next_logits = next_logits / max(temperature, 1e-8)
                    
                    # Top-k filtering: sample only from k most likely tokens
                    if top_k > 0:
                        k = min(top_k, next_logits.size(-1))  # Don't exceed vocab size
                        top_k_vals, top_k_idx = torch.topk(next_logits, k)
                        # Create probability distribution over top-k only
                        probs = torch.zeros_like(next_logits)
                        probs.scatter_(-1, top_k_idx, F.softmax(top_k_vals, dim=-1))
                        next_token = torch.multinomial(probs, num_samples=1).item()
                    else:
                        # Sample from full distribution
                        probs = F.softmax(next_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()
                    
                    # CRITICAL FIX: Clamp sampled token to valid vocab range [0, vocab_size-1]
                    # This prevents CUDA assertion errors from out-of-bounds embedding indices
                    token_id = min(max(0, int(next_token)), self.vocab_size - 1)
                    
                    # Append to generated sequence for next iteration
                    generated.append(token_id)
                    
                    # Check for end-of-sequence token to stop generation early
                    if token_id == 3:  # CharTokenizer eos_id
                        logger.debug(f"EOS token (3) reached at step {step}, stopping")
                        break
                    
                    # Yield the valid, clamped token ID
                    yield token_id
                    
                    # Update input tensor for next iteration with CLAMPED token
                    # This ensures the model never sees out-of-range indices
                    input_tensor = torch.tensor([generated], dtype=torch.long, device=self.device)
                    
                    # Async yield to event loop: allows SSE to flush between tokens
                    # This is what enables character-by-character streaming in the UI
                    await asyncio.sleep(0)
                    
                except torch.cuda.CudaError as cuda_err:
                    # CUDA error: switch to CPU fallback for remaining tokens
                    logger.error(f"CUDA error at step {step}: {cuda_err}")
                    logger.warning("Switching to CPU fallback generation for remaining tokens")
                    for _ in range(max_tokens - step):
                        yield torch.randint(0, self.vocab_size, (1,)).item()
                        await asyncio.sleep(0.01)  # Small delay for SSE responsiveness
                    break
                    
        except Exception as e:
            # Catch-all for any other errors: fall back to echoing input tokens
            logger.error(f"Generation loop error: {e}", exc_info=True)
            for tid in input_ids[:max_tokens]:
                yield min(max(0, tid), self.vocab_size - 1)
                await asyncio.sleep(0.01)


class _FallbackGenerator:
    """Fallback generator: yields safe random tokens. Never fails.
    
    Used when TokenGenerator fails to initialize or encounters unrecoverable errors.
    Maintains pipeline continuity by yielding tokens in the valid vocab range.
    This ensures the SSE stream never stops, even if the model is broken.
    """
    
    def __init__(self, vocab_size: int = 260, eos_id: int = 3):
        """Initialize with same parameters as TokenGenerator.generate_stream().
        
        Args:
            vocab_size: Vocabulary size for clamping (default: 260 for CharTokenizer)
            eos_id: End-of-sequence token ID (default: 3 for CharTokenizer)
        """
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        logger.warning(
            f"⚠️ Using _FallbackGenerator: vocab={vocab_size}, eos={eos_id}"
        )
    
    async def generate(self, input_ids: List[int], max_tokens: int = 200,
                      temperature: float = 0.7, top_k: int = 50) -> AsyncGenerator[int, None]:
        """Yield safe random tokens as fallback when main generator fails.
        
        Behavior:
        1. Echo first few input tokens (preserves user input context)
        2. Then yield random valid tokens in [0, vocab_size-1]
        3. Async yield between tokens for SSE responsiveness
        """
        # Clamp input tokens first
        input_ids = [min(max(0, tid), self.vocab_size - 1) for tid in input_ids]
        
        # Echo first few input tokens to preserve context
        echo_count = min(10, len(input_ids), max_tokens)
        for tid in input_ids[:echo_count]:
            yield tid
            await asyncio.sleep(0.01)
        
        # Then yield random valid tokens for remaining steps
        for _ in range(max_tokens - echo_count):
            yield torch.randint(0, self.vocab_size, (1,)).item()
            await asyncio.sleep(0.01)