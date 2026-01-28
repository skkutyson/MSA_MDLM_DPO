"""
Diffusion sampling strategies for MSAGPT-MDLM.

Provides strategy classes compatible with the existing CLI infrastructure,
but for diffusion-based generation instead of autoregressive.
"""

import torch
import numpy as np
from typing import Optional, List, Tuple, Any


class DiffusionSamplingStrategy:
    """
    Base class for diffusion sampling strategies.

    Unlike autoregressive strategies that operate token-by-token,
    diffusion strategies operate on the entire sequence at once
    through iterative denoising.
    """

    def __init__(
        self,
        num_steps: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        invalid_slices: Optional[List[int]] = None,
        sampler: str = 'ddpm_cache',
    ):
        """
        Args:
            num_steps: Number of diffusion denoising steps
            temperature: Sampling temperature
            top_k: Top-k filtering (0 or None to disable)
            top_p: Nucleus sampling threshold (1.0 or None to disable)
            invalid_slices: Token IDs that should never be generated
            sampler: Sampling method ('ddpm' or 'ddpm_cache')
        """
        self.num_steps = num_steps
        self.temperature = temperature
        self.top_k = top_k if top_k and top_k > 0 else None
        self.top_p = top_p if top_p and top_p < 1.0 else None
        self.invalid_slices = invalid_slices or []
        self.sampler = sampler
        self._is_done = False

    @property
    def is_done(self) -> bool:
        """Check if generation is complete."""
        return self._is_done

    def reset(self):
        """Reset strategy state for new generation."""
        self._is_done = False

    def generate(
        self,
        model,
        context_tokens: torch.Tensor,
        msa_len: int,
        max_gen_length: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate sequences using diffusion sampling.

        Args:
            model: MSAGPT_MDLM model
            context_tokens: Input context tokens (batch, context_len)
            msa_len: Length of each MSA sequence
            max_gen_length: Maximum total length including context

        Returns:
            Generated tokens (batch, total_len)
        """
        self.reset()

        # Call model's generate method
        output = model.generate(
            context_tokens=context_tokens,
            msa_len=msa_len,
            max_gen_length=max_gen_length,
            num_steps=self.num_steps,
            sampler=self.sampler,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            invalid_token_ids=self.invalid_slices,
            **kwargs
        )

        self._is_done = True
        return output

    def finalize(self, tokens: torch.Tensor, mems: Any = None) -> Tuple[torch.Tensor, Any]:
        """
        Finalize generation (for compatibility with AR interface).

        Args:
            tokens: Generated tokens
            mems: Memory state (unused for diffusion)

        Returns:
            Tuple of (tokens, None)
        """
        self.reset()
        return tokens, None


class DDPMStrategy(DiffusionSamplingStrategy):
    """
    Standard DDPM (Denoising Diffusion Probabilistic Model) sampling.

    At each step, predicts the clean sequence and re-masks according
    to the noise schedule.
    """

    def __init__(
        self,
        num_steps: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        invalid_slices: Optional[List[int]] = None,
    ):
        super().__init__(
            num_steps=num_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            invalid_slices=invalid_slices,
            sampler='ddpm',
        )


class DDPMCachingStrategy(DiffusionSamplingStrategy):
    """
    DDPM sampling with caching for efficiency.

    Instead of re-masking all predictions, keeps track of revealed
    positions and progressively reveals more based on confidence scores.
    """

    def __init__(
        self,
        num_steps: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        invalid_slices: Optional[List[int]] = None,
    ):
        super().__init__(
            num_steps=num_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            invalid_slices=invalid_slices,
            sampler='ddpm_cache',
        )


def get_diffusion_strategy(
    strategy_name: str,
    num_steps: int = 256,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    invalid_slices: Optional[List[int]] = None,
) -> DiffusionSamplingStrategy:
    """
    Factory function to create a diffusion sampling strategy.

    Args:
        strategy_name: Name of strategy ('ddpm' or 'ddpm_cache')
        num_steps: Number of diffusion steps
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold
        invalid_slices: Token IDs to never generate

    Returns:
        DiffusionSamplingStrategy instance
    """
    strategies = {
        'ddpm': DDPMStrategy,
        'ddpm_cache': DDPMCachingStrategy,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown diffusion strategy: {strategy_name}. "
                        f"Choose from {list(strategies.keys())}")

    return strategies[strategy_name](
        num_steps=num_steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        invalid_slices=invalid_slices,
    )
