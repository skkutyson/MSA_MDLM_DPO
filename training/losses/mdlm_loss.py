"""
MDLM (Masked Diffusion Language Model) Loss.

Implements the SUBS parameterization loss for discrete diffusion:
L = E_t,x0 [ -log p_θ(x_0 | x_t, t) ]

The model directly predicts the original tokens x_0 given noisy input x_t.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def compute_diffusion_loss(
    model: nn.Module,
    x0: torch.Tensor,
    position_ids: torch.Tensor,
    block_position_ids: torch.Tensor,
    noise_schedule: nn.Module,
    mask_token_id: int,
    attention_mask: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    """
    Compute diffusion training loss.

    Args:
        model: Backbone model that takes (input_ids, sigma, position_ids, block_position_ids)
        x0: Original token IDs (batch, seq)
        position_ids: Sequence positions (batch, seq)
        block_position_ids: MSA block indices (batch, seq)
        noise_schedule: Noise schedule module
        mask_token_id: Token ID for masking
        attention_mask: Optional attention mask
        context_mask: Mask for context tokens (not masked during diffusion)
        loss_mask: Mask for loss computation
        eps: Minimum noise level

    Returns:
        Dictionary with 'loss' and metrics
    """
    batch_size, seq_len = x0.shape
    device = x0.device
    vocab_size = model.vocab_size if hasattr(model, 'vocab_size') else model.output_proj.out_features

    # Sample random timesteps t ~ U(eps, 1)
    t = torch.rand(batch_size, device=device) * (1 - eps) + eps
    sigma = noise_schedule(t)

    # Compute move_chance (probability of masking each token)
    move_chance = sigma.unsqueeze(-1)

    # Only mask non-context tokens
    if context_mask is not None:
        move_chance = move_chance * (1 - context_mask.float())

    # Apply noise: randomly mask tokens
    mask = torch.rand_like(x0.float()) < move_chance
    x_t = torch.where(mask, mask_token_id, x0)

    # Restore context tokens
    if context_mask is not None:
        x_t = torch.where(context_mask.bool(), x0, x_t)

    # Forward pass
    logits = model(
        input_ids=x_t,
        sigma=sigma,
        position_ids=position_ids,
        block_position_ids=block_position_ids,
        attention_mask=attention_mask,
    )

    # Cross-entropy loss
    log_probs = F.log_softmax(logits, dim=-1)
    nll = F.nll_loss(
        log_probs.view(-1, vocab_size),
        x0.view(-1),
        reduction='none',
    ).view(batch_size, seq_len)

    # Apply loss mask
    if loss_mask is None:
        # Default: compute loss on non-context tokens
        if context_mask is not None:
            loss_mask = 1 - context_mask.float()
        else:
            loss_mask = torch.ones_like(nll)

    # Also exclude padding
    if attention_mask is not None:
        loss_mask = loss_mask * attention_mask.float()

    # Compute masked loss
    loss = (nll * loss_mask).sum() / (loss_mask.sum() + 1e-8)

    # Compute accuracy
    preds = logits.argmax(dim=-1)
    correct = (preds == x0).float()
    accuracy = (correct * loss_mask).sum() / (loss_mask.sum() + 1e-8)

    # Compute mask accuracy (on masked positions only)
    masked_positions = mask.float() * loss_mask
    mask_accuracy = (correct * masked_positions).sum() / (masked_positions.sum() + 1e-8)

    return {
        'loss': loss,
        'nll': nll.mean(),
        'accuracy': accuracy,
        'mask_accuracy': mask_accuracy,
        'sigma_mean': sigma.mean(),
        'mask_fraction': mask.float().mean(),
    }


class MDLMLoss(nn.Module):
    """
    MDLM Training Loss Module.

    Wraps the diffusion loss computation with configurable parameters.
    """

    def __init__(
        self,
        noise_schedule: nn.Module,
        mask_token_id: int,
        eps: float = 1e-3,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize loss module.

        Args:
            noise_schedule: Noise schedule for diffusion
            mask_token_id: Token ID used for masking
            eps: Minimum noise level
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.noise_schedule = noise_schedule
        self.mask_token_id = mask_token_id
        self.eps = eps
        self.label_smoothing = label_smoothing

    def forward(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for a batch.

        Args:
            model: The backbone model
            batch: Dictionary with input_ids, position_ids, etc.

        Returns:
            Dictionary with loss and metrics
        """
        return compute_diffusion_loss(
            model=model,
            x0=batch['input_ids'],
            position_ids=batch['position_ids'],
            block_position_ids=batch['block_position_ids'],
            noise_schedule=self.noise_schedule,
            mask_token_id=self.mask_token_id,
            attention_mask=batch.get('attention_mask'),
            context_mask=batch.get('context_mask'),
            loss_mask=batch.get('loss_mask'),
            eps=self.eps,
        )

    def q_xt(
        self,
        x0: torch.Tensor,
        sigma: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample x_t from q(x_t | x_0) by masking.

        Args:
            x0: Original tokens
            sigma: Noise level (masking probability)
            context_mask: Positions to keep unmasked

        Returns:
            Noisy tokens x_t
        """
        move_chance = sigma.unsqueeze(-1)

        if context_mask is not None:
            move_chance = move_chance * (1 - context_mask.float())

        mask = torch.rand_like(x0.float()) < move_chance
        x_t = torch.where(mask, self.mask_token_id, x0)

        if context_mask is not None:
            x_t = torch.where(context_mask.bool(), x0, x_t)

        return x_t


class WeightedDiffusionLoss(nn.Module):
    """
    Diffusion loss with timestep-dependent weighting.

    Optionally applies importance weighting based on the
    signal-to-noise ratio or other criteria.
    """

    def __init__(
        self,
        noise_schedule: nn.Module,
        mask_token_id: int,
        weighting: str = 'uniform',
        eps: float = 1e-3,
    ):
        """
        Args:
            noise_schedule: Noise schedule
            mask_token_id: Mask token ID
            weighting: 'uniform', 'snr', or 'importance'
            eps: Minimum noise level
        """
        super().__init__()
        self.noise_schedule = noise_schedule
        self.mask_token_id = mask_token_id
        self.weighting = weighting
        self.eps = eps

    def _compute_weight(self, t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Compute per-sample loss weight based on timestep."""
        if self.weighting == 'uniform':
            return torch.ones_like(t)
        elif self.weighting == 'snr':
            # Weight by signal-to-noise ratio
            # SNR = (1-sigma)/sigma for masking diffusion
            snr = (1 - sigma) / (sigma + 1e-8)
            return snr
        elif self.weighting == 'importance':
            # Higher weight for intermediate timesteps
            return torch.sin(torch.pi * t) + 0.1
        else:
            return torch.ones_like(t)

    def forward(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted diffusion loss."""
        x0 = batch['input_ids']
        batch_size, seq_len = x0.shape
        device = x0.device

        # Sample timesteps
        t = torch.rand(batch_size, device=device) * (1 - self.eps) + self.eps
        sigma = self.noise_schedule(t)

        # Compute weights
        weights = self._compute_weight(t, sigma)

        # Get base loss
        result = compute_diffusion_loss(
            model=model,
            x0=x0,
            position_ids=batch['position_ids'],
            block_position_ids=batch['block_position_ids'],
            noise_schedule=self.noise_schedule,
            mask_token_id=self.mask_token_id,
            attention_mask=batch.get('attention_mask'),
            context_mask=batch.get('context_mask'),
            eps=self.eps,
        )

        # Apply weighting (note: per-sample loss not implemented, using mean weight)
        result['weighted_loss'] = result['loss'] * weights.mean()

        return result
