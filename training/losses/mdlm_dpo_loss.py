"""
D3PO (Diffusion DPO) Loss for Masked Diffusion Language Models.

Adapts Direct Preference Optimization to discrete diffusion by computing
likelihood ratios via the ELBO. Based on the D3PO paper methodology.

Reference:
- D3PO: https://arxiv.org/abs/2503.08295
- DPO: https://arxiv.org/abs/2305.18290
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class D3POLoss(nn.Module):
    """
    D3PO (Discrete Diffusion Direct Preference Optimization) Loss.

    Computes DPO loss for masked diffusion models by:
    1. Sampling a shared timestep for winner/loser
    2. Computing log-likelihoods under policy and reference models
    3. Applying Bradley-Terry preference model
    """

    def __init__(
        self,
        noise_schedule: nn.Module,
        mask_token_id: int,
        beta: float = 0.1,
        eps: float = 1e-3,
        label_smoothing: float = 0.0,
        loss_type: str = 'sigmoid',
    ):
        """
        Initialize D3PO loss.

        Args:
            noise_schedule: Noise schedule for diffusion
            mask_token_id: Token ID for masking
            beta: DPO temperature (controls preference sharpness)
            eps: Minimum noise level
            label_smoothing: Label smoothing for preference labels
            loss_type: 'sigmoid' (standard DPO) or 'hinge'
        """
        super().__init__()
        self.noise_schedule = noise_schedule
        self.mask_token_id = mask_token_id
        self.beta = beta
        self.eps = eps
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type

    def _apply_noise(
        self,
        x0: torch.Tensor,
        sigma: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply noise (masking) to tokens.

        Args:
            x0: Original tokens
            sigma: Noise level (batch,)
            context_mask: 1 for context (not masked), 0 for generation

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

    def _compute_log_prob(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        x_t: torch.Tensor,
        sigma: torch.Tensor,
        position_ids: torch.Tensor,
        block_position_ids: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute log-probability of x0 given x_t under the model.

        Args:
            model: The diffusion backbone
            x0: Original tokens
            x_t: Noisy tokens
            sigma: Noise level
            position_ids: Sequence positions
            block_position_ids: MSA block indices
            context_mask: Context mask
            attention_mask: Attention mask

        Returns:
            Log-probability summed over generation region (batch,)
        """
        # Get model predictions
        logits = model(
            input_ids=x_t,
            sigma=sigma,
            position_ids=position_ids,
            block_position_ids=block_position_ids,
            attention_mask=attention_mask,
        )

        # Log-probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log-prob for target tokens
        log_p = torch.gather(log_probs, -1, x0.unsqueeze(-1)).squeeze(-1)

        # Sum over generation region only
        if context_mask is not None:
            gen_mask = (1 - context_mask.float())
            if attention_mask is not None:
                gen_mask = gen_mask * attention_mask.float()
            log_p = (log_p * gen_mask).sum(dim=-1)
        else:
            if attention_mask is not None:
                log_p = (log_p * attention_mask.float()).sum(dim=-1)
            else:
                log_p = log_p.sum(dim=-1)

        return log_p

    def forward(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        winner_batch: Dict[str, torch.Tensor],
        loser_batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute D3PO loss.

        Args:
            policy_model: The model being trained
            ref_model: Frozen reference model
            winner_batch: Dictionary with winner inputs
            loser_batch: Dictionary with loser inputs

        Returns:
            Dictionary with 'loss' and metrics
        """
        batch_size = winner_batch['input_ids'].shape[0]
        device = winner_batch['input_ids'].device

        # Sample shared timestep for fair comparison
        t = torch.rand(batch_size, device=device) * (1 - self.eps) + self.eps
        sigma = self.noise_schedule(t)

        # Apply noise to both winner and loser
        winner_x0 = winner_batch['input_ids']
        loser_x0 = loser_batch['input_ids']

        winner_x_t = self._apply_noise(
            winner_x0, sigma,
            winner_batch.get('context_mask'),
        )
        loser_x_t = self._apply_noise(
            loser_x0, sigma,
            loser_batch.get('context_mask'),
        )

        # Compute log-probs under policy model
        winner_log_prob = self._compute_log_prob(
            policy_model,
            winner_x0, winner_x_t, sigma,
            winner_batch['position_ids'],
            winner_batch['block_position_ids'],
            winner_batch.get('context_mask'),
            winner_batch.get('attention_mask'),
        )

        loser_log_prob = self._compute_log_prob(
            policy_model,
            loser_x0, loser_x_t, sigma,
            loser_batch['position_ids'],
            loser_batch['block_position_ids'],
            loser_batch.get('context_mask'),
            loser_batch.get('attention_mask'),
        )

        # Compute log-probs under reference model (no gradients)
        with torch.no_grad():
            winner_log_prob_ref = self._compute_log_prob(
                ref_model,
                winner_x0, winner_x_t, sigma,
                winner_batch['position_ids'],
                winner_batch['block_position_ids'],
                winner_batch.get('context_mask'),
                winner_batch.get('attention_mask'),
            )

            loser_log_prob_ref = self._compute_log_prob(
                ref_model,
                loser_x0, loser_x_t, sigma,
                loser_batch['position_ids'],
                loser_batch['block_position_ids'],
                loser_batch.get('context_mask'),
                loser_batch.get('attention_mask'),
            )

        # Compute log ratios
        winner_ratio = winner_log_prob - winner_log_prob_ref
        loser_ratio = loser_log_prob - loser_log_prob_ref

        # DPO preference loss
        logits = self.beta * (winner_ratio - loser_ratio)

        if self.loss_type == 'sigmoid':
            # Standard DPO loss: -log(sigmoid(beta * (r_w - r_l)))
            loss = -F.logsigmoid(logits)
        elif self.loss_type == 'hinge':
            # Hinge loss variant
            loss = F.relu(1 - logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Label smoothing
        if self.label_smoothing > 0:
            # Mix with uniform loss
            uniform_loss = -torch.log(torch.tensor(0.5, device=device))
            loss = (1 - self.label_smoothing) * loss + self.label_smoothing * uniform_loss

        # Compute metrics
        with torch.no_grad():
            # Preference accuracy: how often winner has higher log-ratio
            accuracy = (winner_ratio > loser_ratio).float().mean()

            # Margin: average difference in log-ratios
            margin = (winner_ratio - loser_ratio).mean()

            # Implicit reward
            winner_reward = self.beta * winner_ratio
            loser_reward = self.beta * loser_ratio

        return {
            'loss': loss.mean(),
            'accuracy': accuracy,
            'margin': margin,
            'winner_reward': winner_reward.mean(),
            'loser_reward': loser_reward.mean(),
            'winner_log_prob': winner_log_prob.mean(),
            'loser_log_prob': loser_log_prob.mean(),
            'sigma_mean': sigma.mean(),
        }


class MDLMDPOLoss(nn.Module):
    """
    Combined MDLM DPO loss with CE regularization.

    Implements the training objective from MSAGPT:
    L = L_DPO + λ * L_CE(winner)

    The CE regularization on winner sequences prevents mode collapse.
    """

    def __init__(
        self,
        noise_schedule: nn.Module,
        mask_token_id: int,
        beta: float = 0.1,
        lambda_ce: float = 0.1,
        eps: float = 1e-3,
    ):
        """
        Initialize combined DPO loss.

        Args:
            noise_schedule: Noise schedule
            mask_token_id: Mask token ID
            beta: DPO temperature
            lambda_ce: Weight for CE regularization
            eps: Minimum noise level
        """
        super().__init__()
        self.noise_schedule = noise_schedule
        self.mask_token_id = mask_token_id
        self.beta = beta
        self.lambda_ce = lambda_ce
        self.eps = eps

        self.dpo_loss = D3POLoss(
            noise_schedule=noise_schedule,
            mask_token_id=mask_token_id,
            beta=beta,
            eps=eps,
        )

    def _compute_ce_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        position_ids: torch.Tensor,
        block_position_ids: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute cross-entropy loss on winner sequences."""
        batch_size, seq_len = x0.shape
        device = x0.device

        # Sample timestep
        t = torch.rand(batch_size, device=device) * (1 - self.eps) + self.eps
        sigma = self.noise_schedule(t)

        # Apply noise
        move_chance = sigma.unsqueeze(-1)
        if context_mask is not None:
            move_chance = move_chance * (1 - context_mask.float())

        mask = torch.rand_like(x0.float()) < move_chance
        x_t = torch.where(mask, self.mask_token_id, x0)

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

        # NLL loss
        vocab_size = logits.shape[-1]
        nll = F.cross_entropy(
            logits.view(-1, vocab_size),
            x0.view(-1),
            reduction='none',
        ).view(batch_size, seq_len)

        # Mask loss to generation region
        if context_mask is not None:
            loss_mask = 1 - context_mask.float()
        else:
            loss_mask = torch.ones_like(nll)

        if attention_mask is not None:
            loss_mask = loss_mask * attention_mask.float()

        loss = (nll * loss_mask).sum() / (loss_mask.sum() + 1e-8)

        return loss

    def forward(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        winner_batch: Dict[str, torch.Tensor],
        loser_batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined DPO + CE loss.

        Args:
            policy_model: Model being trained
            ref_model: Frozen reference model
            winner_batch: Winner batch
            loser_batch: Loser batch

        Returns:
            Dictionary with losses and metrics
        """
        # DPO loss
        dpo_result = self.dpo_loss(
            policy_model, ref_model,
            winner_batch, loser_batch,
        )

        # CE regularization on winner
        ce_loss = self._compute_ce_loss(
            policy_model,
            winner_batch['input_ids'],
            winner_batch['position_ids'],
            winner_batch['block_position_ids'],
            winner_batch.get('context_mask'),
            winner_batch.get('attention_mask'),
        )

        # Combined loss
        total_loss = dpo_result['loss'] + self.lambda_ce * ce_loss

        result = {
            'loss': total_loss,
            'dpo_loss': dpo_result['loss'],
            'ce_loss': ce_loss,
            'accuracy': dpo_result['accuracy'],
            'margin': dpo_result['margin'],
            'winner_reward': dpo_result['winner_reward'],
            'loser_reward': dpo_result['loser_reward'],
        }

        return result
