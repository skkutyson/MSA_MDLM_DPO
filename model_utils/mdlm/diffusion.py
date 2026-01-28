"""
Protein Diffusion wrapper for masked diffusion language modeling.

Implements the core diffusion logic:
- Forward process: q(x_t | x_0) - masking tokens
- Reverse process: p(x_0 | x_t) - predicting original tokens
- Sampling: iterative denoising from fully masked to unmasked

Uses 'subs' (substitution) parameterization where the model directly
predicts p(x_0 | x_t).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .protein_dit import ProteinDIT
from .noise_schedule import NoiseSchedule, LogLinearNoise, get_noise_schedule


class ProteinDiffusion(nn.Module):
    """
    Protein Diffusion model wrapper.

    Handles:
    - Noise injection (masking)
    - Loss computation
    - Sampling (DDPM-style iterative denoising)
    """

    def __init__(
        self,
        backbone: ProteinDIT,
        noise_schedule: NoiseSchedule,
        mask_token_id: int,
        parameterization: str = 'subs',
    ):
        """
        Args:
            backbone: The transformer backbone (ProteinDIT)
            noise_schedule: Noise schedule for diffusion
            mask_token_id: Token ID used for masking
            parameterization: 'subs' for substitution (model predicts x_0)
        """
        super().__init__()
        self.backbone = backbone
        self.noise_schedule = noise_schedule
        self.mask_token_id = mask_token_id
        self.parameterization = parameterization

        assert parameterization == 'subs', "Only 'subs' parameterization is supported"

    @property
    def vocab_size(self) -> int:
        return self.backbone.vocab_size

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def q_xt(
        self,
        x0: torch.Tensor,
        move_chance: torch.Tensor,
        mask_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample x_t from q(x_t | x_0) by randomly masking tokens.

        Args:
            x0: Original token IDs (batch, seq)
            move_chance: Probability of masking each token (batch,) or (batch, 1)
            mask_token_id: Token ID to use for masking

        Returns:
            Noisy tokens x_t (batch, seq)
        """
        if mask_token_id is None:
            mask_token_id = self.mask_token_id

        # Expand move_chance to (batch, seq)
        if move_chance.dim() == 1:
            move_chance = move_chance.unsqueeze(-1)

        # Sample mask: 1 where token should be masked
        mask = torch.rand_like(x0.float()) < move_chance

        # Apply mask
        x_t = torch.where(mask, mask_token_id, x0)

        return x_t

    def forward(
        self,
        x0: torch.Tensor,
        sigma: torch.Tensor,
        position_ids: torch.Tensor,
        block_position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: apply noise and compute log probabilities.

        Args:
            x0: Original token IDs (batch, seq)
            sigma: Noise level (batch,)
            position_ids: Sequence positions (batch, seq)
            block_position_ids: MSA block indices (batch, seq)
            attention_mask: Attention mask
            context_mask: Mask indicating context tokens (1) vs generation tokens (0)
                         Context tokens are not masked.

        Returns:
            Log probabilities p(x_0 | x_t) of shape (batch, seq, vocab_size)
        """
        # Apply noise (masking) to get x_t
        # Only mask non-context tokens
        if context_mask is not None:
            # move_chance is sigma, but only for non-context tokens
            move_chance = sigma.unsqueeze(-1) * (1 - context_mask.float())
        else:
            move_chance = sigma

        x_t = self.q_xt(x0, move_chance)

        # If context_mask provided, restore context tokens
        if context_mask is not None:
            x_t = torch.where(context_mask.bool(), x0, x_t)

        # Get logits from backbone
        logits = self.backbone(
            input_ids=x_t,
            sigma=sigma,
            position_ids=position_ids,
            block_position_ids=block_position_ids,
            attention_mask=attention_mask,
        )

        # Return log probabilities
        return F.log_softmax(logits, dim=-1)

    def get_logits(
        self,
        x_t: torch.Tensor,
        sigma: torch.Tensor,
        position_ids: torch.Tensor,
        block_position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get raw logits for given noisy input.

        Args:
            x_t: Noisy token IDs (batch, seq)
            sigma: Noise level (batch,)
            position_ids: Sequence positions (batch, seq)
            block_position_ids: MSA block indices (batch, seq)
            attention_mask: Attention mask

        Returns:
            Logits of shape (batch, seq, vocab_size)
        """
        return self.backbone(
            input_ids=x_t,
            sigma=sigma,
            position_ids=position_ids,
            block_position_ids=block_position_ids,
            attention_mask=attention_mask,
        )

    def compute_loss(
        self,
        x0: torch.Tensor,
        position_ids: torch.Tensor,
        block_position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion training loss.

        Args:
            x0: Original token IDs (batch, seq)
            position_ids: Sequence positions (batch, seq)
            block_position_ids: MSA block indices (batch, seq)
            attention_mask: Attention mask
            context_mask: Mask for context tokens (not to be masked)
            loss_mask: Mask for loss computation (1 where loss should be computed)

        Returns:
            Dictionary with 'loss' and other metrics
        """
        batch_size = x0.shape[0]

        # Sample random timesteps
        t = torch.rand(batch_size, device=x0.device)
        sigma = self.noise_schedule(t)

        # Forward pass
        log_probs = self.forward(
            x0=x0,
            sigma=sigma,
            position_ids=position_ids,
            block_position_ids=block_position_ids,
            attention_mask=attention_mask,
            context_mask=context_mask,
        )

        # Cross-entropy loss
        # log_probs: (batch, seq, vocab)
        # x0: (batch, seq)
        loss = F.nll_loss(
            log_probs.view(-1, self.vocab_size),
            x0.view(-1),
            reduction='none',
        ).view(batch_size, -1)

        # Apply loss mask if provided
        if loss_mask is not None:
            loss = loss * loss_mask
            loss = loss.sum() / (loss_mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return {'loss': loss, 'sigma_mean': sigma.mean()}

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        position_ids: torch.Tensor,
        block_position_ids: torch.Tensor,
        num_steps: int = 256,
        context_tokens: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        invalid_token_ids: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Sample from the diffusion model using DDPM-style iterative denoising.

        Args:
            batch_size: Number of samples to generate
            seq_len: Sequence length
            position_ids: Sequence positions (batch, seq)
            block_position_ids: MSA block indices (batch, seq)
            num_steps: Number of denoising steps
            context_tokens: Optional context tokens to condition on
            context_mask: Mask indicating context positions (1 = context)
            attention_mask: Attention mask
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            invalid_token_ids: Token IDs to mask out (never generate)

        Returns:
            Sampled token IDs (batch, seq)
        """
        device = self.device

        # Initialize with all mask tokens
        x = torch.full(
            (batch_size, seq_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        # If context provided, set context tokens
        if context_tokens is not None and context_mask is not None:
            x = torch.where(context_mask.bool(), context_tokens, x)

        # Time steps from 1 to 0
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = 1.0 - step * dt
            t_tensor = torch.full((batch_size,), t, device=device)
            sigma = self.noise_schedule(t_tensor)

            # Get log probabilities
            logits = self.get_logits(
                x_t=x,
                sigma=sigma,
                position_ids=position_ids,
                block_position_ids=block_position_ids,
                attention_mask=attention_mask,
            )

            # Apply temperature
            logits = logits / temperature

            # Mask invalid tokens
            if invalid_token_ids is not None:
                for token_id in invalid_token_ids:
                    logits[..., token_id] = float('-inf')

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            # Apply nucleus (top-p) sampling
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            x_pred = torch.multinomial(probs.view(-1, self.vocab_size), num_samples=1)
            x_pred = x_pred.view(batch_size, seq_len)

            # DDPM update: decide which tokens to update
            if step < num_steps - 1:
                # Not the last step: stochastically update some tokens
                t_next = 1.0 - (step + 1) * dt
                t_next_tensor = torch.full((batch_size,), t_next, device=device)
                sigma_next = self.noise_schedule(t_next_tensor)

                # Probability of keeping the mask token
                # move_chance decreases as we get closer to t=0
                move_chance = sigma_next.unsqueeze(-1)

                # Re-mask some predictions
                remask = torch.rand_like(x.float()) < move_chance
                x = torch.where(remask, self.mask_token_id, x_pred)
            else:
                # Last step: use predictions directly
                x = x_pred

            # Restore context tokens
            if context_tokens is not None and context_mask is not None:
                x = torch.where(context_mask.bool(), context_tokens, x)

        return x

    @torch.no_grad()
    def sample_ddpm_caching(
        self,
        batch_size: int,
        seq_len: int,
        position_ids: torch.Tensor,
        block_position_ids: torch.Tensor,
        num_steps: int = 256,
        context_tokens: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        invalid_token_ids: Optional[list] = None,
    ) -> torch.Tensor:
        """
        DDPM sampling with caching for efficiency.

        Instead of re-masking predictions, this method:
        1. Keeps track of which positions are "revealed"
        2. Only unmasks new positions at each step
        3. Caches revealed tokens

        This is more efficient and often produces better results.

        Args:
            Same as sample()

        Returns:
            Sampled token IDs (batch, seq)
        """
        device = self.device

        # Initialize with all mask tokens
        x = torch.full(
            (batch_size, seq_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        # Track which positions are revealed
        revealed = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        # If context provided, set context tokens and mark as revealed
        if context_tokens is not None and context_mask is not None:
            x = torch.where(context_mask.bool(), context_tokens, x)
            revealed = revealed | context_mask.bool()

        # Time steps from 1 to 0
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = 1.0 - step * dt
            t_tensor = torch.full((batch_size,), t, device=device)
            sigma = self.noise_schedule(t_tensor)

            # Get log probabilities
            logits = self.get_logits(
                x_t=x,
                sigma=sigma,
                position_ids=position_ids,
                block_position_ids=block_position_ids,
                attention_mask=attention_mask,
            )

            # Apply temperature
            logits = logits / temperature

            # Mask invalid tokens
            if invalid_token_ids is not None:
                for token_id in invalid_token_ids:
                    logits[..., token_id] = float('-inf')

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            # Apply nucleus (top-p) sampling
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            x_pred = torch.multinomial(probs.view(-1, self.vocab_size), num_samples=1)
            x_pred = x_pred.view(batch_size, seq_len)

            if step < num_steps - 1:
                # Determine how many new positions to reveal
                t_next = 1.0 - (step + 1) * dt
                t_next_tensor = torch.full((batch_size,), t_next, device=device)
                sigma_next = self.noise_schedule(t_next_tensor)

                # Target number of revealed tokens
                # As sigma decreases, more tokens should be revealed
                unrevealed = ~revealed
                num_unrevealed = unrevealed.sum(dim=1).float()
                target_revealed_frac = 1.0 - sigma_next
                num_to_reveal = (target_revealed_frac * (seq_len - (context_mask.sum(dim=1) if context_mask is not None else 0))).long()
                num_currently_revealed = revealed.sum(dim=1) - (context_mask.sum(dim=1) if context_mask is not None else 0)
                num_new_reveal = torch.clamp(num_to_reveal - num_currently_revealed, min=0)

                # Get confidence scores for unrevealed positions
                confidence = probs.max(dim=-1)[0]  # (batch, seq)
                confidence = torch.where(unrevealed, confidence, torch.tensor(-1.0, device=device))

                # For each batch, reveal top-k confident unrevealed positions
                for b in range(batch_size):
                    if num_new_reveal[b] > 0:
                        _, top_indices = torch.topk(confidence[b], k=min(num_new_reveal[b].item(), unrevealed[b].sum().item()))
                        revealed[b, top_indices] = True
                        x[b, top_indices] = x_pred[b, top_indices]
            else:
                # Last step: reveal all remaining positions
                x = torch.where(revealed, x, x_pred)

            # Restore context tokens
            if context_tokens is not None and context_mask is not None:
                x = torch.where(context_mask.bool(), context_tokens, x)

        return x
