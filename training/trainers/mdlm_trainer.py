"""
MDLM Pre-training Trainer.

PyTorch Lightning module for training masked diffusion language models
on protein MSA data.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List, Tuple
import logging
import math

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    pl = None

from ..losses.mdlm_loss import MDLMLoss, compute_diffusion_loss
from ..utils.ema import EMA
from ..utils.metrics import compute_perplexity, compute_accuracy

logger = logging.getLogger(__name__)


class MDLMTrainer(pl.LightningModule if HAS_LIGHTNING else nn.Module):
    """
    PyTorch Lightning module for MDLM pre-training.

    Handles:
    - Diffusion loss computation
    - Optimizer and LR scheduling
    - EMA updates
    - Logging and checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        noise_schedule: nn.Module,
        mask_token_id: int = 36,
        learning_rate: float = 1.2e-4,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.95),
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        ema_decay: float = 0.9999,
        use_ema: bool = True,
        gradient_clip_val: float = 1.0,
        eps: float = 1e-3,
    ):
        """
        Initialize trainer.

        Args:
            model: MDLM backbone model
            noise_schedule: Noise schedule for diffusion
            mask_token_id: Token ID for masking
            learning_rate: Base learning rate
            weight_decay: Weight decay for AdamW
            betas: AdamW beta parameters
            warmup_steps: LR warmup steps
            max_steps: Total training steps
            ema_decay: EMA decay factor
            use_ema: Whether to use EMA
            gradient_clip_val: Gradient clipping value
            eps: Minimum noise level
        """
        super().__init__()

        self.model = model
        self.noise_schedule = noise_schedule
        self.mask_token_id = mask_token_id

        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_clip_val = gradient_clip_val
        self.eps = eps

        # EMA
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema = None
        if use_ema:
            self.ema = EMA(model, decay=ema_decay)

        # Loss function
        self.loss_fn = MDLMLoss(
            noise_schedule=noise_schedule,
            mask_token_id=mask_token_id,
            eps=eps,
        )

        # Save hyperparameters (if Lightning available)
        if HAS_LIGHTNING:
            self.save_hyperparameters(ignore=['model', 'noise_schedule'])

    @property
    def backbone(self) -> nn.Module:
        """Get the backbone model."""
        return self.model

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        block_position_ids: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs (batch, seq)
            position_ids: Sequence positions
            block_position_ids: MSA block indices
            sigma: Noise level (if None, samples randomly)
            attention_mask: Attention mask

        Returns:
            Logits (batch, seq, vocab)
        """
        if sigma is None:
            batch_size = input_ids.shape[0]
            t = torch.rand(batch_size, device=input_ids.device)
            sigma = self.noise_schedule(t)

        return self.model(
            input_ids=input_ids,
            sigma=sigma,
            position_ids=position_ids,
            block_position_ids=block_position_ids,
            attention_mask=attention_mask,
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Single training step.

        Args:
            batch: Dictionary with input tensors
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        # Compute diffusion loss
        result = compute_diffusion_loss(
            model=self.model,
            x0=batch['input_ids'],
            position_ids=batch['position_ids'],
            block_position_ids=batch['block_position_ids'],
            noise_schedule=self.noise_schedule,
            mask_token_id=self.mask_token_id,
            attention_mask=batch.get('attention_mask'),
            context_mask=batch.get('context_mask'),
            eps=self.eps,
        )

        loss = result['loss']

        # Log metrics
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/accuracy', result['accuracy'])
        self.log('train/mask_accuracy', result['mask_accuracy'])
        self.log('train/sigma_mean', result['sigma_mean'])
        self.log('train/perplexity', torch.exp(result['nll']))

        return loss

    def on_fit_start(self):
        """Called when fit begins - ensure EMA is on correct device."""
        if self.use_ema and self.ema is not None:
            self.ema.to(self.device)

    def on_before_optimizer_step(self, optimizer):
        """Update EMA before optimizer step."""
        if self.use_ema and self.ema is not None:
            self.ema.update(self.model)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.

        Uses EMA model if available.
        """
        # Use EMA model for validation
        model = self.ema.get_model() if self.use_ema and self.ema is not None else self.model

        result = compute_diffusion_loss(
            model=model,
            x0=batch['input_ids'],
            position_ids=batch['position_ids'],
            block_position_ids=batch['block_position_ids'],
            noise_schedule=self.noise_schedule,
            mask_token_id=self.mask_token_id,
            attention_mask=batch.get('attention_mask'),
            context_mask=batch.get('context_mask'),
            eps=self.eps,
        )

        self.log('val/loss', result['loss'], prog_bar=True, sync_dist=True)
        self.log('val/accuracy', result['accuracy'], sync_dist=True)
        self.log('val/perplexity', torch.exp(result['nll']), sync_dist=True)

        return result

    def configure_optimizers(self):
        """Configure optimizer and LR scheduler."""
        # Separate parameters by weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'embed' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.learning_rate,
            betas=self.betas,
        )

        # Cosine schedule with warmup
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Save EMA state in checkpoint."""
        if self.use_ema and self.ema is not None:
            checkpoint['ema_state'] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Load EMA state from checkpoint."""
        if self.use_ema and 'ema_state' in checkpoint:
            if self.ema is None:
                self.ema = EMA(self.model, decay=self.ema_decay)
            self.ema.load_state_dict(checkpoint['ema_state'])


def create_trainer(
    model: nn.Module,
    noise_schedule: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    output_dir: str = './checkpoints',
    max_epochs: int = 10,
    max_steps: int = -1,
    gradient_clip_val: float = 1.0,
    accumulate_grad_batches: int = 1,
    precision: str = 'bf16-mixed',
    devices: int = 1,
    strategy: str = 'auto',
    **trainer_kwargs,
) -> Tuple['MDLMTrainer', 'pl.Trainer']:
    """
    Create MDLM trainer and Lightning trainer.

    Args:
        model: MDLM model
        noise_schedule: Noise schedule
        train_dataloader: Training data
        val_dataloader: Validation data
        output_dir: Checkpoint directory
        max_epochs: Maximum epochs
        max_steps: Maximum steps (-1 for unlimited)
        gradient_clip_val: Gradient clipping
        accumulate_grad_batches: Gradient accumulation
        precision: Training precision
        devices: Number of devices
        strategy: Training strategy
        **trainer_kwargs: Additional trainer arguments

    Returns:
        Tuple of (MDLMTrainer, pl.Trainer)
    """
    if not HAS_LIGHTNING:
        raise ImportError("PyTorch Lightning is required for training")

    # Create module
    mdlm_trainer = MDLMTrainer(
        model=model,
        noise_schedule=noise_schedule,
        max_steps=max_steps if max_steps > 0 else 100000,
        **trainer_kwargs,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename='mdlm-{epoch:02d}-{val_loss:.4f}',
            save_top_k=3,
            monitor='val/loss',
            mode='min',
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    # Create Lightning trainer
    lightning_trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=max_epochs,
        max_steps=max_steps,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        precision=precision,
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    return mdlm_trainer, lightning_trainer
