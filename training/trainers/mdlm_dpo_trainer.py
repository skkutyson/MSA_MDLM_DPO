"""
MDLM DPO Trainer.

PyTorch Lightning module for DPO fine-tuning of masked diffusion
language models using D3PO adaptation.
"""

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, Tuple
import logging
import math

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    pl = None

from ..losses.mdlm_dpo_loss import MDLMDPOLoss, D3POLoss

logger = logging.getLogger(__name__)


class MDLMDPOTrainer(pl.LightningModule if HAS_LIGHTNING else nn.Module):
    """
    PyTorch Lightning module for MDLM DPO training.

    Handles:
    - D3PO loss computation with reference model
    - CE regularization to prevent mode collapse
    - Optimizer and LR scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        noise_schedule: nn.Module,
        ref_model: Optional[nn.Module] = None,
        mask_token_id: int = 36,
        beta: float = 0.1,
        lambda_ce: float = 0.1,
        learning_rate: float = 1e-6,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.95),
        warmup_steps: int = 100,
        max_steps: int = 10000,
        gradient_clip_val: float = 1.0,
        eps: float = 1e-3,
        freeze_ref: bool = True,
    ):
        """
        Initialize DPO trainer.

        Args:
            model: MDLM model to train (policy)
            noise_schedule: Noise schedule for diffusion
            ref_model: Reference model (if None, copies from model)
            mask_token_id: Token ID for masking
            beta: DPO temperature
            lambda_ce: Weight for CE regularization
            learning_rate: Learning rate
            weight_decay: Weight decay
            betas: AdamW betas
            warmup_steps: Warmup steps
            max_steps: Total training steps
            gradient_clip_val: Gradient clipping
            eps: Minimum noise level
            freeze_ref: Whether to freeze reference model
        """
        super().__init__()

        self.model = model
        self.noise_schedule = noise_schedule
        self.mask_token_id = mask_token_id

        # Create reference model
        if ref_model is None:
            self.ref_model = copy.deepcopy(model)
        else:
            self.ref_model = ref_model

        # Freeze reference model
        if freeze_ref:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        # DPO parameters
        self.beta = beta
        self.lambda_ce = lambda_ce

        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_clip_val = gradient_clip_val
        self.eps = eps

        # Loss function
        self.loss_fn = MDLMDPOLoss(
            noise_schedule=noise_schedule,
            mask_token_id=mask_token_id,
            beta=beta,
            lambda_ce=lambda_ce,
            eps=eps,
        )

        if HAS_LIGHTNING:
            self.save_hyperparameters(ignore=['model', 'ref_model', 'noise_schedule'])

    @property
    def policy_model(self) -> nn.Module:
        """Get the policy model."""
        return self.model

    @property
    def reference_model(self) -> nn.Module:
        """Get the reference model."""
        return self.ref_model

    def _split_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Split combined batch into winner and loser batches.

        Args:
            batch: Combined batch with winner_* and loser_* keys

        Returns:
            Tuple of (winner_batch, loser_batch)
        """
        winner_batch = {}
        loser_batch = {}

        for key, value in batch.items():
            if key.startswith('winner_'):
                new_key = key.replace('winner_', '')
                winner_batch[new_key] = value
            elif key.startswith('loser_'):
                new_key = key.replace('loser_', '')
                loser_batch[new_key] = value

        return winner_batch, loser_batch

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Single training step.

        Args:
            batch: Dictionary with winner_* and loser_* tensors
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        # Split into winner/loser
        winner_batch, loser_batch = self._split_batch(batch)

        # Compute DPO loss
        result = self.loss_fn(
            self.model,
            self.ref_model,
            winner_batch,
            loser_batch,
        )

        # Log metrics
        self.log('train/loss', result['loss'], prog_bar=True)
        self.log('train/dpo_loss', result['dpo_loss'])
        self.log('train/ce_loss', result['ce_loss'])
        self.log('train/accuracy', result['accuracy'], prog_bar=True)
        self.log('train/margin', result['margin'])
        self.log('train/winner_reward', result['winner_reward'])
        self.log('train/loser_reward', result['loser_reward'])

        return result['loss']

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        """
        winner_batch, loser_batch = self._split_batch(batch)

        result = self.loss_fn(
            self.model,
            self.ref_model,
            winner_batch,
            loser_batch,
        )

        self.log('val/loss', result['loss'], prog_bar=True, sync_dist=True)
        self.log('val/accuracy', result['accuracy'], sync_dist=True)
        self.log('val/margin', result['margin'], sync_dist=True)

        return result

    def configure_optimizers(self):
        """Configure optimizer and LR scheduler."""
        # Only optimize policy model parameters
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

        # Linear warmup + constant LR (typical for fine-tuning)
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            return 1.0

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
        """Save reference model in checkpoint."""
        # Only save policy model state (ref is frozen)
        checkpoint['ref_model_state'] = self.ref_model.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Load reference model from checkpoint."""
        if 'ref_model_state' in checkpoint:
            self.ref_model.load_state_dict(checkpoint['ref_model_state'])


def create_dpo_trainer(
    model: nn.Module,
    noise_schedule: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    output_dir: str = './checkpoints',
    max_epochs: int = 1,
    max_steps: int = -1,
    gradient_clip_val: float = 1.0,
    accumulate_grad_batches: int = 1,
    precision: str = 'bf16-mixed',
    devices: int = 1,
    strategy: str = 'auto',
    **trainer_kwargs,
) -> Tuple['MDLMDPOTrainer', 'pl.Trainer']:
    """
    Create DPO trainer and Lightning trainer.

    Args:
        model: MDLM model
        noise_schedule: Noise schedule
        train_dataloader: Training data
        val_dataloader: Validation data
        output_dir: Checkpoint directory
        max_epochs: Maximum epochs
        max_steps: Maximum steps
        gradient_clip_val: Gradient clipping
        accumulate_grad_batches: Gradient accumulation
        precision: Training precision
        devices: Number of devices
        strategy: Training strategy
        **trainer_kwargs: Additional trainer arguments

    Returns:
        Tuple of (MDLMDPOTrainer, pl.Trainer)
    """
    if not HAS_LIGHTNING:
        raise ImportError("PyTorch Lightning is required for training")

    # Create module
    dpo_trainer = MDLMDPOTrainer(
        model=model,
        noise_schedule=noise_schedule,
        max_steps=max_steps if max_steps > 0 else 10000,
        **trainer_kwargs,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename='mdlm-dpo-{epoch:02d}-{val_loss:.4f}',
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
        log_every_n_steps=1,  # Log every step for DPO
    )

    return dpo_trainer, lightning_trainer


class SimpleDPOTrainer:
    """
    Simple DPO trainer without Lightning dependencies.

    For use in environments without PyTorch Lightning.
    """

    def __init__(
        self,
        model: nn.Module,
        noise_schedule: nn.Module,
        ref_model: Optional[nn.Module] = None,
        mask_token_id: int = 36,
        beta: float = 0.1,
        lambda_ce: float = 0.1,
        learning_rate: float = 1e-6,
        weight_decay: float = 0.01,
        device: str = 'cuda',
    ):
        """Initialize simple DPO trainer."""
        self.model = model.to(device)
        self.noise_schedule = noise_schedule.to(device)
        self.device = device

        if ref_model is None:
            self.ref_model = copy.deepcopy(model).to(device)
        else:
            self.ref_model = ref_model.to(device)

        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.loss_fn = MDLMDPOLoss(
            noise_schedule=noise_schedule,
            mask_token_id=mask_token_id,
            beta=beta,
            lambda_ce=lambda_ce,
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Move to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Split batch
        winner = {k.replace('winner_', ''): v for k, v in batch.items() if 'winner_' in k}
        loser = {k.replace('loser_', ''): v for k, v in batch.items() if 'loser_' in k}

        # Forward
        self.optimizer.zero_grad()
        result = self.loss_fn(self.model, self.ref_model, winner, loser)

        # Backward
        result['loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in result.items()}

    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        torch.save({
            'model': self.model.state_dict(),
            'ref_model': self.ref_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.ref_model.load_state_dict(checkpoint['ref_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
