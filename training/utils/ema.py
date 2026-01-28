"""
Exponential Moving Average (EMA) for model parameters.

Used during training to maintain a smoothed version of the model
for better generalization.
"""

import copy
import torch
import torch.nn as nn
from typing import Optional, Dict, Iterable
import logging

logger = logging.getLogger(__name__)


class EMA:
    """
    Simple Exponential Moving Average wrapper.

    Maintains an EMA of model parameters that can be used
    for evaluation while training continues.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 0,
        update_every: int = 1,
    ):
        """
        Initialize EMA.

        Args:
            model: Model to track
            decay: EMA decay factor (closer to 1 = slower updates)
            update_after_step: Start EMA updates after this step
            update_every: Update EMA every N steps
        """
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.step = 0

        # Create EMA model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA parameters.

        Args:
            model: Current model with updated parameters
        """
        self.step += 1

        if self.step < self.update_after_step:
            # Just copy parameters initially
            self._copy_params(model)
            return

        if self.step % self.update_every != 0:
            return

        for ema_param, model_param in zip(
            self.ema_model.parameters(),
            model.parameters()
        ):
            # Ensure EMA param is on the same device as model param
            if ema_param.device != model_param.device:
                ema_param.data = ema_param.data.to(model_param.device)
            ema_param.data.mul_(self.decay).add_(
                model_param.data, alpha=1 - self.decay
            )

    @torch.no_grad()
    def _copy_params(self, model: nn.Module):
        """Copy parameters from model to EMA model."""
        for ema_param, model_param in zip(
            self.ema_model.parameters(),
            model.parameters()
        ):
            # Ensure EMA param is on the same device as model param
            if ema_param.device != model_param.device:
                ema_param.data = ema_param.data.to(model_param.device)
            ema_param.data.copy_(model_param.data)

    def state_dict(self) -> Dict:
        """Get EMA state dict."""
        return {
            'ema_model': self.ema_model.state_dict(),
            'decay': self.decay,
            'step': self.step,
            'update_after_step': self.update_after_step,
            'update_every': self.update_every,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load EMA state dict."""
        self.ema_model.load_state_dict(state_dict['ema_model'])
        self.decay = state_dict['decay']
        self.step = state_dict['step']
        self.update_after_step = state_dict.get('update_after_step', 0)
        self.update_every = state_dict.get('update_every', 1)

    def to(self, device: torch.device) -> 'EMA':
        """Move EMA model to device."""
        self.ema_model = self.ema_model.to(device)
        return self

    def get_model(self) -> nn.Module:
        """Get the EMA model."""
        return self.ema_model


class ExponentialMovingAverage:
    """
    More feature-rich EMA implementation with warmup and decay scheduling.
    """

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        warmup_steps: int = 0,
        power: float = 1.0,
        use_num_updates: bool = True,
    ):
        """
        Initialize EMA.

        Args:
            parameters: Model parameters to track
            decay: Target EMA decay
            min_decay: Minimum decay during warmup
            warmup_steps: Number of steps to warm up decay
            power: Power for decay schedule
            use_num_updates: Adjust decay based on number of updates
        """
        self.decay = decay
        self.min_decay = min_decay
        self.warmup_steps = warmup_steps
        self.power = power
        self.use_num_updates = use_num_updates
        self.num_updates = 0

        # Store shadow parameters
        self.shadow_params = [p.clone().detach() for p in parameters]
        self.collected_params = None

    def _get_decay(self) -> float:
        """Get current decay value with optional warmup."""
        if self.warmup_steps > 0 and self.num_updates < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.num_updates / self.warmup_steps
            return self.min_decay + (self.decay - self.min_decay) * warmup_factor

        if self.use_num_updates:
            # Adjust decay based on number of updates (from original EMA paper)
            return min(
                self.decay,
                (1 + self.num_updates) / (10 + self.num_updates)
            )

        return self.decay

    @torch.no_grad()
    def update(self, parameters: Iterable[nn.Parameter]):
        """
        Update EMA parameters.

        Args:
            parameters: Current model parameters
        """
        self.num_updates += 1
        decay = self._get_decay()

        for shadow, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                shadow.mul_(decay).add_(param.data, alpha=1 - decay)

    @torch.no_grad()
    def copy_to(self, parameters: Iterable[nn.Parameter]):
        """
        Copy EMA parameters to model.

        Args:
            parameters: Model parameters to update
        """
        for shadow, param in zip(self.shadow_params, parameters):
            param.data.copy_(shadow)

    @torch.no_grad()
    def store(self, parameters: Iterable[nn.Parameter]):
        """Store current parameters for later restoration."""
        self.collected_params = [p.clone() for p in parameters]

    @torch.no_grad()
    def restore(self, parameters: Iterable[nn.Parameter]):
        """Restore previously stored parameters."""
        if self.collected_params is None:
            raise RuntimeError("No parameters stored to restore")

        for stored, param in zip(self.collected_params, parameters):
            param.data.copy_(stored)

        self.collected_params = None

    def state_dict(self) -> Dict:
        """Get state dict."""
        return {
            'shadow_params': self.shadow_params,
            'num_updates': self.num_updates,
            'decay': self.decay,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load state dict."""
        self.shadow_params = state_dict['shadow_params']
        self.num_updates = state_dict['num_updates']
        self.decay = state_dict['decay']


def apply_ema_to_model(
    model: nn.Module,
    ema: EMA,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Apply EMA parameters to a model.

    Args:
        model: Model to update
        ema: EMA wrapper
        device: Device to move EMA model to

    Returns:
        Model with EMA parameters
    """
    ema_model = ema.get_model()
    if device is not None:
        ema_model = ema_model.to(device)

    model.load_state_dict(ema_model.state_dict())
    return model
