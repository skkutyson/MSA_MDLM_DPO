"""
Noise schedule for masked diffusion language models.

Ported from MDLM (https://github.com/kuleshov-group/mdlm)
"""

import abc
import torch
import torch.nn as nn


class NoiseSchedule(abc.ABC, nn.Module):
    """Abstract base class for noise schedules."""

    @abc.abstractmethod
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Return the noise level sigma(t) for timestep t in [0, 1]."""
        pass

    @abc.abstractmethod
    def rate(self, t: torch.Tensor) -> torch.Tensor:
        """Return the rate of change d(sigma)/dt at timestep t."""
        pass


class LogLinearNoise(NoiseSchedule):
    """
    Log-linear noise schedule.

    sigma(t) = 1 - (1 - eps) * t

    This gives:
    - sigma(0) = 1 (fully masked)
    - sigma(1) = eps (nearly unmasked)

    The rate is constant: d(sigma)/dt = -(1 - eps)
    """

    def __init__(self, eps: float = 1e-3):
        """
        Args:
            eps: Small value to prevent sigma from reaching exactly 0.
        """
        super().__init__()
        self.eps = eps

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute sigma(t) = 1 - (1 - eps) * t.

        Args:
            t: Timesteps in [0, 1], shape (batch_size,) or scalar

        Returns:
            Noise level sigma(t), same shape as t
        """
        return 1 - (1 - self.eps) * t

    def rate(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute d(sigma)/dt = -(1 - eps).

        Args:
            t: Timesteps (not used, but included for API consistency)

        Returns:
            Rate of change, same shape as t
        """
        return torch.full_like(t, -(1 - self.eps))


class CosineNoise(NoiseSchedule):
    """
    Cosine noise schedule.

    sigma(t) = cos(pi/2 * t)

    This gives:
    - sigma(0) = 1 (fully masked)
    - sigma(1) = 0 (fully unmasked)
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sigma(t) = cos(pi/2 * t)."""
        import math
        sigma = torch.cos(math.pi / 2 * t)
        return sigma.clamp(min=self.eps)

    def rate(self, t: torch.Tensor) -> torch.Tensor:
        """Compute d(sigma)/dt = -pi/2 * sin(pi/2 * t)."""
        import math
        return -math.pi / 2 * torch.sin(math.pi / 2 * t)


def get_noise_schedule(name: str, **kwargs) -> NoiseSchedule:
    """
    Factory function to get a noise schedule by name.

    Args:
        name: Name of the noise schedule ('loglinear' or 'cosine')
        **kwargs: Additional arguments for the noise schedule

    Returns:
        NoiseSchedule instance
    """
    schedules = {
        'loglinear': LogLinearNoise,
        'cosine': CosineNoise,
    }
    if name not in schedules:
        raise ValueError(f"Unknown noise schedule: {name}. Choose from {list(schedules.keys())}")
    return schedules[name](**kwargs)
