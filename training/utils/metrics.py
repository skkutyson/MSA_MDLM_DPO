"""
Training metrics for MDLM.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional
import math


def compute_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute perplexity from logits.

    Args:
        logits: Model logits (batch, seq, vocab)
        targets: Target token IDs (batch, seq)
        mask: Optional mask for valid positions

    Returns:
        Perplexity value
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Cross-entropy loss
    log_probs = F.log_softmax(logits, dim=-1)
    nll = F.nll_loss(
        log_probs.view(-1, vocab_size),
        targets.view(-1),
        reduction='none',
    ).view(batch_size, seq_len)

    if mask is not None:
        nll = nll * mask
        avg_nll = nll.sum() / (mask.sum() + 1e-8)
    else:
        avg_nll = nll.mean()

    return torch.exp(avg_nll)


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute prediction accuracy.

    Args:
        logits: Model logits (batch, seq, vocab)
        targets: Target token IDs (batch, seq)
        mask: Optional mask for valid positions

    Returns:
        Accuracy value
    """
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float()

    if mask is not None:
        return (correct * mask).sum() / (mask.sum() + 1e-8)
    return correct.mean()


def compute_bits_per_dim(
    loss: torch.Tensor,
    vocab_size: int = 27,
) -> torch.Tensor:
    """
    Convert NLL loss to bits per dimension.

    Args:
        loss: NLL loss value
        vocab_size: Size of vocabulary

    Returns:
        Bits per dimension
    """
    # Convert from nats to bits
    return loss / math.log(2)


class MetricTracker:
    """Track and aggregate training metrics."""

    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: Dict[str, torch.Tensor]):
        """Update with new metric values."""
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().item()

            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0

            self.metrics[key] += value
            self.counts[key] += 1

    def compute(self) -> Dict[str, float]:
        """Compute averaged metrics."""
        return {
            key: self.metrics[key] / max(self.counts[key], 1)
            for key in self.metrics
        }

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}
