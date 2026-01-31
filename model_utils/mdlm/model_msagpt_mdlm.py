"""
MSAGPT_MDLM: SAT-compatible wrapper for MDLM-based protein MSA generation.

This module provides a model interface compatible with the existing MSAGPT
CLI and utilities, but using masked diffusion instead of autoregressive
generation.
"""

import os
import json
import math
import torch
import torch.nn as nn
import argparse
from typing import Optional, Dict, Any, Tuple, List

from .protein_dit import ProteinDIT
from .diffusion import ProteinDiffusion
from .noise_schedule import get_noise_schedule


class MSAGPT_MDLM(nn.Module):
    """
    MSAGPT with MDLM (Masked Diffusion Language Model) backbone.

    This class provides a unified interface for diffusion-based protein
    MSA generation, compatible with the existing CLI infrastructure.
    """

    def __init__(
        self,
        args,
        vocab_size: int = None,  # Will be set from args or default to 128 (SAT padded)
        **kwargs
    ):
        """
        Initialize MSAGPT_MDLM.

        Args:
            args: Argument namespace with model configuration
            vocab_size: Size of token vocabulary
            **kwargs: Additional arguments
        """
        super().__init__()
        self.args = args

        # Extract configuration from args
        # SAT uses hidden_size, num_attention_heads, num_layers
        hidden_size = getattr(args, 'hidden_size', 1024)
        num_heads = getattr(args, 'num_attention_heads', 16)
        num_layers = getattr(args, 'num_layers', 24)
        cond_dim = getattr(args, 'cond_dim', 256)
        mlp_ratio = getattr(args, 'mlp_ratio', 4.0)
        # SAT uses hidden_dropout, but we fall back to 0.0
        dropout = getattr(args, 'hidden_dropout', getattr(args, 'dropout', 0.0))
        noise_type = getattr(args, 'noise_type', 'loglinear')
        mask_token_id = getattr(args, 'mask_token_id', 36)  # DIFFUSION_MASK token ID
        use_flash_attn = getattr(args, 'use_flash_attn', True)

        # Get vocab_size - SAT pads vocab to 128, so always use 128
        # SAT sets args.vocab_size to the unpadded size (e.g., 100), but the
        # actual embedding/output dimensions use the padded size (128)
        if vocab_size is None:
            # Always use 128 (SAT's padded vocab size) regardless of args
            vocab_size = 128

        # Create backbone
        self.backbone = ProteinDIT(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            cond_dim=cond_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
        )

        # Create noise schedule
        noise_schedule = get_noise_schedule(noise_type)

        # Create diffusion wrapper
        self.diffusion = ProteinDiffusion(
            backbone=self.backbone,
            noise_schedule=noise_schedule,
            mask_token_id=mask_token_id,
            parameterization='subs',
        )

        # Store config
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.mask_token_id = mask_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            input_ids: Token IDs (batch, seq)
            position_ids: 2D position IDs (batch, 2, seq) - [seq_pos, block_pos]
            attention_mask: Optional attention mask
            sigma: Optional noise level (batch,). If None, samples randomly.

        Returns:
            Logits (batch, seq, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Parse 2D position IDs
        if position_ids.dim() == 3:
            # (batch, 2, seq) -> separate seq and block positions
            seq_position_ids = position_ids[:, 0, :]
            block_position_ids = position_ids[:, 1, :]
        else:
            # Fallback: use position_ids as seq position, zeros for block
            seq_position_ids = position_ids
            block_position_ids = torch.zeros_like(position_ids)

        # Sample sigma if not provided
        if sigma is None:
            t = torch.rand(batch_size, device=input_ids.device)
            sigma = self.diffusion.noise_schedule(t)

        # Get logits
        logits = self.diffusion.get_logits(
            x_t=input_ids,
            sigma=sigma,
            position_ids=seq_position_ids,
            block_position_ids=block_position_ids,
            attention_mask=attention_mask,
        )

        return logits

    @torch.no_grad()
    def generate(
        self,
        context_tokens: torch.Tensor,
        msa_len: int,
        max_gen_length: int,
        num_steps: int = 256,
        sampler: str = 'ddpm_cache',
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        invalid_token_ids: Optional[List[int]] = None,
        msa_delimiter_id: int = 35,  # <M> token ID
        **kwargs
    ) -> torch.Tensor:
        """
        Generate MSA sequences using diffusion sampling.

        Args:
            context_tokens: Context token IDs (batch, context_len)
            msa_len: Length of each MSA sequence (including delimiter)
            max_gen_length: Maximum total generation length
            num_steps: Number of diffusion steps
            sampler: 'ddpm' or 'ddpm_cache'
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            invalid_token_ids: Token IDs to never generate
            msa_delimiter_id: Token ID for MSA delimiter <M> (default: 35)

        Returns:
            Generated token IDs (batch, total_len)
        """
        batch_size = context_tokens.shape[0]
        context_len = context_tokens.shape[1]
        device = context_tokens.device

        # Calculate generation region
        # max_gen_length includes context
        total_len = max_gen_length

        # Build full sequence with masked generation region
        full_tokens = torch.full(
            (batch_size, total_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        full_tokens[:, :context_len] = context_tokens

        # Build context mask (1 for context, 0 for generation)
        context_mask = torch.zeros(batch_size, total_len, dtype=torch.long, device=device)
        context_mask[:, :context_len] = 1

        # Pre-place <M> delimiter tokens at the end of each MSA block in generation region
        # This ensures proper MSA structure: each MSA ends with <M>
        # msa_len includes the delimiter, so actual sequence length is (msa_len - 1)
        gen_start = context_len
        gen_len = total_len - context_len
        num_gen_msa = gen_len // msa_len

        for msa_idx in range(num_gen_msa):
            # Position of <M> delimiter at end of this MSA block
            delimiter_pos = gen_start + (msa_idx + 1) * msa_len - 1
            if delimiter_pos < total_len:
                full_tokens[:, delimiter_pos] = msa_delimiter_id
                # Mark delimiter positions as fixed (part of "context")
                context_mask[:, delimiter_pos] = 1

        # Build 2D position IDs
        position_ids, block_position_ids = self._build_2d_positions(
            context_len=context_len,
            msa_len=msa_len,
            total_len=total_len,
            batch_size=batch_size,
            device=device,
        )

        # Choose sampling method
        if sampler == 'ddpm_cache':
            sample_fn = self.diffusion.sample_ddpm_caching
        else:
            sample_fn = self.diffusion.sample

        # Generate
        generated = sample_fn(
            batch_size=batch_size,
            seq_len=total_len,
            position_ids=position_ids,
            block_position_ids=block_position_ids,
            num_steps=num_steps,
            context_tokens=full_tokens,
            context_mask=context_mask,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            invalid_token_ids=invalid_token_ids,
        )

        return generated

    def _build_2d_positions(
        self,
        context_len: int,
        msa_len: int,
        total_len: int,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build 2D position IDs for MSA generation.

        Position scheme:
        - position_ids: position within each MSA sequence (0 to msa_len-1)
        - block_position_ids: which MSA in the alignment (0, 1, 2, ...)

        Args:
            context_len: Length of context (query + prompts)
            msa_len: Length of each MSA sequence
            total_len: Total sequence length
            batch_size: Batch size
            device: Device

        Returns:
            Tuple of (position_ids, block_position_ids), each (batch, total_len)
        """
        position_ids = torch.zeros(total_len, dtype=torch.long, device=device)
        block_position_ids = torch.zeros(total_len, dtype=torch.long, device=device)

        # Context region: positions 0 to context_len-1, block 0
        # (simplified - in practice may need more careful handling)
        position_ids[:context_len] = torch.arange(context_len, device=device) % msa_len

        # Generation region: organized by MSA blocks
        gen_start = context_len
        gen_len = total_len - context_len
        num_msa = gen_len // msa_len

        for msa_idx in range(num_msa):
            start = gen_start + msa_idx * msa_len
            end = start + msa_len
            if end > total_len:
                end = total_len
            actual_len = end - start
            position_ids[start:end] = torch.arange(actual_len, device=device)
            block_position_ids[start:end] = msa_idx

        # Expand for batch
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        block_position_ids = block_position_ids.unsqueeze(0).expand(batch_size, -1)

        return position_ids, block_position_ids

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        args: argparse.Namespace,
        overwrite_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple['MSAGPT_MDLM', argparse.Namespace]:
        """
        Load a pretrained MSAGPT_MDLM model.

        Args:
            path: Path to checkpoint directory
            args: Argument namespace
            overwrite_args: Arguments to overwrite in config

        Returns:
            Tuple of (model, args)
        """
        # Load config
        config_path = os.path.join(path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Update args with config
            for key, value in config.items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)

        # Apply overwrites
        if overwrite_args:
            for key, value in overwrite_args.items():
                setattr(args, key, value)

        # Create model
        model = cls(args)

        # Load weights
        ckpt_path = os.path.join(path, 'pytorch_model.bin')
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {ckpt_path}")
        else:
            # Try SAT checkpoint format
            ckpt_path = os.path.join(path, 'mp_rank_00_model_states.pt')
            if os.path.exists(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location='cpu')
                if 'module' in state_dict:
                    state_dict = state_dict['module']
                # Map SAT weights to MDLM structure (would need custom mapping)
                print(f"Warning: SAT checkpoint found but weight mapping not implemented")
            else:
                print(f"Warning: No checkpoint found at {path}")

        return model, args

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add MDLM-specific command line arguments."""
        group = parser.add_argument_group('MSAGPT-MDLM', 'MSAGPT MDLM Configurations')

        # Only add args that don't conflict with SAT's built-in args
        # hidden_size, num_attention_heads, num_layers are already defined by SAT
        group.add_argument('--cond-dim', type=int, default=256,
                          help='Dimension of timestep conditioning')
        group.add_argument('--mlp-ratio', type=float, default=4.0,
                          help='MLP hidden dimension ratio')
        group.add_argument('--noise-type', type=str, default='loglinear',
                          choices=['loglinear', 'cosine'],
                          help='Type of noise schedule')
        group.add_argument('--mask-token-id', type=int, default=36,
                          help='Token ID used for masking in diffusion (default: DIFFUSION_MASK=36)')
        group.add_argument('--use-flash-attn', action='store_true', default=True,
                          help='Use FlashAttention for faster attention computation (default: True)')
        group.add_argument('--no-flash-attn', action='store_false', dest='use_flash_attn',
                          help='Disable FlashAttention')

        return parser

    def save_pretrained(self, path: str):
        """
        Save model to directory.

        Args:
            path: Directory to save to
        """
        os.makedirs(path, exist_ok=True)

        # Save config
        config = {
            'model_class': 'MSAGPT_MDLM',
            'backbone_type': 'mdlm',
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.backbone.num_heads if hasattr(self.backbone, 'num_heads') else 16,
            'num_layers': self.backbone.num_layers,
            'cond_dim': self.args.cond_dim if hasattr(self.args, 'cond_dim') else 256,
            'mlp_ratio': self.args.mlp_ratio if hasattr(self.args, 'mlp_ratio') else 4.0,
            'dropout': self.args.dropout if hasattr(self.args, 'dropout') else 0.0,
            'noise_type': self.args.noise_type if hasattr(self.args, 'noise_type') else 'loglinear',
            'mask_token_id': self.mask_token_id,
            'vocab_size': self.vocab_size,
        }

        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        # Save weights
        torch.save(self.state_dict(), os.path.join(path, 'pytorch_model.bin'))

        print(f"Model saved to {path}")
