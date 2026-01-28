"""
Protein Diffusion Transformer (DIT) with 2D Rotary Embeddings.

Adapted from MDLM's DIT architecture for protein MSA generation.
Uses 2D rotary embeddings: one for sequence position, one for MSA block index.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange


class ProteinRotary2D(nn.Module):
    """
    2D Rotary Position Embedding for protein MSA.

    Splits the head dimension in half:
    - First half: position within MSA sequence (0 to msa_len-1)
    - Second half: MSA block index (which MSA in the alignment)
    """

    def __init__(self, dim: int, base: float = 10000.0):
        """
        Args:
            dim: Head dimension (will be split in half for 2D)
            base: Base for rotary embedding frequencies
        """
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.base = base

        # Precompute inverse frequencies for half dimension
        inv_freq = 1.0 / (base ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))
        self.register_buffer('inv_freq', inv_freq)

    def _compute_rotary_emb(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin for rotary embedding."""
        # positions: (seq_len, batch) or (batch, seq_len)
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)

        # (seq_len, dim/4) or (batch, seq_len, dim/4)
        freqs = torch.einsum('...i,j->...ij', positions.float(), self.inv_freq.to(positions.device))
        # (seq_len, dim/2) - duplicate for real and imaginary
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary embedding to input tensor."""
        # x: (batch, heads, seq, head_dim/2)
        # cos, sin: (seq, head_dim/2) or (batch, seq, head_dim/2)

        # Ensure cos/sin have the right shape
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.dim() == 3:
            cos = cos.unsqueeze(1)  # (batch, 1, seq, dim)
            sin = sin.unsqueeze(1)

        # Split x into two halves for rotation
        x1, x2 = x[..., ::2], x[..., 1::2]

        # Apply rotation
        cos_half = cos[..., ::2]
        sin_half = sin[..., ::2]

        rotated = torch.stack([
            x1 * cos_half - x2 * sin_half,
            x1 * sin_half + x2 * cos_half
        ], dim=-1)

        return rotated.flatten(-2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
        block_position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2D rotary embeddings to query and key.

        Args:
            q: Query tensor (batch, heads, seq, head_dim)
            k: Key tensor (batch, heads, seq, head_dim)
            position_ids: Position within MSA sequence (batch, seq) or (seq,)
            block_position_ids: MSA block index (batch, seq) or (seq,)

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        # Split q and k into two halves
        q1, q2 = q[..., :self.half_dim], q[..., self.half_dim:]
        k1, k2 = k[..., :self.half_dim], k[..., self.half_dim:]

        # Compute rotary embeddings for sequence position
        cos_pos, sin_pos = self._compute_rotary_emb(position_ids)
        # Compute rotary embeddings for block position
        cos_block, sin_block = self._compute_rotary_emb(block_position_ids)

        # Apply rotary to first half with sequence position
        q1_rot = self._apply_rotary(q1, cos_pos, sin_pos)
        k1_rot = self._apply_rotary(k1, cos_pos, sin_pos)

        # Apply rotary to second half with block position
        q2_rot = self._apply_rotary(q2, cos_block, sin_block)
        k2_rot = self._apply_rotary(k2, cos_block, sin_block)

        # Concatenate rotated halves
        q_rot = torch.cat([q1_rot, q2_rot], dim=-1)
        k_rot = torch.cat([k1_rot, k2_rot], dim=-1)

        return q_rot, k_rot


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization for conditioning on timestep.

    Modulates the normalized output with learned scale and shift
    computed from the timestep embedding.
    """

    def __init__(self, hidden_size: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * hidden_size)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq, hidden)
            cond: Conditioning tensor (batch, cond_dim)

        Returns:
            Modulated output (batch, seq, hidden)
        """
        # Project conditioning to scale and shift
        scale_shift = self.proj(cond)  # (batch, 2 * hidden)
        scale, shift = scale_shift.chunk(2, dim=-1)  # each (batch, hidden)

        # Normalize and modulate
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class DITBlock(nn.Module):
    """
    Single Diffusion Transformer block with AdaLN conditioning.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Self-attention
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Rotary embeddings
        self.rotary = ProteinRotary2D(self.head_dim)

        # AdaLN for attention
        self.adaln_attn = AdaLN(hidden_size, cond_dim)

        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

        # AdaLN for MLP
        self.adaln_mlp = AdaLN(hidden_size, cond_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        position_ids: torch.Tensor,
        block_position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq, hidden)
            cond: Timestep conditioning (batch, cond_dim)
            position_ids: Sequence positions (batch, seq)
            block_position_ids: MSA block indices (batch, seq)
            attention_mask: Optional mask (batch, seq) or (batch, 1, seq, seq)

        Returns:
            Output tensor (batch, seq, hidden)
        """
        batch_size, seq_len, _ = x.shape

        # Self-attention with AdaLN
        residual = x
        x = self.adaln_attn(x, cond)

        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply 2D rotary embeddings
        q, k = self.rotary(q, k, position_ids, block_position_ids)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # (batch, seq) -> (batch, 1, 1, seq)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # Residual connection
        x = residual + attn_output

        # MLP with AdaLN
        residual = x
        x = self.adaln_mlp(x, cond)
        x = self.mlp(x)
        x = residual + x

        return x


class ProteinDIT(nn.Module):
    """
    Protein Diffusion Transformer for masked diffusion language modeling.

    Architecture:
    - Token embedding
    - Timestep embedding (sinusoidal + MLP)
    - Stack of DITBlocks with 2D rotary embeddings
    - Final layer norm + output projection
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        cond_dim: int = 256,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ):
        """
        Args:
            vocab_size: Size of token vocabulary
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            cond_dim: Dimension of timestep conditioning
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, hidden_size)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, cond_dim * 4),
            nn.GELU(),
            nn.Linear(cond_dim * 4, cond_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DITBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                cond_dim=cond_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        # Initialize embeddings
        nn.init.normal_(self.token_embed.weight, std=0.02)

        # Initialize output projection to zero for residual learning
        nn.init.zeros_(self.output_proj.weight)

        # Initialize transformer blocks
        for block in self.blocks:
            # Scale attention output for residual
            nn.init.normal_(block.out_proj.weight, std=0.02 / math.sqrt(2 * self.num_layers))

    def forward(
        self,
        input_ids: torch.Tensor,
        sigma: torch.Tensor,
        position_ids: torch.Tensor,
        block_position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the DIT model.

        Args:
            input_ids: Token IDs (batch, seq)
            sigma: Noise level (batch,)
            position_ids: Sequence positions (batch, seq)
            block_position_ids: MSA block indices (batch, seq)
            attention_mask: Optional attention mask

        Returns:
            Logits over vocabulary (batch, seq, vocab_size)
        """
        # Token embedding
        x = self.token_embed(input_ids)  # (batch, seq, hidden)

        # Timestep embedding
        t_emb = self.time_embed(sigma)  # (batch, cond_dim)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, t_emb, position_ids, block_position_ids, attention_mask)

        # Output projection
        x = self.final_norm(x)
        logits = self.output_proj(x)

        return logits


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps (batch,)

        Returns:
            Embeddings (batch, dim)
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return emb
