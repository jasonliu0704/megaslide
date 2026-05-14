"""Baseline models for MegaSlide-DiT paper experiments.

Implements:
1. Dense3DDiT - Global attention baseline (OOMs at 64 frames)
2. SwinDiT - Fixed 3D window attention baseline (256 frames)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import MegaSlideConfig


class Dense3DDiT(nn.Module):
    """Dense 3D-DiT with global attention (Paper Section 5.2 baseline).

    This baseline uses standard global attention over all spatiotemporal tokens.
    Complexity: O(N²) where N = T * H_patches * W_patches

    Expected behavior:
    - 16 frames: ~10 GB HBM
    - 32 frames: ~40 GB HBM
    - 64 frames: ~115 GB HBM (barely fits)
    - 256 frames: OOM (would need ~1.8 TB HBM)
    """

    def __init__(self, config: MegaSlideConfig):
        super().__init__()
        self.config = config

        # Patchify (same as MegaSlideDiT)
        self.patch_embed = nn.Conv3d(
            config.in_channels,
            config.hidden_size,
            kernel_size=(1, config.patch_size, config.patch_size),
            stride=(1, config.patch_size, config.patch_size),
        )

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.num_patches, config.hidden_size) * 0.02
        )

        # Timestep conditioning
        self.time_embed = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )

        # Dense blocks (global attention)
        self.blocks = nn.ModuleList([
            Dense3DBlock(config) for _ in range(config.num_layers)
        ])

        # Output head
        self.norm_out = nn.LayerNorm(config.hidden_size)
        self.out_proj = nn.Linear(
            config.hidden_size, config.in_channels * config.patch_size ** 2
        )

    def forward(
        self,
        latents: Tensor,
        timesteps: Tensor,
        text_embeds: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            latents: [B, C, T, H, W]
            timesteps: [B]
            text_embeds: [B, L, D] optional
            text_mask: [B, L] optional

        Returns:
            pred_noise: [B, C, T, H, W]
        """
        B, C, T, H, W = latents.shape

        # Patchify
        x = self.patch_embed(latents)  # [B, D, T, H', W']
        _, D, T, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]

        # Add positional embeddings
        x = x + self.pos_embed

        # Timestep conditioning
        t_emb = self.time_embed(self._get_timestep_embedding(timesteps))
        x = x + t_emb.unsqueeze(1)

        # Apply dense blocks
        for block in self.blocks:
            x = block(x, text_embeds, text_mask)

        # Output projection
        x = self.norm_out(x)
        x = self.out_proj(x)  # [B, N, C * p^2]

        # Unpatchify
        x = x.view(B, T, H_p, W_p, C, self.config.patch_size, self.config.patch_size)
        x = x.permute(0, 4, 1, 2, 5, 3, 6).contiguous()
        x = x.view(B, C, T, H, W)

        return x

    def _get_timestep_embedding(self, timesteps: Tensor, max_period: int = 10000) -> Tensor:
        """Sinusoidal timestep embeddings."""
        half = self.config.hidden_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=timesteps.device)
            / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding


class Dense3DBlock(nn.Module):
    """Dense block with global multi-head attention."""

    def __init__(self, config: MegaSlideConfig):
        super().__init__()
        self.config = config

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attn = nn.MultiheadAttention(
            config.hidden_size,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(
            config.hidden_size,
            int(config.hidden_size * config.mlp_ratio),
            config.dropout,
        )

        # Optional cross-attention for text conditioning
        if hasattr(config, 'use_text_conditioning') and config.use_text_conditioning:
            self.norm_cross = nn.LayerNorm(config.hidden_size)
            self.cross_attn = nn.MultiheadAttention(
                config.hidden_size,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )

    def forward(
        self,
        x: Tensor,
        text_embeds: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: [B, N, D]
            text_embeds: [B, L, D] optional
            text_mask: [B, L] optional

        Returns:
            y: [B, N, D]
        """
        # Self-attention (global over all N tokens)
        x_normed = self.norm1(x)
        attn_out, _ = self.attn(x_normed, x_normed, x_normed)
        x = x + attn_out

        # Cross-attention (optional)
        if text_embeds is not None and hasattr(self, 'cross_attn'):
            x_normed = self.norm_cross(x)
            attn_out, _ = self.cross_attn(
                x_normed, text_embeds, text_embeds, key_padding_mask=~text_mask
            )
            x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class SwinDiT(nn.Module):
    """Swin-DiT with fixed 3D windows (Paper Section 5.2 baseline).

    This baseline uses shifted window attention with fixed window size.
    Complexity: O(N * w³) where w = 3*16*16 = 768 (fixed)

    Expected behavior:
    - 256 frames: ~115 GB HBM (fits)
    - Lower VBench consistency score (fixed windows can't adapt to motion)
    """

    def __init__(self, config: MegaSlideConfig):
        super().__init__()
        self.config = config
        self.window_size = (3, 16, 16)  # Paper: "window size 16×16×3"

        # Patchify (same as MegaSlideDiT)
        self.patch_embed = nn.Conv3d(
            config.in_channels,
            config.hidden_size,
            kernel_size=(1, config.patch_size, config.patch_size),
            stride=(1, config.patch_size, config.patch_size),
        )

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.num_patches, config.hidden_size) * 0.02
        )

        # Timestep conditioning
        self.time_embed = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )

        # Swin blocks (alternating regular and shifted windows)
        self.blocks = nn.ModuleList([
            SwinBlock(config, self.window_size, shift=(i % 2 == 1))
            for i in range(config.num_layers)
        ])

        # Output head
        self.norm_out = nn.LayerNorm(config.hidden_size)
        self.out_proj = nn.Linear(
            config.hidden_size, config.in_channels * config.patch_size ** 2
        )

    def forward(
        self,
        latents: Tensor,
        timesteps: Tensor,
        text_embeds: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            latents: [B, C, T, H, W]
            timesteps: [B]
            text_embeds: [B, L, D] optional
            text_mask: [B, L] optional

        Returns:
            pred_noise: [B, C, T, H, W]
        """
        B, C, T, H, W = latents.shape

        # Patchify
        x = self.patch_embed(latents)  # [B, D, T, H', W']
        _, D, T, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]

        # Add positional embeddings
        x = x + self.pos_embed

        # Timestep conditioning
        t_emb = self.time_embed(self._get_timestep_embedding(timesteps))
        x = x + t_emb.unsqueeze(1)

        # Apply Swin blocks
        video_shape = (T, H_p, W_p)
        for block in self.blocks:
            x = block(x, video_shape, text_embeds, text_mask)

        # Output projection
        x = self.norm_out(x)
        x = self.out_proj(x)  # [B, N, C * p^2]

        # Unpatchify
        x = x.view(B, T, H_p, W_p, C, self.config.patch_size, self.config.patch_size)
        x = x.permute(0, 4, 1, 2, 5, 3, 6).contiguous()
        x = x.view(B, C, T, H, W)

        return x

    def _get_timestep_embedding(self, timesteps: Tensor, max_period: int = 10000) -> Tensor:
        """Sinusoidal timestep embeddings."""
        half = self.config.hidden_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=timesteps.device)
            / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding


class SwinBlock(nn.Module):
    """Swin block with 3D window attention (regular or shifted)."""

    def __init__(
        self,
        config: MegaSlideConfig,
        window_size: Tuple[int, int, int],
        shift: bool = False,
    ):
        super().__init__()
        self.config = config
        self.window_size = window_size
        self.shift_size = (
            (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
            if shift
            else (0, 0, 0)
        )

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attn = WindowAttention3D(
            config.hidden_size,
            config.num_heads,
            window_size,
            config.dropout,
        )

        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(
            config.hidden_size,
            int(config.hidden_size * config.mlp_ratio),
            config.dropout,
        )

    def forward(
        self,
        x: Tensor,
        video_shape: Tuple[int, int, int],
        text_embeds: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: [B, N, D]
            video_shape: (T, H_patches, W_patches)
            text_embeds: [B, L, D] optional (not used in Swin)
            text_mask: [B, L] optional (not used in Swin)

        Returns:
            y: [B, N, D]
        """
        B, N, D = x.shape
        T, H, W = video_shape

        # Reshape to 3D
        x = x.view(B, T, H, W, D)

        # Cyclic shift
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition_3d(shifted_x, self.window_size)  # [B*nW, wt, wh, ww, D]
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], D)

        # Window attention
        attn_windows = self.attn(self.norm1(x_windows))  # [B*nW, wt*wh*ww, D]

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], D)
        shifted_x = window_reverse_3d(attn_windows, self.window_size, T, H, W)  # [B, T, H, W, D]

        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            x = shifted_x

        # Flatten back
        x = x.view(B, N, D)

        # Residual connection + MLP
        x = x + attn_windows.view(B, N, D)
        x = x + self.mlp(self.norm2(x))

        return x


class WindowAttention3D(nn.Module):
    """Window-based multi-head self-attention for 3D volumes."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: [B*nW, N_win, D] where N_win = wt*wh*ww

        Returns:
            y: [B*nW, N_win, D]
        """
        B_nW, N_win, D = x.shape

        # QKV
        qkv = self.qkv(x).reshape(B_nW, N_win, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*nW, nh, N_win, d]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*nW, nh, N_win, N_win]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Aggregate
        x = (attn @ v).transpose(1, 2).reshape(B_nW, N_win, D)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class MLP(nn.Module):
    """MLP with GELU activation (reused from model.py)."""

    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def window_partition_3d(
    x: Tensor, window_size: Tuple[int, int, int]
) -> Tensor:
    """Partition 3D volume into non-overlapping windows.

    Args:
        x: [B, T, H, W, D]
        window_size: (wt, wh, ww)

    Returns:
        windows: [B*nW, wt, wh, ww, D]
    """
    B, T, H, W, D = x.shape
    wt, wh, ww = window_size

    x = x.view(B, T // wt, wt, H // wh, wh, W // ww, ww, D)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, wt, wh, ww, D)

    return windows


def window_reverse_3d(
    windows: Tensor, window_size: Tuple[int, int, int], T: int, H: int, W: int
) -> Tensor:
    """Reverse window partition to reconstruct 3D volume.

    Args:
        windows: [B*nW, wt, wh, ww, D]
        window_size: (wt, wh, ww)
        T, H, W: Original spatiotemporal dimensions

    Returns:
        x: [B, T, H, W, D]
    """
    wt, wh, ww = window_size
    B_nW, _, _, _, D = windows.shape
    B = B_nW // ((T // wt) * (H // wh) * (W // ww))

    x = windows.view(B, T // wt, H // wh, W // ww, wt, wh, ww, D)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, T, H, W, D)

    return x
