"""MegaSlideDiT: Video Diffusion Transformer with 3D-DSA.

Implements the paper's DiT architecture:
- Patchify video latents (spatial patches, preserve temporal)
- Positional + timestep embeddings
- DiT blocks with 3D-DSA and MLP
- Unpatchify to predict noise
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .attention import DeformableSlideAttention3D
from .config import MegaSlideConfig


class MLP(nn.Module):
    """MLP block with GELU activation."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DiTBlock(nn.Module):
    """DiT block with 3D-DSA and MLP.

    Args:
        config: MegaSlideConfig instance.
    """

    def __init__(self, config: MegaSlideConfig):
        super().__init__()
        self.config = config

        # Pre-norm for self-attention
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # 3D Deformable Slide Attention
        self.attn = DeformableSlideAttention3D(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            kernel_size=config.dsa_kernel_size,
            offset_scale=config.offset_scale,
            dropout=config.dropout,
        )

        # Pre-norm for MLP
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # MLP
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.mlp_hidden_size,
            dropout=config.dropout,
        )

        # Optional: Cross-attention for text conditioning
        if config.use_text_conditioning:
            self.norm_cross = nn.LayerNorm(config.hidden_size, eps=1e-6)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )

    def forward(
        self,
        x: Tensor,
        video_shape: Tuple[int, int, int],
        text_embeds: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through DiT block.

        Args:
            x: Input tokens [B, N, D].
            video_shape: Tuple (T, H_patches, W_patches).
            text_embeds: Optional text embeddings [B, L, D].
            text_mask: Optional text mask [B, L] (True = valid, False = padding).

        Returns:
            Output tokens [B, N, D].
        """
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), video_shape)

        # Cross-attention with text (optional)
        if text_embeds is not None and hasattr(self, 'cross_attn'):
            x_normed = self.norm_cross(x)
            # Convert mask: True -> not masked, False -> masked
            key_padding_mask = ~text_mask if text_mask is not None else None
            attn_out, _ = self.cross_attn(
                x_normed, text_embeds, text_embeds, key_padding_mask=key_padding_mask
            )
            x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class MegaSlideDiT(nn.Module):
    """MegaSlide Diffusion Transformer for video generation.

    Args:
        config: MegaSlideConfig instance.
    """

    def __init__(self, config: MegaSlideConfig):
        super().__init__()
        self.config = config

        # Spatial patchify (temporal dimension not patched)
        # Input: [B, C, T, H, W] -> Output: [B, D, T, H', W'] where H' = H/p, W' = W/p
        self.patch_embed = nn.Conv3d(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=(1, config.patch_size, config.patch_size),
            stride=(1, config.patch_size, config.patch_size),
            padding=0,
            bias=True,
        )

        # Positional embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, config.num_patches, config.hidden_size) * 0.02)

        # Timestep embedding MLP (sinusoidal + MLP projection)
        self.time_embed = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )

        # DiT blocks
        self.blocks = nn.ModuleList([DiTBlock(config) for _ in range(config.num_layers)])

        # Output norm and projection
        self.norm_out = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.out_proj = nn.Linear(
            config.hidden_size, config.in_channels * (config.patch_size ** 2), bias=True
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following standard practices."""
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize positional embeddings
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        latents: Tensor,
        timesteps: Tensor,
        text_embeds: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass to predict noise.

        Args:
            latents: Noisy latent videos [B, C, T, H, W].
            timesteps: Diffusion timesteps [B] (integers in [0, diffusion_steps-1]).
            text_embeds: Optional text embeddings [B, L, D] for conditioning.
            text_mask: Optional text mask [B, L] (True = valid).

        Returns:
            Predicted noise [B, C, T, H, W].
        """
        B, C, T, H, W = latents.shape
        device = latents.device

        # Patchify: [B, C, T, H, W] -> [B, D, T, H', W']
        x = self.patch_embed(latents)
        _, D, T, H_p, W_p = x.shape

        # Flatten spatial dims: [B, D, T, H', W'] -> [B, D, T*H'*W'] -> [B, N, D]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D] where N = T * H' * W'

        # Add positional embeddings
        x = x + self.pos_embed

        # Generate and add timestep embeddings
        t_emb = self._get_timestep_embedding(timesteps, D, device)  # [B, D]
        t_emb = self.time_embed(t_emb)  # [B, D]
        x = x + t_emb.unsqueeze(1)  # Broadcast to [B, N, D]

        # Move text embeddings to same device if provided
        if text_embeds is not None:
            text_embeds = text_embeds.to(device)
            if text_mask is not None:
                text_mask = text_mask.to(device)

        # Apply DiT blocks
        video_shape = (T, H_p, W_p)
        for block in self.blocks:
            x = block(x, video_shape, text_embeds, text_mask)

        # Output norm and projection
        x = self.norm_out(x)
        x = self.out_proj(x)  # [B, N, C * p^2]

        # Unpatchify: [B, N, C * p^2] -> [B, C, T, H, W]
        p = self.config.patch_size
        x = x.view(B, T, H_p, W_p, C, p, p)
        x = x.permute(0, 4, 1, 2, 5, 3, 6).contiguous()  # [B, C, T, H_p, p, W_p, p]
        x = x.view(B, C, T, H, W)

        return x

    def _get_timestep_embedding(
        self, timesteps: Tensor, dim: int, device: torch.device, max_period: int = 10000
    ) -> Tensor:
        """Generate sinusoidal timestep embeddings.

        Args:
            timesteps: Timestep indices [B].
            dim: Embedding dimension.
            device: Torch device.
            max_period: Maximum period for sinusoids.

        Returns:
            Embeddings [B, dim].
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=device, dtype=torch.float32) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # Handle odd dimensions
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding

    def estimate_flops_per_step(self) -> int:
        """Estimate FLOPs per forward pass (for MFU calculation).

        Returns:
            Estimated FLOPs as an integer.
        """
        N = self.config.num_patches
        D = self.config.hidden_size
        L = self.config.num_layers
        k_total = self.config.dsa_kernel_size[0] * self.config.dsa_kernel_size[1] * self.config.dsa_kernel_size[2]

        # Attention: 4 matmuls (Q, K, V, out) + local attention computation
        # QKV: 3 * N * D^2, Out: N * D^2, Local attn: N * k_total * D
        attn_flops = L * (4 * N * D * D + N * k_total * D)

        # MLP: 2 matmuls (up + down)
        mlp_flops = L * 2 * N * D * self.config.mlp_hidden_size

        # Patch embed + output projection
        patch_flops = N * self.config.in_channels * (self.config.patch_size ** 2) * D
        out_flops = N * D * self.config.in_channels * (self.config.patch_size ** 2)

        total_flops = attn_flops + mlp_flops + patch_flops + out_flops

        # Multiply by 2 for forward (FLOPs typically counted for matmuls)
        return total_flops * 2
