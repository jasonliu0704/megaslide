"""Hybrid MegaSlideDiT variants with global register or temporal anchor attention.

Drop-in replacements for MegaSlideDiT that interleave hybrid attention blocks
at configurable intervals.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .config import MegaSlideConfig
from .model import DiTBlock, MLP
from .hybrid_attention import HybridAttention3D, TemporalAnchorAttention3D


class HybridDiTBlock(nn.Module):
    """DiT block using HybridAttention3D (registers + local 3D-DSA)."""

    def __init__(self, config: MegaSlideConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = HybridAttention3D(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            kernel_size=config.dsa_kernel_size,
            num_registers=config.num_registers,
            offset_scale=config.offset_scale,
            dropout=config.dropout,
            gate_init=config.register_gate_init,
        )
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = MLP(config.hidden_size, config.mlp_hidden_size, config.dropout)

        if config.use_text_conditioning:
            self.norm_cross = nn.LayerNorm(config.hidden_size, eps=1e-6)
            self.cross_attn = nn.MultiheadAttention(
                config.hidden_size, config.num_heads, dropout=config.dropout, batch_first=True,
            )

    def forward(self, x: Tensor, video_shape: Tuple[int, int, int],
                text_embeds: Optional[Tensor] = None, text_mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.norm1(x), video_shape)
        if text_embeds is not None and hasattr(self, 'cross_attn'):
            key_padding_mask = ~text_mask if text_mask is not None else None
            x = x + self.cross_attn(self.norm_cross(x), text_embeds, text_embeds, key_padding_mask=key_padding_mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalAnchorDiTBlock(nn.Module):
    """DiT block using TemporalAnchorAttention3D."""

    def __init__(self, config: MegaSlideConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = TemporalAnchorAttention3D(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            kernel_size=config.dsa_kernel_size,
            offset_scale=config.offset_scale,
            dropout=config.dropout,
            gate_init=config.register_gate_init,
        )
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = MLP(config.hidden_size, config.mlp_hidden_size, config.dropout)

        if config.use_text_conditioning:
            self.norm_cross = nn.LayerNorm(config.hidden_size, eps=1e-6)
            self.cross_attn = nn.MultiheadAttention(
                config.hidden_size, config.num_heads, dropout=config.dropout, batch_first=True,
            )

    def forward(self, x: Tensor, video_shape: Tuple[int, int, int],
                text_embeds: Optional[Tensor] = None, text_mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.norm1(x), video_shape)
        if text_embeds is not None and hasattr(self, 'cross_attn'):
            key_padding_mask = ~text_mask if text_mask is not None else None
            x = x + self.cross_attn(self.norm_cross(x), text_embeds, text_embeds, key_padding_mask=key_padding_mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class HybridMegaSlideDiT(nn.Module):
    """MegaSlideDiT with hybrid attention blocks at configurable intervals.

    Layers at indices where (i+1) % register_interval == 0 use HybridDiTBlock,
    other layers use regular DiTBlock. Drop-in replacement for MegaSlideDiT.
    """

    def __init__(self, config: MegaSlideConfig):
        super().__init__()
        self.config = config

        self.patch_embed = nn.Conv3d(
            config.in_channels, config.hidden_size,
            kernel_size=(1, config.patch_size, config.patch_size),
            stride=(1, config.patch_size, config.patch_size), bias=True,
        )
        self.pos_embed = nn.Parameter(torch.randn(1, config.num_patches, config.hidden_size) * 0.02)
        self.time_embed = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4), nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )

        # Interleave regular and hybrid blocks
        self.blocks = nn.ModuleList()
        for i in range(config.num_layers):
            if (i + 1) % config.register_interval == 0:
                self.blocks.append(HybridDiTBlock(config))
            else:
                self.blocks.append(DiTBlock(config))

        self.norm_out = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.out_proj = nn.Linear(config.hidden_size, config.in_channels * config.patch_size ** 2, bias=True)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, latents: Tensor, timesteps: Tensor,
                text_embeds: Optional[Tensor] = None, text_mask: Optional[Tensor] = None) -> Tensor:
        B, C, T, H, W = latents.shape
        x = self.patch_embed(latents)
        _, D, T, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        t_emb = self._get_timestep_embedding(timesteps, D, latents.device)
        x = x + self.time_embed(t_emb).unsqueeze(1)

        video_shape = (T, H_p, W_p)
        for block in self.blocks:
            x = block(x, video_shape, text_embeds, text_mask)

        x = self.norm_out(x)
        x = self.out_proj(x)
        p = self.config.patch_size
        x = x.view(B, T, H_p, W_p, C, p, p)
        x = x.permute(0, 4, 1, 2, 5, 3, 6).contiguous().view(B, C, T, H, W)
        return x

    def _get_timestep_embedding(self, timesteps: Tensor, dim: int, device: torch.device) -> Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device, dtype=torch.float32) / half)
        args = timesteps[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TemporalAnchorMegaSlideDiT(nn.Module):
    """MegaSlideDiT with temporal anchor attention blocks at configurable intervals."""

    def __init__(self, config: MegaSlideConfig):
        super().__init__()
        self.config = config

        self.patch_embed = nn.Conv3d(
            config.in_channels, config.hidden_size,
            kernel_size=(1, config.patch_size, config.patch_size),
            stride=(1, config.patch_size, config.patch_size), bias=True,
        )
        self.pos_embed = nn.Parameter(torch.randn(1, config.num_patches, config.hidden_size) * 0.02)
        self.time_embed = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4), nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )

        self.blocks = nn.ModuleList()
        for i in range(config.num_layers):
            if (i + 1) % config.register_interval == 0:
                self.blocks.append(TemporalAnchorDiTBlock(config))
            else:
                self.blocks.append(DiTBlock(config))

        self.norm_out = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.out_proj = nn.Linear(config.hidden_size, config.in_channels * config.patch_size ** 2, bias=True)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, latents: Tensor, timesteps: Tensor,
                text_embeds: Optional[Tensor] = None, text_mask: Optional[Tensor] = None) -> Tensor:
        B, C, T, H, W = latents.shape
        x = self.patch_embed(latents)
        _, D, T, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        t_emb = self._get_timestep_embedding(timesteps, D, latents.device)
        x = x + self.time_embed(t_emb).unsqueeze(1)

        video_shape = (T, H_p, W_p)
        for block in self.blocks:
            x = block(x, video_shape, text_embeds, text_mask)

        x = self.norm_out(x)
        x = self.out_proj(x)
        p = self.config.patch_size
        x = x.view(B, T, H_p, W_p, C, p, p)
        x = x.permute(0, 4, 1, 2, 5, 3, 6).contiguous().view(B, C, T, H, W)
        return x

    def _get_timestep_embedding(self, timesteps: Tensor, dim: int, device: torch.device) -> Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device, dtype=torch.float32) / half)
        args = timesteps[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
