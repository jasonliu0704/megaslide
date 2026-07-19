"""Hybrid Attention modules for MegaSlide-DiT.

Combines sparse global attention with local 3D-DSA:
- HybridAttention3D: Global register tokens + local 3D-DSA
- TemporalAnchorAttention3D: Per-frame anchor tokens with global self-attention + local 3D-DSA
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .attention import DeformableSlideAttention3D


class HybridAttention3D(nn.Module):
    """3D-DSA augmented with global register tokens.

    Register tokens cross-attend to all spatial tokens (gaining global context),
    then are appended as extra keys/values to the local 3D-DSA attention.
    A learnable sigmoid gate controls register influence (initialized to 0).

    Args:
        hidden_size: Hidden dimension D.
        num_heads: Number of attention heads.
        kernel_size: Tuple (k_t, k_h, k_w) for local neighborhood.
        num_registers: Number of global register tokens G.
        offset_scale: Scale factor for learned offsets.
        dropout: Dropout rate.
        gate_init: Initial value for gate parameter (default 0.0 = no register influence).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kernel_size: Tuple[int, int, int] = (3, 7, 7),
        num_registers: int = 64,
        offset_scale: float = 1.0,
        dropout: float = 0.0,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.num_registers = num_registers

        # Local 3D-DSA (reuse existing implementation)
        self.local_attn = DeformableSlideAttention3D(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kernel_size=kernel_size,
            offset_scale=offset_scale,
            dropout=dropout,
        )

        # Global register tokens
        self.registers = nn.Parameter(torch.randn(1, num_registers, hidden_size) * 0.02)

        # Cross-attention: registers (Q) attend to all spatial tokens (KV)
        self.register_norm = nn.LayerNorm(hidden_size)
        self.register_cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True,
        )

        # Learnable gate controlling register influence
        self.gate = nn.Parameter(torch.full((1,), gate_init))

    def forward(self, x: Tensor, video_shape: Tuple[int, int, int]) -> Tensor:
        """Forward pass.

        Args:
            x: [B, N, D] flattened spatiotemporal tokens.
            video_shape: (T, H, W) patch dimensions.

        Returns:
            Output [B, N, D].
        """
        B = x.shape[0]

        # 1. Expand registers to batch
        regs = self.registers.expand(B, -1, -1)

        # 2. Update registers via cross-attention to all tokens
        regs = regs + self.register_cross_attn(
            self.register_norm(regs), x, x
        )[0]

        # 3. Gate the register contribution
        g = torch.sigmoid(self.gate)
        extra_kv = regs * g

        # 4. Run local 3D-DSA with registers as extra keys/values
        return self.local_attn(x, video_shape, extra_kv=extra_kv)


class TemporalAnchorAttention3D(nn.Module):
    """3D-DSA augmented with per-frame temporal anchor tokens.

    Selects one anchor token per frame (center patch), performs global
    self-attention among anchors, then broadcasts updated info back to
    all tokens in each frame. Much cheaper than register tokens at scale.

    Args:
        hidden_size: Hidden dimension D.
        num_heads: Number of attention heads.
        kernel_size: Tuple (k_t, k_h, k_w) for local neighborhood.
        offset_scale: Scale factor for learned offsets.
        dropout: Dropout rate.
        gate_init: Initial value for gate parameter.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kernel_size: Tuple[int, int, int] = (3, 7, 7),
        offset_scale: float = 1.0,
        dropout: float = 0.0,
        gate_init: float = 0.0,
    ):
        super().__init__()

        # Local 3D-DSA
        self.local_attn = DeformableSlideAttention3D(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kernel_size=kernel_size,
            offset_scale=offset_scale,
            dropout=dropout,
        )

        # Global self-attention among temporal anchors
        self.anchor_norm = nn.LayerNorm(hidden_size)
        self.anchor_self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True,
        )

        # Projection for broadcasting anchor info back to frame tokens
        self.anchor_proj = nn.Linear(hidden_size, hidden_size)

        # Learnable gate
        self.gate = nn.Parameter(torch.full((1,), gate_init))

        # Initialize projection to zero so initial behavior is pure local
        nn.init.zeros_(self.anchor_proj.weight)
        nn.init.zeros_(self.anchor_proj.bias)

    def forward(self, x: Tensor, video_shape: Tuple[int, int, int]) -> Tensor:
        """Forward pass.

        Args:
            x: [B, N, D] flattened spatiotemporal tokens.
            video_shape: (T, H, W) patch dimensions.

        Returns:
            Output [B, N, D].
        """
        B, N, D = x.shape
        T, H, W = video_shape

        # 1. Select anchor indices (center patch of each frame)
        center = (H * W) // 2
        anchor_indices = [t * H * W + center for t in range(T)]

        # 2. Extract anchors: [B, T, D]
        anchors = x[:, anchor_indices, :]

        # 3. Global self-attention among anchors
        anchors_updated = anchors + self.anchor_self_attn(
            self.anchor_norm(anchors), self.anchor_norm(anchors), self.anchor_norm(anchors)
        )[0]

        # 4. Broadcast anchor info back to all tokens in each frame
        g = torch.sigmoid(self.gate)
        anchor_broadcast = self.anchor_proj(anchors_updated)  # [B, T, D]
        # Expand to [B, T, H*W, D] then reshape to [B, N, D]
        anchor_broadcast = anchor_broadcast.unsqueeze(2).expand(B, T, H * W, D).reshape(B, N, D)

        # 5. Add gated anchor info to input, then run local 3D-DSA
        x_augmented = x + g * anchor_broadcast
        return self.local_attn(x_augmented, video_shape)
