"""3D Deformable Slide Attention for MegaSlide-DiT.

Implements the paper's Section 4.2:
- Learns motion-adaptive offsets to sample a local spatiotemporal neighbourhood
- Uses trilinear interpolation for sampling keys and values
- Combines deformable sampling with depthwise 3D convolution for local context
- Linear complexity O(N * k_t * k_h * k_w) instead of quadratic O(N^2)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DeformableSlideAttention3D(nn.Module):
    """3D Deformable Slide Attention (3D-DSA).

    For each query position (t, h, w), learns offsets to sample k_t × k_h × k_w
    neighbors from the key/value volumes. Combines with depthwise 3D convolution
    for local context.

    Args:
        hidden_size: Hidden dimension D.
        num_heads: Number of attention heads.
        kernel_size: Tuple (k_t, k_h, k_w) for local neighborhood size.
        offset_scale: Scale factor for learned offsets (default 1.0).
        dropout: Dropout rate (default 0.0).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kernel_size: Tuple[int, int, int] = (3, 7, 7),
        offset_scale: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, f"hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.k_t, self.k_h, self.k_w = kernel_size
        self.offset_scale = offset_scale

        # Standard QKV projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # Depthwise 3D convolution for local context (paper Equation in Section 4.2)
        self.dw_conv = nn.Conv3d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=(self.k_t // 2, self.k_h // 2, self.k_w // 2),
            groups=hidden_size,  # Depthwise
            bias=True,
        )

        # Offset prediction network
        # Outputs: num_heads * k_t * k_h * k_w * 3 (3 for t, h, w offsets)
        num_offsets = num_heads * self.k_t * self.k_h * self.k_w * 3
        self.offset_net = nn.Sequential(
            nn.Conv3d(hidden_size, hidden_size, 3, padding=1, groups=hidden_size),
            nn.GELU(),
            nn.Conv3d(hidden_size, num_offsets, 1, bias=True),
        )

        # Initialize offset output to zero (start with fixed local windows)
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, video_shape: Tuple[int, int, int], extra_kv: Optional[Tensor] = None) -> Tensor:
        """Forward pass with deformable local attention.

        Args:
            x: Flattened spatiotemporal tokens [B, N, D] where N = T * H * W.
            video_shape: Tuple (T, H_patches, W_patches) of the spatial dimensions.
            extra_kv: Optional extra keys/values [B, G, D] to append to local neighborhood.

        Returns:
            Output tensor [B, N, D].
        """
        B, N, D = x.shape
        T, H, W = video_shape
        assert N == T * H * W, f"Expected N={T*H*W} but got {N}"
        assert D == self.hidden_size

        # Reshape to 5D for 3D operations: [B, D, T, H, W]
        x_5d = x.transpose(1, 2).reshape(B, D, T, H, W)

        # 1. Depthwise 3D convolution term (additive term in paper Equation)
        dw_out = self.dw_conv(x_5d)  # [B, D, T, H, W]

        # 2. Predict offsets for each query position and head
        offsets = self.offset_net(x_5d)  # [B, num_heads * k_t * k_h * k_w * 3, T, H, W]
        offsets = offsets * self.offset_scale

        # Reshape offsets: [B, num_heads, k_t, k_h, k_w, 3, T, H, W]
        offsets = offsets.view(B, self.num_heads, self.k_t, self.k_h, self.k_w, 3, T, H, W)
        # Permute to: [B, num_heads, T, H, W, k_t, k_h, k_w, 3]
        offsets = offsets.permute(0, 1, 6, 7, 8, 2, 3, 4, 5).contiguous()

        # 3. Generate base sampling grid (fixed local window centered at each position)
        base_grid = self._get_base_grid(T, H, W, x.device)  # [T, H, W, k_t, k_h, k_w, 3]

        # 4. Add learned offsets to base grid
        # Normalize offsets to [-1, 1] coordinate system used by grid_sample
        offset_norm = torch.zeros_like(offsets)
        if T > 1:
            offset_norm[..., 0] = offsets[..., 0] / (T - 1)
        if H > 1:
            offset_norm[..., 1] = offsets[..., 1] / (H - 1)
        if W > 1:
            offset_norm[..., 2] = offsets[..., 2] / (W - 1)

        # Broadcast base_grid and add offsets
        grid = base_grid.unsqueeze(0).unsqueeze(0) + offset_norm  # [B, nh, T, H, W, kt, kh, kw, 3]

        # 5. Compute Q, K, V
        q = self.q_proj(x)  # [B, N, D]
        k = self.k_proj(x)  # [B, N, D]
        v = self.v_proj(x)  # [B, N, D]

        # Reshape to multi-head: [B, N, num_heads, head_dim] -> [B, num_heads, N, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Reshape K, V to 5D: [B, num_heads, head_dim, T, H, W]
        k_5d = k.transpose(2, 3).contiguous().view(B, self.num_heads, self.head_dim, T, H, W)
        v_5d = v.transpose(2, 3).contiguous().view(B, self.num_heads, self.head_dim, T, H, W)

        # 6. Sample K and V using trilinear interpolation
        k_sampled = self._trilinear_sample_batched(k_5d, grid)  # [B, nh, d, T, H, W, kt, kh, kw]
        v_sampled = self._trilinear_sample_batched(v_5d, grid)

        # 7. Compute attention over local neighborhood
        # Reshape Q to [B, nh, T, H, W, d]
        q = q.contiguous().view(B, self.num_heads, T, H, W, self.head_dim)

        # Flatten local neighborhood dimension: [B, nh, T, H, W, d, kt*kh*kw]
        k_flat = k_sampled.permute(0, 1, 3, 4, 5, 2, 6, 7, 8).reshape(
            B, self.num_heads, T, H, W, self.head_dim, self.k_t * self.k_h * self.k_w
        )
        v_flat = v_sampled.permute(0, 1, 3, 4, 5, 2, 6, 7, 8).reshape(
            B, self.num_heads, T, H, W, self.head_dim, self.k_t * self.k_h * self.k_w
        )

        # 7b. Append extra keys/values if provided (for hybrid attention)
        if extra_kv is not None:
            G = extra_kv.shape[1]
            # Project extra_kv through k_proj and v_proj: [B, G, D]
            ek = self.k_proj(extra_kv)  # [B, G, D]
            ev = self.v_proj(extra_kv)  # [B, G, D]
            # Reshape to multi-head: [B, nh, G, d]
            ek = ek.view(B, G, self.num_heads, self.head_dim).permute(0, 2, 3, 1)  # [B, nh, d, G]
            ev = ev.view(B, G, self.num_heads, self.head_dim).permute(0, 2, 3, 1)  # [B, nh, d, G]
            # Broadcast to [B, nh, T, H, W, d, G]
            ek = ek.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(B, self.num_heads, T, H, W, self.head_dim, G)
            ev = ev.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(B, self.num_heads, T, H, W, self.head_dim, G)
            # Concatenate: [B, nh, T, H, W, d, k_total + G]
            k_flat = torch.cat([k_flat, ek], dim=-1)
            v_flat = torch.cat([v_flat, ev], dim=-1)

        # Attention scores: [B, nh, T, H, W, k_total(+G)]
        attn_scores = torch.matmul(q.unsqueeze(-2), k_flat).squeeze(-2) / math.sqrt(self.head_dim)

        # Softmax over local neighborhood
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Aggregate values: [B, nh, T, H, W, d]
        attn_out = torch.matmul(attn_weights.unsqueeze(-2), v_flat.transpose(-2, -1)).squeeze(-2)

        # 8. Reshape back to [B, N, D]
        attn_out = attn_out.permute(0, 2, 3, 4, 1, 5).contiguous().reshape(B, N, D)
        attn_out = self.out_proj(attn_out)

        # 9. Add depthwise conv term (paper Equation: DWConv_3D(X) + Attention(...))
        dw_flat = dw_out.reshape(B, D, N).transpose(1, 2)  # [B, N, D]

        return attn_out + dw_flat

    def _get_base_grid(self, T: int, H: int, W: int, device: torch.device) -> Tensor:
        """Generate base sampling grid in normalized coordinates [-1, 1].

        For each position (t, h, w), generates offsets to k_t × k_h × k_w neighbors.

        Args:
            T: Temporal dimension.
            H: Height dimension.
            W: Width dimension.
            device: Torch device.

        Returns:
            Grid of shape [T, H, W, k_t, k_h, k_w, 3] in normalized coords.
        """
        # Create coordinate grids for each dimension
        t_coords = torch.arange(T, device=device, dtype=torch.float32).view(T, 1, 1, 1, 1, 1)
        h_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1, 1, 1, 1)
        w_coords = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W, 1, 1, 1)

        # Kernel offsets (centered at 0)
        t_kernel = torch.arange(self.k_t, device=device, dtype=torch.float32) - self.k_t // 2
        h_kernel = torch.arange(self.k_h, device=device, dtype=torch.float32) - self.k_h // 2
        w_kernel = torch.arange(self.k_w, device=device, dtype=torch.float32) - self.k_w // 2

        t_k = t_kernel.view(1, 1, 1, self.k_t, 1, 1)
        h_k = h_kernel.view(1, 1, 1, 1, self.k_h, 1)
        w_k = w_kernel.view(1, 1, 1, 1, 1, self.k_w)

        # Absolute positions: [T, H, W, k_t, k_h, k_w]
        t_abs = t_coords + t_k
        h_abs = h_coords + h_k
        w_abs = w_coords + w_k

        # Normalize to [-1, 1] (grid_sample convention)
        t_norm = 2 * t_abs / max(T - 1, 1) - 1
        h_norm = 2 * h_abs / max(H - 1, 1) - 1
        w_norm = 2 * w_abs / max(W - 1, 1) - 1

        # Broadcast to common shape [T, H, W, k_t, k_h, k_w] before stacking
        w_norm, h_norm, t_norm = torch.broadcast_tensors(w_norm, h_norm, t_norm)

        # Stack to [T, H, W, k_t, k_h, k_w, 3]
        grid = torch.stack([w_norm, h_norm, t_norm], dim=-1)  # Note: WHD order for grid_sample

        return grid

    def _trilinear_sample_batched(self, volume: Tensor, grid: Tensor) -> Tensor:
        """Sample from 5D volume using trilinear interpolation.

        Args:
            volume: [B, num_heads, head_dim, T, H, W]
            grid: [B, num_heads, T, H, W, k_t, k_h, k_w, 3] in normalized coords [-1, 1]

        Returns:
            Sampled values [B, num_heads, head_dim, T, H, W, k_t, k_h, k_w]
        """
        B, nh, d, T, H, W = volume.shape
        _, _, T_q, H_q, W_q, kt, kh, kw, _ = grid.shape
        K = kt * kh * kw

        # Merge B and nh: volume [B*nh, d, T, H, W], grid [B*nh, T_q*H_q*W_q*K, 3]
        volume_flat = volume.reshape(B * nh, d, T, H, W)
        grid_flat = grid.reshape(B * nh, T_q * H_q * W_q * K, 3)

        # grid_sample on 5D input: [N, C, D, H, W] with grid [N, D_out, H_out, W_out, 3]
        # Reshape grid to [B*nh, T_q*H_q*W_q, K, 1, 3] to treat as 5D spatial output
        grid_for_sample = grid_flat.view(B * nh, T_q * H_q * W_q, K, 1, 3)

        sampled = F.grid_sample(
            volume_flat,
            grid_for_sample,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )  # [B*nh, d, T_q*H_q*W_q, K, 1]

        # Reshape back: [B, nh, d, T_q, H_q, W_q, kt, kh, kw]
        sampled = sampled.squeeze(-1).view(B, nh, d, T_q, H_q, W_q, kt, kh, kw)

        return sampled
