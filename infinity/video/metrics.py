"""Temporal consistency metrics for evaluating hybrid attention.

Measures short-range (consecutive frame) and long-range (distant frame)
feature similarity to quantify how well attention mechanisms maintain
temporal coherence.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def compute_temporal_consistency(
    model: nn.Module,
    latents: Tensor,
    timesteps: Tensor,
    video_shape: Tuple[int, int, int],
) -> Dict[str, float]:
    """Compute temporal consistency metrics using penultimate layer features.

    Args:
        model: A MegaSlideDiT-like model with .blocks attribute.
        latents: Input latents [B, C, T, H, W].
        timesteps: Diffusion timesteps [B].
        video_shape: (T, H_patches, W_patches) after patchification.

    Returns:
        Dict with 'src' (short-range), 'lrc' (long-range), 'tcs' (combined).
    """
    T, H_p, W_p = video_shape
    features = []

    # Hook into penultimate block to capture hidden states
    def hook_fn(module, input, output):
        features.append(output.detach())

    blocks = model.blocks if hasattr(model, 'blocks') else []
    if len(blocks) < 2:
        # Fallback: use output directly
        out = model(latents, timesteps)
        # Treat output channels as features per frame
        # out: [B, C, T, H, W] → per-frame features [B, T, C*H*W]
        B, C, T_out, H_out, W_out = out.shape
        frame_feats = out.permute(0, 2, 1, 3, 4).reshape(B, T_out, -1)
        return _compute_scores(frame_feats, T_out)

    hook = blocks[-2].register_forward_hook(hook_fn)
    try:
        model.eval()
        model(latents, timesteps)
    finally:
        hook.remove()
        model.train()

    if not features:
        return {"src": 0.0, "lrc": 0.0, "tcs": 0.0}

    hidden = features[0]  # [B, N, D]
    B, N, D = hidden.shape

    # Compute per-frame feature vectors via spatial average pooling
    # Reshape to [B, T, H*W, D] then mean over spatial
    frame_feats = hidden.view(B, T, H_p * W_p, D).mean(dim=2)  # [B, T, D]

    return _compute_scores(frame_feats, T)


def _compute_scores(frame_feats: Tensor, T: int) -> Dict[str, float]:
    """Compute SRC, LRC, TCS from per-frame feature vectors.

    Args:
        frame_feats: [B, T, D] per-frame features.
        T: Number of frames.

    Returns:
        Dict with src, lrc, tcs scores.
    """
    if T < 2:
        return {"src": 1.0, "lrc": 1.0, "tcs": 1.0}

    # Short-range consistency: cosine sim between consecutive frames
    f1 = F.normalize(frame_feats[:, :-1, :], dim=-1)
    f2 = F.normalize(frame_feats[:, 1:, :], dim=-1)
    src = (f1 * f2).sum(dim=-1).mean().item()

    # Long-range consistency: cosine sim between frames T//4 apart
    stride = max(T // 4, 1)
    if T > stride:
        f_near = F.normalize(frame_feats[:, :-stride, :], dim=-1)
        f_far = F.normalize(frame_feats[:, stride:, :], dim=-1)
        lrc = (f_near * f_far).sum(dim=-1).mean().item()
    else:
        lrc = src

    # Clamp to [0, 1]
    src = max(0.0, min(1.0, (src + 1.0) / 2.0))  # cosine sim is [-1,1], map to [0,1]
    lrc = max(0.0, min(1.0, (lrc + 1.0) / 2.0))
    tcs = 0.5 * src + 0.5 * lrc

    return {"src": src, "lrc": lrc, "tcs": tcs}


@torch.no_grad()
def compute_offset_magnitude(model: nn.Module) -> float:
    """Compute mean L2 norm of offset prediction network outputs.

    Useful to see if hybrid attention reduces the need for large offsets.

    Args:
        model: Model with DSA layers containing offset_net.

    Returns:
        Mean offset magnitude across all DSA layers, or 0.0 if none found.
    """
    magnitudes = []
    for module in model.modules():
        if hasattr(module, 'offset_net'):
            # Check the last layer's weight norm as proxy for offset magnitude
            last_layer = module.offset_net[-1]
            if hasattr(last_layer, 'weight'):
                magnitudes.append(last_layer.weight.data.norm().item())
            if hasattr(last_layer, 'bias') and last_layer.bias is not None:
                magnitudes.append(last_layer.bias.data.norm().item())

    return sum(magnitudes) / len(magnitudes) if magnitudes else 0.0
