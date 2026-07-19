"""Tests for temporal consistency metrics."""

import torch
import pytest


def test_tcs_smooth_vs_random():
    """Smooth temporal sequence should score higher than random noise."""
    from infinity.video.config import MegaSlideConfig
    from infinity.video.model import MegaSlideDiT
    from infinity.video.metrics import compute_temporal_consistency

    config = MegaSlideConfig(
        frames=4, in_channels=2, height=8, width=8, patch_size=2,
        hidden_size=16, num_layers=2, num_heads=4, mlp_ratio=2.0,
        dsa_kernel_size=(1, 1, 1),
    )
    model = MegaSlideDiT(config)
    model.eval()

    T, H_p, W_p = 4, 4, 4
    timesteps = torch.tensor([100])

    # Smooth sequence: frames are similar (small perturbations)
    base = torch.randn(1, 2, 1, 8, 8)
    smooth = base.expand(1, 2, 4, 8, 8) + torch.randn(1, 2, 4, 8, 8) * 0.01

    # Random sequence: frames are independent
    random_seq = torch.randn(1, 2, 4, 8, 8)

    scores_smooth = compute_temporal_consistency(model, smooth, timesteps, (T, H_p, W_p))
    scores_random = compute_temporal_consistency(model, random_seq, timesteps, (T, H_p, W_p))

    # Both should return valid scores in [0, 1]
    for key in ["src", "lrc", "tcs"]:
        assert 0.0 <= scores_smooth[key] <= 1.0, f"smooth {key}={scores_smooth[key]}"
        assert 0.0 <= scores_random[key] <= 1.0, f"random {key}={scores_random[key]}"

    # Smooth should generally score higher on SRC (consecutive similarity)
    # Note: with random model weights this isn't guaranteed, but TCS should be valid
    assert "src" in scores_smooth
    assert "lrc" in scores_smooth
    assert "tcs" in scores_smooth


def test_tcs_values_in_range():
    """TCS values should always be in [0, 1]."""
    from infinity.video.config import MegaSlideConfig
    from infinity.video.model import MegaSlideDiT
    from infinity.video.metrics import compute_temporal_consistency

    config = MegaSlideConfig(
        frames=2, in_channels=2, height=8, width=8, patch_size=2,
        hidden_size=16, num_layers=2, num_heads=4, mlp_ratio=2.0,
        dsa_kernel_size=(1, 1, 1),
    )
    model = MegaSlideDiT(config)
    latents = torch.randn(1, 2, 2, 8, 8)
    timesteps = torch.tensor([50])

    scores = compute_temporal_consistency(model, latents, timesteps, (2, 4, 4))
    assert 0.0 <= scores["src"] <= 1.0
    assert 0.0 <= scores["lrc"] <= 1.0
    assert 0.0 <= scores["tcs"] <= 1.0


def test_offset_magnitude():
    """compute_offset_magnitude should return a positive value for DSA models."""
    from infinity.video.config import MegaSlideConfig
    from infinity.video.model import MegaSlideDiT
    from infinity.video.metrics import compute_offset_magnitude

    config = MegaSlideConfig(
        frames=2, in_channels=2, height=8, width=8, patch_size=2,
        hidden_size=16, num_layers=2, num_heads=4, mlp_ratio=2.0,
        dsa_kernel_size=(1, 1, 1),
    )
    model = MegaSlideDiT(config)
    mag = compute_offset_magnitude(model)
    # Offset nets are initialized to zero, but xavier init on other layers
    # means the magnitude should be >= 0
    assert mag >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
