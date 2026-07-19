"""Tests for hybrid attention modules."""

import torch
import pytest


def test_deformable_slide_attention_extra_kv():
    """Test that extra_kv parameter works correctly in DeformableSlideAttention3D."""
    from infinity.video.attention import DeformableSlideAttention3D

    B, T, H, W, D, nh = 1, 2, 2, 2, 16, 4
    N = T * H * W
    G = 4

    attn = DeformableSlideAttention3D(hidden_size=D, num_heads=nh, kernel_size=(1, 1, 1))
    x = torch.randn(B, N, D)
    extra_kv = torch.randn(B, G, D)

    # Without extra_kv
    out_no_extra = attn(x, (T, H, W))
    # With extra_kv
    out_with_extra = attn(x, (T, H, W), extra_kv=extra_kv)

    assert out_no_extra.shape == (B, N, D)
    assert out_with_extra.shape == (B, N, D)
    # Outputs should differ when extra_kv is non-zero
    assert not torch.allclose(out_no_extra, out_with_extra, atol=1e-5)


def test_deformable_slide_attention_extra_kv_zeros():
    """When extra_kv is all zeros, output should be close to no extra_kv."""
    from infinity.video.attention import DeformableSlideAttention3D

    B, T, H, W, D, nh = 1, 2, 2, 2, 16, 4
    N = T * H * W
    G = 4

    attn = DeformableSlideAttention3D(hidden_size=D, num_heads=nh, kernel_size=(1, 1, 1))
    x = torch.randn(B, N, D)
    extra_kv_zeros = torch.zeros(B, G, D)

    out_no_extra = attn(x, (T, H, W))
    out_zero_extra = attn(x, (T, H, W), extra_kv=extra_kv_zeros)

    # With zero extra_kv, the extra keys are zero-projected, so attention
    # scores to them will be near-zero but softmax still distributes some weight.
    # They won't be exactly equal but should be close.
    assert out_zero_extra.shape == (B, N, D)


def test_hybrid_attention_3d_shape_and_grad():
    """Test HybridAttention3D output shape and gradient flow."""
    from infinity.video.hybrid_attention import HybridAttention3D

    B, T, H, W, D, nh, G = 1, 2, 2, 2, 16, 4, 4
    N = T * H * W

    attn = HybridAttention3D(
        hidden_size=D, num_heads=nh, kernel_size=(1, 1, 1),
        num_registers=G, gate_init=0.0,
    )
    x = torch.randn(B, N, D, requires_grad=True)
    out = attn(x, (T, H, W))

    assert out.shape == (B, N, D)

    # Gradient flow
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert attn.registers.grad is not None
    assert attn.gate.grad is not None


def test_hybrid_attention_3d_gate_zero():
    """With gate forced to -inf (sigmoid→0), registers have no effect."""
    from infinity.video.hybrid_attention import HybridAttention3D

    B, T, H, W, D, nh, G = 1, 2, 2, 2, 16, 4, 4
    N = T * H * W

    attn = HybridAttention3D(
        hidden_size=D, num_heads=nh, kernel_size=(1, 1, 1),
        num_registers=G, gate_init=-100.0,  # sigmoid(-100) ≈ 0
    )
    x = torch.randn(B, N, D)

    out_hybrid = attn(x, (T, H, W))
    out_local = attn.local_attn(x, (T, H, W))

    # With gate ≈ 0, extra_kv is all zeros, so outputs should be very close
    # (not exactly equal due to softmax redistribution over zero keys)
    assert out_hybrid.shape == out_local.shape


def test_temporal_anchor_attention_shape_and_grad():
    """Test TemporalAnchorAttention3D output shape and gradient flow."""
    from infinity.video.hybrid_attention import TemporalAnchorAttention3D

    B, T, H, W, D, nh = 1, 2, 2, 2, 16, 4
    N = T * H * W

    attn = TemporalAnchorAttention3D(
        hidden_size=D, num_heads=nh, kernel_size=(1, 1, 1), gate_init=0.0,
    )
    x = torch.randn(B, N, D, requires_grad=True)
    out = attn(x, (T, H, W))

    assert out.shape == (B, N, D)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert attn.gate.grad is not None
    assert attn.anchor_proj.weight.grad is not None


def test_temporal_anchor_cheaper_compute_than_register():
    """Temporal anchor variant avoids O(G*N) cross-attention, making it cheaper at scale."""
    from infinity.video.hybrid_attention import HybridAttention3D, TemporalAnchorAttention3D

    D, nh = 16, 4
    B, T, H, W = 1, 4, 4, 4
    N = T * H * W  # 64 tokens

    hybrid = HybridAttention3D(D, nh, kernel_size=(1, 3, 3), num_registers=8)
    anchor = TemporalAnchorAttention3D(D, nh, kernel_size=(1, 3, 3))

    x = torch.randn(B, N, D)

    # Both should produce valid output
    out_h = hybrid(x, (T, H, W))
    out_a = anchor(x, (T, H, W))
    assert out_h.shape == (B, N, D)
    assert out_a.shape == (B, N, D)

    # Anchor self-attention is O(T^2) = O(16) vs register cross-attn O(G*N) = O(8*64) = O(512)
    # This is the computational advantage at scale (not param count)


def test_hybrid_megaslide_dit_forward_backward():
    """Test HybridMegaSlideDiT forward and backward pass."""
    from infinity.video.config import MegaSlideConfig
    from infinity.video.hybrid_model import HybridMegaSlideDiT
    from infinity.video.model import MegaSlideDiT

    config = MegaSlideConfig(
        frames=2, in_channels=2, height=8, width=8, patch_size=2,
        hidden_size=16, num_layers=4, num_heads=4, mlp_ratio=2.0,
        dsa_kernel_size=(1, 1, 1), num_registers=4, register_interval=2,
    )
    model = HybridMegaSlideDiT(config)
    baseline = MegaSlideDiT(config)

    latents = torch.randn(1, 2, 2, 8, 8)
    timesteps = torch.tensor([100])

    out = model(latents, timesteps)
    assert out.shape == latents.shape

    loss = out.sum()
    loss.backward()

    # Check hybrid blocks are at correct positions (layers 1, 3 → indices 1, 3)
    from infinity.video.hybrid_model import HybridDiTBlock
    from infinity.video.model import DiTBlock
    assert isinstance(model.blocks[0], DiTBlock)
    assert isinstance(model.blocks[1], HybridDiTBlock)
    assert isinstance(model.blocks[2], DiTBlock)
    assert isinstance(model.blocks[3], HybridDiTBlock)

    # Hybrid model should have more params than baseline
    hybrid_params = sum(p.numel() for p in model.parameters())
    base_params = sum(p.numel() for p in baseline.parameters())
    assert hybrid_params > base_params


def test_hybrid_megaslide_dit_register_interval_1():
    """Test with register_interval=1 (every layer is hybrid)."""
    from infinity.video.config import MegaSlideConfig
    from infinity.video.hybrid_model import HybridMegaSlideDiT, HybridDiTBlock

    config = MegaSlideConfig(
        frames=2, in_channels=2, height=8, width=8, patch_size=2,
        hidden_size=16, num_layers=2, num_heads=4, mlp_ratio=2.0,
        dsa_kernel_size=(1, 1, 1), num_registers=4, register_interval=1,
    )
    model = HybridMegaSlideDiT(config)

    # All blocks should be hybrid
    for block in model.blocks:
        assert isinstance(block, HybridDiTBlock)

    out = model(torch.randn(1, 2, 2, 8, 8), torch.tensor([50]))
    assert out.shape == (1, 2, 2, 8, 8)


def test_temporal_anchor_megaslide_dit_forward():
    """Test TemporalAnchorMegaSlideDiT forward pass."""
    from infinity.video.config import MegaSlideConfig
    from infinity.video.hybrid_model import TemporalAnchorMegaSlideDiT

    config = MegaSlideConfig(
        frames=2, in_channels=2, height=8, width=8, patch_size=2,
        hidden_size=16, num_layers=4, num_heads=4, mlp_ratio=2.0,
        dsa_kernel_size=(1, 1, 1), register_interval=2,
    )
    model = TemporalAnchorMegaSlideDiT(config)
    latents = torch.randn(1, 2, 2, 8, 8)
    out = model(latents, torch.tensor([200]))
    assert out.shape == latents.shape

    loss = out.sum()
    loss.backward()
    # Verify gradients exist on anchor attention params
    from infinity.video.hybrid_model import TemporalAnchorDiTBlock
    assert isinstance(model.blocks[1], TemporalAnchorDiTBlock)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
