import math

import numpy as np
import pytest
import torch

from infinity import (
    CPUMasterVideoDiT,
    DeformableSlideAttention3D,
    LatentVideoDataset,
    MegaSlideConfig,
    MegaSlideDiT,
    load_megaslide_config,
)


def tiny_config(**overrides):
    values = dict(
        frames=2,
        in_channels=2,
        height=4,
        width=4,
        patch_size=2,
        hidden_size=8,
        num_layers=2,
        num_heads=2,
        mlp_ratio=2.0,
        dsa_kernel_size=(1, 3, 3),
        synthetic_samples=3,
        batch_size=2,
        dtype=torch.float32,
    )
    values.update(overrides)
    return MegaSlideConfig(**values)


def test_deformable_slide_attention_shape_finite_and_gradients():
    torch.manual_seed(0)
    attn = DeformableSlideAttention3D(hidden_size=8, num_heads=2, kernel_size=(1, 3, 3))
    x = torch.randn(2, 18, 8, requires_grad=True)

    y = attn(x, (2, 3, 3))

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert any(param.grad is not None for param in attn.parameters())


def test_deformable_slide_attention_is_non_causal_over_time():
    attn = DeformableSlideAttention3D(hidden_size=2, num_heads=1, kernel_size=(3, 1, 1))
    with torch.no_grad():
        for param in attn.parameters():
            param.zero_()
        attn.v_proj.weight.copy_(torch.eye(2))
        attn.out_proj.weight.copy_(torch.eye(2))
        attn.dw_conv.weight.zero_()
        attn.dw_conv.bias.zero_()

    x = torch.zeros(1, 2, 2)
    changed = x.clone()
    changed[:, 1, :] = torch.tensor([3.0, -2.0])

    y = attn(x, (2, 1, 1))
    y_changed = attn(changed, (2, 1, 1))

    assert not torch.allclose(y[:, 0, :], y_changed[:, 0, :])


def test_deformable_slide_attention_handles_degenerate_grid():
    attn = DeformableSlideAttention3D(hidden_size=4, num_heads=2, kernel_size=(3, 3, 3))
    x = torch.randn(1, 1, 4, requires_grad=True)

    y = attn(x, (1, 1, 1))

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    final_offset = attn.offset_net[-1]
    assert torch.allclose(final_offset.weight, torch.zeros_like(final_offset.weight))
    assert torch.allclose(final_offset.bias, torch.zeros_like(final_offset.bias))


def test_megaslide_dit_shape_and_backward():
    torch.manual_seed(0)
    config = tiny_config()
    model = MegaSlideDiT(config)
    latents = torch.randn(2, config.in_channels, config.frames, config.height, config.width)
    timesteps = torch.tensor([1, 9])

    pred = model(latents, timesteps)
    loss = pred.square().mean()
    loss.backward()

    assert pred.shape == latents.shape
    assert torch.isfinite(pred).all()
    assert any(param.grad is not None for param in model.parameters())


def test_cpu_master_video_dit_cpu_fallback_step():
    torch.manual_seed(0)
    config = tiny_config(batch_size=1)
    trainer = CPUMasterVideoDiT(MegaSlideDiT(config), config, force_cpu=True)
    latents = torch.randn(1, config.in_channels, config.frames, config.height, config.width)
    target = torch.randn_like(latents)
    timesteps = torch.tensor([4])

    loss, timing = trainer.forward_and_backward(latents, timesteps, target)
    grad_norm = trainer.optimizer_step()

    assert math.isfinite(loss)
    assert timing["total"] >= timing["forward"]
    assert math.isfinite(grad_norm)


def test_latent_video_dataset_synthetic_and_files(tmp_path):
    config = tiny_config(in_channels=3, hidden_size=12, num_heads=3)
    synthetic = LatentVideoDataset(config)
    assert len(synthetic) == config.synthetic_samples
    assert synthetic[0]["latents"].shape == (3, 2, 4, 4)

    cthw_path = tmp_path / "sample.pt"
    torch.save(torch.randn(3, 2, 4, 4), cthw_path)
    cthw = LatentVideoDataset(config, path=str(cthw_path))
    assert cthw[0]["latents"].shape == (3, 2, 4, 4)

    tchw_path = tmp_path / "sample.npy"
    np.save(tchw_path, np.random.randn(2, 3, 4, 4).astype("float32"))
    tchw = LatentVideoDataset(config, path=str(tchw_path))
    assert tchw[0]["latents"].shape == (3, 2, 4, 4)


def test_megaslide_yaml_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model:
  frames: 2
  in_channels: 2
  height: 4
  width: 4
  patch_size: 2
  hidden_size: 8
  num_layers: 1
  num_heads: 2
  dsa_kernel_size: [1, 3, 3]
  dtype: "float32"
dataset:
  synthetic_samples: 5
training:
  batch_size: 2
memory:
  checkpoint_interval: 1
""",
        encoding="utf-8",
    )

    config = load_megaslide_config(str(path))

    assert config.dsa_kernel_size == (1, 3, 3)
    assert config.num_patches == 8
    assert config.synthetic_samples == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cpu_master_video_dit_cuda_streaming_smoke():
    torch.manual_seed(0)
    config = tiny_config(batch_size=1, num_grad_slabs=2, dropout=0.2)
    trainer = CPUMasterVideoDiT(MegaSlideDiT(config), config)
    latents = torch.randn(1, config.in_channels, config.frames, config.height, config.width)
    target = torch.randn_like(latents)
    timesteps = torch.tensor([2])

    loss, timing = trainer.forward_and_backward(latents, timesteps, target)
    grad_norm = trainer.optimizer_step()

    assert math.isfinite(loss)
    assert timing["total"] > 0
    assert math.isfinite(grad_norm)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_megaslide_dit_accepts_cpu_conditioning_tensors_on_cuda():
    torch.manual_seed(0)
    config = tiny_config()
    model = MegaSlideDiT(config).cuda()
    latents = torch.randn(1, config.in_channels, config.frames, config.height, config.width, device="cuda")
    timesteps = torch.tensor([3], device="cpu")
    text_embeds = torch.randn(1, 2, config.hidden_size, device="cpu")
    text_mask = torch.ones(1, 2, device="cpu")

    pred = model(latents, timesteps, text_embeds=text_embeds, text_mask=text_mask)

    assert pred.shape == latents.shape
    assert pred.is_cuda
