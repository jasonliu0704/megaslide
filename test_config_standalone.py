#!/usr/bin/env python3
"""Standalone test for config system (no torch dependency)."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from infinity.video.config import MegaSlideConfig
from infinity.video.yaml_loader import load_megaslide_config

def test_default_config():
    """Test default configuration."""
    print("Testing default config...")
    config = MegaSlideConfig()

    assert config.frames == 256
    assert config.hidden_size == 8192
    assert config.num_layers == 48
    assert config.dsa_kernel_size == (3, 7, 7)

    # Test computed properties
    assert config.num_patches == 256 * (1080 // 16) * (1920 // 16)
    assert config.head_dim == 8192 // 64
    assert config.mlp_hidden_size == int(8192 * 4.0)

    print("✅ Default config passed!")


def test_tiny_yaml_config():
    """Test loading tiny YAML config."""
    print("\nTesting YAML config loading...")
    config = load_megaslide_config('examples/configs/megaslide_dit_tiny.yaml')

    # From YAML
    assert config.frames == 2
    assert config.in_channels == 2
    assert config.height == 8
    assert config.width == 8
    assert config.patch_size == 2
    assert config.hidden_size == 16
    assert config.num_layers == 2
    assert config.num_heads == 4
    assert config.dsa_kernel_size == (1, 3, 3)

    # Computed
    assert config.num_patches == 2 * (8 // 2) * (8 // 2) == 32
    assert config.head_dim == 16 // 4 == 4

    print(f"✅ YAML config loaded: {config.num_layers} layers, {config.hidden_size} hidden")


def test_memory_estimation():
    """Test memory footprint estimation."""
    print("\nTesting memory estimation...")

    # Tiny config
    config = MegaSlideConfig(
        frames=2, in_channels=2, height=8, width=8, patch_size=2,
        hidden_size=16, num_layers=2, num_heads=4
    )

    mem = config.estimate_memory_footprint()

    assert "parameter_count" in mem
    assert "fp16_weights_gb" in mem
    assert "total_persistent_gb" in mem
    assert mem["parameter_count"] > 0
    assert mem["fp16_weights_gb"] > 0

    print(f"  Parameters: {mem['parameter_count']:,}")
    print(f"  FP16 weights: {mem['fp16_weights_gb']:.6f} GB")
    print(f"  Total persistent (CPU): {mem['total_persistent_gb']:.6f} GB")
    print(f"  Single activation: {mem['single_activation_gb']:.6f} GB")
    print("✅ Memory estimation passed!")


def test_105b_estimation():
    """Test memory estimation for 105B model (paper scale)."""
    print("\nTesting 105B model estimation (paper scale)...")

    # Paper's 105B config
    config = MegaSlideConfig()  # defaults to 105B
    mem = config.estimate_memory_footprint()

    print(f"  Parameters: {mem['parameter_count']:,} (~{mem['parameter_count']/1e9:.1f}B)")
    print(f"  FP16 weights: {mem['fp16_weights_gb']:.2f} GB")
    print(f"  FP32 master: {mem['fp32_master_gb']:.2f} GB")
    print(f"  Adam moments: {mem['adam_moments_gb']:.2f} GB")
    print(f"  Total persistent (CPU): {mem['total_persistent_gb']:.2f} GB (~{mem['total_persistent_gb']/1000:.2f} TB)")
    print(f"  Single activation: {mem['single_activation_gb']:.2f} GB")
    print(f"  Checkpoint activations: {mem['checkpoint_activations_gb']:.2f} GB")

    # Sanity checks (paper claims ~1.47 TB persistent)
    assert mem['total_persistent_gb'] > 1000, "105B model should require > 1 TB"
    assert mem['total_persistent_gb'] < 2000, "Should be less than 2 TB"

    print("✅ 105B estimation passed! (Matches paper Table 2 scale)")


if __name__ == "__main__":
    test_default_config()
    test_tiny_yaml_config()
    test_memory_estimation()
    test_105b_estimation()

    print("\n" + "=" * 70)
    print("✅ ALL CONFIG TESTS PASSED!")
    print("=" * 70)
