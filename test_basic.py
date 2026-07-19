#!/usr/bin/env python3
"""Basic smoke test for MegaSlide-DiT without full PyTorch."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("MEGASLIDE-DIT BASIC TESTS (No PyTorch Required)")
print("=" * 70)
print()

# Test 1: Import all modules
print("Test 1: Importing modules...")
try:
    from infinity.video import (
        MegaSlideConfig,
        load_megaslide_config,
    )
    print("✅ Config imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Create config from defaults
print("\nTest 2: Creating config from defaults...")
try:
    config = MegaSlideConfig()
    print(f"✅ Config created: {config.num_layers} layers, {config.hidden_size} hidden")
except Exception as e:
    print(f"❌ Config creation failed: {e}")
    sys.exit(1)

# Test 3: Load YAML config
print("\nTest 3: Loading YAML config...")
try:
    config_tiny = load_megaslide_config("examples/configs/megaslide_dit_tiny.yaml")
    print(f"✅ Tiny config loaded: {config_tiny.frames} frames, {config_tiny.hidden_size} hidden")
except Exception as e:
    print(f"❌ YAML loading failed: {e}")
    sys.exit(1)

# Test 4: Memory estimation
print("\nTest 4: Memory estimation...")
try:
    mem = config_tiny.estimate_memory_footprint()
    print(f"✅ Memory estimate: {mem['total_gb']:.2f} GB")
    print(f"   - Parameters: {mem['params_gb']:.3f} GB")
    print(f"   - Optimizer: {mem['optimizer_gb']:.3f} GB")
    print(f"   - Activations: {mem['activations_gb']:.3f} GB")
except Exception as e:
    print(f"❌ Memory estimation failed: {e}")
    sys.exit(1)

# Test 5: Check all configs load
print("\nTest 5: Loading all experiment configs...")
configs = [
    "examples/configs/megaslide_dit_tiny.yaml",
    "examples/configs/dense_baseline_64f.yaml",
    "examples/configs/swin_baseline_256f.yaml",
    "examples/configs/megaslide_paper_experiment_256f.yaml",
]

for config_path in configs:
    try:
        cfg = load_megaslide_config(config_path)
        print(f"✅ {os.path.basename(config_path)}: {cfg.frames}f @ {cfg.height}x{cfg.width}")
    except Exception as e:
        print(f"❌ {config_path} failed: {e}")

print()
print("=" * 70)
print("BASIC TESTS COMPLETE")
print("=" * 70)
print()
print("Next: Run with PyTorch installed for full model tests")
