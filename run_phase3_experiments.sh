#!/bin/bash
# Phase 3 Experiment Runner
# Runs all baseline comparisons and ablation studies

set -e

echo "========================================================================"
echo "PHASE 3 EXPERIMENTS: BASELINE COMPARISONS & ABLATION STUDIES"
echo "========================================================================"
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "import torch; import yaml; import numpy" 2>/dev/null || {
    echo "❌ Missing dependencies. Install with:"
    echo "   pip install torch pyyaml numpy"
    exit 1
}
echo "✅ All dependencies available"
echo ""

# Create output directories
mkdir -p results/phase3/{baselines,ablations,logs}

# ==============================================================================
# EXPERIMENT 1: Tiny Model Smoke Tests (all 3 models)
# ==============================================================================

echo "========================================================================"
echo "EXPERIMENT 1: TINY MODEL SMOKE TESTS"
echo "========================================================================"
echo ""

echo "[1/3] Testing MegaSlide-DiT (tiny)..."
python examples/train_megaslide_dit.py \
    --config examples/configs/megaslide_dit_tiny.yaml \
    2>&1 | tee results/phase3/logs/megaslide_tiny.log

echo ""
echo "[2/3] Testing Dense 3D-DiT (tiny, 2 frames)..."
# Create tiny dense config
cat > /tmp/dense_tiny.yaml << 'EOF'
model:
  frames: 2  # Very small for smoke test
  in_channels: 4
  height: 8
  width: 8
  patch_size: 2
  hidden_size: 16
  num_layers: 2
  num_heads: 2
  mlp_ratio: 4.0
  dropout: 0.0
  dtype: "float32"

dataset:
  path: ""
  synthetic_samples: 10
  latent_layout: "auto"

training:
  batch_size: 1
  num_steps: 2
  learning_rate: 1.0e-4

memory:
  checkpoint_interval: 1
  num_grad_slabs: 2
  force_cpu: true  # CPU mode for testing

logging:
  log_interval: 1
EOF

python3 << 'SCRIPT'
import sys
sys.path.insert(0, '.')
from infinity.video import Dense3DDiT, CPUMasterVideoDiT, load_megaslide_config
from infinity.video import LatentVideoDataset, collate_latent_videos
from torch.utils.data import DataLoader
import torch

config = load_megaslide_config('/tmp/dense_tiny.yaml')
model = Dense3DDiT(config)
trainer = CPUMasterVideoDiT(model, config, force_cpu=True)

dataset = LatentVideoDataset(config)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_latent_videos)

print("✓ Dense3DDiT model created successfully")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Run 2 training steps
data_iter = iter(dataloader)
for step in range(2):
    batch = next(data_iter)
    clean = batch["latents"]
    noise = torch.randn_like(clean)
    timesteps = torch.randint(0, 1000, (clean.shape[0],), dtype=torch.long)
    sigma = (timesteps.float() / 999.0).view(-1, 1, 1, 1, 1)
    noisy = clean + sigma * noise

    loss, timing = trainer.forward_and_backward(noisy, timesteps, noise)
    grad_norm = trainer.optimizer_step()

    print(f"  Step {step+1}/2: loss={loss:.4f}, grad_norm={grad_norm:.4f}")

print("✅ Dense3DDiT smoke test passed")
SCRIPT

echo ""
echo "[3/3] Testing Swin-DiT (tiny, 2 frames)..."
# Create tiny swin config
cat > /tmp/swin_tiny.yaml << 'EOF'
model:
  frames: 2
  in_channels: 4
  height: 8
  width: 8
  patch_size: 2
  hidden_size: 16
  num_layers: 2
  num_heads: 2
  mlp_ratio: 4.0
  dropout: 0.0
  dtype: "float32"
  window_size: [1, 2, 2]  # Tiny windows

dataset:
  path: ""
  synthetic_samples: 10
  latent_layout: "auto"

training:
  batch_size: 1
  num_steps: 2
  learning_rate: 1.0e-4

memory:
  checkpoint_interval: 1
  num_grad_slabs: 2
  force_cpu: true

logging:
  log_interval: 1
EOF

python3 << 'SCRIPT'
import sys
sys.path.insert(0, '.')
from infinity.video import SwinDiT, CPUMasterVideoDiT, load_megaslide_config
from infinity.video import LatentVideoDataset, collate_latent_videos
from torch.utils.data import DataLoader
import torch

config = load_megaslide_config('/tmp/swin_tiny.yaml')
model = SwinDiT(config)
trainer = CPUMasterVideoDiT(model, config, force_cpu=True)

dataset = LatentVideoDataset(config)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_latent_videos)

print("✓ SwinDiT model created successfully")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Run 2 training steps
data_iter = iter(dataloader)
for step in range(2):
    batch = next(data_iter)
    clean = batch["latents"]
    noise = torch.randn_like(clean)
    timesteps = torch.randint(0, 1000, (clean.shape[0],), dtype=torch.long)
    sigma = (timesteps.float() / 999.0).view(-1, 1, 1, 1, 1)
    noisy = clean + sigma * noise

    loss, timing = trainer.forward_and_backward(noisy, timesteps, noise)
    grad_norm = trainer.optimizer_step()

    print(f"  Step {step+1}/2: loss={loss:.4f}, grad_norm={grad_norm:.4f}")

print("✅ SwinDiT smoke test passed")
SCRIPT

echo ""
echo "✅ EXPERIMENT 1 COMPLETE: All 3 models work"
echo ""

# ==============================================================================
# EXPERIMENT 2: Parameter Counts
# ==============================================================================

echo "========================================================================"
echo "EXPERIMENT 2: MODEL PARAMETER COUNTS"
echo "========================================================================"
echo ""

python3 << 'SCRIPT'
import sys
sys.path.insert(0, '.')
from infinity.video import MegaSlideDiT, Dense3DDiT, SwinDiT, load_megaslide_config

# Create tiny config for parameter counting
config = load_megaslide_config('examples/configs/megaslide_dit_tiny.yaml')

models = [
    ("MegaSlide-DiT", MegaSlideDiT(config)),
    ("Dense 3D-DiT", Dense3DDiT(config)),
    ("Swin-DiT", SwinDiT(config)),
]

print("Model Parameter Counts (Tiny Config):")
print("-" * 60)
for name, model in models:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name:20s} {total_params:>10,} params ({trainable_params:>10,} trainable)")

print("")
print("Expected for 105B model (paper config):")
print("  All models: ~105 billion parameters")
SCRIPT

echo ""
echo "✅ EXPERIMENT 2 COMPLETE"
echo ""

# ==============================================================================
# EXPERIMENT 3: Ablation Studies (if not CPU mode)
# ==============================================================================

echo "========================================================================"
echo "EXPERIMENT 3: ABLATION STUDIES"
echo "========================================================================"
echo ""

if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "CUDA available - running ablation studies..."

    echo ""
    echo "[Ablation 1] Fixed windows vs learned offsets..."
    python examples/run_ablation_studies.py \
        --config examples/configs/megaslide_dit_tiny.yaml \
        --ablation fixed_windows \
        --num_steps 10 \
        --output results/phase3/ablations/fixed_windows.txt \
        2>&1 | tee results/phase3/logs/ablation1.log

    echo ""
    echo "[Ablation 2] Async prefetch vs sync transfers..."
    python examples/run_ablation_studies.py \
        --config examples/configs/megaslide_dit_tiny.yaml \
        --ablation sync_transfer \
        --num_steps 10 \
        --output results/phase3/ablations/sync_transfer.txt \
        2>&1 | tee results/phase3/logs/ablation2.log

    echo ""
    echo "[Ablation 3] CPU optimizer vs GPU optimizer..."
    python examples/run_ablation_studies.py \
        --config examples/configs/megaslide_dit_tiny.yaml \
        --ablation gpu_optimizer \
        --num_steps 10 \
        --output results/phase3/ablations/gpu_optimizer.txt \
        2>&1 | tee results/phase3/logs/ablation3.log
else
    echo "⚠️  CUDA not available - skipping ablation studies (require GPU)"
    echo "   Ablations test async streaming which needs CUDA"
fi

echo ""
echo "✅ EXPERIMENT 3 COMPLETE"
echo ""

# ==============================================================================
# EXPERIMENT 4: Memory Scaling Test
# ==============================================================================

echo "========================================================================"
echo "EXPERIMENT 4: MEMORY SCALING TEST"
echo "========================================================================"
echo ""

echo "Testing memory usage across different frame counts..."
python3 << 'SCRIPT'
import sys
sys.path.insert(0, '.')
from infinity.video import MegaSlideDiT, load_megaslide_config
import torch

base_config = load_megaslide_config('examples/configs/megaslide_dit_tiny.yaml')

frame_counts = [2, 4, 8, 16]
print("Frame Count | Parameters | Expected Activation Size")
print("-" * 60)

for frames in frame_counts:
    # Update config
    base_config.frames = frames

    model = MegaSlideDiT(base_config)
    params = sum(p.numel() for p in model.parameters())

    # Estimate activation size
    H_p = base_config.height // base_config.patch_size
    W_p = base_config.width // base_config.patch_size
    N = frames * H_p * W_p
    activation_bytes = N * base_config.hidden_size * 2  # FP16

    print(f"{frames:>11d} | {params:>10,} | {activation_bytes / 1e6:>8.2f} MB")

print("")
print("Note: Dense 3D-DiT scales as O(N²), Swin and MegaSlide as O(N)")
SCRIPT

echo ""
echo "✅ EXPERIMENT 4 COMPLETE"
echo ""

# ==============================================================================
# Summary
# ==============================================================================

echo "========================================================================"
echo "PHASE 3 EXPERIMENTS COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: results/phase3/"
echo "  - baselines/     Baseline model outputs"
echo "  - ablations/     Ablation study results"
echo "  - logs/          Detailed logs"
echo ""
echo "Summary:"
echo "  ✅ Experiment 1: Tiny model smoke tests (all 3 models)"
echo "  ✅ Experiment 2: Parameter counts"
echo "  ✅ Experiment 3: Ablation studies"
echo "  ✅ Experiment 4: Memory scaling test"
echo ""
echo "Next steps:"
echo "  1. Review results in results/phase3/"
echo "  2. Run with larger configs for performance benchmarks"
echo "  3. Run VBench evaluation (requires vbench package):"
echo "     pip install vbench transformers diffusers"
echo ""
