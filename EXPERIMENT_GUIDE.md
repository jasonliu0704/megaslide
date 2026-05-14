# Phase 3 Experiment Guide

**Status:** ⚠️ Requires PyTorch installation  
**Date:** 2026-05-13

---

## Prerequisites

### Install Dependencies
```bash
pip install torch>=2.0 pyyaml numpy pytest

# Optional for VBench evaluation:
pip install vbench transformers diffusers
```

### Check Installation
```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Quick Start

### Option 1: Run All Experiments (Automated)
```bash
./run_phase3_experiments.sh
```

This runs:
1. Tiny model smoke tests (all 3 models)
2. Parameter count comparisons
3. Ablation studies (if CUDA available)
4. Memory scaling tests

**Expected time:** 5-10 minutes for tiny configs

---

### Option 2: Run Individual Experiments

#### Experiment 1: Baseline Comparison (Tiny Config)

**Test MegaSlide-DiT:**
```bash
python examples/train_megaslide_dit.py \
    --config examples/configs/megaslide_dit_tiny.yaml
```

**Expected output:**
```
[CPUMasterVideoDiT] Running on cpu
Step 1/2 | loss 0.1234 | avg 0.1234 | step 0.15s | fwd 0.08s | bwd 0.07s
Step 2/2 | loss 0.1100 | avg 0.1167 | step 0.14s | fwd 0.07s | bwd 0.07s
Training complete
```

**Test Dense 3D-DiT:**
```python
from infinity.video import Dense3DDiT, CPUMasterVideoDiT, load_megaslide_config

config = load_megaslide_config('examples/configs/dense_baseline_64f.yaml')
model = Dense3DDiT(config)
trainer = CPUMasterVideoDiT(model, config)

# Train for a few steps...
```

**Test Swin-DiT:**
```python
from infinity.video import SwinDiT, CPUMasterVideoDiT, load_megaslide_config

config = load_megaslide_config('examples/configs/swin_baseline_256f.yaml')
model = SwinDiT(config)
trainer = CPUMasterVideoDiT(model, config)

# Train for a few steps...
```

---

#### Experiment 2: Parameter Counts

**Compare all three models:**
```python
from infinity.video import MegaSlideDiT, Dense3DDiT, SwinDiT, load_megaslide_config

config = load_megaslide_config('examples/configs/megaslide_dit_tiny.yaml')

models = {
    "MegaSlide-DiT": MegaSlideDiT(config),
    "Dense 3D-DiT": Dense3DDiT(config),
    "Swin-DiT": SwinDiT(config),
}

for name, model in models.items():
    params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {params:,} parameters")
```

**Expected output (tiny config):**
```
MegaSlide-DiT: ~5,000 parameters
Dense 3D-DiT:  ~5,000 parameters
Swin-DiT:      ~5,000 parameters
```

**Expected for 105B config:**
```
MegaSlide-DiT: ~105,000,000,000 parameters
Dense 3D-DiT:  ~105,000,000,000 parameters
Swin-DiT:      ~105,000,000,000 parameters
```

---

#### Experiment 3: Ablation Studies

**Ablation 1: Fixed Windows vs Learned Offsets**
```bash
python examples/run_ablation_studies.py \
    --config examples/configs/megaslide_dit_tiny.yaml \
    --ablation fixed_windows \
    --num_steps 100
```

**Expected results:**
- Loss converges slower than MegaSlide-DiT
- VBench-Consist would drop from 0.88 to ~0.81 (similar to Swin-DiT)
- VBench-Align unchanged (~0.83)

---

**Ablation 2: Async Prefetch vs Sync Transfers**
```bash
python examples/run_ablation_studies.py \
    --config examples/configs/megaslide_dit_tiny.yaml \
    --ablation sync_transfer \
    --num_steps 100
```

**Expected results:**
- With async (baseline): Step time ~3.1s, MFU 61%
- With sync: Step time ~6.8s (2.2× slower), MFU 28%

**Note:** Requires CUDA. CPU mode doesn't benefit from async streaming.

---

**Ablation 3: CPU Optimizer vs GPU Optimizer**
```bash
python examples/run_ablation_studies.py \
    --config examples/configs/megaslide_dit_tiny.yaml \
    --ablation gpu_optimizer \
    --num_steps 100
```

**Expected results:**
- Demonstrates why CPU-master is necessary
- GPU optimizer would require ~300 GB HBM for optimizer state alone (105B model)
- With activations (~115 GB), total exceeds any single GPU

---

#### Experiment 4: Memory Scaling

**Test different frame counts:**
```python
from infinity.video import MegaSlideDiT, Dense3DDiT, SwinDiT, load_megaslide_config
import torch

config = load_megaslide_config('examples/configs/megaslide_dit_tiny.yaml')

for frames in [2, 4, 8, 16, 32, 64]:
    config.frames = frames
    
    # MegaSlide-DiT
    model = MegaSlideDiT(config)
    torch.cuda.reset_peak_memory_stats()
    # ... run forward/backward ...
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"{frames:3d} frames: {peak_mem:.2f} GB")
```

**Expected memory usage (105B model, 1080p):**
| Frames | MegaSlide-DiT | Dense 3D-DiT | Swin-DiT |
|--------|---------------|--------------|----------|
| 16     | ~30 GB        | ~30 GB       | ~30 GB   |
| 32     | ~50 GB        | ~60 GB       | ~50 GB   |
| 64     | ~80 GB        | ~115 GB      | ~80 GB   |
| 256    | ~115 GB       | **OOM**      | ~115 GB  |

**Key observation:** Dense OOMs at 256 frames due to O(N²) complexity.

---

#### Experiment 5: VBench Evaluation

**Prerequisites:**
```bash
pip install vbench transformers diffusers
```

**Run evaluation (small scale test):**
```bash
python examples/run_vbench_evaluation.py \
    --config examples/configs/megaslide_paper_experiment_256f.yaml \
    --checkpoint checkpoints/megaslide_5k.pt \
    --model_type megaslide \
    --num_prompts 10 \
    --num_inference_steps 30
```

**Expected output:**
```
Generating videos...
[1/10] Generating: A cat playing with a ball...
[2/10] Generating: A sunset over the ocean...
...
✓ Generated 10 videos

Evaluating with VBench...

VBENCH RESULTS
======================================================================
VBench-Align:   0.83 ± 0.02
VBench-Consist: 0.88 ± 0.02
======================================================================
```

**Full scale (paper reproduction):**
```bash
python examples/run_vbench_evaluation.py \
    --config examples/configs/megaslide_paper_experiment_256f.yaml \
    --checkpoint checkpoints/megaslide_5k.pt \
    --model_type megaslide \
    --num_prompts 300 \
    --num_inference_steps 30 \
    --save_videos
```

**Expected time:** ~2-3 hours for 300 prompts (depends on GPU)

---

## Expected Results Summary

### Paper Table 3: VBench Scores

| Model | VBench-Align | VBench-Consist | Frames | Memory (HBM) |
|-------|--------------|----------------|--------|--------------|
| **Dense 3D-DiT** | 0.78 ± 0.02 | 0.85 ± 0.03 | 64 (max) | ~115 GB |
| **Swin-DiT** | 0.81 ± 0.02 | 0.79 ± 0.03 | 256 | ~115 GB |
| **MegaSlide-DiT** | **0.83 ± 0.02** | **0.88 ± 0.02** | 256 | ~115 GB |

**Key findings:**
1. **Dense 3D-DiT:**
   - Best temporal consistency at 64 frames (0.85)
   - OOMs at 256 frames (O(N²) complexity)
   - Lower alignment than MegaSlide (0.78 vs 0.83)

2. **Swin-DiT:**
   - Scales to 256 frames (fixed window attention)
   - Lowest temporal consistency (0.79) - fixed windows can't adapt to motion
   - Better alignment than Dense (0.81 vs 0.78)

3. **MegaSlide-DiT:**
   - Best alignment (0.83) and best consistency (0.88)
   - Scales to 256 frames with learned deformable attention
   - Adapts to motion patterns via learned offsets

---

### Paper Section 8: Ablation Studies

**8.1 Fixed Windows vs Learned Offsets:**
- Fixed windows (frozen offsets): VBench-Consist 0.88 → 0.81
- Impact: Temporal consistency drops significantly
- Conclusion: Learned offsets are crucial for motion adaptation

**8.2 Async Prefetch vs Sync Transfers:**
- Async (baseline): Step time 3.1s, MFU 61%
- Sync: Step time 6.8s (2.2× slower), MFU 28%
- Impact: Double-buffering provides 2.2× speedup
- Conclusion: Async overlapping is essential for efficiency

**8.3 CPU Optimizer vs GPU Optimizer:**
- CPU optimizer: Fits on single H200 (~115 GB HBM)
- GPU optimizer: Would need ~415 GB HBM (210 GB params + 420 GB master + 840 GB moments)
- Impact: OOM with GPU-resident optimizer
- Conclusion: CPU-master architecture enables 105B model training

---

## Troubleshooting

### Issue 1: ModuleNotFoundError
```bash
# Fix:
pip install torch pyyaml numpy
```

### Issue 2: CUDA out of memory
```bash
# Use smaller config:
python examples/train_megaslide_dit.py \
    --config examples/configs/megaslide_dit_tiny.yaml
```

### Issue 3: Dense baseline OOM at 256 frames
**Expected behavior!** Dense uses O(N²) attention and cannot fit 256 frames.
```bash
# Use 64 frames instead:
python examples/train_megaslide_dit.py \
    --config examples/configs/dense_baseline_64f.yaml
```

### Issue 4: VBench not installed
```bash
# Install VBench dependencies:
pip install vbench transformers diffusers
```

### Issue 5: Slow CPU training
**Expected!** Tiny config is for smoke tests only. Real training requires GPU.
```bash
# Check if CUDA available:
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## Next Steps After Experiments

### 1. Unit Tests
```bash
pytest tests/test_megaslide_video.py -v
```

**Expected:** All 8 tests pass

---

### 2. Phase 4: Profiling & Metrics

Implement:
- Memory profiling (reproduce paper Table 2)
- MFU calculation (verify 61% with async)
- Bandwidth analysis (18 GB H2D + 18 GB D2H)

---

### 3. Full-Scale Experiments (Requires H200 + 1.5 TB RAM)

**Training:**
```bash
python examples/train_megaslide_dit.py \
    --config examples/configs/megaslide_paper_experiment_256f.yaml
```

**VBench Evaluation:**
```bash
python examples/run_vbench_evaluation.py \
    --config examples/configs/megaslide_paper_experiment_256f.yaml \
    --checkpoint checkpoints/megaslide_5k.pt \
    --model_type megaslide \
    --num_prompts 300 \
    --save_videos
```

**Expected training time:** ~4-5 hours for 5,000 steps

---

## Summary

**What to run:**
1. `./run_phase3_experiments.sh` - Automated test suite
2. Individual experiments as needed
3. VBench evaluation (requires vbench package)

**What to expect:**
- All 3 models train without errors on tiny configs
- Dense OOMs at 256 frames (expected)
- Swin and MegaSlide scale to 256 frames
- MegaSlide achieves best VBench scores
- Ablations demonstrate importance of learned offsets and async streaming

**Hardware requirements:**
- **Development:** Any GPU (>= 2 GB for tiny config)
- **Full 105B model:** H200 (141 GB HBM) + 1.5 TB CPU RAM

---

**Status:** Ready to run once PyTorch is installed!
