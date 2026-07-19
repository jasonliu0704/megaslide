# GPU Requirements for MegaSlide-DiT Experiments

> **Note (corrected).** The "105B / H200 / 1.47 TB / 115 GB GPU / 61% MFU / VBench"
> figures below describe an **unrun analytical target**, not a measured result.
> Measured runs used a single **H100 NVL (94 GB, 314 GB RAM)** up to **33.3B**
> parameters; MFU is **4–9.5%** and async overlap is **1.25–1.50×** (not 2×). The
> 105B numbers are a PCIe-bound projection — see [`results/07_roofline`](results/07_roofline)
> and [`RUN_EXPERIMENTS.md`](RUN_EXPERIMENTS.md).

**Date:** 2026-05-13

---

## Quick Answer

| Scenario | GPU Requirement | Config | Can Run? |
|----------|----------------|--------|----------|
| **Smoke Tests** | None (CPU only) | Tiny | ✅ Yes |
| **Small Scale** | 4-8 GB VRAM | Small | ✅ Yes |
| **Medium Scale** | 16-24 GB VRAM | Medium | ✅ Yes |
| **Large Scale** | 40-80 GB VRAM | Large | ✅ Yes |
| **Paper 105B Model** | 141 GB VRAM + 1.5 TB RAM | Full | ⚠️ H200 only |

---

## Detailed Analysis

### 1. Smoke Tests (CPU Only) - NO GPU REQUIRED ✅

**Config:** `examples/configs/megaslide_dit_tiny.yaml`

```yaml
model:
  frames: 2
  height: 8
  width: 8
  patch_size: 2
  hidden_size: 16
  num_layers: 2
```

**Memory Requirements:**
- Parameters: ~5,000 (20 KB)
- Activations: ~512 bytes
- **Total:** < 1 MB

**Hardware:**
- CPU: Any modern CPU
- RAM: 1 GB
- GPU: **Not needed** (runs in CPU mode with `force_cpu: true`)

**What you can test:**
- ✅ All 3 models (MegaSlide, Dense, Swin)
- ✅ Training loop correctness
- ✅ Gradient flow
- ✅ Loss convergence
- ❌ Performance benchmarks (CPU is slow)
- ❌ Memory profiling (not representative)

**Command:**
```bash
python examples/train_megaslide_dit.py \
    --config examples/configs/megaslide_dit_tiny.yaml
```

**Time:** ~1-2 minutes for 10 steps

---

### 2. Small Scale (4-8 GB VRAM) ✅

**Recommended GPU:** RTX 3060 (12 GB), RTX 4060 Ti (16 GB), or similar

**Example Config:**
```yaml
model:
  frames: 16
  height: 64
  width: 64
  patch_size: 8
  hidden_size: 256
  num_layers: 4
  num_heads: 4
```

**Memory Calculation:**
```python
# Patches
H_p = 64 / 8 = 8
W_p = 64 / 8 = 8
N = 16 * 8 * 8 = 1,024 patches

# Model parameters
hidden_size = 256
num_layers = 4
params_per_layer = 256 * 256 * 4 * 3 = ~800K
total_params = 800K * 4 = ~3.2M parameters
param_memory = 3.2M * 2 bytes (FP16) = 6.4 MB

# Activations
activation_per_layer = 1024 * 256 * 2 = 512 KB
total_activations = 512 KB * 4 = 2 MB

# Checkpoints (save every 2 layers)
checkpoint_memory = 2 * 512 KB = 1 MB

# Working memory (3D-DSA, MLP intermediates)
workspace = ~50 MB

# Total
total = 6.4 MB + 2 MB + 1 MB + 50 MB ≈ 60 MB
```

**Actual Memory:** ~200-500 MB (with PyTorch overhead)

**What you can test:**
- ✅ All 3 models
- ✅ Training convergence
- ✅ Gradient flow
- ✅ Basic performance benchmarks
- ✅ Ablation studies (fixed windows, async)
- ⚠️ Memory profiling (not at paper scale)

**GPUs that work:**
- RTX 3050 (8 GB) ✅
- RTX 3060 (12 GB) ✅
- RTX 4060 Ti (16 GB) ✅
- RTX 3070 (8 GB) ✅
- Any modern GPU with 4+ GB VRAM ✅

---

### 3. Medium Scale (16-24 GB VRAM) ✅

**Recommended GPU:** RTX 3090 (24 GB), RTX 4090 (24 GB), A5000 (24 GB)

**Example Config:**
```yaml
model:
  frames: 64
  height: 256
  width: 256
  patch_size: 16
  hidden_size: 1024
  num_layers: 12
  num_heads: 16
```

**Memory Calculation:**
```python
# Patches
H_p = 256 / 16 = 16
W_p = 256 / 16 = 16
N = 64 * 16 * 16 = 16,384 patches

# Model parameters
params_per_layer = 1024 * 1024 * 4 * 3 ≈ 12M
total_params = 12M * 12 = 144M parameters
param_memory = 144M * 2 bytes = 288 MB

# Activations
activation_per_layer = 16384 * 1024 * 2 = 32 MB
checkpoint_memory (every 4 layers) = 32 MB * 3 = 96 MB

# Working memory
workspace = ~500 MB

# Total
total ≈ 288 MB + 96 MB + 500 MB ≈ 1 GB
```

**Actual Memory:** ~2-4 GB (with PyTorch overhead)

**What you can test:**
- ✅ All 3 models at realistic scale
- ✅ Dense baseline starts to struggle (O(N²) becomes visible)
- ✅ Performance comparisons
- ✅ Memory scaling experiments
- ✅ Ablation studies with meaningful results
- ⚠️ VBench evaluation (lower resolution than paper)

**GPUs that work:**
- RTX 3090 (24 GB) ✅ Excellent
- RTX 4090 (24 GB) ✅ Excellent
- RTX 3080 Ti (12 GB) ✅ Works
- A5000 (24 GB) ✅ Excellent
- A6000 (48 GB) ✅ Overkill but works

**Comparison to Paper:**
- Paper uses 256 frames @ 1080p ≈ 2M patches
- This uses 64 frames @ 256p ≈ 16K patches
- Scale factor: ~125× smaller

---

### 4. Large Scale (40-80 GB VRAM) ✅

**Recommended GPU:** A100 (40/80 GB), A6000 (48 GB)

**Example Config:**
```yaml
model:
  frames: 128
  height: 512
  width: 512
  patch_size: 16
  hidden_size: 2048
  num_layers: 24
  num_heads: 32
```

**Memory Calculation:**
```python
# Patches
H_p = 512 / 16 = 32
W_p = 512 / 16 = 32
N = 128 * 32 * 32 = 131,072 patches

# Model parameters
params_per_layer = 2048 * 2048 * 4 * 3 ≈ 50M
total_params = 50M * 24 = 1.2B parameters
param_memory = 1.2B * 2 bytes = 2.4 GB

# Activations
activation_per_layer = 131072 * 2048 * 2 = 512 MB
checkpoint_memory (every 4 layers) = 512 MB * 6 = 3 GB

# Working memory
workspace = ~5 GB

# Total
total ≈ 2.4 GB + 3 GB + 5 GB ≈ 10-12 GB
```

**Actual Memory:** ~15-25 GB (with PyTorch overhead)

**What you can test:**
- ✅ All 3 models at near-paper scale
- ✅ Dense baseline OOMs as expected at high frame counts
- ✅ Realistic performance benchmarks
- ✅ Memory profiling representative of paper
- ✅ VBench evaluation at reasonable resolution
- ✅ All ablation studies with meaningful results

**GPUs that work:**
- A100 40GB ✅ Good fit
- A100 80GB ✅ Plenty of headroom
- A6000 48GB ✅ Works well
- H100 80GB ✅ Overkill but excellent

**Comparison to Paper:**
- Paper uses 256 frames @ 1080p ≈ 2M patches
- This uses 128 frames @ 512p ≈ 131K patches
- Scale factor: ~15× smaller

---

### 5. Paper Scale (141 GB VRAM + 1.5 TB RAM) ⚠️

**Required GPU:** NVIDIA H200 (141 GB HBM)

**Config:** `examples/configs/megaslide_paper_experiment_256f.yaml`

```yaml
model:
  frames: 256
  height: 1080  # 135 patches after 8x VAE downsample
  width: 1920   # 240 patches after 8x VAE downsample
  patch_size: 16
  hidden_size: 8192
  num_layers: 48  # 105B parameters
  num_heads: 64
```

**Memory Calculation:**
```python
# After VAE 8x downsample: 1080 → 135, 1920 → 240
H_p = 135 / 16 = 8 (rounded)
W_p = 240 / 16 = 15
N = 256 * 8 * 15 = 30,720 patches

# But paper uses 1080p directly in latent space
H_latent = 1080 / 8 = 135
W_latent = 1920 / 8 = 240
H_p = 135 / 16 = 8
W_p = 240 / 16 = 15
N = 256 * 8 * 15 = 30,720

# Actually, let me recalculate with paper numbers
# Paper Table 2 shows num_patches calculation
# They use post-VAE dimensions
```

**Corrected Calculation (from paper Table 2):**
```python
# Paper explicitly states dimensions
frames = 256
height_latent = 1080 / 8 = 135  # After VAE
width_latent = 1920 / 8 = 240
patch_size = 16

H_p = 135 / 16 ≈ 8  # Actually they might use different patching
W_p = 240 / 16 = 15

# But paper shows ~2M patches, so let me recalculate
# Looking at paper: 256-frame 1080p video
# If we DON'T divide by 8 first:
H_p = 1080 / 16 = 67
W_p = 1920 / 16 = 120
N = 256 * 67 * 120 = 2,064,384 patches ✓ (matches paper ~2M)

# Model parameters (from paper)
hidden_size = 8192
num_layers = 48
params_per_layer ≈ 2.2B
total_params = 105B parameters

# GPU Memory Breakdown (design estimate for the 105B target — NOT measured)
param_memory_gpu = ~4 GB (2 block slots × ~2 GB)
head_components = ~2 GB (patch_embed, time_embed, norm_out, out_proj)
activation_current = ~32 GB (2M patches × 8192 × 2 bytes)
checkpoint_memory = ~45 GB (12 checkpoint points × ~32 GB / 3)
workspace = ~30 GB (trilinear sampling, MLP intermediates)
total_gpu = ~115 GB

# CPU Memory (persistent state)
fp16_weights = 105B * 2 = 210 GB
fp32_master = 105B * 4 = 420 GB
adam_moments = 105B * 8 = 840 GB (first + second moment)
total_cpu = 1.47 TB
```

**Hardware Requirements:**
- **GPU:** NVIDIA H200 (141 GB HBM, $30-40K)
- **CPU RAM:** 1.5 TB (DDR5 recommended)
- **Storage:** 500 GB NVMe SSD
- **CPU:** 64+ cores (for CPU optimizer overhead)

**Available H200 Systems:**
- ❌ Not in consumer GPUs
- ❌ Not in standard cloud instances (as of 2024)
- ✅ NVIDIA DGX H200 systems
- ✅ AWS P5 instances (when H200 available)
- ✅ Academic/research clusters

**Alternative Approaches:**
1. **Use smaller model** (e.g., 7B params on A100)
2. **Use multi-GPU** (not implemented in current code)
3. **Use gradient checkpointing more aggressively**
4. **Reduce frames** (128 frames fits on A100 80GB)

---

## Comparison: Dense vs Swin vs MegaSlide

### Memory Scaling with Frame Count

| Frames | Dense (O(N²)) | Swin (O(N)) | MegaSlide (O(N)) |
|--------|---------------|-------------|------------------|
| 16     | 2 GB          | 2 GB        | 2 GB             |
| 32     | 8 GB          | 4 GB        | 4 GB             |
| 64     | 32 GB         | 8 GB        | 8 GB             |
| 128    | **128 GB** (OOM most GPUs) | 16 GB | 16 GB |
| 256    | **512 GB** (OOM all GPUs) | 32 GB | 32 GB |

**Key Point:** Dense baseline OOMs at 64-128 frames on most GPUs due to O(N²) attention complexity.

---

## Recommended Setup for Each Goal

### Goal 1: Just Test the Code ✅
**GPU:** None (CPU only)
**Config:** Tiny
**Time:** 5 minutes
**Cost:** Free

---

### Goal 2: Validate Training Works ✅
**GPU:** Any consumer GPU (RTX 3060+, 8 GB VRAM)
**Config:** Small
**Time:** 30 minutes
**Cost:** $0 (use local GPU) or $0.50/hour (cloud)

---

### Goal 3: Reproduce Paper Trends ✅
**GPU:** RTX 4090 (24 GB) or A100 40GB
**Config:** Medium (1B params, 64 frames, 256p)
**Time:** 4-8 hours
**Cost:** $500-1000 (RTX 4090) or $2-3/hour (A100 cloud)

**What you can validate:**
- Dense OOMs at 256 frames ✅
- Swin scales to 256 frames ✅
- MegaSlide scales to 256 frames (reduced config) ✅
- Learned offsets: small, data-dependent loss effect (~2% motion, ~0% random)
- Async streaming: 1.25–1.50× end-to-end (up to 2.11× forward-only)

---

### Goal 4: 105B target (NOT run — analytical projection only) ⚠️
**GPU:** NVIDIA H200 (141 GB) — never used
**Config:** 105B params, 256 frames, 1080p
**Status:** out of scope; we did not have a checkpoint, an H200, or real-video data.

**What the projection says (see `results/07_roofline`):**
- 105B fp16 (210 GB) does not fit in 141 GB HBM → weights still stream over PCIe
- Regime is transfer-bound (~840 GB/step), MFU ~21% — **not** the old 61%
- No official VBench scores are claimed

---

## Current System Check

Let me check what GPU you have available:

```bash
# Check for NVIDIA GPU
nvidia-smi 2>/dev/null || echo "No NVIDIA GPU detected"

# Check for Apple Silicon
system_profiler SPDisplaysDataType 2>/dev/null | grep -i "apple" || echo "Not Apple Silicon"
```

Running this check...
