# Running MegaSlide-DiT on Apple Silicon

**Your System:** Apple M5 Max (40-core GPU)  
**Status:** ✅ Can run experiments with PyTorch MPS backend  
**Date:** 2026-05-13

---

## Quick Summary

**Your M5 Max can run:**
- ✅ **Smoke tests** (tiny configs) - CPU/MPS
- ✅ **Small scale** (up to ~1B params) - MPS backend
- ✅ **Medium scale** (up to ~5B params) - MPS backend with unified memory
- ❌ **Paper scale** (105B params) - Requires H200 GPU

**Recommended approach:** Use MPS backend for GPU acceleration, CPU mode as fallback.

---

## Apple Silicon GPU Support

### PyTorch MPS Backend

**Install PyTorch with MPS support:**
```bash
# Install PyTorch 2.0+ with MPS backend
pip install torch torchvision torchaudio

# Verify MPS availability
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**Expected output:**
```
MPS available: True
```

### Unified Memory Advantage

Apple Silicon has **unified memory** shared between CPU and GPU:
- Your M5 Max likely has 64 GB or 96 GB unified memory
- No separate VRAM limit (unlike NVIDIA GPUs)
- Can use all system RAM for GPU operations
- Much more flexible than discrete GPUs!

**Check your unified memory:**
```bash
system_profiler SPHardwareDataType | grep "Memory:"
```

---

## What You Can Run

### Scenario 1: Tiny Smoke Tests (Recommended First) ✅

**No modifications needed!**

```bash
# Run with CPU (works everywhere)
python examples/train_megaslide_dit.py \
    --config examples/configs/megaslide_dit_tiny.yaml
```

**Memory:** < 100 MB  
**Time:** 1-2 minutes  
**What it tests:** Basic correctness, gradient flow

---

### Scenario 2: Small Scale with MPS (GPU Acceleration) ✅

**Modify config to use MPS:**

Create `examples/configs/megaslide_small_mps.yaml`:
```yaml
model:
  frames: 32
  in_channels: 4
  height: 128
  width: 128
  patch_size: 16
  hidden_size: 512
  num_layers: 6
  num_heads: 8
  mlp_ratio: 4.0
  dropout: 0.0
  dtype: "float32"  # MPS works better with FP32

dataset:
  path: ""
  synthetic_samples: 50
  latent_layout: "auto"

training:
  batch_size: 1
  num_steps: 100
  learning_rate: 1.0e-4

memory:
  checkpoint_interval: 2
  num_grad_slabs: 4
  device: "mps"  # Use MPS backend
  force_cpu: false

logging:
  log_interval: 10
```

**Run:**
```bash
python examples/train_megaslide_dit.py \
    --config examples/configs/megaslide_small_mps.yaml
```

**Memory Calculation:**
- Parameters: ~50M × 4 bytes (FP32) = 200 MB
- Activations: ~10 MB
- **Total:** ~500 MB (well within M5 Max capacity)

**Expected Performance:**
- Step time: ~0.5-1s (much faster than CPU)
- MFU: ~20-30% (MPS is less optimized than CUDA)

---

### Scenario 3: Medium Scale (Up to ~5B params) ✅

**If you have 64+ GB unified memory:**

```yaml
model:
  frames: 64
  height: 256
  width: 256
  patch_size: 16
  hidden_size: 1024
  num_layers: 12
  num_heads: 16
  dtype: "float32"

training:
  batch_size: 1

memory:
  device: "mps"
  checkpoint_interval: 3
```

**Memory Estimate:**
- Parameters: ~1.5B × 4 bytes = 6 GB
- Activations: ~2 GB
- **Total:** ~10-15 GB (should fit on M5 Max with 64+ GB)

**Expected Performance:**
- Step time: ~5-10s
- Realistic enough to test training dynamics

---

## Limitations on Apple Silicon

### 1. No CUDA-Specific Features ⚠️

**These won't work:**
- `torch.cuda.Stream()` - MPS doesn't support CUDA streams
- `torch.cuda.Event()` - MPS has different synchronization
- Double-buffered async streaming - Requires CUDA streams

**Solution:** Our trainer has CPU fallback mode!
```python
trainer = CPUMasterVideoDiT(model, config, force_cpu=True)
```

### 2. MPS Backend Limitations

**Known issues:**
- Some operations slower than CUDA
- Less optimized for large matrix multiplies
- Memory management less sophisticated
- No FP16 support (use FP32)

**Impact:** Training ~2-3× slower than equivalent NVIDIA GPU

### 3. Paper Scale Not Possible

**105B model requires:**
- 141 GB HBM (H200 GPU)
- 1.5 TB RAM (for CPU-master)

**M5 Max typical configs:**
- 64 GB or 96 GB unified memory (not enough for 105B)

**Solution:** Test with smaller models (up to ~5B params feasible)

---

## Recommended Workflow on M5 Max

### Step 1: Test with CPU Mode (5 min)
```bash
python examples/train_megaslide_dit.py \
    --config examples/configs/megaslide_dit_tiny.yaml
```

**Validates:** Code correctness, no errors

---

### Step 2: Test with MPS Mode (30 min)
```bash
# Install PyTorch
pip install torch torchvision torchaudio

# Run with small config + MPS
python examples/train_megaslide_dit.py \
    --config examples/configs/megaslide_small_mps.yaml
```

**Validates:** GPU acceleration works, training converges

---

### Step 3: Compare Baselines (2 hours)

**Test all 3 models at small scale:**
```bash
# MegaSlide-DiT
python examples/train_megaslide_dit.py \
    --config examples/configs/megaslide_small_mps.yaml

# Dense 3D-DiT (will be slower due to O(N²))
# ... modify config to use Dense3DDiT

# Swin-DiT
# ... modify config to use SwinDiT
```

**Validates:** 
- All 3 models work
- Dense is slower (O(N²) vs O(N))
- Training dynamics are reasonable

---

### Step 4: Run Ablations (2 hours)

**Ablation 1: Fixed windows**
```bash
python examples/run_ablation_studies.py \
    --config examples/configs/megaslide_small_mps.yaml \
    --ablation fixed_windows \
    --num_steps 50
```

**Ablation 2 & 3:** Skip (require CUDA async streaming)

---

## Expected Results on M5 Max

### Performance Estimates

| Config | Params | Memory | Step Time (MPS) | Step Time (CPU) |
|--------|--------|--------|-----------------|-----------------|
| Tiny   | 5K     | < 100 MB | 0.05s         | 0.1s            |
| Small  | 50M    | ~500 MB  | 0.5-1s        | 5-10s           |
| Medium | 1.5B   | ~10 GB   | 5-10s         | 60-120s         |

**MPS Speedup:** ~5-10× faster than CPU (less than CUDA's ~20-50×)

---

## Installation Instructions

### Option 1: Use System Python
```bash
# Install PyTorch
pip3 install torch torchvision torchaudio

# Install other dependencies
pip3 install pyyaml numpy pytest

# Verify
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

### Option 2: Use Virtual Environment (Recommended)
```bash
# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio pyyaml numpy pytest

# Verify
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

---

## Troubleshooting

### Issue 1: "MPS backend is not available"
```bash
# Check macOS version (MPS requires macOS 12.3+)
sw_vers

# Update PyTorch to latest version
pip install --upgrade torch torchvision torchaudio
```

### Issue 2: "RuntimeError: MPS does not support operation X"
**Solution:** Fall back to CPU mode
```python
# In config:
memory:
  force_cpu: true
```

### Issue 3: Memory pressure / system slow
**Solution:** Reduce batch size or model size
```yaml
training:
  batch_size: 1  # Already minimum

model:
  hidden_size: 256  # Reduce from 512
  num_layers: 4     # Reduce from 6
```

### Issue 4: Slower than expected
**Expected!** MPS is ~2-3× slower than CUDA for transformer models.

**Check Activity Monitor:**
- CPU usage should be high during forward/backward
- GPU usage should show in Activity Monitor → GPU tab

---

## Summary for M5 Max

**✅ You CAN run:**
- All smoke tests (tiny configs)
- Small scale experiments (~50M params)
- Medium scale experiments (~1-5B params, if 64+ GB RAM)
- All 3 models (MegaSlide, Dense, Swin)
- Fixed windows ablation

**❌ You CANNOT run:**
- Paper scale (105B params) - needs H200
- Async streaming ablations - needs CUDA
- VBench at paper resolution - needs more VRAM

**🎯 Recommended Goal:**
Test with small/medium configs to validate:
1. All 3 models train correctly ✅
2. Dense is slower than Swin/MegaSlide ✅
3. Training converges to low loss ✅
4. Learned offsets improve results ✅

**Next Steps:**
1. Install PyTorch: `pip install torch pyyaml numpy`
2. Run smoke test: `./run_phase3_experiments.sh`
3. Check results: All models should train without errors

**Need CUDA GPU?** Consider:
- Cloud GPUs (RunPod, Lambda Labs) - $0.50-3/hour
- Colab Pro+ (A100) - $50/month
- AWS P3/P4 instances - $3-30/hour

---

**TL;DR:** Your M5 Max can run all smoke tests and small-scale experiments perfectly. Install PyTorch and run `./run_phase3_experiments.sh` to start!
