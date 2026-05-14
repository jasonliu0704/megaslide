# MegaSlide-DiT Phase 3 Experiments: Step-by-Step Procedure

**Date:** 2026-05-13  
**Purpose:** Reproduce paper experiments on a VM with working PyTorch  
**Time:** 5-10 minutes for smoke tests, 2-4 hours for full experiments

---

## Prerequisites

### System Requirements
- **GPU:** NVIDIA with 16+ GB VRAM (RTX 3090/4090, A100)
- **RAM:** 32+ GB system memory
- **Disk:** 10 GB free space
- **OS:** Linux (Ubuntu 20.04+ recommended) or macOS

### Software Dependencies
- Python 3.8+
- CUDA 11.8+ (for GPU experiments)
- Git (to clone/transfer code)

---

## Step 1: Transfer Code to VM

**Option A: Git (if repo is accessible)**
```bash
git clone <your-repo-url>
cd megaslide
```

**Option B: SCP from Mac**
```bash
# On your Mac:
tar -czf megaslide.tar.gz megaslide/
scp megaslide.tar.gz your-vm:/home/user/

# On VM:
tar -xzf megaslide.tar.gz
cd megaslide
```

**Option C: Sync specific files**
```bash
# Minimum files needed:
rsync -av --include='*.py' --include='*.yaml' --include='*.sh' \
    --exclude='*' megaslide/ your-vm:~/megaslide/
```

---

## Step 2: Set Up Python Environment

### Option A: Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pyyaml numpy pytest einops

# Verify PyTorch installation
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Expected output:**
```
PyTorch 2.x.x
CUDA available: True
```

### Option B: Conda
```bash
conda create -n megaslide python=3.10
conda activate megaslide
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pyyaml numpy pytest einops
```

---

## Step 3: Pre-Flight Checks

### 3.1 Verify Code Structure
```bash
python3 verify_without_torch.py
```

**Expected output:**
```
✅ Baseline models: infinity/video/baselines.py (513 lines)
✅ VBench evaluation: examples/run_vbench_evaluation.py (368 lines)
✅ All Phase 3 files present and verified
```

### 3.2 Test Basic Imports
```bash
python3 test_basic.py
```

**Expected output:**
```
✅ Config imports successful
✅ Tiny config loaded: 16 frames, 16 hidden
✅ Memory estimate: 0.02 GB
BASIC TESTS COMPLETE
```

### 3.3 Check GPU Access
```bash
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

**Expected output:**
```
GPU: NVIDIA GeForce RTX 3090
Memory: 24.0 GB
```

---

## Step 4: Smoke Tests (5 minutes)

### 4.1 Test MegaSlide-DiT (Tiny Config)
```bash
python3 examples/train_megaslide_dit.py \
    --config examples/configs/megaslide_dit_tiny.yaml \
    --num_steps 2
```

**Expected output:**
```
[MegaSlideDiT] Model: 2 layers, 16 hidden, 5.1K params
[Step 0] Loss: 0.xxxx, Forward: 0.xx s, Backward: 0.xx s
[Step 1] Loss: 0.xxxx, Forward: 0.xx s, Backward: 0.xx s
✅ Training complete
```

**Success criteria:**
- ✅ No errors
- ✅ Loss is finite (not NaN/Inf)
- ✅ Step time < 1s

### 4.2 Test Dense3DDiT (Tiny Config)
```bash
# Create tiny dense config
cat > examples/configs/dense_baseline_tiny.yaml << 'EOF'
model:
  model_type: "dense"
  frames: 16
  in_channels: 4
  height: 32
  width: 32
  patch_size: 4
  hidden_size: 16
  num_layers: 2
  num_heads: 4

dataset:
  synthetic_samples: 10

training:
  batch_size: 1
  num_steps: 2
  learning_rate: 1.0e-4

memory:
  checkpoint_interval: 1
  device: 0

logging:
  log_interval: 1
EOF

python3 examples/train_megaslide_dit.py \
    --config examples/configs/dense_baseline_tiny.yaml \
    --num_steps 2
```

**Expected output:**
```
[Dense3DDiT] Global attention, O(N²) complexity
[Step 0] Loss: 0.xxxx
✅ Training complete
```

### 4.3 Test SwinDiT (Tiny Config)
```bash
# Create tiny swin config
cat > examples/configs/swin_baseline_tiny.yaml << 'EOF'
model:
  model_type: "swin"
  frames: 16
  in_channels: 4
  height: 32
  width: 32
  patch_size: 4
  hidden_size: 16
  num_layers: 2
  num_heads: 4
  window_size: [4, 4, 4]

dataset:
  synthetic_samples: 10

training:
  batch_size: 1
  num_steps: 2
  learning_rate: 1.0e-4

memory:
  checkpoint_interval: 1
  device: 0

logging:
  log_interval: 1
EOF

python3 examples/train_megaslide_dit.py \
    --config examples/configs/swin_baseline_tiny.yaml \
    --num_steps 2
```

**Expected output:**
```
[SwinDiT] Fixed 3D windows, O(N·w³) complexity
[Step 0] Loss: 0.xxxx
✅ Training complete
```

---

## Step 5: Parameter Count Verification (2 minutes)

### 5.1 Compare Model Sizes
```bash
python3 << 'EOF'
from infinity.video import MegaSlideDiT, load_megaslide_config
from infinity.video.baselines import Dense3DDiT, SwinDiT
import torch

configs = {
    "Tiny": "examples/configs/megaslide_dit_tiny.yaml",
    "Dense (64f)": "examples/configs/dense_baseline_64f.yaml",
}

for name, path in configs.items():
    try:
        config = load_megaslide_config(path)
        
        # Determine model type
        model_type = getattr(config, 'model_type', 'megaslide')
        if model_type == 'dense':
            model = Dense3DDiT(config)
        elif model_type == 'swin':
            model = SwinDiT(config)
        else:
            model = MegaSlideDiT(config)
        
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:20s}: {params:>12,} params ({params*4/1e6:.1f} MB)")
    except Exception as e:
        print(f"{name:20s}: ERROR - {e}")

EOF
```

**Expected output:**
```
Tiny                :        5,120 params (0.0 MB)
Dense (64f)         :  105,123,456 params (420.5 MB)
```

**Verify:** All 3 models have same parameter count for same config (attention mechanism doesn't change params).

---

## Step 6: Memory Scaling Test (10 minutes)

### 6.1 Test Dense Baseline at Increasing Frames
```bash
for frames in 16 32 64; do
    echo "Testing Dense3DDiT at $frames frames..."
    
    cat > /tmp/dense_${frames}f.yaml << EOF
model:
  model_type: "dense"
  frames: $frames
  in_channels: 4
  height: 64
  width: 64
  patch_size: 8
  hidden_size: 256
  num_layers: 4
  num_heads: 8

dataset:
  synthetic_samples: 5

training:
  batch_size: 1
  num_steps: 1

memory:
  checkpoint_interval: 2
  device: 0

logging:
  log_interval: 1
EOF
    
    python3 examples/train_megaslide_dit.py \
        --config /tmp/dense_${frames}f.yaml \
        --num_steps 1 2>&1 | grep -E "(Step|Memory|OOM)" || echo "OOM at $frames frames"
done
```

**Expected results:**
- 16 frames: ✅ Runs successfully
- 32 frames: ✅ Runs successfully (slower)
- 64 frames: ⚠️ Very slow or OOM (depending on GPU)
- 128 frames: ❌ OOM (expected)

### 6.2 Test MegaSlide at 256 Frames
```bash
# Only run if you have 16+ GB VRAM
cat > /tmp/megaslide_256f_small.yaml << 'EOF'
model:
  frames: 256
  in_channels: 4
  height: 64
  width: 64
  patch_size: 8
  hidden_size: 256
  num_layers: 4
  num_heads: 8
  dsa_kernel_size: [3, 7, 7]

dataset:
  synthetic_samples: 2

training:
  batch_size: 1
  num_steps: 1

memory:
  checkpoint_interval: 2
  device: 0

logging:
  log_interval: 1
EOF

python3 examples/train_megaslide_dit.py \
    --config /tmp/megaslide_256f_small.yaml \
    --num_steps 1
```

**Expected output:**
```
[MegaSlideDiT] 256 frames, 4×64×64 latent
[Step 0] Loss: 0.xxxx
✅ Proves 3D-DSA scales to 256 frames (Dense would OOM)
```

---

## Step 7: Baseline Comparison (30 minutes)

### 7.1 Train All 3 Models at Same Scale
```bash
# Use small-scale config that all 3 can handle
for model_type in megaslide dense swin; do
    echo "Training $model_type..."
    
    cat > /tmp/${model_type}_compare.yaml << EOF
model:
  model_type: "$model_type"
  frames: 32
  in_channels: 4
  height: 64
  width: 64
  patch_size: 8
  hidden_size: 256
  num_layers: 4
  num_heads: 8
  $([ "$model_type" = "megaslide" ] && echo "dsa_kernel_size: [3, 7, 7]")
  $([ "$model_type" = "swin" ] && echo "window_size: [4, 8, 8]")

dataset:
  synthetic_samples: 50

training:
  batch_size: 1
  num_steps: 50
  learning_rate: 1.0e-4

memory:
  checkpoint_interval: 2
  device: 0

logging:
  log_interval: 10
EOF
    
    python3 examples/train_megaslide_dit.py \
        --config /tmp/${model_type}_compare.yaml \
        --num_steps 50 > results/${model_type}_train.log 2>&1
    
    echo "✅ $model_type complete"
done
```

### 7.2 Compare Training Speed
```bash
python3 << 'EOF'
import re

for model_type in ["megaslide", "dense", "swin"]:
    log_file = f"results/{model_type}_train.log"
    try:
        with open(log_file) as f:
            content = f.read()
        
        # Extract step times
        times = re.findall(r'Forward: ([\d.]+) s', content)
        if times:
            avg_time = sum(float(t) for t in times) / len(times)
            print(f"{model_type:12s}: {avg_time:.3f} s/step (avg over {len(times)} steps)")
    except FileNotFoundError:
        print(f"{model_type:12s}: Log not found")

EOF
```

**Expected results:**
```
megaslide   : 0.2-0.5 s/step (O(N·k) complexity)
dense       : 0.5-2.0 s/step (O(N²) complexity, slower)
swin        : 0.2-0.5 s/step (O(N·w³) complexity)
```

**Key finding:** Dense is slower than MegaSlide/Swin at same scale.

---

## Step 8: Ablation Study 1 - Fixed Windows vs Learned Offsets (1 hour)

### 8.1 Run Baseline (Learned Offsets)
```bash
python3 examples/train_megaslide_dit.py \
    --config examples/configs/megaslide_dit_tiny.yaml \
    --num_steps 100 \
    > results/ablation1_learned.log 2>&1
```

### 8.2 Run Ablation (Fixed Windows)
```bash
python3 examples/run_ablation_studies.py \
    --config examples/configs/megaslide_dit_tiny.yaml \
    --ablation fixed_windows \
    --num_steps 100 \
    --output_dir results/ablation1_fixed
```

**Expected output:**
```
Ablation 1: Fixed Windows
  - Freezing offset prediction networks...
  - Setting all offsets to zero...
  ✅ Model modified: offsets are fixed

[Training with fixed windows...]
[Step 0] Loss: 0.xxxx
...
[Step 99] Loss: 0.yyyy

Results saved to: results/ablation1_fixed/
```

### 8.3 Compare Results
```bash
python3 << 'EOF'
import json

# Read final losses
with open("results/ablation1_learned.log") as f:
    learned_lines = f.readlines()
    learned_loss = float(learned_lines[-2].split("Loss: ")[1].split(",")[0])

with open("results/ablation1_fixed/train.log") as f:
    fixed_lines = f.readlines()
    fixed_loss = float(fixed_lines[-2].split("Loss: ")[1].split(",")[0])

print(f"Learned offsets: Final loss = {learned_loss:.4f}")
print(f"Fixed windows:   Final loss = {fixed_loss:.4f}")
print(f"Impact: {((fixed_loss - learned_loss) / learned_loss * 100):+.1f}% change")

if fixed_loss > learned_loss:
    print("✅ Confirms: Learned offsets improve training")
else:
    print("⚠️  Unexpected: Fixed windows performed better (may need more steps)")

EOF
```

**Expected result:**
```
Learned offsets: Final loss = 0.2xxx
Fixed windows:   Final loss = 0.3xxx
Impact: +30-50% change
✅ Confirms: Learned offsets improve training
```

---

## Step 9: Unit Tests (5 minutes)

```bash
pytest tests/test_megaslide_video.py -v
```

**Expected output:**
```
tests/test_megaslide_video.py::test_deformable_slide_attention_shape_finite_and_gradients PASSED
tests/test_megaslide_video.py::test_deformable_slide_attention_is_non_causal_over_time PASSED
tests/test_megaslide_video.py::test_deformable_slide_attention_handles_degenerate_grid PASSED
tests/test_megaslide_video.py::test_megaslide_dit_shape_and_backward PASSED
tests/test_megaslide_video.py::test_cpu_master_video_dit_cpu_fallback_step PASSED
tests/test_megaslide_video.py::test_cpu_master_video_dit_cuda_streaming_smoke PASSED
tests/test_megaslide_video.py::test_latent_video_dataset_synthetic_and_files PASSED
tests/test_megaslide_video.py::test_megaslide_yaml_config PASSED

==================== 8 passed in X.XXs ====================
```

**Success criteria:** All 8 tests pass.

---

## Step 10: Full Automated Test Suite (Optional, 2-4 hours)

**⚠️ Only run if you have time and sufficient GPU resources**

```bash
# Make script executable
chmod +x run_phase3_experiments.sh

# Run full suite
./run_phase3_experiments.sh
```

This runs:
1. All smoke tests (3 models)
2. Parameter count verification
3. Memory scaling tests (Dense at 16/32/64, MegaSlide at 256)
4. Training convergence tests (100 steps each)
5. All ablation studies
6. Generates comprehensive results

**Results location:** `results/phase3/`

---

## Step 11: Collect Results

### 11.1 Generate Summary Report
```bash
python3 << 'EOF'
import os
import glob

print("=" * 70)
print("PHASE 3 EXPERIMENT RESULTS SUMMARY")
print("=" * 70)
print()

# Check smoke tests
print("1. SMOKE TESTS")
for model in ["megaslide", "dense", "swin"]:
    log_path = f"results/{model}_train.log"
    if os.path.exists(log_path):
        print(f"   ✅ {model:12s}: Completed")
    else:
        print(f"   ⬜ {model:12s}: Not run")
print()

# Check unit tests
print("2. UNIT TESTS")
if os.path.exists("results/pytest.log"):
    with open("results/pytest.log") as f:
        if "8 passed" in f.read():
            print("   ✅ All 8 tests passed")
        else:
            print("   ⚠️  Some tests failed")
else:
    print("   ⬜ Not run")
print()

# Check ablations
print("3. ABLATION STUDIES")
for abl in ["ablation1_fixed", "ablation2_sync", "ablation3_gpu_opt"]:
    if os.path.exists(f"results/{abl}"):
        print(f"   ✅ {abl}")
    else:
        print(f"   ⬜ {abl}: Not run")
print()

print("=" * 70)
print("To view detailed results:")
print("  - Training logs: results/*_train.log")
print("  - Ablation results: results/ablation*/")
print("  - Full suite: results/phase3/")
print("=" * 70)

EOF
```

### 11.2 Package Results for Transfer Back
```bash
# Create results archive
tar -czf phase3_results.tar.gz results/ *.log

# Transfer back to Mac
# From VM:
scp phase3_results.tar.gz your-mac:/path/to/megaslide/
```

---

## Troubleshooting

### Issue 1: CUDA Out of Memory
**Solution:** Reduce frames, hidden_size, or batch_size
```yaml
model:
  frames: 16  # Reduce from 32
  hidden_size: 128  # Reduce from 256
```

### Issue 2: "Module not found: infinity.video"
**Solution:** Ensure you're in the megaslide root directory
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue 3: Slow Training
**Expected:** Dense baseline should be slower than MegaSlide/Swin (this proves the paper's claims)

### Issue 4: PyTorch not detecting GPU
**Check:**
```bash
nvidia-smi  # Should show GPU
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Fix:**
```bash
# Reinstall PyTorch with correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Expected Timeline

| Step | Time | Can Skip? |
|------|------|-----------|
| 1-3: Setup & Pre-flight | 10 min | No |
| 4: Smoke tests | 5 min | No |
| 5: Parameter counts | 2 min | Yes |
| 6: Memory scaling | 10 min | Yes |
| 7: Baseline comparison | 30 min | No (core result) |
| 8: Ablation study | 1 hour | Yes (but recommended) |
| 9: Unit tests | 5 min | No |
| 10: Full suite | 2-4 hours | Yes (only if time permits) |
| 11: Collect results | 5 min | No |

**Minimum viable run:** Steps 1-4, 7, 9 (~50 minutes)
**Recommended run:** Steps 1-9 (~2 hours)
**Complete run:** All steps (~4-5 hours)

---

## Success Criteria

### Minimum (MVP)
- ✅ All 3 models train without errors (Step 4)
- ✅ All 8 unit tests pass (Step 9)
- ✅ Dense is slower than MegaSlide at same scale (Step 7)

### Recommended
- ✅ Dense OOMs or is very slow at 64+ frames (Step 6)
- ✅ MegaSlide scales to 256 frames (Step 6)
- ✅ Learned offsets improve over fixed windows (Step 8)

### Complete (Paper Reproduction)
- ✅ Memory usage matches theoretical estimates
- ✅ VBench evaluation runs (requires additional setup)
- ✅ All ablations reproduce paper trends

---

## Next Steps After Experiments

1. **Review results:** Check logs in `results/`
2. **Document findings:** Note any deviations from expected results
3. **Transfer back:** Copy results archive to original system
4. **Phase 4:** If experiments pass, proceed to profiling & metrics
5. **Report:** Summarize which paper claims were validated

---

## Quick Reference Commands

```bash
# Activate environment
source venv/bin/activate

# Run single smoke test
python3 examples/train_megaslide_dit.py --config examples/configs/megaslide_dit_tiny.yaml --num_steps 2

# Run all unit tests
pytest tests/test_megaslide_video.py -v

# Run baseline comparison
./run_phase3_experiments.sh

# Check GPU memory
nvidia-smi

# Monitor training
tail -f results/megaslide_train.log
```

---

## Contact / Issues

If experiments fail or produce unexpected results, document:
1. Which step failed
2. Full error message
3. GPU model and memory
4. PyTorch version (`python3 -c "import torch; print(torch.__version__)"`)
5. Contents of failing log file

This helps diagnose whether the issue is environmental or implementation-related.
