# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/jasonliu2050/megaslide.git
cd megaslide

# Install (automatically builds CUDA extension if CUDA is available)
pip install -e .

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pyyaml numpy pytest einops psutil

# Optional: faster attention & optimizer
pip install flash-attn
pip install deepspeed
```

The installation will automatically:
- Install all Python dependencies
- Build and install the CUDA pipeline extension (if CUDA is available)
- Set up the infinity library for import

## Usage

### Run MegaSlide-DiT Smoke Test

```bash
# Quick validation (any GPU, <1 min)
PYTHONPATH=. python examples/train_megaslide_dit.py --config examples/configs/megaslide_dit_tiny.yaml
```

### Run Paper Experiments

```bash
# Memory scaling experiment (proves dense OOMs at 256 frames)
PYTHONPATH=. python examples/run_megaslide_paper_experiment.py --experiment memory_scaling

# Quality ablation (proves learned offsets matter)
PYTHONPATH=. python examples/run_megaslide_paper_experiment.py --experiment quality_ablation

# Async speedup scaling
PYTHONPATH=. python examples/run_megaslide_paper_experiment.py --experiment efficiency_scaling

# Long training convergence (28.4B, 100 steps)
PYTHONPATH=. python examples/run_megaslide_paper_experiment.py --experiment long_training
```

### Run Unit Tests

```bash
PYTHONPATH=. pytest tests/test_megaslide_video.py -v
```

### Using YAML Configuration

```bash
# Tiny model for development
PYTHONPATH=. python examples/train_megaslide_dit.py --config examples/configs/megaslide_dit_tiny.yaml

# Full 256-frame paper experiment
PYTHONPATH=. python examples/train_megaslide_dit.py --config examples/configs/megaslide_paper_experiment_256f.yaml
```

### Using in Python

```python
from infinity.video import MegaSlideDiT, CPUMasterVideoDiT, load_megaslide_config

# Load configuration
config = load_megaslide_config("examples/configs/megaslide_dit_tiny.yaml")

# Create model
model = MegaSlideDiT(config)

# Create CPU-master trainer (streams weights from host memory)
trainer = CPUMasterVideoDiT(model, config)

# Train...
trainer.train_step(batch)
```

## Configuration Files

Configuration files are located in `examples/configs/`:

- **megaslide_dit_tiny.yaml** — Tiny model for smoke tests
- **megaslide_paper_experiment_tiny.yaml** — Paper experiments at reduced scale
- **megaslide_paper_experiment_256f.yaml** — Full 256-frame paper experiments
- **dense_baseline_64f.yaml** — Dense 3D-DiT baseline (64 frames max)
- **swin_baseline_256f.yaml** — Swin-DiT fixed-window baseline

See `examples/configs/README.md` for detailed configuration guide.

## What Gets Installed

When you run `pip install -e .`:

1. **Python Package**: `megaslide-dit` package with all modules
2. **CUDA Extension**: `cuda_pipeline` for optimized GPU operations (if CUDA available)
3. **Dependencies**: PyTorch, Transformers, einops, etc.

## Verify Installation

```python
# Test imports
from infinity.video import MegaSlideDiT, CPUMasterVideoDiT, load_megaslide_config
print("✓ MegaSlide-DiT library installed successfully")

# Test CUDA extension (optional)
try:
    import cuda_pipeline
    print("✓ CUDA pipeline extension available")
except ImportError:
    print("✗ CUDA pipeline extension not available (optional)")
```

## Requirements

- **GPU**: NVIDIA GPU (2+ GB for smoke tests; measured runs used one 94 GB H100 NVL up to 33.3B). The 141 GB H200 / 105B "full scale" was **not run** — it is an analytical PCIe-bound projection (~21% MFU); see `RUN_EXPERIMENTS.md`.
- **CPU RAM**: 8 GB minimum; ~291 GB used at 33.3B. 1.5 TB is the design estimate for the unrun 105B target.
- **CUDA**: 11.8+ (for CUDA extension)
- **PyTorch**: 2.0+
- **Python**: 3.9+

## Troubleshooting

### CUDA Extension Build Failed

If CUDA extension fails to build, the package will still install without it. Training will work but may be slower for memory transfers.

To manually build CUDA extension:
```bash
cd infinity/cuda_pipeline
python setup.py install
```

### Import Errors

Make sure you're in the correct directory:
```bash
cd megaslide
pip install -e .
```

### Out of Memory

Use a smaller config or reduce frame count:
```yaml
data:
  num_frames: 64    # Reduce from 256
  resolution: 512   # Reduce from 1080
```

## Next Steps

- Read the [Configuration Guide](examples/configs/README.md)
- Check the [Main README](README.md) for architecture details
- Read the [Paper](paper/megaslide_dit_paper.md) for full technical details
- Explore experiment results in `results/`
