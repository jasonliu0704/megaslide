<div align="center">

# MegaSlide-DiT

### Training Video Diffusion Transformers Larger Than GPU Memory via CPU-Master Streaming

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

**Train full-parameter video diffusion models far larger than GPU memory on a single GPU by keeping weights and optimizer state in host RAM and streaming per-layer shards on demand — validated to 33.3B parameters on one 94 GB H100 NVL.**

[Quick Start](#quick-start) | [Architecture](#architecture) | [Experiments](#experiments) | [Paper](#paper) | [Citation](#citation)

</div>

---

## Overview

MegaSlide-DiT is a **CPU-master** training system that decouples model size from GPU memory. Persistent weights and optimizer state live in host RAM; the GPU holds only a transient per-layer shard plus activations. The central, measured finding is a **capacity-for-throughput trade-off**: streaming makes models far larger than HBM trainable on one GPU, but PCIe transfer of host-resident weights bounds utilisation. It addresses two walls:

- **Parameter memory wall:** large DiTs require terabyte-scale persistent state (weights + optimizer). MegaSlide-DiT keeps all of this in host CPU memory and streams only transient shards to the GPU.
- **Activation memory wall:** high-resolution video produces very long token sequences. MegaSlide-DiT uses **3D Deformable Slide Attention (3D-DSA)**, one of three interchangeable linear-complexity local-attention back-ends, as an alternative to O(N²) global attention.

### Key Results (measured on a single H100 NVL, 94 GB HBM / 314 GB RAM)

| Metric | Value |
|--------|-------|
| Max model trained (single 94 GB GPU) | **33.3B parameters** (133 GB weights, 1.4× HBM) |
| Model FLOPs Utilisation | 4–9.5% (PCIe transfer-bound) |
| Async vs sync streaming | 1.25–1.50× end-to-end (up to 2.11× forward-only) |
| Dense baseline at 256 frames | OOM (O(N²) attention) |
| Long-training stability | 28.4B, 100 SGD steps, 13.9% loss reduction |

> **Scope:** No 105B run, no H200 run, and no official VBench evaluation. The 105B figures in the paper are an analytical, PCIe-bound projection (~21% MFU) — see [`results/07_roofline`](results/07_roofline) and the paper's Table 2.

## Features

- **Models larger than GPU memory** — Full-parameter training of video DiTs to 33.3B on a single 94 GB GPU using CPU RAM for persistent state
- **3D Deformable Slide Attention** — Motion-adaptive local attention with learned offsets; linear complexity (one of three interchangeable back-ends)
- **Async Double-Buffered Streaming** — Overlaps CPU↔GPU weight transfers with compute (measured 1.25–1.50× end-to-end; up to 2.11× forward-only)
- **CPU-Resident Optimizer** — AdamW/SGD runs on the host, keeping GPU memory free for forward/backward
- **Gradient Checkpointing** — Selective activation checkpointing keeps GPU memory under budget
- **Validated PCIe roofline** — Effective host↔device bandwidth fitted from measured runs (~12.6 GB/s); used for the corrected 105B projection

## Quick Start

```bash
# Clone and setup
git clone https://github.com/jasonliu2050/megaslide.git
cd megaslide
pip install -e .

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pyyaml numpy pytest einops psutil

# Run smoke test (any GPU, <1 min)
PYTHONPATH=. python examples/train_megaslide_dit.py --config examples/configs/megaslide_dit_tiny.yaml

# Run unit tests
PYTHONPATH=. pytest tests/test_megaslide_video.py -v
```

## Architecture

MegaSlide-DiT treats the GPU as a stateless worker that processes one layer at a time:

1. **Host → GPU:** Stream weight shard for layer *l* via pinned staging buffer
2. **Compute:** Forward pass on current activation tensor using 3D-DSA
3. **GPU → Host:** Return gradients; the host optimizer (AdamW or SGD) updates the master weights
4. **Overlap:** Prefetch layer *l+1* weights while computing layer *l*

```
infinity/video/
├── model.py        # MegaSlideDiT (3D-DSA + DiT blocks)
├── attention.py    # DeformableSlideAttention3D (learned offsets, grid_sample)
├── trainer.py      # CPUMasterVideoDiT (async streaming, double-buffered)
├── baselines.py    # Dense3DDiT (O(N²)), SwinDiT (fixed windows)
├── config.py       # MegaSlideConfig (YAML-driven)
└── dataset.py      # LatentVideoDataset (synthetic + file loading)
```

### 3D Deformable Slide Attention (3D-DSA)

For each token at position (t, h, w), 3D-DSA:
1. Predicts offsets Δp via lightweight depthwise 3D convolution
2. Samples keys/values from positions (t+Δt, h+Δh, w+Δw) using trilinear interpolation
3. Computes softmax attention over the local neighbourhood (k_t=3, k_h=k_w=7)

Complexity: O(N · k_t · k_h · k_w) — linear in sequence length for fixed window sizes.

## Experiments

All experiments are self-contained and reproducible. Results are stored in `results/`:

| Experiment | Key Finding |
|------------|-------------|
| Memory scaling | Dense attention OOMs at 256 frames; MegaSlide scales |
| Quality ablation | Learned offsets converge; Swin-DiT diverges on motion data |
| Async speedup | 2.11× forward speedup at 28B parameters |
| Max scale | 33.3B model trained on single 94 GB GPU |
| Long training | 28.4B converges 13.9% over 100 steps (63 min) |

```text
results/
├── 01_smoke_tests/          # 3 models × 2 steps, tiny config
├── 02_memory_scaling/       # Dense OOM at 256 frames
├── 03_quality_ablation/     # Learned offsets vs fixed windows (256F, motion data)
├── 04_efficiency_scaling/   # Async vs sync (171M → 12.6B)
├── 05_max_scale/            # 19.7B, 28.4B, 33.3B experiments
├── 06_long_training/        # 28.4B, 100 steps convergence
└── reports/                 # Detailed analysis reports
```

### Hardware Requirements

| Experiment | GPU VRAM | RAM | Time |
|------------|----------|-----|------|
| Smoke tests | 2+ GB | 8 GB | <1 min |
| Memory scaling | 16+ GB | 32 GB | 10 min |
| Quality ablation | 8+ GB | 16 GB | 30 min |
| Efficiency (12.6B) | 24+ GB | 128 GB | 20 min |
| Max scale (33.3B) | 48+ GB | 300 GB | 30 min |
| Long training (28.4B) | 48+ GB | 250 GB | 63 min |
| 105B target (projection only) | 141 GB (H200) | 1.5 TB | analytical, not run |

### Reproducing Experiments

See [`RUN_EXPERIMENTS.md`](RUN_EXPERIMENTS.md) for the full, authoritative runbook
(experiment → paper-table map, exact configs, and the A1/A2/A3 scripts). Quick entry points:

```bash
# PCIe roofline analysis (no GPU; produces results/07_roofline)
PYTHONPATH=. python examples/analyze_roofline.py

# Core measured ladder (H100 NVL); vary config dims for each rung
PYTHONPATH=. python examples/run_megaslide_paper_experiment.py \
    --config examples/configs/megaslide_paper_experiment_256f.yaml \
    --output-dir runs/max_scale --num-steps 20

# A2: CPU-AdamW vs SGD at ~12.6B (GPU run pending)
PYTHONPATH=. python examples/run_adamw_validation.py --scale 12.6b --steps 300

# A3: fair matched-config attention comparison (GPU run pending)
PYTHONPATH=. python examples/run_fair_attention_comparison.py --layers 12 --hidden 768 --steps 400
```

## Configuration

Experiments are configured via YAML files in `examples/configs/`:

| Config | Description |
|--------|-------------|
| `megaslide_dit_tiny.yaml` | Tiny model for smoke tests |
| `megaslide_paper_experiment_tiny.yaml` | Paper experiments at reduced scale |
| `megaslide_paper_experiment_256f.yaml` | Full 256-frame paper experiments |
| `dense_baseline_64f.yaml` | Dense 3D-DiT baseline (64 frames max) |
| `swin_baseline_256f.yaml` | Swin-DiT fixed-window baseline |

## Installation

```bash
git clone https://github.com/jasonliu2050/megaslide.git
cd megaslide
pip install -e .

# Core dependencies
pip install torch>=2.0.0 transformers>=4.30.0 einops psutil pyyaml numpy

# Optional: faster attention
pip install flash-attn

# Optional: faster CPU optimizer
pip install deepspeed
```

The installation automatically builds the CUDA pipeline extension if CUDA is available.

## Paper

The full paper is available at [`paper/megaslide_dit_paper.tex`](paper/megaslide_dit_paper.tex) (LaTeX) and [`paper/megaslide_dit_paper.md`](paper/megaslide_dit_paper.md) (Markdown).

**Abstract:** We introduce MegaSlide-DiT, a CPU-master training system whose key insight is that the GPU need not own the model state: all persistent weights, master weights and optimizer moments remain in host memory, while only transient per-layer shards are streamed to the GPU on demand. We validate it on a single NVIDIA H100 NVL (94 GB HBM, 314 GB host RAM), training full-parameter video DiTs up to 33.3B parameters — whose 133 GB of weights exceed GPU memory by 1.4× — on one commodity GPU. The central finding is a capacity-for-throughput trade-off: streaming makes models far larger than HBM trainable on one GPU, but PCIe transfer becomes the bottleneck (measured MFU 4–9.5%, async overlap 1.25–1.50×). We characterise this regime with a validated PCIe roofline and project a 105B target as explicitly transfer-bound (~21% MFU). We also study 3D Deformable Slide Attention (3D-DSA) as one of three interchangeable linear-complexity attention back-ends.

## Citation

If you use MegaSlide-DiT in your research, please cite:

```bibtex
@article{liu2026megaslide,
  title={MegaSlide-DiT: Training Video Diffusion Transformers Larger Than GPU Memory via CPU-Master Streaming},
  author={Jason Liu},
  year={2026},
  institution={Trendinsight Lab / UC San Diego}
}
```

## Acknowledgements

- [HuggingFace Transformers](https://github.com/huggingface/transformers) — Model loading infrastructure
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) — SIMD-accelerated CPUAdam optimizer
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) — Memory-efficient attention kernels
- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) — Inspiration for deformable attention design

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).
