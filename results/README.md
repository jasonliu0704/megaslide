# Experiment Results

All experiments run on: **NVIDIA H100 NVL 94GB, 314 GB RAM, 40 CPUs**  
Date: 2026-05-16

---

## Directory Structure

```
results/
├── 01_smoke_tests/          # Basic validation (3 models, unit tests)
├── 02_memory_scaling/       # Dense OOM proof at 256 frames
├── 03_quality_ablation/     # Learned offsets vs fixed windows on motion data
├── 04_efficiency_scaling/   # Async vs sync from 171M to 12.6B
├── 05_max_scale/            # 19.7B, 28.4B, 33.3B experiments
├── 06_long_training/        # 28.4B, 100 steps convergence
├── reports/                 # Detailed markdown reports
└── motion_dataset_256f.pt   # Structured motion data (50 samples, 256F)
```

## Experiments Summary

| # | Experiment | Key Result |
|---|-----------|------------|
| 1 | Smoke tests + unit tests | ✅ 9/9 tests pass, all 3 models train |
| 2 | Memory scaling (16→256 frames) | ✅ Dense OOMs at 256F, MegaSlide scales |
| 3 | Quality ablation (256F, motion data) | ✅ Offsets 2% better, Swin diverges |
| 4 | Efficiency (171M→12.6B, async/sync) | ✅ 1.06×→1.26× speedup |
| 5 | Max scale (19.7B→33.3B) | ✅ 1.50× speedup, 2.11× forward |
| 6 | Long training (28.4B, 100 steps) | ✅ 13.9% loss reduction in 63 min |

## Key Numbers

| Metric | Best Result |
|--------|------------|
| Largest model | 33.3B params |
| Best forward speedup | 2.11× (forward-only; end-to-end is 1.25–1.50×) |
| Best overall speedup | 1.50× |
| GPU utilization | 78% (73/94 GB) |
| Training convergence | 13.9% loss reduction (28B, 100 steps) |
| Dense OOM confirmed | ✅ at 256 frames |
| Swin divergence | ✅ loss 1.3→3.4 on motion data |
