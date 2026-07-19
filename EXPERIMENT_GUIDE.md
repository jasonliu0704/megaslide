# Experiment Guide

> **Superseded by [`RUN_EXPERIMENTS.md`](RUN_EXPERIMENTS.md).** This guide has been
> corrected. Earlier versions listed fabricated VBench scores (0.83/0.88), a "61%
> MFU / 2.2x async" claim, a 105B headline, and `--num_steps` / `--model_type`
> flags that the scripts do not accept. None of those are real: the measured
> evidence ceiling is 33.3B on a single H100 NVL, MFU is 4–9.5%, async overlap is
> 1.25–1.5x, and VBench was never run. Use `RUN_EXPERIMENTS.md` as the source of
> truth.

---

## Prerequisites

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install pyyaml numpy einops pytest psutil
export PYTHONPATH="$PWD:$PYTHONPATH"
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Quick start

```bash
# A1: PCIe roofline — pure Python, no GPU; already produces results/07_roofline
python3 examples/analyze_roofline.py

# smoke train (tiny model; train_megaslide_dit.py takes ONLY --config)
python3 examples/train_megaslide_dit.py --config examples/configs/megaslide_dit_tiny.yaml

# unit tests
pytest tests/test_megaslide_video.py -v
```

---

## Experiments

### Core measured ladder (H100 NVL)

Driven by `examples/run_megaslide_paper_experiment.py` (`-h` for flags:
`--config`, `--output-dir`, `--num-steps`, `--force-cpu`). Train a `MegaSlideDiT`
through the CPU-master streaming trainer and record async/sync step time,
transfer GB/step, peak GPU, peak host RAM, and MFU.

```bash
python3 examples/run_megaslide_paper_experiment.py \
    --config examples/configs/megaslide_paper_experiment_256f.yaml \
    --output-dir runs/max_scale_33b \
    --num-steps 20
```

Reproduce each rung by editing the YAML dims (keep `batch_size: 1`):

| Run | layers | hidden | heads | frames | params | backs |
|-----|--------|--------|-------|--------|--------|-------|
| 12.6B | 48 | 4096 | 32 | 16 | 12.6B | Table 6 |
| 19.7B | 48 | 5120 | 40 | 16 | 19.7B | Table 6 |
| 28.4B | 48 | 6144 | 48 | 16–64 | 28.4B | Tables 6–7 |
| 33.3B | 48 | 6656 | 52 | 32 | 33.3B | Table 7 |

**Memory scaling (Table 4):** sweep `frames` in {16,32,64,128,256} at a ~1B config
(12 layers, 2048 hidden) for MegaSlide / Dense / Swin. **Dense is expected to OOM
at 256 frames** — that OOM is the result. Note that 3D-DSA carries a large
constant overhead and uses *more* memory than Dense at low/mid frame counts; Swin
is the most memory-efficient operator.

### A2 — CPU-AdamW vs SGD (pending GPU run)

```bash
python3 examples/run_adamw_validation.py --scale 12.6b --steps 300   # ~2–4 h
python3 examples/run_adamw_validation.py --scale 7b   --steps 300    # fallback
```
Flags: `--scale {1b,7b,12.6b}`, `--steps`, `--adamw-lr`, `--sgd-lr`, `--seed`.
Output: `results/08_adamw_validation/adamw_vs_sgd_<scale>.json`. The 28–33B runs
use SGD because AdamW state exceeds 314 GB host RAM at that scale; A2 validates the
CPU-AdamW path at ~12.6B where it fits. Cite the AdamW-vs-SGD final loss in
Section 9.5.

### A3 — Fair, matched-config attention (pending GPU run)

```bash
python3 examples/run_fair_attention_comparison.py --layers 12 --hidden 768 --steps 400
```
Flags: `--layers`, `--hidden`, `--heads`, `--frames`, `--size`, `--steps`, `--lr`,
`--n-train`, `--n-val`, `--seed`. Builds MegaSlide (learned offsets), MegaSlide
(frozen offsets), Dense, and Swin at **matched** depth/width on structured motion
with a **held-out split**, reporting each model's `val_mse`. This replaces the old
unfair 4-layer Swin baseline. Expectation: learned offsets give a *small,
data-dependent* edge; a matched-depth Swin should not catastrophically diverge.

---

## Expected results (measured, H100 NVL)

| Claim | Result |
|-------|--------|
| CPU-master enables models >> GPU memory | 33.3B (133 GB weights) on a 94 GB GPU |
| Dense global attention OOMs at long sequences | OOM at 256 frames (Table 4) |
| Async streaming speeds up training | 1.25–1.50x end-to-end (2.11x forward-only) |
| MFU | 4–9.5% across the ladder (PCIe transfer-bound) |
| Long-training stability | 28.4B, 100 SGD steps, 13.9% loss reduction, stable grad norm |
| Learned offsets help | small, data-dependent (+2% motion, −0.6% random); A3 re-test pending |

**Not measured / out of scope:** 105B training, H200 hardware, official VBench,
real-video data, custom CUDA kernels. The 105B numbers are analytical and
PCIe-bound (~21% MFU); see `results/07_roofline` and paper Section 6, Table 2.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No module named 'infinity'` | Run from repo root; `export PYTHONPATH="$PWD:$PYTHONPATH"`. |
| `No module named 'torch'` | A1 needs no torch; everything else does. |
| Host OOM at 28–33B | Expected near 314 GB — use SGD, not AdamW, or drop a rung. |
| CUDA OOM at high frames | Reduce `frames`/`hidden_size` (3D-DSA `grid_sample` buffers are large). |
| Dense OOM at 256 frames | Expected — it is the O(N^2) result for Table 4. |
| A2/A3 very slow | They warn if no CUDA; meant for the H100 NVL box. |

---

## Unit tests

```bash
pytest tests/test_megaslide_video.py -v
```
