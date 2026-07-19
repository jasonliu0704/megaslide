# Running the MegaSlide-DiT Experiments

This is the authoritative runbook for reproducing the experiments behind the
paper (`paper/megaslide_dit_paper.md`). It reflects the **current** code and the
**corrected thesis**: CPU-master streaming trains full-parameter video DiTs
*larger than GPU memory* (validated to **33.3B on a single 94 GB H100 NVL**),
trading throughput (4–9.5% MFU, 1.25–1.5x async overlap) for capacity. There is
no 105B run, no H200 run, and no official VBench evaluation — those are scoped as
future work, and the 105B numbers are an analytical, PCIe-bound projection.

> **TL;DR**
> - **A1 (roofline)** runs anywhere (pure Python, no GPU) and is already done.
> - **Core measured ladder (memory / async / max-scale / long-training)** needs the H100 NVL box.
> - **A2 (CPU-AdamW)** and **A3 (fair attention)** are scripted and *pending* a GPU run.

---

## 1. Hardware and environment

### Reference machine (where the measured results come from)
- **GPU:** 1x NVIDIA H100 NVL, 94 GB HBM, PCIe Gen 4 x16
- **Host:** 40 CPU cores, **314 GB DDR5 RAM**
- **Software:** PyTorch 2.11 / CUDA 13 (any PyTorch >= 2.0 with CUDA works)

The host RAM is the real constraint: persistent weights + optimizer state live on
the CPU, so the 28–33B SGD runs use ~290 GB RAM. AdamW at those sizes does **not**
fit in 314 GB (hence A2 validates AdamW at ~12.6B instead).

### Setup
```bash
cd megaslide
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install pyyaml numpy einops pytest psutil

# from the repo root, so `import infinity` resolves
export PYTHONPATH="$PWD:$PYTHONPATH"

# sanity check
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`psutil` is optional but recommended (A2 uses it to log host RAM).

---

## 2. Experiment → paper map

| ID | Script | GPU? | Output dir | Backs |
|----|--------|------|-----------|-------|
| Smoke | `examples/train_megaslide_dit.py` | optional | stdout | sanity only |
| Unit | `pytest tests/test_megaslide_video.py` | optional | stdout | correctness |
| Core: memory scaling | config-driven run (Sec. 4) | yes | `results/02_memory_scaling/` | Table 4 (Sec. 9.1) |
| Core: async vs sync | config-driven run (Sec. 4) | yes | `results/04_efficiency_scaling/`, `results/05_max_scale/` | Table 6 (Sec. 9.3) |
| Core: max scale 33.3B | config-driven run (Sec. 4) | yes | `results/05_max_scale/` | Table 7 (Sec. 9.4) |
| Core: long training | config-driven run (Sec. 4) | yes | `results/06_long_training/` | Table 8 (Sec. 9.5) |
| **A1** roofline | `examples/analyze_roofline.py` | **no** | `results/07_roofline/` | **Table 2, Sec. 6 + 9.4** |
| **A2** CPU-AdamW vs SGD | `examples/run_adamw_validation.py` | yes | `results/08_adamw_validation/` | Sec. 9.5 (optimizer) |
| **A3** fair attention | `examples/run_fair_attention_comparison.py` | yes | `results/09_fair_attention/` | Sec. 9.1–9.2 (attention) |

Status: A1 is **complete** (`results/07_roofline/` has outputs). The core ladder
JSONs are **already present** under `results/02..06`. A2 and A3 are **authored,
execution pending** (their dirs currently hold only a README).

---

## 3. A1 — PCIe roofline (no GPU, run this anywhere)

This is the most important analysis and needs no GPU or third-party packages. It
reads the existing measured runs (`results/04`, `results/05`), fits an effective
PCIe bandwidth (~12.6 GB/s), validates the roofline against the measured step
times, and emits the **corrected, transfer-bound 105B projection** that replaces
the old 61% MFU / 2.2x claim.

```bash
python3 examples/analyze_roofline.py
```

Outputs:
- `results/07_roofline/roofline_analysis.json`
- `results/07_roofline/ROOFLINE_REPORT.md`

Expected console line: `Effective PCIe bandwidth: ~12.6 GB/s`. If you add new
max-scale runs, re-run this to refresh the projection.

---

## 4. Core measured ladder (H100 NVL)

These are the experiments that produce the paper's headline evidence. They are
driven by `examples/run_megaslide_paper_experiment.py`, which trains a
`MegaSlideDiT` through the CPU-master streaming trainer and writes a JSON/summary
to `--output-dir`.

```bash
python3 examples/run_megaslide_paper_experiment.py \
    --config examples/configs/megaslide_paper_experiment_256f.yaml \
    --output-dir runs/max_scale_33b \
    --num-steps 20
```

Key flags (run with `-h` for the full list):
- `--config` — YAML model/training config (see `examples/configs/`).
- `--num-steps` — overrides `training.num_steps`.
- `--force-cpu` — disables CUDA streaming (debug only; very slow).

To reproduce each rung of the ladder, vary the model dimensions in a config
(`hidden_size`, `num_layers`, `num_heads`, `frames`) to hit the target parameter
count, and keep `batch_size: 1`. The measured configurations are:

| Run | layers | hidden | heads | frames | params | dir |
|-----|--------|--------|-------|--------|--------|-----|
| 12.6B | 48 | 4096 | 32 | 16 | 12.6B | `results/04_efficiency_scaling/` |
| 19.7B | 48 | 5120 | 40 | 16 | 19.7B | `results/05_max_scale/` |
| 28.4B | 48 | 6144 | 48 | 16–64 | 28.4B | `results/05_max_scale/` |
| 33.3B | 48 | 6656 | 52 | 32 | 33.3B | `results/05_max_scale/` |

For each run, record from the trainer output: `async`/`sync` avg step time,
`transfer_gb`, peak GPU (`torch.cuda.max_memory_allocated`), peak host RAM, and
MFU. These feed Tables 6 and 7. Memory scaling (Table 4) sweeps `frames` in
{16,32,64,128,256} at a ~1B config (12 layers, 2048 hidden) for MegaSlide, Dense,
and Swin — Dense is expected to OOM at 256 frames.

> Tip: start small (12.6B/16F, a few steps) to confirm RAM headroom before
> launching 33.3B, which sits at ~290/314 GB host RAM.

---

## 5. A2 — CPU-AdamW vs SGD (H100 NVL, pending)

Validates the CPU-resident AdamW path the design depends on. The 28–33B runs used
SGD because AdamW state (~12 bytes/param) exceeds 314 GB RAM; ~12.6B fits and lets
us compare convergence head-to-head from an identical init.

```bash
python3 examples/run_adamw_validation.py --scale 12.6b --steps 300   # primary
python3 examples/run_adamw_validation.py --scale 7b   --steps 300    # lighter fallback
```

Flags: `--scale {1b,7b,12.6b}`, `--steps`, `--adamw-lr` (default 1e-4),
`--sgd-lr` (default 1e-3), `--seed`.

Output: `results/08_adamw_validation/adamw_vs_sgd_<scale>.json`, with per-arm loss
curves, final-loss, improvement-%, and step times. Cite `adamw_final_loss` vs
`sgd_final_loss` in Section 9.5.

Cost at 12.6B: roughly 25–50 s/step x 2 arms x 300 steps ≈ 2–4 h. Use `--scale 7b`
or fewer steps if time-constrained.

---

## 6. A3 — Fair, matched-config attention comparison (H100 NVL, pending)

Removes the "rigged shallow Swin" concern: builds MegaSlide (learned offsets),
MegaSlide (frozen/zero offsets), Dense, and Swin at the **same** depth/width/heads
on a structured-motion dataset with a **held-out validation split**.

```bash
python3 examples/run_fair_attention_comparison.py --layers 12 --hidden 768 --steps 400
```

Flags: `--layers`, `--hidden`, `--heads`, `--frames` (default 32), `--size`
(default 64), `--steps`, `--lr`, `--n-train` (40), `--n-val` (10), `--seed`.

Output: `results/09_fair_attention/fair_attention_<L>L_<H>H.json`, reporting per
model: `params`, `train_loss_first20/last20`, **`val_mse`**, `peak_gpu_gb`, and a
`diverged` flag. The honest expectation: learned offsets give a *small,
data-dependent* edge; a matched-depth Swin should **not** catastrophically diverge
the way the old 4-layer baseline did. Update Section 9.2 with the real `val_mse`
numbers once run.

---

## 7. Smoke + unit tests (fast, any machine)

```bash
# 2-step smoke train (CPU-master trainer, tiny model)
python3 examples/train_megaslide_dit.py --config examples/configs/megaslide_dit_tiny.yaml

# unit tests (attention shapes/gradients, non-causality, CPU + CUDA smoke)
pytest tests/test_megaslide_video.py -v
```

Note: `train_megaslide_dit.py` takes only `--config` and always builds a
`MegaSlideDiT` (step count and model dims come from the YAML). For Dense/Swin
baselines and richer overrides, use `run_megaslide_paper_experiment.py` (Sec. 4)
or `run_fair_attention_comparison.py` (Sec. 6).

---

## 8. After running: refresh the paper

1. Drop new JSONs into the matching `results/<NN>_*/` directory.
2. Re-run `python3 examples/analyze_roofline.py` if you added max-scale runs.
3. Update the corresponding paper section/table with the measured numbers:
   - A2 → Section 9.5 (replace "execution pending" with the AdamW-vs-SGD result).
   - A3 → Section 9.2 + Table 4/5 (replace the "shallow-Swin, indicative only" caveat with the matched-depth `val_mse`).
4. Keep `results/RECONCILIATION.md` as the single source of truth for any number
   that appears in more than one place (MFU, transfer volumes, offset deltas).

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `No module named 'infinity'` | Run from repo root with `export PYTHONPATH="$PWD:$PYTHONPATH"`. |
| `No module named 'torch'` | A1 (`analyze_roofline.py`) needs no torch; everything else does — install it. |
| Host OOM at 28–33B | Expected near 314 GB; use SGD (not AdamW) at that scale, or drop a rung. |
| CUDA OOM at high frames | 3D-DSA `grid_sample` intermediates are large; reduce `frames`/`hidden_size`. |
| Dense OOMs at 256 frames | Expected — that is the O(N^2) result behind Table 4. |
| A2/A3 extremely slow | They print a warning if no CUDA is found; they are meant for the H100 NVL box. |

---

## 10. What is intentionally *not* run

Per the paper's scope, the following remain future work and should not be
fabricated: a 105B training run, any H200 measurement, official VBench scores,
real-video datasets, and custom CUDA kernels. The 105B figures are analytical and
explicitly PCIe-bound (see A1 / Section 6, Table 2).
