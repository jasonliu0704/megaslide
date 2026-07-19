# MegaSlide-DiT: Step-by-Step Experiment Procedure

> **Superseded by [`RUN_EXPERIMENTS.md`](RUN_EXPERIMENTS.md).** This file has been
> corrected to match the current code. Previous versions referenced CLI flags
> (`--num_steps`, `--model_type`) that the training script does not accept, plus
> VBench scores and a "matches paper's 2.2x" claim that are no longer part of the
> thesis. Use `RUN_EXPERIMENTS.md` as the source of truth; this is a condensed
> walkthrough.

**Purpose:** reproduce the measured results behind the paper.
**Where:** core/A2/A3 need the H100 NVL box; A1 and tests run anywhere.

---

## Step 1 — Get the code onto the box

```bash
git clone <repo-url> megaslide   # or rsync/scp the tree
cd megaslide
```

## Step 2 — Python environment

```bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install pyyaml numpy einops pytest psutil
export PYTHONPATH="$PWD:$PYTHONPATH"

python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

If you have a GPU:
```bash
python3 -c "import torch; p=torch.cuda.get_device_properties(0); print(p.name, f'{p.total_memory/1e9:.0f} GB')"
```

## Step 3 — Pre-flight: roofline + tests (no GPU needed for A1)

```bash
# PCIe roofline (pure Python; reads existing results/04 + results/05 JSONs)
python3 examples/analyze_roofline.py
#   -> results/07_roofline/roofline_analysis.json
#   -> results/07_roofline/ROOFLINE_REPORT.md
#   console prints the fitted effective PCIe bandwidth (~12.6 GB/s)

# unit tests (attention shapes/gradients, non-causality, CPU + CUDA smoke)
pytest tests/test_megaslide_video.py -v
```

## Step 4 — Smoke train

```bash
# train_megaslide_dit.py takes ONLY --config; model size and step count come
# from the YAML. It always builds a MegaSlideDiT.
python3 examples/train_megaslide_dit.py --config examples/configs/megaslide_dit_tiny.yaml
```

Success: no errors, finite loss, sub-second steps on the tiny config.

## Step 5 — Core measured ladder (H100 NVL)

Driven by `run_megaslide_paper_experiment.py` (use `-h` for all flags):

```bash
python3 examples/run_megaslide_paper_experiment.py \
    --config examples/configs/megaslide_paper_experiment_256f.yaml \
    --output-dir runs/max_scale_33b \
    --num-steps 20
```

Reproduce each rung by editing `hidden_size` / `num_layers` / `num_heads` /
`frames` in a config (keep `batch_size: 1`):

| Run | layers | hidden | heads | frames | params |
|-----|--------|--------|-------|--------|--------|
| 12.6B | 48 | 4096 | 32 | 16 | 12.6B |
| 19.7B | 48 | 5120 | 40 | 16 | 19.7B |
| 28.4B | 48 | 6144 | 48 | 16–64 | 28.4B |
| 33.3B | 48 | 6656 | 52 | 32 | 33.3B |

Record per run: async/sync avg step time, transfer GB/step, peak GPU
(`torch.cuda.max_memory_allocated`), peak host RAM, MFU. These feed Tables 6–7.

**Memory scaling (Table 4):** sweep `frames` in {16,32,64,128,256} at a ~1B
config (12 layers, 2048 hidden) for MegaSlide, Dense, and Swin. Dense is expected
to OOM at 256 frames — that OOM is the result.

> Launch 12.6B/16F for a few steps first to confirm RAM headroom; 33.3B sits near
> ~290/314 GB host RAM.

## Step 6 — A2: CPU-AdamW vs SGD (pending GPU run)

```bash
python3 examples/run_adamw_validation.py --scale 12.6b --steps 300   # primary (~2–4 h)
python3 examples/run_adamw_validation.py --scale 7b   --steps 300    # lighter fallback
#   -> results/08_adamw_validation/adamw_vs_sgd_<scale>.json
```

Cite `adamw_final_loss` vs `sgd_final_loss` (and improvement-%) in Section 9.5.

## Step 7 — A3: fair attention comparison (pending GPU run)

```bash
python3 examples/run_fair_attention_comparison.py --layers 12 --hidden 768 --steps 400
#   -> results/09_fair_attention/fair_attention_12L_768H.json
```

Builds MegaSlide (learned offsets), MegaSlide (frozen offsets), Dense, and Swin at
**matched** depth/width on structured motion with a held-out split. Report each
model's `val_mse` in Section 9.2. Expectation: learned offsets give a small,
data-dependent edge; a matched-depth Swin should not catastrophically diverge.

## Step 8 — Fold results back into the paper

1. Place new JSONs under the matching `results/<NN>_*/`.
2. Re-run `examples/analyze_roofline.py` if you added max-scale runs.
3. Update the cited section/table (A2 → 9.5, A3 → 9.2 + Tables 4/5).
4. Keep `results/RECONCILIATION.md` authoritative for any number used in >1 place.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No module named 'infinity'` | Run from repo root; `export PYTHONPATH="$PWD:$PYTHONPATH"`. |
| `No module named 'torch'` | A1 needs no torch; everything else does. |
| Host OOM at 28–33B | Expected near 314 GB — use SGD (not AdamW) or drop a rung. |
| CUDA OOM at high frames | Reduce `frames`/`hidden_size`; 3D-DSA `grid_sample` buffers are large. |
| Dense OOM at 256 frames | Expected — it is the O(N^2) result for Table 4. |
| A2/A3 very slow | They warn if no CUDA; meant for the H100 NVL box. |

## Out of scope (do not fabricate)

105B training, H200 hardware, official VBench, real-video datasets, custom CUDA
kernels. The 105B numbers are analytical and PCIe-bound (see A1 / Section 6, Table 2).
