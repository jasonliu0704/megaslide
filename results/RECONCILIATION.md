# Results Reconciliation (single source of truth)

This note resolves contradictions between the earlier per-experiment reports so
the paper cites one coherent set of numbers. Where two measurements disagree,
the authoritative choice is stated and the reason given.

## 1. Authoritative systems numbers: the `05_max_scale` series

Use `results/05_max_scale/*.json` (12.6B in `04_efficiency_scaling/max_scale_full_experiment.json`)
as the authoritative end-to-end systems results. They share one measurement
script (full step incl. optimizer + overhead) and report params, tokens,
transfer volume, async/sync step, and peak GPU/RAM consistently.

| Config | Params | Tokens | Transfer/step (GB) | Async step (s) | Sync step (s) | Overall speedup | Peak GPU (GB) | Peak RAM (GB) | MFU |
|--------|-------:|-------:|-------------------:|---------------:|--------------:|----------------:|--------------:|--------------:|----:|
| 48L/4096H/16F | 12.6B | 1024 | 99.8 | 51.9 | 60.2 | 1.16x | 47.8 | 218 | 5.7% |
| 48L/5120H/16F | 19.7B | 1024 | 155.9 | 26.1 | 29.2 | 1.12x | 19.9 | 177 | 4.4% |
| 48L/6144H/16F | 28.4B | 1024 | 224.3 | 26.6 | 40.0 | 1.50x | 25.5 | 252 | 6.3% |
| 48L/6656H/32F | 33.3B | 2048 | 263.2 | 43.0 | 64.0 | 1.49x | 45.6 | 291 | 9.1% |
| 48L/6144H/48F | 28.4B | 3072 | 224.3 | 53.7 | 71.8 | 1.34x (fwd 2.11x) | 57.2 | 252 | 9.3% |
| 48L/6144H/64F | 28.4B | 4096 | 224.3 | 70.1 | 87.7 | 1.25x (fwd 1.89x) | 73.1 | 252 | 9.5% |

Largest model trained: **33.3B params (133 GB fp32 weights) on a 94 GB GPU.**

## 2. The 12.6B "two step times" discrepancy

- `04_efficiency_scaling/max_scale_full_experiment.json`: 12.6B/16F, async 51.9 s, step FLOPs 1.99e14.
- `04_efficiency_scaling/efficiency_ablation_10b.json` + `mfu_calculation.json`: same nominal 12.6B/1024-tok, async 25.5 s, step FLOPs 4.97e13.

These are not the same measurement. The `efficiency_ablation`/`mfu_calculation`
pair is an earlier, lighter loop (≈4x lower counted FLOPs, ≈2x faster step) and
should NOT be mixed with the max-scale series. **Authoritative: the max-scale
51.9 s / 1.99e14 numbers.** The `mfu_calculation.json` 2.9% MFU figure is
superseded by the per-run MFU in the table above (5.7% for this config).

## 3. Transfer-volume figures

Per-step transfer is total host<->device traffic (weights H2D for forward +
weights H2D for backward recompute + gradients D2H), so it is roughly 2x the
fp32 weight size, scaled by passes:

- 12.6B -> 99.8 GB; 19.7B -> 155.9 GB; 28.4B -> 224.3 GB; 33.3B -> 263.2 GB.
- 105B: fp32 weights alone are 420 GB; **~840 GB/step** is the per-step total
  (the older "420 GB" figure was the one-pass weight size, not per-step traffic).

## 4. Effective PCIe bandwidth and the corrected 105B projection

From `results/07_roofline` (fit from the overlap saving `sync - async`):
**effective PCIe bandwidth ~= 12.6 GB/s** (well below theoretical Gen4/Gen5 due
to pinned-buffer copies and per-layer chunking).

**Superseded claim:** "61% MFU / 2.2x at 105B because H200 HBM is 4.8 TB/s."
This is wrong: a 105B fp16 model (210 GB) does not fit in 141 GB H200 HBM, so
weights must stream over PCIe every step. HBM bandwidth is irrelevant to the
PCIe transfer. **Corrected projection:** at the measured ~12.6 GB/s, 105B is
transfer-bound with end-to-end MFU ~= 21% (not 61%); even ideal PCIe Gen4 only
reaches ~40%. Async speedup is bounded near ~2x because overlap hides at most
one of {compute, transfer}. Present 105B as PCIe-bound future work.

## 5. Offset (deformable) benefit: contested, scale/data-dependent

- `03_quality_ablation/ablation1_offsets.json` (random noise, 171M, 100 steps):
  learned 1.7802 vs fixed 1.7975 final -> **-0.6% (no benefit)**. Expected:
  random noise has no motion for offsets to track.
- `03_quality_ablation/quality_ablation_motion.json` (structured motion, 31M,
  200 steps): learned 1.314 vs fixed 1.341 -> **+2%**; the 4-layer Swin arm
  diverges (1.3 -> 3.4).

**Authoritative framing:** the learned-offset benefit is small (~2%) and only
appears on structured-motion toy data; it is absent on random data. The Swin
"divergence" used an unfairly shallow 4-layer baseline; the fair matched-depth
re-run is now executed (A3,
`results/09_fair_attention/fair_attention_12L_768H.json`, 12L/768H/12 heads,
400 steps, held-out val): MegaSlide-learned val MSE **0.117**, MegaSlide-fixed
0.127, Dense 0.279, Swin **1.001** — all `diverged: false`. So Swin no longer
catastrophically diverges at matched depth; it just converges to a worse
val MSE on this motion task. The learned-vs-fixed gap is ~8% on val MSE
(0.117 vs 0.127). Present 3D-DSA's benefit as small, data-dependent, but now
measured-fair, not as a decisive win.

## 6. Optimizer

All max-scale systems runs and the 28B long-training run
(`06_long_training/28b_long_training.json`, SGD, 100 steps, loss 3.0 -> 2.58)
used **SGD**. CPU-AdamW (the design's intended optimizer) is now validated
separately at ~12.6B in A2
(`results/08_adamw_validation/adamw_vs_sgd_12.6b.json`, 300 steps each):
**AdamW final loss 1.005** (improvement 67.0%, avg 19.5 s/step, 198 GB host RAM)
vs **SGD 1.790** (improvement 35.8%, avg 14.5 s/step, 151 GB host RAM). CPU-
AdamW is the better optimizer when it fits; SGD remains the only viable choice
at 28-33B because AdamW state there would exceed 314 GB host RAM.

## 7. What remains NOT measured

105B params; H200 hardware; official VBench; real-video data; custom CUDA
kernels; 256-frame full-scale (grid_sample OOMs at high hidden x 16k tokens).
