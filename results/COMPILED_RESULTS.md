# MegaSlide-DiT: Compiled Experiment Results

All results measured on a single NVIDIA H100 NVL (94 GB HBM, 314 GB DDR5 host RAM).
Authoritative numbers per `results/RECONCILIATION.md`. All JSON artefacts are in `results/`.

---

## 1. Core Systems Ladder — Maximum Model Scale (§9.4, Table 7)

Source: `results/05_max_scale/*.json`, `results/04_efficiency_scaling/max_scale_full_experiment.json`

| Config | Params | Tokens | Transfer/step | Async step | Sync step | Speedup | Peak GPU | Peak RAM | MFU |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 48L / 4096H / 16F | 12.6B | 1,024 | 99.8 GB | 51.9 s | 60.2 s | 1.16× | 47.8 GB | 218 GB | 5.7% |
| 48L / 5120H / 16F | 19.7B | 1,024 | 155.9 GB | 26.1 s | 29.2 s | 1.12× | 19.9 GB | 177 GB | 4.4% |
| 48L / 6144H / 16F | 28.4B | 1,024 | 224.3 GB | 26.6 s | 40.0 s | 1.50× | 25.5 GB | 252 GB | 6.3% |
| **48L / 6656H / 32F** | **33.3B** | **2,048** | **263.2 GB** | **43.0 s** | **64.0 s** | **1.49×** | **45.6 GB** | **291 GB** | **9.1%** |
| 48L / 6144H / 48F | 28.4B | 3,072 | 224.3 GB | 53.7 s | 71.8 s | 1.34× (fwd 2.11×) | 57.2 GB | 252 GB | 9.3% |
| 48L / 6144H / 64F | 28.4B | 4,096 | 224.3 GB | 70.1 s | 87.7 s | 1.25× (fwd 1.89×) | 73.1 GB | 252 GB | 9.5% |

**Headline:** 33.3B params (133 GB BF16 weights) trained on a 94 GB GPU. Weights exceed HBM by 1.4×.

---

## 2. Head-to-Head: MegaSlide vs FSDP-CPU Offload (§9.6, Table 9b)

Source: `results/10_deepspeed/fsdp_cpu_*.json`

| Scale | Dense3DDiT Params | Avg Step (s) | Peak GPU | Peak RAM | OOM? |
|:---|---:|---:|---:|---:|:---:|
| 1B | 0.64B | 0.56 | 2.7 GB | 12.7 GB | No |
| 3B | 1.94B | 1.74 | 7.6 GB | 40.6 GB | No |
| 7B | 6.28B | 5.31 | 24.1 GB | 129.8 GB | No |
| 12.6B | 9.81B | 8.48 | 37.4 GB | 172.8 GB | No |
| 19.7B | 16.49B | 14.05 | 62.6 GB | 254.6 GB | No |
| **28.4B** | **25.39B** | **—** | **91.4 GB** | **94.1 GB** | **YES** |
| **33.3B** | **30.42B** | **—** | **91.4 GB** | **112.9 GB** | **YES** |

**Headline:** FSDP-CPU OOMs at 25.4B params on the same H100 NVL where MegaSlide fits 33.3B (~30% more headroom).

---

## 3. Memory Scaling: MegaSlide vs Dense vs Swin (§9.1, Table 4)

Source: `results/02_memory_scaling/memory_scaling.json`

| Frames | Tokens | MegaSlide (GPU / step) | Dense (GPU / step) | Swin (GPU / step) |
|---:|---:|:---|:---|:---|
| 16 | 256 | 6.6 GB / 1.39 s | 2.7 GB / 0.44 s | 3.1 GB / 0.44 s |
| 32 | 512 | 14.4 GB / 2.07 s | 7.0 GB / 0.50 s | 5.9 GB / 0.44 s |
| 64 | 1,024 | 27.7 GB / 4.24 s | 16.3 GB / 1.10 s | 9.0 GB / 0.64 s |
| 128 | 2,048 | 51.6 GB / 8.48 s | 45.4 GB / 3.19 s | 12.9 GB / 1.30 s |
| **256** | **4,096** | **OOM (74.7 GB)** | **OOM (81.6 GB)** | **17.7 GB / 2.69 s** |

MegaSlide and Dense both OOM at 256 frames; Swin's fixed windows survive.

---

## 4. Quality Ablation: Attention Back-ends (§9.2)

### 4a. Offset Ablation — Random Noise (Table 5a)

Source: `results/03_quality_ablation/ablation1_offsets.json`

| Variant | Params | Final Loss | Δ |
|:---|---:|---:|:---|
| Learned offsets | 171M | 1.780 | — |
| Fixed windows | 113M | 1.797 | −0.6% (within noise) |

No benefit on random data (no motion structure for offsets to exploit).

### 4b. Offset Ablation — Structured Motion (Table 5a)

Source: `results/03_quality_ablation/quality_ablation_motion.json`

| Variant | Params | Final Loss | Improvement |
|:---|---:|---:|---:|
| Learned offsets | 31M | 1.241 | 43.0% |
| Fixed windows | 24M | 1.370 | 35.3% |
| Swin (4-layer) | 24M | 2.311 | **diverged** |

+2% benefit for learned offsets on motion data. Swin divergence is due to unfairly shallow (4-layer) baseline — see 4c below.

### 4c. Fair Matched-Depth Comparison (Table 5b)

Source: `results/09_fair_attention/fair_attention_12L_768H.json`

| Variant | Params | Train Loss (last-20) | **Val MSE** | Peak GPU |
|:---|---:|---:|---:|---:|
| MegaSlide (learned) | 142M | 0.128 | **0.117** | 26.6 GB |
| MegaSlide (fixed) | 142M | 0.123 | 0.127 | 26.1 GB |
| Dense | 92M | 0.308 | 0.279 | 6.0 GB |
| Swin | 92M | 1.001 | 1.001 | 3.6 GB |

At matched depth (12L/768H/12 heads), Swin **does not diverge** — it simply converges to a worse val MSE. Learned-vs-fixed gap is ~8% on val MSE (0.117 vs 0.127).

---

## 5. Hybrid Attention: Registers + 3D-DSA

### 5a. Small Scale — 1.5M params (§9.2, Table 5c)

Source: `results/07_hybrid_attention/all_results.json`

| Variant | Params | Final Loss | Δ vs Baseline | Avg Step |
|:---|---:|---:|:---|---:|
| Baseline (pure 3D-DSA) | 940K | 0.996 | — | 0.106 s |
| Register-64 | 1.09M | 0.968 | **−2.9%** | 0.197 s (+86%) |
| Register-128 | 1.11M | 0.982 | **−1.5%** | 0.263 s (+148%) |
| Temporal anchor | 1.11M | 0.994 | −0.3% | 0.111 s (+5%) |

Registers help at small scale, but with significant step-time cost.

### 5b. 240M Scale — Scaling Test (§9.2, Table 5d)

Source: `results/10_hybrid_120M/combined_results.json`, `results/10_hybrid_120M/baseline_results.json`

| Variant | Params | Avg-Last-50 Loss | Avg Step |
|:---|---:|---:|---:|
| Baseline (pure 3D-DSA) | 205.8M | 0.371 | 0.613 s |
| Register-128 | 240.4M | 0.367 | 0.715 s (+17%) |

**Register gain collapses to ~1% (within noise) at 240M**, with +17% step-time cost.

---

## 6. Async Streaming Speedup Scaling (§9.3, Table 6)

Source: `results/05_max_scale/*.json`, `results/07_roofline/roofline_analysis.json`

| Model | Transfer/Step | End-to-End Speedup | Forward Speedup |
|:---|---:|---:|---:|
| 171M | 1.4 GB | 1.06× | — |
| 12.6B / 16F | 100 GB | 1.26× | 1.46× |
| 28.4B / 16F | 224 GB | 1.50× | — |
| 33.3B / 32F | 263 GB | 1.49× | 1.84× |
| 28.4B / 48F | 224 GB | 1.34× | **2.11×** |
| 28.4B / 64F | 224 GB | 1.25× | 1.89× |

Forward-only speedup peaks at 2.11×; end-to-end is 1.25–1.50× (backward pass dominates).

---

## 7. PCIe Roofline and 105B Projection (§9.4, Table 2)

Source: `results/07_roofline/roofline_analysis.json`

| Parameter | Value |
|:---|:---|
| Effective PCIe bandwidth | **~12.6 GB/s** |
| Roofline prediction error | <1% for 4/6 configs; 11% and 32% for two outliers |
| 105B async step (projected) | 66.6 s (transfer-bound) |
| 105B sync step (projected) | 101.0 s |
| 105B MFU (projected) | **~21%** (corrected from earlier 61% claim) |

---

## 8. CPU-AdamW vs SGD Validation (§9.5, Table 8b)

Source: `results/08_adamw_validation/adamw_vs_sgd_12.6b.json`

| Optimizer | Final Loss (last-10 avg) | Improvement | Avg Step | Peak RAM |
|:---|---:|---:|---:|---:|
| **AdamW** (lr=1e-4) | **1.005** | **67.0%** | 19.5 s | 198 GB |
| SGD (lr=1e-3) | 1.790 | 35.8% | 14.5 s | 151 GB |

AdamW is vastly better when it fits in RAM. SGD is the only option at ≥28B (AdamW state exceeds 314 GB).

---

## 9. Long Training Stability (§9.5, Table 8)

Source: `results/06_long_training/28b_long_training.json`

| Metric | Value |
|:---|:---|
| Model | 28.4B params, SGD, 100 steps |
| Duration | 62.7 min |
| Loss trajectory | first-10 avg 2.997 → last-10 avg 2.580 |
| Improvement | 13.9% |

No drift or divergence over 63-minute horizon at 28.4B scale.

---

## 10. Real-Video Validation (§9.7, Table 10)

Source: `results/10_real_video/real_video_results.json`

| Metric | Value |
|:---|:---|
| Model | 22L / 1024H / 16 heads = **370.2M params** |
| Data | 69 Disney videos → 207 clips (176 train / 31 val) |
| Latent shape | [4, 16, 32, 32] (SD VAE `sd-vae-ft-mse`) |
| Train loss | 2.84 → 2.53 (avg-last-50: **2.49**, −12.3%) |
| Best val loss | **5.54** (step 800) |
| Final val loss | 33.7 (severe overfitting) |
| Avg step time | 1.44 s |
| Peak GPU | 1.75 GB |
| Peak RAM | 7.1 GB |
| Total time | 1,493 s (~25 min) |

Pipeline validated on real VAE-encoded video. Val diverges due to data scarcity (176 clips for 370M params). No generative quality claim.

---

## Summary Scorecard

| Claim | Status | Key Evidence |
|:---|:---|:---|
| CPU-master enables models >> GPU memory | **Supported** | 33.3B (133 GB weights) on 94 GB GPU |
| MegaSlide fits where FSDP-CPU OOMs | **Supported** | FSDP OOMs at 25.4B; MegaSlide fits 33.3B |
| Dense attention OOMs at 256 frames | **Supported** | Dense + MegaSlide OOM; Swin survives |
| Async streaming speeds up training | **Supported (modest)** | 1.25–1.50× end-to-end; up to 2.11× fwd-only |
| Learned offsets improve quality | **Small, data-dependent** | +8% val MSE on motion; 0% on random data |
| Registers improve local attention | **At small scale only** | +2.9% at 1.5M; collapses to ~1% at 240M |
| CPU-AdamW is numerically stable | **Supported** | 67% improvement at 12.6B (300 steps) |
| Pipeline works on real video | **Supported (pipeline only)** | Train loss ↓12.3%; val diverges (data-scarce) |
| 105B / H200 / VBench | **Not done** | Projection only: ~21% MFU (PCIe-bound) |

---

## What Remains NOT Measured

- 105B parameters
- H200 hardware
- Official VBench evaluation
- WebVid/Panda-70M scale real video (need >10K clips for quality claim)
- Custom CUDA/Triton kernels for 3D-DSA
- 256-frame training at full hidden size (grid_sample OOM)
