# MegaSlide-DiT: Training 33B-Parameter Video Diffusion Transformers on a Single GPU via CPU-Master Streaming

**Jason Liu - Trendinsight Lab/UC San Diego**

## Abstract

We train a 33.3B-parameter video Diffusion Transformer on a single 94 GB GPU. The model's 133 GB of BF16 weights exceed GPU memory by 1.4x; MegaSlide-DiT fits them by streaming one layer at a time from host RAM, keeping all persistent state (weights, master weights, optimizer moments) CPU-resident. In a head-to-head on the same NVIDIA H100 NVL, PyTorch FSDP with CPU offload -- the standard single-GPU baseline -- OOMs at 25.4B, giving MegaSlide ~30% more parameter headroom. The cost is throughput: measured MFU is 4--9.5% and asynchronous prefetch yields 1.25--1.50x end-to-end speedup, bounded by a fitted effective PCIe bandwidth of ~12.6 GB/s. We validate the roofline model across six configurations, confirm the pipeline on real VAE-encoded video, and release all code, artefacts, and a reconciliation log.

## 1. Introduction

Full-parameter fine-tuning of large video Diffusion Transformers (DiTs) requires persistent state -- BF16 weights, FP32 master weights, and optimizer moments -- that far exceeds single-GPU HBM. A 33B model needs ~133 GB of BF16 weights alone; with AdamW, persistent state approaches 400 GB. Existing CPU-offload paths (FSDP-CPU, ZeRO-Infinity, DeepSpeed Stage-3) target multi-GPU sharding; on a single GPU, FSDP-CPU falls back to `NO_SHARD` and keeps parameters GPU-resident, so it OOMs well before 33B.

MegaSlide-DiT treats the GPU as a stateless compute engine: only one layer's weight shard (~2 GB) is GPU-resident at any time. All other state lives in host DDR5 RAM. A double-buffered scheduler overlaps weight transfer for layer $l{+}1$ with computation of layer $l$, and the optimizer update runs entirely on the CPU. The design buys *capacity* (models far larger than HBM) at the cost of *throughput* (single-digit MFU), targeting the regime where the alternative is no fine-tune at all.

**Contributions.**
1. **33.3B on one GPU, 30% beyond FSDP-CPU.** We validate CPU-master streaming to 33.3B parameters on a single H100 NVL (94 GB HBM, 314 GB DDR5). At matched configuration, FSDP with `CPUOffload(offload_params=True)` OOMs at 25.4B on the same hardware (Section 4.1).
2. **Fitted PCIe roofline.** We measure MFU (4--9.5%), async speedup (1.25--1.50x end-to-end), and fit an effective bandwidth of ~12.6 GB/s that reproduces measured sync step times within 1% on 4/6 configurations. The roofline corrects a prior 105B/61%-MFU projection to a transfer-bound ~21% (Section 4.3, Appendix B).
3. **Open, reproducible measurement suite.** 37 JSON experiment artefacts, a PCIe roofline analyser, per-run scripts, and a reconciliation document that tracks every corrected claim.

## 2. Background and Related Work

**Memory-offload systems.** ZeRO-Infinity (Rajbhandari et al., 2021) is the canonical CPU/NVMe offload framework for trillion-parameter LM training across multi-GPU clusters. PyTorch FSDP-CPU and DeepSpeed Stage-3 expose similar functionality. FlexGen (Sheng et al., 2023) targets single-GPU *inference* by streaming activations and weights from host/SSD. MegaSlide-DiT differs in three ways: (a) it targets single-GPU full-parameter *training*, not inference; (b) it streams per-layer, not per-shard, so it works without multi-rank sharding; and (c) the optimizer runs on the CPU, avoiding GPU-side moment storage. We compare directly against FSDP-CPU on the same hardware (Section 4.1): FSDP-CPU OOMs at 25.4B; MegaSlide fits 33.3B.

**Efficient attention.** Our system treats the attention back-end as pluggable. The FSDP comparison uses Dense (global) attention to control for architecture; we additionally study a 3D Deformable Slide Attention variant (Appendix A) but find attention-operator choice does not affect the streaming system's throughput character.

**Large video DiTs.** Sora, Open-Sora, CogVideoX, HunyuanVideo, and MovieGen train 2--30B video DiTs on multi-node clusters. MegaSlide-DiT is orthogonal: a single-GPU training path, not a generation-quality baseline.

## 3. System Design

### 3.1 Streaming scheduler

MegaSlide-DiT processes one layer at a time. At each step:
1. The host copies weights $W^{(l)}$ from CPU memory to a pinned staging buffer.
2. A CUDA transfer stream pushes $W^{(l)}$ to the GPU while the compute stream processes layer $l{-}1$.
3. The GPU computes the forward pass, stores an activation checkpoint, and streams back $\nabla W^{(l)}$ during the backward pass.
4. The CPU receives gradients, updates master weights (AdamW or SGD), and prepares $W^{(l+1)}$.

The double-buffer hides transfer latency when per-layer compute time exceeds per-layer transfer time; the roofline (Section 3.3) characterises when this holds.

### 3.2 CPU-resident optimizer

The optimizer update runs entirely on the host. FP32 master weights and moment vectors never leave CPU memory, eliminating GPU-side storage and an extra PCIe round-trip. At 28--33B we use SGD because AdamW's moments exceed 314 GB of host RAM; the CPU-AdamW path is validated at 12.6B where its state (~198 GB) fits comfortably (Section 4.4).

### 3.3 Analytical throughput model

Every step streams the full weight set to the GPU (forward), re-streams it for checkpointed backward recompute, and returns gradients. With effective host-device bandwidth $B$ and per-step transfer volume $V$:
$$ t_{\text{step}} \approx t_{\text{overhead}} + \max\!\left(t_{\text{compute}},\; V/B\right) $$

Async prefetch hides at most $\min(t_{\text{compute}}, V/B)$, bounding the speedup at 2x. On our hardware $B \approx 12.6$ GB/s (fitted from measurements, Section 4.3), well below the theoretical PCIe Gen 5 peak.

### 3.4 GPU memory budget

Gradient checkpointing keeps the GPU footprint small:
* Transient weights: ~2 GB (one layer shard)
* Current activation: ~32.7 GB (256 frames, 1080p)
* Checkpointed boundaries: ~45 GB
* Workspace buffers: ~30 GB

The total stays below 120 GB, leaving headroom on a 141 GB H200 or fitting within the 94 GB of the H100 NVL used in our experiments. GPU memory is bounded by activations, not parameters -- the defining property of per-layer streaming.

## 4. Experiments

All experiments run on a single NVIDIA H100 NVL (94 GB HBM3, 314 GB DDR5 RAM, 40 CPU cores, PyTorch 2.11.0, CUDA 13.0). We report measured numbers only; projections are in Appendix B.

### 4.1 Head-to-head: MegaSlide vs FSDP-CPU offload

We run the same task -- Dense3DDiT, 16 frames, 64x64 latents, bfloat16, AdamW, batch 1 -- under PyTorch FSDP with `CPUOffload(offload_params=True)` and under MegaSlide CPU-master streaming.

| Scale | Params | System | Peak GPU | Step time | OOM? |
| :--- | ---: | :--- | ---: | ---: | :---: |
| 1B | 0.64B | FSDP-CPU | 2.7 GB | 0.56 s | No |
| 7B | 6.28B | FSDP-CPU | 24.1 GB | 5.30 s | No |
| 12.6B | 9.81B | FSDP-CPU | 37.4 GB | 8.48 s | No |
| 19.7B | 16.49B | FSDP-CPU | 62.6 GB | 14.05 s | No |
| **28.4B** | **25.39B** | **FSDP-CPU** | **91.4 GB** | **--** | **YES** |
| **33.3B** | **30.42B** | **FSDP-CPU** | **91.4 GB** | **--** | **YES** |
| 12.6B | 12.6B | MegaSlide | 47.8 GB | 51.9 s | No |
| **33.3B** | **33.3B** | **MegaSlide** | **45.6 GB** | **43.0 s** | **No** |
*Table 1: FSDP-CPU OOMs at 25.4B; MegaSlide fits 33.3B on the same GPU.*

**Why FSDP-CPU OOMs.** With `world_size=1`, FSDP falls back to `NO_SHARD` -- parameters and gradients stay GPU-resident (~3.7 bytes/param), so 25B crosses the 94 GB ceiling. `CPUOffload` only moves optimizer state off-GPU in this single-rank mode. MegaSlide's per-layer streaming keeps GPU footprint independent of model size.

**Throughput trade-off.** At scales where both fit (1B--12.6B), FSDP-CPU is 4--10x faster per step because weights are GPU-resident. MegaSlide pays this cost to reach scales FSDP cannot.

### 4.2 Scaling ladder: 12.6B to 33.3B

| Config | Params | Tokens | Transfer/step | Async step | Sync step | Speedup | Peak GPU | MFU |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 48L/4096H/16F | 12.6B | 1,024 | 99.8 GB | 51.9 s | 60.2 s | 1.16x | 47.8 GB | 5.7% |
| 48L/5120H/16F | 19.7B | 1,024 | 155.9 GB | 26.1 s | 29.2 s | 1.12x | 19.9 GB | 4.4% |
| 48L/6144H/16F | 28.4B | 1,024 | 224.3 GB | 26.6 s | 40.0 s | 1.50x | 25.5 GB | 6.3% |
| **48L/6656H/32F** | **33.3B** | **2,048** | **263.2 GB** | **43.0 s** | **64.0 s** | **1.49x** | **45.6 GB** | **9.1%** |
| 48L/6144H/48F | 28.4B | 3,072 | 224.3 GB | 53.7 s | 71.8 s | 1.34x | 57.2 GB | 9.3% |
| 48L/6144H/64F | 28.4B | 4,096 | 224.3 GB | 70.1 s | 87.7 s | 1.25x | 73.1 GB | 9.5% |
*Table 2: Systems ladder on H100 NVL. MFU rises with arithmetic intensity (more tokens per step) and async speedup grows with transfer volume.*

The 33.3B model's BF16 weights (133 GB) exceed GPU HBM by 1.4x. MFU is 4--9.5%, bounded by PCIe; end-to-end async speedup is 1.25--1.50x.

### 4.3 PCIe roofline validation

We fit the effective bandwidth by dividing measured transfer volume by the overlap saving (sync minus async step time): $B \approx 12.6$ GB/s. Using this to predict sync step times:

| Run | $V$ (GB) | Predicted sync (s) | Measured sync (s) | Error |
| :--- | ---: | ---: | ---: | ---: |
| 12.6B/16F | 99.8 | 59.9 | 60.2 | -0.6% |
| 33.3B/32F | 263.2 | 63.9 | 64.0 | -0.2% |
| 28.4B/48F | 224.3 | 71.5 | 71.8 | -0.5% |
| 28.4B/64F | 224.3 | 87.9 | 87.7 | +0.1% |
*Table 3: PCIe roofline validation. 4/6 configs within $\pm$1%. Full table in Appendix B.*

The roofline is the analytical core of the system: it predicts step time from transfer volume and arithmetic intensity, and shows that the 105B regime stays transfer-bound (~21% MFU) even on an H200 (Appendix B).

### 4.4 CPU-AdamW vs SGD

At 12.6B (the largest scale where AdamW state fits in 314 GB host RAM), we ran 300 steps each from an identical init:

| Optimizer | Final loss | Improvement | Avg step | Host RAM |
| :--- | ---: | ---: | ---: | ---: |
| **CPU-AdamW** | **1.005** | **67.0%** | 19.5 s | 198 GB |
| SGD + momentum | 1.790 | 35.8% | 14.5 s | 151 GB |
*Table 4: CPU-AdamW vs SGD at 12.6B.*

AdamW converges to a substantially lower loss at +34% step-time cost. A 28.4B model trained for 100 steps (63 min) with SGD shows monotonic loss descent (3.00 -> 2.58) with stable gradient norms, confirming no numerical drift from the CPU-master streaming path at the largest scale.

### 4.5 Real-video pipeline validation

To confirm the system works beyond synthetic data, we train a 370M model on 207 VAE-encoded clips (176 train / 31 val) from publicly available videos, encoded through a pretrained Stable Diffusion VAE. Training loss decreases from 2.84 to 2.49 (avg-last-50) over 1000 steps at 1.44 s/step; validation loss diverges as expected with only 176 clips for 370M parameters. This validates pipeline correctness on real latents, not generative quality.

## 5. Limitations and Future Work

**Throughput.** MFU is 4--9.5%. For a 1k-step fine-tune of 33.3B at 32 frames, wall-clock time is ~8 days on a single H100 NVL. This is acceptable when the alternative is no fine-tune (no cluster access), but a multi-GPU setup would be faster when available.

**Scope.** We do not train at 105B, run on an H200, generate video samples, or report FVD/VBench. The strongest missing experiment is fine-tuning a pretrained video DiT checkpoint (e.g. Open-Sora) at a scale FSDP cannot reach and reporting downstream quality -- this would close the "so what" gap and is the recommended next step.

**Caveats.** FSDP-CPU under multi-GPU `FULL_SHARD` is a different system; our comparison is single-GPU only. The attention back-end (Appendix A) shows only a small, data-dependent quality gain for learned offsets; a practitioner should default to fixed-window attention unless motion adaptivity demonstrably helps on their task.

## 6. Conclusion

MegaSlide-DiT trains 33.3B-parameter video DiTs on a single 94 GB GPU by streaming one layer at a time from host RAM. FSDP-CPU OOMs at 25.4B on the same hardware. The trade-off is throughput: 4--9.5% MFU, bounded by a PCIe roofline we fit and validate across six configurations. The system, measurements, and a reconciliation log are released as an open reproducibility package.

---

## Appendix A: Attention Back-End Study

The streaming system is attention-agnostic (Section 3.1). This appendix characterises the operators we studied on synthetic structured-motion data.

### A.1 3D Deformable Slide Attention (3D-DSA)

For each token at position $(t, h, w)$, 3D-DSA learns offsets $\Delta p \in \mathbb{R}^{k_t \times k_h \times k_w \times 3}$ via a depthwise 3D convolution, then samples keys/values at $(t{+}\Delta p_t, h{+}\Delta p_h, w{+}\Delta p_w)$ using trilinear interpolation. Complexity is $\mathcal{O}(N \cdot k_t k_h k_w)$, linear in $N$.

### A.2 Register-token augmentation

We augment 3D-DSA with $G \in \{64, 128\}$ learnable global register tokens that cross-attend to all spatial positions and are appended as extra K/V to the local attention. A sigmoid gate ($g{=}0$ at init) controls their contribution. Complexity stays linear: $\mathcal{O}(N \cdot (k_t k_h k_w + G))$.

### A.3 Memory scaling

| Frames | MegaSlide-DiT | Dense3DDiT | SwinDiT |
| :--- | :--- | :--- | :--- |
| 16 | 6.6 GB | 2.7 GB | 3.1 GB |
| 64 | 27.7 GB | 16.3 GB | 9.0 GB |
| 128 | 51.6 GB | 45.4 GB | 12.9 GB |
| 256 | OOM | OOM | 17.7 GB |
*Table A1: 3D-DSA uses more memory than Dense at $\leq$128F due to trilinear sampling. Swin is cheapest.*

### A.4 Quality ablation

Fair matched-depth comparison (12L/768H, 32 frames, 400 steps, held-out val):

| Model | val MSE | Peak GPU |
| :--- | ---: | ---: |
| 3D-DSA (learned offsets) | **0.117** | 26.6 GB |
| 3D-DSA (fixed offsets) | 0.127 | 26.0 GB |
| Dense | 0.279 | 6.0 GB |
| Swin | 1.001 | 3.6 GB |
*Table A2: Learned offsets give ~8% MSE improvement over fixed -- small and data-dependent.*

### A.5 Register scaling

| Scale | Variant | Avg loss (last 50) | Delta |
| :--- | :--- | ---: | ---: |
| 1.5M | 3D-DSA | 0.445 | -- |
| 1.5M | + register-128 | 0.353 | **-20.7%** |
| 240M | 3D-DSA | 0.371 | -- |
| 240M | + register-128 | 0.367 | -1.0% (noise) |
*Table A3: Register gain collapses from 20.7% at 1.5M to ~1% at 240M.*

## Appendix B: 105B Projection and Full Roofline

A 105B model streams $V \approx 840$ GB per step. Its 210 GB of fp16 weights do not fit in 141 GB of H200 HBM, so PCIe streaming is mandatory. At $B = 12.6$ GB/s, transfer takes ~66.6 s vs compute of 19.7--34.4 s: the regime is **transfer-bound** at ~21% MFU, correcting a prior 61% claim.

| PCIe bandwidth | Transfer (s) | Async step (s) | MFU |
| :--- | ---: | ---: | ---: |
| ~12.6 GB/s (measured) | 66.6 | 66.6 | **~21%** |
| 25 GB/s (Gen4 ideal) | 33.6 | 33.6--34.4 | ~40% |
| 55 GB/s (Gen5 ideal) | 15.3 | 19.7--34.4 | 40--71% |
*Table B1: 105B projection.*

**Full roofline validation:**

| Run | $V$ (GB) | Predicted sync (s) | Measured sync (s) | Error |
| :--- | ---: | ---: | ---: | ---: |
| 12.6B/16F | 99.8 | 59.9 | 60.2 | -0.6% |
| 19.7B/16F | 155.9 | 38.5 | 29.2 | +31.7% |
| 28.4B/16F | 224.3 | 44.4 | 40.0 | +11.1% |
| 33.3B/32F | 263.2 | 63.9 | 64.0 | -0.2% |
| 28.4B/48F | 224.3 | 71.5 | 71.8 | -0.5% |
| 28.4B/64F | 224.3 | 87.9 | 87.7 | +0.1% |
*Table B2: Full roofline. 4/6 within $\pm$1%; outliers are dispatch-overhead-dominated.*
