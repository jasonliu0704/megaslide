# Future Work: MegaSlide-DiT

Prioritized directions for extending MegaSlide-DiT, organized by impact and feasibility.

---

## 1. Hybrid Attention (Global + Local 3D-DSA) ✅ IMPLEMENTED

**Problem:** Pure local attention misses long-range dependencies — scene-level consistency suffers for prompts with far-apart object interactions or abrupt scene changes.

**Approach:**
- Insert sparse global attention tokens (e.g., every 64th token attends globally)
- Combine with existing 3D-DSA for local detail
- Could use a "global register" approach (like ViT register tokens)

**Status:** Implemented and validated. See `results/07_hybrid_attention/report.md`.

**Results (200 steps, synthetic motion data):**
| Variant | Final Loss | Δ vs Baseline | Step Overhead |
|---------|-----------|---------------|---------------|
| Baseline (pure 3D-DSA) | 0.9963 | — | — |
| Register-64 | **0.9676** | **-2.9%** | +86% |
| Register-128 | 0.9817 | -1.5% | +148% |
| Temporal Anchor | 0.9937 | -0.3% | +5% |

**Recommendation:** Register-64 with `register_interval=4` for production. Temporal anchors for latency-sensitive applications.

**Files:** `infinity/video/hybrid_attention.py`, `infinity/video/hybrid_model.py`, `infinity/video/metrics.py`

---

## 2. INT8/FP16 Quantization

**Problem:** 105B model requires 1.5 TB RAM. Quantizing master weights and optimizer states to INT8 would halve this to ~750 GB.

**Approach:**
- Quantize CPU-resident master weights to INT8, dequantize on-the-fly during streaming
- Keep FP32 optimizer moments but explore INT8 moment compression (like 8-bit Adam)
- FP16 gradients already used; could push to FP8 on Hopper GPUs

**Effort:** Medium (~2 weeks)  
**Impact:** High — doubles accessible model size or halves hardware cost  
**Files to modify:** `infinity/video/trainer.py`, new `infinity/video/quantization.py`

---

## 3. NVMe Offloading

**Problem:** Not everyone has 1.5 TB RAM. NVMe SSDs (8+ TB, 7 GB/s read) could store optimizer states.

**Approach:**
- Tier storage: hot layers in RAM, cold optimizer states on NVMe
- Prefetch optimizer states for upcoming layers during backward pass
- Follow DeepSpeed ZeRO-Infinity patterns

**Effort:** High (~3-4 weeks)  
**Impact:** High — enables 105B training on machines with 256-512 GB RAM  
**Dependencies:** Fast NVMe (PCIe 5.0 recommended)

---

## 4. Multi-GPU Tensor Parallelism

**Problem:** Single-GPU throughput is limited. MFU is only 9.5% on PCIe-bound H100 NVL.

**Approach:**
- Shard layers across 2-4 GPUs with NVLink
- Keep CPU-master architecture but stream to multiple GPUs
- Could enable pre-training from scratch (not just fine-tuning)

**Effort:** High (~4-6 weeks)  
**Impact:** Very high — unlocks pre-training and dramatically improves throughput  
**Files to modify:** `infinity/video/trainer.py`, new `infinity/video/parallel.py`

---

## 5. Adaptive Kernel Size Learning

**Problem:** Fixed DSA kernel size `(k_t, k_h, k_w)` is suboptimal — fast motion needs larger temporal windows, static scenes need smaller ones.

**Approach:**
- Predict per-token kernel size from content features
- Use a lightweight MLP to output scale factors for the offset grid
- Differentiable masking of kernel positions

**Effort:** Low-Medium (~1-2 weeks)  
**Impact:** Medium — improves quality on diverse motion patterns  
**Files to modify:** `infinity/video/attention.py`

---

## 6. Custom Triton Kernels for 3D-DSA

**Problem:** Current implementation uses `F.grid_sample` with complex reshaping. Custom kernels could fuse operations and reduce memory.

**Approach:**
- Fused trilinear sampling + attention kernel in Triton
- Eliminate intermediate tensors from grid_sample reshaping
- Could reduce activation memory by 30-50%

**Effort:** High (~3-4 weeks, requires Triton expertise)  
**Impact:** Medium-High — faster forward/backward, lower memory  
**New files:** `infinity/video/triton_kernels/`

---

## 7. Real Video Data Training

**Problem:** All current experiments use synthetic/motion data. Need validation on real video latents.

**Approach:**
- Encode real videos through a pre-trained VAE (SD-VAE or CogVideoX-VAE)
- Train on WebVid-10M or Panda-70M latents
- Evaluate with FVD, FID-VID, and VBench

**Effort:** Medium (~2-3 weeks, mostly data pipeline)  
**Impact:** High — validates the approach on real-world data  
**Files to modify:** `infinity/video/dataset.py`, new data pipeline scripts

---

## 8. Conditional Generation (Text-to-Video)

**Problem:** Current training is unconditional noise prediction. Need text conditioning for practical use.

**Approach:**
- Integrate CLIP/T5 text encoder (CPU-offloaded like other weights)
- Cross-attention already stubbed in `model.py`
- Classifier-free guidance during sampling

**Effort:** Medium (~2 weeks)  
**Impact:** High — required for any practical video generation  
**Files to modify:** `infinity/video/model.py`, `examples/run_vbench_evaluation.py`

---

## 9. Gradient Accumulation Optimization

**Problem:** Large effective batch sizes require many micro-batches, each streaming all layers.

**Approach:**
- Accumulate gradients across micro-batches before optimizer step
- Amortize weight streaming cost over multiple forward/backward passes
- Could improve effective throughput by 2-4x for large batch training

**Effort:** Low (~1 week)  
**Impact:** Medium — better throughput for large-batch regimes  
**Files to modify:** `infinity/video/trainer.py`

---

## 10. LoRA + Full-Parameter Hybrid

**Problem:** Full-parameter adaptation is slow. LoRA is fast but limited.

**Approach:**
- LoRA on attention layers + full-parameter on MLP layers
- Or: LoRA warmup → full-parameter fine-tuning
- Reduces streaming volume while maintaining quality

**Effort:** Medium (~2 weeks)  
**Impact:** Medium — practical speedup for adaptation tasks  
**New files:** `infinity/video/lora.py`

---

## Priority Ranking

| Priority | Direction | Reason |
|:---------|:----------|:-------|
| P0 | Hybrid Attention | Fixes main quality limitation |
| P0 | Real Video Data | Validates approach on real data |
| P1 | INT8 Quantization | Doubles accessible scale |
| P1 | Text Conditioning | Required for practical use |
| P1 | Custom Triton Kernels | Performance + memory gains |
| P2 | NVMe Offloading | Broadens hardware accessibility |
| P2 | Adaptive Kernel Size | Quality improvement |
| P2 | Gradient Accumulation | Easy throughput win |
| P3 | Multi-GPU TP | Major effort, major payoff |
| P3 | LoRA Hybrid | Nice-to-have optimization |

---

## Quick Wins (< 1 week each)

1. **Gradient accumulation** — straightforward loop change in trainer
2. **FP16 optimizer moments** — swap dtype in CPU Adam, measure quality impact
3. **Cosine LR schedule** — add to trainer, currently only constant LR
4. **EMA weights** — exponential moving average for better generation quality
5. **Mixed kernel sizes per layer** — early layers use larger temporal kernel, later layers smaller
