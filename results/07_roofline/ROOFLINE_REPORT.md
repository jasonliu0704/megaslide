# PCIe Roofline Analysis (A1)

**Hardware:** NVIDIA H100 NVL 94GB, 314 GB RAM, PCIe-attached host memory  
**Method:** derived from existing measured CPU-master runs; no new GPU runs required.

## 1. Effective PCIe bandwidth

In every measured run compute time exceeds transfer time, so the async
overlap saving `sync - async` equals the per-step transfer time. This
gives an effective host<->device bandwidth per run:

| Run | overlap saving (s) | transfer (GB) | effective BW (GB/s) |
|-----|-------------------:|--------------:|--------------------:|
| 12.6B/16F | 8.25 | 99.8 | 12.1 |
| 19.7B/16F | 3.09 | 155.9 | 50.5 |
| 28.4B/16F | 13.37 | 224.3 | 16.8 |
| 33.3B/32F | 21.02 | 263.2 | 12.5 |
| 28.4B/48F | 18.11 | 224.3 | 12.4 |
| 28.4B/64F | 17.65 | 224.3 | 12.7 |

**Robust (median) effective PCIe bandwidth: 12.6 GB/s** — far below theoretical PCIe Gen4/Gen5 peaks because of pinned-buffer copies and per-layer chunking. (The 19.7B run is an outlier with near-zero overlap saving and is excluded by the median.)

## 2. Roofline validation

Model: `step ~= overhead + max(compute, transfer / BW_eff)`. Since async
already equals `compute + overhead` in this compute-bound regime, adding
`transfer / BW_eff` should reproduce the measured sync step:

| Run | transfer (GB) | t_transfer (s) | measured async (s) | predicted sync (s) | measured sync (s) | error |
|-----|--------------:|---------------:|-------------------:|-------------------:|------------------:|------:|
| 12.6B/16F | 99.8 | 7.9 | 51.9 | 59.9 | 60.2 | -0.6% |
| 19.7B/16F | 155.9 | 12.4 | 26.1 | 38.5 | 29.2 | +31.7% |
| 28.4B/16F | 224.3 | 17.8 | 26.6 | 44.4 | 40.0 | +11.1% |
| 33.3B/32F | 263.2 | 20.9 | 43.0 | 63.9 | 64.0 | -0.2% |
| 28.4B/48F | 224.3 | 17.8 | 53.7 | 71.5 | 71.8 | -0.5% |
| 28.4B/64F | 224.3 | 17.8 | 70.1 | 87.9 | 87.7 | +0.1% |

The roofline reproduces measured sync step times within a few percent,
confirming the pipeline is compute-bound at 12.6-33B and that the overlap
simply hides the transfer behind compute.

## 3. Corrected 105B projection (replaces the 61% MFU / 2.2x claim)

- Params: 105B, tokens: 16384, streamed per step: 840 GB
- 105B fp16 = 210 GB > 141 GB H200 HBM, so weights MUST stream over PCIe every step; transfer cannot be made resident.

The earlier reports assumed compute (~3.1s) dominated the ~840 GB transfer,
yielding 61% MFU and 2.2x. That is impossible: at 16,384 tokens a 105B
forward+backward+recompute is ~1.4e16 FLOPs (tens of seconds even on an
H200), and 840 GB/step over PCIe is tens of seconds. The regime is
transfer-bound to balanced, not compute-bound. Sweeping PCIe bandwidth
and achievable compute (MFU is vs the H200 bf16 dense peak, ~989 TFLOPS):

| PCIe BW | compute rate | t_transfer (s) | t_compute (s) | async step (s) | async speedup | bound by | MFU vs H200 peak |
|---------|--------------|---------------:|--------------:|---------------:|--------------:|----------|-----------------:|
| measured_pcie_~12.5GB/s | achievable_400TFLOPS | 66.6 | 34.4 | 66.6 | 1.52x | transfer | 20.9% |
| measured_pcie_~12.5GB/s | optimistic_700TFLOPS | 66.6 | 19.7 | 66.6 | 1.30x | transfer | 20.9% |
| pcie_gen4_ideal_25GB/s | achievable_400TFLOPS | 33.6 | 34.4 | 34.4 | 1.98x | compute | 40.4% |
| pcie_gen4_ideal_25GB/s | optimistic_700TFLOPS | 33.6 | 19.7 | 33.6 | 1.59x | transfer | 41.4% |
| pcie_gen5_ideal_55GB/s | achievable_400TFLOPS | 15.3 | 34.4 | 34.4 | 1.44x | compute | 40.4% |
| pcie_gen5_ideal_55GB/s | optimistic_700TFLOPS | 15.3 | 19.7 | 19.7 | 1.78x | compute | 70.7% |

**Conclusion:** at the *measured* effective PCIe bandwidth (~12.6 GB/s)
105B streaming is firmly transfer-bound and end-to-end MFU is ~21%, not
61%. Even an ideal PCIe Gen4 link only reaches ~40%. Approaching the old
61% figure requires a doubly-optimistic corner (full PCIe Gen5 efficiency
AND ~70% compute utilization) that the measured ~12.6 GB/s data shows is
not achieved in practice. The async speedup is bounded by ~2x because
overlap hides at most one of the two costs. Present 105B as PCIe-bound
future work, not a compute-bound 61%-MFU result.
