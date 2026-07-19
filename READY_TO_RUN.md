# MegaSlide-DiT: Ready to Run

> **Superseded.** The authoritative, up-to-date runbook is
> [`RUN_EXPERIMENTS.md`](RUN_EXPERIMENTS.md). This file is kept only as a quick
> index and has been corrected to match the current code and thesis. Earlier
> versions of this doc listed VBench scores, a 105B headline, and CLI flags that
> no longer exist — those were wrong and have been removed.

---

## What this project measures

CPU-master weight streaming trains full-parameter video DiTs **larger than GPU
memory** on a single GPU. Validated to **33.3B parameters on one 94 GB H100 NVL**
(133 GB of weights, 1.4x HBM). The method trades throughput (4–9.5% MFU,
1.25–1.5x async overlap) for capacity. There is **no** 105B run, **no** H200 run,
and **no** official VBench evaluation — those are future work.

---

## Install

```bash
cd megaslide
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install pyyaml numpy einops pytest psutil
export PYTHONPATH="$PWD:$PYTHONPATH"
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Fastest checks

```bash
# 1. PCIe roofline analysis — no GPU, no extra deps, already produces results/07_roofline
python3 examples/analyze_roofline.py

# 2. Smoke train (tiny model, CPU-master trainer; takes only --config)
python3 examples/train_megaslide_dit.py --config examples/configs/megaslide_dit_tiny.yaml

# 3. Unit tests
pytest tests/test_megaslide_video.py -v
```

---

## Full experiment set

See [`RUN_EXPERIMENTS.md`](RUN_EXPERIMENTS.md) for:

- The experiment → paper-table map.
- The core measured ladder (12.6B → 33.3B) via `examples/run_megaslide_paper_experiment.py`.
- **A1** roofline (`examples/analyze_roofline.py`) — done.
- **A2** CPU-AdamW vs SGD (`examples/run_adamw_validation.py`) — scripted, GPU run pending.
- **A3** fair matched-config attention (`examples/run_fair_attention_comparison.py`) — scripted, GPU run pending.

---

## Hardware

| Use | GPU | Host RAM |
|-----|-----|----------|
| Roofline (A1), unit tests | none | any |
| Core ladder / A2 / A3 | 1x H100 NVL (94 GB) | ~314 GB (28–33B SGD sits near the limit) |

> AdamW at 28–33B does not fit in 314 GB host RAM; those runs use SGD. A2
> validates the CPU-AdamW path at ~12.6B, where its state fits.
