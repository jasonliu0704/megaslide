# A2: CPU-AdamW vs SGD validation (pending GPU run)

Closes the "CPU-AdamW never validated at scale" gap. The trainer defaults to
CPU-resident AdamW; prior 28-33B runs used SGD only because AdamW state exceeds
314 GB RAM at that scale. AdamW state is ~12 bytes/param, so ~12.6B fits.

Run on the H100 NVL box (CUDA + ~314 GB RAM):

```bash
python examples/run_adamw_validation.py --scale 12.6b --steps 300   # primary
python examples/run_adamw_validation.py --scale 7b   --steps 300    # lighter fallback
```

Writes `adamw_vs_sgd_<scale>.json` here. Expected runtime at 12.6B: roughly
25-50 s/step x 2 arms x 300 steps. Cite the AdamW-vs-SGD final loss and
improvement-% in the paper's optimizer/convergence section.
