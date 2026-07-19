# A3: Fair matched-config attention comparison (pending GPU run)

Replaces the earlier 4-layer Swin "divergence" result (an unfairly shallow
baseline) with MegaSlide / Dense / Swin all built at the same depth, hidden
size and head count, on a structured-motion dataset with a held-out validation
split. Also includes the learned-offset vs frozen-offset arm.

Run on the H100 NVL box:

```bash
python examples/run_fair_attention_comparison.py --layers 12 --hidden 768 --steps 400
```

Writes `fair_attention_<L>L_<H>H.json` here. Cite held-out val MSE, peak memory
and params per model in the (demoted) attention section. If a matched-depth Swin
no longer diverges, report that honestly: it removes the prior over-claim.
