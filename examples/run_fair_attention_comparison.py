"""A3: Fair, matched-config comparison of the three attention operators.

The earlier motion-quality ablation (results/03_quality_ablation) compared a
4-layer Swin against MegaSlide and reported that Swin "diverges". A 4-layer
network is an unusually shallow, easy-to-destabilise baseline, so that result is
not a fair test of fixed-window attention. This script re-runs the comparison
with MegaSlide-DiT, Dense3DDiT and SwinDiT all built at the *same* depth, hidden
size and head count, on a structured-motion dataset with a held-out validation
split, and also includes the learned-offset vs frozen-offset arm.

Must run on the H100 NVL box. Small models, so it uses a plain GPU optimizer
(no streaming needed) and reports train loss, held-out validation MSE, peak
memory and parameter counts per model.

Example:
    python examples/run_fair_attention_comparison.py --layers 12 --hidden 768 --steps 400
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from infinity import MegaSlideConfig, MegaSlideDiT
from infinity.video.baselines import Dense3DDiT, SwinDiT

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results" / "09_fair_attention"


def make_motion_dataset(n, C, T, H, W, seed):
    """Structured-motion latents: Gaussian blobs translating across frames.

    Unlike random noise, these have temporal motion for deformable offsets to
    track, and unlike the tiny prior set we generate a held-out split.
    """
    g = torch.Generator().manual_seed(seed)
    data = torch.zeros(n, C, T, H, W)
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    for i in range(n):
        cy = torch.randint(H // 4, 3 * H // 4, (1,), generator=g).item()
        cx = torch.randint(W // 4, 3 * W // 4, (1,), generator=g).item()
        vy = torch.randint(-2, 3, (1,), generator=g).item()
        vx = torch.randint(-2, 3, (1,), generator=g).item()
        sigma = float(torch.randint(2, 5, (1,), generator=g).item())
        for t in range(T):
            yy = (ys - (cy + vy * t)) ** 2
            xx = (xs - (cx + vx * t)) ** 2
            blob = torch.exp(-(yy + xx) / (2 * sigma ** 2))
            data[i, :, t] = blob.unsqueeze(0).expand(C, H, W)
    data = data + 0.05 * torch.randn(data.shape, generator=g)
    return data


def base_config(layers, hidden, heads, T, H, W):
    return MegaSlideConfig(
        frames=T, height=H, width=W, patch_size=8, in_channels=4,
        hidden_size=hidden, num_layers=layers, num_heads=heads, mlp_ratio=4.0,
        dsa_kernel_size=(3, 7, 7), batch_size=1, dtype="float32",
    )


def train_eval(model, train_data, val_data, steps, lr, device, diffusion_steps, seed):
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    g = torch.Generator().manual_seed(seed + 7)
    n = train_data.shape[0]
    losses = []
    model.train()
    for step in range(steps):
        idx = torch.randint(0, n, (1,), generator=g).item()
        clean = train_data[idx:idx + 1].to(device)
        noise = torch.randn(clean.shape, generator=g).to(device)
        ts = torch.randint(0, diffusion_steps, (1,), device=device)
        noisy = clean + (ts.float() / diffusion_steps).view(-1, 1, 1, 1, 1) * noise
        pred = model(noisy, ts)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for j in range(val_data.shape[0]):
            clean = val_data[j:j + 1].to(device)
            noise = torch.randn(clean.shape, generator=g).to(device)
            ts = torch.randint(0, diffusion_steps, (1,), device=device)
            noisy = clean + (ts.float() / diffusion_steps).view(-1, 1, 1, 1, 1) * noise
            val_losses.append(F.mse_loss(model(noisy, ts), noise).item())

    return {
        "params": sum(p.numel() for p in model.parameters()),
        "train_loss_first20": sum(losses[:20]) / min(20, len(losses)),
        "train_loss_last20": sum(losses[-20:]) / min(20, len(losses)),
        "val_mse": sum(val_losses) / len(val_losses),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0,
        "diverged": (sum(losses[-20:]) / min(20, len(losses))) > 2 * (sum(losses[:20]) / min(20, len(losses))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-val", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA; intended for the H100 NVL box.")

    C, T, H, W = 4, args.frames, args.size, args.size
    train_data = make_motion_dataset(args.n_train, C, T, H, W, args.seed)
    val_data = make_motion_dataset(args.n_val, C, T, H, W, args.seed + 100)
    cfg = base_config(args.layers, args.hidden, args.heads, T, H, W)

    results = {}

    # MegaSlide with learned offsets.
    results["megaslide_learned"] = train_eval(
        MegaSlideDiT(cfg), train_data, val_data, args.steps, args.lr, device,
        cfg.diffusion_steps, args.seed)

    # MegaSlide with frozen (zero) offsets == fixed local windows.
    m_fixed = MegaSlideDiT(cfg)
    for blk in m_fixed.blocks:
        for p in blk.attn.offset_net.parameters():
            p.requires_grad_(False)
    results["megaslide_fixed_offsets"] = train_eval(
        m_fixed, train_data, val_data, args.steps, args.lr, device,
        cfg.diffusion_steps, args.seed)

    # Dense global attention (matched depth/width).
    results["dense"] = train_eval(
        Dense3DDiT(cfg), train_data, val_data, args.steps, args.lr, device,
        cfg.diffusion_steps, args.seed)

    # Swin fixed windows (matched depth/width -- the fair baseline).
    results["swin"] = train_eval(
        SwinDiT(cfg), train_data, val_data, args.steps, args.lr, device,
        cfg.diffusion_steps, args.seed)

    report = {
        "experiment": "A3: matched-config attention comparison on structured motion",
        "config": {"layers": args.layers, "hidden": args.hidden, "heads": args.heads,
                   "frames": T, "size": H, "steps": args.steps, "lr": args.lr,
                   "n_train": args.n_train, "n_val": args.n_val},
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"fair_attention_{args.layers}L_{args.hidden}H.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    print(json.dumps({k: {"val_mse": v["val_mse"], "params": v["params"],
                          "peak_gpu_gb": v["peak_gpu_gb"], "diverged": v["diverged"]}
                      for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
