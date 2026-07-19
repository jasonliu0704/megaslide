"""A2: Validate CPU-resident AdamW vs SGD convergence under weight streaming.

Prior max-scale runs used plain SGD because AdamW state for 28-33B exceeds the
314 GB host RAM. AdamW state is ~12 bytes/param (fp32 master + m + v) plus a
transient gradient copy, so a ~12.6B model (~150-200 GB of state) fits and lets
us validate the CPU-AdamW path the paper's design depends on.

This script must run on the H100 NVL box (CUDA + ~314 GB RAM). It builds a
MegaSlide-DiT at the requested scale, trains it twice from an identical
initialization -- once with CPU-AdamW (the trainer default) and once with SGD --
through the CPU-master streaming trainer, and records the loss curves, step
times, and host/device memory.

Example (on the GPU box):
    python examples/run_adamw_validation.py --scale 12.6b --steps 300
    python examples/run_adamw_validation.py --scale 7b --steps 300   # lighter fallback
"""

import argparse
import json
import time
from pathlib import Path

import torch

from infinity import CPUMasterVideoDiT, MegaSlideConfig, MegaSlideDiT

try:
    import psutil
except ImportError:
    psutil = None

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results" / "08_adamw_validation"

# (hidden, layers, heads) presets chosen to land near the named parameter count
# at frames=16, 64x64 latents, patch 8 -> 1024 tokens (matches the 12.6B/16F run).
SCALES = {
    "1b": (2048, 12, 32),
    "7b": (3584, 40, 28),
    "12.6b": (4096, 48, 32),
}


def build_config(scale: str, lr: float) -> MegaSlideConfig:
    hidden, layers, heads = SCALES[scale]
    return MegaSlideConfig(
        frames=16, height=64, width=64, patch_size=8, in_channels=4,
        hidden_size=hidden, num_layers=layers, num_heads=heads, mlp_ratio=4.0,
        dsa_kernel_size=(3, 7, 7), batch_size=1, learning_rate=lr,
        checkpoint_interval=8, num_grad_slabs=4, dtype="float32",
    )


def host_mem_gb() -> float:
    return psutil.virtual_memory().used / 1024 ** 3 if psutil else 0.0


def run_arm(optimizer_name: str, scale: str, steps: int, lr: float, seed: int):
    torch.manual_seed(seed)
    config = build_config(scale, lr)
    model = MegaSlideDiT(config)
    num_params = sum(p.numel() for p in model.parameters())
    trainer = CPUMasterVideoDiT(model, config)

    # Swap optimizer for the SGD arm; AdamW is the trainer default.
    if optimizer_name == "sgd":
        trainer.optimizer = torch.optim.SGD(trainer.get_parameters(), lr=lr, momentum=0.9)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    C, T, H, W = config.in_channels, config.frames, config.height, config.width
    gen = torch.Generator().manual_seed(seed + 1)
    losses, step_times = [], []
    for step in range(steps):
        clean = torch.randn(1, C, T, H, W, generator=gen)
        noise = torch.randn(1, C, T, H, W, generator=gen)
        ts = torch.randint(0, config.diffusion_steps, (1,))
        noisy = clean + (ts.float() / config.diffusion_steps).view(-1, 1, 1, 1, 1) * noise

        t0 = time.perf_counter()
        trainer.zero_grad()
        loss, _ = trainer.forward_and_backward(noisy, ts, noise)
        trainer.optimizer_step()
        step_times.append(time.perf_counter() - t0)
        losses.append(loss)
        if (step + 1) % 10 == 0:
            print(f"[{optimizer_name}] step {step+1}/{steps} loss={loss:.4f} "
                  f"t={step_times[-1]:.2f}s ram={host_mem_gb():.1f}GB")

    result = {
        "optimizer": optimizer_name,
        "scale": scale,
        "params": num_params,
        "lr": lr,
        "steps": steps,
        "losses": losses,
        "avg_step_time_s": sum(step_times) / len(step_times),
        "loss_first10": sum(losses[:10]) / min(10, len(losses)),
        "loss_last10": sum(losses[-10:]) / min(10, len(losses)),
        "peak_host_ram_gb": host_mem_gb(),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0,
    }
    result["improvement_pct"] = (
        (result["loss_first10"] - result["loss_last10"]) / result["loss_first10"] * 100.0
        if result["loss_first10"] else 0.0
    )
    trainer.cleanup()
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=list(SCALES), default="12.6b")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--adamw-lr", type=float, default=1e-4)
    ap.add_argument("--sgd-lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: no CUDA detected. This experiment is intended for the "
              "H100 NVL box; CPU execution will be extremely slow.")

    adamw = run_arm("adamw", args.scale, args.steps, args.adamw_lr, args.seed)
    sgd = run_arm("sgd", args.scale, args.steps, args.sgd_lr, args.seed)

    report = {
        "experiment": "A2: CPU-AdamW vs SGD under weight streaming",
        "scale": args.scale,
        "params_b": adamw["params"] / 1e9,
        "adamw": adamw,
        "sgd": sgd,
        "summary": {
            "adamw_final_loss": adamw["loss_last10"],
            "sgd_final_loss": sgd["loss_last10"],
            "adamw_improvement_pct": adamw["improvement_pct"],
            "sgd_improvement_pct": sgd["improvement_pct"],
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"adamw_vs_sgd_{args.scale}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
