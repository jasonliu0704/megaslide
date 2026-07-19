"""Hybrid Attention Experiment Runner.

Trains and evaluates baseline (pure 3D-DSA), register, and anchor variants,
measuring loss, temporal consistency (SRC/LRC/TCS), step time, and memory.

Usage:
    # Single variant
    PYTHONPATH=. python examples/run_hybrid_attention_experiment.py \
        --config examples/configs/hybrid_register_64_256f.yaml --variant register

    # All variants comparison
    PYTHONPATH=. python examples/run_hybrid_attention_experiment.py \
        --config examples/configs/hybrid_register_64_256f.yaml --all

    # Tiny smoke test
    PYTHONPATH=. python examples/run_hybrid_attention_experiment.py \
        --config examples/configs/hybrid_register_64_tiny.yaml --variant register --num-steps 3
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from infinity.video.config import MegaSlideConfig
from infinity.video.model import MegaSlideDiT
from infinity.video.hybrid_model import HybridMegaSlideDiT, TemporalAnchorMegaSlideDiT
from infinity.video.metrics import compute_temporal_consistency, compute_offset_magnitude


def load_config(path: str) -> MegaSlideConfig:
    """Load config from YAML, mapping nested sections to flat MegaSlideConfig fields."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    flat = {}
    for section in raw.values():
        if isinstance(section, dict):
            flat.update(section)
        else:
            pass  # skip non-dict top-level values

    # Convert list to tuple for kernel size
    if "dsa_kernel_size" in flat and isinstance(flat["dsa_kernel_size"], list):
        flat["dsa_kernel_size"] = tuple(flat["dsa_kernel_size"])

    # Filter to only valid MegaSlideConfig fields
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(MegaSlideConfig)}
    flat = {k: v for k, v in flat.items() if k in valid_fields}

    return MegaSlideConfig(**flat)


def create_model(config: MegaSlideConfig, variant: str):
    """Create model based on variant name."""
    if variant == "baseline":
        return MegaSlideDiT(config)
    elif variant == "register":
        return HybridMegaSlideDiT(config)
    elif variant == "anchor":
        return TemporalAnchorMegaSlideDiT(config)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def generate_motion_data(config: MegaSlideConfig, num_samples: int) -> torch.Tensor:
    """Generate motion data requiring long-range temporal reasoning.

    Creates patterns where distant frames are correlated in ways that
    local-only attention cannot capture:
    - Recurring spatial patterns (object appears, disappears, reappears)
    - Global motion trajectories spanning many frames
    - Periodic signals with period > local temporal kernel
    """
    C, T, H, W = config.in_channels, config.frames, config.height, config.width
    data = []
    for i in range(num_samples):
        sample = torch.zeros(1, C, T, H, W)

        # 1. Background: slow global oscillation (period = T frames)
        t_phase = torch.linspace(0, 2 * 3.14159, T).view(1, 1, T, 1, 1)
        bg = torch.randn(1, C, 1, H, W) * 0.3
        sample = sample + bg * torch.sin(t_phase + i * 0.7)

        # 2. Recurring object: a spatial blob that appears at frames 0, T//4, T//2, 3T//4
        blob_y, blob_x = torch.randint(2, H - 2, (1,)).item(), torch.randint(2, W - 2, (1,)).item()
        blob_pattern = torch.randn(C, 1, 1) * 2.0
        for anchor_t in [0, T // 4, T // 2, 3 * T // 4]:
            for dt in range(min(2, T - anchor_t)):
                sample[0, :, anchor_t + dt, blob_y-1:blob_y+2, blob_x-1:blob_x+2] += blob_pattern

        # 3. Trajectory: a feature moves diagonally across frames
        speed_y = (H - 4) / max(T - 1, 1)
        speed_x = (W - 4) / max(T - 1, 1)
        traj_val = torch.randn(C, 1, 1) * 1.5
        for t in range(T):
            ty = int(1 + speed_y * t)
            tx = int(1 + speed_x * t)
            ty = min(ty, H - 2)
            tx = min(tx, W - 2)
            sample[0, :, t, ty:ty+2, tx:tx+2] += traj_val

        # 4. Add noise
        sample = sample + torch.randn_like(sample) * 0.2
        data.append(sample)
    return torch.cat(data, dim=0)


def run_experiment(config: MegaSlideConfig, variant: str, num_steps: int, eval_interval: int = 50):
    """Run a single experiment variant.

    Returns dict with loss_curve, step_times, tcs_history, final metrics.
    """
    torch.manual_seed(config.seed)
    model = create_model(config, variant)
    model.train()

    if torch.cuda.is_available() and config.device is not None:
        device = torch.device(f"cuda:{config.device}")
        model = model.to(device)
        print(f"  [{variant}] using device={device}")
    else:
        device = torch.device("cpu")
        print(f"  [{variant}] using device=cpu (no CUDA)")

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )

    # Generate motion data (CPU), keep there; we'll move per-batch to device.
    data = generate_motion_data(config, max(config.synthetic_samples, config.batch_size * 2))

    T_p = config.frames
    H_p = config.height // config.patch_size
    W_p = config.width // config.patch_size
    video_shape = (T_p, H_p, W_p)

    loss_curve = []
    step_times = []
    tcs_history = []

    use_amp = device.type == "cuda" and str(getattr(config, "dtype", "float32")).lower() in {"bfloat16", "bf16", "float16", "fp16"}
    amp_dtype = torch.bfloat16 if str(getattr(config, "dtype", "")).lower() in {"bfloat16", "bf16"} else torch.float16

    for step in range(num_steps):
        # Sample batch
        idx = torch.randint(0, len(data), (config.batch_size,))
        clean = data[idx].to(device, non_blocking=True)
        noise = torch.randn_like(clean)
        timesteps = torch.randint(0, config.diffusion_steps, (config.batch_size,), device=device)
        sigma = (timesteps.float() / max(config.diffusion_steps - 1, 1)).view(-1, 1, 1, 1, 1)
        noisy = clean + sigma * noise

        # Forward + backward
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_t = time.perf_counter()
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                pred = model(noisy, timesteps)
                loss = F.mse_loss(pred, noise)
        else:
            pred = model(noisy, timesteps)
            loss = F.mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_time = time.perf_counter() - start_t

        loss_curve.append(loss.item())
        step_times.append(step_time)

        # Evaluate TCS periodically
        if (step + 1) % eval_interval == 0 or step == num_steps - 1:
            model.eval()
            eval_batch = data[:min(2, len(data))].to(device, non_blocking=True)
            eval_ts = torch.randint(0, config.diffusion_steps, (eval_batch.shape[0],), device=device)
            tcs = compute_temporal_consistency(model, eval_batch, eval_ts, video_shape)
            tcs_history.append({"step": step + 1, **tcs})
            model.train()

        if (step + 1) % max(num_steps // 100, 10) == 0 or step == 0:
            print(f"  [{variant}] step {step+1}/{num_steps} loss={loss.item():.4f} time={step_time:.3f}s", flush=True)
        # Milestone marker every 10% (separate, easy-to-scan log line)
        if (step + 1) % max(num_steps // 10, 1) == 0:
            pct = round(100 * (step + 1) / num_steps)
            print(f"  [{variant}] MILESTONE {pct}%  step={step+1}/{num_steps}  loss={loss.item():.4f}  avg_step_time={sum(step_times[-50:])/min(len(step_times),50):.3f}s", flush=True)

    # Final metrics
    offset_mag = compute_offset_magnitude(model)

    return {
        "variant": variant,
        "num_params": num_params,
        "trainable_params": trainable_params,
        "loss_curve": loss_curve,
        "final_loss": loss_curve[-1] if loss_curve else None,
        "avg_step_time": sum(step_times[1:]) / max(len(step_times) - 1, 1),
        "tcs_history": tcs_history,
        "final_tcs": tcs_history[-1] if tcs_history else None,
        "offset_magnitude": offset_mag,
        "num_steps": num_steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Hybrid Attention Experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--variant", choices=["baseline", "register", "anchor"], default="register")
    parser.add_argument("--num-steps", type=int, default=None, help="Override num_steps from config")
    parser.add_argument("--output-dir", default="results/07_hybrid_attention")
    parser.add_argument("--all", action="store_true", help="Run all variants")
    args = parser.parse_args()

    config = load_config(args.config)
    num_steps = args.num_steps or config.num_steps
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = ["baseline", "register", "anchor"] if args.all else [args.variant]
    eval_interval = max(num_steps // 4, 1)

    all_results = {}
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Running variant: {variant}")
        print(f"{'='*60}")
        result = run_experiment(config, variant, num_steps, eval_interval)
        all_results[variant] = result

        # Save individual result
        out_path = output_dir / f"{variant}_results.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}")

    # Print comparison
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Variant':<12} {'Params':<10} {'Final Loss':<12} {'Avg Step(s)':<12} {'SRC':<6} {'LRC':<6} {'TCS':<6}")
        print("-" * 70)
        for name, r in all_results.items():
            tcs = r["final_tcs"] or {}
            print(f"{name:<12} {r['num_params']:<10} {r['final_loss']:<12.4f} "
                  f"{r['avg_step_time']:<12.4f} {tcs.get('src', 0):<6.3f} "
                  f"{tcs.get('lrc', 0):<6.3f} {tcs.get('tcs', 0):<6.3f}")

    # Save combined results
    combined_path = output_dir / "combined_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results: {combined_path}")


if __name__ == "__main__":
    main()
