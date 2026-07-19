"""Phase 2 baseline: PyTorch FSDP + CPU offload (single GPU) at multiple scales.

This script runs the standard CPU-offload baseline using PyTorch FSDP with
CPUOffload(offload_params=True), on the same Dense3DDiT architecture, and
records the same metrics as MegaSlide §9.4 (peak GPU mem, peak host RAM,
step time, OOM-or-not). The comparison against MegaSlide is made by reading
both JSON sets side-by-side in the paper (no need to re-run MegaSlide here).

Why FSDP-CPU and not DeepSpeed Stage-3? Both are listed as comparators in
Section 2.4. FSDP-CPU is the official PyTorch path and ships with torch.
DeepSpeed Stage-3 with CPU offload requires nvcc (not in this environment),
and the *quantitative claim* — whether a standard CPU-offload trainer fits the
same model as MegaSlide — is identical regardless of which standard offload
system we compare against. The paper notes both equivalences explicitly.

Usage:
    PYTHONPATH=. python examples/run_offload_comparison.py --scale small --num-steps 10
    PYTHONPATH=. python examples/run_offload_comparison.py --scale medium --num-steps 10
"""

import argparse
import functools
import gc
import json
import os
import socket
import time
from pathlib import Path
from typing import Dict

import psutil
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from infinity.video.config import MegaSlideConfig
from infinity.video.baselines import Dense3DDiT, Dense3DBlock


class Dense3DDiTBF16(Dense3DDiT):
    """Dense3DDiT with timestep embedding cast to model dtype.

    Needed because FSDP MixedPrecision converts params to bf16 while
    Dense3DDiT._get_timestep_embedding returns fp32 (calls .float()).
    """

    def forward(self, latents, timesteps, text_embeds=None, text_mask=None):
        B, C, T, H, W = latents.shape
        x = self.patch_embed(latents)
        _, D, T, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        t_emb = self.time_embed(self._get_timestep_embedding(timesteps).to(x.dtype))
        x = x + t_emb.unsqueeze(1)
        for block in self.blocks:
            x = block(x, text_embeds, text_mask)
        x = self.norm_out(x)
        x = self.out_proj(x)
        x = x.view(B, T, H_p, W_p, C, self.config.patch_size, self.config.patch_size)
        x = x.permute(0, 4, 1, 2, 5, 3, 6).contiguous()
        x = x.view(B, C, T, H, W)
        return x


SCALES: Dict[str, Dict] = {
    # Matched to MegaSlide §9.4 measurement ladder (frames=16, 64x64 latents,
    # patch=8 -> 1024 tokens). hidden, layers, heads chosen to land near the
    # named parameter count for Dense3DDiT (same shape as MegaSlideDiT minus
    # the deformable offsets, so ~equal flops per layer).
    "smoke":  dict(num_layers=12, hidden_size=1024, num_heads=16),  # ~160M
    "1b":     dict(num_layers=12, hidden_size=2048, num_heads=32),  # ~0.9B
    "3b":     dict(num_layers=24, hidden_size=2560, num_heads=20),  # ~2.4B
    "7b":     dict(num_layers=40, hidden_size=3584, num_heads=28),  # ~7B (Dense3DDiT ~6.3B)
    "12.6b":  dict(num_layers=48, hidden_size=4096, num_heads=32),  # ~12.6B (Dense3DDiT ~8B)
    # Scales designed to land near MegaSlide's named scales for Dense3DDiT
    # (which has fewer params per (L, H) than MegaSlideDiT). Calibrated from
    # ratio measured at 12L/2048H: MegaSlide ~1B vs Dense3DDiT 641M (0.64x).
    "19.7b":  dict(num_layers=64, hidden_size=4608, num_heads=32),  # ~14B Dense3DDiT
    "28.4b":  dict(num_layers=80, hidden_size=5120, num_heads=32),  # ~20B Dense3DDiT
    "33.3b":  dict(num_layers=96, hidden_size=5120, num_heads=32),  # ~24B Dense3DDiT
}


def build_config(scale_name: str, batch_size: int, frames: int) -> MegaSlideConfig:
    s = SCALES[scale_name]
    return MegaSlideConfig(
        in_channels=4, height=64, width=64, patch_size=8,
        frames=frames,
        mlp_ratio=4.0,
        dsa_kernel_size=(3, 7, 7),
        dtype="bfloat16", device=0, attention_mode="dsa",
        batch_size=batch_size, num_steps=20,
        learning_rate=1.0e-4, weight_decay=0.01, max_grad_norm=1.0,
        seed=42, diffusion_steps=1000, synthetic_samples=4,
        checkpoint_interval=2, num_grad_slabs=4,
        **s,
    )


def free_port() -> int:
    s = socket.socket(); s.bind(("", 0))
    p = s.getsockname()[1]; s.close(); return p


def _peak_host_ram_gb() -> float:
    me = psutil.Process(os.getpid())
    rss = me.memory_info().rss
    for ch in me.children(recursive=True):
        try:
            rss += ch.memory_info().rss
        except Exception:
            pass
    return rss / (1024 ** 3)


def _setup_dist() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(free_port()))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        # Use gloo backend: world_size=1 means no real communication, and
        # this environment's NCCL/NVML pair is mismatched. Gloo (CPU
        # collective) is the safe single-GPU fallback.
        dist.init_process_group("gloo", rank=0, world_size=1)


def _teardown_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def run(config: MegaSlideConfig, num_steps: int) -> Dict:
    _setup_dist()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(config.seed)

    print(f"  [fsdp_cpu] building Dense3DDiTBF16 (layers={config.num_layers} hidden={config.hidden_size})", flush=True)
    model = Dense3DDiTBF16(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [fsdp_cpu] params: {n_params/1e6:.1f}M", flush=True)

    fsdp_model = FullyShardedDataParallel(
        model,
        cpu_offload=CPUOffload(offload_params=True),
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Dense3DBlock},
        ),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=0,
    )
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )

    B, C = config.batch_size, config.in_channels
    T, H, W = config.frames, config.height, config.width
    losses, step_times, host_ram = [], [], []

    for step in range(num_steps):
        clean = torch.randn(B, C, T, H, W, device=device, dtype=torch.bfloat16)
        noise = torch.randn_like(clean)
        timesteps = torch.randint(0, config.diffusion_steps, (B,), device=device)
        sigma = (timesteps.float() / max(config.diffusion_steps - 1, 1)).view(-1, 1, 1, 1, 1).to(torch.bfloat16)
        noisy = clean + sigma * noise

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = fsdp_model(noisy, timesteps)
        loss = F.mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        st = time.perf_counter() - t0

        losses.append(loss.item())
        step_times.append(st)
        host_ram.append(_peak_host_ram_gb())
        print(f"  [fsdp_cpu] step {step+1}/{num_steps} loss={loss.item():.4f} time={st:.2f}s host_ram={host_ram[-1]:.1f}GB peak_gpu={torch.cuda.max_memory_allocated(device)/(1024**3):.1f}GB", flush=True)

    result = {
        "system": "fsdp_cpu",
        "n_params": n_params,
        "losses": losses,
        "step_times": step_times,
        "avg_step_time": (sum(step_times[1:]) / max(len(step_times) - 1, 1)) if len(step_times) > 1 else step_times[0] if step_times else None,
        "peak_gpu_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "peak_host_ram_gb": max(host_ram) if host_ram else 0.0,
        "completed_steps": len(losses),
        "oom": False,
    }

    _teardown_dist()
    return result


def main():
    p = argparse.ArgumentParser(description="FSDP-CPU offload baseline at multiple scales")
    p.add_argument("--scale", choices=list(SCALES.keys()), default="smoke")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--output-dir", default="results/10_deepspeed")
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"fsdp_cpu_{args.scale}.json"

    config = build_config(args.scale, batch_size=args.batch_size, frames=args.frames)

    print(f"\n=== fsdp_cpu @ scale={args.scale} (steps={args.num_steps}) ===", flush=True)
    print(f"  config: layers={config.num_layers} hidden={config.hidden_size} heads={config.num_heads} frames={config.frames} batch={config.batch_size}", flush=True)

    try:
        result = run(config, args.num_steps)
    except torch.cuda.OutOfMemoryError as e:
        print(f"  [fsdp_cpu] CUDA OOM at scale={args.scale}: {str(e)[:200]}", flush=True)
        try:
            peak_gpu = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        except Exception:
            peak_gpu = 0.0
        result = {
            "system": "fsdp_cpu", "n_params": None,
            "losses": [], "step_times": [], "avg_step_time": None,
            "peak_gpu_gb": peak_gpu, "peak_host_ram_gb": _peak_host_ram_gb(),
            "completed_steps": 0, "oom": True, "oom_message": str(e)[:300],
        }
        _teardown_dist()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  [fsdp_cpu] RuntimeError OOM at scale={args.scale}: {str(e)[:200]}", flush=True)
            try:
                peak_gpu = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
            except Exception:
                peak_gpu = 0.0
            result = {
                "system": "fsdp_cpu", "n_params": None,
                "losses": [], "step_times": [], "avg_step_time": None,
                "peak_gpu_gb": peak_gpu, "peak_host_ram_gb": _peak_host_ram_gb(),
                "completed_steps": 0, "oom": True, "oom_message": str(e)[:300],
            }
            _teardown_dist()
        else:
            _teardown_dist()
            raise

    result["scale"] = args.scale
    result["config"] = {
        "num_layers": config.num_layers, "hidden_size": config.hidden_size,
        "num_heads": config.num_heads, "frames": config.frames,
        "height": config.height, "width": config.width, "patch_size": config.patch_size,
        "batch_size": config.batch_size,
    }
    result["host_total_ram_gb"] = psutil.virtual_memory().total / (1024 ** 3)
    result["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    result["gpu_name"] = torch.cuda.get_device_name(0)
    result["torch_version"] = torch.__version__

    out_path.write_text(json.dumps(result, indent=2))
    print(f"  saved: {out_path}", flush=True)
    print(f"  summary: params={result['n_params']} peak_gpu={result['peak_gpu_gb']:.1f}GB peak_host_ram={result['peak_host_ram_gb']:.1f}GB avg_step={result['avg_step_time']} oom={result['oom']}", flush=True)


if __name__ == "__main__":
    main()
