#!/usr/bin/env python3
"""Train MegaSlide-DiT on latent video tensors with validation and JSON metrics.

Supports both synthetic data (smoke tests) and real VAE-encoded latents.

Usage:
    python examples/train_megaslide_dit.py --config examples/configs/megaslide_dit_280M_realvideo.yaml
    python examples/train_megaslide_dit.py --config examples/configs/megaslide_dit_tiny.yaml
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import psutil
import torch
from torch.utils.data import DataLoader

from infinity.video import (
    CPUMasterVideoDiT,
    LatentVideoDataset,
    MegaSlideDiT,
    collate_latent_videos,
    load_megaslide_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train MegaSlide-DiT")
    p.add_argument("--config", default="examples/configs/megaslide_dit_tiny.yaml")
    p.add_argument("--output-dir", default=None,
                   help="Directory for results JSON and checkpoints")
    p.add_argument("--val-path", default=None,
                   help="Path to validation latents .pt (overrides config)")
    p.add_argument("--val-interval", type=int, default=50,
                   help="Run validation every N steps")
    p.add_argument("--val-steps", type=int, default=0,
                   help="Number of val batches (0 = full pass)")
    return p.parse_args()


@torch.no_grad()
def validate(trainer, val_loader, config, max_steps=0):
    """Run validation pass and return average loss."""
    total_loss = 0.0
    n = 0
    for i, batch in enumerate(val_loader):
        if max_steps > 0 and i >= max_steps:
            break
        clean = batch["latents"]
        noise = torch.randn_like(clean)
        t = torch.randint(0, config.diffusion_steps, (clean.shape[0],), dtype=torch.long)
        sigma = (t.float() / max(config.diffusion_steps - 1, 1)).view(-1, 1, 1, 1, 1)
        noisy = clean + sigma * noise

        if trainer.use_cuda:
            noisy_gpu = noisy.to(trainer.device)
            t_gpu = t.to(trainer.device)
            noise_gpu = noise.to(trainer.device)

            B, C, T_f, H, W = noisy_gpu.shape
            x = trainer.patch_embed(noisy_gpu)
            _, D, T, H_p, W_p = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = x + trainer.pos_embed

            import inspect
            sig = inspect.signature(trainer.model._get_timestep_embedding)
            if len(sig.parameters) > 2:
                t_emb = trainer.model._get_timestep_embedding(t_gpu, D, trainer.device)
            else:
                t_emb = trainer.model._get_timestep_embedding(t_gpu)
            t_emb = trainer.time_embed(t_emb)
            x = x + t_emb.unsqueeze(1)

            video_shape = (T, H_p, W_p)
            for blk_idx in range(trainer.num_blocks):
                buf = blk_idx % 2
                trainer._load_block_to_gpu(blk_idx, buf)
                trainer.weight_stream.synchronize()
                trainer.compute_stream.wait_event(trainer.weight_ready_events[buf])
                with torch.cuda.stream(trainer.compute_stream):
                    if trainer._block_needs_video_shape:
                        x = trainer.gpu_blocks[buf](x, video_shape, None, None)
                    else:
                        x = trainer.gpu_blocks[buf](x, None, None)
            torch.cuda.current_stream().wait_stream(trainer.compute_stream)

            x = trainer.norm_out(x)
            x = trainer.out_proj(x)

            p = config.patch_size
            x = x.view(B, T, H_p, W_p, C, p, p)
            x = x.permute(0, 4, 1, 2, 5, 3, 6).contiguous()
            pred = x.view(B, C, T_f, H, W)
            loss = torch.nn.functional.mse_loss(pred, noise_gpu).item()
        else:
            pred = trainer.model(noisy, t)
            loss = torch.nn.functional.mse_loss(pred, noise).item()

        total_loss += loss
        n += 1

    return total_loss / max(n, 1)


def main():
    args = parse_args()
    config = load_megaslide_config(args.config)
    torch.manual_seed(config.seed)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"results/10_real_video"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MEGASLIDE-DIT TRAINING — PHASE 1 (REAL VIDEO)")
    logger.info("=" * 70)
    logger.info(
        "Shape: B=%s C=%s T=%s H=%s W=%s hidden=%s layers=%s",
        config.batch_size, config.in_channels, config.frames,
        config.height, config.width, config.hidden_size, config.num_layers,
    )

    # Dataset
    val_path = args.val_path
    if val_path is None and config.path:
        candidate = str(Path(config.path).parent / "val_latents.pt")
        if os.path.exists(candidate):
            val_path = candidate
            logger.info("Auto-detected val file: %s", val_path)

    dataset = LatentVideoDataset(config, val_path=val_path)
    train_loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_latent_videos, drop_last=True,
    )

    val_loader = None
    if dataset.val_dataset is not None:
        val_loader = DataLoader(
            dataset.val_dataset, batch_size=config.batch_size,
            shuffle=False, collate_fn=collate_latent_videos,
        )
        logger.info("Train: %d samples, Val: %d samples", len(dataset), len(dataset.val_dataset))
    else:
        logger.info("Train: %d samples, no validation set", len(dataset))

    # Model
    model = MegaSlideDiT(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %.2fM", n_params / 1e6)

    trainer = CPUMasterVideoDiT(model, config)

    # Metrics tracking
    process = psutil.Process()
    train_losses = []
    val_losses = []
    step_times = []
    best_val = float("inf")

    logger.info("=" * 70)
    logger.info("Starting training (%d steps)", config.num_steps)
    logger.info("=" * 70)

    data_iter = iter(train_loader)
    total_loss = 0.0
    t_start = time.time()

    for step in range(config.num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        clean = batch["latents"]
        noise = torch.randn_like(clean)
        t = torch.randint(0, config.diffusion_steps, (clean.shape[0],), dtype=torch.long)
        sigma = (t.float() / max(config.diffusion_steps - 1, 1)).view(-1, 1, 1, 1, 1)
        noisy = clean + sigma * noise

        step_t0 = time.perf_counter()
        trainer.zero_grad()
        loss_val, timing = trainer.forward_and_backward(noisy, t, noise)
        grad_norm = trainer.optimizer_step()
        step_time = time.perf_counter() - step_t0

        total_loss += loss_val
        train_losses.append(loss_val)
        step_times.append(step_time)

        if (step + 1) % config.log_interval == 0:
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            cpu_mem = process.memory_info().rss / 1024**3
            logger.info(
                "Step %d/%d | loss %.4f | avg %.4f | %.2fs/step | GPU %.1fGB | RAM %.1fGB",
                step + 1, config.num_steps, loss_val,
                total_loss / (step + 1), step_time, gpu_mem, cpu_mem,
            )

        # Validation
        if val_loader and (step + 1) % args.val_interval == 0:
            val_loss = validate(trainer, val_loader, config, max_steps=args.val_steps)
            val_losses.append({"step": step + 1, "val_loss": val_loss})
            tag = ""
            if val_loss < best_val:
                best_val = val_loss
                tag = " (best)"
            logger.info("  VAL step %d: %.4f%s", step + 1, val_loss, tag)

    elapsed = time.time() - t_start

    # Final validation
    if val_loader:
        final_val = validate(trainer, val_loader, config, max_steps=0)
        val_losses.append({"step": config.num_steps, "val_loss": final_val})
        if final_val < best_val:
            best_val = final_val
        logger.info("Final VAL: %.4f (best: %.4f)", final_val, best_val)

    # Summary
    gpu_peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    cpu_peak = process.memory_info().rss / 1024**3
    avg_step = sum(step_times) / len(step_times) if step_times else 0

    last_n = min(50, len(train_losses))
    avg_last = sum(train_losses[-last_n:]) / last_n if train_losses else 0

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info("Loss: %.4f -> %.4f (avg-last-%d: %.4f)", train_losses[0], train_losses[-1], last_n, avg_last)
    logger.info("Elapsed: %.1fs | Avg step: %.3fs", elapsed, avg_step)
    logger.info("Peak GPU: %.2f GB | Peak RAM: %.2f GB", gpu_peak, cpu_peak)
    if val_losses:
        logger.info("Best val loss: %.4f", best_val)

    # Save results JSON
    results = {
        "config": {
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "frames": config.frames,
            "height": config.height,
            "width": config.width,
            "patch_size": config.patch_size,
            "in_channels": config.in_channels,
            "num_steps": config.num_steps,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "data_path": config.path,
        },
        "params": n_params,
        "train_loss_first": train_losses[0],
        "train_loss_last": train_losses[-1],
        f"avg_last_{last_n}": avg_last,
        "best_val_loss": best_val if val_losses else None,
        "val_history": val_losses,
        "train_losses": train_losses,
        "avg_step_sec": avg_step,
        "elapsed_sec": elapsed,
        "peak_gpu_gb": gpu_peak,
        "peak_ram_gb": cpu_peak,
        "data_source": "real_video_vae_latents",
    }

    results_path = os.path.join(output_dir, "real_video_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved: %s", results_path)

    trainer.cleanup()


if __name__ == "__main__":
    main()
