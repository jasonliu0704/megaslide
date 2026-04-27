"""Train the MegaSlide-DiT prototype on latent video tensors."""

import argparse
import logging
import time

import torch
from torch.utils.data import DataLoader

from infinity import (
    CPUMasterVideoDiT,
    LatentVideoDataset,
    MegaSlideDiT,
    collate_latent_videos,
    load_megaslide_config,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MegaSlide-DiT with CPU-backed parameters")
    parser.add_argument("--config", default="examples/configs/megaslide_dit_tiny.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_megaslide_config(args.config)
    torch.manual_seed(config.seed)

    logger.info("=" * 70)
    logger.info("MEGASLIDE-DIT: CPU-MASTER VIDEO DIFFUSION PROTOTYPE")
    logger.info("=" * 70)
    logger.info(
        "Shape: B=%s C=%s T=%s H=%s W=%s hidden=%s layers=%s",
        config.batch_size,
        config.in_channels,
        config.frames,
        config.height,
        config.width,
        config.hidden_size,
        config.num_layers,
    )

    dataset = LatentVideoDataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_latent_videos,
    )
    data_iter = iter(dataloader)

    model = MegaSlideDiT(config)
    trainer = CPUMasterVideoDiT(model, config)

    total_loss = 0.0
    for step in range(config.num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        clean = batch["latents"]
        noise = torch.randn_like(clean)
        timesteps = torch.randint(0, config.diffusion_steps, (clean.shape[0],), dtype=torch.long)
        sigma = (timesteps.float() / max(config.diffusion_steps - 1, 1)).view(-1, 1, 1, 1, 1)
        noisy = clean + sigma * noise

        start = time.perf_counter()
        loss, timing = trainer.forward_and_backward(noisy, timesteps, noise)

        grad_norm = None
        if (step + 1) % config.gradient_accumulation_steps == 0:
            grad_norm = trainer.optimizer_step()

        total_loss += loss
        if (step + 1) % config.log_interval == 0:
            logger.info(
                "Step %s/%s | loss %.4f | avg %.4f | step %.2fs | fwd %.2fs | bwd %.2fs%s",
                step + 1,
                config.num_steps,
                loss,
                total_loss / (step + 1),
                time.perf_counter() - start,
                timing["forward"],
                timing["backward"],
                f" | grad_norm {grad_norm:.4f}" if grad_norm is not None else "",
            )

    trainer.cleanup()
    logger.info("Training complete")


if __name__ == "__main__":
    main()
