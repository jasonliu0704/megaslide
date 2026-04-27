"""End-to-end MegaSlide-DiT prototype experiment runner.

This script follows the paper-shaped workflow available in this repository:
train a CPU-master MegaSlide-DiT prototype, evaluate denoising and temporal
consistency proxy metrics, optionally condition on prompts via deterministic
text embeddings, sample short latent videos, and write a JSON report.

It does not reproduce private 105B checkpoints, official VBench, or fused CUDA
kernels. Those are outside this prototype's available artifacts.
"""

import argparse
import json
import logging
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a project dependency.
    psutil = None

from infinity import (
    CPUMasterVideoDiT,
    LatentVideoDataset,
    MegaSlideConfig,
    MegaSlideDiT,
    collate_latent_videos,
    load_megaslide_config,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


DEFAULT_PROMPTS = [
    "a person walking through a city street",
    "a camera pans across a mountain landscape",
    "a small object moves across a tabletop",
    "a slow zoom into a futuristic workstation",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a MegaSlide-DiT prototype experiment")
    parser.add_argument("--config", default="examples/configs/megaslide_paper_experiment_tiny.yaml")
    parser.add_argument("--output-dir", default="runs/megaslide_paper_experiment")
    parser.add_argument("--prompts", default=None, help="Optional text file with one prompt per line")
    parser.add_argument("--num-steps", type=int, default=None, help="Override training.num_steps")
    parser.add_argument("--eval-batches", type=int, default=None, help="Override experiment.eval_batches")
    parser.add_argument("--sample-steps", type=int, default=None, help="Override experiment.sample_steps")
    parser.add_argument("--force-cpu", action="store_true", help="Disable CUDA streaming even when CUDA is available")
    return parser.parse_args()


def load_raw_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_prompts(path: Optional[str]) -> List[str]:
    if path is None:
        return DEFAULT_PROMPTS
    prompt_path = Path(path)
    prompts = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_path}")
    return prompts


def prompt_seed(prompt: str) -> int:
    return int.from_bytes(sha256(prompt.encode("utf-8")).digest()[:8], "little") % (2**31)


def make_text_batch(
    prompts: List[str],
    batch_size: int,
    hidden_size: int,
    num_tokens: int,
    step: int,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    selected = [prompts[(step * batch_size + i) % len(prompts)] for i in range(batch_size)]
    embeddings = []
    for prompt in selected:
        generator = torch.Generator(device="cpu").manual_seed(prompt_seed(prompt))
        embeddings.append(torch.randn(num_tokens, hidden_size, generator=generator))
    text_embeds = torch.stack(embeddings).to(device=device)
    text_mask = torch.ones(batch_size, num_tokens, device=device)
    return text_embeds, text_mask, selected


def add_noise(clean: torch.Tensor, timesteps: torch.Tensor, diffusion_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(clean)
    sigma = (timesteps.to(device=clean.device).float() / max(diffusion_steps - 1, 1)).view(-1, 1, 1, 1, 1)
    return clean + sigma * noise, noise


def temporal_consistency_score(latents: torch.Tensor) -> float:
    if latents.shape[2] < 2:
        return 1.0
    diffs = latents[:, :, 1:] - latents[:, :, :-1]
    return float((1.0 / (1.0 + diffs.float().pow(2).mean().sqrt())).item())


def latent_feature_vector(latents: torch.Tensor, hidden_size: int) -> torch.Tensor:
    flat = latents.float().flatten(1)
    stats = torch.stack(
        [
            flat.mean(dim=1),
            flat.std(dim=1),
            flat.abs().mean(dim=1),
            flat.square().mean(dim=1).sqrt(),
        ],
        dim=1,
    )
    repeats = (hidden_size + stats.shape[1] - 1) // stats.shape[1]
    return stats.repeat(1, repeats)[:, :hidden_size]


def text_alignment_proxy(latents: torch.Tensor, text_embeds: torch.Tensor) -> float:
    video_vec = latent_feature_vector(latents, text_embeds.shape[-1])
    text_vec = text_embeds.float().mean(dim=1).cpu()
    score = F.cosine_similarity(video_vec, text_vec, dim=-1).mean()
    return float(score.item())


def process_memory_gb() -> float:
    if psutil is None:
        return 0.0
    return psutil.Process().memory_info().rss / 1024**3


def model_parameter_summary(model: MegaSlideDiT, config: MegaSlideConfig) -> Dict[str, float]:
    num_params = sum(param.numel() for param in model.parameters())
    dtype_bytes = torch.tensor([], dtype=config.dtype).element_size()
    return {
        "parameters": num_params,
        "working_weight_gb": num_params * dtype_bytes / 1024**3,
        "fp32_master_gb": num_params * 4 / 1024**3,
        "adam_moments_gb": num_params * 8 / 1024**3,
        "estimated_persistent_state_gb": num_params * (dtype_bytes + 12) / 1024**3,
    }


def train(
    trainer: CPUMasterVideoDiT,
    dataloader: DataLoader,
    config: MegaSlideConfig,
    prompts: List[str],
    text_tokens: int,
) -> List[Dict[str, float]]:
    data_iter = iter(dataloader)
    history = []
    total_loss = 0.0

    for step in range(config.num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        clean = batch["latents"]
        timesteps = torch.randint(0, config.diffusion_steps, (clean.shape[0],), dtype=torch.long)
        noisy, noise = add_noise(clean, timesteps, config.diffusion_steps)
        text_embeds, text_mask, _ = make_text_batch(
            prompts,
            clean.shape[0],
            config.hidden_size,
            text_tokens,
            step,
        )

        start = time.perf_counter()
        loss, timing = trainer.forward_and_backward(noisy, timesteps, noise, text_embeds=text_embeds, text_mask=text_mask)
        grad_norm = None
        if (step + 1) % config.gradient_accumulation_steps == 0:
            grad_norm = trainer.optimizer_step()

        step_time = time.perf_counter() - start
        total_loss += loss
        record = {
            "step": step + 1,
            "loss": loss,
            "avg_loss": total_loss / (step + 1),
            "step_time_s": step_time,
            "forward_s": timing["forward"],
            "backward_s": timing["backward"],
            "grad_norm": grad_norm,
            "cpu_memory_gb": process_memory_gb(),
        }
        if torch.cuda.is_available():
            record["cuda_peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        history.append(record)

        if (step + 1) % config.log_interval == 0:
            logger.info(
                "train step %s/%s loss=%.4f avg=%.4f time=%.2fs",
                step + 1,
                config.num_steps,
                loss,
                record["avg_loss"],
                step_time,
            )

    return history


@torch.no_grad()
def evaluate_denoising(
    model: MegaSlideDiT,
    dataloader: DataLoader,
    config: MegaSlideConfig,
    prompts: List[str],
    text_tokens: int,
    eval_batches: int,
    device: torch.device,
) -> Dict[str, float]:
    eval_model = model.to(device)
    eval_model.eval()
    losses = []
    consistency = []
    alignment = []

    data_iter = iter(dataloader)
    for idx in range(eval_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        clean = batch["latents"].to(device)
        timesteps = torch.randint(0, config.diffusion_steps, (clean.shape[0],), dtype=torch.long, device=device)
        noisy, noise = add_noise(clean, timesteps.cpu(), config.diffusion_steps)
        noisy = noisy.to(device)
        noise = noise.to(device)
        text_embeds, text_mask, _ = make_text_batch(
            prompts,
            clean.shape[0],
            config.hidden_size,
            text_tokens,
            idx,
            device=device,
        )
        pred = eval_model(noisy, timesteps.cpu(), text_embeds=text_embeds.cpu(), text_mask=text_mask.cpu())
        denoised = noisy - (timesteps.float() / max(config.diffusion_steps - 1, 1)).view(-1, 1, 1, 1, 1) * pred
        losses.append(float(F.mse_loss(pred.float(), noise.float()).item()))
        consistency.append(temporal_consistency_score(denoised.cpu()))
        alignment.append(text_alignment_proxy(denoised.cpu(), text_embeds.cpu()))

    eval_model.cpu()
    return {
        "denoise_mse": sum(losses) / len(losses),
        "temporal_consistency_proxy": sum(consistency) / len(consistency),
        "text_alignment_proxy": sum(alignment) / len(alignment),
    }


@torch.no_grad()
def sample_latents(
    model: MegaSlideDiT,
    config: MegaSlideConfig,
    prompts: List[str],
    text_tokens: int,
    sample_steps: int,
    num_samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    eval_model = model.to(device)
    eval_model.eval()
    latents = torch.randn(
        num_samples,
        config.in_channels,
        config.frames,
        config.height,
        config.width,
        device=device,
    )
    text_embeds, text_mask, selected_prompts = make_text_batch(
        prompts,
        num_samples,
        config.hidden_size,
        text_tokens,
        0,
        device=device,
    )

    if sample_steps <= 0:
        raise ValueError("sample_steps must be positive")
    schedule = torch.linspace(config.diffusion_steps - 1, 0, sample_steps, device=device).long()
    for idx, timestep in enumerate(schedule):
        t = timestep.repeat(num_samples)
        sigma = t.float().view(-1, 1, 1, 1, 1) / max(config.diffusion_steps - 1, 1)
        next_t = schedule[idx + 1] if idx + 1 < len(schedule) else torch.tensor(0, device=device)
        next_sigma = next_t.float().view(1, 1, 1, 1, 1) / max(config.diffusion_steps - 1, 1)
        pred_noise = eval_model(latents, t.cpu(), text_embeds=text_embeds.cpu(), text_mask=text_mask.cpu())
        pred_clean = latents - sigma * pred_noise
        latents = pred_clean + next_sigma * pred_noise

    samples = latents.cpu()
    metrics = {
        "sample_temporal_consistency_proxy": temporal_consistency_score(samples),
        "sample_text_alignment_proxy": text_alignment_proxy(samples, text_embeds.cpu()),
        "num_samples": num_samples,
        "sample_steps": sample_steps,
        "prompt_count": len(selected_prompts),
    }
    eval_model.cpu()
    return samples, metrics


def main():
    args = parse_args()
    raw = load_raw_yaml(args.config)
    exp_cfg = raw.get("experiment", {})

    config = load_megaslide_config(args.config)
    if args.num_steps is not None:
        config.num_steps = args.num_steps
    eval_batches = args.eval_batches if args.eval_batches is not None else exp_cfg.get("eval_batches", 2)
    sample_steps = args.sample_steps if args.sample_steps is not None else exp_cfg.get("sample_steps", 4)
    num_samples = exp_cfg.get("num_samples", min(2, config.batch_size))
    text_tokens = exp_cfg.get("text_tokens", 4)
    save_samples = exp_cfg.get("save_samples", True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, output_dir / "config.yaml")

    prompts = load_prompts(args.prompts or exp_cfg.get("prompts_path"))
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    dataset = LatentVideoDataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_latent_videos,
    )

    model = MegaSlideDiT(config)
    trainer = CPUMasterVideoDiT(model, config, force_cpu=args.force_cpu)
    device = torch.device(f"cuda:{config.device}") if torch.cuda.is_available() and not args.force_cpu else torch.device("cpu")

    started = datetime.now(timezone.utc).isoformat()
    param_summary = model_parameter_summary(model, config)
    logger.info("parameters=%s estimated_persistent_state_gb=%.4f", param_summary["parameters"], param_summary["estimated_persistent_state_gb"])

    train_history = train(trainer, dataloader, config, prompts, text_tokens)
    eval_metrics = evaluate_denoising(trainer.model, dataloader, config, prompts, text_tokens, eval_batches, device)
    samples, sample_metrics = sample_latents(trainer.model, config, prompts, text_tokens, sample_steps, num_samples, device)

    if save_samples:
        torch.save(
            {
                "samples": samples,
                "prompts": prompts[:num_samples],
                "shape": tuple(samples.shape),
            },
            output_dir / "samples.pt",
        )

    report = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "scope": "prototype experiment; not official VBench or 105B checkpoint reproduction",
        "config": {
            **asdict(config),
            "dtype": str(config.dtype).replace("torch.", ""),
            "dsa_kernel_size": list(config.dsa_kernel_size),
        },
        "runtime": {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cpu_memory_gb": process_memory_gb(),
            "cuda_peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
        },
        "parameter_summary": param_summary,
        "train": {
            "history": train_history,
            "final_loss": train_history[-1]["loss"] if train_history else None,
            "avg_step_time_s": sum(item["step_time_s"] for item in train_history) / len(train_history) if train_history else None,
        },
        "evaluation": {
            **eval_metrics,
            **sample_metrics,
            "metric_note": "alignment and temporal consistency are lightweight proxy metrics, not official VBench",
        },
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    trainer.cleanup()

    logger.info("wrote report: %s", report_path)
    logger.info("evaluation: %s", json.dumps(report["evaluation"], indent=2))


if __name__ == "__main__":
    main()
