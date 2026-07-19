"""VBench evaluation for MegaSlide-DiT paper experiments (Section 7).

Evaluates video generation quality using the VBench suite:
- Video-text alignment metrics
- Temporal consistency metrics
- 300 prompts from VBench dataset
- 30 diffusion timesteps for sampling

Usage:
    python examples/run_vbench_evaluation.py \
        --config examples/configs/megaslide_paper_experiment_256f.yaml \
        --checkpoint checkpoints/megaslide_5k.pt \
        --model_type megaslide \
        --num_prompts 300 \
        --output_dir vbench_results
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

# VBench and related imports (install: pip install vbench transformers diffusers)
try:
    from vbench import VBench
except ImportError:
    print("Warning: vbench not installed. Install with: pip install vbench")
    VBench = None

try:
    from transformers import CLIPTextModel, CLIPTokenizer
except ImportError:
    print("Warning: transformers not installed. Install with: pip install transformers")
    CLIPTextModel = None
    CLIPTokenizer = None

try:
    from diffusers import AutoencoderKL
except ImportError:
    print("Warning: diffusers not installed. Install with: pip install diffusers")
    AutoencoderKL = None

from infinity.video import (
    MegaSlideDiT,
    Dense3DDiT,
    SwinDiT,
    load_megaslide_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run VBench evaluation for paper experiments")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["megaslide", "dense", "swin"],
        default="megaslide",
        help="Model type to evaluate",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=300,
        help="Number of VBench prompts to evaluate (paper uses 300)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of diffusion timesteps (paper uses 30)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vbench_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Text encoder model (CLIP or T5)",
    )
    parser.add_argument(
        "--vae_model",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="VAE model for latent decoding",
    )
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Save generated videos to disk",
    )
    return parser.parse_args()


class DDPMSampler:
    """DDPM sampling for diffusion models.

    Implements the DDPM sampling algorithm from Ho et al. 2020.
    """

    def __init__(self, num_steps: int = 1000, beta_schedule: str = "linear"):
        self.num_steps = num_steps

        # Create noise schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(1e-4, 0.02, num_steps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Precompute sampling coefficients
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        num_inference_steps: int,
        text_embeds: Tensor = None,
        device: str = "cuda",
    ) -> Tensor:
        """Sample from diffusion model.

        Args:
            model: Diffusion model
            shape: Output shape (B, C, T, H, W)
            num_inference_steps: Number of denoising steps
            text_embeds: Optional text conditioning
            device: Device to run on

        Returns:
            Generated latent video
        """
        # Start from pure noise
        latent = torch.randn(shape, device=device)

        # Sampling timesteps (evenly spaced)
        timesteps = torch.linspace(
            self.num_steps - 1, 0, num_inference_steps, dtype=torch.long, device=device
        )

        # Denoising loop
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Predict noise
            with torch.no_grad():
                noise_pred = model(latent, t_batch, text_embeds=text_embeds)

            # DDPM update step
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)

            # Predict x_0
            pred_x0 = (latent - self.sqrt_one_minus_alphas_cumprod[t] * noise_pred) / self.sqrt_alphas_cumprod[t]

            # Compute x_{t-1}
            if t > 0:
                noise = torch.randn_like(latent)
                sigma_t = torch.sqrt((1.0 - alpha_t_prev) / (1.0 - alpha_t) * self.betas[t])
                latent = (
                    torch.sqrt(alpha_t_prev) * pred_x0
                    + torch.sqrt(1.0 - alpha_t_prev - sigma_t ** 2) * noise_pred
                    + sigma_t * noise
                )
            else:
                latent = pred_x0

        return latent


def encode_text(prompt: str, tokenizer, text_encoder, device: str = "cuda") -> Tensor:
    """Encode text prompt using CLIP.

    Args:
        prompt: Text prompt
        tokenizer: CLIP tokenizer
        text_encoder: CLIP text encoder
        device: Device to run on

    Returns:
        Text embeddings [1, L, D]
    """
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = tokens.input_ids.to(device)

    with torch.no_grad():
        text_embeds = text_encoder(input_ids)[0]

    return text_embeds


def decode_latents(latents: Tensor, vae: nn.Module) -> Tensor:
    """Decode latent video using VAE.

    Args:
        latents: [B, C, T, H, W] latent video
        vae: VAE decoder

    Returns:
        [B, 3, T, H*8, W*8] decoded video
    """
    B, C, T, H, W = latents.shape

    # Decode frame by frame (VAE is 2D)
    frames = []
    for t in range(T):
        frame_latent = latents[:, :, t, :, :]  # [B, C, H, W]
        with torch.no_grad():
            frame = vae.decode(frame_latent / 0.18215).sample  # [B, 3, H*8, W*8]
        frames.append(frame)

    video = torch.stack(frames, dim=2)  # [B, 3, T, H*8, W*8]

    # Normalize to [0, 1]
    video = (video + 1.0) / 2.0
    video = video.clamp(0, 1)

    return video


def main():
    args = parse_args()

    # Check dependencies
    if VBench is None:
        raise ImportError("VBench not installed. Install with: pip install vbench")
    if CLIPTextModel is None or CLIPTokenizer is None:
        raise ImportError("transformers not installed. Install with: pip install transformers")
    if AutoencoderKL is None:
        raise ImportError("diffusers not installed. Install with: pip install diffusers")

    # Load config
    config = load_megaslide_config(args.config)
    device = f"cuda:{config.device}" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("VBENCH EVALUATION")
    print("=" * 70)
    print(f"Model type: {args.model_type}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num prompts: {args.num_prompts}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Device: {device}")
    print("=" * 70)

    # Load model
    print("Loading model...")
    if args.model_type == "megaslide":
        model = MegaSlideDiT(config)
    elif args.model_type == "dense":
        model = Dense3DDiT(config)
    else:
        model = SwinDiT(config)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    print(f"✓ Loaded {args.model_type} model")

    # Load text encoder
    print("Loading text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(args.text_encoder)
    text_encoder = CLIPTextModel.from_pretrained(args.text_encoder).to(device)
    text_encoder.eval()
    print(f"✓ Loaded text encoder: {args.text_encoder}")

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(args.vae_model).to(device)
    vae.eval()
    print(f"✓ Loaded VAE: {args.vae_model}")

    # Initialize VBench
    print("Initializing VBench...")
    vbench = VBench(device=device, num_workers=8)
    prompts = vbench.get_prompts()[:args.num_prompts]
    print(f"✓ Loaded {len(prompts)} prompts")

    # Initialize DDPM sampler
    sampler = DDPMSampler(num_steps=config.diffusion_steps, beta_schedule=config.noise_schedule)

    # Generate videos
    print("\nGenerating videos...")
    videos = []
    os.makedirs(args.output_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] Generating: {prompt[:60]}...")

        # Encode text
        text_embeds = encode_text(prompt, tokenizer, text_encoder, device)

        # Sample latent video
        latent_shape = (1, config.in_channels, config.frames,
                       config.height // 8, config.width // 8)  # VAE 8x downsample
        latent = sampler.sample(
            model, latent_shape, args.num_inference_steps, text_embeds, device
        )

        # Decode to pixel space
        video = decode_latents(latent, vae)  # [1, 3, T, H, W]
        videos.append(video.cpu())

        # Optionally save video
        if args.save_videos:
            video_path = Path(args.output_dir) / f"video_{i:04d}.pt"
            torch.save({"video": video.cpu(), "prompt": prompt}, video_path)

    print(f"✓ Generated {len(videos)} videos")

    # Evaluate with VBench
    print("\nEvaluating with VBench...")
    results = vbench.evaluate(videos, prompts)

    # Print results
    print("\n" + "=" * 70)
    print("VBENCH RESULTS")
    print("=" * 70)
    print(f"VBench-Align:   {results['alignment']:.2f} ± {results['alignment_ci']:.2f}")
    print(f"VBench-Consist: {results['consistency']:.2f} ± {results['consistency_ci']:.2f}")
    print("=" * 70)

    # Save results
    results_path = Path(args.output_dir) / f"{args.model_type}_vbench_results.pt"
    torch.save(results, results_path)
    print(f"\n✓ Results saved to: {results_path}")

    # NOTE: This is an optional, out-of-scope harness. The paper does NOT report
    # official VBench scores (no pre-trained checkpoint or real-video data), so we
    # deliberately do not print any "expected" reference numbers here.


if __name__ == "__main__":
    main()
