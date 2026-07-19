#!/usr/bin/env python3
"""Prepare real-video latent dataset from publicly available videos.

Downloads videos from HuggingFace, extracts fixed-length clips,
resizes, encodes through a pretrained SD VAE, and saves train/val
latent tensors as .pt files.

Usage:
    python scripts/prepare_real_video_latents.py \
        --output-dir data/real_video_latents \
        --num-frames 16 --resolution 256 --val-frac 0.15
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download


def parse_args():
    p = argparse.ArgumentParser(description="Encode real videos into VAE latents")
    p.add_argument("--repo-id", default="Wild-Heart/Disney-VideoGeneration-Dataset")
    p.add_argument("--output-dir", default="data/real_video_latents")
    p.add_argument("--video-cache", default="data/disney_videos")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--clips-per-video", type=int, default=3,
                   help="Number of non-overlapping clips to extract per video")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Frames per VAE encode batch")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def download_videos(repo_id: str, cache_dir: str) -> list[str]:
    """Download all .mp4 files from a HuggingFace dataset repo."""
    api = HfApi()
    all_files = api.list_repo_files(repo_id, repo_type="dataset")
    video_files = [f for f in all_files if f.endswith(".mp4")]
    print(f"Found {len(video_files)} videos in {repo_id}")

    local_paths = []
    for vf in video_files:
        local_path = os.path.join(cache_dir, vf)
        if os.path.exists(local_path):
            local_paths.append(local_path)
            continue
        try:
            path = hf_hub_download(
                repo_id=repo_id, filename=vf,
                repo_type="dataset", local_dir=cache_dir,
            )
            local_paths.append(path)
            print(f"  Downloaded {vf} ({os.path.getsize(path)/1e6:.1f} MB)")
        except Exception as e:
            print(f"  SKIP {vf}: {e}")
    print(f"Total local videos: {len(local_paths)}")
    return sorted(local_paths)


def extract_clips(video_path: str, num_frames: int, resolution: int,
                  clips_per_video: int) -> list[np.ndarray]:
    """Extract fixed-length clips from a video, uniformly spaced.

    Returns list of arrays, each [num_frames, H, W, 3] in uint8 RGB.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        cap.release()
        return []

    clips = []
    max_clips = min(clips_per_video, total_frames // num_frames)

    for clip_idx in range(max_clips):
        start = clip_idx * (total_frames // max_clips)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resolution, resolution),
                               interpolation=cv2.INTER_LANCZOS4)
            frames.append(frame)

        if len(frames) == num_frames:
            clips.append(np.stack(frames))

    cap.release()
    return clips


@torch.no_grad()
def encode_clips_to_latents(
    clips: list[np.ndarray], vae, device: torch.device, batch_size: int = 4,
) -> list[torch.Tensor]:
    """Encode [T, H, W, 3] uint8 clips to VAE latents [C, T, H/8, W/8].

    Encodes frame-by-frame (SD VAE is 2D), then stacks along temporal dim.
    """
    latent_clips = []
    for clip in clips:
        T = clip.shape[0]
        # [T, H, W, 3] -> [T, 3, H, W], float32 in [-1, 1]
        frames_t = torch.from_numpy(clip).permute(0, 3, 1, 2).float() / 127.5 - 1.0

        frame_latents = []
        for i in range(0, T, batch_size):
            batch = frames_t[i : i + batch_size].to(device)
            posterior = vae.encode(batch)
            z = posterior.latent_dist.sample()
            z = z * vae.config.scaling_factor
            frame_latents.append(z.cpu())

        # [T, C, H/8, W/8] -> [C, T, H/8, W/8]
        stacked = torch.cat(frame_latents, dim=0)  # [T, C, h, w]
        stacked = stacked.permute(1, 0, 2, 3)  # [C, T, h, w]
        latent_clips.append(stacked)

    return latent_clips


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.video_cache, exist_ok=True)

    t0 = time.time()

    # 1. Download videos
    print("=" * 60)
    print("Step 1: Downloading videos")
    print("=" * 60)
    video_paths = download_videos(args.repo_id, args.video_cache)
    if not video_paths:
        raise RuntimeError("No videos downloaded")

    # 2. Extract clips
    print("\n" + "=" * 60)
    print("Step 2: Extracting clips")
    print("=" * 60)
    all_clips = []
    for vp in video_paths:
        clips = extract_clips(vp, args.num_frames, args.resolution,
                              args.clips_per_video)
        if clips:
            print(f"  {Path(vp).name}: {len(clips)} clips")
            all_clips.extend(clips)
    print(f"Total clips: {len(all_clips)}")

    if len(all_clips) < 5:
        raise RuntimeError(f"Too few clips ({len(all_clips)}), need at least 5")

    # 3. Load VAE
    print("\n" + "=" * 60)
    print("Step 3: Loading VAE")
    print("=" * 60)
    from diffusers import AutoencoderKL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained(args.vae_model, torch_dtype=torch.float32)
    vae = vae.to(device).eval()
    print(f"VAE loaded on {device}")

    # 4. Encode to latents
    print("\n" + "=" * 60)
    print("Step 4: Encoding clips to latents")
    print("=" * 60)
    latent_list = encode_clips_to_latents(
        all_clips, vae, device, batch_size=args.batch_size,
    )
    print(f"Encoded {len(latent_list)} clips, shape per clip: {latent_list[0].shape}")

    # Free GPU memory
    del vae
    torch.cuda.empty_cache()

    # 5. Train/val split
    print("\n" + "=" * 60)
    print("Step 5: Splitting train/val")
    print("=" * 60)
    indices = np.random.permutation(len(latent_list))
    n_val = max(1, int(len(latent_list) * args.val_frac))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_latents = torch.stack([latent_list[i] for i in train_idx])
    val_latents = torch.stack([latent_list[i] for i in val_idx])

    print(f"Train: {train_latents.shape}, Val: {val_latents.shape}")

    train_path = os.path.join(args.output_dir, "train_latents.pt")
    val_path = os.path.join(args.output_dir, "val_latents.pt")
    torch.save(train_latents, train_path)
    torch.save(val_latents, val_path)
    print(f"Saved: {train_path} ({os.path.getsize(train_path)/1e6:.1f} MB)")
    print(f"Saved: {val_path} ({os.path.getsize(val_path)/1e6:.1f} MB)")

    # 6. Save metadata
    meta = {
        "source": args.repo_id,
        "num_frames": args.num_frames,
        "resolution": args.resolution,
        "clips_per_video": args.clips_per_video,
        "vae_model": args.vae_model,
        "total_clips": len(latent_list),
        "train_clips": len(train_idx),
        "val_clips": len(val_idx),
        "latent_shape": list(latent_list[0].shape),
        "train_tensor_shape": list(train_latents.shape),
        "val_tensor_shape": list(val_latents.shape),
        "elapsed_sec": time.time() - t0,
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata: {meta_path}")
    print(f"Total time: {meta['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
