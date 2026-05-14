"""Dataset for loading latent video tensors."""

import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .config import MegaSlideConfig


class LatentVideoDataset(Dataset):
    """Dataset for pre-computed latent video tensors.

    Supports:
    - Loading from .pt or .npy files
    - Auto-detection of CTHW vs TCHW layout
    - Synthetic random latents for smoke tests

    Args:
        config: MegaSlideConfig instance.
        path: Optional path to data file (overrides config.path).
    """

    def __init__(self, config: MegaSlideConfig, path: Optional[str] = None):
        super().__init__()
        self.config = config
        self.path = path or config.path

        if self.path and os.path.exists(self.path):
            # Load real latent file
            self._load_data()
        else:
            # Generate synthetic latents
            print(f"No dataset path provided, generating {config.synthetic_samples} synthetic samples")
            self.data = torch.randn(
                config.synthetic_samples,
                config.in_channels,
                config.frames,
                config.height,
                config.width,
            )

    def _load_data(self):
        """Load data from file and handle layout."""
        if self.path.endswith(".pt"):
            self.data = torch.load(self.path, map_location="cpu")
        elif self.path.endswith(".npy"):
            self.data = torch.from_numpy(np.load(self.path))
        else:
            raise ValueError(f"Unsupported file format: {self.path}")

        # Auto-detect layout if needed
        if self.config.latent_layout == "auto":
            # Heuristic: Channels typically in {1, 3, 4, 16} while T is often {8, 16, 32, 64, 256}
            if self.data.ndim == 4:
                # Single sample: C, T, H, W or T, C, H, W
                dim0, dim1 = self.data.shape[0], self.data.shape[1]
                # If dim0 is small (<=16) AND dim1 is larger, likely CTHW
                # If dim1 is small (<=16) AND dim0 is larger, likely TCHW
                if dim0 <= 16 and dim1 > 16:
                    self.layout = "CTHW"
                elif dim1 <= 16 and dim0 > 16:
                    self.layout = "TCHW"
                    self.data = self.data.permute(1, 0, 2, 3)  # TCHW -> CTHW
                elif dim0 < dim1:
                    # Fallback: smaller is likely channels
                    self.layout = "CTHW"
                else:
                    self.layout = "TCHW"
                    self.data = self.data.permute(1, 0, 2, 3)  # TCHW -> CTHW
            elif self.data.ndim == 5:
                # Batched: B, C, T, H, W or B, T, C, H, W
                dim1, dim2 = self.data.shape[1], self.data.shape[2]
                if dim1 <= 16 and dim2 > 16:
                    self.layout = "BCTHW"
                elif dim2 <= 16 and dim1 > 16:
                    self.layout = "BTCHW"
                    self.data = self.data.permute(0, 2, 1, 3, 4)  # BTCHW -> BCTHW
                elif dim1 < dim2:
                    self.layout = "BCTHW"
                else:
                    self.layout = "BTCHW"
                    self.data = self.data.permute(0, 2, 1, 3, 4)  # BTCHW -> BCTHW
        elif self.config.latent_layout == "TCHW":
            if self.data.ndim == 4:
                self.data = self.data.permute(1, 0, 2, 3)
            elif self.data.ndim == 5:
                self.data = self.data.permute(0, 2, 1, 3, 4)

        print(f"Loaded {len(self)} samples from {self.path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {"latents": self.data[idx]}


def collate_latent_videos(batch):
    """Collate function for latent video batches.

    Args:
        batch: List of dicts with 'latents' key.

    Returns:
        Dict with stacked 'latents' tensor.
    """
    latents = torch.stack([item["latents"] for item in batch])
    return {"latents": latents}
