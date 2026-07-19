"""MegaSlide-DiT: Video Diffusion Transformers with CPU-Master Training.

Implements the MegaSlide-DiT paper's approach:
- 3D Deformable Slide Attention (3D-DSA) for linear-complexity spatiotemporal processing
- CPU-resident parameter storage with async GPU streaming
- Full-parameter adaptation of massive video DiT models on single GPU
"""

from .config import MegaSlideConfig
from .yaml_loader import load_megaslide_config
from .attention import DeformableSlideAttention3D
from .model import MegaSlideDiT, DiTBlock, MLP
from .trainer import CPUMasterVideoDiT
from .dataset import LatentVideoDataset, collate_latent_videos
from .baselines import Dense3DDiT, SwinDiT
from .hybrid_attention import HybridAttention3D, TemporalAnchorAttention3D
from .hybrid_model import HybridMegaSlideDiT, TemporalAnchorMegaSlideDiT
from .metrics import compute_temporal_consistency, compute_offset_magnitude

__all__ = [
    "MegaSlideConfig",
    "load_megaslide_config",
    "DeformableSlideAttention3D",
    "MegaSlideDiT",
    "DiTBlock",
    "MLP",
    "CPUMasterVideoDiT",
    "LatentVideoDataset",
    "collate_latent_videos",
    "Dense3DDiT",
    "SwinDiT",
    "HybridAttention3D",
    "TemporalAnchorAttention3D",
    "HybridMegaSlideDiT",
    "TemporalAnchorMegaSlideDiT",
    "compute_temporal_consistency",
    "compute_offset_magnitude",
]
