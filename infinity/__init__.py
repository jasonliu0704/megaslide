"""MegaSlide-DiT: Memory-Centric Adaptation and Deformable Local Attention for Efficient Video Diffusion.

Adapts 105B+ parameter video Diffusion Transformers on a single GPU by:
- Streaming weight shards from host CPU memory (1.5 TB DDR5)
- Replacing O(N²) global attention with 3D Deformable Slide Attention (linear complexity)
- Async double-buffered CPU↔GPU communication with overlap
- CPU-bound AdamW optimizer with AVX-512
"""

from .model.cpu_master import CPUMasterModel
from .config.training import CPUMasterConfig
from .data.datasets import ChatDataset, MetaMathDataset, collate_fn

# Video DiT components (lazy import to avoid torch dependency at package level)
def __getattr__(name):
    """Lazy import for video module to avoid torch dependency."""
    if name in ["MegaSlideConfig", "load_megaslide_config", "DeformableSlideAttention3D",
                "MegaSlideDiT", "DiTBlock", "MLP", "CPUMasterVideoDiT",
                "LatentVideoDataset", "collate_latent_videos"]:
        from . import video
        return getattr(video, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "0.3.0"

__all__ = [
    "CPUMasterModel",
    "CPUMasterConfig",
    "ChatDataset",
    "MetaMathDataset",
    "collate_fn",
    # Video DiT exports
    "MegaSlideConfig",
    "load_megaslide_config",
    "DeformableSlideAttention3D",
    "MegaSlideDiT",
    "DiTBlock",
    "MLP",
    "CPUMasterVideoDiT",
    "LatentVideoDataset",
    "collate_latent_videos",
]
