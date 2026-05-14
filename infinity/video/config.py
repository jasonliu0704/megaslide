"""Configuration for MegaSlide-DiT video diffusion models."""

from dataclasses import dataclass, field
from typing import Tuple, Union


@dataclass
class MegaSlideConfig:
    """Configuration for MegaSlide-DiT training and inference.

    Defaults are set for the paper's 105B parameter model. Override via YAML
    for different scales (e.g., tiny configs for smoke tests).
    """

    # ========== Model Architecture (Paper Table 2) ==========
    frames: int = 256
    """Number of video frames."""

    in_channels: int = 4
    """Number of latent channels (e.g., 4 for Stable Diffusion VAE)."""

    height: int = 1080
    """Video height in pixels (before VAE encoding)."""

    width: int = 1920
    """Video width in pixels (before VAE encoding)."""

    patch_size: int = 16
    """Spatial patch size for patchification (temporal dim not patched)."""

    hidden_size: int = 8192
    """Hidden dimension D."""

    num_layers: int = 48
    """Number of DiT blocks (105B model uses 48 layers)."""

    num_heads: int = 64
    """Number of attention heads."""

    mlp_ratio: float = 4.0
    """MLP expansion ratio (intermediate size = hidden_size * mlp_ratio)."""

    # ========== 3D-DSA Parameters (Paper Section 4.2) ==========
    dsa_kernel_size: Tuple[int, int, int] = (3, 7, 7)
    """Kernel size (k_t, k_h, k_w) for local deformable attention."""

    offset_scale: float = 1.0
    """Scale factor for learned offsets."""

    use_dwconv: bool = True
    """Whether to add depthwise 3D conv term (Equation in Section 4.2)."""

    dropout: float = 0.0
    """Dropout rate."""

    # ========== Diffusion Settings ==========
    diffusion_steps: int = 1000
    """Total diffusion timesteps for training."""

    noise_schedule: str = "linear"
    """Noise schedule type (linear, cosine, etc.)."""

    # ========== Training (Paper Section 5.1) ==========
    batch_size: int = 1
    """Batch size per step."""

    gradient_accumulation_steps: int = 1
    """Accumulate gradients over this many steps before optimizer update."""

    num_steps: int = 5000
    """Total training steps (paper uses 5,000 fine-tuning steps)."""

    learning_rate: float = 1e-5
    """Learning rate for optimizer."""

    weight_decay: float = 0.01
    """Weight decay for AdamW."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    seed: int = 42
    """Random seed for reproducibility."""

    # ========== Memory Management (Paper Section 3.4) ==========
    checkpoint_interval: int = 4
    """Save activation checkpoints every K layers."""

    num_grad_slabs: int = 12
    """Number of gradient slabs in the pool for async collection."""

    # ========== Optimizer (Paper Section 3.3) ==========
    optimizer: str = "cpu_adamw"
    """Optimizer type (cpu_adamw for CPU-resident optimizer)."""

    beta1: float = 0.9
    """Adam beta1 parameter."""

    beta2: float = 0.999
    """Adam beta2 parameter."""

    eps: float = 1e-8
    """Adam epsilon parameter."""

    # ========== System (Paper Section 5.1) ==========
    dtype: str = "bfloat16"
    """Data type for GPU computation (bfloat16, float16, float32)."""

    device: Union[int, str] = 0
    """CUDA device ID or 'cpu'."""

    force_cpu: bool = False
    """Force CPU-only mode (no GPU streaming)."""

    # ========== Dataset ==========
    path: str = ""
    """Path to latent video dataset (.pt or .npy files)."""

    synthetic_samples: int = 100
    """Number of synthetic samples to generate if no dataset path provided."""

    latent_layout: str = "auto"
    """Latent tensor layout: 'CTHW', 'TCHW', or 'auto' to detect."""

    # ========== Logging ==========
    log_interval: int = 10
    """Log metrics every N steps."""

    save_interval: int = 1000
    """Save checkpoint every N steps."""

    # ========== Optional Features ==========
    use_text_conditioning: bool = False
    """Enable cross-attention for text-to-video conditioning."""

    # ========== Computed Properties ==========

    @property
    def num_patches(self) -> int:
        """Total number of spatiotemporal patches after patchification."""
        H_patches = self.height // self.patch_size
        W_patches = self.width // self.patch_size
        return self.frames * H_patches * W_patches

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_size // self.num_heads

    @property
    def mlp_hidden_size(self) -> int:
        """MLP intermediate hidden size."""
        return int(self.hidden_size * self.mlp_ratio)

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"

        assert self.height % self.patch_size == 0, \
            f"height ({self.height}) must be divisible by patch_size ({self.patch_size})"

        assert self.width % self.patch_size == 0, \
            f"width ({self.width}) must be divisible by patch_size ({self.patch_size})"

        assert len(self.dsa_kernel_size) == 3, \
            f"dsa_kernel_size must be a 3-tuple (k_t, k_h, k_w), got {self.dsa_kernel_size}"

        assert self.checkpoint_interval > 0, \
            f"checkpoint_interval must be positive, got {self.checkpoint_interval}"

        assert self.num_grad_slabs >= 2, \
            f"num_grad_slabs must be at least 2 for async collection, got {self.num_grad_slabs}"

    def estimate_parameter_count(self) -> int:
        """Estimate total model parameters (for reference)."""
        # Patch embedding: in_channels * patch_size^2 * hidden_size
        patch_embed_params = self.in_channels * (self.patch_size ** 2) * self.hidden_size

        # Positional embeddings: num_patches * hidden_size
        pos_embed_params = self.num_patches * self.hidden_size

        # Timestep embedding MLP: hidden_size * (4 * hidden_size) * 2 layers
        time_embed_params = self.hidden_size * (4 * self.hidden_size) + (4 * self.hidden_size) * self.hidden_size

        # DiT blocks (approximate)
        # Each block: 3D-DSA + LayerNorm + MLP
        # 3D-DSA: QKV (3 * hidden_size^2) + out_proj (hidden_size^2) + offset_net + dwconv
        # MLP: 2 * hidden_size * mlp_hidden_size
        dsa_params = 4 * (self.hidden_size ** 2)  # Simplified
        mlp_params = 2 * self.hidden_size * self.mlp_hidden_size
        block_params = (dsa_params + mlp_params + 2 * self.hidden_size) * self.num_layers

        # Output head: hidden_size * (in_channels * patch_size^2)
        out_params = self.hidden_size * self.in_channels * (self.patch_size ** 2)

        total = patch_embed_params + pos_embed_params + time_embed_params + block_params + out_params
        return total

    def estimate_memory_footprint(self) -> dict:
        """Estimate memory footprint for paper Table 2."""
        param_count = self.estimate_parameter_count()

        # FP16/BF16 weights (GPU transient)
        fp16_weights_gb = param_count * 2 / 1e9

        # FP32 master weights (CPU persistent)
        fp32_master_gb = param_count * 4 / 1e9

        # Adam moments (CPU persistent): 2 * FP32
        adam_moments_gb = 2 * param_count * 4 / 1e9

        # Total persistent state (CPU RAM)
        total_persistent_gb = fp16_weights_gb + fp32_master_gb + adam_moments_gb

        # Single activation tensor: num_patches * hidden_size * 2 bytes (FP16)
        activation_gb = self.num_patches * self.hidden_size * 2 / 1e9

        # Checkpoint activations: (num_layers / checkpoint_interval) * activation_size
        checkpoint_gb = (self.num_layers / self.checkpoint_interval) * activation_gb

        return {
            "parameter_count": param_count,
            "fp16_weights_gb": fp16_weights_gb,
            "fp32_master_gb": fp32_master_gb,
            "adam_moments_gb": adam_moments_gb,
            "total_persistent_gb": total_persistent_gb,
            "single_activation_gb": activation_gb,
            "checkpoint_activations_gb": checkpoint_gb,
        }
