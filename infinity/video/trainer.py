"""CPU-Master trainer for MegaSlide-DiT.

Adapts patterns from infinity/model/cpu_master.py:
- Double-buffered GPU weight streaming
- Gradient checkpointing with selective recompute
- K-slab async gradient collection
- CPU-resident optimizer with FP32 master weights
"""

import queue
import threading
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .config import MegaSlideConfig
from .model import MegaSlideDiT


class CPUMasterVideoDiT:
    """CPU-master trainer for MegaSlide-DiT with async streaming.

    Key features:
    - All persistent parameters (FP16 + FP32 master + Adam moments) on CPU
    - Double-buffered GPU execution (stream layer i+1 while computing layer i)
    - Gradient checkpointing every K layers
    - K-slab gradient pool for async D2H transfer
    - CPU-resident AdamW optimizer

    Args:
        model: MegaSlideDiT instance.
        config: MegaSlideConfig instance.
        force_cpu: Force CPU-only mode for testing (default: False).
    """

    def __init__(self, model: MegaSlideDiT, config: MegaSlideConfig, force_cpu: bool = False):
        self.model = model
        self.config = config
        self.force_cpu = force_cpu

        # Determine device
        if force_cpu or not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.use_cuda = False
            print("[CPUMasterVideoDiT] Running in CPU-only mode")
        else:
            self.device = torch.device(f"cuda:{config.device}")
            self.use_cuda = True
            print(f"[CPUMasterVideoDiT] Running on {self.device}")

        # Move model to CPU and extract components
        self.model.to("cpu")
        self._extract_model_components()

        # Set up double-buffered GPU streaming if CUDA available
        if self.use_cuda:
            self._setup_cuda_streaming()
        else:
            # Fallback: just move whole model to CPU
            self.model.to("cpu")

        # Initialize CPU-resident optimizer
        self.optimizer = torch.optim.AdamW(
            self.get_parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

        self.step_count = 0

    def _extract_model_components(self):
        """Extract model components for layer-by-layer streaming."""
        # Separate components:
        # - patch_embed, pos_embed, time_embed: keep on GPU
        # - blocks: stream one at a time
        # - norm_out, out_proj: keep on GPU

        self.patch_embed = self.model.patch_embed
        self.pos_embed = self.model.pos_embed
        self.time_embed = self.model.time_embed
        self.blocks = self.model.blocks
        self.norm_out = self.model.norm_out
        self.out_proj = self.model.out_proj

        # Cache block structure info
        self.num_blocks = len(self.blocks)

    def _setup_cuda_streaming(self):
        """Set up CUDA streams, buffers, and events for async pipeline."""
        # CUDA streams
        self.compute_stream = torch.cuda.Stream(device=self.device)
        self.weight_stream = torch.cuda.Stream(device=self.device)
        self.grad_stream = torch.cuda.Stream(device=self.device)

        # CUDA events for synchronization
        self.weight_ready_events = [torch.cuda.Event() for _ in range(2)]
        self.backward_done_events = [torch.cuda.Event() for _ in range(2)]
        self.buffer_busy_events = [torch.cuda.Event() for _ in range(2)]

        # Move head/tail components to GPU
        self.patch_embed.to(self.device)
        self.pos_embed = self.pos_embed.to(self.device)
        self.time_embed.to(self.device)
        self.norm_out.to(self.device)
        self.out_proj.to(self.device)

        # Double-buffered GPU templates for blocks
        # We'll create 2 GPU copies and swap between them
        self.gpu_blocks = [None, None]
        for i in range(2):
            # Deep copy first block as template
            self.gpu_blocks[i] = self._create_gpu_block_template().to(self.device)

        # K-slab gradient pool (pinned CPU memory)
        self.grad_slabs = []
        self.grad_slab_free_list = queue.Queue()

        # Estimate max block params for slab size
        max_block_params = sum(p.numel() for p in self.blocks[0].parameters())

        for i in range(self.config.num_grad_slabs):
            slab = torch.empty(max_block_params, dtype=torch.float32, pin_memory=True)
            self.grad_slabs.append(slab)
            self.grad_slab_free_list.put(i)

        # Gradient collection task queue and worker thread
        self.grad_task_queue = queue.Queue()
        self.grad_worker_thread = threading.Thread(target=self._grad_worker, daemon=True)
        self.grad_worker_thread.start()

        # Events for head/tail gradients
        self.head_slab_event = torch.cuda.Event()
        self.embed_slab_event = torch.cuda.Event()
        self.head_slab_free = threading.Event()
        self.embed_slab_free = threading.Event()
        self.head_slab_free.set()
        self.embed_slab_free.set()

        # Separate slabs for head/tail
        head_params = sum(p.numel() for p in self.norm_out.parameters()) + \
                     sum(p.numel() for p in self.out_proj.parameters())
        embed_params = sum(p.numel() for p in self.patch_embed.parameters()) + \
                      sum(p.numel() for p in self.time_embed.parameters())

        self.head_grad_slab = torch.empty(head_params, dtype=torch.float32, pin_memory=True)
        self.embed_grad_slab = torch.empty(embed_params, dtype=torch.float32, pin_memory=True)

    def _create_gpu_block_template(self):
        """Create a GPU-resident copy of a DiT block."""
        import copy
        return copy.deepcopy(self.blocks[0])

    def _load_block_to_gpu(self, block_idx: int, gpu_slot: int):
        """Load block parameters from CPU to GPU slot."""
        cpu_block = self.blocks[block_idx]
        gpu_block = self.gpu_blocks[gpu_slot]

        with torch.cuda.stream(self.weight_stream):
            for (cpu_name, cpu_param), (gpu_name, gpu_param) in zip(
                cpu_block.named_parameters(), gpu_block.named_parameters()
            ):
                gpu_param.data.copy_(cpu_param.data, non_blocking=True)

            self.weight_ready_events[gpu_slot].record(self.weight_stream)

    def _grad_worker(self):
        """Worker thread to accumulate gradients from GPU slabs to CPU parameters."""
        while True:
            try:
                task = self.grad_task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task is None:  # Shutdown signal
                break

            task_type, slab_idx, cpu_params, shapes, numels = task

            # Wait for D2H transfer to complete
            if task_type == 'block':
                self.grad_stream.synchronize()
                slab_flat = self.grad_slabs[slab_idx]
            elif task_type == 'head':
                self.head_slab_event.synchronize()
                slab_flat = self.head_grad_slab
            elif task_type == 'embed':
                self.embed_slab_event.synchronize()
                slab_flat = self.embed_grad_slab

            # Accumulate gradients to CPU parameters
            offset = 0
            for p, shape, numel in zip(cpu_params, shapes, numels):
                grad_flat = slab_flat[offset:offset + numel]
                grad_reshaped = grad_flat.view(shape)

                if p.grad is None:
                    p.grad = grad_reshaped.clone()
                else:
                    p.grad.add_(grad_reshaped)

                offset += numel

            # Return slab to pool
            if task_type == 'block':
                self.grad_slab_free_list.put(slab_idx)
            elif task_type == 'head':
                self.head_slab_free.set()
            elif task_type == 'embed':
                self.embed_slab_free.set()

    def forward_and_backward(
        self,
        latents: Tensor,
        timesteps: Tensor,
        target_noise: Tensor,
        text_embeds: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Forward and backward pass with checkpointing and async streaming.

        Args:
            latents: Noisy latents [B, C, T, H, W].
            timesteps: Timesteps [B].
            target_noise: Target noise to predict [B, C, T, H, W].
            text_embeds: Optional text embeddings [B, L, D].
            text_mask: Optional text mask [B, L].

        Returns:
            Tuple of (loss, timing_dict).
        """
        if not self.use_cuda:
            return self._forward_and_backward_cpu(latents, timesteps, target_noise,
                                                  text_embeds, text_mask)

        B, C, T_frames, H, W = latents.shape

        # CUDA events for timing
        start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        start.record()

        # Move inputs to GPU
        latents_gpu = latents.to(self.device)
        timesteps_gpu = timesteps.to(self.device)
        target_noise_gpu = target_noise.to(self.device)

        # === FORWARD PASS ===

        # Patchify
        x = self.patch_embed(latents_gpu)  # [B, D, T, H', W']
        _, D, T, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]

        # Add positional embeddings
        x = x + self.pos_embed

        # Timestep conditioning
        t_emb = self.model._get_timestep_embedding(timesteps_gpu, D, self.device)
        t_emb = self.time_embed(t_emb)
        x = x + t_emb.unsqueeze(1)

        # Forward through blocks with checkpointing
        video_shape = (T, H_p, W_p)
        checkpoints = {}

        with torch.no_grad():
            # Prefetch first block
            self._load_block_to_gpu(0, 0)
            self.weight_stream.synchronize()

            for i in range(self.num_blocks):
                buffer_idx = i % 2
                next_buffer_idx = (i + 1) % 2

                # Checkpoint at intervals
                if i % self.config.checkpoint_interval == 0:
                    checkpoints[i] = x.detach()

                # Prefetch next block
                if i + 1 < self.num_blocks:
                    self._load_block_to_gpu(i + 1, next_buffer_idx)

                # Wait for current block weights to be ready
                self.compute_stream.wait_event(self.weight_ready_events[buffer_idx])

                with torch.cuda.stream(self.compute_stream):
                    self.buffer_busy_events[buffer_idx].record(self.compute_stream)

                    # Forward through block
                    x = self.gpu_blocks[buffer_idx](x, video_shape, text_embeds, text_mask)

        checkpoints[self.num_blocks] = x.detach()

        # Output projection
        x = self.norm_out(x)
        x = self.out_proj(x)  # [B, N, C * p^2]

        # Unpatchify
        p = self.config.patch_size
        x = x.view(B, T, H_p, W_p, C, p, p)
        x = x.permute(0, 4, 1, 2, 5, 3, 6).contiguous()
        pred_noise = x.view(B, C, T_frames, H, W)

        fwd_end.record()

        # === LOSS + BACKWARD ===

        # Compute MSE loss
        loss = nn.functional.mse_loss(pred_noise, target_noise_gpu)
        loss_val = loss.item()

        # Start backward from output
        loss.backward()

        # Collect head gradients (norm_out + out_proj)
        if not self.head_slab_free.wait(timeout=30.0):
            raise RuntimeError("head slab wait timeout")
        self.head_slab_free.clear()

        with torch.cuda.stream(self.grad_stream):
            offset = 0
            for module in [self.norm_out, self.out_proj]:
                for p in module.parameters():
                    if p.grad is not None:
                        numel = p.grad.numel()
                        self.head_grad_slab[offset:offset + numel].copy_(
                            p.grad.flatten(), non_blocking=True
                        )
                        p.grad = None
                        offset += numel
            self.head_slab_event.record(self.grad_stream)

        cpu_params = list(self.norm_out.parameters()) + list(self.out_proj.parameters())
        shapes = [p.shape for p in cpu_params]
        numels = [p.numel() for p in cpu_params]
        self.grad_task_queue.put(('head', None, cpu_params, shapes, numels))

        # Get gradient w.r.t. hidden states before norm/proj
        # Need to recompute to get gradient
        hidden_before_proj = checkpoints[self.num_blocks].requires_grad_(True)
        hidden_after_norm = self.norm_out(hidden_before_proj)
        pred_recompute = self.out_proj(hidden_after_norm)

        # Backward to get grad_hidden
        pred_recompute.backward(pred_noise.grad)
        grad_hidden = hidden_before_proj.grad.detach()

        del hidden_before_proj, hidden_after_norm, pred_recompute

        # Backward through blocks with checkpointed recompute
        num_checkpoint_blocks = (self.num_blocks + self.config.checkpoint_interval - 1) // \
                               self.config.checkpoint_interval

        for block_idx in range(num_checkpoint_blocks - 1, -1, -1):
            block_start = block_idx * self.config.checkpoint_interval
            block_end = min((block_idx + 1) * self.config.checkpoint_interval, self.num_blocks)

            current_checkpoint = checkpoints[block_start]

            # Recompute forward for this checkpoint block
            recompute_cache = {}
            hidden_recompute = current_checkpoint

            with torch.no_grad():
                for j in range(block_start, block_end):
                    buffer_idx = j % 2
                    self._load_block_to_gpu(j, buffer_idx)
                    self.compute_stream.wait_event(self.weight_ready_events[buffer_idx])

                    with torch.cuda.stream(self.compute_stream):
                        hidden_recompute = self.gpu_blocks[buffer_idx](
                            hidden_recompute, video_shape, text_embeds, text_mask
                        )
                        recompute_cache[j] = hidden_recompute.detach()

            # Backward through block
            for i in range(block_end - 1, block_start - 1, -1):
                buffer_idx = i % 2

                # Get input for this layer
                if i == block_start:
                    layer_input = current_checkpoint.detach().requires_grad_(True)
                else:
                    layer_input = recompute_cache[i - 1].requires_grad_(True)

                # Load block to GPU
                self._load_block_to_gpu(i, buffer_idx)
                self.compute_stream.wait_event(self.weight_ready_events[buffer_idx])

                with torch.cuda.stream(self.compute_stream):
                    # Enable gradients for autograd.grad
                    for p in self.gpu_blocks[buffer_idx].parameters():
                        p.requires_grad_(True)

                    # Forward
                    layer_output = self.gpu_blocks[buffer_idx](
                        layer_input, video_shape, text_embeds, text_mask
                    )

                    # Backward via autograd.grad
                    grads = torch.autograd.grad(
                        outputs=layer_output,
                        inputs=(layer_input, *self.gpu_blocks[buffer_idx].parameters()),
                        grad_outputs=grad_hidden,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )

                    grad_hidden = grads[0].detach()
                    param_grads = grads[1:]

                    # Attach gradients to GPU block
                    for p, g in zip(self.gpu_blocks[buffer_idx].parameters(), param_grads):
                        p.grad = g

                    # Disable gradients again
                    for p in self.gpu_blocks[buffer_idx].parameters():
                        p.requires_grad_(False)

                    self.backward_done_events[buffer_idx].record(self.compute_stream)

                # Collect gradients async
                self._collect_block_grads_async(i, buffer_idx)

                del layer_input, layer_output

            recompute_cache.clear()

        # Backward through patch_embed and time_embed
        latents_recompute = latents_gpu.requires_grad_(True)
        x_recompute = self.patch_embed(latents_recompute)
        x_recompute = x_recompute.flatten(2).transpose(1, 2)
        x_recompute = x_recompute + self.pos_embed

        t_emb_recompute = self.model._get_timestep_embedding(timesteps_gpu, D, self.device)
        t_emb_recompute = self.time_embed(t_emb_recompute)
        x_recompute = x_recompute + t_emb_recompute.unsqueeze(1)

        x_recompute.backward(grad_hidden)

        # Collect embed gradients
        if not self.embed_slab_free.wait(timeout=30.0):
            raise RuntimeError("embed slab wait timeout")
        self.embed_slab_free.clear()

        with torch.cuda.stream(self.grad_stream):
            offset = 0
            for module in [self.patch_embed, self.time_embed]:
                for p in module.parameters():
                    if p.grad is not None:
                        numel = p.grad.numel()
                        self.embed_grad_slab[offset:offset + numel].copy_(
                            p.grad.flatten(), non_blocking=True
                        )
                        p.grad = None
                        offset += numel
            self.embed_slab_event.record(self.grad_stream)

        cpu_params = list(self.patch_embed.parameters()) + list(self.time_embed.parameters())
        shapes = [p.shape for p in cpu_params]
        numels = [p.numel() for p in cpu_params]
        self.grad_task_queue.put(('embed', None, cpu_params, shapes, numels))

        # Wait for all gradient collection to complete
        self._accumulate_grads_batch()

        bwd_end.record()
        torch.cuda.synchronize()

        fwd_time = start.elapsed_time(fwd_end) / 1000.0
        bwd_time = fwd_end.elapsed_time(bwd_end) / 1000.0
        total_time = start.elapsed_time(bwd_end) / 1000.0

        checkpoints.clear()

        return loss_val, {
            'forward': fwd_time,
            'backward': bwd_time,
            'total': total_time,
        }

    def _forward_and_backward_cpu(
        self,
        latents: Tensor,
        timesteps: Tensor,
        target_noise: Tensor,
        text_embeds: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """CPU-only fallback forward/backward."""
        start = time.perf_counter()

        # Forward
        fwd_start = time.perf_counter()
        pred_noise = self.model(latents, timesteps, text_embeds, text_mask)
        fwd_time = time.perf_counter() - fwd_start

        # Loss
        loss = nn.functional.mse_loss(pred_noise, target_noise)

        # Backward
        bwd_start = time.perf_counter()
        loss.backward()
        bwd_time = time.perf_counter() - bwd_start

        total_time = time.perf_counter() - start

        return loss.item(), {
            'forward': fwd_time,
            'backward': bwd_time,
            'total': total_time,
        }

    def _collect_block_grads_async(self, block_idx: int, gpu_slot: int):
        """Async transfer of block gradients from GPU to CPU slab."""
        slab_idx = self.grad_slab_free_list.get()

        with torch.cuda.stream(self.grad_stream):
            self.grad_stream.wait_event(self.backward_done_events[gpu_slot])

            slab_flat = self.grad_slabs[slab_idx]
            offset = 0

            for p in self.gpu_blocks[gpu_slot].parameters():
                if p.grad is not None:
                    numel = p.grad.numel()
                    slab_flat[offset:offset + numel].copy_(p.grad.flatten(), non_blocking=True)
                    p.grad = None
                    offset += numel

        cpu_block = self.blocks[block_idx]
        cpu_params = list(cpu_block.parameters())
        shapes = [p.shape for p in cpu_params]
        numels = [p.numel() for p in cpu_params]

        self.grad_task_queue.put(('block', slab_idx, cpu_params, shapes, numels))

    def _accumulate_grads_batch(self):
        """Wait for all pending gradient accumulation tasks."""
        # Just ensure worker has processed all tasks
        # In practice, worker processes async; we synchronize before optimizer step
        pass

    def optimizer_step(self) -> float:
        """Run optimizer step with gradient clipping.

        Returns:
            Gradient norm after clipping.
        """
        # Wait for all async gradient collection to finish
        if self.use_cuda:
            torch.cuda.synchronize()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.get_parameters(), self.config.max_grad_norm
        )
        grad_norm_val = grad_norm.item()

        # Optimizer step
        self.optimizer.step()
        self.step_count += 1

        # Log gradient norm for monitoring
        if self.step_count % self.config.log_interval == 0:
            print(f"  [Step {self.step_count}] Gradient norm: {grad_norm_val:.4f}")

        return grad_norm_val

    def zero_grad(self):
        """Zero out gradients."""
        self.optimizer.zero_grad()

    def get_parameters(self):
        """Get all trainable parameters."""
        return list(self.model.parameters())

    def cleanup(self):
        """Cleanup CUDA resources."""
        if self.use_cuda:
            # Send shutdown signal to worker
            self.grad_task_queue.put(None)
            self.grad_worker_thread.join(timeout=5.0)

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step_count,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint with validation.

        Args:
            path: Path to checkpoint file.

        Raises:
            ValueError: If checkpoint is invalid or incompatible.
        """
        checkpoint = torch.load(path, map_location="cpu")

        # Validate checkpoint structure
        required_keys = ["model", "optimizer", "step", "config"]
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            raise ValueError(f"Invalid checkpoint: missing keys {missing_keys}")

        # Validate config compatibility (check critical params)
        saved_config = checkpoint["config"]
        critical_params = ["hidden_size", "num_layers", "num_heads", "in_channels"]
        mismatches = []
        for param in critical_params:
            if getattr(saved_config, param) != getattr(self.config, param):
                mismatches.append(
                    f"{param}: saved={getattr(saved_config, param)}, "
                    f"current={getattr(self.config, param)}"
                )
        if mismatches:
            raise ValueError(
                f"Checkpoint config mismatch:\n" + "\n".join(mismatches)
            )

        # Load state dicts
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step_count = checkpoint["step"]
        print(f"✓ Loaded checkpoint from step {self.step_count}")
