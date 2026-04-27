# MegaSlide-DiT Prototype Plan

## Summary
- Add a separate MegaSlide-DiT video path while preserving the existing LLM/VLM `CPUMasterModel` behavior.
- Implement the paper’s core ideas as a working PyTorch prototype: CPU-resident model state, async layer streaming, checkpointed recompute, CPU AdamW, and 3D Deformable Slide Attention.
- Do not implement custom CUDA/Triton kernels or claim reproduction of the paper’s H200/VBench numbers in this first version.

## Key Changes
- Add a new `infinity.video` package with:
  - `MegaSlideConfig`: video-DiT settings for frames, latent size, patch size, hidden size, layers, heads, DSA kernel sizes, checkpoint interval, dtype, device, and optimizer fields.
  - `DeformableSlideAttention3D`: non-causal 3D local attention using depthwise `Conv3d`, learned offsets, trilinear `grid_sample`, softmax over local neighborhoods, and output projection.
  - `MegaSlideDiT`: patchifies video latents with spatial patches, adds timestep conditioning, runs DiT blocks with 3D-DSA + MLP, then unpatchifies to predict noise with the same shape as input latents.
  - `CPUMasterVideoDiT`: CPU-master trainer wrapper adapted from the existing async layer-streaming pattern, with pinned host buffers, double GPU buffers, gradient slabs, checkpointed block recompute, and CPU optimizer state.
- Add a standalone training entrypoint and tiny config:
  - `examples/train_megaslide_dit.py`
  - `examples/configs/megaslide_dit_tiny.yaml`
- Add a simple latent-video dataset path:
  - Accept `.pt` / `.npy` latent tensors shaped `[C, T, H, W]` or `[T, C, H, W]`.
  - Provide a synthetic random latent mode for smoke tests when no real dataset is configured.
- Export the new public classes from `infinity/__init__.py` without changing existing exports.

## Public Interfaces
- `MegaSlideDiT.forward(latents, timesteps, text_embeds=None, text_mask=None) -> pred_noise`
  - `latents`: `[B, C, T, H, W]`
  - `timesteps`: `[B]`
  - output: `[B, C, T, H, W]`
- `DeformableSlideAttention3D.forward(x, video_shape) -> x`
  - `x`: `[B, N, D]`
  - `video_shape`: `(T, H_patches, W_patches)`
  - no causal mask; no optical-flow output or claims.
- `CPUMasterVideoDiT.forward_and_backward(latents, timesteps, target_noise, text_embeds=None, text_mask=None)` returns loss and timing metrics.

## Test Plan
- Add CPU unit tests for 3D-DSA shape, finite outputs, gradients, non-causal operation, and zero/small-offset behavior.
- Add model tests verifying patchify/unpatchify shape preservation and one forward/backward pass on a tiny synthetic video.
- Add config/dataset tests for YAML parsing and `.pt`/synthetic latent loading.
- Add CUDA smoke tests, skipped when CUDA is unavailable, for one streamed training step with a tiny MegaSlide-DiT.
- Re-run existing focused tests to ensure the LLM/VLM path is unchanged: `pytest -q tests/test_gradients.py tests/test_fixes.py`.

## Assumptions
- First implementation is the selected PyTorch prototype, not a production CUDA/Triton kernel.
- MegaSlide-DiT lives in a new video path, not inside the existing HF decoder trainer.
- No new heavy dependencies are required for v1; use existing PyTorch/einops/yaml stack.
- The prototype supports full-parameter adaptation mechanics, but does not require private 105B weights, VBench integration, or H200-scale validation.
