"""Core HRRRCast inference engine: model load, sampling loop, and one forecast hour.

`diffusion_loop` is the single shared reverse-diffusion sampler; `forecast_hour`
wraps it (diffusion -> clip -> normalized NHWC) and is the per-hour unit used by
both the multi-hour `rollout` and the `validate` parity harness, so production
and verification run identical numerics.
"""

from __future__ import annotations

import torch

from .config import DEFAULT_MODULE_STATE
from .diffusion import (
    INFERENCE_STEPS,
    NUM_DIFFUSION_STEPS,
    NUM_INFERENCE_STEPS,
    compute_epsilon,
    ddim,
    dpmpp_2m,
)
from .model import GFS_CHANNELS, HRRR_CHANNELS, HRRRCast

# 138 diffusion-predicted channels (the HRRR analysis state).
PREDICTED_CHANNELS = HRRR_CHANNELS

# Upstream ea1b4ed samples with DPM-Solver++(2M); "ddim" is kept for reference.
DEFAULT_SAMPLER = "dpmpp"


def load_hrrrcast(
    state_path: str | None = None,
    *,
    device: torch.device | str | None = None,
    compile_model: bool = False,
    compile_mode: str = "default",
) -> HRRRCast:
    """Construct `HRRRCast` and load the module state dict for inference.

    When `compile_model` is set, the module is wrapped with `torch.compile`
    (Inductor), which fuses the elementwise/LayerNorm ops that dominate GPU
    time and reduces kernel-launch overhead. The first forward call pays a
    one-time compilation cost; subsequent same-shape calls run the fused graph.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = HRRRCast().to(device).eval()
    state = torch.load(state_path or str(DEFAULT_MODULE_STATE), map_location=device, weights_only=True)
    model.load_state_dict(state)
    # Store conv weights in channels_last so cuDNN runs its NHWC tensor-core
    # kernels on H100 without inserting per-conv NCHW<->NHWC transposes.
    model = model.to(memory_format=torch.channels_last)
    for param in model.parameters():
        param.requires_grad_(False)
    if compile_model:
        model = torch.compile(model, mode=compile_mode)
    return model


def diffusion_loop(
    model: HRRRCast,
    x_batch: torch.Tensor,
    xn: torch.Tensor,
    *,
    sampler: str = DEFAULT_SAMPLER,
    predicted_channels: int = PREDICTED_CHANNELS,
    gfs_channels: int = GFS_CHANNELS,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    num_diffusion_steps: int = NUM_DIFFUSION_STEPS,
) -> torch.Tensor:
    """Reverse-diffusion sampling loop for a stacked-member NCHW batch.

    `sampler` selects the per-step update (all ops are elementwise, so the loop
    stays in NCHW):
      * ``"dpmpp"`` - DPM-Solver++(2M) in x0 space (upstream ea1b4ed default).
      * ``"ddim"``  - first-order DDIM in epsilon space.

    The noised channels of the model input are replaced by the running iterate
    `xn` and the per-step time encoding each iteration. Returns the final `xn`
    (NCHW); callers clip to physical bounds.
    """
    if sampler not in ("dpmpp", "ddim"):
        raise ValueError(f"unknown sampler {sampler!r}; expected 'dpmpp' or 'ddim'")
    start = predicted_channels + gfs_channels
    prev_x0 = prev_h = None
    with torch.no_grad():
        for t_i in range(num_inference_steps - 1):
            ti = num_inference_steps - 1 - t_i
            t = int(INFERENCE_STEPS[ti])
            step_encoding = torch.full(
                (x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3]),
                t / num_diffusion_steps,
                dtype=x_batch.dtype,
                device=x_batch.device,
            )
            x_iter = torch.cat(
                [
                    x_batch[:, :start],
                    xn,
                    x_batch[:, start + predicted_channels : -2],
                    step_encoding,
                    x_batch[:, -1:],
                ],
                dim=1,
            )
            x0 = model(x_iter)
            if sampler == "dpmpp":
                xn, prev_x0, prev_h = dpmpp_2m(xn, x0, ti, prev_x0=prev_x0, prev_h=prev_h)
            else:
                epsilon = compute_epsilon(xn, x0, t)
                xn = ddim(xn, epsilon, ti)
    return xn


def forecast_hour(
    model: HRRRCast,
    x_batch: torch.Tensor,
    xn: torch.Tensor,
    channel_mins: torch.Tensor,
    channel_maxs: torch.Tensor,
    *,
    sampler: str = DEFAULT_SAMPLER,
    predicted_channels: int = PREDICTED_CHANNELS,
    gfs_channels: int = GFS_CHANNELS,
) -> torch.Tensor:
    """Run one forecast hour for a stacked-member batch.

    inputs (assembled NCHW `x_batch` + initial noise `xn`) -> `diffusion_loop`
    -> clip to physical channel bounds -> normalized NHWC `(B, H, W, C_pred)`.

    Args:
        x_batch: assembled model input, NCHW `(B, C_in, H, W)`.
        xn: initial Gaussian noise for the predicted channels, NCHW `(B, C_pred, H, W)`.
        channel_mins/channel_maxs: per-channel normalized clip bounds.
    """
    y = diffusion_loop(
        model,
        x_batch,
        xn,
        sampler=sampler,
        predicted_channels=predicted_channels,
        gfs_channels=gfs_channels,
    )
    mins = channel_mins[:predicted_channels].to(device=y.device, dtype=y.dtype)[None, :, None, None]
    maxs = channel_maxs[:predicted_channels].to(device=y.device, dtype=y.dtype)[None, :, None, None]
    y = torch.clip(y, mins, maxs)
    return y.permute(0, 2, 3, 1).contiguous()
