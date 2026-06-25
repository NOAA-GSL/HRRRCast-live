from __future__ import annotations

import math

import numpy as np
import torch

NUM_DIFFUSION_STEPS = 200
# Upstream ea1b4ed: 25 uniformly-spaced inference steps with the dpmpp-2m sampler
# (down from 50-step DDIM), allowing ~2x members at similar wall time.
NUM_INFERENCE_STEPS = 25


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    beta_start = 0.0001
    beta_end = 0.9999

    def alpha_bar_fn(t: int) -> float:
        return np.cos((t / timesteps + s) / (1 + s) * np.pi / 2) ** 2

    alphas_bar = np.array([alpha_bar_fn(t) for t in range(timesteps + 1)])
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return np.clip(betas, beta_start, beta_end)


BETA = cosine_beta_schedule(NUM_DIFFUSION_STEPS).astype(np.float32)
ALPHA = (1.0 - BETA).astype(np.float32)
ALPHA_BAR = np.cumprod(ALPHA, axis=0).astype(np.float32)
INFERENCE_STEPS = np.linspace(0, NUM_DIFFUSION_STEPS - 1, NUM_INFERENCE_STEPS).astype(np.int32)


def compute_epsilon(x_t: torch.Tensor, x_0: torch.Tensor, t: int) -> torch.Tensor:
    alpha_bar_t = torch.as_tensor(ALPHA_BAR[t], dtype=x_t.dtype, device=x_t.device)
    return (x_t - x_0 * torch.sqrt(alpha_bar_t)) / torch.sqrt(1.0 - alpha_bar_t)


def ddim(x_t: torch.Tensor, pred_noise: torch.Tensor, t_index: int, eta: float = 0.0) -> torch.Tensor:
    t = int(INFERENCE_STEPS[t_index])
    tm1 = int(INFERENCE_STEPS[t_index - 1])
    alpha_bar_t = torch.as_tensor(ALPHA_BAR[t], dtype=x_t.dtype, device=x_t.device)
    alpha_bar_tm1 = torch.as_tensor(ALPHA_BAR[tm1], dtype=x_t.dtype, device=x_t.device)

    x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
    if eta > 0.0:
        r1 = (1.0 - alpha_bar_tm1) / (1.0 - alpha_bar_t + 1e-12)
        r2 = 1.0 - (alpha_bar_t / (alpha_bar_tm1 + 1e-12))
        sigma_t = eta * torch.sqrt(r1 * r2)
    else:
        sigma_t = torch.zeros_like(alpha_bar_t)
    return torch.sqrt(alpha_bar_tm1) * x0_pred + torch.sqrt(1.0 - alpha_bar_tm1 - sigma_t**2) * pred_noise


def dpmpp_2m(
    x_t: torch.Tensor,
    pred_x0: torch.Tensor,
    t_index: int,
    prev_x0: torch.Tensor | None = None,
    prev_h: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """DPM-Solver++(2M) multistep update in x0-prediction space.

    Faithful port of `src/diffusion_params.dpmpp_2m` (upstream ea1b4ed). First
    step is first-order DPM-Solver++; subsequent steps add the 2nd-order
    multistep correction using the previous step's x0 and lambda-step `prev_h`.
    Returns `(x_{t-1}, x0_t, h)`; carry `x0_t`/`h` into the next call.
    """
    t = int(INFERENCE_STEPS[t_index])
    tm1 = int(INFERENCE_STEPS[t_index - 1])
    alpha_bar_t = float(ALPHA_BAR[t])
    alpha_bar_tm1 = float(ALPHA_BAR[tm1])

    alpha_t = math.sqrt(alpha_bar_t)
    alpha_tm1 = math.sqrt(alpha_bar_tm1)
    sigma_t = math.sqrt(1.0 - alpha_bar_t)
    sigma_tm1 = math.sqrt(1.0 - alpha_bar_tm1)

    lambda_t = math.log(alpha_t + 1e-12) - math.log(sigma_t + 1e-12)
    lambda_tm1 = math.log(alpha_tm1 + 1e-12) - math.log(sigma_tm1 + 1e-12)
    h = lambda_tm1 - lambda_t

    sample_coeff = sigma_tm1 / (sigma_t + 1e-12)
    phi_1 = math.expm1(-h)

    d0 = pred_x0
    x_tm1 = sample_coeff * x_t - alpha_tm1 * phi_1 * d0

    if prev_x0 is not None and prev_h is not None:
        r = prev_h / (h + 1e-12)
        d1 = (d0 - prev_x0) / (r + 1e-12)
        x_tm1 = x_tm1 - 0.5 * alpha_tm1 * phi_1 * d1

    return x_tm1, d0, h
