"""
diffusion_params.py
-------------------
Diffusion and CRPS parameter schedules and utilities for weather model training.

This module provides beta schedules, constants, and noise functions for diffusion models
and CRPS ensemble training. It supports linear, quadratic, sigmoid, and cosine beta schedules,
and provides forward and reverse noise functions for DDPM and DDIM sampling.

Constants:
    USE_DIFFUSION: Enable/disable diffusion model.
    USE_CRPS: Enable/disable CRPS ensemble training.
    USE_VECTORIZED: Enable/disable vectorized operations for CRPS.
    NUM_DIFFUSION_STEPS: Number of diffusion steps.
    NUM_INFERENCE_STEPS: Number of inference steps for sampling.
    NUM_CRPS_ENSEMBLES: Number of CRPS ensemble members.

Functions:
    linear_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule, cosine_beta_schedule
    forward_noise, compute_epsilon, ddpm, ddim, ddim_heun, dpmpp_2m
"""

import numpy as np
import tensorflow as tf
from typing import Union


# ==== Diffusion and CRPS configuration ====
USE_DIFFUSION: bool = True
USE_CRPS: bool = False
USE_VECTORIZED: bool = False

# ==== Diffusion/CRPS step counts ====
NUM_DIFFUSION_STEPS: int = 200
NUM_INFERENCE_STEPS: int = 50
INFERENCE_STEP_SPACING: str = "uniform"  # "uniform", "logsnr", "powerlaw"
POWERLAW_GAMMA: float = 0.45
NUM_CRPS_ENSEMBLES: int = 4


# ==== Beta Schedules ====
def linear_beta_schedule(timesteps: int) -> np.ndarray:
    """
    Linear schedule for beta values.
    Args:
        timesteps (int): Number of timesteps.
    Returns:
        np.ndarray: Linearly spaced beta values.
    """
    beta_start = 0.0001 * 1000 / timesteps
    beta_end = 0.02 * 1000 / timesteps
    return np.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps: int) -> np.ndarray:
    """
    Quadratic schedule for beta values.
    Args:
        timesteps (int): Number of timesteps.
    Returns:
        np.ndarray: Quadratically spaced beta values.
    """
    beta_start = 0.0001 * 1000 / timesteps
    beta_end = 0.02 * 1000 / timesteps
    return np.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps: int) -> np.ndarray:
    """
    Sigmoid schedule for beta values.
    Args:
        timesteps (int): Number of timesteps.
    Returns:
        np.ndarray: Sigmoid-spaced beta values.
    """
    beta_start = 0.0001 * 1000 / timesteps
    beta_end = 0.02 * 1000 / timesteps
    betas = np.linspace(-6, 6, timesteps)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    return sigmoid(betas) * (beta_end - beta_start) + beta_start


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    """
    Cosine schedule for beta values.
    Args:
        timesteps (int): Number of timesteps.
        s (float): Small offset for stability.
    Returns:
        np.ndarray: Cosine-spaced beta values.
    """
    beta_start = 0.0001
    beta_end = 0.9999

    def alpha_bar_fn(t):
        return np.cos((t / timesteps + s) / (1 + s) * np.pi / 2) ** 2

    alphas_bar = np.array([alpha_bar_fn(t) for t in range(timesteps + 1)])
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    betas = np.clip(betas, beta_start, beta_end)
    return betas


# ==== Diffusion constants (reparameterization trick) ====
BETA: np.ndarray = cosine_beta_schedule(NUM_DIFFUSION_STEPS).astype(np.float32)
ALPHA: np.ndarray = (1.0 - BETA).astype(np.float32)
ALPHA_BAR: np.ndarray = np.cumprod(ALPHA, axis=0).astype(np.float32)
SQRT_ALPHA_BAR: np.ndarray = np.sqrt(ALPHA_BAR).astype(np.float32)
SQRT_ONE_MINUS_ALPHA_BAR: np.ndarray = np.sqrt(1.0 - ALPHA_BAR).astype(np.float32)

# ==== Inference timestep clustering utilities ====
def _powerlaw_deltas_to_0(gamma: float) -> np.ndarray:
    T = NUM_DIFFUSION_STEPS - 1
    n = NUM_INFERENCE_STEPS - 1

    i = np.arange(1, n + 1)

    # power-law weights
    w = i ** gamma

    # normalize to sum T
    d = (w / w.sum()) * T

    # integer projection
    d = np.floor(d).astype(int)

    # ---- FIX 1: ensure no zeros without destroying structure ----
    # instead of forcing all zeros to 1, track deficit
    zero_mask = (d == 0)
    deficit = np.sum(zero_mask)
    d[zero_mask] = 1

    # ---- FIX 2: renormalize after fixing zeros ----
    current_sum = np.sum(d)
    diff = T - current_sum

    # distribute correction across all elements (NOT just last)
    if diff != 0:
        # spread adjustment proportionally to weights
        order = np.argsort(w)[::-1]  # largest weights get priority

        for k in range(abs(diff)):
            idx = order[k % len(order)]
            d[idx] += 1 if diff > 0 else -1

            # prevent going below 1
            if d[idx] < 1:
                d[idx] = 1

    return d

def _build_timesteps(deltas: np.ndarray) -> np.ndarray:
    t = np.zeros(len(deltas) + 1, dtype=int)
    t[1:] = np.cumsum(deltas)
    return t

def cluster_to_T(gamma: float = POWERLAW_GAMMA) -> np.ndarray:
    d0 = _powerlaw_deltas_to_0(gamma)
    dT = d0[::-1]
    return _build_timesteps(dT)

# ==== Log-SNR steps ====
_LOGSNR_EPS = 1e-10
_ALPHA_BAR_CLIP = np.clip(ALPHA_BAR, _LOGSNR_EPS, 1.0 - _LOGSNR_EPS)
LOG_SNR: np.ndarray = np.log(_ALPHA_BAR_CLIP / (1.0 - _ALPHA_BAR_CLIP)).astype(np.float32)
LOG_SNR_MIN: float = float(np.min(LOG_SNR))
LOG_SNR_MAX: float = float(np.max(LOG_SNR))

def _compute_log_snr_spaced_steps() -> np.ndarray:
    """
    Compute inference timesteps by evenly spacing the log-SNR values.
    Returns integer timestep indices in ascending order.
    """

    # Start with more candidates than needed
    num_candidates = NUM_INFERENCE_STEPS * 178 // 100
    target_log_snr = np.linspace(LOG_SNR_MAX, LOG_SNR_MIN, num_candidates)

    # For each target, find the closest actual timestep
    timesteps = []
    for target in target_log_snr:
        idx = np.argmin(np.abs(LOG_SNR - target))
        timesteps.append(idx)

    # Remove duplicates while preserving order
    seen = set()
    unique_timesteps = []
    for t in timesteps:
        if t not in seen:
            unique_timesteps.append(t)
            seen.add(t)
            if len(unique_timesteps) == NUM_INFERENCE_STEPS:
                break

    # If we still don't have enough (shouldn't happen with 2x candidates)
    # fall back to selecting evenly from remaining pool
    if len(unique_timesteps) < NUM_INFERENCE_STEPS:
        remaining = [t for t in range(NUM_DIFFUSION_STEPS) if t not in seen]
        # Select from remaining based on their log-SNR values
        remaining_log_snr = [(t, LOG_SNR[t]) for t in remaining]
        remaining_log_snr.sort(key=lambda x: x[1], reverse=True)

        for t, _ in remaining_log_snr:
            unique_timesteps.append(t)
            if len(unique_timesteps) == NUM_INFERENCE_STEPS:
                break

    steps = np.array(sorted(unique_timesteps[:NUM_INFERENCE_STEPS]), dtype=np.int32)

    # Ensure endpoint coverage without creating a large last-step jump:
    # if T is missing, insert it and drop earliest interior points so any
    # spacing distortion is pushed toward small timesteps near 0.
    T = NUM_DIFFUSION_STEPS - 1
    if steps[-1] != T:
        steps = np.unique(np.append(steps, T)).astype(np.int32)
        excess = len(steps) - NUM_INFERENCE_STEPS
        if excess > 0:
            # Keep the first point and trim from early interior indices.
            steps = np.concatenate([steps[:1], steps[1 + excess:]]).astype(np.int32)

    return steps.astype(np.int32)

# ==== Uniform steps ====
def _compute_uniform_spaced_steps() -> np.ndarray:
    """Compute approximately uniform inference timesteps in ascending order."""
    return np.linspace(0, NUM_DIFFUSION_STEPS - 1, NUM_INFERENCE_STEPS, dtype=np.int32)

# ==== Power-law steps ====
def _compute_powerlaw_spaced_steps(gamma: float = POWERLAW_GAMMA) -> np.ndarray:
    """Compute power-law-clustered timesteps toward T in ascending order."""
    return cluster_to_T(gamma=gamma).astype(np.int32)


# ==== Inference steps computation ====
if INFERENCE_STEP_SPACING == "logsnr":
    INFERENCE_STEPS = _compute_log_snr_spaced_steps()
elif INFERENCE_STEP_SPACING == "uniform":
    INFERENCE_STEPS = _compute_uniform_spaced_steps()
elif INFERENCE_STEP_SPACING == "powerlaw":
    INFERENCE_STEPS = _compute_powerlaw_spaced_steps()
else:
    raise ValueError(
        f"Unknown INFERENCE_STEP_SPACING={INFERENCE_STEP_SPACING!r}. "
        "Expected one of: 'uniform', 'logsnr', 'powerlaw'."
    )

# ==== DDPM diffusion functions ====
def forward_noise(x_0: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    """
    Add forward noise to input tensor x_0 at timestep t.
    Args:
        x_0 (tf.Tensor): Original input tensor.
        t (tf.Tensor): Timestep indices.
    Returns:
        tf.Tensor: Noised tensor x_t.
    """
    SQRT_ALPHA_BAR_t = tf.reshape(tf.gather(SQRT_ALPHA_BAR, t), (-1, 1, 1, 1))
    SQRT_ONE_MINUS_ALPHA_BAR_t = tf.reshape(tf.gather(SQRT_ONE_MINUS_ALPHA_BAR, t), (-1, 1, 1, 1))
    x_t = tf.random.normal(shape=tf.shape(x_0), dtype=tf.float32)
    x_t *= SQRT_ONE_MINUS_ALPHA_BAR_t
    x_t += tf.cast(x_0, tf.float32) * SQRT_ALPHA_BAR_t
    return tf.cast(x_t, x_0.dtype)


def compute_epsilon(x_t: tf.Tensor, x_0: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    """
    Compute epsilon (noise) given x_t, x_0, and timestep t.
    Args:
        x_t (tf.Tensor): Noised tensor.
        x_0 (tf.Tensor): Original tensor.
        t (tf.Tensor): Timestep indices.
    Returns:
        tf.Tensor: Computed epsilon noise.
    """
    SQRT_ALPHA_BAR_t = tf.reshape(tf.gather(SQRT_ALPHA_BAR, t), (-1, 1, 1, 1))
    SQRT_ONE_MINUS_ALPHA_BAR_t = tf.reshape(tf.gather(SQRT_ONE_MINUS_ALPHA_BAR, t), (-1, 1, 1, 1))
    epsilon = (x_t - x_0 * SQRT_ALPHA_BAR_t) / SQRT_ONE_MINUS_ALPHA_BAR_t
    return epsilon


def ddpm(x_t: tf.Tensor, pred_noise: tf.Tensor, t_: int, seed: Union[int, list] = 0) -> tf.Tensor:
    """
    Reverse diffusion step using DDPM.
    Args:
        x_t (tf.Tensor): Noised tensor at time t, shape (batch_size, H, W, C).
        pred_noise (tf.Tensor): Predicted noise, shape (batch_size, H, W, C).
        t_ (int): Inference step index.
        seed (int or list): Random seed for reproducibility. If list, must match batch_size.
    Returns:
        x_{t-1} (tf.Tensor): tensor after one reverse step.
    """
    # Normalize seed to list
    batch_size = tf.shape(x_t)[0]
    if isinstance(seed, int):
        seeds = [seed] * batch_size
    else:
        seeds = seed
    
    t = tf.gather(list(INFERENCE_STEPS), t_)
    ALPHA_t = tf.gather(ALPHA, t)
    ALPHA_BAR_t = tf.gather(ALPHA_BAR, t)
    BETA_t = tf.gather(BETA, t)
    eps_coef = (1.0 - ALPHA_t) / tf.sqrt(1.0 - ALPHA_BAR_t)
    mean = (1.0 / tf.sqrt(ALPHA_t)) * (x_t - eps_coef * pred_noise)
    var = tf.where(t > 0, BETA_t, tf.zeros([], tf.float32))

    # add stochasticity per sample
    z_shape = tf.shape(x_t)[1:]
    z_samples = []
    for batch_idx, s in enumerate(seeds):
        z = tf.random.stateless_normal(
            shape=tf.concat([[1], z_shape], axis=0),
            seed=tf.stack([s, t]),
            dtype=tf.float32
        )
        z_samples.append(z)
    z = tf.concat(z_samples, axis=0)  # (batch_size, H, W, C)

    return mean + tf.sqrt(var) * z


def ddim(
    x_t: tf.Tensor,
    pred_noise: tf.Tensor,
    t_: int,
    seed: Union[int, list] = 0,
    eta: float = 0.0,
) -> tf.Tensor:
    """
    DDIM + stochasticity using DDIM η-schedule (stable).
    Args:
        x_t: Noised sample at DDIM inference step, shape (batch_size, H, W, C).
        pred_noise: Predicted noise ε_θ(x_t, t), shape (batch_size, H, W, C).
        t_: Inference index (not training timestep index).
        seed: RNG seed. If int, use for all samples. If list, must match batch_size.
        eta: DDIM stochasticity parameter (0 = deterministic, 1 = DDPM-level variance).
    Returns:
        x_{t-1} (tf.Tensor): tensor after one reverse step.
    """
    # Normalize seed to list
    batch_size = tf.shape(x_t)[0]
    if isinstance(seed, int):
        seeds = [seed] * batch_size
    else:
        seeds = seed
    
    t = tf.gather(list(INFERENCE_STEPS), t_)
    tm1 = tf.gather(list(INFERENCE_STEPS), t_ - 1)
    ALPHA_BAR_t = tf.gather(ALPHA_BAR, t)
    ALPHA_BAR_tm1 = tf.gather(ALPHA_BAR, tm1)

    # Predicted x0
    x0_pred = (x_t - tf.sqrt(1.0 - ALPHA_BAR_t) * pred_noise) / tf.sqrt(ALPHA_BAR_t)

    # Compute the correct DDIM sigma variance term
    if eta > 0.0:
        r1 = (1.0 - ALPHA_BAR_tm1) / (1.0 - ALPHA_BAR_t + 1e-12)
        r2 = 1.0 - (ALPHA_BAR_t / (ALPHA_BAR_tm1 + 1e-12))
        sigma_t = eta * tf.sqrt(r1 * r2)
    else:
        sigma_t = tf.zeros_like(ALPHA_BAR_t)

    # Deterministic DDIM part
    mean = (
        tf.sqrt(ALPHA_BAR_tm1) * x0_pred +
        tf.sqrt(1.0 - ALPHA_BAR_tm1 - sigma_t**2) * pred_noise
    )

    # Add stochastic residual noise per sample
    if eta > 0.0:
        z_shape = tf.shape(x_t)[1:]
        z_samples = []
        for batch_idx, s in enumerate(seeds):
            z = tf.random.stateless_normal(
                shape=tf.concat([[1], z_shape], axis=0),
                seed=tf.stack([s, t]),
                dtype=tf.float32
            )
            z_samples.append(z)
        z = tf.concat(z_samples, axis=0)  # (batch_size, H, W, C)
        x_tm1 = mean + sigma_t * z
    else:
        x_tm1 = mean

    return x_tm1


def ddim_heun(
    x_t: tf.Tensor,
    pred_noise_t: tf.Tensor,
    pred_noise_tm1: tf.Tensor,
    t_: int,
    seed: Union[int, list] = 0,
    eta: float = 0.0,
) -> tf.Tensor:
    """
    Second-order DDIM-Heun step using predictor-corrector averaging in noise space.
    Args:
        x_t: Current sample at DDIM inference index t_.
        pred_noise_t: Predicted noise at x_t.
        pred_noise_tm1: Predicted noise at Euler-predicted x_{t-1}.
        t_: Inference index (not training timestep index).
        seed: RNG seed. If int, use for all samples. If list, must match batch_size.
        eta: DDIM stochasticity parameter.
    Returns:
        x_{t-1} after Heun correction.
    """
    pred_noise_avg = 0.5 * (pred_noise_t + pred_noise_tm1)
    return ddim(x_t, pred_noise_avg, t_, seed=seed, eta=eta)


def dpmpp_2m(
    x_t: tf.Tensor,
    pred_x0_t: tf.Tensor,
    t_: int,
    prev_x0: tf.Tensor | None = None,
    prev_h: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    DPM-Solver++(2M) multistep update in x0-prediction space.

    Falls back to first-order DPM-Solver++ on the first step (when prev_x0/prev_h
    are not available), and uses second-order multistep correction afterwards.

    Args:
        x_t: Current sample at inference index t_.
        pred_x0_t: Direct x0 prediction from the network at x_t.
        t_: Inference index (not training timestep index).
        prev_x0: Previous step x0 prediction (for multistep correction).
        prev_h: Previous lambda-step size h.
    Returns:
        tuple: (x_{t-1}, x0_t, h)
    """
    t = tf.gather(list(INFERENCE_STEPS), t_)
    tm1 = tf.gather(list(INFERENCE_STEPS), t_ - 1)

    alpha_bar_t = tf.gather(ALPHA_BAR, t)
    alpha_bar_tm1 = tf.gather(ALPHA_BAR, tm1)

    alpha_t = tf.sqrt(alpha_bar_t)
    alpha_tm1 = tf.sqrt(alpha_bar_tm1)
    sigma_t = tf.sqrt(1.0 - alpha_bar_t)
    sigma_tm1 = tf.sqrt(1.0 - alpha_bar_tm1)

    x0_t = pred_x0_t

    lambda_t = tf.math.log(alpha_t + 1e-12) - tf.math.log(sigma_t + 1e-12)
    lambda_tm1 = tf.math.log(alpha_tm1 + 1e-12) - tf.math.log(sigma_tm1 + 1e-12)
    h = lambda_tm1 - lambda_t

    sample_coeff = sigma_tm1 / (sigma_t + 1e-12)
    phi_1 = tf.math.expm1(-h)

    # First-order DPM-Solver++ update.
    d0 = x0_t
    x_tm1 = sample_coeff * x_t - alpha_tm1 * phi_1 * d0

    # Second-order multistep correction when previous model output is available.
    if prev_x0 is not None and prev_h is not None:
        r = prev_h / (h + 1e-12)
        d1 = (d0 - prev_x0) / (r + 1e-12)
        x_tm1 = x_tm1 - 0.5 * alpha_tm1 * phi_1 * d1

    return x_tm1, x0_t, h
