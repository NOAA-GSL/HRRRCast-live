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
    forward_noise, compute_epsilon, ddpm, ddim
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
USE_LOGSNR_SPACED: bool = True
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

# ==== Inference steps computation ====
def _compute_log_snr_spaced_steps(num_inference_steps: int, num_diffusion_steps: int, alpha_bar: np.ndarray) -> np.ndarray:
    """
    Compute inference timesteps by evenly spacing the log-SNR values.
    Returns integer timestep indices in ascending order.
    """
    alpha_bar = np.array(alpha_bar, dtype=np.float64)

    # Clipping is essential to avoid log(0) or log(inf)
    eps = 1e-10
    alpha_bar = np.clip(alpha_bar, eps, 1.0 - eps)

    # Compute log-SNR for all timesteps
    log_snr = np.log(alpha_bar / (1.0 - alpha_bar))

    # Create evenly spaced log-SNR values
    log_snr_min = np.min(log_snr)
    log_snr_max = np.max(log_snr)

    # Start with more candidates than needed
    num_candidates = num_inference_steps * 178 // 100
    target_log_snr = np.linspace(log_snr_max, log_snr_min, num_candidates)

    # For each target, find the closest actual timestep
    timesteps = []
    for target in target_log_snr:
        idx = np.argmin(np.abs(log_snr - target))
        timesteps.append(idx)

    # Remove duplicates while preserving order
    seen = set()
    unique_timesteps = []
    for t in timesteps:
        if t not in seen:
            unique_timesteps.append(t)
            seen.add(t)
            if len(unique_timesteps) == num_inference_steps:
                break

    # If we still don't have enough (shouldn't happen with 2x candidates)
    # fall back to selecting evenly from remaining pool
    if len(unique_timesteps) < num_inference_steps:
        remaining = [t for t in range(num_diffusion_steps) if t not in seen]
        # Select from remaining based on their log-SNR values
        remaining_log_snr = [(t, log_snr[t]) for t in remaining]
        remaining_log_snr.sort(key=lambda x: x[1], reverse=True)

        for t, _ in remaining_log_snr:
            unique_timesteps.append(t)
            if len(unique_timesteps) == num_inference_steps:
                break

    return np.array(sorted(unique_timesteps[:num_inference_steps]))

# Compute inference steps
if USE_LOGSNR_SPACED:
    INFERENCE_STEPS = _compute_log_snr_spaced_steps(NUM_INFERENCE_STEPS, NUM_DIFFUSION_STEPS, ALPHA_BAR)
else:
    INFERENCE_STEPS = np.arange(0, NUM_DIFFUSION_STEPS, max(1, NUM_DIFFUSION_STEPS // NUM_INFERENCE_STEPS))

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
    
    t = tf.gather(list(INFERENCE_STEPS), t_ - 1)
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
