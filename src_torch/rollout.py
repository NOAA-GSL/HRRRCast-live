"""Multi-member, multi-hour autoregressive HRRRCast rollout core.

This module is the PyTorch equivalent of the autoregressive forecast loop
inside `src/fcst.py::WeatherForecaster.autoregressive_rollout`. It contains
only the tensor/numpy logic; per-hour I/O is delegated through a callback.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from .inference import DEFAULT_SAMPLER, GFS_CHANNELS, PREDICTED_CHANNELS, forecast_hour
from .model import HRRRCast


logger = logging.getLogger(__name__)


STATIC_CHANNELS = 2  # HRRR static (LAND/OROG) appended after predicted channels in the npz
DATE_CHANNELS = 6
STEP_CHANNELS = 1
LEAD_CHANNELS = 1


def _seed_from_pair(member: int, hour: int) -> int:
    """Map TensorFlow-style stateless seed pair `[member, hour]` to a torch seed.

    This does not reproduce TensorFlow's Philox stream bit-for-bit; it preserves
    the same seed structure and deterministic member/hour independence.
    """
    return ((int(member) & 0xFFFFFFFF) << 32) ^ (int(hour) & 0xFFFFFFFF)


def member_noise(
    member: int,
    shape: tuple[int, ...],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    *,
    hour: int = 0,
) -> torch.Tensor:
    """Deterministic Gaussian noise for a member/hour seed pair.

    Uses a `torch.Generator` seeded from `[member, hour]`, matching the
    TensorFlow code's stateless seed convention at the algorithm level. This is
    NOT bit-identical to TensorFlow `stateless_normal`; exact pointwise
    validation uses offline TF-dumped noise artifacts, not production forecast.
    """
    device_obj = torch.device(device)
    generator = torch.Generator(device=device_obj).manual_seed(_seed_from_pair(member, hour))
    return torch.randn(shape, generator=generator, dtype=dtype, device=device_obj)


def advance_member_noise(
    anchor_noise: torch.Tensor,
    *,
    member: int,
    hour: int,
    rho: float = 0.9,
) -> torch.Tensor:
    """Advance TF-style AR(1) member noise: rho * previous + sqrt(1-rho^2) * eps."""
    sigma = math.sqrt(1.0 - rho * rho)
    eps = member_noise(member, tuple(anchor_noise.shape), anchor_noise.device, anchor_noise.dtype, hour=hour)
    return anchor_noise * rho + eps * sigma


def phase_angles(num_members: int) -> dict[int, float]:
    """Per-member phase angles used to phase-shift GFS forcing index.

    Mirrors `src/fcst.py`: member 0 is always unshifted, even-sized
    ensembles also keep member 1 unshifted, then add symmetric +/- offsets.
    """
    members = list(range(num_members))
    half_count = num_members // 2 - ((num_members + 1) % 2)
    step = 1.0 / half_count if half_count > 0 else 0.0
    seq: list[float] = []
    seq.append(0.0)
    if num_members % 2 == 0:
        seq.append(0.0)
    for i in range(half_count):
        seq.append(step * (i + 1))
        seq.append(-step * (i + 1))
    return {m: seq[i] for i, m in enumerate(members)}


def compute_time_features(init_datetime: datetime, hour: int) -> np.ndarray:
    """Cyclic hour-of-day + day-of-year features plus HRRR v3/v4 era masks. Returns (1, 6)."""
    valid = pd.to_datetime([init_datetime]) + pd.to_timedelta([hour], unit="h")
    h = pd.DatetimeIndex(valid).hour.astype(np.float32)
    doy = pd.DatetimeIndex(valid).dayofyear.astype(np.float32)
    v4 = (valid >= np.datetime64("2021-03-23T00")).astype(np.float32)
    v3 = ((valid >= np.datetime64("2018-07-12T00")) & (valid < np.datetime64("2021-03-23T00"))).astype(np.float32)
    return np.stack(
        [
            np.sin(2 * np.pi * h / 24.0),
            np.cos(2 * np.pi * h / 24.0),
            np.sin(2 * np.pi * doy / 365.0),
            np.cos(2 * np.pi * doy / 365.0),
            v4.astype(np.float32),
            v3.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def date_encoding_field(init_datetime: datetime, hour: int, height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Tile the (1, 6) date encoding across the spatial grid, return NCHW (1, 6, H, W)."""
    feats = compute_time_features(init_datetime, hour)  # (1, 6)
    t = torch.from_numpy(feats).to(device=device, dtype=dtype)
    return t.view(1, 6, 1, 1).expand(1, 6, height, width).contiguous()


def build_initial_input(
    hrrr_npz: Mapping[str, np.ndarray],
    gfs_model_input: np.ndarray,
    *,
    predicted_channels: int = PREDICTED_CHANNELS,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the initial (1, C, H, W) NCHW input tensor used as the starting state.

    Channel layout (mirrors `src/fcst.py::main` for the diffusion path):
      [predicted (138)] + [gfs (42)] + [noise placeholder (138)]
      + [hrrr_static (2)] + [date (6)] + [step (1)] + [lead (1)]   = 328
    """
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    hrrr_input = np.asarray(hrrr_npz["model_input"], dtype=np.float32)
    gfs_input = np.asarray(gfs_model_input, dtype=np.float32)
    if hrrr_input.shape[0] != 1:
        raise ValueError(f"Expected HRRR model_input with batch=1, got shape {hrrr_input.shape}")
    nlat, nlon = hrrr_input.shape[1:3]
    static_count = hrrr_input.shape[-1] - predicted_channels
    if static_count < 0:
        raise ValueError(f"HRRR model_input has only {hrrr_input.shape[-1]} channels; needs at least {predicted_channels}.")

    noise_placeholder = np.ones((1, nlat, nlon, predicted_channels), dtype=np.float32)
    date_channel = np.ones((1, nlat, nlon, DATE_CHANNELS), dtype=np.float32)
    step_channel = np.ones((1, nlat, nlon, STEP_CHANNELS), dtype=np.float32)
    lead_channel = np.ones((1, nlat, nlon, LEAD_CHANNELS), dtype=np.float32)

    nhwc = np.concatenate(
        [
            hrrr_input[:, :, :, :predicted_channels],
            gfs_input[0:1, :, :, :],
            noise_placeholder,
            hrrr_input[:, :, :, predicted_channels:],
            date_channel,
            step_channel,
            lead_channel,
        ],
        axis=-1,
    )
    return torch.from_numpy(nhwc).permute(0, 3, 1, 2).contiguous().to(device=device_obj, dtype=dtype)


# Type alias for the per-hour callback. Receives normalized NHWC tensor (1, H, W, C_out).
HourCallback = Callable[[int, int, torch.Tensor], None]


def autoregressive_rollout(
    model: HRRRCast,
    *,
    init_input: torch.Tensor,
    gfs_forcing: torch.Tensor,
    members: list[int],
    num_members: int,
    lead_hours: int,
    init_datetime: datetime,
    channel_mins: torch.Tensor,
    channel_maxs: torch.Tensor,
    batch_size: int = 1,
    on_hour: HourCallback,
    predicted_channels: int = PREDICTED_CHANNELS,
    gfs_channels: int = GFS_CHANNELS,
    sampler: str = DEFAULT_SAMPLER,
    noise_rho: float = 0.9,
) -> None:
    """Run the multi-member, multi-hour HRRRCast rollout.

    Args:
        init_input: (1, C, H, W) NCHW initial model input (from `build_initial_input`).
        gfs_forcing: (N_fcst, C_gfs, H, W) NCHW GFS forcing block.
        members: list of integer member ids to forecast (sorted, deduplicated upstream).
        num_members: total ensemble size (used for symmetric phase shift weighting).
        lead_hours: max forecast lead time in hours (inclusive).
        on_hour: callback invoked once per (hour, member) with a normalized NHWC tensor.
            Called for hour=0 (initial state) before the loop, and once per forecast hour.
        noise_rho: AR(1) noise correlation coefficient, matching `src/fcst.py`.
    """
    device = init_input.device
    dtype = init_input.dtype
    height, width = init_input.shape[2], init_input.shape[3]

    on_hour(0, members[0] if members else 0, _state_to_nhwc(init_input[:, :predicted_channels]))
    # f00 is the same for every member (no diffusion yet); broadcast to remaining members.
    f00_state = init_input[:, :predicted_channels]
    for member in members[1:]:
        on_hour(0, member, _state_to_nhwc(f00_state))

    state_from_hour: dict[int, torch.Tensor] = {m: f00_state.clone() for m in members}
    noise_state: dict[int, torch.Tensor] = {
        m: member_noise(m, (1, predicted_channels, height, width), device=device, dtype=dtype, hour=0)
        for m in members
    }
    start_pred_noise = predicted_channels + gfs_channels
    phase_lookup = phase_angles(num_members)
    forcing_count = gfs_forcing.shape[0]

    for hour in range(1, lead_hours + 1):
        from_hour = ((hour - 1) // 6) * 6
        step = hour - from_hour

        date_enc = date_encoding_field(init_datetime, hour, height, width, dtype=dtype, device=device)
        lead_enc = torch.full((1, LEAD_CHANNELS, height, width), step / 6.0, dtype=dtype, device=device)

        # Static/noise placeholder + date + (step placeholder) + lead. The step
        # placeholder is overwritten inside the DDIM loop, so we just reuse the
        # existing slice of init_input for it.
        x_base = torch.cat(
            [
                init_input[:, start_pred_noise:-8],
                date_enc,
                init_input[:, -2:-1],
                lead_enc,
            ],
            dim=1,
        )

        for batch_start in range(0, len(members), batch_size):
            batch_members = members[batch_start : batch_start + batch_size]

            x_members = []
            xn_members = []
            for member in batch_members:
                phase_width = from_hour // 12
                phase_shift = round(phase_width * phase_lookup[member])
                forcing_idx = int(np.clip(hour - 1 + phase_shift, 0, forcing_count - 1))
                x_member = torch.cat(
                    [
                        state_from_hour[member],
                        gfs_forcing[forcing_idx : forcing_idx + 1],
                        x_base,
                    ],
                    dim=1,
                )
                x_members.append(x_member)
                xn = advance_member_noise(noise_state[member], member=member, hour=hour, rho=noise_rho)
                noise_state[member] = xn
                xn_members.append(xn)
            x_batch = torch.cat(x_members, dim=0)
            xn = torch.cat(xn_members, dim=0)

            t0 = time.time()
            y_nhwc = forecast_hour(
                model,
                x_batch,
                xn,
                channel_mins,
                channel_maxs,
                sampler=sampler,
                predicted_channels=predicted_channels,
                gfs_channels=gfs_channels,
            )
            elapsed = time.time() - t0
            logger.info(
                "hour=%d batch=%s/%s (members %s) predict %.3fs",
                hour,
                batch_start // batch_size + 1,
                (len(members) + batch_size - 1) // batch_size,
                batch_members,
                elapsed,
            )

            for batch_idx, member in enumerate(batch_members):
                y_member = y_nhwc[batch_idx : batch_idx + 1]  # NHWC (1, H, W, C)
                if hour % 6 == 0:
                    state_from_hour[member] = y_member.permute(0, 3, 1, 2).contiguous()
                on_hour(hour, member, y_member)


def _state_to_nhwc(t: torch.Tensor) -> torch.Tensor:
    return t.permute(0, 2, 3, 1).contiguous()


def gfs_forcing_to_nchw(gfs_model_input: np.ndarray, *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert the GFS preprocessed forcing block to (N_fcst, C, H, W) NCHW."""
    device_obj = torch.device(device or "cpu")
    arr = np.asarray(gfs_model_input, dtype=np.float32)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device=device_obj, dtype=dtype)
