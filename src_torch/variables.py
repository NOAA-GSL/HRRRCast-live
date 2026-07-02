"""Channel layout, normalization stats, inverse transforms, and the diagnostics bridge."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch

from .config import REPO_ROOT

# Per-channel normalization statistics shipped with the model. Override with
# the HRRRCAST_NORM_FILE environment variable; defaults to the repo copy.
DEFAULT_NORM_FILE = Path(
    os.environ.get("HRRRCAST_NORM_FILE", REPO_ROOT / "net-diffusion" / "normalize-stats.nc")
)


PL_VARS = ["UGRD", "VGRD", "VVEL", "TMP", "HGT", "SPFH"]
SFC_VARS = [
    "PRES",
    "MSLMA",
    "REFC",
    "T2M",
    "UGRD10M",
    "VGRD10M",
    "UGRD80M",
    "VGRD80M",
    "D2M",
    "TCDC",
    "LCDC",
    "MCDC",
    "HCDC",
    "VIS",
    "APCP",
    "HGTCC",
    "CAPE",
    "CIN",
]
LEVELS = [200, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
LOG_TRANSFORM_VARS = {"VIS", "APCP", "HGTCC", "CAPE"}
NEG_LOG_TRANSFORM_VARS = {"CIN"}
RAW_BOUNDS = {
    "UGRD": (-120, 120),
    "VGRD": (-120, 120),
    "VVEL": (-30, 30),
    "TMP": (180, 340),
    "HGT": (-600, 20000),
    "SPFH": (0, 0.05),
    "PRES": (50000, 110000),
    "MSLMA": (50000, 110000),
    "REFC": (0, 80),
    "T2M": (180, 340),
    "UGRD10M": (-100, 100),
    "VGRD10M": (-100, 100),
    "UGRD80M": (-100, 100),
    "VGRD80M": (-100, 100),
    "D2M": (180, 340),
    "TCDC": (0, 100),
    "LCDC": (0, 100),
    "MCDC": (0, 100),
    "HCDC": (0, 100),
    "VIS": (0, 100000),
    "APCP": (0, 500),
    "HGTCC": (0, 20000),
    "CAPE": (0, 7000),
    "CIN": (-2000, 0),
}


def _fallback_bounds() -> tuple[np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    for i, var in enumerate(RAW_BOUNDS):
        vmin, vmax = RAW_BOUNDS[var]
        if var in LOG_TRANSFORM_VARS:
            vmin = np.log1p(vmin)
            vmax = np.log1p(vmax)
        elif var in NEG_LOG_TRANSFORM_VARS:
            vmin = np.sign(vmin) * np.log1p(abs(vmin))
            vmax = np.sign(vmax) * np.log1p(abs(vmax))
        count = len(LEVELS) if i < len(PL_VARS) else 1
        mins.extend([vmin] * count)
        maxs.extend([vmax] * count)
    return np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)


def channel_bounds(
    norm_file: str | Path = DEFAULT_NORM_FILE,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    means, stds, raw_mins, raw_maxs = read_channel_stats(norm_file)
    mins = (raw_mins - means) / stds
    maxs = (raw_maxs - means) / stds
    return torch.from_numpy(mins).to(device=device), torch.from_numpy(maxs).to(device=device)


def channel_mean_std(
    norm_file: str | Path = DEFAULT_NORM_FILE,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    means, stds, _raw_mins, _raw_maxs = read_channel_stats(norm_file)
    return torch.from_numpy(means).to(device=device), torch.from_numpy(stds).to(device=device)


def read_channel_stats(
    norm_file: str | Path = DEFAULT_NORM_FILE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fallback_mins, fallback_maxs = _fallback_bounds()
    raw_means = []
    raw_stds = []
    raw_mins = []
    raw_maxs = []
    channel_idx = 0
    with h5py.File(norm_file, "r") as h5:
        for var in PL_VARS:
            stats = np.asarray(h5[var])
            nlev_stats = stats.shape[1] if stats.ndim > 1 else 1
            for i, _level in enumerate(LEVELS):
                if i < nlev_stats and stats.shape[0] >= 2:
                    mean = float(stats[0, i])
                    std = float(stats[1, i]) or 1.0
                    vmin = float(stats[2, i]) if stats.shape[0] > 2 else float(fallback_mins[channel_idx])
                    vmax = float(stats[3, i]) if stats.shape[0] > 3 else float(fallback_maxs[channel_idx])
                    if np.isnan(vmin):
                        vmin = float(fallback_mins[channel_idx])
                    if np.isnan(vmax):
                        vmax = float(fallback_maxs[channel_idx])
                else:
                    mean = 0.0
                    std = 1.0
                    vmin = float(fallback_mins[channel_idx])
                    vmax = float(fallback_maxs[channel_idx])
                raw_means.append(mean)
                raw_stds.append(std)
                raw_mins.append(vmin)
                raw_maxs.append(vmax)
                channel_idx += 1

        for var in SFC_VARS:
            stats = np.asarray(h5[var])
            if stats.shape[0] >= 2:
                mean = float(np.nanmean(stats[0]))
                std = float(np.nanmean(stats[1])) or 1.0
                vmin = float(np.nanmean(stats[2])) if stats.shape[0] > 2 else float(fallback_mins[channel_idx])
                vmax = float(np.nanmean(stats[3])) if stats.shape[0] > 3 else float(fallback_maxs[channel_idx])
                if np.isnan(vmin):
                    vmin = float(fallback_mins[channel_idx])
                if np.isnan(vmax):
                    vmax = float(fallback_maxs[channel_idx])
            else:
                mean = 0.0
                std = 1.0
                vmin = float(fallback_mins[channel_idx])
                vmax = float(fallback_maxs[channel_idx])
            raw_means.append(mean)
            raw_stds.append(std)
            raw_mins.append(vmin)
            raw_maxs.append(vmax)
            channel_idx += 1

    return (
        np.asarray(raw_means, dtype=np.float32),
        np.asarray(raw_stds, dtype=np.float32),
        np.asarray(raw_mins, dtype=np.float32),
        np.asarray(raw_maxs, dtype=np.float32),
    )


def denormalize(output: torch.Tensor) -> torch.Tensor:
    means, stds = channel_mean_std(device=output.device)
    return output * stds[: output.shape[-1]] + means[: output.shape[-1]]


def inverse_transforms_physical(output: torch.Tensor) -> torch.Tensor:
    result = output.clone()
    offset = len(PL_VARS) * len(LEVELS)
    for name in LOG_TRANSFORM_VARS:
        idx = SFC_VARS.index(name)
        channel = offset + idx
        result[..., channel] = torch.expm1(result[..., channel])
    for name in NEG_LOG_TRANSFORM_VARS:
        idx = SFC_VARS.index(name)
        channel = offset + idx
        value = result[..., channel]
        result[..., channel] = torch.sign(value) * torch.expm1(torch.abs(value))
    return result


def compute_diagnostics(ds):
    """Add HRRRCast diagnostic variables, delegating to the model's `src/diagnostics.py`.

    Imported lazily so merely importing this module does not pull the `src/`
    post-processing dependencies (xarray, etc.).
    """
    from .config import add_src_to_path

    add_src_to_path()
    from diagnostics import compute_diagnostics as _impl  # type: ignore[import-not-found]

    return _impl(ds)
