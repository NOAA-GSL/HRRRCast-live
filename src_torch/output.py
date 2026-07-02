"""Per-hour HRRRCast NetCDF assembly and writing helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping, Union

import numpy as np
import torch
import xarray as xr

from .variables import (
    LEVELS,
    PL_VARS,
    SFC_VARS,
    compute_diagnostics,
    denormalize,
    inverse_transforms_physical,
)


def create_dataset(
    init_datetime: datetime,
    hour: int,
    lats: np.ndarray,
    lons: np.ndarray,
    data: np.ndarray,
) -> xr.Dataset:
    """Build an xarray dataset for a single (init_time, lead_time=hour) from a (H, W, C) array."""
    data_vars: dict[str, xr.DataArray] = {}
    var_index = 0
    for var in PL_VARS:
        pl_data = np.transpose(data[..., var_index : var_index + len(LEVELS)], (2, 0, 1))
        data_vars[var] = xr.DataArray(
            np.expand_dims(np.expand_dims(pl_data, 0), 0),
            dims=("time", "lead_time", "level", "latitude", "longitude"),
            coords={
                "time": [init_datetime],
                "lead_time": ("lead_time", [hour], {"units": "hours"}),
                "level": ("level", LEVELS, {"units": "hPa"}),
                "latitude": (("latitude", "longitude"), lats),
                "longitude": (("latitude", "longitude"), lons),
            },
            name=var,
        )
        var_index += len(LEVELS)

    for var in SFC_VARS:
        data_vars[var] = xr.DataArray(
            np.expand_dims(np.expand_dims(data[..., var_index], 0), 0),
            dims=("time", "lead_time", "latitude", "longitude"),
            coords={
                "time": [init_datetime],
                "lead_time": ("lead_time", [hour], {"units": "hours"}),
                "latitude": (("latitude", "longitude"), lats),
                "longitude": (("latitude", "longitude"), lons),
            },
            name=var,
        )
        var_index += 1
    return xr.Dataset(data_vars)


def add_static_fields(
    ds: xr.Dataset,
    static_arrays: Mapping[str, np.ndarray],
    init_datetime: datetime,
    hour: int,
    lats: np.ndarray,
    lons: np.ndarray,
) -> xr.Dataset:
    """Inject 2D constant fields (e.g. LAND, OROG) into the dataset for this hour."""
    for name, values in static_arrays.items():
        ds[name] = xr.DataArray(
            np.asarray(values, dtype=np.float32)[None, None, :, :],
            dims=("time", "lead_time", "latitude", "longitude"),
            coords={
                "time": [init_datetime],
                "lead_time": ("lead_time", [hour], {"units": "hours"}),
                "latitude": (("latitude", "longitude"), lats),
                "longitude": (("latitude", "longitude"), lons),
            },
            name=name,
        )
    return ds


def static_fields_from_npz(npz: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    """Extract LAND/OROG-style constant fields from a preprocessed HRRR npz."""
    out: dict[str, np.ndarray] = {}
    for name in ("LAND", "OROG"):
        key = f"{name}_raw"
        if key in npz.files:
            out[name] = np.asarray(npz[key], dtype=np.float32)
    return out


def normalized_to_physical(normalized: torch.Tensor) -> np.ndarray:
    """Denormalize a (1, H, W, C) NHWC tensor and undo log transforms; returns numpy (H, W, C)."""
    return inverse_transforms_physical(denormalize(normalized)).detach().cpu().numpy()[0]


def write_hour(
    *,
    normalized: torch.Tensor,
    hour: int,
    static_fields: Mapping[str, np.ndarray],
    init_datetime: datetime,
    lats: np.ndarray,
    lons: np.ndarray,
    output_dir: Path,
    member: Union[int, str],
) -> Path:
    """Build per-hour dataset, run diagnostics, and write `hrrrcast_mem{member}_f{hour:02d}.nc`."""
    physical = normalized_to_physical(normalized)
    ds = create_dataset(init_datetime, hour, lats, lons, physical)
    ds = add_static_fields(ds, static_fields, init_datetime, hour, lats, lons)
    ds = compute_diagnostics(ds)
    # Upstream naming convention: per-member `m{NN}`, ensemble mean `avg`, spread `spr`.
    mem_str = str(member) if str(member) in ("avg", "spr") else f"m{int(member):02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"hrrrcast_{mem_str}_f{hour:02d}.nc"
    ds.to_netcdf(out)
    ds.close()
    return out


def convert_netcdf_hours_to_grib2(
    *,
    init_time: str,
    member: int | str,
    hours: list[int],
    in_dir: str | Path,
    out_dir: str | Path,
) -> list[Path]:
    """Convert per-hour HRRRCast NetCDF files to GRIB2 via the model's `src/nc2grib.py`.

    `nc2grib` (and its `grib2io`/`eccodes` deps) is imported lazily so importing
    this module never requires the GRIB stack.
    """
    from .config import add_src_to_path

    add_src_to_path()
    from nc2grib import Netcdf2Grib  # type: ignore[import-not-found]

    init_datetime = datetime.fromisoformat(init_time)
    in_path = Path(in_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    converter = Netcdf2Grib()
    written: list[Path] = []
    for hour in hours:
        mem_str = member if member in ("avg", "spr") else f"m{int(member):02d}"
        nc_path = in_path / f"hrrrcast_{mem_str}_f{hour:02d}.nc"
        ds = xr.open_dataset(nc_path, decode_timedelta=False)
        out_file = out_path / f"hrrrcast.{mem_str}.t{init_datetime.hour:02d}z.pgrb2.f{hour:02d}"
        try:
            # Upstream nc2grib.save_grib2 takes the full output path; the
            # member->filename mapping lives here in the caller.
            converter.save_grib2(init_datetime, ds, str(out_file))
            written.append(out_file)
        finally:
            ds.close()
    return written
