#!/usr/bin/env python3
"""Full multi-member, multi-hour HRRRCast forecast (PyTorch port of src/fcst.py).

Pipeline:
  1. Load HRRR + GFS preprocessed npz files.
  2. Load the clean HRRRCast nn.Module.
  3. Assemble the initial (1, 328, H, W) model input.
  4. Run the autoregressive rollout, per-hour writing NetCDF (and optional GRIB2)
     per ensemble member, with optional PMM nudging on REFC/APCP.

Example:
  python -m src_torch.cli forecast --init-time 2024-05-06T23 --lead-hours 1 --members 0
  python -m src_torch.cli forecast --init-time 2024-05-06T23 --members 0-3 --batch-size 2
  python -m src_torch.cli validate          # parity vs the TF reference dump

Data and output locations default to ``$HRRRCAST_DATA`` (else ``<repo>/data``)
and ``$HRRRCAST_ARTIFACTS/torch_ref/out`` (else ``<repo>/artifacts/...``);
override with --base-dir / --output-dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .config import ARTIFACTS_ROOT
from .inference import DEFAULT_SAMPLER, load_hrrrcast
from .output import convert_netcdf_hours_to_grib2, static_fields_from_npz, write_hour
from .variables import LEVELS, PL_VARS, SFC_VARS, channel_bounds
from .rollout import (
    autoregressive_rollout,
    build_initial_input,
    gfs_forcing_to_nchw,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA = Path(os.environ.get("HRRRCAST_DATA", _REPO_ROOT / "data"))
_DEFAULT_OUTPUT = ARTIFACTS_ROOT / "torch_ref" / "out"


def expand_member_spec(specs: Iterable[str]) -> list[int]:
    """Accept '0,1,2', '0-3', '0 1 2', and mixes thereof."""
    out: list[int] = []
    for spec in specs:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-")
                out.extend(range(int(lo), int(hi) + 1))
            else:
                out.append(int(part))
    return sorted(set(out))


def cycle_dir_from_init(init_time: str) -> Path:
    date, hour = init_time.split("T")
    return Path(date.replace("-", "")) / hour


def data_files(base_dir: Path, init_time: str) -> tuple[Path, Path]:
    date, hour = init_time.split("T")
    yyyymmdd = date.replace("-", "")
    cycle = base_dir / yyyymmdd / hour
    return cycle / f"hrrr_{yyyymmdd}_{hour}.npz", cycle / f"gfs_{yyyymmdd}_{hour}.npz"


# --- PMM nudging (REFC/APCP only) -----------------------------------------

def _maybe_nudge(
    hour_outputs: dict[int, torch.Tensor],
    *,
    alpha: float,
) -> dict[int, torch.Tensor]:
    """Blend each member's REFC/APCP fields toward the PMM mean.

    Returns a dict of (1, H, W, C) NHWC float32 tensors. Falls back to identity
    when scikit-image or the `src/compute_pmm.compute_PMM` is unavailable, or when there
    are fewer than two members.
    """
    if len(hour_outputs) < 2 or alpha >= 1.0:
        return hour_outputs

    try:
        import xarray as xr  # local import to avoid mandatory dependency at module import time
        from skimage.exposure import match_histograms

        from .config import add_src_to_path

        add_src_to_path()
        from compute_pmm import compute_PMM  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning("PMM nudging disabled: %s", exc)
        return hour_outputs

    num_pl = len(PL_VARS) * len(LEVELS)
    refc_idx = num_pl + SFC_VARS.index("REFC") if "REFC" in SFC_VARS else None
    apcp_idx = num_pl + SFC_VARS.index("APCP") if "APCP" in SFC_VARS else None
    channels = [c for c in (refc_idx, apcp_idx) if c is not None]
    if not channels:
        return hour_outputs

    members_sorted = sorted(hour_outputs.keys())
    stack = np.stack([hour_outputs[m].detach().cpu().numpy()[0] for m in members_sorted], axis=0)
    pmm_arrays: dict[int, np.ndarray] = {}
    for channel in channels:
        if channel >= stack.shape[-1]:
            continue
        da = xr.DataArray(
            np.transpose(stack[:, :, :, channel], (1, 2, 0)),
            dims=("latitude", "longitude", "member"),
            coords={"member": np.arange(len(members_sorted))},
        )
        pmm_arrays[channel] = compute_PMM(da, method=2).values

    nudged: dict[int, torch.Tensor] = {}
    for i, member in enumerate(members_sorted):
        arr = stack[i].copy()
        for channel, pmm_vals in pmm_arrays.items():
            member_channel = stack[i, :, :, channel]
            blended = alpha * member_channel + (1.0 - alpha) * pmm_vals
            arr[:, :, channel] = match_histograms(blended, member_channel, channel_axis=None)
        nudged[member] = torch.from_numpy(arr[None, ...]).to(device=hour_outputs[member].device, dtype=hour_outputs[member].dtype)
    return nudged


# --- main -----------------------------------------------------------------

def add_forecast_args(parser: argparse.ArgumentParser) -> None:
    # TF-compatible positionals so jobs/job-fcst.sh can drive this as a drop-in
    # for `src/fcst.py <model> <INIT> <LEAD> ...`. All optional; the --init-time
    # / --lead-hours flags still work for standalone use.
    parser.add_argument("model_path", nargs="?", default=None,
                        help="(TF-compat) model path; ignored unless it ends with .pt (then used as --state-path)")
    parser.add_argument("init_pos", nargs="?", default=None, help="(TF-compat) init time positional YYYY-MM-DDTHH")
    parser.add_argument("lead_pos", nargs="?", default=None, help="(TF-compat) lead hours positional")
    parser.add_argument("--init-time", dest="init_time", default="2024-05-06T23", help="YYYY-MM-DDTHH initialization time")
    parser.add_argument("--lead-hours", dest="lead_hours", type=int, default=1, help="Maximum lead time in hours")
    parser.add_argument("--members", nargs="+", default=["0"], help="Member ids, e.g. '0 1 2' or '0-3' or '0,1,2'")
    parser.add_argument("--num-members", "--num_members", dest="num_members", type=int, default=None, help="Ensemble size (defaults to len(--members))")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1, help="How many members to predict together each hour")
    parser.add_argument("--sampler", default=DEFAULT_SAMPLER, choices=["dpmpp", "ddim"], help="Reverse-diffusion sampler (dpmpp = upstream ea1b4ed)")
    parser.add_argument("--base-dir", "--base_dir", dest="base_dir", default=str(_DEFAULT_DATA))
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--state-path", "--state_path", dest="state_path", default=None, help="Override module state dict path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--pmm-alpha", "--pmm_alpha", dest="pmm_alpha", type=float, default=0.7, help="Blend factor toward PMM mean (1.0 disables nudging)")
    parser.add_argument("--noise-rho", "--noise_rho", dest="noise_rho", type=float, default=0.9, help="AR(1) member-noise correlation coefficient")
    parser.add_argument("--no-nudging", "--no_nudging", dest="no_nudging", action="store_true", help="Disable PMM nudging regardless of --pmm-alpha")
    parser.add_argument("--no-diffusion", "--no_diffusion", dest="no_diffusion", action="store_true", help="(TF-compat) accepted but unsupported; the PyTorch port always runs diffusion")
    parser.add_argument("--no-grib2", "--no_grib2", dest="no_grib2", action="store_true", help="Skip GRIB2 conversion (NetCDF only)")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model (fuses elementwise/LayerNorm, cuts launch overhead)")
    parser.add_argument(
        "--compile-mode", "--compile_mode", dest="compile_mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="torch.compile mode (only used with --compile)",
    )
    parser.add_argument("--log-level", "--log_level", dest="log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def run_forecast(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("fcst")

    # Resolve TF-compatible positionals (if given) over the named flags.
    init_time = args.init_pos if args.init_pos is not None else args.init_time
    lead_hours = int(args.lead_pos) if args.lead_pos is not None else args.lead_hours
    state_path = args.state_path
    if args.model_path and args.model_path.endswith(".pt"):
        state_path = args.model_path
    if args.no_diffusion:
        logger.warning("--no-diffusion is not supported by the PyTorch port; ignoring (diffusion always on).")

    members = expand_member_spec(args.members)
    if not members:
        raise SystemExit("No members requested")
    num_members = args.num_members or len(members)

    hrrr_path, gfs_path = data_files(Path(args.base_dir), init_time)
    if not hrrr_path.exists():
        raise SystemExit(f"HRRR npz missing: {hrrr_path}")
    if not gfs_path.exists():
        raise SystemExit(f"GFS npz missing: {gfs_path}")

    hrrr_npz = np.load(hrrr_path)
    gfs_npz = np.load(gfs_path)
    init_datetime = datetime.fromisoformat(str(hrrr_npz["init_datetime"]))
    lats = np.asarray(hrrr_npz["lats"])
    lons = np.asarray(hrrr_npz["lons"])
    # np.load's NpzFile re-reads/decompresses on every __getitem__, so pull the
    # large GFS forcing block out once and reuse it (shape, initial input, forcing).
    gfs_model_input = np.asarray(gfs_npz["model_input"])

    forcing_hours_available = int(gfs_model_input.shape[0])
    if lead_hours > forcing_hours_available:
        logger.warning(
            "Requested %d lead hours but GFS npz only has %d forcing hour(s); rollout will clip to the last available index.",
            lead_hours,
            forcing_hours_available,
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Loading HRRRCast model on %s%s", device, " (torch.compile)" if args.compile else "")
    model = load_hrrrcast(state_path, device=device, compile_model=args.compile, compile_mode=args.compile_mode)

    initial_input = build_initial_input(hrrr_npz, gfs_model_input, device=device)
    gfs_forcing = gfs_forcing_to_nchw(gfs_model_input, device=device)
    mins, maxs = channel_bounds(device=device)
    cycle_out = Path(args.output_dir) / cycle_dir_from_init(init_time)
    cycle_out.mkdir(parents=True, exist_ok=True)
    static_fields = static_fields_from_npz(hrrr_npz)

    # Buffer one hour's worth of member outputs so PMM can run on the full
    # ensemble before we hand off to the writer. Writes happen in a background
    # thread to overlap with the next hour's inference.
    io_executor = ThreadPoolExecutor(max_workers=1)
    io_futures: list = []
    nudging_enabled = (not args.no_nudging) and len(members) >= 2 and args.pmm_alpha < 1.0
    hour_buffer: dict[int, dict[int, torch.Tensor]] = {}

    def schedule_write(hour: int, member: int, normalized_nhwc: torch.Tensor) -> None:
        io_futures.append(
            io_executor.submit(
                write_hour,
                normalized=normalized_nhwc,
                hour=hour,
                static_fields=static_fields,
                init_datetime=init_datetime,
                lats=lats,
                lons=lons,
                output_dir=cycle_out,
                member=member,
            )
        )

    def flush_hour(hour: int) -> None:
        outputs = hour_buffer.pop(hour, {})
        if not outputs:
            return
        finalized = _maybe_nudge(outputs, alpha=args.pmm_alpha) if nudging_enabled else outputs
        for member, tensor in finalized.items():
            schedule_write(hour, member, tensor)

    def on_hour(hour: int, member: int, normalized_nhwc: torch.Tensor) -> None:
        hour_buffer.setdefault(hour, {})[member] = normalized_nhwc.detach().cpu()
        if len(hour_buffer[hour]) == len(members):
            flush_hour(hour)

    autoregressive_rollout(
        model,
        init_input=initial_input,
        gfs_forcing=gfs_forcing,
        members=members,
        num_members=num_members,
        lead_hours=lead_hours,
        init_datetime=init_datetime,
        channel_mins=mins,
        channel_maxs=maxs,
        batch_size=args.batch_size,
        on_hour=on_hour,
        sampler=args.sampler,
        noise_rho=args.noise_rho,
    )

    # Flush any partially filled hour buckets (shouldn't happen with correct
    # member tracking but is a safety net).
    for hour in sorted(hour_buffer):
        flush_hour(hour)

    written_nc: list[Path] = []
    for future in as_completed(io_futures):
        written_nc.append(future.result())
    io_executor.shutdown(wait=True)
    logger.info("Wrote %d NetCDF files under %s", len(written_nc), cycle_out)

    if not args.no_grib2:
        try:
            hours = list(range(0, lead_hours + 1))
            for member in members:
                convert_netcdf_hours_to_grib2(
                    init_time=init_time,
                    member=member,
                    hours=hours,
                    in_dir=cycle_out,
                    out_dir=cycle_out,
                )
            logger.info("Wrote GRIB2 for members %s hours %s", members, hours)
        except ModuleNotFoundError as exc:
            if exc.name in {"grib2io", "eccodes"}:
                logger.warning(
                    "grib2io / eccodes not available; skipping GRIB2 in-process. "
                    "Run scripts/convert_torch_netcdf_to_grib2.py to convert via the micromamba fallback."
                )
            else:
                raise

    print(
        {
            "init_time": init_time,
            "lead_hours": lead_hours,
            "members": members,
            "output_dir": str(cycle_out),
            "netcdf_count": len(written_nc),
            "device": str(device),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="HRRRCast PyTorch inference (forecast / validate)")
    sub = parser.add_subparsers(dest="command")
    add_forecast_args(sub.add_parser("forecast", help="Multi-member, multi-hour forecast (NetCDF + GRIB2)"))
    vp = sub.add_parser("validate", help="Parity check vs a TF reference dump")
    vp.add_argument("--dump", default=None, help="TF reference npz (defaults to $HRRRCAST_ARTIFACTS/tf_ref/...)")
    vp.add_argument("--state", default=None, help="Override module state dict path")
    vp.add_argument("--device", default=None)
    vp.add_argument("--sampler", default=DEFAULT_SAMPLER, choices=["dpmpp", "ddim"])
    vp.add_argument("--rollout-dump", default=None, help="TF rollout dump directory for validation-only noise replay")
    vp.add_argument("--base-dir", default=None, help="Data root for rollout replay")
    vp.add_argument("--init-time", default="2024-05-06T23")
    vp.add_argument("--lead-hours", type=int, default=6)
    vp.add_argument("--member", type=int, default=0)
    vp.add_argument("--out", default=None, help="Optional path to write the JSON report")
    args = parser.parse_args()

    if args.command == "validate":
        from .validate import run as run_validate

        kwargs = {
            "state": args.state,
            "device": args.device,
            "sampler": args.sampler,
            "out": args.out,
            "rollout_dump": args.rollout_dump,
            "base_dir": args.base_dir,
            "init_time": args.init_time,
            "lead_hours": args.lead_hours,
            "member": args.member,
        }
        if args.dump:
            kwargs["dump"] = args.dump
        report, ok = run_validate(**kwargs)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if ok else 1)

    if args.command == "forecast":
        run_forecast(args)
        return

    parser.print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
