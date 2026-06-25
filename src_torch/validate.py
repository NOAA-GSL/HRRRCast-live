#!/usr/bin/env python3
"""Parity harness: validate the PyTorch port against a TensorFlow reference dump.

Runs the two checks that gate the port and asserts both stay within acceptance
thresholds (normalized space):

  1. First model forward : ``HRRRCast(first_model_input)`` vs ``first_model_output``
  2. dpmpp-2m loop (f01)  : ``inference.forecast_hour`` vs ``final_normalized_f01``

Both checks exercise the production code path (`inference.load_hrrrcast` and the
shared `inference.forecast_hour` / `inference.diffusion_loop`), so a pass
certifies the same functions the CLI runs. No production I/O.

Usage:
    python -m src_torch.validate [--dump REF.npz] [--state STATE.pt] [--out report.json]
    python -m src_torch.cli validate ...        # equivalent
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from .config import DEFAULT_TENSOR_DUMP
from .inference import forecast_hour, load_hrrrcast
from .rollout import (
    build_initial_input,
    date_encoding_field,
    gfs_forcing_to_nchw,
    phase_angles,
)
from .variables import channel_bounds

# Acceptance thresholds (normalized space) for the frozen reference case.
FIRST_FORWARD_LIMITS = {"mean_abs": 2.0e-4, "rmse": 5.0e-4, "max_abs": 5.0e-2}
DIFFUSION_LOOP_LIMITS = {"mean_abs": 1.0e-3, "rmse": 2.0e-3, "p99_abs": 5.0e-3}
ROLLOUT_REPLAY_LIMITS = {"mean_abs": 1.5e-3, "rmse": 3.0e-3, "p99_abs": 1.0e-2}


def _metrics(diff: torch.Tensor) -> dict:
    ad = diff.abs()
    flat = ad.flatten()
    stride = max(flat.numel() // 5_000_000, 1)
    return {
        "max_abs": float(ad.max().item()),
        "mean_abs": float(ad.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "p99_abs": float(torch.quantile(flat[::stride], 0.99).item()),
    }


def _check(name: str, metrics: dict, limits: dict, failures: list[str]) -> None:
    for key, limit in limits.items():
        if metrics[key] > limit:
            failures.append(f"{name} {key} too high: {metrics[key]:.3e} > {limit:.1e}")


def _data_files(base_dir: Path, init_time: str) -> tuple[Path, Path]:
    date, hour = init_time.split("T")
    yyyymmdd = date.replace("-", "")
    cycle = base_dir / yyyymmdd / hour
    return cycle / f"hrrr_{yyyymmdd}_{hour}.npz", cycle / f"gfs_{yyyymmdd}_{hour}.npz"


def _load_tf_noise(path: Path, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    dump = np.load(path)
    key = "noise" if "noise" in dump.files else "member_noise"
    if key not in dump.files:
        raise ValueError(f"{path} has no 'noise' or 'member_noise' array")
    return torch.from_numpy(dump[key]).permute(0, 3, 1, 2).contiguous().to(device=device, dtype=dtype)


def run_rollout_replay(
    *,
    rollout_dump: str,
    state: str | None = None,
    device: str | None = None,
    sampler: str = "dpmpp",
    base_dir: str | None = None,
    init_time: str = "2024-05-06T23",
    lead_hours: int = 6,
    member: int = 0,
) -> dict:
    """Replay a TF rollout dump using TF per-hour noise and compare normalized outputs.

    This is a validation-only parity harness. It intentionally lives outside the
    production forecast CLI: production forecasts use native PyTorch AR(1) noise,
    while this path replays TF-dumped `tf_noise_mNN_fHH.npz` tensors to isolate
    cross-framework numerical differences.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dump_dir = Path(rollout_dump)
    data_root = Path(base_dir or os.environ.get("HRRRCAST_DATA", Path(__file__).resolve().parents[1] / "data"))
    hrrr_path, gfs_path = _data_files(data_root, init_time)
    hrrr_npz = np.load(hrrr_path)
    gfs_model_input = np.asarray(np.load(gfs_path)["model_input"])

    model = load_hrrrcast(state, device=dev)
    init_input = build_initial_input(hrrr_npz, gfs_model_input, device=dev)
    gfs_forcing = gfs_forcing_to_nchw(gfs_model_input, device=dev)
    mins, maxs = channel_bounds(device=dev)

    init_datetime = datetime.fromisoformat(str(hrrr_npz["init_datetime"]))
    predicted_channels = init_input[:, :138].shape[1]
    gfs_channels = gfs_forcing.shape[1]
    height, width = init_input.shape[2], init_input.shape[3]
    dtype = init_input.dtype
    start_pred_noise = predicted_channels + gfs_channels
    phase_lookup = phase_angles(member + 1)
    forcing_count = gfs_forcing.shape[0]

    state_from_hour = init_input[:, :predicted_channels].clone()
    report: dict = {
        "dump": str(dump_dir),
        "base_dir": str(data_root),
        "init_time": init_time,
        "lead_hours": lead_hours,
        "member": member,
        "sampler": sampler,
        "device": str(dev),
        "hours": {},
    }

    failures: list[str] = []

    def compare_hour(hour: int, y_nhwc: torch.Tensor) -> None:
        target_path = dump_dir / f"tf_norm_m{member:02d}_f{hour:02d}.npz"
        if not target_path.exists():
            return
        target = torch.from_numpy(np.load(target_path)["normalized"]).to(device=dev, dtype=y_nhwc.dtype)
        m = _metrics(y_nhwc - target)
        report["hours"][f"f{hour:02d}"] = m
        _check(f"rollout_replay_f{hour:02d}", m, ROLLOUT_REPLAY_LIMITS, failures)

    compare_hour(0, state_from_hour.permute(0, 2, 3, 1).contiguous())

    for hour in range(1, lead_hours + 1):
        from_hour = ((hour - 1) // 6) * 6
        step = hour - from_hour
        date_enc = date_encoding_field(init_datetime, hour, height, width, dtype=dtype, device=dev)
        lead_enc = torch.full((1, 1, height, width), step / 6.0, dtype=dtype, device=dev)
        x_base = torch.cat([init_input[:, start_pred_noise:-8], date_enc, init_input[:, -2:-1], lead_enc], dim=1)

        phase_width = from_hour // 12
        phase_shift = round(phase_width * phase_lookup[member])
        forcing_idx = int(np.clip(hour - 1 + phase_shift, 0, forcing_count - 1))
        x_member = torch.cat(
            [
                state_from_hour,
                gfs_forcing[forcing_idx : forcing_idx + 1],
                x_base,
            ],
            dim=1,
        )
        noise_path = dump_dir / f"tf_noise_m{member:02d}_f{hour:02d}.npz"
        if not noise_path.exists():
            raise FileNotFoundError(noise_path)
        xn = _load_tf_noise(noise_path, device=dev, dtype=dtype)
        y_nhwc = forecast_hour(model, x_member, xn, mins, maxs, sampler=sampler)
        if hour % 6 == 0:
            state_from_hour = y_nhwc.permute(0, 3, 1, 2).contiguous()
        compare_hour(hour, y_nhwc)

    report["status"] = "fail" if failures else "pass"
    report["failures"] = failures
    return report


def run(
    *,
    dump: str = str(DEFAULT_TENSOR_DUMP),
    state: str | None = None,
    device: str | None = None,
    sampler: str = "dpmpp",
    out: str | None = None,
    rollout_dump: str | None = None,
    base_dir: str | None = None,
    init_time: str = "2024-05-06T23",
    lead_hours: int = 6,
    member: int = 0,
) -> tuple[dict, bool]:
    """Run both parity checks; return (report, ok)."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tensors = np.load(dump)
    model = load_hrrrcast(state, device=dev)

    failures: list[str] = []
    report: dict = {"device": str(dev), "dump": str(dump), "sampler": sampler}

    # 1. first model forward
    x = torch.from_numpy(tensors["first_model_input"]).permute(0, 3, 1, 2).contiguous().to(dev)
    ff_target = torch.from_numpy(tensors["first_model_output"]).to(dev)
    with torch.no_grad():
        ff = model(x).permute(0, 2, 3, 1).contiguous()
    report["first_forward"] = _metrics(ff - ff_target)
    _check("first_forward", report["first_forward"], FIRST_FORWARD_LIMITS, failures)

    # 2. diffusion loop (f01) via the shared production per-hour function
    x_batch = torch.from_numpy(tensors["x_batch"]).permute(0, 3, 1, 2).contiguous().to(dev)
    xn = torch.from_numpy(tensors["member_noise"]).permute(0, 3, 1, 2).contiguous().to(dev)
    loop_target = torch.from_numpy(tensors["final_normalized_f01"]).to(dev)
    mins, maxs = channel_bounds(device=dev)
    out_nhwc = forecast_hour(model, x_batch, xn, mins, maxs, sampler=sampler)
    report["diffusion_loop"] = _metrics(out_nhwc - loop_target)
    _check("diffusion_loop", report["diffusion_loop"], DIFFUSION_LOOP_LIMITS, failures)

    if rollout_dump:
        replay_report = run_rollout_replay(
            rollout_dump=rollout_dump,
            state=state,
            device=str(dev),
            sampler=sampler,
            base_dir=base_dir,
            init_time=init_time,
            lead_hours=lead_hours,
            member=member,
        )
        report["rollout_replay"] = replay_report
        failures.extend(f"rollout_replay {f}" for f in replay_report.get("failures", []))

    report["status"] = "fail" if failures else "pass"
    report["failures"] = failures
    if out:
        from pathlib import Path

        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2) + "\n")
    return report, not failures


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate PyTorch HRRRCast against a TF reference dump.")
    ap.add_argument("--dump", default=str(DEFAULT_TENSOR_DUMP))
    ap.add_argument("--state", default=None, help="Override module state dict path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sampler", default="dpmpp", choices=["dpmpp", "ddim"])
    ap.add_argument("--rollout-dump", default=None, help="TF rollout dump directory with tf_noise_*.npz and tf_norm_*.npz for validation-only replay")
    ap.add_argument("--base-dir", default=None, help="Data root containing preprocessed HRRR/GFS npz files (defaults to $HRRRCAST_DATA or repo/data)")
    ap.add_argument("--init-time", default="2024-05-06T23")
    ap.add_argument("--lead-hours", type=int, default=6)
    ap.add_argument("--member", type=int, default=0)
    ap.add_argument("--out", default=None, help="Optional path to write the JSON report")
    args = ap.parse_args()

    report, ok = run(
        dump=args.dump,
        state=args.state,
        device=args.device,
        sampler=args.sampler,
        out=args.out,
        rollout_dump=args.rollout_dump,
        base_dir=args.base_dir,
        init_time=args.init_time,
        lead_hours=args.lead_hours,
        member=args.member,
    )
    print(json.dumps(report, indent=2))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
