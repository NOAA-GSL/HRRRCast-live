"""Optional benchmark/profiling helpers for HRRRCast inference.

Instrumentation is disabled by default. Enable with:
  HRRRCAST_PROFILE=1        # timing + CUDA sync + NVTX
  HRRRCAST_PROFILE_TIMING=1 # timing logs only
  HRRRCAST_NVTX=1           # NVTX ranges only
  HRRRCAST_SYNC_TIMING=1    # synchronize CUDA around timed regions
  HRRRCAST_PROFILE_DETAIL=1 # include nested per-forward/per-op regions
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


_TRUE = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "") in _TRUE


def timing_enabled() -> bool:
    return _env_enabled("HRRRCAST_PROFILE") or _env_enabled("HRRRCAST_PROFILE_TIMING")


def nvtx_enabled() -> bool:
    return torch.cuda.is_available() and (_env_enabled("HRRRCAST_PROFILE") or _env_enabled("HRRRCAST_NVTX"))


def sync_enabled() -> bool:
    return torch.cuda.is_available() and (_env_enabled("HRRRCAST_PROFILE") or _env_enabled("HRRRCAST_SYNC_TIMING"))


def detail_enabled() -> bool:
    return _env_enabled("HRRRCAST_PROFILE_DETAIL")


def synchronize_if_enabled() -> None:
    if sync_enabled():
        torch.cuda.synchronize()


@dataclass
class ProfileRecord:
    elapsed: float = 0.0


@contextmanager
def profile_region(
    name: str,
    *,
    logger: logging.Logger | None = None,
    extra: str = "",
    detail: bool = False,
) -> Iterator[ProfileRecord]:
    """Optionally time and/or NVTX-mark a source-code region."""
    if detail and not detail_enabled():
        yield ProfileRecord()
        return

    do_timing = timing_enabled()
    do_nvtx = nvtx_enabled()
    record = ProfileRecord()
    if not (do_timing or do_nvtx or sync_enabled()):
        yield record
        return

    synchronize_if_enabled()
    if do_nvtx:
        torch.cuda.nvtx.range_push(name)
    t0 = time.perf_counter()
    try:
        yield record
    finally:
        synchronize_if_enabled()
        record.elapsed = time.perf_counter() - t0
        if do_nvtx:
            torch.cuda.nvtx.range_pop()
        if do_timing:
            log = logger or logging.getLogger(__name__)
            log.info("BENCH phase=%s%s seconds=%.6f", name, extra, record.elapsed)
