"""Repository paths, artifact locations, and the `src/` utilities import shim."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Root for generated artifacts (converted weights, TF reference dumps, outputs).
# Override with the HRRRCAST_ARTIFACTS environment variable; defaults to
# `<repo>/artifacts`.
ARTIFACTS_ROOT = Path(os.environ.get("HRRRCAST_ARTIFACTS", REPO_ROOT / "artifacts"))

DEFAULT_MODEL_EXPORT = ARTIFACTS_ROOT / "model_export"
DEFAULT_MODULE_STATE = Path(os.environ.get("HRRRCAST_MODULE_STATE", DEFAULT_MODEL_EXPORT / "hrrrcast_module_state_dict.pt"))
DEFAULT_TENSOR_DUMP = ARTIFACTS_ROOT / "tf_ref" / "tensor_dump_ref_mem0.npz"


def add_src_to_path() -> Path:
    """Add the model's `src/` to sys.path so its shared post-processing utilities
    (`diagnostics.compute_diagnostics`, `compute_pmm.compute_PMM`,
    `nc2grib.Netcdf2Grib`) can be imported and reused by the port."""
    src_dir = REPO_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return src_dir
