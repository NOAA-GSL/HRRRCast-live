#!/usr/bin/env python3
"""Convert exported Keras weights to a clean `HRRRCast` state dict.

By default this reads the committed `net-diffusion/model.keras` archive,
extracts its `config.json` and `model.weights.h5`, and writes
`$HRRRCAST_ARTIFACTS/model_export/hrrrcast_module_state_dict.pt`, whose keys
match the `src_torch.model.HRRRCast` module hierarchy
(e.g. `hrrr_pre.blocks.0.conv1.weight`).

No intermediate Keras-named state dict is required.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from .config import DEFAULT_MODEL_EXPORT, DEFAULT_MODULE_STATE, REPO_ROOT
from .model import HRRRCast


SPATIAL_GROUPED_NAMES = {
    "encoder_0": [
        ["spatial_grouped_conv2d", "spatial_grouped_conv2d_1"],
        ["spatial_grouped_conv2d_2", "spatial_grouped_conv2d_3"],
    ],
    "decoder_2": [
        ["spatial_grouped_conv2d_4", "spatial_grouped_conv2d_5"],
        ["spatial_grouped_conv2d_6", "spatial_grouped_conv2d_7"],
    ],
}


STACK_SPECS: list[dict] = [
    {"module": "hrrr_pre", "keras_prefix": "hrrr_preprocessor", "n_blocks": 2},
    {"module": "gfs_pre", "keras_prefix": "gfs_preprocessor", "n_blocks": 2},
    {"module": "noised_pre", "keras_prefix": "hrrr_noised_preprocessor", "n_blocks": 2},
    {"module": "enc0", "keras_prefix": "encoder_0", "n_blocks": 2},
    {"module": "enc1", "keras_prefix": "encoder_1", "n_blocks": 3},
    {"module": "enc2", "keras_prefix": "encoder_2", "n_blocks": 4},
    {"module": "dec0", "keras_prefix": "decoder_0", "n_blocks": 4},
    {"module": "dec1", "keras_prefix": "decoder_1", "n_blocks": 3},
    {"module": "dec2", "keras_prefix": "decoder_2", "n_blocks": 2},
    {"module": "output_refine", "keras_prefix": "output_refine", "n_blocks": 1},
]


def keras_group_name(class_name: str, index: int) -> str:
    base = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", class_name)
    base = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", base).lower()
    base = base.replace("2_d", "2d")
    return base if index == 0 else f"{base}_{index}"


def assign_h5_paths(layers: list[dict], prefix: str, paths: dict[str, str]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for layer in layers:
        class_name = layer["class_name"]
        name = layer["config"]["name"]
        group = keras_group_name(class_name, counts[class_name])
        counts[class_name] += 1
        path = f"{prefix}/{group}"
        paths[name] = path
        if class_name == "RecomputeSubModel":
            assign_h5_paths(layer["config"]["submodel"]["config"]["layers"], f"{path}/submodel/layers", paths)


def build_keras_named_tensors(h5: h5py.File, layers: list[dict], paths: dict[str, str]) -> dict[str, torch.Tensor]:
    """Pull tensors out of the Keras HDF5 file keyed by Keras layer name + suffix."""
    native: dict[str, torch.Tensor] = {}
    for layer in layers:
        class_name = layer["class_name"]
        name = layer["config"]["name"]
        path = paths[name]
        if class_name == "Dense":
            native[f"{name}.weight"] = torch.from_numpy(np.asarray(h5[f"{path}/vars/0"])).t().contiguous()
            if f"{path}/vars/1" in h5:
                native[f"{name}.bias"] = torch.from_numpy(np.asarray(h5[f"{path}/vars/1"]))
        elif class_name == "Conv2D":
            native[f"{name}.weight"] = torch.from_numpy(np.asarray(h5[f"{path}/vars/0"])).permute(3, 2, 0, 1).contiguous()
            if f"{path}/vars/1" in h5:
                native[f"{name}.bias"] = torch.from_numpy(np.asarray(h5[f"{path}/vars/1"]))
        elif class_name == "SpatialGroupedConv2D":
            native[f"{name}.weight"] = torch.from_numpy(np.asarray(h5[f"{path}/conv/vars/0"])).permute(3, 2, 0, 1).contiguous()
        elif class_name == "LayerNormalization":
            group = f"{path}/vars"
            if group not in h5:
                continue
            keys = sorted(h5[group].keys(), key=int)
            offset = 0
            if layer["config"].get("scale", True):
                native[f"{name}.gamma"] = torch.from_numpy(np.asarray(h5[f"{group}/{keys[offset]}"]))
                offset += 1
            if layer["config"].get("center", True):
                native[f"{name}.beta"] = torch.from_numpy(np.asarray(h5[f"{group}/{keys[offset]}"]))
        elif class_name == "RecomputeSubModel":
            native.update(build_keras_named_tensors(h5, layer["config"]["submodel"]["config"]["layers"], paths))
    return native


def block_conv_key(keras_prefix: str, block_idx_1based: int, sub: str) -> str:
    if keras_prefix in SPATIAL_GROUPED_NAMES:
        return SPATIAL_GROUPED_NAMES[keras_prefix][block_idx_1based - 1][int(sub) - 1] + ".weight"
    return f"{keras_prefix}_res_{block_idx_1based}_{sub}_conv.weight"


def remap_block(source: dict[str, torch.Tensor], keras_prefix: str, block_idx_1based: int, module_path: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    src_block_prefix = f"{keras_prefix}_res_{block_idx_1based}"

    out[f"{module_path}.conv1.weight"] = source[block_conv_key(keras_prefix, block_idx_1based, "1")]
    out[f"{module_path}.conv2.weight"] = source[block_conv_key(keras_prefix, block_idx_1based, "2")]

    out[f"{module_path}.norm1.beta"] = source[f"{src_block_prefix}_1_lnorm.beta"]
    out[f"{module_path}.norm2.gamma"] = source[f"{src_block_prefix}_2_lnorm.gamma"]
    out[f"{module_path}.norm2.beta"] = source[f"{src_block_prefix}_2_lnorm.beta"]

    for film in ("film_gamma", "film_beta"):
        out[f"{module_path}.{film}.weight"] = source[f"{src_block_prefix}_{film}_dense.weight"]
        out[f"{module_path}.{film}.bias"] = source[f"{src_block_prefix}_{film}_dense.bias"]

    for mlp in ("cbam_ch_mlp_1", "cbam_ch_mlp_2"):
        out[f"{module_path}.{mlp}.weight"] = source[f"{src_block_prefix}_{mlp}.weight"]
        out[f"{module_path}.{mlp}.bias"] = source[f"{src_block_prefix}_{mlp}.bias"]

    out[f"{module_path}.cbam_sp_conv7.weight"] = source[f"{src_block_prefix}_cbam_sp_conv7.weight"]

    shortcut_key = f"{src_block_prefix}_0_conv.weight"
    if shortcut_key in source:
        out[f"{module_path}.shortcut.weight"] = source[shortcut_key]
    return out


def remap_processor(source: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for module_idx in range(14):
        block_idx_1based = module_idx + 1
        prefix = f"processor_{module_idx}_res_{block_idx_1based}"
        module_path = f"processor.blocks.{module_idx}"

        out[f"{module_path}.conv1.weight"] = source[f"{prefix}_1_conv.weight"]
        out[f"{module_path}.conv2.weight"] = source[f"{prefix}_2_conv.weight"]
        out[f"{module_path}.norm1.beta"] = source[f"{prefix}_1_lnorm.beta"]
        out[f"{module_path}.norm2.gamma"] = source[f"{prefix}_2_lnorm.gamma"]
        out[f"{module_path}.norm2.beta"] = source[f"{prefix}_2_lnorm.beta"]
        for film in ("film_gamma", "film_beta"):
            out[f"{module_path}.{film}.weight"] = source[f"{prefix}_{film}_dense.weight"]
            out[f"{module_path}.{film}.bias"] = source[f"{prefix}_{film}_dense.bias"]
        for mlp in ("cbam_ch_mlp_1", "cbam_ch_mlp_2"):
            out[f"{module_path}.{mlp}.weight"] = source[f"{prefix}_{mlp}.weight"]
            out[f"{module_path}.{mlp}.bias"] = source[f"{prefix}_{mlp}.bias"]
        out[f"{module_path}.cbam_sp_conv7.weight"] = source[f"{prefix}_cbam_sp_conv7.weight"]
    return out


def build_module_state(source: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}

    out["time_dense.weight"] = source["time_dense.weight"]
    out["time_dense.bias"] = source["time_dense.bias"]
    out["time_norm.gamma"] = source["time_bn_relu_lnorm.gamma"]
    out["time_norm.beta"] = source["time_bn_relu_lnorm.beta"]
    out["skip1.weight"] = source["skip_scale_dense_1.weight"]
    out["skip1.bias"] = source["skip_scale_dense_1.bias"]
    out["skip2.weight"] = source["skip_scale_dense_2.weight"]
    out["skip2.bias"] = source["skip_scale_dense_2.bias"]
    out["output_conv.weight"] = source["output_refine_output_conv.weight"]

    for spec in STACK_SPECS:
        for module_idx in range(spec["n_blocks"]):
            block_idx_1based = module_idx + 1
            module_path = f"{spec['module']}.blocks.{module_idx}"
            out.update(remap_block(source, spec["keras_prefix"], block_idx_1based, module_path))

    out.update(remap_processor(source))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keras", default=str(REPO_ROOT / "net-diffusion" / "model.keras"), help="Keras .keras archive containing config.json and model.weights.h5")
    parser.add_argument("--h5", default=None, help="Optional extracted model.weights.h5 override")
    parser.add_argument("--config", default=None, help="Optional extracted config.json override")
    parser.add_argument("--out", default=str(DEFAULT_MODULE_STATE))
    args = parser.parse_args()

    if bool(args.h5) != bool(args.config):
        raise SystemExit("--h5 and --config must be provided together")

    with tempfile.TemporaryDirectory() as tmp:
        if args.h5 and args.config:
            h5_path = Path(args.h5)
            config_path = Path(args.config)
        else:
            keras_path = Path(args.keras)
            if not keras_path.exists():
                raise SystemExit(f"Keras archive not found: {keras_path}")
            with zipfile.ZipFile(keras_path) as zf:
                zf.extract("config.json", tmp)
                zf.extract("model.weights.h5", tmp)
            h5_path = Path(tmp) / "model.weights.h5"
            config_path = Path(tmp) / "config.json"

        config_layers = json.loads(config_path.read_text())["config"]["layers"]
        paths: dict[str, str] = {}
        assign_h5_paths(config_layers, "layers", paths)

        with h5py.File(h5_path, "r") as h5:
            keras_named = build_keras_named_tensors(h5, config_layers, paths)

    module_state = build_module_state(keras_named)

    model = HRRRCast()
    expected = set(model.state_dict().keys())
    got = set(module_state.keys())
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing or extra:
        raise SystemExit({"missing": missing[:20], "missing_count": len(missing), "extra": extra[:20], "extra_count": len(extra)})
    for key in expected:
        if module_state[key].shape != model.state_dict()[key].shape:
            raise SystemExit(f"shape mismatch for {key}: source {tuple(module_state[key].shape)} vs module {tuple(model.state_dict()[key].shape)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module_state, out_path)
    print({
        "out": str(out_path),
        "tensors": len(module_state),
        "params": sum(t.numel() for t in module_state.values()),
    })


if __name__ == "__main__":
    main()
