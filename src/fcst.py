#!/usr/bin/env python3
"""
Weather Forecast Runner Script

This script loads preprocessed GRIB data and runs the neural network forecast model.
This stage is GPU-intensive and handles the autoregressive model inference.

Usage:
    python run_forecast.py <model_path> <preprocessed_data> <lead_hours> [--output_dir DIR]
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tensorflow as tf
import xarray as xr
import pandas as pd
from skimage.exposure import match_histograms

from nc2grib import Netcdf2Grib

# Import custom modules (assuming they exist)
try:
    import resnet
except ImportError as e:
    logging.warning(f"Could not import custom modules: {e}")

from diffusion_params import (
    NUM_DIFFUSION_STEPS,
    NUM_INFERENCE_STEPS,
    INFERENCE_STEPS,
    compute_epsilon,
    ddpm,
    ddim,
)
from transform import (
    inverse_log_transform_array,
    inverse_neg_log_transform_array,
)
import utils
from utils import setup_logging
from diagnostics import compute_diagnostics
from compute_pmm import compute_PMM

logger = None


class PreprocessedDataLoader:
    """Handles loading and validation of preprocessed data."""
    
    def __init__(self, preprocessed_file: str):
        self.preprocessed_file = preprocessed_file
        self.data = None
        self.metadata = None
        self._load_data()
    
    def _load_data(self):
        """Load preprocessed data from file."""
        if not os.path.exists(self.preprocessed_file):
            raise FileNotFoundError(f"Preprocessed data file not found: {self.preprocessed_file}")
        
        try:
            logger.info(f"Loading preprocessed data from {self.preprocessed_file}")
            self.data = np.load(self.preprocessed_file)
            
            # Extract metadata
            self.metadata = {
                'init_year': str(self.data['init_year']),
                'init_month': str(self.data['init_month']),
                'init_day': str(self.data['init_day']),
                'init_hh': str(self.data['init_hh']),
                'init_datetime': str(self.data['init_datetime']),
                'pl_vars': self.data['pl_vars'].tolist(),
                'sfc_vars': self.data['sfc_vars'].tolist(),
                'levels': self.data['levels'].tolist(),
                'grid_height': int(self.data['grid_height']),
                'grid_width': int(self.data['grid_width']),
                'downsample_factor': int(self.data['downsample_factor']),
                'norm_file': str(self.data['norm_file'])
            }
            
            logger.info("Preprocessed data loaded successfully")
            logger.info(f"Model input shape: {self.data['model_input'].shape}")
            logger.info(f"Initialization time: {self.metadata['init_datetime']}")
            
        except Exception as e:
            logger.error(f"Error loading preprocessed data: {e}")
            raise
    
    def get_model_input(self) -> np.ndarray:
        """Get the model input array."""
        return self.data['model_input']
    
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get latitude and longitude arrays."""
        return self.data['lats'], self.data['lons']
    
    def get_init_datetime(self) -> datetime:
        """Get initialization datetime."""
        return datetime.fromisoformat(self.metadata['init_datetime'])


class ForecastModel:
    """Handles model loading and inference."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._setup_tf_environment()
        self._load_model()

    def _setup_tf_environment(self) -> None:
        """ 
        Set up the TensorFlow environment for optimal performance.
        """
        # use only 1 gpu
        num_gpus = 1
        # Improved CPU/GPU device handling
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            logger.info(f"Num GPUs available: {len(gpus)}")
            tf.config.set_visible_devices(gpus[:num_gpus], "GPU")
            visible_gpus = tf.config.get_visible_devices("GPU")
            logger.info(f"Using GPUs: {[gpu.name for gpu in visible_gpus]}")
            for gpu in tf.config.get_visible_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory growth set for all visible GPUs.")
        else:
            tf.config.set_visible_devices([], "GPU")
            logger.warning("No GPUs used, running on CPU only.")

        # set JIT compilation of graphs
        tf.config.optimizer.set_jit(True)
    
    def _load_model(self):
        """Load the TensorFlow model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = tf.keras.models.load_model(
                self.model_path, 
                safe_mode=False, 
                compile=False
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Make prediction using the loaded model."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            return self.model(input_data, training=False)
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            raise


class WeatherForecaster:
    """Handles the forecasting pipeline."""

    def __init__(
        self,
        data_loader_hrrr: PreprocessedDataLoader,
        data_loader_gfs: PreprocessedDataLoader,
        num_members: int,
        members: List[int],
        batch_size: int,
        use_diffusion: bool,
        predicted_channels: Optional[int] = None,
        gfs_channels: Optional[int] = None,
        static_channels: Optional[int] = None,
        pmm_alpha: float = 0.65,
        use_nudging: bool = True,
    ):
        self.data_loader_hrrr = data_loader_hrrr
        self.data_loader_gfs = data_loader_gfs
        self.metadata = data_loader_hrrr.metadata
        self.num_members = num_members
        self.members = members
        self.batch_size = batch_size
        self.use_diffusion = use_diffusion
        self.pmm_alpha = pmm_alpha
        self.use_nudging = use_nudging and len(members) > 1

        # log-transform variables list
        self.LOG_TRANSFORM_VARS = [
            "VIS",
            "APCP",
            "HGTCC",
            "CAPE",
        ]
        self.NEG_LOG_TRANSFORM_VARS = [
            "CIN",
        ]

        # Derive dynamic channel counts if not provided
        pl_vars = self.metadata["pl_vars"]
        sfc_vars = self.metadata["sfc_vars"]
        levels = self.metadata["levels"]
        default_predicted = len(pl_vars) * len(levels) + len(sfc_vars)
        self.input_shape = data_loader_hrrr.get_model_input().shape
        hrrr_channels = self.input_shape[-1]
        nlat, nlon = self.input_shape[1], self.input_shape[2] 

        if predicted_channels is None:
            predicted_channels = default_predicted
        if gfs_channels is None:
            gfs_channels = data_loader_gfs.get_model_input().shape[-1]
        if static_channels is None:
            static_channels = max(hrrr_channels - predicted_channels, 0)

        self.predicted_channels = predicted_channels
        self.gfs_channels = gfs_channels
        self.static_channels = static_channels

        # Load normalization file and construct per-channel mean/std vectors consistent with preprocessing
        norm_file = self.metadata["norm_file"]
        try:
            ds_norm = xr.open_dataset(norm_file)
            self._init_channel_stats(ds_norm)
            ds_norm.close()
            logger.info(
                f"Normalization file loaded and channel stats constructed: {norm_file}"
            )
        except Exception as e:
            logger.error(f"Error loading/processing normalization file: {e}")
            raise

        # set member noise for all members
        self.member_noise = {}
        for member in self.members:
            if self.use_diffusion:
                noise = tf.random.stateless_normal(
                    shape=(nlat, nlon, self.predicted_channels),
                    dtype=tf.float32,
                    seed=[member, 0]
                )
                noise = tf.expand_dims(noise, axis=0)
            else:
                noise = tf.random.stateless_uniform(
                    shape=(),
                    minval=0.0,
                    maxval=1.0,
                    dtype=tf.float32,
                    seed=[member, 0]
                )
                noise = tf.tile(
                    tf.reshape(noise, (1, 1, 1, 1)),
                    [1, nlat, nlon, 1]
                )
            self.member_noise[member] = noise

    def _compute_pmm_mean(self, member_outputs: Dict[int, np.ndarray], method: int = 2) -> Tuple[np.ndarray, List[int]]:
        """Compute PMM mean for REFC and APCP channels only.

        Args:
            member_outputs: Dict mapping member -> array shape (1, H, W, C) or (H, W, C)
            method: PMM method (1 or 2)

        Returns:
            Tuple of (pmm_values, channel_indices)
            - pmm_values: shape (H, W, num_pmm_channels) containing only REFC/APCP
            - channel_indices: list of channel indices for REFC/APCP
        """
        if not member_outputs:
            raise ValueError("member_outputs is empty; cannot compute PMM")

        # Identify REFC and APCP channel indices
        pl_vars = self.metadata['pl_vars']
        sfc_vars = self.metadata['sfc_vars']
        levels = self.metadata['levels']
        num_pl_channels = len(pl_vars) * len(levels)
        
        pmm_channels = []
        for var_name in ['REFC', 'APCP']:
            if var_name in sfc_vars:
                sfc_idx = sfc_vars.index(var_name)
                channel_idx = num_pl_channels + sfc_idx
                pmm_channels.append(channel_idx)
        
        if not pmm_channels:
            # No REFC/APCP channels
            members_sorted = sorted(member_outputs.keys())
            first_arr = member_outputs[members_sorted[0]]
            if first_arr.ndim == 4:
                first_arr = first_arr[0]
            ny, nx = first_arr.shape[:2]
            return np.empty((ny, nx, 0), dtype=first_arr.dtype), []
        
        # Stack members for PMM computation
        members_sorted = sorted(member_outputs.keys())
        stack_list = []
        for m in members_sorted:
            arr = member_outputs[m]
            if arr.ndim == 4:
                arr = arr[0]
            stack_list.append(arr)
        stack = np.stack(stack_list, axis=0)
        m_count = stack.shape[0]
        ny, nx, nchan = stack.shape[1:]
        
        # Compute PMM only for REFC and APCP channels
        pmm_results = []
        valid_channels = []
        for c in pmm_channels:
            if c < nchan:
                channel_stack = np.transpose(stack[:, :, :, c], (1, 2, 0))
                da = xr.DataArray(
                    channel_stack,
                    dims=("latitude", "longitude", "member"),
                    coords={
                        "member": np.arange(m_count),
                    },
                )
                pmm_da = compute_PMM(da, method=method)
                pmm_results.append(pmm_da.values)
                valid_channels.append(c)
        
        if not pmm_results:
            return np.empty((ny, nx, 0), dtype=stack.dtype), []
        
        # Stack PMM results (H, W, num_pmm_channels)
        pmm_values = np.stack(pmm_results, axis=-1)
        logger.debug(f"Computed PMM for channels {valid_channels} (REFC/APCP)")
        return pmm_values, valid_channels

    def _nudge_members_toward_pmm(
        self,
        member_outputs: Dict[int, np.ndarray],
        pmm_values: np.ndarray,
        pmm_channels: List[int],
        alpha: float,
    ) -> Dict[int, np.ndarray]:
        """Nudge each member towards PMM mean with histogram matching.

        Only applies nudging to REFC and APCP channels; other channels remain unchanged.
        Blends: blended = alpha * member + (1 - alpha) * PMM
        Then applies histogram matching: nudged = match_histograms(blended, member)
        
        Args:
            member_outputs: Dict mapping member -> array shape (1, H, W, C) or (H, W, C)
            pmm_values: PMM values shape (H, W, num_pmm_channels) for REFC/APCP only
            pmm_channels: List of channel indices corresponding to pmm_values
            alpha: Blending factor
        """
        if not pmm_channels or pmm_values.shape[-1] == 0:
            # No nudging needed, return original outputs
            return member_outputs
        
        nudged_outputs: Dict[int, np.ndarray] = {}
        for member, arr in member_outputs.items():
            if arr.ndim == 4:
                arr2 = arr[0]
            else:
                arr2 = arr
            
            # Start with original member data
            nudged = arr2.copy()
            
            # Apply nudging only to REFC and APCP channels
            for i, c in enumerate(pmm_channels):
                if c < arr2.shape[-1] and i < pmm_values.shape[-1]:
                    # Extract single channel
                    member_channel = arr2[:, :, c]
                    pmm_channel = pmm_values[:, :, i]
                    
                    # Blend with PMM
                    blended_channel = alpha * member_channel + (1.0 - alpha) * pmm_channel
                    
                    # Apply histogram matching to preserve member's distribution
                    nudged_channel = match_histograms(blended_channel, member_channel, channel_axis=None)
                    
                    # Replace channel in output
                    nudged[:, :, c] = nudged_channel
            
            nudged_outputs[member] = nudged[None, ...]
        
        logger.debug(f"Applied nudging to channels {pmm_channels} (REFC/APCP)")
        return nudged_outputs



    def _init_channel_stats(self, ds_norm: xr.Dataset):
        """Build flattened mean/std vectors matching channel ordering in preprocessing.

        Ordering used in make_ics preprocessing:
          1. Pressure-level vars in the order (UGRD, VGRD, VVEL, TMP, HGT, SPFH) for each level.
          2. Surface vars in the order stored in metadata['sfc_vars'] (no constants).
        Constants (e.g., LAND, OROG) were appended in preprocessing but are not predicted
        by the diffusion / deterministic heads (first 74 channels). We still include them
        at the tail of the vectors if present so slicing remains safe.
        """
        pl_vars = self.metadata['pl_vars']
        sfc_vars = self.metadata['sfc_vars']
        levels = self.metadata['levels']

        fallback_mins_raw, fallback_maxs_raw = self.get_variable_bounds()

        raw_means: List[float] = []
        raw_stds: List[float] = []
        raw_mins: List[float] = []
        raw_maxs: List[float] = []
        channel_idx = 0

        # Pressure-level variables
        for var in pl_vars:
            if var not in ds_norm.variables:
                # Fallback: fill with zeros/ones to avoid crash
                logger.warning(f"Normalization stats missing for pressure var {var}; using mean=0,std=1")
                for _ in levels:
                    raw_means.append(0.0)
                    raw_stds.append(1.0)
                    raw_mins.append(float(fallback_mins_raw[channel_idx]))
                    raw_maxs.append(float(fallback_maxs_raw[channel_idx]))
                    channel_idx += 1
                continue

            stats = ds_norm[var].values  # shape (stat, level)
            # Safeguard shape
            if stats.shape[0] < 2:
                logger.warning(f"Stats for {var} malformed; using zeros/ones")
                for _ in levels:
                    raw_means.append(0.0)
                    raw_stds.append(1.0)
                    raw_mins.append(float(fallback_mins_raw[channel_idx]))
                    raw_maxs.append(float(fallback_maxs_raw[channel_idx]))
                    channel_idx += 1
                continue

            # If level dimension differs, broadcast or truncate
            nlev_stats = stats.shape[1] if stats.ndim > 1 else 1
            for i, lvl in enumerate(levels):
                if i < nlev_stats:
                    stat_mean = float(stats[0, i])
                    stat_std = float(stats[1, i]) if float(stats[1, i]) != 0 else 1.0
                    stat_min = float(stats[2, i]) if stats.shape[0] > 2 and i < nlev_stats else float(fallback_mins_raw[channel_idx])
                    stat_max = float(stats[3, i]) if stats.shape[0] > 3 and i < nlev_stats else float(fallback_maxs_raw[channel_idx])
                    if np.isnan(stat_min):
                        stat_min = float(fallback_mins_raw[channel_idx])
                    if np.isnan(stat_max):
                        stat_max = float(fallback_maxs_raw[channel_idx])
                else:
                    stat_mean = 0.0
                    stat_std = 1.0
                    stat_min = float(fallback_mins_raw[channel_idx])
                    stat_max = float(fallback_maxs_raw[channel_idx])

                raw_means.append(stat_mean)
                raw_stds.append(stat_std)
                raw_mins.append(stat_min)
                raw_maxs.append(stat_max)
                channel_idx += 1

        # Surface variables (single value per variable)
        for var in sfc_vars:
            if var not in ds_norm.variables:
                logger.warning(f"Normalization stats missing for surface var {var}; using mean=0,std=1")
                raw_means.append(0.0)
                raw_stds.append(1.0)
                raw_mins.append(float(fallback_mins_raw[channel_idx]))
                raw_maxs.append(float(fallback_maxs_raw[channel_idx]))
                channel_idx += 1
                continue

            stats = ds_norm[var].values  # expect (stat, ...)
            if stats.shape[0] < 2:
                logger.warning(f"Stats for {var} malformed; using mean=0,std=1")
                raw_means.append(0.0)
                raw_stds.append(1.0)
                raw_mins.append(float(fallback_mins_raw[channel_idx]))
                raw_maxs.append(float(fallback_maxs_raw[channel_idx]))
                channel_idx += 1
                continue

            stat_mean = float(np.nanmean(stats[0]))
            stat_std = float(np.nanmean(stats[1])) if np.nanmean(stats[1]) != 0 else 1.0
            stat_min = float(np.nanmean(stats[2])) if stats.shape[0] > 2 else float(fallback_mins_raw[channel_idx])
            stat_max = float(np.nanmean(stats[3])) if stats.shape[0] > 3 else float(fallback_maxs_raw[channel_idx])
            if np.isnan(stat_min):
                stat_min = float(fallback_mins_raw[channel_idx])
            if np.isnan(stat_max):
                stat_max = float(fallback_maxs_raw[channel_idx])

            raw_means.append(stat_mean)
            raw_stds.append(stat_std)
            raw_mins.append(stat_min)
            raw_maxs.append(stat_max)
            channel_idx += 1

        self.raw_means = np.array(raw_means, dtype=np.float32)
        self.raw_stds = np.array(raw_stds, dtype=np.float32)
        self.raw_mins = np.array(raw_mins, dtype=np.float32)
        self.raw_maxs = np.array(raw_maxs, dtype=np.float32)

        self.channel_means = self.raw_means
        self.channel_stds = self.raw_stds
        self.channel_mins = (self.raw_mins - self.channel_means) / self.channel_stds
        self.channel_maxs = (self.raw_maxs - self.channel_means) / self.channel_stds

    def denormalize(self, output: np.ndarray) -> np.ndarray:
        """Convert model output back to physical units using stored per-channel stats.

        output: shape (1, H, W, C_out) or (H,W,C_out). We slice stats to C_out.
        """
        try:
            if output.ndim == 3:
                output = output[None, ...]
            C_out = output.shape[-1]
            means = self.channel_means[:C_out][None, None, None, :]
            stds = self.channel_stds[:C_out][None, None, None, :]
            return np.squeeze(output * stds + means)
        except Exception as e:
            logger.error(f"Error in denormalization: {e}")
            raise

    def _apply_inverse_transforms(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply inverse transforms to variables stored in log/signed-log space.

        Returns the same dataset instance with values modified in place for the affected
        variables. Safe to call when variables are absent.
        """
        try:
            applied = []
            for var in self.LOG_TRANSFORM_VARS:
                if var in ds.variables:
                    data_arr = ds[var].values
                    ds[var].values[:] = inverse_log_transform_array(data_arr)
                    applied.append(var)
            for var in self.NEG_LOG_TRANSFORM_VARS:
                if var in ds.variables:
                    data_arr = ds[var].values
                    ds[var].values[:] = inverse_neg_log_transform_array(data_arr)
                    applied.append(var)
            if applied:
                logger.info(f"Applied inverse transforms to: {', '.join(applied)}")
        except Exception as e:
            logger.error(f"Failed applying inverse transforms: {e}")
        return ds

    def build_single_hour_dataset(
        self,
        init_datetime: datetime,
        hour: int,
        lats: np.ndarray,
        lons: np.ndarray,
        forecast_norm: np.ndarray,
    ) -> xr.Dataset:
        """Build an xarray.Dataset for a single lead hour from a normalized forecast slice.

        Args:
            init_datetime: initialization datetime
            hour: lead time in hours (int)
            lats, lons: 2D latitude/longitude arrays (Ny, Nx)
            forecast_norm: normalized model output for this hour, shape (1, Ny, Nx, C)

        Returns:
            xr.Dataset with dims (time=1, lead_time=1, [level], latitude, longitude)
        """
        # Denormalize to physical units
        denorm = self.denormalize(forecast_norm)
        # Ensure shape (time=1, Ny, Nx, C)
        if denorm.ndim == 3:
            denorm = denorm[None, ...]
        times = [hour]
        ds_hour = self.create_xarray_dataset(init_datetime, times, lats, lons, denorm)

        # Inject constants if present in preprocessed NPZ (repeat across lead_time length 1)
        for cname in ["LAND", "OROG"]:
            raw_key = f"{cname}_raw"
            if hasattr(self.data_loader_hrrr, "data") and raw_key in self.data_loader_hrrr.data.files and cname not in ds_hour:
                cvals = self.data_loader_hrrr.data[raw_key].astype(np.float32)
                const_4d = np.tile(cvals[None, None, :, :], (1, len(times), 1, 1))
                ds_hour[cname] = xr.DataArray(
                    const_4d,
                    dims=("time", "lead_time", "latitude", "longitude"),
                    coords={
                        "time": [init_datetime],
                        "lead_time": ("lead_time", times, {"units": "hours"}),
                        "latitude": (("latitude", "longitude"), lats),
                        "longitude": (("latitude", "longitude"), lons),
                    },
                    name=cname,
                )
                logger.debug(f"Injected constant field {cname} for hour {hour}")

        # Apply inverse transforms to recover physical units
        ds_hour = self._apply_inverse_transforms(ds_hour)

        # compute diagnostics
        ds_hour = compute_diagnostics(ds_hour)

        return ds_hour

    def write_single_hour_netcdf(
        self,
        init_datetime: datetime,
        hour: int,
        ds_hour: xr.Dataset,
        output_dir: str,
        member: Union[int, str],
    ) -> str:
        """Write a NetCDF file for a single lead time.

        Returns the output file path.
        """
        init_year = self.metadata['init_year']
        init_month = self.metadata['init_month']
        init_day = self.metadata['init_day']
        init_hh = self.metadata['init_hh']
        date_str = f"{init_year}{init_month}{init_day}/{init_hh}"
        utils.make_directory(f"{output_dir}/{date_str}")
        outdir = Path(f"{output_dir}/{date_str}")
        outdir.mkdir(parents=True, exist_ok=True)
        mem_str = f"avg" if str(member) == "avg" else f"mem{int(member)}"
        nc_path = outdir / f"hrrrcast_{mem_str}_f{hour:02d}.nc"
        logger.info(f"Saving single-hour NetCDF to {nc_path}")
        ds_hour.to_netcdf(nc_path)
        return str(nc_path)

    def write_single_hour_grib2(
        self,
        init_datetime: datetime,
        hour: int,
        ds_hour: xr.Dataset,
        output_dir: str,
        member: Union[int, str],
    ) -> None:
        """Write a GRIB2 file for a single lead time using Netcdf2Grib.

        Netcdf2Grib iterates over available time points; with a single-hour dataset,
        it will produce only the requested f{hour:02d} product.
        """
        init_year = self.metadata['init_year']
        init_month = self.metadata['init_month']
        init_day = self.metadata['init_day']
        init_hh = self.metadata['init_hh']
        date_str = f"{init_year}{init_month}{init_day}/{init_hh}"
        utils.make_directory(f"{output_dir}/{date_str}")
        outdir = Path(f"{output_dir}/{date_str}")
        outdir.mkdir(parents=True, exist_ok=True)

        converter = Netcdf2Grib()
        # Ensure ds_hour has exactly one lead_time equal to 'hour'
        if 'lead_time' in ds_hour.coords:
            try:
                # If needed, overwrite lead_time coord to match the requested hour
                ds_hour = ds_hour.assign_coords(lead_time=("lead_time", [hour]))
            except Exception:
                pass
        converter.save_grib2(init_datetime, ds_hour, member, outdir)

    def get_variable_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (mins, maxs) numpy arrays each shaped (output_channels,).
        """
        raw_bounds = {
            "UGRD":    (-120, 120),
            "VGRD":    (-120, 120),
            "VVEL":    (-30, 30),
            "TMP":     (180, 340),
            "HGT":     (-600, 20000),
            "SPFH":    (0, 0.05),
            "PRES":    (50000, 110000),
            "MSLMA":   (50000, 110000),
            "REFC":    (0, 80),
            "T2M":     (180, 340),
            "UGRD10M": (-100, 100),
            "VGRD10M": (-100, 100),
            "UGRD80M": (-100, 100),
            "VGRD80M": (-100, 100),
            "D2M":     (180, 340),
            "TCDC":    (0, 100),
            "LCDC":    (0, 100),
            "MCDC":    (0, 100),
            "HCDC":    (0, 100),
            "VIS":     (0, 100000),
            "APCP":    (0, 500),
            "HGTCC":   (0, 20000),
            "CAPE":    (0, 7000),
            "CIN":     (-2000, 0),
        }
        mins = []
        maxs = []
        num_levels = len(self.metadata['levels'])
        # Merge 3D and 2D targets into a single loop
        for i, var in enumerate(raw_bounds):
            vmin, vmax = raw_bounds[var]
            if var in self.LOG_TRANSFORM_VARS:
                vmin = np.log1p(vmin)
                vmax = np.log1p(vmax)
            elif var in self.NEG_LOG_TRANSFORM_VARS:
                vmin = np.sign(vmin) * np.log1p(abs(vmin))
                vmax = np.sign(vmax) * np.log1p(abs(vmax))
            # Repeat for each pressure level if 3D, else once
            n_levels = num_levels if i < 6 else 1
            for _ in range(n_levels):
                mins.append(vmin)
                maxs.append(vmax)
        return np.array(mins, dtype=np.float32), np.array(maxs, dtype=np.float32)


    @staticmethod
    def compute_time_features(init_times_np, lead_times_np):
        # Ensure inputs are array-like
        if not isinstance(init_times_np, (list, np.ndarray)):
            init_times_np = [init_times_np]
        if lead_times_np is not None and not isinstance(lead_times_np, (list, np.ndarray)):
            lead_times_np = [lead_times_np]
        # compute valid times
        time_coord = pd.to_datetime(init_times_np)
        if lead_times_np is not None:
            time_coord += pd.to_timedelta(lead_times_np, unit='h')
        # compute cyclical features
        hours = pd.DatetimeIndex(time_coord).hour.astype(np.float32)
        doy = pd.DatetimeIndex(time_coord).dayofyear.astype(np.float32)
        # version masks
        v4 = (time_coord >= np.datetime64("2021-03-23T00")).astype(np.float32)
        v3 = ((time_coord >= np.datetime64("2018-07-12T00")) & (time_coord < np.datetime64("2021-03-23T00"))).astype(np.float32)
        # Stack features into shape [B, 6]
        features = np.stack([
            np.sin(2 * np.pi * hours / 24.0).astype(np.float32),
            np.cos(2 * np.pi * hours / 24.0).astype(np.float32),
            np.sin(2 * np.pi * doy / 365.0).astype(np.float32),
            np.cos(2 * np.pi * doy / 365.0).astype(np.float32),
            v4.astype(np.float32),
            v3.astype(np.float32),
        ], axis=-1)
        return features

    def date_encoding_tensor(self, init_times_np, lead_times_np):
        """Compute cyclical time encodings and HRRR version masks"""

        def get_encoding_tensor(enc):
            enc = tf.cast(enc, dtype=tf.float32)
            batch_size, lat, lon = tf.shape(enc)[0], self.input_shape[1], self.input_shape[2]
            enc = tf.reshape(enc, (batch_size, 1, 1, 6))
            enc = tf.broadcast_to(enc, (batch_size, lat, lon, 6))
            return enc

        enc = self.compute_time_features(init_times_np, lead_times_np)
        enc = get_encoding_tensor(enc)
        return enc

    def predict(self, model: ForecastModel, X: tf.Tensor, members: Union[int, List[int]]) -> tf.Tensor:
        """Predict using diffusion or CRPS model.
        
        Args:
            model: ForecastModel to use for predictions
            X: Input tensor, shape (batch_size, H, W, C)
            members: Single member ID (int) for batch_size=1, or list of member IDs for batch_size>1
                    Length must match batch_size of X.
        
        Returns:
            Predicted tensor of shape (batch_size, H, W, predicted_channels)
        """
        if self.use_diffusion:
            num_output_channels = self.predicted_channels
            start = self.predicted_channels + self.gfs_channels

            # Start from complete gaussian noise (per member)
            Xn_list = [self.member_noise[m] for m in members]
            Xn = tf.concat(Xn_list, axis=0)  # (batch_size, H, W, C)

            # iterate over diffusion steps
            for t_ in range(NUM_INFERENCE_STEPS - 1):
                ti = NUM_INFERENCE_STEPS - 1 - t_
                t = INFERENCE_STEPS[ti]

                # set the correct time embedding
                step_encoding = tf.fill(
                    tf.concat([tf.shape(X)[:-1], [1]], axis=0),
                    tf.cast(t / NUM_DIFFUSION_STEPS, tf.float32),
                )
                X = tf.concat(
                    [
                        X[:, :, :, :start],
                        Xn,
                        X[:, :, :, start + num_output_channels :-2],
                        step_encoding,
                        X[:, :, :, -1:],
                    ],
                    axis=-1,
                )

                # predict total noise
                x_0 = model.predict(X)
                epsilon_t = compute_epsilon(Xn, x_0, t)
                Xn = ddim(Xn, epsilon_t, ti, seed=members)

            return Xn
        else:
            # set CRPS member (per-member noise)
            Xn_list = [self.member_noise[m] for m in members]
            Xn = tf.concat(Xn_list, axis=0)  # (batch_size, H, W, C)
            X = tf.concat(
                [
                    X[:, :, :, :-2],
                    Xn,
                    X[:, :, :, -1:]
                ],
                axis=-1,
            )
            y = model.predict(X)
            return y

    def autoregressive_rollout(self, initial_input: np.ndarray, forcing_input: np.ndarray, model: ForecastModel, 
                             target_hour: int,
                             output_dir: Optional[str] = None,
                             init_datetime: Optional[datetime] = None,
                             write_per_hour: bool = False) -> Dict[int, Dict]:
        """Perform greedy autoregressive rollout with overlapped I/O.

        When write_per_hour=True, persist single-hour NetCDF and GRIB2 files for each lead hour,
        including f00 representing the initial state. I/O is done in background threads to overlap
        with forecasting.
        """
        logger.info(f"Starting autoregressive rollout for {target_hour} hours")
        
        # Initial input (updated during rollout)
        X = tf.convert_to_tensor(initial_input, dtype=tf.float32)
        
        # Track state_from_hour per member
        state_from_hour = {
            member: tf.identity(X[0:1, :, :, :self.predicted_channels]) for member in self.members
        }

        start_pred_noise = self.predicted_channels + self.gfs_channels

        # Lazily fetch coordinates if writing outputs per hour
        lats = lons = None
        if write_per_hour:
            lats, lons = self.data_loader_hrrr.get_coordinates()

        # Local helper to write outputs for any hour using shared context
        def write_hour_outputs(hour: int, data: np.ndarray, member: int) -> None:
            """Build dataset and write NetCDF/GRIB2 for a given hour if output context is available."""
            if not (write_per_hour and output_dir is not None and init_datetime is not None and lats is not None and lons is not None):
                return
            try:
                ds_hour = self.build_single_hour_dataset(init_datetime, hour, lats, lons, data)
                _ = self.write_single_hour_netcdf(init_datetime, hour, ds_hour, output_dir, member)
                self.write_single_hour_grib2(init_datetime, hour, ds_hour, output_dir, member)
                logger.debug(f"Completed writing hour {hour} for member {member}")
            except Exception as e:
                logger.error(f"Failed writing hour {hour} outputs for member {member}: {e}")

        # ThreadPoolExecutor for non-blocking I/O submission
        # Note: Only 1 worker since the _io_write_lock serializes all writes anyway
        io_executor = ThreadPoolExecutor(max_workers=1) if write_per_hour else None
        io_futures = []

        # Write out hour 0 (f00) products representing the initial state for each member
        hour0_outputs: Dict[int, np.ndarray] = {
            member: state_from_hour[member].numpy().copy() for member in self.members
        }
        if self.use_nudging:
            pmm0_values, pmm0_channels = self._compute_pmm_mean(hour0_outputs)
            nudged_hour0 = self._nudge_members_toward_pmm(hour0_outputs, pmm0_values, pmm0_channels, self.pmm_alpha)
        else:
            nudged_hour0 = hour0_outputs
        for member in self.members:
            if io_executor:
                future = io_executor.submit(write_hour_outputs, 0, nudged_hour0[member], member)
                io_futures.append(future)
            else:
                write_hour_outputs(0, nudged_hour0[member], member)

        # phase shift of GFS forcing input
        num_members = self.num_members
        members_sorted = list(range(num_members))
        half_count = num_members // 2  # Half count for symmetry
        step = 1.0 / (half_count + (num_members % 2))  # Adjust step for odd/even
        seq = []
        if num_members % 2 == 1:  # Add zero phase shift for odd
            seq.append(0.0)
        for i in range(half_count):
            seq.append(step * (i + 1))  # Positive phase shifts
            seq.append(-step * (i + 1))  # Negative phase shifts
        phase_angle = {member: seq[i] for i, member in enumerate(members_sorted)}

        # Process all hourly steps
        for hour in range(1, target_hour + 1):
            from_hour = ((hour - 1) // 6) * 6
            step = hour - from_hour
            logger.info(f"Forecasting hour {hour:2d}: from hour {from_hour:2d} using step {step}h")

            # date encoding
            date_encoding = self.date_encoding_tensor(init_datetime, hour)
            lead_encoding = tf.fill(
                tf.concat([tf.shape(X)[:-1], [1]], axis=0),
                tf.cast(step / 6.0, tf.float32),
            )

            # NOTE: forcing_input no longer includes hour 0, so hour=1 corresponds to index 0
            X_base = tf.concat(
                [
                    X[:, :, :, start_pred_noise:-8],
                    date_encoding,
                    X[:, :, :, -2:-1],
                    lead_encoding,
                ],
                axis=-1,
            )

            # Process members in batches
            batch_size = self.batch_size
            hour_member_outputs: Dict[int, np.ndarray] = {}
            for batch_start in range(0, len(self.members), batch_size):
                batch_end = min(batch_start + batch_size, len(self.members))
                batch_members_list = self.members[batch_start:batch_end]
                
                # Collect inputs for this batch of members
                batch_X_members = []
                
                for member in batch_members_list:
                    # apply phase shift to forcing input index for this member
                    phase_width = from_hour // 12
                    phase_shift = round(phase_width * phase_angle[member])
                    forcing_idx = hour - 1 + phase_shift
                    forcing_idx = np.clip(forcing_idx, 0, forcing_input.shape[0] - 1)

                    # Assemble input for this member
                    X_member = tf.concat(
                        [
                            state_from_hour[member],
                            forcing_input[forcing_idx:forcing_idx + 1, :, :, :],
                            X_base,
                        ],
                        axis=-1,
                    )
                    batch_X_members.append(X_member)
                
                # Stack batch inputs
                X_batch = tf.concat(batch_X_members, axis=0)
                
                # Predict next-hour fields for entire batch
                t0 = time.time()
                y_batch = self.predict(model, X_batch, batch_members_list)
                predict_time = time.time() - t0
                logger.info(f"Hour {hour}, batch {batch_start//batch_size + 1}: predict took {predict_time:.3f}s")
                
                y_batch = tf.clip_by_value(
                    y_batch,
                    self.channel_mins[:y_batch.shape[-1]],
                    self.channel_maxs[:y_batch.shape[-1]]
                )
                
                # Process outputs for each member in batch
                for batch_idx, member in enumerate(batch_members_list):
                    y = y_batch[batch_idx:batch_idx+1]
                    
                    # When we reach a 6-hour boundary, update the reference 
                    # state for this member
                    if hour % 6 == 0:
                        state_from_hour[member] = y
                    hour_member_outputs[member] = y.numpy().copy()

            # Compute PMM mean for this hour and nudge members before writing (if enabled)
            if self.use_nudging:
                pmm_values, pmm_channels = self._compute_pmm_mean(hour_member_outputs)
                nudged_outputs = self._nudge_members_toward_pmm(
                    hour_member_outputs, pmm_values, pmm_channels, self.pmm_alpha
                )
            else:
                nudged_outputs = hour_member_outputs

            # Write out per-hour products asynchronously after all members computed
            for member in self.members:
                if io_executor:
                    future = io_executor.submit(write_hour_outputs, hour, nudged_outputs[member], member)
                    io_futures.append(future)
                else:
                    write_hour_outputs(hour, nudged_outputs[member], member)

        # Wait for all I/O operations to complete
        if io_executor:
            logger.info(f"Waiting for {len(io_futures)} I/O operations to complete...")
            for future in as_completed(io_futures):
                try:
                    future.result()  # Raise any exceptions that occurred
                except Exception as e:
                    logger.error(f"I/O operation failed: {e}")
            io_executor.shutdown(wait=True)
            logger.info("All I/O operations completed")

        logger.info("Autoregressive rollout completed")

    def create_xarray_dataset(self, init_datetime: datetime, times: List[int], 
                            lats: np.ndarray, lons: np.ndarray, data: np.ndarray) -> xr.Dataset:
        """Convert numpy array to xarray.Dataset."""
        data_vars = {}
        var_index = 0
        
        pl_vars = self.metadata['pl_vars']
        sfc_vars = self.metadata['sfc_vars']
        levels = self.metadata['levels']
        
        # Pressure-level variables: (time, level, y, x)
        for pl_var in pl_vars:
            pl_data = np.transpose(data[..., var_index:var_index+len(levels)], (0, 3, 1, 2))
            data_vars[pl_var] = xr.DataArray(
                np.expand_dims(pl_data, 0),
                dims=("time", "lead_time", "level", "latitude", "longitude"),
                coords={
                    "time": [init_datetime],
                    "lead_time": ("lead_time", times, {"units": "hours"}),
                    "level": ("level", levels, {"units": "hPa"}),
                    "latitude": (("latitude", "longitude"), lats),
                    "longitude": (("latitude", "longitude"), lons),
                },
                name=pl_var
            )
            var_index += len(levels)
        
        # Surface variables: (time, y, x)
        for sfc_var in sfc_vars:
            sfc_data = data[..., var_index]
            data_vars[sfc_var] = xr.DataArray(
                np.expand_dims(sfc_data, 0),
                dims=("time", "lead_time", "latitude", "longitude"),
                coords={
                    "time": [init_datetime],
                    "lead_time": ("lead_time", times, {"units": "hours"}),
                    "latitude": (("latitude", "longitude"), lats),
                    "longitude": (("latitude", "longitude"), lons),
                },
                name=sfc_var
            )
            var_index += 1
        
        ds = xr.Dataset(data_vars)

        return ds
    
    def run_forecast(self, model: ForecastModel, lead_hours: int, model_input: np.ndarray, output_dir: str = "./"):
        """Run the forecasting pipeline with per-hour streaming outputs. Requires precomputed model_input.

        This function now avoids building a single multi-hour xarray Dataset and avoids bulk NetCDF/GRIB2 writes.
        Instead, per-hour NetCDF/GRIB2 files are written during the autoregressive rollout.
        """
        try:
            init_datetime = self.data_loader_hrrr.get_init_datetime()

            logger.info(f"Running forecast for {init_datetime} with {lead_hours} hour lead time")
            logger.info(f"Model input shape: {model_input.shape}")
            logger.info(self.metadata)

            # Run autoregressive forecast and write per-hour outputs to disk
            self.autoregressive_rollout(
                model_input,
                self.data_loader_gfs.get_model_input(),
                model,
                lead_hours,
                output_dir=output_dir,
                init_datetime=init_datetime,
                write_per_hour=True,
            )
            logger.info("Forecast completed successfully (per-hour outputs written during rollout)")
        except Exception as e:
            logger.error(f"Forecast failed: {e}")
            raise

def run_weather_forecast(forecaster: WeatherForecaster, model: ForecastModel, lead_hours: int, model_input: np.ndarray, output_dir: str):
    """Run forecast for a single member. Requires precomputed model_input."""
    try:
        forecaster.run_forecast(model, lead_hours, model_input, output_dir)
    except Exception as e:
        logger.error(f"Forecast failed : {e}")
        raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Weather Forecast Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("model_path", help="Path to the trained model")
    parser.add_argument('inittime', help='Forecast initialization time in format YYYY-MM-DDTHH (e.g., "2024-05-06T23")')
    parser.add_argument("lead_hours", type=int, help="Lead time in hours")
    parser.add_argument("--num_members", type=int, default=1, help="Number of ensemble members to generate")
    parser.add_argument("--members", nargs='+', required=True, help="List of ensemble member IDs (e.g., 0 1 2 or 0,1,2)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for model inference")
    parser.add_argument("--no_diffusion", default=False, action="store_true", help="Turn off diffusion")
    parser.add_argument("--base_dir", default="./", help="Base directory for input preprocessed files")
    parser.add_argument("--output_dir", default="./", help="Output directory for forecast files")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--pmm_alpha", type=float, default=0.7,
                        help="Nudge factor toward PMM mean for member outputs (0..1)")
    parser.add_argument("--no_nudging", default=False, action="store_true",
                        help="Disable nudging (member perturbation toward consensus)")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    global logger
    args = parse_arguments()
    logger = setup_logging(args.log_level)

    try:
        # Parse members argument (support space/comma separated and ranges like 0-2)
        def expand_member_arg(m):
            result = []
            for part in m.split(","):
                part = part.strip()
                if "-" in part:
                    start, end = part.split("-")
                    result.extend(list(range(int(start), int(end)+1)))
                elif part != "":
                    result.append(int(part))
            return result
        members = []
        for m in args.members:
            members.extend(expand_member_arg(m))
        members = sorted(set(members))  # Remove duplicates and sort

        # Load preprocessed data and model ONCE
        init_datetime, init_year, init_month, init_day, init_hh = utils.validate_datetime(args.inittime)
        date_str = f"{init_year}{init_month}{init_day}/{init_hh}"
        filedate_str = f"{init_year}{init_month}{init_day}_{init_hh}"
        hrrr_preprocessed_file = f"{args.base_dir}/{date_str}/hrrr_{filedate_str}.npz"
        gfs_preprocessed_file = f"{args.base_dir}/{date_str}/gfs_{filedate_str}.npz"
        data_loader_hrrr = PreprocessedDataLoader(hrrr_preprocessed_file)
        data_loader_gfs = PreprocessedDataLoader(gfs_preprocessed_file)
        model = ForecastModel(args.model_path)

        # Precompute model_input ONCE
        model_input_hrrr = data_loader_hrrr.get_model_input()
        model_input_gfs = data_loader_gfs.get_model_input()

        pl_vars = data_loader_hrrr.metadata["pl_vars"]
        sfc_vars = data_loader_hrrr.metadata["sfc_vars"]
        levels = data_loader_hrrr.metadata["levels"]
        predicted_channels = len(pl_vars) * len(levels) + len(sfc_vars)
        gfs_channels = model_input_gfs.shape[-1]
        static_channels = max(model_input_hrrr.shape[-1] - predicted_channels, 0)

        nlat = model_input_hrrr.shape[1]
        nlon = model_input_hrrr.shape[2]
        date_channel = np.ones((1, nlat, nlon, 6), dtype=model_input_hrrr.dtype)
        lead_channel = np.ones((1, nlat, nlon, 1), dtype=model_input_hrrr.dtype)
        step_channel = np.ones((1, nlat, nlon, 1), dtype=model_input_hrrr.dtype)

        if not args.no_diffusion:
            rand_channel = np.ones((1, nlat, nlon, predicted_channels), dtype=model_input_hrrr.dtype)
            model_input = np.concatenate(
                [
                    model_input_hrrr[:, :, :, :predicted_channels],
                    model_input_gfs[0:1, :, :, :],
                    rand_channel,
                    model_input_hrrr[:, :, :, predicted_channels:],
                    date_channel,
                    step_channel,
                    lead_channel
                ],
                axis=-1
            )
        else:
            model_input = np.concatenate(
                [
                    model_input_hrrr[:, :, :, :predicted_channels],
                    model_input_gfs[0:1, :, :, :],
                    model_input_hrrr[:, :, :, predicted_channels:],
                    date_channel,
                    step_channel,
                    lead_channel
                ],
                axis=-1
            )
        
        forecaster = WeatherForecaster(data_loader_hrrr, data_loader_gfs,
                                        args.num_members, members,
                                        args.batch_size, not args.no_diffusion,
                                        predicted_channels=predicted_channels,
                                        gfs_channels=gfs_channels,
                                        static_channels=static_channels,
                                        pmm_alpha=args.pmm_alpha,
                                        use_nudging=not args.no_nudging)
        run_weather_forecast(
            forecaster, model, args.lead_hours, model_input, args.output_dir
        )
        logger.info(f"All forecasts complete.")
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
