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
from datetime import datetime, timedelta
from dateutil import parser
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import xarray as xr
from tqdm import tqdm

from nc2grib import Netcdf2Grib

# Import custom modules (assuming they exist)
try:
    import resnet
    import losses
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
    DEFAULT_LOG_EPS,
)
import utils
from utils import setup_logging

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
        tf.config.optimizer.set_jit(False)
    
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
        member: int,
        use_diffusion: bool,
        predicted_channels: Optional[int] = None,
        gfs_channels: Optional[int] = None,
        static_channels: Optional[int] = None,
    ):
        self.data_loader_hrrr = data_loader_hrrr
        self.data_loader_gfs = data_loader_gfs
        self.metadata = data_loader_hrrr.metadata
        self.member = member
        self.use_diffusion = use_diffusion

        # Derive dynamic channel counts if not provided
        pl_vars = self.metadata["pl_vars"]
        sfc_vars = self.metadata["sfc_vars"]
        levels = self.metadata["levels"]
        default_predicted = len(pl_vars) * len(levels) + len(sfc_vars)
        total_hrrr = data_loader_hrrr.get_model_input().shape[-1]

        if predicted_channels is None:
            predicted_channels = default_predicted
        if gfs_channels is None:
            gfs_channels = data_loader_gfs.get_model_input().shape[-1]
        if static_channels is None:
            static_channels = max(total_hrrr - predicted_channels, 0)

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
    

    def _init_channel_stats(self, ds_norm: xr.Dataset):
        """Build flattened mean/std vectors matching channel ordering in preprocessing.

        Ordering used in make_ics preprocessing:
          1. Pressure-level vars in the order (UGRD, VGRD, VVEL, TMP, HGT, SPFH) for each level.
          2. Surface vars in the order stored in metadata['sfc_vars'] (no constants).
        Constants (e.g., LAND, OROG) were appended in preprocessing but are not predicted
        by the diffusion / deterministic heads (first 74 channels). We still include them
        at the tail of the vectors if present so slicing remains safe.
        """
        pl_vars = self.metadata['pl_vars']  # ["UGRD", ...]
        sfc_vars = self.metadata['sfc_vars']
        levels = self.metadata['levels']

        means = []
        stds = []

        # Pressure-level variables
        # Preprocessing loops over raw GRIB shortNames: u,v,w,t,gh,q corresponding to pl_vars order below
        pl_order = ["UGRD", "VGRD", "VVEL", "TMP", "HGT", "SPFH"]
        for var in pl_order:
            if var not in ds_norm.variables:
                # Fallback: fill with zeros/ones to avoid crash
                logger.warning(f"Normalization stats missing for pressure var {var}; using mean=0,std=1")
                means.extend([0.0] * len(levels))
                stds.extend([1.0] * len(levels))
                continue
            stats = ds_norm[var].values  # shape (2, level)
            # Safeguard shape
            if stats.shape[0] < 2:
                logger.warning(f"Stats for {var} malformed; using zeros/ones")
                means.extend([0.0] * len(levels))
                stds.extend([1.0] * len(levels))
                continue
            # If level dimension differs, broadcast or truncate
            nlev_stats = stats.shape[1] if stats.ndim > 1 else 1
            for i, lvl in enumerate(levels):
                if i < nlev_stats:
                    means.append(float(stats[0, i]))
                    stds.append(float(stats[1, i]))
                else:
                    means.append(0.0)
                    stds.append(1.0)

        # Surface variables (single value per variable)
        for var in sfc_vars:
            if var not in ds_norm.variables:
                logger.warning(f"Normalization stats missing for surface var {var}; using mean=0,std=1")
                means.append(0.0)
                stds.append(1.0)
                continue
            stats = ds_norm[var].values  # expect (2, ...) first dim=stat
            if stats.shape[0] < 2:
                logger.warning(f"Stats for {var} malformed; using mean=0,std=1")
                means.append(0.0)
                stds.append(1.0)
                continue
            # Reduce any remaining dims to scalar with nanmean
            means.append(float(np.nanmean(stats[0])))
            stds.append(float(np.nanmean(stats[1])) if np.nanmean(stats[1]) != 0 else 1.0)

        self.channel_means = np.array(means, dtype=np.float32)
        self.channel_stds = np.array(stds, dtype=np.float32)
        vmin, vmax = self.get_variable_bounds()
        self.channel_mins = (vmin - self.channel_means) / self.channel_stds
        self.channel_maxs = (vmax - self.channel_means) / self.channel_stds

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
    
    def predict(self, model: ForecastModel, X: tf.Tensor):
        if self.use_diffusion:
            num_output_channels = self.predicted_channels
            start = self.predicted_channels + self.gfs_channels
            batch_size = 1

            # start from complete gaussian noise
            tf.random.set_seed(self.member)
            Xn = tf.random.normal(
                shape=tf.shape(X[0, :, :, start : start + num_output_channels])
            )
            Xn = tf.tile(tf.expand_dims(Xn, axis=0), [batch_size, 1, 1, 1])
            X = tf.concat(
                [
                    X[:, :, :, :start],
                    Xn,
                    X[:, :, :, start + num_output_channels :],
                ],
                axis=-1,
            )

            # iterate over diffusion steps
            for t_ in tqdm(range(NUM_INFERENCE_STEPS - 1)):
                ti = NUM_INFERENCE_STEPS - 1 - t_
                t = INFERENCE_STEPS[ti]
                # set the correct time embedding
                X = tf.concat(
                    [
                        X[:, :, :, :-2],
                        tf.fill(
                            tf.concat([tf.shape(X)[:-1], [1]], axis=0),
                            t / NUM_DIFFUSION_STEPS,
                        ),
                        X[:, :, :, -1:],
                    ],
                    axis=-1,
                )

                # predict total noise
                x_0 = model.predict(X)
                epsilon_t = compute_epsilon(Xn, x_0, t)

                Xn = ddim(Xn, epsilon_t, ti, seed=self.member)
                X = tf.concat(
                    [
                        X[:, :, :, :start],
                        Xn,
                        X[:, :, :, start + num_output_channels :],
                    ],
                    axis=-1,
                )

            return Xn
        else:
            return model.predict(X)


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
            "R2M":     (0, 100),
            "TCDC":    (0, 100),
            "VIS":     (0, 100000),
            "APCP":    (0, 500),
            "HGTCC":   (0, 20000),
            "CAPE":    (0, 7000),
            "CIN":     (-2000, 0),
        }
        eps = 1e-3
        mins = []
        maxs = []
        log_vars = {"SPFH", "VIS", "APCP", "HGTCC", "CAPE"}
        neg_log_vars = {"CIN"}
        num_levels = len(self.metadata['levels'])
        # Merge 3D and 2D targets into a single loop
        for i, var in enumerate(raw_bounds):
            vmin, vmax = raw_bounds[var]
            if var in log_vars:
                vmin = np.log(vmin + eps) - np.log(eps)
                vmax = np.log(vmax + eps) - np.log(eps)
            elif var in neg_log_vars:
                vmin = np.sign(vmin) * (np.log(abs(vmin) + eps) - np.log(eps))
                vmax = np.sign(vmax) * (np.log(abs(vmax) + eps) - np.log(eps))
            # Repeat for each pressure level if 3D, else once
            n_levels = num_levels if i < 6 else 1
            for _ in range(n_levels):
                mins.append(vmin)
                maxs.append(vmax)
        return np.array(mins, dtype=np.float32), np.array(maxs, dtype=np.float32)

    def autoregressive_rollout(self, initial_input: np.ndarray, forcing_input: np.ndarray, model: ForecastModel, 
                             target_hour: int) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
        """Perform greedy autoregressive rollout."""
        logger.info(f"Starting autoregressive rollout for {target_hour} hours")
        
        # Initial input (updated during rollout)
        X = tf.convert_to_tensor(initial_input, dtype=tf.float32)
        
        # Stores forecasts and history
        hourly_forecasts = {0: tf.identity(X[0:1, :, :, :self.predicted_channels])}
        history = {0: {'step': 0, 'from': None}}

        start_pred_noise = self.predicted_channels + self.gfs_channels

        # Process all hourly steps
        for hour in tqdm(range(1, target_hour + 1), desc="Forecasting"):
            from_hour = ((hour - 1) // 6) * 6
            step = hour - from_hour

            # NOTE: forcing_input no longer includes hour 0, so hour=1 corresponds to index 0
            X = tf.concat(
                [
                    hourly_forecasts[from_hour],
                    forcing_input[hour-1:hour, :, :, :],
                    X[:, :, :, start_pred_noise:-1],
                    tf.fill(
                        tf.concat([tf.shape(X)[:-1], [1]], axis=0),
                        step / 6.0,
                    ),
                ],
                axis=-1,
            )

            # Predict next-hour fields (predicted channels only)
            y = self.predict(model, X)
            y = tf.clip_by_value(y, self.channel_mins[:y.shape[-1]], self.channel_maxs[:y.shape[-1]])

            # Apply REFC noise suppression: set reflectivity < 5 dBZ to 0 (operate in normalized space)
            refc_channel = 122
            refc = y[..., refc_channel]
            refc = tf.where(refc < 0.05, 0.0, refc)
            y = tf.concat([
                y[..., :refc_channel],
                tf.expand_dims(refc, axis=-1),
                y[..., refc_channel + 1:]
            ], axis=-1)

            hourly_forecasts[hour] = y
            history[hour] = {"step": step, "from": from_hour}

        logger.info("Autoregressive rollout completed")
        return hourly_forecasts, history
    
    @staticmethod
    def compute_r2m(ds: xr.Dataset) -> xr.Dataset:
        """
        Compute relative humidity at 2m (R2M) using T2M and D2M.
        Assumes T2M and D2M are in Kelvin.
        """
        # Convert to Celsius
        T_c = ds["T2M"] - 273.15
        Td_c = ds["D2M"] - 273.15
        # Magnus formula for saturation vapor pressure (hPa)
        es = 6.112 * np.exp((17.67 * T_c) / (T_c + 243.5))
        e = 6.112 * np.exp((17.67 * Td_c) / (Td_c + 243.5))
        # Relative humidity (%)
        RH = 100.0 * (e / es)
        # Clip to [0, 100]
        ds["R2M"] = RH.clip(min=0.0, max=100.0)
        return ds

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

        # compute relative humidity at 2m
        ds = self.compute_r2m(ds)

        return ds
    
    def run_forecast(self, model: ForecastModel, lead_hours: int, model_input: np.ndarray, output_dir: str = "./", return_history: bool = False):
        """Run the complete forecasting pipeline. Requires precomputed model_input."""
        try:
            lats, lons = self.data_loader_hrrr.get_coordinates()
            init_datetime = self.data_loader_hrrr.get_init_datetime()

            logger.info(f"Running forecast for {init_datetime} with {lead_hours} hour lead time")
            logger.info(f"Model input shape: {model_input.shape}")
            logger.info(self.metadata)

            # Run autoregressive forecast
            hourly_forecasts, history = self.autoregressive_rollout(model_input, self.data_loader_gfs.get_model_input(), model, lead_hours)

            # Denormalize all outputs
            logger.info("Denormalizing outputs...")
            denorm_outputs = {}
            for hour, forecast in hourly_forecasts.items():
                denorm_outputs[hour] = self.denormalize(forecast[None, ...])

            # Stack all timesteps into a single numpy array
            outdata = np.array([denorm_outputs[i] for i in range(0, lead_hours + 1)])

            # Create timestamps for each forecast hour
            times = list(range(0, lead_hours + 1))

            # Convert numpy to xarray
            logger.info("Creating xarray dataset...")
            outdata_xr = self.create_xarray_dataset(init_datetime, times, lats, lons, outdata)

            # Inject raw constant LAND / OROG if present (repeat across lead_time so GRIB conversion sees time axis)
            for cname in ["LAND", "OROG"]:
                raw_key = f"{cname}_raw"
                if raw_key in self.data_loader_hrrr.data.files and cname not in outdata_xr:
                    cvals = self.data_loader_hrrr.data[raw_key]
                    const_4d = np.tile(cvals[None, None, :, :], (1, len(times), 1, 1))
                    outdata_xr[cname] = xr.DataArray(
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
                    logger.info(f"Added constant field {cname} to forecast output")

            # Apply inverse log / signed-log transforms to recover physical units
            try:
                log_vars = {"SPFH", "VIS", "APCP", "HGTCC", "CAPE"}
                neg_log_vars = {"CIN"}
                applied = []
                for var in log_vars:
                    if var in outdata_xr.variables:
                        data_arr = outdata_xr[var].values  # shape (time, lead_time, [level], y, x)
                        outdata_xr[var].values[:] = inverse_log_transform_array(data_arr, eps=DEFAULT_LOG_EPS)
                        applied.append(var)
                for var in neg_log_vars:
                    if var in outdata_xr.variables:
                        data_arr = outdata_xr[var].values
                        outdata_xr[var].values[:] = inverse_neg_log_transform_array(data_arr, eps=DEFAULT_LOG_EPS)
                        applied.append(var)
                if applied:
                    logger.info(f"Applied inverse log transforms to variables: {', '.join(applied)}")
                else:
                    logger.info("No inverse log transforms applied (variables not found in dataset)")
            except Exception as e:
                logger.error(f"Failed applying inverse log transforms: {e}")

            # Save output
            init_year = self.metadata['init_year']
            init_month = self.metadata['init_month']
            init_day = self.metadata['init_day']
            init_hh = self.metadata['init_hh']
            date_str = f"{init_year}{init_month}{init_day}/{init_hh}"
            utils.make_directory(f"{output_dir}/{date_str}")

            # Create a new directory for grib2 files
            outdir = Path(f"{output_dir}/{date_str}")
            outdir.mkdir(parents=True, exist_ok=True)

            output_file = f"{output_dir}/{date_str}/hrrrcast_mem{self.member}.nc"
            logger.info(f"Saving forecast to {output_file}")
            outdata_xr.to_netcdf(output_file)

            converter = Netcdf2Grib()
            converter.save_grib2(init_datetime, outdata_xr, self.member, outdir)
            
            logger.info("Forecast completed successfully")
            if return_history:
                return outdata_xr, output_file, history
            else:
                return outdata_xr, output_file
        except Exception as e:
            logger.error(f"Forecast failed: {e}")
            raise

def run_weather_forecast_for_member(forecaster: WeatherForecaster, model: ForecastModel, lead_hours: int, model_input: np.ndarray, output_dir: str, member: int, print_history: bool = False):
    """Run forecast for a single member, optionally printing forecast step history. Requires precomputed model_input."""
    try:
        forecast_dataset, output_file, history = forecaster.run_forecast(model, lead_hours, model_input, output_dir, return_history=True)
        if print_history:
            logger.info("Forecast schedule:")
            for hour in range(1, min(lead_hours + 1, 25)):
                info = history[hour]
                logger.info(f"Hour {hour:2d}: from hour {info['from']:2d} using step {info['step']}h")
            if lead_hours > 24:
                logger.info(f"... (showing first 24 hours of {lead_hours} total)")
        return forecast_dataset, output_file
    except Exception as e:
        logger.error(f"Forecast failed for member {member}: {e}")
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
    parser.add_argument("--members", nargs='+', required=True, help="List of ensemble member IDs (e.g., 0 1 2 or 0,1,2)")
    parser.add_argument("--no_diffusion", default=False, action="store_true", help="Turn off diffusion")
    parser.add_argument("--base_dir", default="./", help="Base directory for input preprocessed files")
    parser.add_argument("--output_dir", default="./", help="Output directory for forecast files")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
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
        model_input_hrrr = np.nan_to_num(model_input_hrrr, nan=0.0)
        model_input_gfs = np.nan_to_num(model_input_gfs, nan=0.0)
        pl_vars = data_loader_hrrr.metadata["pl_vars"]
        sfc_vars = data_loader_hrrr.metadata["sfc_vars"]
        levels = data_loader_hrrr.metadata["levels"]
        predicted_channels = len(pl_vars) * len(levels) + len(sfc_vars)
        gfs_channels = model_input_gfs.shape[-1]
        static_channels = max(model_input_hrrr.shape[-1] - predicted_channels, 0)

        lead_channel = np.ones((1, model_input_hrrr.shape[1], model_input_hrrr.shape[2], 1), dtype=model_input_hrrr.dtype)
        if not args.no_diffusion:
            rand_channel = np.ones((1, model_input_hrrr.shape[1], model_input_hrrr.shape[2], predicted_channels), dtype=model_input_hrrr.dtype)
            step_channel = np.ones((1, model_input_hrrr.shape[1], model_input_hrrr.shape[2], 1), dtype=model_input_hrrr.dtype)
            model_input = np.concatenate([
                model_input_hrrr[:, :, :, :predicted_channels],
                model_input_gfs[0:1, :, :, :],
                rand_channel,
                model_input_hrrr[:, :, :, predicted_channels:],
                step_channel, lead_channel], axis=-1)
        else:
            model_input = np.concatenate([
                model_input_hrrr[:, :, :, :predicted_channels],
                model_input_gfs[0:1, :, :, :],
                model_input_hrrr[:, :, :, predicted_channels:],
                lead_channel], axis=-1)
        
        output_files = []
        for i, member in enumerate(members):
            forecaster = WeatherForecaster(data_loader_hrrr, data_loader_gfs, member, not args.no_diffusion,
                                           predicted_channels=predicted_channels,
                                           gfs_channels=gfs_channels,
                                           static_channels=static_channels)
            forecast_dataset, output_file = run_weather_forecast_for_member(
                forecaster, model, args.lead_hours, model_input, args.output_dir, member, print_history=(i==len(members)-1)
            )
            logger.info(f"Forecast complete for member {member}. Output saved to: {output_file}")
            output_files.append(output_file)
        logger.info(f"All forecasts complete. Output files: {output_files}")
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
