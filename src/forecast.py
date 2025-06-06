#!/usr/bin/env python3
"""
Weather Forecasting Model Inference Script

This script performs autoregressive weather forecasting using a trained neural network model.
It processes HRRR GRIB files and generates forecasts at hourly intervals.

Usage:
    python forecast.py <model_path> <year> <month> <day> <hour> <lead_hours>
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pygrib as pg
import tensorflow as tf
import xarray as xr
from tqdm import tqdm

# Import custom modules (assuming they exist)
try:
    import resnet
    import losses
except ImportError as e:
    logging.warning(f"Could not import custom modules: {e}")

# suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeatherForecastConfig:
    """Configuration class for weather forecasting parameters."""
    
    def __init__(self):
        # Pressure level and surface variables
        self.pl_vars = ["UGRD", "VGRD", "VVEL", "TMP", "HGT", "SPFH"]
        self.sfc_vars = ["T2M", "REFC"]
        
        # Pressure levels (hPa)
        self.levels = [200, 300, 475, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
        
        # Grid downsampling factor
        self.downsample_factor = 2
        
        # Expected grid dimensions after downsampling
        self.grid_height = 530
        self.grid_width = 900


class GRIBProcessor:
    """Handles GRIB file processing and normalization."""
    
    def __init__(self, config: WeatherForecastConfig):
        self.config = config
    
    @staticmethod
    def normalize(data: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Normalize data using mean and standard deviation."""
        if std == 0:
            logger.warning("Standard deviation is zero, skipping normalization")
            return data - mean
        return (data - mean) / std
    
    @staticmethod
    def denormalize(output: np.ndarray, norm_file: str) -> np.ndarray:
        """Convert model output back to physical units using stored mean/std."""
        try:
            norms = xr.open_dataset(norm_file)['UGRD']
            mean = norms[0, :74].values[None, None, None, :]
            std = norms[1, :74].values[None, None, None, :]
            return np.squeeze(output * std + mean)
        except Exception as e:
            logger.error(f"Error in denormalization: {e}")
            raise
    
    def process_pressure_levels(self, pres_file: str, norm_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and normalize pressure-level variables."""
        if not os.path.exists(pres_file):
            raise FileNotFoundError(f"Pressure file not found: {pres_file}")
        
        try:
            grbs = pg.open(pres_file)
            norms = xr.open_dataset(norm_file)['UGRD'].values
            varnames = ['u', 'v', 'w', 't', 'gh', 'q']
            
            normalized_vals, raw_vals = [], []
            
            logger.info("Processing pressure level variables...")
            for v_idx, var in enumerate(varnames):
                selected = grbs.select(shortName=var, level=self.config.levels)
                
                if len(selected) != len(self.config.levels):
                    logger.warning(f"Expected {len(self.config.levels)} levels for {var}, got {len(selected)}")
                
                for l_idx, grb in enumerate(selected):
                    vals = grb.values[::self.config.downsample_factor, ::self.config.downsample_factor]
                    idx = v_idx * len(self.config.levels) + l_idx
                    
                    if idx < len(norms[0]):
                        mean, std = norms[0, idx], norms[1, idx]
                        raw_vals.append(vals)
                        normalized_vals.append(self.normalize(vals, mean, std))
                    else:
                        logger.error(f"Normalization index {idx} out of bounds")
                        raise IndexError(f"Normalization index {idx} out of bounds")
            
            grbs.close()
            return np.array(normalized_vals), np.array(raw_vals)
            
        except Exception as e:
            logger.error(f"Error processing pressure levels: {e}")
            raise
    
    def process_surface_variables(self, sfc_file: str, norm_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and normalize surface variables."""
        if not os.path.exists(sfc_file):
            raise FileNotFoundError(f"Surface file not found: {sfc_file}")
        
        try:
            grbs = pg.open(sfc_file)
            norms = xr.open_dataset(norm_file)['UGRD'].values
            
            # Mean/std for t2m and refc
            t2_mean, t2_std = norms[0, 72], norms[1, 72]
            refc_mean, refc_std = norms[0, 73], norms[1, 73]
            
            # Get 2D lat/lon
            lats, lons = grbs[1].latlons()
            lats = lats[::self.config.downsample_factor, ::self.config.downsample_factor]
            lons = lons[::self.config.downsample_factor, ::self.config.downsample_factor]
            
            # Extract necessary variables and downsample
            t2_vals = grbs.select(shortName="2t")[0].values[::self.config.downsample_factor, ::self.config.downsample_factor]
            refc_vals = grbs.select(shortName="refc")[0].values[::self.config.downsample_factor, ::self.config.downsample_factor]
            refc_vals = np.maximum(refc_vals, 0)  # Remove invalid reflectivity values
            lsm_vals = grbs.select(shortName="lsm")[0].values[::self.config.downsample_factor, ::self.config.downsample_factor]
            orog_vals = grbs.select(shortName="orog")[0].values[::self.config.downsample_factor, ::self.config.downsample_factor]
            
            # Normalize extracted variables
            normalized = [
                self.normalize(t2_vals, t2_mean, t2_std),
                self.normalize(refc_vals, refc_mean, refc_std),
                self.normalize(lsm_vals, np.mean(lsm_vals), np.std(lsm_vals)),
                self.normalize(orog_vals, np.mean(orog_vals), np.std(orog_vals)),
            ]
            
            raw = [t2_vals, refc_vals]
            
            grbs.close()
            return np.array(normalized), np.array(raw), lats, lons
            
        except Exception as e:
            logger.error(f"Error processing surface variables: {e}")
            raise


class ForecastModel:
    """Handles model loading and inference."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
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
    """Main forecasting class that orchestrates the entire process."""
    
    def __init__(self, config: WeatherForecastConfig):
        self.config = config
        self.grib_processor = GRIBProcessor(config)
    
    def create_xarray_dataset(self, init_datetime: datetime, times: List[np.timedelta64], 
                            lats: np.ndarray, lons: np.ndarray, data: np.ndarray) -> xr.Dataset:
        """Convert numpy array to xarray.Dataset."""
        data_vars = {}
        var_index = 0
        
        # Pressure-level variables: (time, level, y, x)
        for pl_var in self.config.pl_vars:
            pl_data = np.transpose(data[..., var_index:var_index+len(self.config.levels)], (0, 3, 1, 2))
            data_vars[pl_var] = xr.DataArray(
                np.expand_dims(pl_data, 0),
                dims=("time", "lead_time", "level", "latitude", "longitude"),
                coords={
                    "time": [init_datetime],
                    "lead_time": times,
                    "level": self.config.levels,
                    "latitude": (("latitude", "longitude"), lats),
                    "longitude": (("latitude", "longitude"), lons),
                },
                name=pl_var
            )
            var_index += len(self.config.levels)
        
        # Surface variables: (time, y, x)
        for sfc_var in self.config.sfc_vars:
            sfc_data = data[..., var_index]
            data_vars[sfc_var] = xr.DataArray(
                np.expand_dims(sfc_data, 0),
                dims=("time", "lead_time", "latitude", "longitude"),
                coords={
                    "time": [init_datetime],
                    "lead_time": times,
                    "latitude": (("latitude", "longitude"), lats),
                    "longitude": (("latitude", "longitude"), lons),
                },
                name=sfc_var
            )
            var_index += 1
        
        return xr.Dataset(data_vars)
    
    def autoregressive_rollout(self, initial_input: np.ndarray, model: ForecastModel, 
                             target_hour: int) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
        """Perform greedy autoregressive rollout."""
        logger.info(f"Starting autoregressive rollout for {target_hour} hours")
        
        # Initial input (updated during rollout)
        current_input = initial_input.copy()
        
        # Stores forecasts and history
        hourly_forecasts = {0: current_input[0, :, :, :74].copy()}
        history = {0: {'step': 0, 'from': None}}
        
        # Process all hourly steps
        for hour in tqdm(range(1, target_hour + 1), desc="Forecasting"):
            from_hour = ((hour - 1) // 6) * 6
            step = hour - from_hour
            
            current_input[0, :, :, :74] = hourly_forecasts[from_hour]
            current_input[0, :, :, 76] = step / 6.0
            
            output = model.predict(current_input)
            hourly_forecasts[hour] = output[0]
            history[hour] = {
                'step': step,
                'from': from_hour,
            }
        
        logger.info("Autoregressive rollout completed")
        return hourly_forecasts, history
    
    def run_forecast(self, model_path: str, init_year: str, init_month: str, 
                    init_day: str, init_hh: str, lead_hours: int, base_dir: str = "./"):
        """Run the complete forecasting pipeline."""
        try:
            # Validate inputs
            init_datetime = datetime.strptime(f"{init_year}{init_month}{init_day}_{init_hh}", "%Y%m%d_%H")
            logger.info(f"Running forecast for {init_datetime} with {lead_hours} hour lead time")
            
            # Setup paths
            norm_file = Path(model_path).parent / "normalize.nc"
            hrrr_pres_file = f"{base_dir}/{init_year}{init_month}{init_day}_{init_hh}_pres"
            hrrr_sfc_file = f"{base_dir}/{init_year}{init_month}{init_day}_{init_hh}_sfc"
            
            # Validate required files exist
            for file_path in [norm_file, hrrr_pres_file, hrrr_sfc_file]:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Required file not found: {file_path}")
            
            # Load model
            model = ForecastModel(model_path)
            
            # Process GRIB data
            logger.info("Processing GRIB data...")
            pres_norm, pres_raw = self.grib_processor.process_pressure_levels(hrrr_pres_file, str(norm_file))
            sfc_norm, sfc_raw, lats, lons = self.grib_processor.process_surface_variables(hrrr_sfc_file, str(norm_file))
            
            # Validate grid dimensions
            expected_shape = (self.config.grid_height, self.config.grid_width)
            if pres_norm.shape[1:] != expected_shape:
                logger.warning(f"Unexpected grid shape: {pres_norm.shape[1:]} vs expected {expected_shape}")
            
            # Format input shape: (1, y, x, channels)
            lead_channel = np.ones((1, self.config.grid_height, self.config.grid_width))
            model_input = np.concatenate((pres_norm, sfc_norm, lead_channel), axis=0)
            model_input = np.transpose(model_input, (1, 2, 0))[None, ...]
            
            logger.info(f"Model input shape: {model_input.shape}")
            
            # Run autoregressive forecast
            hourly_forecasts, history = self.autoregressive_rollout(model_input, model, lead_hours)
            
            # Denormalize all outputs
            logger.info("Denormalizing outputs...")
            denorm_outputs = {}
            for hour, forecast in hourly_forecasts.items():
                denorm_outputs[hour] = self.grib_processor.denormalize(forecast[None, ...], str(norm_file))
            
            # Stack all timesteps into a single numpy array
            outdata = np.array([denorm_outputs[i] for i in range(0, lead_hours + 1)])
            
            # Create timestamps for each forecast hour
            times = [np.timedelta64(i, 'h') for i in range(0, lead_hours + 1)]
            
            # Convert numpy to xarray
            logger.info("Creating xarray dataset...")
            outdata_xr = self.create_xarray_dataset(init_datetime, times, lats, lons, outdata)
            
            # Print the forecast step history
            logger.info("Forecast schedule:")
            for hour in range(1, min(lead_hours + 1, 25)):  # Limit output for readability
                info = history[hour]
                logger.info(f"Hour {hour:2d}: from hour {info['from']:2d} using step {info['step']}h")
            
            if lead_hours > 24:
                logger.info(f"... (showing first 24 hours of {lead_hours} total)")
            
            # Save output
            output_file = f"{base_dir}/{init_year}{init_month}{init_day}_{init_hh}.nc"
            logger.info(f"Saving output to {output_file}")
            outdata_xr.to_netcdf(output_file)
            
            # Optional: Save to Zarr format
            # zarr_file = f"{base_dir}/{init_year}{init_month}{init_day}_{init_hh}.zarr"
            # outdata_xr.to_zarr(zarr_file, mode="w")
            
            logger.info("Forecast completed successfully")
            return outdata_xr
            
        except Exception as e:
            logger.error(f"Forecast failed: {e}")
            raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Weather Forecasting Model Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("model_path", help="Path to the trained model")
    parser.add_argument("init_year", help="Initialization year (YYYY)")
    parser.add_argument("init_month", help="Initialization month (MM)")
    parser.add_argument("init_day", help="Initialization day (DD)")
    parser.add_argument("init_hh", help="Initialization hour (HH)")
    parser.add_argument("lead_hours", type=int, help="Lead time in hours")
    parser.add_argument("--base_dir", default="./", help="Base directory for input/output files")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    try:
        args = parse_arguments()
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Initialize configuration and forecaster
        config = WeatherForecastConfig()
        forecaster = WeatherForecaster(config)
        
        # Run forecast
        forecaster.run_forecast(
            model_path=args.model_path,
            init_year=args.init_year,
            init_month=args.init_month,
            init_day=args.init_day,
            init_hh=args.init_hh,
            lead_hours=args.lead_hours,
            base_dir=args.base_dir
        )
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
