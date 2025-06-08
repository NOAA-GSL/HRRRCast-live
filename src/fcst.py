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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import xarray as xr
from tqdm import tqdm

# Import custom modules (assuming they exist)
try:
    import resnet
    import losses
except ImportError as e:
    logging.warning(f"Could not import custom modules: {e}")

from diffusion_params import (
    USE_DIFFUSION,
    NUM_DIFFUSION_STEPS,
    NUM_INFERENCE_STEPS,
    INFERENCE_STEPS,
    compute_epsilon,
    ddpm,
    ddim,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    """Handles the forecasting pipeline."""
    
    def __init__(self, data_loader_hrrr: PreprocessedDataLoader, data_loader_gfs: PreprocessedDataLoader):
        self.data_loader_hrrr = data_loader_hrrr
        self.data_loader_gfs = data_loader_gfs
        self.metadata = data_loader_hrrr.metadata
    
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
    
    def predict(self, model: ForecastModel, current_input: np.ndarray):
        if USE_DIFFUSION:
            member=0
            np.random.seed(member)
            Xn = current_input[:, :, :, 102:176] 
            Xn = np.random.normal(size=Xn.shape)
            for t_ in tqdm(range(NUM_INFERENCE_STEPS - 1)):
                ti = NUM_INFERENCE_STEPS - 1 - t_
                t = INFERENCE_STEPS[ti]
                current_input[:, :, :, -2] = t / NUM_DIFFUSION_STEPS
                current_input[:, :, :, 102:176] = Xn
                X0 = model.predict(current_input)
                epsilon_t = compute_epsilon(Xn, X0, t)
                Xn = ddim(Xn, epsilon_t, ti, seed=member)
            return Xn
        else:
            return model.predict(current_input)

    def autoregressive_rollout(self, initial_input: np.ndarray, forcing_input: np.ndarray, model: ForecastModel, 
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
            
            # setup ICs and BCs
            current_input[0, :, :, :74] = hourly_forecasts[from_hour]
            current_input[0, :, :, 74:102] = forcing_input[hour, :, :, :]
            current_input[0, :, :, -1] = step / 6.0
            
            # predict
            output = self.predict(model, current_input)

            # set to 0 negative REFC values
            refc = output[..., -1]
            refc = tf.where(refc < 0, tf.zeros_like(refc), refc)
            output = tf.concat([output[...,:-1], tf.expand_dims(refc, axis=-1)], axis=-1)

            hourly_forecasts[hour] = output
            history[hour] = {
                'step': step,
                'from': from_hour,
            }
        
        logger.info("Autoregressive rollout completed")
        return hourly_forecasts, history
    
    def create_xarray_dataset(self, init_datetime: datetime, times: List[np.timedelta64], 
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
                    "lead_time": times,
                    "level": levels,
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
                    "lead_time": times,
                    "latitude": (("latitude", "longitude"), lats),
                    "longitude": (("latitude", "longitude"), lons),
                },
                name=sfc_var
            )
            var_index += 1
        
        return xr.Dataset(data_vars)
    
    def run_forecast(self, model: ForecastModel, lead_hours: int, output_dir: str = "./"):
        """Run the complete forecasting pipeline."""
        try:
            # Get preprocessed data
            model_input_hrrr = self.data_loader_hrrr.get_model_input()
            model_input_gfs = self.data_loader_gfs.get_model_input()
            lead_channel = np.ones((1, model_input_hrrr.shape[1], model_input_hrrr.shape[2], 1))
            if USE_DIFFUSION:
                rand_channel = np.ones((1, model_input_hrrr.shape[1], model_input_hrrr.shape[2], 74))
                step_channel = np.ones((1, model_input_hrrr.shape[1], model_input_hrrr.shape[2], 1))
                model_input = np.concatenate([
                    model_input_hrrr[:, :, :, :74],
                    model_input_gfs[0:1, :, :, :],
                    rand_channel,
                    model_input_hrrr[:, :, :, 74:],
                    step_channel, lead_channel], axis=-1)
            else:
                model_input = np.concatenate([
                    model_input_hrrr[:, :, :, :74],
                    model_input_gfs[0:1, :, :, :],
                    model_input_hrrr[:, :, :, 74:],
                    lead_channel], axis=-1)

            lats, lons = self.data_loader_hrrr.get_coordinates()
            init_datetime = self.data_loader_hrrr.get_init_datetime()

            logger.info(f"Running forecast for {init_datetime} with {lead_hours} hour lead time")
            logger.info(f"Model input shape: {model_input.shape}")
            
            logger.info(self.metadata)

            # Run autoregressive forecast
            hourly_forecasts, history = self.autoregressive_rollout(model_input, model_input_gfs, model, lead_hours)
            
            # Denormalize all outputs
            logger.info("Denormalizing outputs...")
            norm_file = self.metadata['norm_file']
            denorm_outputs = {}
            for hour, forecast in hourly_forecasts.items():
                denorm_outputs[hour] = self.denormalize(forecast[None, ...], norm_file)
            
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
            init_year = self.metadata['init_year']
            init_month = self.metadata['init_month']
            init_day = self.metadata['init_day']
            init_hh = self.metadata['init_hh']
            date_str = f"{init_year}{init_month}{init_day}_{init_hh}"
            Path(f"{output_dir}/{date_str}").mkdir(parents=True, exist_ok=True)
            
            output_file = f"{output_dir}/{date_str}/hrrrcast_{date_str}.nc"
            logger.info(f"Saving forecast to {output_file}")
            outdata_xr.to_netcdf(output_file)
            
            logger.info("Forecast completed successfully")
            return outdata_xr, output_file
            
        except Exception as e:
            logger.error(f"Forecast failed: {e}")
            raise


def run_weather_forecast(model_path: str, init_year: str, init_month: str,
                        init_day: str, init_hh: str, lead_hours: int,
                        base_dir: str = "./", output_dir: str = "./"):
    """Main forecasting function."""
    try:
        # Load preprocessed data
        date_str = f"{init_year}{init_month}{init_day}_{init_hh}"
        hrrr_preprocessed_file = f"{base_dir}/{date_str}/hrrr_{date_str}.npz"
        gfs_preprocessed_file = f"{base_dir}/{date_str}/gfs_{date_str}.npz"
        data_loader_hrrr = PreprocessedDataLoader(hrrr_preprocessed_file)
        data_loader_gfs = PreprocessedDataLoader(gfs_preprocessed_file)
        
        # Load model
        model = ForecastModel(model_path)
        
        # Initialize forecaster
        forecaster = WeatherForecaster(data_loader_hrrr, data_loader_gfs)
        
        # Run forecast
        forecast_dataset, output_file = forecaster.run_forecast(model, lead_hours, output_dir)
        
        return forecast_dataset, output_file
        
    except Exception as e:
        logger.error(f"Weather forecast failed: {e}")
        raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Weather Forecast Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("model_path", help="Path to the trained model")
    parser.add_argument("init_year", help="Initialization year (YYYY)")
    parser.add_argument("init_month", help="Initialization month (MM)")
    parser.add_argument("init_day", help="Initialization day (DD)")
    parser.add_argument("init_hh", help="Initialization hour (HH)")
    parser.add_argument("lead_hours", type=int, help="Lead time in hours")
    parser.add_argument("--base_dir", default="./", help="Base directory for input preprocessed files")
    parser.add_argument("--output_dir", default="./", help="Output directory for forecast files")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    try:
        args = parse_arguments()
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Run forecast
        forecast_dataset, output_file = run_weather_forecast(
            model_path=args.model_path,
            init_year=args.init_year,
            init_month=args.init_month,
            init_day=args.init_day,
            init_hh=args.init_hh,
            lead_hours=args.lead_hours,
            base_dir=args.base_dir,
            output_dir=args.output_dir
        )
        
        logger.info(f"Forecast complete. Output saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
