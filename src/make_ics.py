#!/usr/bin/env python3
"""
GRIB Preprocessing Script

This script processes HRRR GRIB files and saves the preprocessed data for use
by the forecasting model. This stage is CPU-intensive and handles all the 
GRIB file parsing and normalization.

Usage:
    python preprocess_grib.py <norm_file> <year> <month> <day> <hour> [--base_dir DIR] [--output_dir DIR]
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pygrib as pg
import xarray as xr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeatherPreprocessConfig:
    """Configuration class for weather preprocessing parameters."""
    
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


class GRIBPreprocessor:
    """Handles GRIB file processing and normalization."""
    
    def __init__(self, config: WeatherPreprocessConfig):
        self.config = config
    
    @staticmethod
    def normalize(data: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Normalize data using mean and standard deviation."""
        if std == 0:
            logger.warning("Standard deviation is zero, skipping normalization")
            return data - mean
        return (data - mean) / std
    
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
    
    def save_preprocessed_data(self, output_file: str, pres_norm: np.ndarray, pres_raw: np.ndarray,
                              sfc_norm: np.ndarray, sfc_raw: np.ndarray, lats: np.ndarray, 
                              lons: np.ndarray, metadata: Dict) -> None:
        """Save preprocessed data to compressed numpy format."""
        try:
            logger.info(f"Saving preprocessed data to {output_file}")
            
            # Create model input array
            lead_channel = np.ones((1, self.config.grid_height, self.config.grid_width))
            model_input = np.concatenate((pres_norm, sfc_norm, lead_channel), axis=0)
            model_input = np.transpose(model_input, (1, 2, 0))[None, ...]
            
            # Save all data in compressed format
            np.savez_compressed(
                output_file,
                # Model input (ready for inference)
                model_input=model_input,
                # Raw data for potential analysis
                pres_raw=pres_raw,
                sfc_raw=sfc_raw,
                # Coordinate information
                lats=lats,
                lons=lons,
                # Metadata
                **metadata
            )
            
            logger.info(f"Preprocessed data saved successfully")
            logger.info(f"Model input shape: {model_input.shape}")
            
        except Exception as e:
            logger.error(f"Error saving preprocessed data: {e}")
            raise


def preprocess_grib_data(norm_file: str, init_year: str, init_month: str, 
                        init_day: str, init_hh: str, base_dir: str = "./", 
                        output_dir: str = "./"):
    """Main preprocessing function."""
    try:
        # Validate inputs
        init_datetime = datetime.strptime(f"{init_year}{init_month}{init_day}_{init_hh}", "%Y%m%d_%H")
        logger.info(f"Preprocessing GRIB data for {init_datetime}")
        
        # Setup paths
        date_str = f"{init_year}{init_month}{init_day}_{init_hh}"
        hrrr_pres_file = f"{base_dir}/{date_str}/hrrr_{date_str}_pressure.grib2"
        hrrr_sfc_file = f"{base_dir}/{date_str}/hrrr_{date_str}_surface.grib2"
        
        # Create output directory if it doesn't exist
        Path(f"{output_dir}/{date_str}").mkdir(parents=True, exist_ok=True)
        output_file = f"{output_dir}/{date_str}/hrrr_{date_str}.npz"
        
        # Validate required files exist
        for file_path in [norm_file, hrrr_pres_file, hrrr_sfc_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Initialize preprocessor
        config = WeatherPreprocessConfig()
        preprocessor = GRIBPreprocessor(config)
        
        # Process GRIB data
        logger.info("Processing pressure level data...")
        pres_norm, pres_raw = preprocessor.process_pressure_levels(hrrr_pres_file, norm_file)
        
        logger.info("Processing surface data...")
        sfc_norm, sfc_raw, lats, lons = preprocessor.process_surface_variables(hrrr_sfc_file, norm_file)
        
        # Validate grid dimensions
        expected_shape = (config.grid_height, config.grid_width)
        if pres_norm.shape[1:] != expected_shape:
            logger.warning(f"Unexpected grid shape: {pres_norm.shape[1:]} vs expected {expected_shape}")
        
        # Prepare metadata
        metadata = {
            'init_year': init_year,
            'init_month': init_month,
            'init_day': init_day,
            'init_hh': init_hh,
            'init_datetime': init_datetime.isoformat(),
            'pl_vars': config.pl_vars,
            'sfc_vars': config.sfc_vars,
            'levels': config.levels,
            'grid_height': config.grid_height,
            'grid_width': config.grid_width,
            'downsample_factor': config.downsample_factor,
            'norm_file': norm_file
        }
        
        # Save preprocessed data
        preprocessor.save_preprocessed_data(
            output_file, pres_norm, pres_raw, sfc_norm, sfc_raw, lats, lons, metadata
        )
        
        logger.info("GRIB preprocessing completed successfully")
        return output_file
        
    except Exception as e:
        logger.error(f"GRIB preprocessing failed: {e}")
        raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GRIB Data Preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("norm_file", help="Path to the normalization file")
    parser.add_argument("init_year", help="Initialization year (YYYY)")
    parser.add_argument("init_month", help="Initialization month (MM)")
    parser.add_argument("init_day", help="Initialization day (DD)")
    parser.add_argument("init_hh", help="Initialization hour (HH)")
    parser.add_argument("--base_dir", default="./", help="Base directory for input GRIB files")
    parser.add_argument("--output_dir", default="./", help="Output directory for preprocessed data")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    try:
        args = parse_arguments()
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Run preprocessing
        output_file = preprocess_grib_data(
            norm_file=args.norm_file,
            init_year=args.init_year,
            init_month=args.init_month,
            init_day=args.init_day,
            init_hh=args.init_hh,
            base_dir=args.base_dir,
            output_dir=args.output_dir
        )
        
        logger.info(f"Preprocessing complete. Output saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
