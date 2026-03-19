#!/usr/bin/env python3
"""
GRIB Preprocessing Script

This script processes HRRR GRIB files and saves the preprocessed data for use
by the forecasting model. This stage is CPU-intensive and handles all the 
GRIB file parsing and normalization.

Usage:
    python preprocess_grib.py <norm_file> <inittime> [--base_dir DIR] [--output_dir DIR]
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pygrib as pg
import xarray as xr

import utils
from transform import (
    log_transform_array,
    neg_log_transform_array,
)

from utils import setup_logging

logger = None

class WeatherPreprocessConfig:
    """Configuration class for weather preprocessing parameters."""
    
    def __init__(self):
        # Pressure level and surface variables
        self.pl_vars = ["UGRD", "VGRD", "VVEL", "TMP", "HGT", "SPFH"]
        self.sfc_vars = ["PRES", "MSLMA", "REFC", "T2M", "UGRD10M", "VGRD10M", "UGRD80M", "VGRD80M", "D2M", "TCDC", "LCDC", "MCDC", "HCDC", "VIS", "APCP", "HGTCC", "CAPE", "CIN"]
        self.consts = ["LAND", "OROG"]
        
        # Pressure levels (hPa)
        self.levels = [200, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
        
        # Grid downsampling factor (1 = no downsampling; full HRRR grid)
        self.downsample_factor = 1
        
        # Expected grid dimensions (full HRRR grid)
        self.grid_height = 1059
        self.grid_width = 1799

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
        # Mapping from logical surface variable names to GRIB query parameters
        # Keys correspond to entries in sfc_vars or consts. Values specify how to
        # retrieve the field from the GRIB file using pygrib.select(). If 'level'
        # and 'typeOfLevel' are provided they are included in the selection.
        self.SFC_GRIB_MAP = {
            # Basic surface / near-sfc fields
            "PRES": {"shortName": "sp"},
            "MSLMA": {"shortName": "mslma"},  # MSL pressure (may differ on some systems)
            "REFC": {"shortName": "refc"},
            "T2M": {"shortName": "2t"},
            "UGRD10M": {"shortName": "10u"},
            "VGRD10M": {"shortName": "10v"},
            # 80 m winds often stored as ugrd/vgrd at heightAboveGround=80
            "UGRD80M": {"shortName": "u", "typeOfLevel": "heightAboveGround", "level": 80},
            "VGRD80M": {"shortName": "v", "typeOfLevel": "heightAboveGround", "level": 80},
            "D2M": {"shortName": "2d"},
            "TCDC": {"shortName": "tcc", "typeOfLevel": "atmosphere"},
            "LCDC": {"shortName": "lcc"},
            "MCDC": {"shortName": "mcc"},
            "HCDC": {"shortName": "hcc"},
            "VIS": {"shortName": "vis"},
            "APCP": {"shortName": "tp"},  # accumulated precip
            "HGTCC": {"shortName": "gh", "typeOfLevel": "cloudCeiling"},  # cloud ceiling height (may vary)
            "CAPE": {"shortName": "cape"},
            "CIN": {"shortName": "cin"},
            # Constants
            "LAND": {"shortName": "lsm"},
            "OROG": {"shortName": "orog"},
        }


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
        """Load and normalize pressure-level variables using per-variable stats.

        The normalization file now contains separate variables (UGRD, VGRD, VVEL, TMP, HGT, SPFH)
        each with shape (stat, level) where stat is ordered [mean, std, min, max]. We apply log
        transforms (e.g., SPFH) before normalization consistent with transform.log_transform_dataset
        logic.
        """
        if not os.path.exists(pres_file):
            raise FileNotFoundError(f"Pressure file not found: {pres_file}")

        try:
            grbs = pg.open(pres_file)
            ds_norm = xr.open_dataset(norm_file)

            # Mapping between GRIB shortNames and normalization variable names
            grib_to_norm = {
                'u': 'UGRD',
                'v': 'VGRD',
                'w': 'VVEL',
                't': 'TMP',
                'gh': 'HGT',
                'q': 'SPFH',
            }
            varnames = list(grib_to_norm.keys())

            normalized_vals, raw_vals = [], []
            logger.info("Processing pressure level variables with per-variable normalization...")

            for var in varnames:
                norm_name = grib_to_norm[var]
                if norm_name not in ds_norm.variables:
                    logger.warning(f"Normalization stats for {norm_name} not found; skipping variable {var}")
                    continue
                stats = ds_norm[norm_name].values  # shape (stat, nLevels)
                if stats.shape[0] < 2:
                    logger.error(f"Stats for {norm_name} malformed (expected first dim size >=2), skipping")
                    continue

                selected = grbs.select(shortName=var, level=self.config.levels)
                if len(selected) != len(self.config.levels):
                    logger.warning(f"Expected {len(self.config.levels)} levels for {var}, got {len(selected)}")

                for l_idx, grb in enumerate(selected):
                    try:
                        vals = grb.values[::self.config.downsample_factor, ::self.config.downsample_factor]
                    except Exception as e:
                        logger.warning(f"Failed reading data for {var} level index {l_idx}: {e}")
                        continue

                    # Store raw (pre-transform) values
                    raw_vals.append(vals)

                    # Apply log / signed-log transforms if configured (e.g., SPFH)
                    if norm_name in self.config.LOG_TRANSFORM_VARS:
                        vals_proc = log_transform_array(vals)
                    elif norm_name in self.config.NEG_LOG_TRANSFORM_VARS:
                        vals_proc = neg_log_transform_array(vals)
                    else:
                        vals_proc = vals

                    # Per-level stats
                    if l_idx >= stats.shape[1]:
                        logger.error(f"Level index {l_idx} out of bounds for stats {norm_name} (shape {stats.shape})")
                        continue
                    stat_mean = float(stats[0, l_idx])
                    stat_std = float(stats[1, l_idx])
                    stat_min = float(stats[2, l_idx]) if stats.shape[0] > 2 else np.nan
                    stat_max = float(stats[3, l_idx]) if stats.shape[0] > 3 else np.nan
                    norm_vals = self.normalize(vals_proc, stat_mean, stat_std)
                    fillv = (stat_max - stat_mean) / stat_std
                    if isinstance(norm_vals, np.ma.MaskedArray):
                        norm_vals = norm_vals.filled(fillv)
                    norm_vals[np.isnan(norm_vals)] = fillv
                    logger.info(
                        f"Variable {var}: stats mean {stat_mean} std {stat_std} min {stat_min} max {stat_max}; "
                        f"data min {np.min(vals_proc)} max {np.max(vals_proc)}; "
                        f"norm min {np.min(norm_vals)} max {np.max(norm_vals)}"
                    )
                    normalized_vals.append(norm_vals)

            grbs.close()

            if not normalized_vals:
                raise RuntimeError("No pressure level variables were successfully processed")

            return np.array(normalized_vals)

        except Exception as e:
            logger.error(f"Error processing pressure levels: {e}")
            raise
    
    def process_surface_variables(self, sfc_file: str, norm_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Load and normalize surface and constant variables using per-variable stats.

        Returns:
            sfc_norm (np.ndarray): Normalized surface/constant variables (Nvar, Ny, Nx)
            sfc_raw (np.ndarray): Raw (unnormalized) surface variables (Nvar_raw, Ny, Nx)
            lats (np.ndarray): Latitude grid (Ny, Nx)
            lons (np.ndarray): Longitude grid (Ny, Nx)
        """
        if not os.path.exists(sfc_file):
            raise FileNotFoundError(f"Surface file not found: {sfc_file}")

        try:
            grbs = pg.open(sfc_file)
            ds_norm = xr.open_dataset(norm_file)  # new format: each variable has its own (stat, ...)

            # Lat/lon grid (take from first record)
            lats, lons = grbs[1].latlons()
            lats = lats[::self.config.downsample_factor, ::self.config.downsample_factor]
            lons = lons[::self.config.downsample_factor, ::self.config.downsample_factor]

            norm_arrays = []  # normalized data for all surface + constants
            const_raw: Dict[str, np.ndarray] = {}  # store original (unnormalized) LAND/OROG

            def fetch_grib_field(var_name: str):
                """Retrieve a GRIB field based on mapping; return None if not found."""
                if var_name not in self.config.SFC_GRIB_MAP:
                    return None
                query = self.config.SFC_GRIB_MAP[var_name]
                try:
                    select_args = {k: v for k, v in query.items() if k != 'shortName'}
                    msgs = grbs.select(shortName=query['shortName'], **select_args)
                    if not msgs:
                        logger.warning(f"GRIB field for {var_name} ({query}) not found")
                        return None
                    return msgs[0].values[::self.config.downsample_factor, ::self.config.downsample_factor]
                except Exception as e:
                    logger.warning(f"Failed to retrieve {var_name}: {e}")
                    return None

            # Attempt to load previous hour 1h surface forecast file (downloaded alongside analysis) for APCP replacement
            prev_apcp_vals = None
            try:
                base_name = os.path.basename(sfc_file)  # hrrr_YYYYMMDD_HH_surface.grib2
                parts = base_name.split('_')
                if len(parts) >= 4:
                    ymd = parts[1]
                    hh = parts[2]
                    cur_dt = datetime.strptime(ymd+hh, "%Y%m%d%H")
                    prev_dt = cur_dt - timedelta(hours=1)
                    prev_ymd = prev_dt.strftime("%Y%m%d")
                    prev_hh = prev_dt.strftime("%H")
                    f01_name = f"hrrr_{prev_ymd}_{prev_hh}_surface_f01.grib2"
                    f01_path = os.path.join(os.path.dirname(sfc_file), f01_name)
                    if os.path.exists(f01_path):
                        try:
                            grbs_prev = pg.open(f01_path)
                            tp_msgs = grbs_prev.select(shortName='tp')
                            if tp_msgs:
                                prev_apcp_vals = tp_msgs[0].values[::self.config.downsample_factor, ::self.config.downsample_factor]
                                logger.info(f"Loaded APCP from previous hour forecast file {f01_name} (valid ending {cur_dt})")
                            grbs_prev.close()
                        except Exception as e:
                            logger.warning(f"Failed reading previous hour forecast APCP from {f01_path}: {e}")
                    else:
                        logger.warning(f"Previous hour APCP forecast file not found: {f01_name}")
            except Exception as e:
                logger.warning(f"Error preparing previous hour APCP replacement: {e}")

            for var in self.config.sfc_vars + self.config.consts:
                vals = fetch_grib_field(var)
                if vals is None:
                    continue

                # Clean / transform as needed
                if var == "REFC":
                    vals = np.maximum(vals, 0)
                # Always prefer previous hour 1h accumulation APCP if available
                if var == "APCP" and prev_apcp_vals is not None:
                    logger.info("Using previous hour 1h forecast APCP values (replacing analysis APCP)")
                    vals = prev_apcp_vals
                
                if var in self.config.LOG_TRANSFORM_VARS:
                    vals_proc = log_transform_array(vals)
                elif var in self.config.NEG_LOG_TRANSFORM_VARS:
                    vals_proc = neg_log_transform_array(vals)
                else:
                    vals_proc = vals

                # Obtain mean/std from normalization file if present; else compute on-the-fly
                if var in ds_norm.variables:
                    stats = ds_norm[var].values
                    # stats shape: (stat, ...). Reduce any extra dimensions via nanmean
                    stat_mean = float(np.nanmean(stats[0]))
                    stat_std = float(np.nanmean(stats[1]))
                    stat_min = float(np.nanmean(stats[2])) if stats.shape[0] > 2 else np.nan
                    stat_max = float(np.nanmean(stats[3])) if stats.shape[0] > 3 else np.nan
                else:
                    logger.warning(f"Normalization stats for {var} not found in {norm_file}; computing from data")
                    stat_mean = float(np.nanmean(vals_proc))
                    stat_std = float(np.nanstd(vals_proc))
                    stat_min = np.nan
                    stat_max = np.nan

                norm_vals = self.normalize(vals_proc, stat_mean, stat_std)
                fillv = (stat_max - stat_mean) / stat_std
                if isinstance(norm_vals, np.ma.MaskedArray):
                    norm_vals = norm_vals.filled(fillv)
                norm_vals[np.isnan(norm_vals)] = fillv
                logger.info(
                    f"Variable {var}: stats mean {stat_mean}, std {stat_std}, min {stat_min}, max {stat_max}; "
                    f"data min {np.min(vals_proc)}, max {np.max(vals_proc)}; "
                    f"norm min {np.min(norm_vals)}, max {np.max(norm_vals)}"
                )
                norm_arrays.append(norm_vals)

                if var in self.config.consts:
                    # Store original unnormalized constant values for later denormalized output
                    const_raw[var] = vals

            grbs.close()
            if not norm_arrays:
                raise RuntimeError("No surface variables were successfully processed")

            return np.array(norm_arrays), lats, lons, const_raw

        except Exception as e:
            logger.error(f"Error processing surface variables: {e}")
            raise
    
    def save_preprocessed_data(self, output_file: str, pres_norm: np.ndarray,
                              sfc_norm: np.ndarray, const_raw: Dict[str, np.ndarray],
                              lats: np.ndarray, lons: np.ndarray, metadata: Dict) -> None:
        """Save preprocessed data to compressed numpy format."""
        try:
            logger.info(f"Saving preprocessed data to {output_file}")
            
            # Create model input array
            model_input = np.concatenate((pres_norm, sfc_norm), axis=0)
            model_input = np.transpose(model_input, (1, 2, 0))[None, ...]
            
            # Save all data in compressed format
            save_kwargs = dict(
                model_input=model_input,
                lats=lats,
                lons=lons,
                **metadata,
            )
            # Add raw constants if present
            for cname in ["LAND", "OROG"]:
                if cname in const_raw:
                    save_kwargs[f"{cname}_raw"] = const_raw[cname]

            np.savez_compressed(output_file, **save_kwargs)
            
            logger.info(f"Preprocessed data saved successfully")
            logger.info(f"Model input shape: {model_input.shape}")
            
        except Exception as e:
            logger.error(f"Error saving preprocessed data: {e}")
            raise


def preprocess_grib_data(norm_file: str, datetime_str: str,
                        base_dir: str = "./", output_dir: str = "./"):
    """Main preprocessing function."""
    try:
        # Validate inputs
        logger.info(f"Preprocessing GRIB data for {datetime_str}")
        
        # Setup paths
        init_datetime, init_year, init_month, init_day, init_hh = utils.validate_datetime(datetime_str)
        date_str = f"{init_year}{init_month}{init_day}/{init_hh}"
        filedate_str = f"{init_year}{init_month}{init_day}_{init_hh}"
        hrrr_pres_file = f"{base_dir}/{date_str}/hrrr_{filedate_str}_pressure.grib2"
        hrrr_sfc_file = f"{base_dir}/{date_str}/hrrr_{filedate_str}_surface.grib2"
        
        # Create output directory if it doesn't exist
        utils.make_directory(f"{output_dir}/{date_str}")
        output_file = f"{output_dir}/{date_str}/hrrr_{filedate_str}.npz"
        
        # Validate required files exist
        for file_path in [norm_file, hrrr_pres_file, hrrr_sfc_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Initialize preprocessor
        config = WeatherPreprocessConfig()
        preprocessor = GRIBPreprocessor(config)
        
        # Process GRIB data
        logger.info("Processing pressure level data...")
        pres_norm = preprocessor.process_pressure_levels(hrrr_pres_file, norm_file)
        
        logger.info("Processing surface data...")
        sfc_norm, lats, lons, const_raw = preprocessor.process_surface_variables(hrrr_sfc_file, norm_file)
        
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
            output_file, pres_norm, sfc_norm, const_raw, lats, lons, metadata
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
    parser.add_argument('inittime',
                       help='Forecast initialization time in format YYYY-MM-DDTHH (e.g., "2024-05-06T23")')
    parser.add_argument("--base_dir", default="./", help="Base directory for input GRIB files")
    parser.add_argument("--output_dir", default="./", help="Output directory for preprocessed data")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    global logger
    args = parse_arguments()
    logger = setup_logging(args.log_level)

    try:
        output_file = preprocess_grib_data(
            norm_file=args.norm_file,
            datetime_str=args.inittime,
            base_dir=args.base_dir,
            output_dir=args.output_dir
        )
        logger.info(f"Preprocessing complete. Output saved to: {output_file}")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
