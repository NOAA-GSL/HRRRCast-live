#!/usr/bin/env python3
"""
GRIB Preprocessing Script for GFS with HRRR Grid Interpolation

This script processes GFS GRIB files (separate file per lead hour), interpolates them onto the HRRR grid,
and saves the preprocessed data for use by the forecasting model. This stage 
is CPU-intensive and handles all the GRIB file parsing, interpolation, and normalization.

Usage:
    python preprocess_grib_gfs.py <norm_file> <inititme> <lead_hours> [--base_dir DIR] [--output_dir DIR] [--hrrr_grid_file FILE]
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pygrib as pg
import xarray as xr
import xesmf as xe

import utils
from utils import setup_logging

from transform import (
    log_transform_array,
    neg_log_transform_array,
)

logger = None


class WeatherPreprocessConfig:
    """Configuration class for weather preprocessing parameters."""
    
    def __init__(self, hrrr_grid_file: Optional[str] = None):
        # 3D and 2D variables for GFS
        self.pl_vars = ["GFS-HGT", "GFS-SPFH", "GFS-TMP", "GFS-UGRD", "GFS-VGRD", "GFS-VVEL"]
        self.sfc_vars = ["GFS-PRES", "GFS-PRMSL", "GFS-REFC", "GFS-T2M", "GFS-UGRD10M", "GFS-VGRD10M", "GFS-UGRD80M", "GFS-VGRD80M", "GFS-D2M", "GFS-TCDC", "GFS-LCDC", "GFS-MCDC", "GFS-HCDC", "GFS-VIS", "GFS-APCP", "GFS-HGTCC", "GFS-CAPE", "GFS-CIN"]
        
        # Pressure levels (hPa)
        self.levels = [250, 500, 850, 1000]
        
        # Grid downsampling factor (1 = no downsampling; full HRRR grid)
        self.downsample_factor = 1
        
        # HRRR grid specifications (CONUS domain)
        # These are typical HRRR grid dimensions - adjust as needed
        self.hrrr_grid_height = 1059  # Full HRRR grid height
        self.hrrr_grid_width = 1799   # Full HRRR grid width
        
        # Final grid dimensions (full HRRR grid)
        self.grid_height = 1059
        self.grid_width = 1799
        
        # HRRR grid file for reference coordinates
        self.hrrr_grid_file = hrrr_grid_file
        
        # HRRR grid coordinates (will be loaded from file or defined)
        self.hrrr_lats = None
        self.hrrr_lons = None

          # log-transform variables list
        self.LOG_TRANSFORM_VARS = [
            "GFS-VIS",
            "GFS-APCP",
            "GFS-HGTCC",
            "GFS-CAPE",
        ]
        self.NEG_LOG_TRANSFORM_VARS = [
            "GFS-CIN",
        ]    


class GridInterpolator:
    """Handles grid interpolation from GFS to HRRR grid using xESMF."""

    def __init__(self, config: WeatherPreprocessConfig):
        self.config = config
        self.regridder = None
        self.hrrr_coords_loaded = False
        self.hrrr_ds = None

    def load_hrrr_grid_coordinates(self, hrrr_file: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        if hrrr_file and os.path.exists(hrrr_file):
            logger.info(f"Loading HRRR grid coordinates from {hrrr_file}")
            grbs = pg.open(hrrr_file)
            lats, lons = grbs[1].latlons()
            grbs.close()
        else:
            logger.info("Creating HRRR-like CONUS grid coordinates")
            lat_min, lat_max = 21.0, 53.0
            lon_min, lon_max = -134.0, -60.0
            lats = np.linspace(lat_min, lat_max, self.config.hrrr_grid_height)
            lons = np.linspace(lon_min, lon_max, self.config.hrrr_grid_width)
            lons, lats = np.meshgrid(lons, lats)

        # Downsample before interpolation (noop when downsample_factor=1)
        lats_ds = lats[::self.config.downsample_factor, ::self.config.downsample_factor]
        lons_ds = lons[::self.config.downsample_factor, ::self.config.downsample_factor]

        self.config.hrrr_lats, self.config.hrrr_lons = lats_ds, lons_ds
        self.hrrr_ds = xr.Dataset({
            "lat": (("y", "x"), lats_ds),
            "lon": (("y", "x"), lons_ds)
        })
        self.hrrr_coords_loaded = True
        return lats_ds, lons_ds

    def get_regridder(self, gfs_lats, gfs_lons) -> xe.Regridder:
        if self.regridder is None:
            logger.info("Initializing reusable xESMF regridder")
            src_ds = xr.Dataset({
                "lat": ("y", gfs_lats[:, 0]),
                "lon": ("x", gfs_lons[0, :])
            })
            filename = "gfs_to_hrrr_weights.nc"
            self.regridder = xe.Regridder(src_ds, self.hrrr_ds, "bilinear", reuse_weights=True, filename=filename)
        return self.regridder

    def interpolate_to_hrrr_grid(self, gfs_data: np.ndarray, gfs_lats: np.ndarray,
                                 gfs_lons: np.ndarray) -> np.ndarray:
        if not self.hrrr_coords_loaded:
            self.load_hrrr_grid_coordinates(self.config.hrrr_grid_file)

        da = xr.DataArray(gfs_data, dims=("y", "x"), coords={"lat": ("y", gfs_lats[:, 0]), "lon": ("x", gfs_lons[0, :])})
        regridder = self.get_regridder(gfs_lats, gfs_lons)
        regridded = regridder(da)
        return regridded.values


def process_single_lead_hour(args):
    """Process a single lead hour for both pressure and surface variables."""
    lead_time, init_datetime, base_dir, norm_file, hrrr_grid_file = args
    
    # Create a new preprocessor instance for this process
    config = WeatherPreprocessConfig(hrrr_grid_file)
    preprocessor = GRIBPreprocessor(config)
    
    # Get filename for this lead time
    gfs_file = preprocessor.get_valid_time_filename(init_datetime, lead_time, base_dir)
    
    if not os.path.exists(gfs_file):
        raise FileNotFoundError(f"GFS file not found for lead time {lead_time}h: {gfs_file}")
    
    logger.info(f"Processing lead time {lead_time}h from file: {os.path.basename(gfs_file)}")
    
    # Load normalization dataset (per-variable stats)
    ds_norm = xr.open_dataset(norm_file)

    # Open GRIB file
    grbs = pg.open(gfs_file)

    # Get GFS grid coordinates using first geopotential height level
    first_grb = grbs.select(shortName='gh', level=config.levels[0])[0]
    gfs_lats, gfs_lons = first_grb.latlons()

    # Pressure-level variable mappings: GRIB shortName -> (norm var, config name)
    pl_mappings = [
        {'shortName': 'gh', 'cfg': 'GFS-HGT'},
        {'shortName': 'q',  'cfg': 'GFS-SPFH'},
        {'shortName': 't',  'cfg': 'GFS-TMP'},
        {'shortName': 'u',  'cfg': 'GFS-UGRD'},
        {'shortName': 'v',  'cfg': 'GFS-VGRD'},
        {'shortName': 'w',  'cfg': 'GFS-VVEL'},
    ]

    normalized_pl, raw_pl = [], []
    for mapping in pl_mappings:
        short = mapping['shortName']
        cfg_name = mapping['cfg']

        try:
            selected = grbs.select(shortName=short, level=config.levels)
        except Exception as e:
            logger.warning(f"Unable to select pressure variable {short}: {e}")
            continue

        if len(selected) != len(config.levels):
            logger.warning(f"Expected {len(config.levels)} levels for {short} got {len(selected)} (lead {lead_time}h)")

        # Fetch stats array: shape (2, nLevels) if available
        stats = ds_norm[cfg_name].values if cfg_name in ds_norm.variables else None
        if stats is None:
            logger.warning(f"No normalization stats for {cfg_name}; will compute per-level mean/std")

        for l_idx, grb_var in enumerate(selected):
            try:
                vals = grb_var.values
            except Exception as e:
                logger.warning(f"Failed reading values for {short} level {l_idx}: {e}")
                continue

            # Interpolate to HRRR grid
            hrrr_vals = preprocessor.interpolator.interpolate_to_hrrr_grid(vals, gfs_lats, gfs_lons)
            raw_pl.append(hrrr_vals)

            # Apply log transforms where configured
            if cfg_name in config.LOG_TRANSFORM_VARS:
                proc_vals = log_transform_array(hrrr_vals)
            elif cfg_name in config.NEG_LOG_TRANSFORM_VARS:
                proc_vals = neg_log_transform_array(hrrr_vals)
            else:
                proc_vals = hrrr_vals

            if stats is not None and l_idx < stats.shape[1]:
                stat_mean = float(stats[0, l_idx])
                stat_std = float(stats[1, l_idx])
                stat_min = float(stats[2, l_idx]) if stats.shape[0] > 2 else np.nan
                stat_max = float(stats[3, l_idx]) if stats.shape[0] > 3 else np.nan
            else:
                stat_mean = float(np.nanmean(proc_vals))
                stat_std = float(np.nanstd(proc_vals))
                stat_min = np.nan
                stat_max = np.nan
            norm_vals = preprocessor.normalize(proc_vals, stat_mean, stat_std)
            fillv = (stat_max - stat_mean) / stat_std
            if isinstance(norm_vals, np.ma.MaskedArray):
                norm_vals = norm_vals.filled(fillv)
            norm_vals[np.isnan(norm_vals)] = fillv
            logger.info(
                f"Variable {mapping['cfg']}-{l_idx}: stats mean {stat_mean} std {stat_std} min {stat_min} max {stat_max}; "
                f"data min {np.min(proc_vals)} max {np.max(proc_vals)}; "
                f"norm min {np.min(norm_vals)} max {np.max(norm_vals)}"
            )
            normalized_pl.append(norm_vals)

    grbs.close()
    pres_norm = np.array(normalized_pl)
    pres_raw = np.array(raw_pl)

    # Surface variable mappings including height-specific variants
    sfc_mappings = [
        {'shortName': 'sp',     'cfg': 'GFS-PRES'},
        {'shortName': 'prmsl',  'cfg': 'GFS-PRMSL'},
        {'shortName': 'refc',   'cfg': 'GFS-REFC'},
        {'shortName': '2t',     'cfg': 'GFS-T2M'},
        {'shortName': '10u',    'cfg': 'GFS-UGRD10M'},
        {'shortName': '10v',    'cfg': 'GFS-VGRD10M'},
        {'shortName': 'u',      'cfg': 'GFS-UGRD80M', 'typeOfLevel': 'heightAboveGround', 'level': 80},
        {'shortName': 'v',      'cfg': 'GFS-VGRD80M', 'typeOfLevel': 'heightAboveGround', 'level': 80},
        {'shortName': '2d',     'cfg': 'GFS-D2M'},
        {'shortName': 'tcc',    'cfg': 'GFS-TCDC', "typeOfLevel": "atmosphere"},
        {'shortName': 'lcc',    'cfg': 'GFS-LCDC'},
        {'shortName': 'mcc',    'cfg': 'GFS-MCDC'},
        {'shortName': 'hcc',    'cfg': 'GFS-HCDC'},
        {'shortName': 'vis',    'cfg': 'GFS-VIS'},
        {'shortName': 'tp',     'cfg': 'GFS-APCP'},
        {'shortName': 'gh',     'cfg': 'GFS-HGTCC',  'typeOfLevel': 'cloudCeiling'},
        {'shortName': 'cape',   'cfg': 'GFS-CAPE'},
        {'shortName': 'cin',    'cfg': 'GFS-CIN'},
    ]

    grbs = pg.open(gfs_file)
    normalized_sfc, raw_sfc = [], []
    for mapping in sfc_mappings:
        kwargs = {'shortName': mapping['shortName']}
        if 'typeOfLevel' in mapping:
            kwargs['typeOfLevel'] = mapping['typeOfLevel']
        if 'level' in mapping:
            kwargs['level'] = mapping['level']
        try:
            msgs = grbs.select(**kwargs)
            if not msgs:
                logger.warning(f"Surface var {mapping['cfg']} ({kwargs}) not found")
                continue
            vals = msgs[0].values
        except Exception as e:
            logger.warning(f"Failed selecting surface var {mapping['cfg']}: {e}")
            continue

        # Interpolate
        hrrr_vals = preprocessor.interpolator.interpolate_to_hrrr_grid(vals, gfs_lats, gfs_lons)

        # APCP replacement: use nearest synoptic hour strictly greater than valid time if available
        if mapping['cfg'] == 'GFS-APCP':
            try:
                valid_datetime = init_datetime + timedelta(hours=lead_time)
                syn_hours = [0, 6, 12, 18]
                next_syn_hour = next((h for h in syn_hours if h > valid_datetime.hour), None)
                if next_syn_hour is None:
                    future_syn_dt = (valid_datetime + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    future_syn_dt = valid_datetime.replace(hour=next_syn_hour, minute=0, second=0, microsecond=0)
                # Build expected local path (all GFS files stored under init directory)
                # Use preprocessor helper to construct path for future synoptic valid time
                future_lead = int((future_syn_dt - init_datetime).total_seconds() // 3600)
                future_path = preprocessor.get_valid_time_filename(init_datetime, future_lead, base_dir)
                future_fname = os.path.basename(future_path)
                if os.path.exists(future_path):
                    try:
                        grbs_future = pg.open(future_path)
                        future_msgs = grbs_future.select(shortName='tp')
                        if future_msgs:
                            future_vals = future_msgs[0].values
                            future_interp = preprocessor.interpolator.interpolate_to_hrrr_grid(future_vals, gfs_lats, gfs_lons)
                            hrrr_vals = future_interp
                            logger.info(f"Replaced GFS-APCP at lead {lead_time}h using future synoptic APCP from {future_fname} (> valid {valid_datetime:%Y-%m-%d %H}Z)")
                        grbs_future.close()
                    except Exception as fe:
                        logger.warning(f"Failed using future synoptic APCP {future_path}: {fe}")
                else:
                    logger.info(f"Future synoptic APCP file not found ({future_fname}); keeping current lead file values")
            except Exception as e_apcp:
                logger.warning(f"APCP future synoptic replacement error (lead {lead_time}h): {e_apcp}")

        # Clean / enforce constraints
        if mapping['cfg'] == 'GFS-REFC':
            hrrr_vals = np.maximum(hrrr_vals, 0)

        # Transform if needed
        if mapping['cfg'] in config.LOG_TRANSFORM_VARS:
            proc_vals = log_transform_array(hrrr_vals)
        elif mapping['cfg'] in config.NEG_LOG_TRANSFORM_VARS:
            proc_vals = neg_log_transform_array(hrrr_vals)
        else:
            proc_vals = hrrr_vals

        # Stats
        if mapping['cfg'] in ds_norm.variables:
            stats = ds_norm[mapping['cfg']].values
            if stats.shape[0] >= 2:
                stat_mean = float(np.nanmean(stats[0]))
                stat_std = float(np.nanmean(stats[1]))
                stat_min = float(np.nanmean(stats[2])) if stats.shape[0] > 2 else np.nan
                stat_max = float(np.nanmean(stats[3])) if stats.shape[0] > 3 else np.nan
            else:
                stat_mean = float(np.nanmean(proc_vals))
                stat_std = float(np.nanstd(proc_vals))
                stat_min = np.nan
                stat_max = np.nan
        else:
            logger.warning(f"No normalization stats for surface var {mapping['cfg']}; computing from data")
            stat_mean = float(np.nanmean(proc_vals))
            stat_std = float(np.nanstd(proc_vals))
            stat_min = np.nan
            stat_max = np.nan

        norm_vals = preprocessor.normalize(proc_vals, stat_mean, stat_std)
        fillv = (stat_max - stat_mean) / stat_std
        if isinstance(norm_vals, np.ma.MaskedArray):
            norm_vals = norm_vals.filled(fillv)
        norm_vals[np.isnan(norm_vals)] = fillv
        logger.info(
            f"Variable {mapping['cfg']}: stats mean {stat_mean} std {stat_std} min {stat_min} max {stat_max}; "
            f"data min {np.min(proc_vals)} max {np.max(proc_vals)}; "
            f"norm min {np.min(norm_vals)} max {np.max(norm_vals)}"
        )
        normalized_sfc.append(norm_vals)
        raw_sfc.append(hrrr_vals)

    grbs.close()

    return lead_time, np.array(pres_norm), np.array(pres_raw), np.array(normalized_sfc), np.array(raw_sfc)


class GRIBPreprocessor:
    """Handles GRIB file processing and normalization with HRRR grid interpolation."""
    
    def __init__(self, config: WeatherPreprocessConfig):
        self.config = config
        self.interpolator = GridInterpolator(config)
    
    @staticmethod
    def normalize(data: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Normalize data using mean and standard deviation."""
        if std == 0:
            logger.warning("Standard deviation is zero, skipping normalization")
            return data - mean
        return (data - mean) / std
    
    def get_valid_time_filename(self, init_datetime: datetime, lead_hour: int, base_dir: str) -> str:
        """Generate filename based on valid time (init_time + lead_hour)."""
        valid_datetime = init_datetime + timedelta(hours=lead_hour)
        init_date_str = init_datetime.strftime("%Y%m%d/%H")
        valid_date_str = valid_datetime.strftime("%Y%m%d_%H")
        return f"{base_dir}/{init_date_str}/gfs_{valid_date_str}.grib2"
    
    def process_pressure_levels(self, init_datetime: datetime, base_dir: str, norm_file: str, max_lead_hours: int, n_workers: int = 1, skip_zero: bool = False) -> np.ndarray:
        """Load, interpolate to HRRR grid, and normalize pressure-level variables from separate GFS files for all lead times."""
        try:
            logger.info(f"Processing pressure level variables for lead times {'1' if skip_zero else '0'} to {max_lead_hours}h using {n_workers} workers...")
            
            # Prepare arguments for parallel processing
            if skip_zero:
                args_list = [(lead_time, init_datetime, base_dir, norm_file, self.config.hrrr_grid_file) 
                            for lead_time in range(1, max_lead_hours + 1)]
            else:
                args_list = [(lead_time, init_datetime, base_dir, norm_file, self.config.hrrr_grid_file) 
                            for lead_time in range(max_lead_hours + 1)]
            
            all_normalized_vals = [None] * len(args_list)
            
            if n_workers == 1:
                # Sequential processing
                for i, args in enumerate(args_list):
                    lead_time, pres_norm, _, _, _ = process_single_lead_hour(args)
                    all_normalized_vals[i] = pres_norm
            else:
                # Parallel processing
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    # Submit all jobs
                    future_to_idx = {executor.submit(process_single_lead_hour, args): i for i, args in enumerate(args_list)}
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_idx):
                        i = future_to_idx[future]
                        lead_time, pres_norm, _, _, _ = future.result()
                        all_normalized_vals[i] = pres_norm
                        logger.info(f"Completed processing lead time {lead_time}h")
            
            return np.array(all_normalized_vals)
            
        except Exception as e:
            logger.error(f"Error processing pressure levels: {e}")
            raise
    
    def process_surface_variables(self, init_datetime: datetime, base_dir: str, norm_file: str, max_lead_hours: int, n_workers: int = 1, skip_zero: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load, interpolate to HRRR grid, and normalize surface variables from separate GFS files for all lead times."""
        try:
            logger.info(f"Processing surface variables for lead times {'1' if skip_zero else '0'} to {max_lead_hours}h using {n_workers} workers...")
            
            # Get HRRR coordinates for output
            if not self.interpolator.hrrr_coords_loaded:
                self.config.hrrr_lats, self.config.hrrr_lons = self.interpolator.load_hrrr_grid_coordinates(
                    self.config.hrrr_grid_file
                )
                self.interpolator.hrrr_coords_loaded = True
            
            # HRRR coordinates (full grid when downsample_factor=1)
            hrrr_lats_ds = self.config.hrrr_lats
            hrrr_lons_ds = self.config.hrrr_lons
            
            logger.info(f"Final grid shape after HRRR interpolation: {hrrr_lats_ds.shape}")
            
            # Prepare arguments for parallel processing
            if skip_zero:
                args_list = [(lead_time, init_datetime, base_dir, norm_file, self.config.hrrr_grid_file) 
                            for lead_time in range(1, max_lead_hours + 1)]
            else:
                args_list = [(lead_time, init_datetime, base_dir, norm_file, self.config.hrrr_grid_file) 
                            for lead_time in range(max_lead_hours + 1)]
            
            all_normalized = [None] * len(args_list)
            
            if n_workers == 1:
                # Sequential processing
                for i, args in enumerate(args_list):
                    lead_time, _, _, sfc_norm, _ = process_single_lead_hour(args)
                    all_normalized[i] = sfc_norm
            else:
                # Parallel processing
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    # Submit all jobs
                    future_to_idx = {executor.submit(process_single_lead_hour, args): i for i, args in enumerate(args_list)}
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_idx):
                        i = future_to_idx[future]
                        lead_time, _, _, sfc_norm, _ = future.result()
                        all_normalized[i] = sfc_norm
                        logger.info(f"Completed processing surface variables for lead time {lead_time}h")
            
            return np.array(all_normalized), hrrr_lats_ds, hrrr_lons_ds
            
        except Exception as e:
            logger.error(f"Error processing surface variables: {e}")
            raise
    
    def save_preprocessed_data(self, output_file: str, pres_norm: np.ndarray,
                              sfc_norm: np.ndarray, lats: np.ndarray, 
                              lons: np.ndarray, metadata: Dict) -> None:
        """Save preprocessed data for all lead times to compressed numpy format."""
        try:
            logger.info(f"Saving preprocessed data to {output_file}")
            
            # Create model input arrays for all lead times
            num_lead_times = pres_norm.shape[0]
            model_inputs = []
            
            for lead_idx in range(num_lead_times):
                # Concatenate pressure, surface
                model_input = np.concatenate((pres_norm[lead_idx], sfc_norm[lead_idx]), axis=0)
                model_input = np.transpose(model_input, (1, 2, 0))
                model_inputs.append(model_input)
            
            # Stack all lead times
            model_inputs = np.array(model_inputs)  # Shape: (lead_times, height, width, channels)
            
            # Save all data in compressed format
            np.savez_compressed(
                output_file,
                # Model input (ready for inference) - all lead times
                model_input=model_inputs,
                # Coordinate information (same for all lead times)
                lats=lats,
                lons=lons,
                # Metadata
                **metadata
            )
            
            logger.info(f"Preprocessed data saved successfully")
            logger.info(f"Model input shape: {model_inputs.shape}")
            logger.info(f"Number of lead times processed: {num_lead_times}")
            logger.info(f"Grid dimensions (HRRR-based): {self.config.grid_height} x {self.config.grid_width}")
            
        except Exception as e:
            logger.error(f"Error saving preprocessed data: {e}")
            raise


def preprocess_grib_data(norm_file: str, datetime_str: str,
                        lead_hours: str,
                        base_dir: str = "./", output_dir: str = "./", 
                        hrrr_grid_file: Optional[str] = None):
    """Main preprocessing function for GFS data with HRRR grid interpolation - processes all lead times from separate files (skipping 0th hour)."""
    try:
        # Validate inputs
        max_lead_time = int(lead_hours)
        logger.info(f"Preprocessing GFS data initialized at {datetime_str} with lead times 1 to {max_lead_time}h (skipping 0th hour)")
        logger.info("Data will be interpolated to HRRR grid (no downsampling)")
        logger.info("Reading from separate GRIB files for each lead time based on valid time")
        
        # Set n_workers to number of lead hours (1 to max_lead_time inclusive)
        n_workers = max_lead_time if max_lead_time > 0 else 1
        logger.info(f"Using {n_workers} worker processes for parallel processing (one per lead hour, skipping 0th hour)")
        
        # Setup paths
        init_datetime, init_year, init_month, init_day, init_hh = utils.validate_datetime(datetime_str)
        date_str = f"{init_year}{init_month}{init_day}/{init_hh}"
        filedate_str = f"{init_year}{init_month}{init_day}_{init_hh}"
        
        # Create output directory if it doesn't exist
        utils.make_directory(f"{output_dir}/{date_str}")
        output_file = f"{output_dir}/{date_str}/gfs_{filedate_str}.npz"
        
        # Validate normalization file exists
        if not os.path.exists(norm_file):
            raise FileNotFoundError(f"Normalization file not found: {norm_file}")
        
        # Check if all required GRIB files exist
        missing_files = []
        preprocessor = GRIBPreprocessor(WeatherPreprocessConfig(hrrr_grid_file))
        
        for lead_time in range(1, max_lead_time + 1):
            gfs_file = preprocessor.get_valid_time_filename(init_datetime, lead_time, base_dir)
            if not os.path.exists(gfs_file):
                missing_files.append(f"Lead {lead_time}h: {gfs_file}")
        
        if missing_files:
            logger.error("Missing GRIB files:")
            for missing in missing_files:
                logger.error(f"  {missing}")
            raise FileNotFoundError(f"Missing {len(missing_files)} GRIB files")
        
        logger.info(f"All required GRIB files found for lead times 1 to {max_lead_time}h")
        
        # --- Ensure xESMF weights file exists before parallel jobs ---
        weights_file = "gfs_to_hrrr_weights.nc"
        if not os.path.exists(weights_file):
            logger.info(f"Weights file {weights_file} not found. Creating it serially before parallel processing...")
            # Use the first available GFS file to get grid info
            first_gfs_file = preprocessor.get_valid_time_filename(init_datetime, 1, base_dir)
            grbs = pg.open(first_gfs_file)
            first_grb = grbs.select(shortName='gh', level=preprocessor.config.levels[0])[0]
            gfs_lats, gfs_lons = first_grb.latlons()
            grbs.close()
            # Create src_ds and tgt_ds
            src_ds = xr.Dataset({
                "lat": ("y", gfs_lats[:, 0]),
                "lon": ("x", gfs_lons[0, :])
            })
            # Load HRRR grid
            lats_ds, lons_ds = preprocessor.interpolator.load_hrrr_grid_coordinates(preprocessor.config.hrrr_grid_file)
            tgt_ds = xr.Dataset({
                "lat": (("y", "x"), lats_ds),
                "lon": (("y", "x"), lons_ds)
            })
            # Create weights file
            xe.Regridder(src_ds, tgt_ds, "bilinear", filename=weights_file, reuse_weights=False)
            logger.info(f"Weights file {weights_file} created successfully.")
        else:
            logger.info(f"Weights file {weights_file} already exists. Will be reused by all parallel jobs.")
        # --- End weights file creation ---
        
        # Initialize preprocessor with HRRR grid configuration
        config = WeatherPreprocessConfig(hrrr_grid_file)
        preprocessor = GRIBPreprocessor(config)
        
        # Process GRIB data for all lead times (skipping 0th hour)
        logger.info("Processing pressure level data for all lead times with HRRR grid interpolation...")
        pres_norm = preprocessor.process_pressure_levels(init_datetime, base_dir, norm_file, max_lead_time, n_workers, skip_zero=True)
        
        logger.info("Processing surface data for all lead times with HRRR grid interpolation...")
        sfc_norm, lats, lons = preprocessor.process_surface_variables(init_datetime, base_dir, norm_file, max_lead_time, n_workers, skip_zero=True)
        
        # Validate grid dimensions
        expected_shape = (config.grid_height, config.grid_width)
        if pres_norm.shape[2:] != expected_shape:
            logger.warning(f"Unexpected grid shape: {pres_norm.shape[2:]} vs expected {expected_shape}")
        
        # Prepare metadata
        metadata = {
            'init_year': init_year,
            'init_month': init_month,
            'init_day': init_day,
            'init_hh': init_hh,
            'max_lead_hours': lead_hours,
            'lead_times': list(range(1, max_lead_time + 1)),
            'init_datetime': init_datetime.isoformat(),
            'pl_vars': config.pl_vars,
            'sfc_vars': config.sfc_vars,
            'levels': config.levels,
            'grid_height': config.grid_height,
            'grid_width': config.grid_width,
            'hrrr_grid_height': config.hrrr_grid_height,
            'hrrr_grid_width': config.hrrr_grid_width,
            'downsample_factor': config.downsample_factor,
            'norm_file': norm_file,
            'hrrr_grid_file': hrrr_grid_file,
            'model': 'GFS',
            'target_grid': 'HRRR',
            'interpolation_method': 'linear',
            'file_structure': 'separate_files_per_lead_time',
            'filename_convention': 'gfs_YYYYMMDD_HH.grib2 (valid_time based)'
        }
        
        # Save preprocessed data
        preprocessor.save_preprocessed_data(
            output_file, pres_norm, sfc_norm, lats, lons, metadata
        )
        
        logger.info("GFS to HRRR grid preprocessing completed successfully")
        return output_file
        
    except Exception as e:
        logger.error(f"GFS to HRRR grid preprocessing failed: {e}")
        raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GFS GRIB Data Preprocessing with HRRR Grid Interpolation (Separate files per lead time)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("norm_file", help="Path to the normalization file")
    parser.add_argument('inittime',
                       help='Forecast initialization time in format YYYY-MM-DDTHH (e.g., "2024-05-06T23")')
    parser.add_argument("lead_hours", help="Maximum lead time in hours (will process 0 to lead_hours)")
    parser.add_argument("--base_dir", default="./", help="Base directory for input GRIB files")
    parser.add_argument("--output_dir", default="./", help="Output directory for preprocessed data")
    parser.add_argument("--hrrr_grid_file", help="Optional HRRR GRIB file to extract grid coordinates from")
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
            lead_hours=args.lead_hours,
            base_dir=args.base_dir,
            output_dir=args.output_dir,
            hrrr_grid_file=args.hrrr_grid_file
        )
        logger.info(f"Preprocessing complete. Output saved to: {output_file}")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
