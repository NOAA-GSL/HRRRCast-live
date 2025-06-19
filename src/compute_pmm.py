#!/usr/bin/env python3
"""
Ensemble Post-Processing Script for HRRR Forecasts

This script processes ensemble forecast data from HRRR (High-Resolution Rapid Refresh) 
model runs and computes post-processed ensemble products. It applies different statistical
methods based on the variable type:

- REFC (Reflectivity): Uses Probability-Matched Mean (PMM) to preserve the natural 
  distribution and spatial structure of precipitation-related fields
- All other variables: Uses standard ensemble mean which is appropriate for variables
  like temperature, wind, pressure, etc.

The Probability-Matched Mean method addresses the common problem where simple ensemble
averaging of precipitation-related variables creates unrealistically smooth fields with
underestimated extremes. PMM preserves the distribution of the ensemble mean while
maintaining the spatial structure of individual ensemble members.

Input files should follow the naming convention:
  YYYYMMDD_HH/hrrrcast_YYYYMMDD_HH_memN.nc

Output files are saved as:
  YYYYMMDD_HH/hrrrcast_YYYYMMDD_HH_processed.nc

Usage:
  python ensemble_postprocess.py "2024-05-06T23" --forecast_dir /path/to/data
"""
import argparse
import logging
import os
import sys 
from datetime import datetime, timedelta
from dateutil import parser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import xarray as xr
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def compute_PMM(fields: xr.DataArray, method=2) -> xr.DataArray:
    """ 
    Compute Probability-Matched Mean (PMM) for an xarray DataArray.
    
    Expects input with spatial dimensions (latitude, longitude) and member dimension.
    For the HRRR dataset, this will typically be called on slices with dimensions (lat, lon, member).
    
    Parameters:
    - fields: xarray.DataArray with dimensions (lat, lon, member) or similar spatial + member dims
    - method: 1 for sorting per member, 2 for sorting all values together
    Returns:
    - PMM: xarray.DataArray with the same dimensions as input, minus 'member'
    """
    if "member" not in fields.dims:
        raise ValueError("Input DataArray must have a 'member' dimension.")
    
    # Determine spatial dimension names (handle latitude/lat and longitude/lon variations)
    spatial_dims = []
    for dim in fields.dims:
        if dim in ['latitude', 'lat', 'longitude', 'lon', 'x', 'y'] and dim != 'member':
            spatial_dims.append(dim)
    
    if len(spatial_dims) < 2:
        raise ValueError(f"Could not identify spatial dimensions. Available dims: {fields.dims}")
    
    # print info for debugging
    if 'lead_time' in fields.coords:
        lt = fields.lead_time.values / np.timedelta64(1, "h")
        print(f"Lead time {lt}h", end=" ")
    if 'level' in fields.coords:
        lv = fields.level.values
        print(f"Level {lv}", end=" ")
    if 'time' in fields.coords:
        print(f"Time {fields.time.values}")
    else:
        print()  # Just newline if no time coord
    
    # Load data
    fields = fields.compute()
    
    # Compute the simple ensemble mean along the member dimension
    field_mean = fields.mean(dim="member")
    
    # Get sorted indices of the flattened mean field
    sort_indices = np.argsort(field_mean.data.flatten())
    
    # Reshape the input fields for easier manipulation (flatten spatial dims)
    stacked_fields = fields.stack(flat=spatial_dims, create_index=False)
    
    if method == 1:
        # Method 1: Sort each case individually, then average
        sorted_per_member = []
        for member in fields.member:
            member_data = fields.sel(member=member)
            sorted_per_member.append(np.ma.sort(np.ma.ravel(member_data.values)))
        sorted_per_member = np.ma.array(
            sorted_per_member
        ).T  # Transpose to get (space, member) dimensions
        sorted_1D = np.ma.mean(sorted_per_member, axis=1)
    elif method == 2:
        # Sort all values from all members together
        sorted_all = np.sort(stacked_fields.data.flatten())
        # Select every Nth element where N is the number of members
        N = fields.sizes["member"]
        sorted_1D = sorted_all[::N]
    else:
        raise ValueError("Invalid method. Choose 1 or 2.")
    
    # Initialize the PMM array
    PMM_1D = np.empty_like(field_mean.data.flatten())
    
    # Assign sorted values to locations based on sort_indices
    for count, idx in enumerate(sort_indices):
        PMM_1D[idx] = sorted_1D[count]
    
    # Reshape back to original spatial dimensions
    PMM = PMM_1D.reshape(field_mean.shape)
    
    # Return as a DataArray with original coordinates (minus 'member')
    return xr.DataArray(PMM, coords=field_mean.coords, dims=field_mean.dims)

def process_variable_pmm(var_data: xr.DataArray, method: int = 2) -> xr.DataArray:
    """
    Process a variable using Probability-Matched Mean method.
    
    Handles datasets with dimensions:
    - 3D variables: (time, lead_time, level, lat, lon, member)
    - 2D variables: (time, lead_time, lat, lon, member)
    """
    
    # Initialize list to collect results across all dimensions
    time_results = []
    
    # Loop over time dimension
    for t in range(var_data.sizes['time']):
        logger.debug(f"Processing time step {t+1}/{var_data.sizes['time']}")
        time_slice = var_data.isel(time=t)
        
        # Loop over lead_time dimension
        lead_time_results = []
        for lt in range(time_slice.sizes['lead_time']):
            lead_time_slice = time_slice.isel(lead_time=lt)
            
            # Check if level dimension exists (3D vs 2D variable)
            if 'level' in lead_time_slice.dims:
                # 3D variable: process each level separately
                level_results = []
                for lev in range(lead_time_slice.sizes['level']):
                    level_slice = lead_time_slice.isel(level=lev)
                    # Now we have (lat, lon, member) - ready for PMM
                    pmm_result = compute_PMM(level_slice, method=method)
                    level_results.append(pmm_result)
                
                # Concatenate results back along level dimension
                lead_time_pmm = xr.concat(level_results, dim='level')
            else:
                # 2D variable: direct PMM computation on (lat, lon, member)
                lead_time_pmm = compute_PMM(lead_time_slice, method=method)
            
            lead_time_results.append(lead_time_pmm)
        
        # Concatenate results back along lead_time dimension
        time_pmm = xr.concat(lead_time_results, dim='lead_time')
        time_results.append(time_pmm)
    
    # Concatenate results back along time dimension
    var_processed = xr.concat(time_results, dim='time')
    
    return var_processed

def process_variable_mean(var_data: xr.DataArray) -> xr.DataArray:
    """
    Process a variable using standard ensemble mean.
    
    Simply computes the mean across the member dimension, preserving all other dimensions:
    - 3D variables: (time, lead_time, level, lat, lon, member) -> (time, lead_time, level, lat, lon)
    - 2D variables: (time, lead_time, lat, lon, member) -> (time, lead_time, lat, lon)
    """
    processed_var = var_data.mean(dim='member')
    return processed_var

def validate_datetime(datetime_str: str) -> Tuple[str, str, str, str]:
    """Validate and format any datetime string that Python can parse."""
    try:
        # Parse the datetime string using dateutil parser (very flexible)
        dt = parser.parse(datetime_str)
        # Format components with proper padding
        year = f"{dt.year:04d}"
        month = f"{dt.month:02d}"
        day = f"{dt.day:02d}"
        hour = f"{dt.hour:02d}"

        return dt, year, month, day, hour
    except (ValueError, TypeError, parser.ParserError) as e:
        raise ValueError(f"Invalid date/time: {e}")

def find_ensemble_files(date_str: str, forecast_dir: str) -> List[str]:
    """Find all ensemble member files for a given date."""
    # Look for files in the date directory
    date_dir = os.path.join(forecast_dir, date_str)
    if not os.path.exists(date_dir):
        raise FileNotFoundError(f"Directory {date_dir} does not exist")
    
    # Pattern for ensemble files
    pattern = os.path.join(date_dir, f"hrrrcast_{date_str}_mem*.nc")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No ensemble files found matching pattern: {pattern}")
    
    # Sort files to ensure consistent ordering
    files.sort()
    logger.info(f"Found {len(files)} ensemble files")
    return files

def load_ensemble_data(files: List[str]) -> xr.Dataset:
    """Load ensemble data from multiple files and concatenate along member dimension."""
    datasets = []
    
    for i, file in enumerate(files):
        logger.info(f"Loading file {i+1}/{len(files)}: {os.path.basename(file)}")
        ds = xr.open_dataset(file)
        # Add member coordinate
        ds = ds.expand_dims(member=[i])
        datasets.append(ds)
    
    # Concatenate along member dimension
    ensemble_ds = xr.concat(datasets, dim='member')
    logger.info(f"Loaded ensemble dataset with shape: {dict(ensemble_ds.dims)}")
    
    return ensemble_ds

def compute_ensemble_pmm(datetime_str: str,
                        forecast_dir: str = "./", 
                        output_dir: str = "./",
                        method: int = 2):
    """Main ensemble post-processing function."""
    try:
        # Validate inputs
        init_datetime, init_year, init_month, init_day, init_hh = validate_datetime(datetime_str)
        date_str = f"{init_year}{init_month}{init_day}_{init_hh}"
        
        logger.info(f"Computing ensemble post-processing for initialization time: {date_str}")
        
        # Find ensemble files
        ensemble_files = find_ensemble_files(date_str, forecast_dir)
        
        # Load ensemble data
        ensemble_ds = load_ensemble_data(ensemble_files)
        
        # Create output directory if it doesn't exist
        output_date_dir = os.path.join(output_dir, date_str)
        os.makedirs(output_date_dir, exist_ok=True)
        
        # Process each variable with appropriate method
        processed_datasets = {}
        
        for var_name in ensemble_ds.data_vars:
            var_data = ensemble_ds[var_name]
            
            # Check if variable has the required dimensions
            if 'member' not in var_data.dims:
                logger.warning(f"Variable {var_name} does not have 'member' dimension, copying as-is")
                processed_datasets[var_name] = var_data
                continue
            
            # Apply PMM only to REFC (reflectivity), use standard mean for others
            if var_name == 'REFC':
                logger.info(f"Computing PMM for reflectivity variable: {var_name}")
                processed_var = process_variable_pmm(var_data, method=method)
                processed_var.attrs['processing_method'] = 'probability_matched_mean'
            else:
                logger.info(f"Computing ensemble mean for variable: {var_name}")
                processed_var = process_variable_mean(var_data)
                processed_var.attrs['processing_method'] = 'ensemble_mean'
            
            processed_datasets[var_name] = processed_var
        
        # Create output dataset
        processed_ds = xr.Dataset(processed_datasets)
        
        # Copy attributes from original dataset
        processed_ds.attrs = ensemble_ds.attrs.copy()
        processed_ds.attrs['postprocessing_method'] = 'PMM for REFC, ensemble mean for others'
        processed_ds.attrs['pmm_method'] = method
        processed_ds.attrs['processed_timestamp'] = str(datetime.now())
        processed_ds.attrs['source_files'] = [os.path.basename(f) for f in ensemble_files]
        
        # Save processed result
        output_filename = f"hrrrcast_{date_str}_mem_mean.nc"
        output_path = os.path.join(output_date_dir, output_filename)
        
        logger.info(f"Saving processed ensemble data to: {output_path}")
        processed_ds.to_netcdf(output_path)
        
        logger.info(f"Ensemble post-processing completed successfully")
        
        # Close datasets to free memory
        ensemble_ds.close()
        processed_ds.close()
        
    except Exception as e:
        logger.error(f"Ensemble post-processing failed: {e}")
        raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process HRRR ensemble forecasts: PMM for reflectivity, ensemble mean for other variables",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('inittime',
                       help='Forecast initialization time in format YYYY-MM-DDTHH (e.g., "2024-05-06T23")')
    parser.add_argument("--forecast_dir", default="./", help="Directory containing forecast files")
    parser.add_argument("--output_dir", default="./", help="Output directory for processed files")
    parser.add_argument("--method", type=int, default=2, choices=[1, 2],
                       help="PMM method for REFC: 1 for sorting per member, 2 for sorting all values together")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    return parser.parse_args()

def main():
    """Main execution function."""
    try:
        args = parse_arguments()
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Run ensemble post-processing
        compute_ensemble_pmm(
            datetime_str=args.inittime,
            forecast_dir=args.forecast_dir,
            output_dir=args.output_dir,
            method=args.method
        )
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
