#!/usr/bin/env python3
"""
Ensemble Post-Processing Script for HRRR Forecasts

This script processes ensemble forecast data from HRRR (High-Resolution Rapid Refresh) 
model runs and computes post-processed ensemble products. It applies different statistical
methods based on the variable type:

- REFC (Reflectivity) and APCP (Precipitation Accumulation): Uses Probability-Matched Mean (PMM) 
  to preserve the natural distribution and spatial structure of precipitation-related fields
- All other variables: Uses standard ensemble mean which is appropriate for variables
  like temperature, wind, pressure, etc.

The Probability-Matched Mean method addresses the common problem where simple ensemble
averaging of precipitation-related variables creates unrealistically smooth fields with
underestimated extremes. PMM preserves the distribution of the ensemble mean while
maintaining the spatial structure of individual ensemble members.

Input files should follow the naming convention (per-member, per-hour):
    YYYYMMDD/HH/hrrrcast_memN_fXX.nc

Output files are saved per hour:
    YYYYMMDD/HH/hrrrcast_memavg_fXX.nc

Usage:
    python compute_pmm.py "2024-05-06T23" 18 --forecast_dir /path/to/data --n_ensembles 4
"""
import argparse
import logging
import os
import sys 
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import xarray as xr
import time
import utils

from nc2grib import Netcdf2Grib

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
        logger.debug(f"Lead time {lt}h")
    if 'level' in fields.coords:
        lv = fields.level.values
        logger.debug(f"Level {lv}")
    if 'time' in fields.coords:
        logger.debug(f"Time {fields.time.values}")
    else:
        logger.debug("")  # Just newline if no time coord
    
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

def build_member_file_list(date_str: str, forecast_dir: str, hour: int, n_ensembles: int) -> List[str]:
    """Construct expected per-member file paths for a given hour and validate existence.

    Uses naming convention hrrrcast_memN_fXX.nc for N in [0..n_ensembles-1].
    """
    date_dir = os.path.join(forecast_dir, date_str)
    if not os.path.isdir(date_dir):
        raise FileNotFoundError(f"Directory not found: {date_dir}")

    files: List[str] = []
    for m in range(n_ensembles):
        fname = os.path.join(date_dir, f"hrrrcast_mem{m}_f{hour:02d}.nc")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Missing expected file: {fname}")
        files.append(fname)
    return files

def wait_for_hour_files(date_str: str,
                        forecast_dir: str,
                        hour: int,
                        n_ensembles: int,
                        poll_seconds: int = 60,
                        min_age_seconds: int = 90) -> List[str]:
    """Wait until all expected member files exist and are stable for the given hour.

    Stability is defined as not modified within the last min_age_seconds.
    Checks every poll_seconds. Returns the list of file paths when ready.
    """
    date_dir = os.path.join(forecast_dir, date_str)
    if not os.path.isdir(date_dir):
        raise FileNotFoundError(f"Directory not found: {date_dir}")

    def file_path(m: int) -> str:
        return os.path.join(date_dir, f"hrrrcast_mem{m}_f{hour:02d}.nc")

    while True:
        files: List[str] = []
        all_present = True
        for m in range(n_ensembles):
            fp = file_path(m)
            if not os.path.exists(fp):
                all_present = False
                break
            files.append(fp)

        if not all_present:
            logger.info(f"Waiting for ensemble files for hour f{hour:02d} ({poll_seconds}s)...")
            time.sleep(poll_seconds)
            continue

        # Check stability (no recent modifications)
        now = time.time()
        all_stable = True
        for fp in files:
            try:
                mtime = os.path.getmtime(fp)
            except FileNotFoundError:
                all_stable = False
                break
            if (now - mtime) < min_age_seconds:
                all_stable = False
                break

        if all_stable:
            logger.info(f"Files ready for hour f{hour:02d}: {len(files)} members, stable >= {min_age_seconds}s")
            return files
        else:
            logger.info(f"Files present for hour f{hour:02d} but not yet stable (age < {min_age_seconds}s). Sleeping {poll_seconds}s...")
            time.sleep(poll_seconds)

def load_hour_ensemble_data(files: List[str]) -> xr.Dataset:
    """Load per-hour ensemble files and concatenate along member dimension."""
    datasets = []
    for idx, file in enumerate(files):
        logger.info(f"Loading file {idx+1}/{len(files)}: {os.path.basename(file)}")
        ds = xr.open_dataset(file)
        # Use member index derived from filename order; assume sorted by member
        ds = ds.expand_dims(member=[idx])
        datasets.append(ds)
    ensemble_ds = xr.concat(datasets, dim='member')
    logger.info(f"Loaded per-hour ensemble dataset with dims: {dict(ensemble_ds.dims)}")
    return ensemble_ds

def compute_ensemble_pmm(datetime_str: str,
                        lead_hour: int,
                        forecast_dir: str = "./", 
                        output_dir: str = "./",
                        method: int = 2,
                        n_ensembles: Optional[int] = None):
    """Main ensemble post-processing function: loop hours 1..lead_hour and write per-hour outputs."""
    try:
        # Validate inputs
        init_datetime, init_year, init_month, init_day, init_hh = utils.validate_datetime(datetime_str)
        date_str = f"{init_year}{init_month}{init_day}/{init_hh}"

        logger.info(f"Computing ensemble post-processing for initialization time: {date_str}, lead_hour: {lead_hour}, n_ensembles: {n_ensembles}")

        # Create output directory if it doesn't exist
        output_date_dir = os.path.join(output_dir, date_str)
        utils.make_directory(output_date_dir)

        converter = Netcdf2Grib()

        # Polling configuration (overridable via env)
        poll_seconds = int(os.environ.get("PMM_POLL_SECONDS", "60"))
        min_age_seconds = int(os.environ.get("PMM_MIN_AGE_SECONDS", "90"))

        for h in range(0, int(lead_hour) + 1):
            # Wait until files are present and stable before processing this hour
            files = wait_for_hour_files(date_str, forecast_dir, h, n_ensembles, poll_seconds, min_age_seconds)
            logger.info(f"Processing forecast hour f{h:02d} with {len(files)} member files")

            # Load per-hour ensemble
            ensemble_ds = load_hour_ensemble_data(files)

            processed_datasets: Dict[str, xr.DataArray] = {}

            for var_name in ensemble_ds.data_vars:
                var_data = ensemble_ds[var_name]
                if 'member' not in var_data.dims:
                    logger.warning(f"Variable {var_name} missing 'member' dim at f{lead_hour:02d}, copying as-is")
                    da = var_data
                else:
                    if var_name in ['REFC', 'APCP']:
                        logger.info(f"PMM for {var_name} at f{h:02d}")
                        da = process_variable_pmm(var_data, method=method)
                        da.attrs['processing_method'] = 'probability_matched_mean'
                    else:
                        logger.info(f"Mean for {var_name} at f{h:02d}")
                        da = process_variable_mean(var_data)
                        da.attrs['processing_method'] = 'ensemble_mean'

                # Ensure time and lead_time coords/dims exist for downstream writer
                # If dims already exist, just set their coordinate values; else expand dims
                if 'time' in da.dims and 'lead_time' in da.dims:
                    da = da.assign_coords(time=[np.datetime64(init_datetime)],
                                          lead_time=[int(h)])
                else:
                    da = da.expand_dims({
                        'time': [np.datetime64(init_datetime)],
                        'lead_time': [int(h)]
                    })
                processed_datasets[var_name] = da

            processed_ds = xr.Dataset(processed_datasets)
            # Copy attributes and annotate
            processed_ds.attrs = ensemble_ds.attrs.copy()
            processed_ds.attrs['postprocessing_method'] = 'PMM for REFC/APCP, mean for others'
            processed_ds.attrs['pmm_method'] = method
            processed_ds.attrs['processed_timestamp'] = str(datetime.now())
            processed_ds.attrs['source_files'] = [os.path.basename(f) for f in files]

            # Save per-hour NetCDF
            out_nc = os.path.join(output_date_dir, f"hrrrcast_memavg_f{h:02d}.nc")
            logger.info(f"Saving per-hour processed ensemble to: {out_nc}")
            processed_ds.to_netcdf(out_nc)

            # Save per-hour GRIB2 for avg
            converter.save_grib2(init_datetime, processed_ds, 'avg', output_date_dir)

            # Close datasets to free memory
            ensemble_ds.close()
            processed_ds.close()

        logger.info("Ensemble per-hour post-processing completed successfully")

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
    parser.add_argument('lead_hour', type=int, help='Process all lead hours from 1..lead_hour (e.g., 18)')
    parser.add_argument("--forecast_dir", default="./", help="Directory containing forecast files")
    parser.add_argument("--output_dir", default="./", help="Output directory for processed files")
    parser.add_argument("--method", type=int, default=2, choices=[1, 2],
                       help="PMM method for REFC: 1 for sorting per member, 2 for sorting all values together")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--n_ensembles", type=int, default=None, help="Number of ensemble members (fallback to N_ENSEMBLES env)")
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
            lead_hour=args.lead_hour,
            forecast_dir=args.forecast_dir,
            output_dir=args.output_dir,
            method=args.method,
            n_ensembles=args.n_ensembles
        )
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
