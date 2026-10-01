#!/usr/bin/env python3
"""
Ensemble Post-Processing Script for HRRR Forecasts

This script processes ensemble forecast data from HRRRCast model runs and computes post-processed 
ensemble products. It applies different statistical methods based on the variable type:

- REFC (Reflectivity) and APCP (Precipitation Accumulation): Uses Probability-Matched Mean (PMM) 
  to preserve the natural distribution and spatial structure of precipitation-related fields
- All other variables: Uses standard ensemble mean which is appropriate for variables
  like temperature, wind, pressure, etc.

The Probability-Matched Mean method addresses the common problem where simple ensemble
averaging of precipitation-related variables creates unrealistically smooth fields with
underestimated extremes. PMM preserves the distribution of the ensemble mean while
maintaining the spatial structure of individual ensemble members.

Input files should follow the naming convention (per-member, per-hour):
    YYYYMMDD/HH/hrrrcast_mNN_fXX.nc

Output files are saved per hour:
    YYYYMMDD/HH/hrrrcast_avg_fXX.nc

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
from scipy.signal import fftconvolve
from scipy import ndimage

from nc2grib import Netcdf2Grib
from cf_attributes import get_cf_encoding, apply_cf_attributes, PROBABILITY_THRESHOLD_MAP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _format_threshold_token(threshold: float) -> str:
    if threshold < 0:
        return f"m{abs(threshold):g}".replace(".", "p")
    return f"{threshold:g}".replace(".", "p")


def _probability_var_name(base_var: str, threshold: float, operator: str) -> str:
    suffix = "GT" if operator == "gt" else "LT"
    return f"{base_var}_PROB_{suffix}{_format_threshold_token(threshold)}"


def _forecast_spatial_dims(var_data: xr.DataArray) -> List[str]:
    return [dim for dim in var_data.dims if dim in ["latitude", "lat", "y", "longitude", "lon", "x"]]


def _transpose_forecast_dims(var_data: xr.DataArray) -> xr.DataArray:
    if "level" in var_data.dims:
        desired_order = ["lead_time", "time", "level", "y", "x", "latitude", "longitude", "lat", "lon"]
    else:
        desired_order = ["lead_time", "time", "y", "x", "latitude", "longitude", "lat", "lon"]
    dims = [dim for dim in desired_order if dim in var_data.dims]
    return var_data.transpose(*dims)


def _attach_forecast_coords(var_data: xr.DataArray, init_datetime: datetime, lead_hour: int) -> xr.DataArray:
    if "time" in var_data.dims and "lead_time" in var_data.dims:
        return var_data.assign_coords(time=[np.datetime64(init_datetime)], lead_time=[int(lead_hour)])
    return var_data.expand_dims({"time": [np.datetime64(init_datetime)], "lead_time": [int(lead_hour)]})


def _get_probability_footprint(radius_km: float, dx_km: float) -> np.ndarray:
    """Build a DESI-style circular footprint for a neighborhood radius in km."""
    if radius_km <= 0:
        return np.ones((1, 1), dtype=np.float32)

    size = int((radius_km // dx_km) * 2 + 1)
    footprint = np.ones((size, size), dtype=np.int32)
    center = size // 2
    footprint[center, center] = 0
    dist = ndimage.distance_transform_edt(footprint, sampling=[dx_km, dx_km])
    footprint = np.where(np.greater(dist, radius_km), 0, 1).astype(np.float32)
    return footprint


def _apply_probability_smoothing(probability: xr.DataArray,
                                 valid_mask: xr.DataArray,
                                 smoothing_radius_km: float,
                                 grid_spacing_km: float) -> xr.DataArray:
    """Apply DESI-style Gaussian smoothing to a probability field.

    DESI smooths the final probability grid after the member hits have already
    been converted to percentages. The Gaussian width is specified in km and
    converted to grid-space sigma by dividing by the grid spacing.
    """
    if smoothing_radius_km <= 0:
        return probability

    sigma = smoothing_radius_km / grid_spacing_km

    smoothed = ndimage.gaussian_filter(
        probability.fillna(0.0).values,
        sigma=sigma,
        mode="constant",
        cval=0.0,
    )

    smoothed = np.where(valid_mask.values, 0.0, smoothed).astype(np.float32)
    return xr.DataArray(smoothed, coords=probability.coords, dims=probability.dims)

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
    - 3D variables: (lead_time, time, level, lat, lon, member)
    - 2D variables: (lead_time, time, lat, lon, member)
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

    # Transpose to CF-compliant dimension order.
    var_processed = _transpose_forecast_dims(var_processed)
 
    return var_processed

def process_variable_mean(var_data: xr.DataArray) -> xr.DataArray:
    """
    Process a variable using standard ensemble mean.
    
    Simply computes the mean across the member dimension, preserving all other dimensions:
    - 3D variables: (lead_time, time, level, lat, lon, member) -> (lead_time, time, level, lat, lon)
    - 2D variables: (lead_time, time, lat, lon, member) -> (lead_time, time, lat, lon)
    """
    processed_var = var_data.mean(dim='member')
    return processed_var

def process_variable_spread(var_data: xr.DataArray) -> xr.DataArray:
    """
    Process a variable using ensemble spread (standard deviation).

    Computes spread across the member dimension while preserving all other dimensions.
    """
    processed_var = var_data.std(dim='member')
    return processed_var


def process_variable_probability(var_data: xr.DataArray,
                                 threshold: float,
                                 operator: str = "gt",
                                 neighborhood_radius_km: float = 0.0,
                                 grid_spacing_km: float = 3.0,
                                 smoothing_radius_km: float = 0.0) -> xr.DataArray:
    """Process a variable into a point or neighborhood probability field.

    This follows the DESI-style probability workflow more closely: threshold
    each member, optionally apply a neighborhood hit test per member, then
    convert the member hits into a percentage field.
    """

    def _probability_on_slice(slice_data: xr.DataArray) -> xr.DataArray:
        valid_mask = slice_data.isnull()
        if operator == "gt":
            exceedance = (slice_data > threshold).astype(np.float32)
        elif operator == "lt":
            exceedance = (slice_data < threshold).astype(np.float32)
        else:
            raise ValueError(f"Unsupported probability operator: {operator}")
        if neighborhood_radius_km > 0 and neighborhood_radius_km >= grid_spacing_km:
            spatial_dims = _forecast_spatial_dims(exceedance)
            if len(spatial_dims) < 2:
                raise ValueError(f"Could not identify spatial dimensions for probability field. Available dims: {slice_data.dims}")
            footprint = _get_probability_footprint(neighborhood_radius_km, grid_spacing_km)

            def _convolve_spatial(values: np.ndarray) -> np.ndarray:
                return fftconvolve(values, footprint, mode="same")

            exceedance = xr.apply_ufunc(
                _convolve_spatial,
                exceedance,
                input_core_dims=[spatial_dims],
                output_core_dims=[spatial_dims],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[np.float32],
            )
            exceedance = exceedance > 0.5

        valid_members = (~valid_mask).sum(dim='member')
        valid_hits = exceedance.where(~valid_mask, other=0.0).sum(dim='member')
        probability = xr.where(valid_members > 0, valid_hits / valid_members * 100.0, np.nan)
        probability = _apply_probability_smoothing(
            probability,
            valid_members == 0,
            smoothing_radius_km,
            grid_spacing_km,
        )
        return probability.astype(np.float32)

    time_results = []

    for t in range(var_data.sizes['time']):
        time_slice = var_data.isel(time=t)
        lead_time_results = []

        for lt in range(time_slice.sizes['lead_time']):
            lead_time_slice = time_slice.isel(lead_time=lt)

            if 'level' in lead_time_slice.dims:
                level_results = []
                for lev in range(lead_time_slice.sizes['level']):
                    level_slice = lead_time_slice.isel(level=lev)
                    level_results.append(_probability_on_slice(level_slice))
                lead_time_prob = xr.concat(level_results, dim='level')
            else:
                lead_time_prob = _probability_on_slice(lead_time_slice)

            lead_time_results.append(lead_time_prob)

        time_prob = xr.concat(lead_time_results, dim='lead_time')
        time_results.append(time_prob)

    var_processed = xr.concat(time_results, dim='time')
    return _transpose_forecast_dims(var_processed)

def build_member_file_list(date_str: str, forecast_dir: str, hour: int, n_ensembles: int) -> List[str]:
    """Construct expected per-member file paths for a given hour and validate existence.

    Uses naming convention hrrrcast_mN_fXX.nc for N in [0..n_ensembles-1].
    """
    date_dir = os.path.join(forecast_dir, date_str)
    if not os.path.isdir(date_dir):
        raise FileNotFoundError(f"Directory not found: {date_dir}")

    files: List[str] = []
    for m in range(n_ensembles):
        fname = os.path.join(date_dir, f"hrrrcast_m{m:02d}_f{hour:02d}.nc")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Missing expected file: {fname}")
        files.append(fname)
    return files

def wait_for_hour_files(date_str: str,
                        forecast_dir: str,
                        hour: int,
                        n_ensembles: int,
                        poll_seconds: int = 60,
                        min_age_seconds: int = 90,
                        timeout_seconds: Optional[int] = None) -> List[str]:
    """Wait until all expected member files exist and are stable for the given hour.

    Stability is defined as not modified within the last min_age_seconds.
    Checks every poll_seconds. Returns the list of file paths when ready.
    
    Parameters:
    - timeout_seconds: Maximum time to wait in seconds. None means wait indefinitely.
                      If timeout is reached, raises TimeoutError.
    """
    date_dir = os.path.join(forecast_dir, date_str)
    if not os.path.isdir(date_dir):
        raise FileNotFoundError(f"Directory not found: {date_dir}")

    def file_path(m: int) -> str:
        return os.path.join(date_dir, f"hrrrcast_m{m:02d}_f{hour:02d}.nc")

    start_time = time.time()
    while True:
        # Check timeout
        if timeout_seconds is not None:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"Timeout after {elapsed:.0f}s waiting for hour f{hour:02d} files. "
                    f"Forecast job may have failed or died."
                )
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
                        n_ensembles: Optional[int] = None,
                        prob_neighborhood_radius_km: float = 0.0,
                        prob_smoothing_radius_km: float = 0.0):
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
        timeout_seconds = int(os.environ.get("PMM_TIMEOUT_SECONDS", "600"))

        for h in range(0, int(lead_hour) + 1):
            # Wait until files are present and stable before processing this hour
            # Hour 0: wait indefinitely; subsequent hours: max timeout_seconds
            timeout = None if h == 0 else timeout_seconds
            try:
                files = wait_for_hour_files(date_str, forecast_dir, h, n_ensembles, poll_seconds, min_age_seconds, timeout_seconds=timeout)
            except TimeoutError as e:
                logger.error(f"Files not found for hour f{h:02d}: {e}")
                logger.error("Forecast job appears to be dead or has failed. Exiting PMM computation.")
                sys.exit(1)
            logger.info(f"Processing forecast hour f{h:02d} with {len(files)} member files")

            # Load per-hour ensemble
            ensemble_ds = load_hour_ensemble_data(files)

            processed_datasets: Dict[str, xr.DataArray] = {}
            spread_datasets: Dict[str, xr.DataArray] = {}
            probability_datasets: Dict[str, xr.DataArray] = {}

            for var_name in ensemble_ds.data_vars:
                # Skip CF metadata variables
                if var_name == 'grid_mapping':
                    continue
                    
                var_data = ensemble_ds[var_name]
                if 'member' not in var_data.dims:
                    logger.warning(f"Variable {var_name} missing 'member' dim at f{lead_hour:02d}, copying as-is")
                    da = var_data
                    spread_da = var_data
                else:
                    if var_name in ['REFC', 'APCP']:
                        logger.info(f"PMM for {var_name} at f{h:02d}")
                        da = process_variable_pmm(var_data, method=method)
                        da.attrs['processing_method'] = 'probability_matched_mean'
                    else:
                        logger.info(f"Mean for {var_name} at f{h:02d}")
                        da = process_variable_mean(var_data)
                        da.attrs['processing_method'] = 'ensemble_mean'

                    logger.info(f"Spread for {var_name} at f{h:02d}")
                    spread_da = process_variable_spread(var_data)
                    spread_da.attrs['processing_method'] = 'ensemble_spread_stddev'

                    if var_name in PROBABILITY_THRESHOLD_MAP:
                        prob_config = PROBABILITY_THRESHOLD_MAP[var_name]
                        for threshold in prob_config["thresholds"]:
                            prob_name = _probability_var_name(var_name, threshold, prob_config["operator"])
                            logger.info(
                                f"Probability for {var_name} threshold {threshold:g} at f{h:02d} "
                                f"using {prob_neighborhood_radius_km:g} km neighborhood radius"
                            )
                            prob_da = process_variable_probability(
                                var_data,
                                threshold=threshold,
                                operator=prob_config["operator"],
                                neighborhood_radius_km=prob_neighborhood_radius_km,
                                smoothing_radius_km=prob_smoothing_radius_km,
                            )
                            prob_da = _attach_forecast_coords(prob_da, init_datetime, h)
                            prob_da.attrs.update({
                                'long_name': f"Probability of {var_name} {'>' if prob_config['operator'] == 'gt' else '<'} {threshold:g}",
                                'units': '%',
                                'processing_method': 'ensemble_neighborhood_probability'
                                if prob_neighborhood_radius_km > 0 else 'ensemble_probability',
                                'base_variable': var_name,
                                'probability_threshold': threshold,
                                'probability_operator': '>' if prob_config['operator'] == 'gt' else '<',
                                'probability_neighborhood_radius_km': prob_neighborhood_radius_km,
                                'probability_neighborhood_grid_spacing_km': 3.0,
                                'probability_neighborhood_method': 'desi_footprint'
                                if prob_neighborhood_radius_km > 0 else 'point',
                                'probability_smoothing_radius_km': prob_smoothing_radius_km,
                                'probability_smoothing_method': 'gaussian_filter'
                                if prob_smoothing_radius_km > 0 else 'none',
                            })
                            probability_datasets[prob_name] = prob_da

                # Ensure time and lead_time coords/dims exist for downstream writer
                # If dims already exist, just set their coordinate values; else expand dims
                da = _attach_forecast_coords(da, init_datetime, h)
                spread_da = _attach_forecast_coords(spread_da, init_datetime, h)

                processed_datasets[var_name] = da
                spread_datasets[var_name] = spread_da

            processed_ds = xr.Dataset(processed_datasets)
            spread_ds = xr.Dataset(spread_datasets)
            probability_ds = xr.Dataset(probability_datasets)

            # Apply CF attributes
            processed_ds = apply_cf_attributes(processed_ds, init_datetime=init_datetime)
            spread_ds = apply_cf_attributes(spread_ds, init_datetime=init_datetime)
            probability_ds = apply_cf_attributes(probability_ds, init_datetime=init_datetime)

            # Add processing-specific attributes
            processed_ds.attrs.update({
                'postprocessing_method': 'PMM for selected variables, mean for others',
                'pmm_method': method,
                'processed_timestamp': str(datetime.now()),
                'source_files': [os.path.basename(f) for f in files]
            })

            spread_ds.attrs.update({
                'postprocessing_method': 'ensemble spread (standard deviation)',
                'processed_timestamp': str(datetime.now()),
                'source_files': [os.path.basename(f) for f in files]
            })

            probability_ds.attrs.update({
                'postprocessing_method': 'ensemble probabilities',
                'probability_neighborhood_radius_km': prob_neighborhood_radius_km,
                'probability_neighborhood_grid_spacing_km': 3.0,
                'probability_neighborhood_method': 'desi_footprint'
                if prob_neighborhood_radius_km > 0 else 'point',
                'probability_smoothing_radius_km': prob_smoothing_radius_km,
                'probability_smoothing_method': 'gaussian_filter'
                if prob_smoothing_radius_km > 0 else 'none',
                'ensemble_size': n_ensembles,
                'processed_timestamp': str(datetime.now()),
                'source_files': [os.path.basename(f) for f in files]
            })
            for var_name in probability_ds.data_vars:
                if var_name == 'grid_mapping':
                    continue
                da = probability_ds[var_name]
                base_var = da.attrs.get('base_variable', var_name)
                threshold = da.attrs.get('probability_threshold')
                operator = da.attrs.get('probability_operator', '>')
                if threshold is not None:
                    da.attrs['long_name'] = f"Probability of {base_var} {operator} {threshold:g}"
                else:
                    logger.warning(f"Probability variable {var_name} missing threshold metadata; preserving existing long_name")
                da.attrs['units'] = '%'
                da.attrs['grid_mapping'] = 'grid_mapping'
                da.attrs['ensemble_size'] = n_ensembles

            cycle = init_datetime.hour

            # Save per-hour NetCDF
            out_nc = os.path.join(output_date_dir, f"hrrrcast_avg_f{h:02d}.nc")
            # Get CF-compliant encoding
            encoding = get_cf_encoding(processed_ds, init_datetime)
            processed_ds.to_netcdf(out_nc, encoding=encoding)
            logger.info(f"Wrote NetCDF : {out_nc}")

            out_nc_spread = os.path.join(output_date_dir, f"hrrrcast_spr_f{h:02d}.nc")
            encoding = get_cf_encoding(spread_ds, init_datetime)
            spread_ds.to_netcdf(out_nc_spread, encoding=encoding)
            logger.info(f"Wrote NetCDF : {out_nc_spread}")

            out_nc_prob = os.path.join(output_date_dir, f"hrrrcast_prob_f{h:02d}.nc")
            encoding = get_cf_encoding(probability_ds, init_datetime)
            probability_ds.to_netcdf(out_nc_prob, encoding=encoding)
            logger.info(f"Wrote NetCDF : {out_nc_prob}")

            # Save per-hour GRIB2
            avg_grib2 = os.path.join(output_date_dir, f"hrrrcast.avg.t{cycle:02d}z.pgrb2.f{h:02d}")
            converter.save_grib2(init_datetime, processed_ds, avg_grib2)
            logger.info(f"Wrote GRIB2 : {avg_grib2}")

            spr_grib2 = os.path.join(output_date_dir, f"hrrrcast.spr.t{cycle:02d}z.pgrb2.f{h:02d}")
            converter.save_grib2(init_datetime, spread_ds, spr_grib2)
            logger.info(f"Wrote GRIB2 : {spr_grib2}")

            prob_grib2 = os.path.join(output_date_dir, f"hrrrcast.prob.t{cycle:02d}z.pgrb2.f{h:02d}")
            converter.save_grib2(init_datetime, probability_ds, prob_grib2)
            logger.info(f"Wrote GRIB2 : {prob_grib2}")

            # Close datasets to free memory
            ensemble_ds.close()
            processed_ds.close()
            spread_ds.close()
            probability_ds.close()

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
    parser.add_argument("--prob_neighborhood_radius_km", type=float, default=float(os.environ.get("PMM_PROB_NEIGHBORHOOD_RADIUS_KM", "18")),
                       help="Neighborhood radius in kilometers used for exceedance probabilities")
    parser.add_argument("--prob_smoothing_radius_km", type=float, default=float(os.environ.get("PMM_PROB_SMOOTHING_RADIUS_KM", "36")),
                       help="DESI-style Gaussian smoothing radius in kilometers applied to probability fields")
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
            n_ensembles=args.n_ensembles,
            prob_neighborhood_radius_km=args.prob_neighborhood_radius_km,
            prob_smoothing_radius_km=args.prob_smoothing_radius_km,
        )
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
