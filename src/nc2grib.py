"""
GRIB2 writer using grib2io for HRRRCast outputs.

This module converts NetCDF forecast output to GRIB2 format using grib2io,
inspired by NOAA-EMC MLGlobal's grib2writer.py.

The Netcdf2Grib class handles per-member forecast writes, supporting both
single-hour and multi-hour datasets. Per-hour writes enable overlapped I/O
during autoregressive forecasting.

Notes/assumptions:
- Grid Definition: We require a valid GRIB2 Section 3 for the HRRRCast Lambert
    Conformal grid. Provide via Netcdf2Grib(section3=...) constructor or set the
    environment variable NETCDF2GRIB_SECTION3 to a .npy file. If neither is provided,
    we auto-construct a canonical HRRR Lambert Conformal Section 3 for the full
    3 km grid (Nx=1799, Ny=1059).
- Template Numbers: Product Definition Template Numbers (pdtn) and Data Representation
  Template Numbers (drtn) default to 0 (instantaneous forecast, simple packing).
  For accumulated fields (e.g., APCP), adjust pdtn and duration semantics to match
  downstream consumers.
- Member IDs: Each member forecast can be written independently, allowing per-member
  outputs with consistent naming conventions.
"""

import os
import subprocess
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import xarray as xr
import grib2io

from cf_attributes import (
    VARIABLE_METADATA,
    PRECIP_ACCUMULATION_THRESHOLDS,
    PROBABILITY_THRESHOLD_MAP,
)
from utils import setup_logging

logger = setup_logging("INFO")


# Derive GRIB parameter map from consolidated metadata
# Format: var -> (discipline, category, number, surface_type, surface_value)
GRIB_PARAM_MAP = {
    var: meta["grib2"]
    for var, meta in VARIABLE_METADATA.items()
    if "grib2" in meta
}

class Netcdf2Grib:
    # Class-level lock for grib2io operations (g2c library may not be thread-safe)
    _grib2io_lock = threading.Lock()

    def __init__(self, section3: Optional[np.ndarray] = None, pdtn_default: int = 0, drtn_default: int = 3):
        self.section3 = self._resolve_section3(section3)
        self.pdtn_default = pdtn_default
        self.drtn_default = drtn_default

    def construct_section3_hrrr(self, nx: int = 1799, ny: int = 1059) -> np.ndarray:
        """Construct GRIB2 Section 3 for HRRR-like CONUS Lambert Conformal grid at 3 km.

        This uses canonical HRRR projection parameters and the full-resolution dimensions
        defined in preprocessing (grid_width=1799, grid_height=1059).

        Parameters used:
        - First grid point (La1/Lo1): 21.138123N, 237.280472E
        - Orientation longitude (LoV): 262.5E
        - Standard parallels (Latin1, Latin2): 38.5N, 38.5N
        - Grid spacing (Dx/Dy): 3000 m
        - Earth radius: 6371229 m

        Returns a numpy array suitable for the `section3` argument of grib2io.Grib2Message.

        Note: If grib2io provides a helper for LCC Section 3 creation in your environment,
        this function will attempt to use it. Otherwise, it constructs a fixed array using
        canonical HRRR parameters. You can override via NETCDF2GRIB_SECTION3.
        """
        # Canonical HRRR LCC parameters (matching HRRR docs)
        lat1 = 21.138123    # degrees North
        lon1 = 237.280472   # degrees East
        lov = 262.5         # degrees East
        latin1 = 38.5       # degrees North
        latin2 = 38.5       # degrees North
        dx = 3000           # meters
        dy = 3000           # meters
        earth_radius = 6371229  # meters (spherical)

        # Build a best-effort fixed array for GRIB2 Template 3.30 (Lambert Conformal)
        # Values are encoded as scaled integers:
        # - Lat/Lon in microdegrees (deg * 1e6)
        # - Dx/Dy in millimeters (m * 1e3)
        # Note: Field positions follow common GRIB2 3.30 usage; some decoders may require
        # exact scan mode or earth-shape codes. Adjust if downstream tools complain.

        micro = 1_000_000
        milli = 1_000

        la1 = int(round(lat1 * micro))
        lo1 = int(round(lon1 * micro))
        lov_i = int(round(lov * micro))
        latin1_i = int(round(latin1 * micro))
        latin2_i = int(round(latin2 * micro))
        dx_mm = int(round(dx * milli))
        dy_mm = int(round(dy * milli))

        # Common defaults
        shape_of_earth = 1  # spherical with given radius
        # Resolution and component flags: 8 -> winds(grid) per wgrib2 'res 8'
        res_flags = 8
        # Projection centre flag: 0 = north, 1 = south
        proj_center_flag = 0

        # Section 3 structure (template 3.30 Lambert Conformal) matching grib_dump order:
        # Fields reflect wgrib2/grib_dump output: res=8, scanningMode=64 (WE:SN), LaD=38500000
        section3 = np.array([
            0,                   # Source of grid definition
            nx * ny,             # Number of data points = Ni * Nj
            0,                   # Number of octets for number of points
            0,                   # Interpretation of number of points
            30,                  # Grid definition template number (3.30)
            shape_of_earth,      # Shape of Earth (1 = spherical, producer-specified radius)
            0,                   # Scale factor of radius of spherical Earth
            earth_radius,        # Scaled value of spherical Earth radius (meters)
            0,                   # Scale factor of Earth major axis
            0,                   # Scaled value of Earth major axis
            0,                   # Scale factor of Earth minor axis
            0,                   # Scaled value of Earth minor axis
            nx,                  # Nx
            ny,                  # Ny
            la1,                 # Latitude of first grid point (microdegrees)
            lo1,                 # Longitude of first grid point (microdegrees)
            res_flags,           # Resolution and component flags (8 -> winds(grid))
            38_500_000,          # LaD (Latitude of grid orientation, microdegrees)
            lov_i,               # LoV (orientation longitude, microdegrees)
            dx_mm,               # Dx (grid length in x, millimeters)
            dy_mm,               # Dy (grid length in y, millimeters)
            proj_center_flag,    # Projection centre flag (0 = north)
            64,                  # Scanning mode (WE:SN)
            latin1_i,            # Latin1 (first standard parallel, microdegrees)
            latin2_i,            # Latin2 (second standard parallel, microdegrees)
            0,                   # Latitude of southern pole
            0,                   # Longitude of southern pole
        ], dtype=np.int64)

        return section3

    def _resolve_section3(self, section3: Optional[np.ndarray]) -> np.ndarray:
        if section3 is not None:
            return np.asarray(section3, dtype=np.int64)
        env_path = os.environ.get("NETCDF2GRIB_SECTION3", "")
        if env_path and os.path.isfile(env_path):
            try:
                arr = np.load(env_path)
                return np.asarray(arr, dtype=np.int64)
            except Exception as e:
                raise RuntimeError(f"Failed to load section3 from {env_path}: {e}")
        # Fallback: attempt to construct HRRR-like 3 km LCC Section 3 using known dims (Nx=1799, Ny=1059)
        try:
            return self.construct_section3_hrrr(nx=1799, ny=1059)
        except Exception as e:
            raise RuntimeError(
                "GRIB2 Section 3 (grid definition) is required and could not be auto-constructed. "
                "Provide 'section3' to Netcdf2Grib, set NETCDF2GRIB_SECTION3 to a .npy file, or ensure grib2io LCC helper is available. "
                f"Error: {e}"
            )

    def _apply_accumulation_metadata(
        self,
        msg: grib2io.Grib2Message,
        ref_time: datetime,
        lead_hour: int,
        var_name: str,
        accumulation_hours: Optional[int] = None,
    ) -> None:
        # GRIB2 accumulation fields are not instantaneous forecasts.
        # APCP (hourly accumulation) spans [lead-1, lead]. APCP_TOTAL is a cumulative
        # diagnostic from the forecast start, so it should begin at step 0.
        if accumulation_hours is not None:
            start_hour = max(int(lead_hour) - int(accumulation_hours), 0)
        else:
            start_hour = 0 if var_name == "APCP_TOTAL" else max(int(lead_hour) - 1, 0)
        end_hour = int(lead_hour)
        duration_hours = end_hour - start_hour
        end_time = ref_time + timedelta(hours=end_hour)

        # Product Definition Template 8 represents an interval as a forecast
        # time at the beginning of the interval plus its duration.
        msg.leadTime = timedelta(hours=start_hour)
        msg.yearOfEndOfTimePeriod = end_time.year
        msg.monthOfEndOfTimePeriod = end_time.month
        msg.dayOfEndOfTimePeriod = end_time.day
        msg.hourOfEndOfTimePeriod = end_time.hour
        msg.minuteOfEndOfTimePeriod = end_time.minute
        msg.secondOfEndOfTimePeriod = end_time.second
        msg.numberOfTimeRanges = 1
        msg.numberOfMissingValues = 0
        msg.statisticalProcess = 1  # Code Table 4.10: accumulation
        msg.typeOfTimeIncrementOfStatisticalProcess = 2  # Code Table 4.11
        msg.unitOfTimeRangeOfStatisticalProcess = 1  # hours
        msg.timeRangeOfStatisticalProcess = duration_hours
        msg.unitOfTimeRangeOfSuccessiveFields = 1  # hours
        msg.timeIncrementOfSuccessiveFields = 0

    def _apply_probability_metadata(
        self,
        msg: grib2io.Grib2Message,
        var_name: str,
        probability_threshold: Optional[float],
        probability_operator: str,
        base_variable: Optional[str] = None,
        accumulation_hours: Optional[int] = None,
    ) -> None:
        if probability_threshold is None:
            return

        # PDT 5/9 represent probability forecasts. Code Table 4.9 uses the lower
        # limit for "below" and the upper limit for "above".
        probability_type = 1 if probability_operator == ">" else 0
        threshold_value = float(probability_threshold)
        parsed = self._parse_probability_var_name(var_name)
        probability_number = 1
        total_probabilities = 1
        base_var = base_variable
        if base_var is None and parsed is not None:
            base_var = parsed[0]
        if base_var == "APCP" and accumulation_hours in PRECIP_ACCUMULATION_THRESHOLDS:
            thresholds = PRECIP_ACCUMULATION_THRESHOLDS[accumulation_hours]
            total_probabilities = len(thresholds)
            probability_number = next(
                index
                for index, configured_threshold in enumerate(thresholds, start=1)
                if abs(float(configured_threshold) - threshold_value) < 1e-6
            )
        elif base_var in PROBABILITY_THRESHOLD_MAP:
            thresholds = PROBABILITY_THRESHOLD_MAP[base_var]["thresholds"]
            total_probabilities = len(thresholds)
            probability_number = next(
                index
                for index, configured_threshold in enumerate(thresholds, start=1)
                if abs(float(configured_threshold) - threshold_value) < 1e-6
            )

        msg.forecastProbabilityNumber = probability_number
        msg.totalNumberOfForecastProbabilities = total_probabilities
        msg.typeOfProbability = probability_type
        if probability_operator == ">":
            msg.thresholdUpperLimit = threshold_value
        else:
            msg.thresholdLowerLimit = threshold_value

    def _build_message(
        self,
        var_name: str,
        ref_time: datetime,
        lead_hour: int,
        surface_type: Optional[int] = None,
        surface_value: Optional[float] = None,
        pdtn: Optional[int] = None,
        drtn: Optional[int] = None,
        ensemble_size: Optional[int] = None,
        probability_threshold: Optional[float] = None,
        probability_operator: str = ">",
        base_variable: Optional[str] = None,
        accumulation_hours: Optional[int] = None,
    ) -> grib2io.Grib2Message:

        # 1. Define Section 1 (Identification Section)
        section1 = np.array([
            7,               # Center: 7 (NCEP)
            0,               # Subcenter: 0
            2,               # Master Tables Version: 2
            1,               # Local Tables Version: 1
            1,               # Significance of Ref Time: 1 (Start of Forecast)
            ref_time.year,
            ref_time.month,
            ref_time.day,
            ref_time.hour,
            ref_time.minute,
            ref_time.second,
            0,               # Production Status: 0 (Operational)
            1                # Type of Data: 1 (Forecast)
        ], dtype=np.int64)

        # 2. Construct message
        # A lead-zero APCP record is an analysis, not a zero-length forecast
        # accumulation. Use PDT 8 only once a forecast interval exists.
        is_accumulation = (
            probability_threshold is None
            and int(lead_hour) > 0
            and var_name in ["APCP", "APCP_TOTAL"]
        )
        is_probability_accumulation = (
            probability_threshold is not None
            and int(lead_hour) > 0
            and base_variable == "APCP"
            and accumulation_hours is not None
        )
        message_pdtn = self.pdtn_default if pdtn is None else pdtn
        if is_accumulation:
            # grib2io initializes Section 4 from pdtn in the constructor. Changing
            # productDefinitionTemplateNumber afterward can leave Section 4 using
            # the old template and fail in some grib2io versions.
            message_pdtn = 8
        elif is_probability_accumulation:
            # PDT 9 is the probability counterpart to PDT 8 and includes the
            # accumulation interval plus its statistical-processing range.
            message_pdtn = 9
        elif probability_threshold is not None:
            # PDT 5 is a probability forecast at a horizontal level or layer at
            # a point in time. PDT 8 is for statistical processing over an
            # interval and requires at least one time-range specification.
            message_pdtn = 5
        msg = grib2io.Grib2Message(
            section1=section1,
            section3=self.section3,
            pdtn=message_pdtn,
            drtn=self.drtn_default if drtn is None else drtn,
        )

        self._apply_probability_metadata(
            msg,
            var_name,
            probability_threshold,
            probability_operator,
            base_variable,
            accumulation_hours,
        )

        # 3. Set parameter keys
        parameter_var_name = base_variable if base_variable is not None else var_name
        disc, cat, num, default_surface, _ = self._get_grib_param_info(parameter_var_name)
        msg.discipline = disc
        msg.parameterCategory = cat
        msg.parameterNumber = num
        msg.typeOfFirstFixedSurface = surface_type if surface_type is not None else default_surface

        if surface_value is not None:
            # Check if surface_value is a tuple (layer) or a single value
            if isinstance(surface_value, tuple):
                # Layer specification: (top, bottom)
                top_value, bottom_value = surface_value
                msg.scaledValueOfFirstFixedSurface = int(top_value)
                msg.scaleFactorOfFirstFixedSurface = 0
                msg.typeOfSecondFixedSurface = surface_type if surface_type is not None else default_surface
                msg.scaledValueOfSecondFixedSurface = int(bottom_value)
                msg.scaleFactorOfSecondFixedSurface = 0
            else:
                # Single level specification
                msg.scaledValueOfFirstFixedSurface = int(surface_value)
                msg.scaleFactorOfFirstFixedSurface = 0
                msg.typeOfSecondFixedSurface = 255
                msg.scaleFactorOfSecondFixedSurface = 0
                msg.scaledValueOfSecondFixedSurface = 0
        else:
            msg.scaledValueOfFirstFixedSurface = 0
            msg.scaleFactorOfFirstFixedSurface = 0

            msg.typeOfSecondFixedSurface = 255
            msg.scaleFactorOfSecondFixedSurface = 0
            msg.scaledValueOfSecondFixedSurface = 0

        # 4. Time metadata
        msg.unitOfForecastTime = 1  # hours
        if is_accumulation or is_probability_accumulation:
            # APCP is the previous one-hour accumulation; APCP_TOTAL spans from
            # forecast initialization through this lead time.
            accumulation_var = "APCP" if is_probability_accumulation else var_name
            self._apply_accumulation_metadata(
                msg,
                ref_time,
                lead_hour,
                accumulation_var,
                accumulation_hours=accumulation_hours,
            )
        else:
            msg.leadTime = timedelta(hours=int(lead_hour))

        if ensemble_size is not None:
            for attr_name, attr_value in (
                ("perturbationNumber", 0),
                ("numberOfForecastsInEnsemble", int(ensemble_size)),
                ("numberOfMembersInEnsemble", int(ensemble_size)),
                ("totalNumberOfEnsembleForecasts", int(ensemble_size)),
            ):
                try:
                    setattr(msg, attr_name, attr_value)
                except Exception:
                    continue

        # 6. Adjust decimal scale factor to improve precision for select variables
        msg.binaryScaleFactor = 0
        if var_name == "SPFH" or var_name == "SPFH_0C" or var_name == "SPFH2M":
            if surface_value and surface_value >= 5000 and surface_value <= 10000:
                msg.decScaleFactor = 12
            elif surface_value and surface_value >= 15000 and surface_value <= 40000:
                msg.decScaleFactor = 10
            else:
                msg.decScaleFactor = 8
        elif var_name in ["PWAT"]:
            # Precipitable water: typically 0-80 mm, use higher precision
            msg.decScaleFactor = 3
        elif var_name in ["CRAIN", "CFRZR", "APCP"]:
            # Precipitation: typically small values in mm/hr, use high precision
            msg.decScaleFactor = 4
        elif var_name in ["VUCSH_0_1km", "VVCSH_0_1km", "VUCSH_0_6km", "VVCSH_0_6km"]:
            # Wind shear: typically small values (1/s), use high precision
            msg.decScaleFactor = 5
        elif var_name in ["RELV_max_0_1km", "RELV_max_0_2km"]:
            # Relative vorticity: typically 1e-3 to 1e-2 s^-1, use high precision
            msg.decScaleFactor = 5
        else:
            msg.decScaleFactor = 2

        # 7. Spatial differencing order (disable for discontinuous fields like visibility)
        if var_name in ["VIS", "HGTCC"]:
            msg.spatialDifferenceOrder = 0
        else:
            msg.spatialDifferenceOrder = 2

        return msg

    def _resolve_probability_grib_mapping(self, var_name: str, base_var: str, operator: str) -> Tuple[int, int, int, int, Optional[float]]:
        if base_var not in GRIB_PARAM_MAP:
            raise ValueError(f"Unknown base variable for probability field {var_name}")
        if base_var not in PROBABILITY_THRESHOLD_MAP:
            raise ValueError(f"Unsupported probability field {base_var}")

        prob_config = PROBABILITY_THRESHOLD_MAP[base_var]
        expected_operator = prob_config["operator"]
        if (operator in (">", "GT") and expected_operator != "gt") or (operator in ("<", "LT") and expected_operator != "lt"):
            raise ValueError(f"Unsupported probability operator for {var_name}")

        disc, _, _, default_surface, surface_value = GRIB_PARAM_MAP[base_var]
        return disc, GRIB_PARAM_MAP[base_var][1], GRIB_PARAM_MAP[base_var][2], default_surface, surface_value

    def _parse_probability_var_name(self, var_name: str) -> Optional[Tuple[str, str, float]]:
        if "_PROB_" not in var_name:
            return None

        base_var, suffix = var_name.split("_PROB_", 1)
        if not base_var or not suffix:
            return None

        operator = None
        rest = suffix
        for token in ("GT", "LT"):
            if suffix.startswith(token):
                operator = token
                rest = suffix[len(token):]
                break
        if operator is None:
            return None

        if not rest:
            return None

        if rest.startswith("m"):
            threshold = -float(rest[1:].replace("p", "."))
        else:
            threshold = float(rest.replace("p", "."))

        return base_var, operator, threshold

    def _get_grib_param_info(self, var_name: str, da: Optional[xr.DataArray] = None) -> Tuple[int, int, int, int, Optional[float]]:
        if da is not None and "base_variable" in da.attrs and "probability_threshold" in da.attrs:
            base_var = str(da.attrs["base_variable"])
            operator = str(da.attrs.get("probability_operator", ">"))
            return self._resolve_probability_grib_mapping(var_name, base_var, operator)

        parsed = self._parse_probability_var_name(var_name)
        if parsed is not None:
            base_var, operator, threshold = parsed
            if base_var in PROBABILITY_THRESHOLD_MAP:
                if threshold not in PROBABILITY_THRESHOLD_MAP[base_var]["thresholds"]:
                    # Thresholds are stored as floats. Allow a small tolerance to cover
                    # representation differences like 12.7 vs 12.700000762939453.
                    threshold_matches = any(abs(float(th) - float(threshold)) < 1e-6 for th in PROBABILITY_THRESHOLD_MAP[base_var]["thresholds"])
                    if not threshold_matches:
                        raise ValueError(f"Unsupported probability threshold {threshold} for {base_var}")
                return self._resolve_probability_grib_mapping(var_name, base_var, operator)

        if var_name in GRIB_PARAM_MAP:
            return GRIB_PARAM_MAP[var_name]

        raise ValueError(f"Unknown variable {var_name} not in GRIB_PARAM_MAP")

    def _get_surface_type_and_value(self, var_name: str, ds: xr.Dataset, da: xr.DataArray) -> Tuple[int, Optional[float]]:
        _, _, _, surface_type, surface_value = self._get_grib_param_info(var_name, da)
        return surface_type, surface_value

    def _get_ensemble_size(self, ds: xr.Dataset, da: xr.DataArray) -> Optional[int]:
        ensemble_size = da.attrs.get("ensemble_size", ds.attrs.get("ensemble_size"))
        if ensemble_size is None:
            return None
        try:
            return int(ensemble_size)
        except (TypeError, ValueError):
            return None

    def save_grib2(self, forecast_starttime: datetime, ds_hour: xr.Dataset, output_path: str) -> None:
        """Write a single-hour GRIB2 file from an xarray.Dataset using grib2io.

        ds_hour is expected to have dims (lead_time=1, time=1, [level], y, x) and contain
        both pressure-level and surface variables.
        """
        # Extract lead hour
        try:
            lead = int(np.asarray(ds_hour["lead_time"]).item())
        except Exception:
            lead = 0

        outfile = output_path

        # Prepare all messages outside the lock (parallel-safe operations)
        # This includes dataset iteration, numpy operations, and message building
        messages_to_write = []
        current_var_name = None

        try:
            # Ensure y,x dims exist (rename from latitude/longitude if needed)
            ds_loc = ds_hour
            if "y" not in ds_loc.dims or "x" not in ds_loc.dims:
                if "latitude" in ds_loc.dims and "longitude" in ds_loc.dims:
                    ds_loc = ds_loc.rename_dims({"latitude": "y", "longitude": "x"})
                else:
                    logger.warning("Dataset missing y/x dims; attempting to infer from data variable shapes.")

            # Loop over variables in sorted order for stable output
            for var_name in sorted(ds_loc.data_vars):
                current_var_name = var_name
                da = ds_loc[var_name]
                try:
                    self._get_grib_param_info(var_name, da)
                except ValueError:
                    logger.debug(f"Skipping unknown variable {var_name}")
                    continue

                surface_type, surface_value = self._get_surface_type_and_value(var_name, ds_loc, da)
                ensemble_size = self._get_ensemble_size(ds_loc, da)
                probability_threshold = da.attrs.get("probability_threshold")
                probability_operator = str(da.attrs.get("probability_operator", ">"))
                base_variable = da.attrs.get("base_variable")
                accumulation_hours = da.attrs.get("accumulation_hours")
                if accumulation_hours is not None:
                    accumulation_hours = int(accumulation_hours)

                # Pressure-level variables
                if "level" in da.coords:
                    for level in np.atleast_1d(da["level"].values):
                        # Ensure pressure level is in Pa (convert from hPa/mb if necessary)
                        plevel = float(level)
                        if plevel < 2000:  # assume provided in hPa
                            plevel *= 100.0
                        msg = self._build_message(
                            var_name,
                            forecast_starttime,
                            lead,
                            surface_type=100,
                            surface_value=plevel,
                            ensemble_size=ensemble_size,
                            probability_threshold=probability_threshold,
                            probability_operator=probability_operator,
                            base_variable=base_variable,
                            accumulation_hours=accumulation_hours,
                        )
                        # Expect data shape (lead_time=1, time=1, level=1, y, x) or (lead_time=1, level=1, y, x)
                        # Squeeze removes singleton dimensions regardless of order
                        vals = np.squeeze(da.sel(level=level).values)
                        # Slice out time/lead if present
                        if vals.ndim == 4:
                            vals2d = vals[0, 0, :, :]
                        elif vals.ndim == 3:
                            vals2d = vals[0, :, :]
                        else:
                            vals2d = vals
                        msg.data = np.asarray(vals2d)
                        messages_to_write.append(msg)
                else:
                    msg = self._build_message(
                        var_name,
                        forecast_starttime,
                        lead,
                        surface_type=surface_type,
                        surface_value=surface_value,
                        ensemble_size=ensemble_size,
                        probability_threshold=probability_threshold,
                        probability_operator=probability_operator,
                        base_variable=base_variable,
                        accumulation_hours=accumulation_hours,
                    )
                    vals = np.squeeze(da.values)
                    if vals.ndim == 3:
                        vals2d = vals[0, 0, :, :]
                    elif vals.ndim == 2:
                        vals2d = vals
                    else:
                        vals2d = np.squeeze(vals)
                    msg.data = np.asarray(vals2d)
                    messages_to_write.append(msg)
        except Exception as e:
            raise RuntimeError(
                f"Error preparing GRIB message for {current_var_name} "
                f"({type(e).__name__}): {e!r}"
            ) from e

        # Now serialize the grib2io operations (g2c library may not be thread-safe)
        with self._grib2io_lock:
            # Remove existing file if present
            if os.path.isfile(outfile):
                os.remove(outfile)

            # Open GRIB2 file for writing
            g2 = grib2io.open(outfile, mode="w")

            try:
                # Write all prepared messages
                for msg in messages_to_write:
                    msg.pack()  # g2c packing may use global state
                    g2.write(msg)
            except Exception:
                g2.close()
                if os.path.isfile(outfile):
                    os.remove(outfile)
                raise
            else:
                g2.close()

        # Optionally create an index via wgrib2 if available
        try:
            wgrib2 = os.environ.get("WGRIB2", "wgrib2")
            idxfile = f"{outfile}.idx"
            t0 = time.time()
            with open(idxfile, "w") as f_out:
                subprocess.run([wgrib2, "-s", outfile], stdout=f_out, check=True)
            t1 = time.time()
            logger.info(f"Index created in {t1 - t0:.2f}s: {idxfile}")
        except Exception as e:
            logger.warning(f"Skipping index creation with wgrib2: {e}")
