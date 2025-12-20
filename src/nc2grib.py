"""
GRIB2 writer using grib2io for HRRRCast outputs.

This module replaces the earlier iris/eccodes-based writer with a direct grib2io
implementation inspired by NOAA-EMC MLGlobal's grib2writer.py.

Notes/assumptions:
- We require a valid GRIB2 Section 3 (grid definition) for the HRRRCast Lambert
    Conformal grid. Provide this via the Netcdf2Grib(section3=...) constructor or
    by setting the environment variable NETCDF2GRIB_SECTION3 to a .npy file containing
    the section3 integer array. If neither is provided, we auto-construct a canonical
    HRRR-like Lambert Conformal Section 3 for the downsampled 6 km grid (Nx=900, Ny=530).
- Product Definition Template Numbers (pdtn) and Data Representation Template
  Numbers (drtn) default to 0 (instantaneous forecast, simple packing). For
  accumulated fields (e.g., APCP) you may wish to adjust pdtn and the duration
  semantics to match downstream consumers.
"""

import os
import json
import subprocess
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import xarray as xr
import grib2io

from utils import setup_logging

logger = setup_logging("INFO")


# Minimal GRIB parameter map: var -> (discipline, category, number, default_surface_type)
GRIB_PARAM_MAP = {
    # Pressure level fields
    "UGRD": (0, 2, 2, 100),   # u-wind
    "VGRD": (0, 2, 3, 100),   # v-wind
    "VVEL": (0, 2, 8, 100),   # vertical velocity (Pa/s)
    "TMP":  (0, 0, 0, 100),   # temperature
    "HGT":  (0, 3, 5, 100),   # geopotential height
    "SPFH": (0, 1, 1, 100),   # specific humidity
    # Surface/height fields
    "PRES":    (0, 3, 0, 1),    # pressure (surface)
    "MSLMA":   (0, 3, 1, 102),  # mean sea level pressure
    "T2M":     (0, 0, 0, 103),  # temperature at 2m
    "UGRD10M": (0, 2, 2, 103),
    "VGRD10M": (0, 2, 3, 103),
    "UGRD80M": (0, 2, 2, 103),
    "VGRD80M": (0, 2, 3, 103),
    "D2M":     (0, 0, 6, 103),  # dewpoint at 2m
    "R2M":     (0, 1, 1, 103),  # RH at 2m
    "TCDC":    (0, 6, 1, 10),   # total cloud cover, entire atmosphere
    "VIS":     (0, 19, 0, 1),   # visibility at surface
    "APCP":    (0, 1, 8, 1),    # total precipitation at surface
    "HGTCC":   (0, 6, 13, 1),   # cloud ceiling height (approx)
    "CAPE":    (0, 7, 6, 1),
    "CIN":     (0, 7, 7, 1),
    "REFC":    (0, 16, 196, 10),# reflectivity, entire atmosphere
    "LAND":    (2, 0, 0, 1),    # land-sea mask
    "OROG":    (0, 3, 5, 1),    # orography
}


class Netcdf2Grib:
    def __init__(self, section3: Optional[np.ndarray] = None, pdtn_default: int = 0, drtn_default: int = 0):
        self.section3 = self._resolve_section3(section3)
        self.pdtn_default = pdtn_default
        self.drtn_default = drtn_default

    def construct_section3_hrrr_6km(self, nx: int = 900, ny: int = 530) -> np.ndarray:
        """Construct GRIB2 Section 3 for HRRR-like CONUS Lambert Conformal grid at 6 km.

        This uses canonical HRRR projection parameters and the downsampled dimensions
        defined in preprocessing (grid_width=900, grid_height=530).

        Parameters used:
        - First grid point (La1/Lo1): 21.138123N, 237.280472E
        - Orientation longitude (LoV): 262.5E
        - Standard parallels (Latin1, Latin2): 38.5N, 38.5N
        - Grid spacing (Dx/Dy): 6000 m
        - Earth radius: 6371229 m

        Returns a numpy array suitable for the `section3` argument of grib2io.Grib2Message.

        Note: If grib2io provides a helper for LCC Section 3 creation in your environment,
        this function will attempt to use it. Otherwise, it constructs a fixed array using
        canonical HRRR parameters. You can override via NETCDF2GRIB_SECTION3.
        """
        # Canonical HRRR LCC parameters (matching earlier code and HRRR docs)
        lat1 = 21.138123    # degrees North
        lon1 = 237.280472   # degrees East
        lov = 262.5         # degrees East
        latin1 = 38.5       # degrees North
        latin2 = 38.5       # degrees North
        dx = 6000           # meters
        dy = 6000           # meters
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
        # Scale factors for latitude/longitude (assume default: not used)
        lat_scale = 0
        lon_scale = 0
        # Resolution and component flags: 8 -> winds(grid) per wgrib2 'res 8'
        res_flags = 8
        # Projection centre flag: 0 = north, 1 = south
        proj_center_flag = 0
        # Scanning mode: 0 typically yields input WE:SN, output WE:SN in wgrib2
        scan_mode = 0

        # Section 3 structure (template 3.30 Lambert Conformal) matching grib_dump order:
        # Fields reflect wgrib2/grib_dump output: res=8, scanningMode=64 (WE:SN), LaD=38500000, Dx/Dy=6000000
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
        # Fallback: attempt to construct HRRR-like 6 km LCC Section 3 using known dims (Nx=900, Ny=530)
        try:
            return self.construct_section3_hrrr_6km(nx=900, ny=530)
        except Exception as e:
            raise RuntimeError(
                "GRIB2 Section 3 (grid definition) is required and could not be auto-constructed. "
                "Provide 'section3' to Netcdf2Grib, set NETCDF2GRIB_SECTION3 to a .npy file, or ensure grib2io LCC helper is available. "
                f"Error: {e}"
            )

    def _build_message(
        self,
        var_name: str,
        ref_time: datetime,
        lead_hour: int,
        surface_type: Optional[int] = None,
        surface_value: Optional[float] = None,
        pdtn: Optional[int] = None,
        drtn: Optional[int] = None,
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

        # 2. Determine PDT
        pdtn = pdtn if pdtn is not None else self.pdtn_default

        # 3. Construct message
        msg = grib2io.Grib2Message(
            section1=section1,
            section3=self.section3,
            pdtn=pdtn,
            drtn=self.drtn_default if drtn is None else drtn,
        )

        # 4. Set parameter keys
        if var_name in GRIB_PARAM_MAP:
            disc, cat, num, default_surface = GRIB_PARAM_MAP[var_name]
            msg.discipline = disc
            msg.parameterCategory = cat
            msg.parameterNumber = num
            msg.typeOfFirstFixedSurface = surface_type if surface_type is not None else default_surface
        else:
            msg.discipline = 0
            msg.parameterCategory = 255
            msg.parameterNumber = 255
            msg.typeOfFirstFixedSurface = surface_type if surface_type is not None else 1

        if surface_value is not None:
            msg.scaledValueOfFirstFixedSurface = int(surface_value)
            msg.scaleFactorOfFirstFixedSurface = 0
        else:
            msg.scaledValueOfFirstFixedSurface = 0
            msg.scaleFactorOfFirstFixedSurface = 0

        # 5. Time metadata
        msg.unitOfForecastTime = 1  # hours
        msg.leadTime = timedelta(hours=int(lead_hour))

        # 6. Statistical processing
        msg.typeOfStatisticalProcessing = 0
        msg.numberOfTimeRanges = 0

        # 8. Second surface (unused)
        msg.typeOfSecondFixedSurface = 255
        msg.scaleFactorOfSecondFixedSurface = 0
        msg.scaledValueOfSecondFixedSurface = 0

        return msg

    def _get_surface_type_and_value(self, var_name: str, ds: xr.Dataset, da: xr.DataArray) -> Tuple[int, Optional[float]]:
        # Pressure-level variables have a 'level' coordinate in Pa
        if "level" in da.coords:
            return 100, None  # pressure surface, value set per-level during loop
        # Height AGL variables
        if var_name in ("T2M", "D2M", "R2M"):
            return 103, 2.0
        if var_name in ("UGRD10M", "VGRD10M"):
            return 103, 10.0
        if var_name in ("UGRD80M", "VGRD80M"):
            return 103, 80.0
        # Entire atmosphere
        if var_name in ("TCDC", "REFC"):
            return 10, None
        # Surface
        return GRIB_PARAM_MAP.get(var_name, (0, 0, 0, 1))[3], None

    def save_grib2(self, forecast_starttime: datetime, ds_hour: xr.Dataset, member, outdir: str) -> None:
        """Write a single-hour GRIB2 file from an xarray.Dataset using grib2io.

        ds_hour is expected to have dims (time=1, lead_time=1, [level], y, x) and contain
        both pressure-level and surface variables.
        """
        # Extract lead hour
        try:
            lead = int(np.asarray(ds_hour["lead_time"]).item())
        except Exception:
            lead = 0

        cycle = forecast_starttime.hour
        if member == "avg":
            outfile = os.path.join(outdir, f"hrrrcast.avg.t{cycle:02d}z.pgrb2.f{lead:02d}")
        else:
            outfile = os.path.join(outdir, f"hrrrcast.m{int(member):02d}.t{cycle:02d}z.pgrb2.f{lead:02d}")

        # Remove existing file if present
        if os.path.isfile(outfile):
            os.remove(outfile)

        # Open GRIB2 file for writing
        g2 = grib2io.open(outfile, mode="w")
        logger.info(f"Writing GRIB2: {outfile}")

        # Ensure y,x dims exist (rename from latitude/longitude if needed)
        ds_loc = ds_hour
        if "y" not in ds_loc.dims or "x" not in ds_loc.dims:
            if "latitude" in ds_loc.dims and "longitude" in ds_loc.dims:
                ds_loc = ds_loc.rename_dims({"latitude": "y", "longitude": "x"})
            else:
                logger.warning("Dataset missing y/x dims; attempting to infer from data variable shapes.")

        # Loop over variables in sorted order for stable output
        for var_name in sorted(ds_loc.data_vars):
            da = ds_loc[var_name]
            if var_name not in GRIB_PARAM_MAP:
                logger.debug(f"Skipping unknown variable {var_name}")
                continue

            surface_type, surface_value = self._get_surface_type_and_value(var_name, ds_loc, da)

            # Pressure-level variables
            if "level" in da.coords:
                for level in np.atleast_1d(da["level"].values):
                    # Ensure pressure level is in Pa (convert from hPa/mb if necessary)
                    plevel = float(level)
                    if plevel < 2000:  # assume provided in hPa
                        plevel *= 100.0
                    msg = self._build_message(var_name, forecast_starttime, lead, surface_type=100, surface_value=plevel)
                    # Expect data shape (time=1, lead_time=1, level=1, y, x) or (lead_time=1, level=1, y, x)
                    vals = np.squeeze(da.sel(level=level).values)
                    # Slice out time/lead if present
                    if vals.ndim == 4:
                        _, _, ny, nx = vals.shape
                        vals2d = vals[0, 0, :, :]
                    elif vals.ndim == 3:
                        _, ny, nx = vals.shape
                        vals2d = vals[0, :, :]
                    else:
                        vals2d = vals
                    msg.data = np.asarray(vals2d)
                    msg.pack()
                    g2.write(msg)
            else:
                msg = self._build_message(var_name, forecast_starttime, lead, surface_type=surface_type, surface_value=surface_value)
                vals = np.squeeze(da.values)
                if vals.ndim == 3:
                    # (time, lead, y, x)
                    vals2d = vals[0, 0, :, :]
                elif vals.ndim == 2:
                    vals2d = vals
                else:
                    # Attempt to reduce
                    vals2d = np.squeeze(vals)
                msg.data = np.asarray(vals2d)
                msg.pack()
                g2.write(msg)

        g2.close()

        # Optionally create an index via wgrib2 if available
        try:
            wgrib2 = os.environ.get("WGRIB2", "wgrib2")
            idxfile = f"{outfile}.idx"
            with open(idxfile, "w") as f_out:
                subprocess.run([wgrib2, "-s", outfile], stdout=f_out, check=True)
            logger.info(f"Index created: {idxfile}")
        except Exception as e:
            logger.warning(f"Skipping index creation with wgrib2: {e}")
