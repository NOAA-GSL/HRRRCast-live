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
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import xarray as xr
import grib2io

from utils import setup_logging

logger = setup_logging("INFO")


# Minimal GRIB parameter map: var -> (discipline, category, number, default_surface_type, default_surface_value)
GRIB_PARAM_MAP = {
    # Pressure level fields
    "UGRD": (0, 2, 2, 100, None),   # u-wind
    "VGRD": (0, 2, 3, 100, None),   # v-wind
    "VVEL": (0, 2, 8, 100, None),   # vertical velocity (Pa/s)
    "TMP":  (0, 0, 0, 100, None),   # temperature
    "HGT":  (0, 3, 5, 100, None),   # geopotential height
    "SPFH": (0, 1, 0, 100, None),   # specific humidity
    # Surface/height fields
    "PRES":    (0, 3, 0, 1, None),    # pressure (surface)
    "MSLMA":   (0, 3, 198, 101, None),# mean sea level pressure
    "T2M":     (0, 0, 0, 103, 2.0),   # temperature at 2m
    "UGRD10M": (0, 2, 2, 103, 10.0),
    "VGRD10M": (0, 2, 3, 103, 10.0),
    "UGRD80M": (0, 2, 2, 103, 80.0),
    "VGRD80M": (0, 2, 3, 103, 80.0),
    "D2M":     (0, 0, 6, 103, 2.0),   # dewpoint at 2m
    "R2M":     (0, 1, 1, 103, 2.0),   # RH at 2m
    "SPFH2M":  (0, 1, 0, 103, 2.0),   # specific humidity at 2m
    "POT2M":   (0, 0, 2, 103, 2.0),   # potential temperature at 2m
    "TCDC":    (0, 6, 1, 10, None),   # total cloud cover, entire atmosphere
    "LCDC":    (0, 6, 3, 214, None),  # low cloud cover, entire atmosphere
    "MCDC":    (0, 6, 4, 224, None),  # medium cloud cover, entire atmosphere
    "HCDC":    (0, 6, 5, 234, None),  # high cloud cover, entire atmosphere
    "VIS":     (0, 19, 0, 1, None),   # visibility at surface
    "APCP":    (0, 1, 8, 1, None),    # total precipitation at surface
    "HGTCC":   (0, 3, 5, 215, None),  # cloud ceiling height (approx)
    "CAPE":    (0, 7, 6, 1, None),
    "CIN":     (0, 7, 7, 1, None),
    "REFC":    (0, 16, 196, 10, None),# reflectivity, entire atmosphere
    "LAND":    (2, 0, 0, 1, None),    # land-sea mask
    "OROG":    (0, 3, 5, 1, None),    # orography
    # Diagnostic fields - precipitable water
    "PWAT":    (0, 1, 3, 10, None),   # precipitable water, entire atmosphere
    # Diagnostic fields - conditional rain/freezing rain
    "CRAIN":   (0, 1, 33, 1, None),  # conditional rain rate at surface
    #"RAIN_FRACTION": (0, 1, 194, 1, None), # rain fraction
    #"RAIN_MASK": (0, 1, 193, 1, None),# rain mask at surface
    "CFRZR":   (0, 1, 34, 1, None),  # conditional freezing rain at surface
    #"FRZR_FRACTION": (0, 1, 197, 1, None), # freezing rain fraction
    #"FRZR_MASK": (0, 1, 196, 1, None),# freezing rain mask at surface
    #"WARM_LAYER_DEPTH": (0, 3, 192, 1, None), # warm layer depth (hPa)
    #"COLD_LAYER_DEPTH": (0, 3, 193, 1, None), # cold layer depth (hPa)
    # Diagnostic fields - wind gust
    "GUST":    (0, 2, 22, 1, None),   # wind gust at surface
    #"GUST_FACTOR": (0, 2, 192, 1, None), # empirical gust factor
    #"GUST_CONV": (0, 2, 193, 1, None),# convective gust enhancement
    "WIND_10M": (0, 2, 1, 103, 10.0), # 10m wind speed
    #"WIND_MAX": (0, 2, 194, 1, None), # maximum wind in column
    # Diagnostic fields - convective parameters (shear)
    "VUCSH_0_1km": (0, 2, 15, 103, (1000.0, 0.0)),  # U shear rate 1000-0m AGL (layer)
    "VVCSH_0_1km": (0, 2, 16, 103, (1000.0, 0.0)),  # V shear rate 1000-0m AGL (layer)
    "VUCSH_0_6km": (0, 2, 15, 103, (6000.0, 0.0)),  # U shear rate 6000-0m AGL (layer)
    "VVCSH_0_6km": (0, 2, 16, 103, (6000.0, 0.0)),  # V shear rate 6000-0m AGL (layer)
    # Diagnostic fields - convective parameters (vorticity)
    "RELV_max_0_1km": (0, 2, 12, 103, (1000.0, 0.0)), # max relative vorticity 1000-0m AGL (layer)
    "RELV_max_0_2km": (0, 2, 12, 103, (2000.0, 0.0)), # max relative vorticity 2000-0m AGL (layer)
    # Diagnostic fields - storm motion
    "USTM_0_6km":    (0, 2, 194, 103, (0.0, 6000.0)),  # U-component storm motion 6000-0m AGL (layer)
    "VSTM_0_6km":    (0, 2, 195, 103, (0.0, 6000.0)),  # V-component storm motion 6000-0m AGL (layer)
    # Diagnostic fields - helicity
    "HLCY_0_1km": (0, 7, 8, 103, (1000.0, 0.0)),  # storm-relative helicity 1000-0m AGL (layer)
    "HLCY_0_3km": (0, 7, 8, 103, (3000.0, 0.0)),  # storm-relative helicity 3000-0m AGL (layer)
    # Diagnostic fields - updraft helicity
    "MXUPHL_max_0_2km": (0, 7, 199, 103, (2000.0, 0.0)),  # max updraft helicity 2000-0m AGL (layer)
    "MNUPHL_min_0_2km": (0, 7, 200, 103, (2000.0, 0.0)),  # min updraft helicity 2000-0m AGL (layer)
    "MXUPHL_max_0_3km": (0, 7, 199, 103, (3000.0, 0.0)),  # max updraft helicity 3000-0m AGL (layer)
    "MNUPHL_min_0_3km": (0, 7, 200, 103, (3000.0, 0.0)),  # min updraft helicity 3000-0m AGL (layer)
    "MXUPHL_max_2_5km": (0, 7, 199, 103, (5000.0, 2000.0)),  # max updraft helicity 5000-2000m AGL (layer)
    "MNUPHL_min_2_5km": (0, 7, 200, 103, (5000.0, 2000.0)),  # min updraft helicity 5000-2000m AGL (layer)
    # Diagnostic fields - vertical velocity extrema
    "MAXUVV_max_100_1000mb": (0, 2, 220, 100, (10000.0, 100000.0)),  # max upward vert velocity 100-1000mb (layer, in Pa)
    "MAXDVV_max_100_1000mb": (0, 2, 221, 100, (10000.0, 100000.0)),  # max downward vert velocity 100-1000mb (layer, in Pa)
    # Diagnostic fields - 0C isotherm
    "HGT_0C":  (0, 3, 5, 4, None),    # height AGL at 0C isotherm
    "UGRD_0C": (0, 2, 2, 4, None),    # U-wind at 0C isotherm
    "VGRD_0C": (0, 2, 3, 4, None),    # V-wind at 0C isotherm
    "WIND_0C": (0, 2, 1, 4, None),    # wind speed at 0C isotherm
    "SPFH_0C": (0, 1, 0, 4, None),    # specific humidity at 0C isotherm
    #"DU_SFC_0C": (0, 2, 192, 4, None),# U-wind shear surface to 0C
    #"DV_SFC_0C": (0, 2, 193, 4, None),# V-wind shear surface to 0C
    #"SHEAR_SFC_0C": (0, 2, 194, 4, None), # wind shear magnitude surface to 0C
    "RH_0C":   (0, 1, 1, 4, None),    # relative humidity at 0C isotherm
}


class Netcdf2Grib:
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

        # 2. Construct message
        msg = grib2io.Grib2Message(
            section1=section1,
            section3=self.section3,
            pdtn=self.pdtn_default if pdtn is None else pdtn,
            drtn=self.drtn_default if drtn is None else drtn,
        )

        # 3. Set parameter keys
        if var_name not in GRIB_PARAM_MAP:
            raise ValueError(f"Unknown variable {var_name} not in GRIB_PARAM_MAP")
        disc, cat, num, default_surface, _ = GRIB_PARAM_MAP[var_name]
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
        msg.leadTime = timedelta(hours=int(lead_hour))

        # 5. Statistical processing
        msg.typeOfStatisticalProcessing = 0
        msg.numberOfTimeRanges = 0

        # 6. Adjust decimal scale factor to improve precision for select variables
        msg.binaryScaleFactor = 0
        if var_name == "SPFH" or var_name == "SPFH_0C":
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

    def _get_surface_type_and_value(self, var_name: str, ds: xr.Dataset, da: xr.DataArray) -> Tuple[int, Optional[float]]:
        if var_name not in GRIB_PARAM_MAP:
            raise ValueError(f"Unknown variable {var_name} not in GRIB_PARAM_MAP")
        _, _, _, surface_type, surface_value = GRIB_PARAM_MAP[var_name]
        return surface_type, surface_value

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
                            vals2d = vals[0, 0, :, :]
                        elif vals.ndim == 3:
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
                        vals2d = vals[0, 0, :, :]
                    elif vals.ndim == 2:
                        vals2d = vals
                    else:
                        vals2d = np.squeeze(vals)
                    msg.data = np.asarray(vals2d)
                    msg.pack()
                    g2.write(msg)
        except Exception as e:
            logger.warning("Error writing GRIB messages: %s", e)

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
