"""
CF (Climate and Forecast) metadata conventions for HRRRCast output.

This module centralizes everything related to CF-1.6 metadata and GRIB2 encoding:
the per-variable metadata table (:data:`VARIABLE_METADATA`) containing both CF
attributes (long_name/units) and GRIB2 encoding parameters, plus the
:func:`apply_cf_attributes` helper that annotates a HRRRCast dataset with
the grid mapping, projection coordinates, and global attributes required
by the CF conventions.

The consolidated :data:`VARIABLE_METADATA` dictionary ensures consistency between
NetCDF (CF-1.6) and GRIB2 outputs by maintaining all variable metadata in one place.
"""

from datetime import datetime

import numpy as np
import xarray as xr


# Consolidated metadata for all model output and diagnostic variables
# Each entry contains:
#   - long_name: CF-compliant descriptive name
#   - units: CF-compliant units string
#   - cmap: matplotlib colormap name for plotting
#   - grib2: (discipline, category, number, surface_type, surface_value) for GRIB2 encoding
VARIABLE_METADATA = {
    # Pressure-level variables
    "UGRD": {
        "long_name": "U-component of wind",
        "units": "m s-1",
        "cmap": "RdBu_r",
        "grib2": (0, 2, 2, 100, None),
    },
    "VGRD": {
        "long_name": "V-component of wind",
        "units": "m s-1",
        "cmap": "RdBu_r",
        "grib2": (0, 2, 3, 100, None),
    },
    "VVEL": {
        "long_name": "Vertical velocity (pressure coordinate)",
        "units": "Pa s-1",
        "cmap": "RdBu_r",
        "grib2": (0, 2, 8, 100, None),
    },
    "TMP": {
        "long_name": "Temperature",
        "units": "K",
        "cmap": "coolwarm",
        "grib2": (0, 0, 0, 100, None),
    },
    "HGT": {
        "long_name": "Geopotential height",
        "units": "m",
        "cmap": "terrain",
        "grib2": (0, 3, 5, 100, None),
    },
    "SPFH": {
        "long_name": "Specific humidity",
        "units": "kg kg-1",
        "cmap": "Blues",
        "grib2": (0, 1, 0, 100, None),
    },
    # Surface variables
    "PRES": {
        "long_name": "Surface pressure",
        "units": "Pa",
        "cmap": "viridis",
        "grib2": (0, 3, 0, 1, None),
    },
    "MSLMA": {
        "long_name": "Mean sea level pressure (MAPS reduction)",
        "units": "Pa",
        "cmap": "viridis",
        "grib2": (0, 3, 198, 101, None),
    },
    "REFC": {
        "long_name": "Composite reflectivity",
        "units": "dBZ",
        "cmap": "pyart_NWSRef",
        "grib2": (0, 16, 196, 10, None),
    },
    "T2M": {
        "long_name": "2-meter temperature",
        "units": "K",
        "cmap": "coolwarm",
        "grib2": (0, 0, 0, 103, 2.0),
    },
    "UGRD10M": {
        "long_name": "U-component of 10-meter wind",
        "units": "m s-1",
        "cmap": "RdBu_r",
        "grib2": (0, 2, 2, 103, 10.0),
    },
    "VGRD10M": {
        "long_name": "V-component of 10-meter wind",
        "units": "m s-1",
        "cmap": "RdBu_r",
        "grib2": (0, 2, 3, 103, 10.0),
    },
    "UGRD80M": {
        "long_name": "U-component of 80-meter wind",
        "units": "m s-1",
        "cmap": "RdBu_r",
        "grib2": (0, 2, 2, 103, 80.0),
    },
    "VGRD80M": {
        "long_name": "V-component of 80-meter wind",
        "units": "m s-1",
        "cmap": "RdBu_r",
        "grib2": (0, 2, 3, 103, 80.0),
    },
    "D2M": {
        "long_name": "2-meter dew point temperature",
        "units": "K",
        "cmap": "coolwarm",
        "grib2": (0, 0, 6, 103, 2.0),
    },
    "TCDC": {
        "long_name": "Total cloud cover",
        "units": "%",
        "cmap": "Greys",
        "grib2": (0, 6, 1, 10, None),
    },
    "LCDC": {
        "long_name": "Low-level cloud cover",
        "units": "%",
        "cmap": "Blues",
        "grib2": (0, 6, 3, 214, None),
    },
    "MCDC": {
        "long_name": "Mid-level cloud cover",
        "units": "%",
        "cmap": "Greens",
        "grib2": (0, 6, 4, 224, None),
    },
    "HCDC": {
        "long_name": "High-level cloud cover",
        "units": "%",
        "cmap": "Reds",
        "grib2": (0, 6, 5, 234, None),
    },
    "VIS": {
        "long_name": "Visibility",
        "units": "m",
        "cmap": "plasma_r",
        "grib2": (0, 19, 0, 1, None),
    },
    "APCP": {
        "long_name": "Total precipitation",
        "units": "kg m-2",
        "cmap": "Blues",
        "grib2": (0, 1, 8, 1, None),
    },
    "HGTCC": {
        "long_name": "Convective cloud top height",
        "units": "m",
        "cmap": "cividis",
        "grib2": (0, 3, 5, 215, None),
    },
    "CAPE": {
        "long_name": "Convective available potential energy",
        "units": "J kg-1",
        "cmap": "Spectral_r",
        "grib2": (0, 7, 6, 1, None),
    },
    "CIN": {
        "long_name": "Convective inhibition",
        "units": "J kg-1",
        "cmap": "PuOr",
        "grib2": (0, 7, 7, 1, None),
    },
    # Constant/static fields
    "LAND": {
        "long_name": "Land-sea mask",
        "units": "1",
        "cmap": "Greys",
        "grib2": (2, 0, 0, 1, None),
    },
    "OROG": {
        "long_name": "Surface orography",
        "units": "m",
        "cmap": "terrain",
        "grib2": (0, 3, 5, 1, None),
    },
    # Surface diagnostics
    "R2M": {
        "long_name": "Relative humidity at 2 m",
        "units": "%",
        "cmap": "YlGnBu",
        "grib2": (0, 1, 1, 103, 2.0),
    },
    "SPFH2M": {
        "long_name": "Specific humidity at 2 m",
        "units": "kg kg-1",
        "cmap": "Blues",
        "grib2": (0, 1, 0, 103, 2.0),
    },
    "POT2M": {
        "long_name": "Potential temperature at 2 m",
        "units": "K",
        "cmap": "coolwarm",
        "grib2": (0, 0, 2, 103, 2.0),
    },
    # Column-integrated
    "PWAT": {
        "long_name": "Precipitable water",
        "units": "kg m-2",
        "cmap": "YlGnBu",
        "grib2": (0, 1, 3, 10, None),
    },
    # Precipitation diagnostics
    "CRAIN": {
        "long_name": "Conditional rain rate",
        "units": "kg m-2",
        "cmap": "Blues",
        "grib2": (0, 1, 33, 1, None),
    },
    "RAIN_MASK": {
        "long_name": "Rain occurrence mask",
        "units": "1",
        "cmap": "Greys",
    },
    "RAIN_FRACTION": {
        "long_name": "Rain area fraction",
        "units": "1",
        "cmap": "Blues",
    },
    "CFRZR": {
        "long_name": "Conditional freezing rain rate",
        "units": "kg m-2",
        "cmap": "PuBu",
        "grib2": (0, 1, 34, 1, None),
    },
    "FRZR_MASK": {
        "long_name": "Freezing rain occurrence mask",
        "units": "1",
        "cmap": "Greys",
    },
    "FRZR_FRACTION": {
        "long_name": "Freezing rain area fraction",
        "units": "1",
        "cmap": "PuBu",
    },
    "WARM_LAYER_DEPTH": {
        "long_name": "Warm layer depth above surface",
        "units": "hPa",
        "cmap": "YlOrRd",
    },
    "COLD_LAYER_DEPTH": {
        "long_name": "Cold layer depth near surface",
        "units": "hPa",
        "cmap": "PuBuGn",
    },
    # Wind diagnostics
    "GUST": {
        "long_name": "Wind gust speed",
        "units": "m s-1",
        "cmap": "viridis",
        "grib2": (0, 2, 22, 1, None),
    },
    "GUST_FACTOR": {
        "long_name": "Empirical gust factor estimate",
        "units": "m s-1",
        "cmap": "magma",
    },
    "GUST_CONV": {
        "long_name": "Convective gust estimate",
        "units": "m s-1",
        "cmap": "magma",
    },
    "WIND_10M": {
        "long_name": "10-meter wind speed",
        "units": "m s-1",
        "cmap": "viridis",
        "grib2": (0, 2, 1, 103, 10.0),
    },
    "WIND_MAX": {
        "long_name": "Maximum wind speed in lower atmospheric column",
        "units": "m s-1",
        "cmap": "viridis",
    },
    # Convective diagnostics
    "VUCSH_0_1km": {
        "long_name": "U-component wind shear rate 0-1 km AGL",
        "units": "s-1",
        "cmap": "RdBu_r",
        "grib2": (0, 2, 15, 103, (1000.0, 0.0)),
    },
    "VVCSH_0_1km": {
        "long_name": "V-component wind shear rate 0-1 km AGL",
        "units": "s-1",
        "cmap": "RdBu_r",
        "grib2": (0, 2, 16, 103, (1000.0, 0.0)),
    },
    "VUCSH_0_6km": {
        "long_name": "U-component wind shear rate 0-6 km AGL",
        "units": "s-1",
        "cmap": "RdBu_r",
        "grib2": (0, 2, 15, 103, (6000.0, 0.0)),
    },
    "VVCSH_0_6km": {
        "long_name": "V-component wind shear rate 0-6 km AGL",
        "units": "s-1",
        "cmap": "RdBu_r",
        "grib2": (0, 2, 16, 103, (6000.0, 0.0)),
    },
    "RELV_max_0_1km": {
        "long_name": "Maximum relative vorticity 0-1 km AGL",
        "units": "s-1",
        "cmap": "Spectral_r",
        "grib2": (0, 2, 12, 103, (1000.0, 0.0)),
    },
    "RELV_max_0_2km": {
        "long_name": "Maximum relative vorticity 0-2 km AGL",
        "units": "s-1",
        "cmap": "Spectral_r",
        "grib2": (0, 2, 12, 103, (2000.0, 0.0)),
    },
    "USTM_0_6km": {
        "long_name": "U-component of storm motion 0-6 km (Bunkers)",
        "units": "m s-1",
        "cmap": "RdBu_r",
        "grib2": (0, 2, 194, 103, (0.0, 6000.0)),
    },
    "VSTM_0_6km": {
        "long_name": "V-component of storm motion 0-6 km (Bunkers)",
        "units": "m s-1",
        "cmap": "RdBu_r",
        "grib2": (0, 2, 195, 103, (0.0, 6000.0)),
    },
    "HLCY_0_1km": {
        "long_name": "Storm-relative helicity 0-1 km AGL",
        "units": "m2 s-2",
        "cmap": "PuOr",
        "grib2": (0, 7, 8, 103, (1000.0, 0.0)),
    },
    "HLCY_0_3km": {
        "long_name": "Storm-relative helicity 0-3 km AGL",
        "units": "m2 s-2",
        "cmap": "PuOr",
        "grib2": (0, 7, 8, 103, (3000.0, 0.0)),
    },
    # Diagnostic fields - updraft helicity
    "MXUPHL_max_0_2km": {
        "long_name": "Maximum updraft helicity 0-2 km AGL",
        "units": "m2 s-2",
        "cmap": "RdPu",
        "grib2": (0, 7, 199, 103, (2000.0, 0.0)),
    },
    "MNUPHL_min_0_2km": {
        "long_name": "Minimum updraft helicity 0-2 km AGL",
        "units": "m2 s-2",
        "cmap": "RdPu",
        "grib2": (0, 7, 200, 103, (2000.0, 0.0)),
    },
    "MXUPHL_max_0_3km": {
        "long_name": "Maximum updraft helicity 0-3 km AGL",
        "units": "m2 s-2",
        "cmap": "RdPu",
        "grib2": (0, 7, 199, 103, (3000.0, 0.0)),
    },
    "MNUPHL_min_0_3km": {
        "long_name": "Minimum updraft helicity 0-3 km AGL",
        "units": "m2 s-2",
        "cmap": "RdPu",
        "grib2": (0, 7, 200, 103, (3000.0, 0.0)),
    },
    "MXUPHL_max_2_5km": {
        "long_name": "Maximum updraft helicity 2-5 km AGL",
        "units": "m2 s-2",
        "cmap": "RdPu",
        "grib2": (0, 7, 199, 103, (5000.0, 2000.0)),
    },
    "MNUPHL_min_2_5km": {
        "long_name": "Minimum updraft helicity 2-5 km AGL",
        "units": "m2 s-2",
        "cmap": "RdPu",
        "grib2": (0, 7, 200, 103, (5000.0, 2000.0)),
    },
    # Diagnostic fields - vertical velocity extrema
    "MAXUVV_max_100_1000mb": {
        "long_name": "Maximum upward vertical velocity 100-1000 hPa",
        "units": "Pa s-1",
        "cmap": "Reds",
        "grib2": (0, 2, 220, 100, (10000.0, 100000.0)),
    },
    "MAXDVV_max_100_1000mb": {
        "long_name": "Maximum downward vertical velocity 100-1000 hPa",
        "units": "Pa s-1",
        "cmap": "Blues",
        "grib2": (0, 2, 221, 100, (10000.0, 100000.0)),
    },
    # 0 degC isotherm diagnostics
    "HGT_0C": {
        "long_name": "Height AGL at the 0 degC isotherm",
        "units": "m",
        "cmap": "terrain",
        "grib2": (0, 3, 5, 4, None),
    },
    "UGRD_0C": {
        "long_name": "U-component of wind at 0 degC isotherm",
        "units": "m s-1",
        "cmap": "RdBu_r",
        "grib2": (0, 2, 2, 4, None),
    },
    "VGRD_0C": {
        "long_name": "V-component of wind at 0 degC isotherm",
        "units": "m s-1",
        "cmap": "RdBu_r",
        "grib2": (0, 2, 3, 4, None),
    },
    "WIND_0C": {
        "long_name": "Wind speed at 0 degC isotherm",
        "units": "m s-1",
        "cmap": "viridis",
        "grib2": (0, 2, 1, 4, None),
    },
    "SPFH_0C": {
        "long_name": "Specific humidity at 0 degC isotherm",
        "units": "kg kg-1",
        "cmap": "Blues",
        "grib2": (0, 1, 0, 4, None),
    },
    "DU_SFC_0C": {
        "long_name": "U-component wind shear surface to 0 degC isotherm",
        "units": "m s-1",
        "cmap": "RdBu_r",
    },
    "DV_SFC_0C": {
        "long_name": "V-component wind shear surface to 0 degC isotherm",
        "units": "m s-1",
        "cmap": "RdBu_r",
    },
    "SHEAR_SFC_0C": {
        "long_name": "Wind shear magnitude surface to 0 degC isotherm",
        "units": "m s-1",
        "cmap": "viridis",
    },
    "RH_0C": {
        "long_name": "Relative humidity at 0 degC isotherm",
        "units": "%",
        "cmap": "YlGnBu",
        "grib2": (0, 1, 1, 4, None),
    },
}


# Derived dictionary for backward compatibility: CF attributes only
CF_ATTRS = {
    var: {k: v for k, v in meta.items() if k in ["long_name", "units"]}
    for var, meta in VARIABLE_METADATA.items()
}


# CF-compliant coordinate attributes for standard forecast dimensions
CF_COORD_ATTRS = {
    "time": {
        "standard_name": "time",
        "long_name": "time",
        "axis": "T",
    },
    "lead_time": {
        "standard_name": "forecast_period",
        "long_name": "forecast period",
        "units": "hours",
    },
    "level": {
        "standard_name": "air_pressure",
        "long_name": "pressure level",
        "units": "hPa",
        "positive": "down",
        "axis": "Z",
    },
    "latitude": {
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
    },
    "longitude": {
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east",
    },
    "forecast_reference_time": {
        "standard_name": "forecast_reference_time",
        "long_name": "model initialization time",
    },
    "x": {
        "standard_name": "projection_x_coordinate",
        "long_name": "x coordinate of projection",
        "units": "m",
        "axis": "X",
    },
    "y": {
        "standard_name": "projection_y_coordinate",
        "long_name": "y coordinate of projection",
        "units": "m",
        "axis": "Y",
    },
}


def apply_cf_attributes(ds: xr.Dataset, init_datetime=None) -> xr.Dataset:
    """Apply CF-1.6 metadata to a HRRRCast dataset.

    Sets variable-level ``long_name``/``units`` from :data:`CF_ATTRS`, attaches the
    Lambert Conformal Conic ``grid_mapping`` variable, adds 1D projection x/y
    auxiliary coordinates required by CF \u00a75.6, marks data variables with
    ``coordinates="latitude longitude"`` so the 2D auxiliary lat/lon arrays are
    discoverable, and writes the required global attributes.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to annotate with CF metadata.
    init_datetime : datetime.datetime, optional
        Forecast initialization time (UTC). When provided it is written as the
        ``initialization_time`` global attribute in ISO 8601 form.
    """
    # CF best practice (and an IOOS compliance-checker requirement): a 2D
    # auxiliary coordinate variable must not share its name with a dimension.
    # Internally the dataset uses dim names "latitude"/"longitude"; rename
    # those dims to y/x at write time so the 2D ``latitude(y, x)`` and
    # ``longitude(y, x)`` arrays no longer collide with their dimensions.
    rename_map = {}
    if "latitude" in ds.dims:
        rename_map["latitude"] = "y"
    if "longitude" in ds.dims:
        rename_map["longitude"] = "x"
    if rename_map:
        ds = ds.rename_dims(rename_map)

    # Variable-level: long_name, units, grid_mapping, and coordinates reference.
    for var in ds.data_vars:
        if var == "grid_mapping":
            continue
        if var in CF_ATTRS:
            ds[var].attrs.update(CF_ATTRS[var])
        ds[var].attrs["grid_mapping"] = "grid_mapping"
        # CF \u00a75.5: only declare the 2D lat/lon auxiliary coordinates on
        # variables whose dims actually contain the spatial axes. Domain-
        # aggregated diagnostics like RAIN_FRACTION/FRZR_FRACTION are reduced
        # to (time,) and would otherwise fail the auxiliary-coord-subset rule.
        if "y" in ds[var].dims and "x" in ds[var].dims:
            ds[var].attrs["coordinates"] = "latitude longitude"

    # Grid mapping variable (scalar int, CF convention)
    ds["grid_mapping"] = xr.DataArray(
        np.int32(0),
        attrs={
            "grid_mapping_name":             "lambert_conformal_conic",
            "standard_parallel":             38.5,
            "longitude_of_central_meridian": -97.5,
            "latitude_of_projection_origin": 38.5,
            "false_easting":                 0.0,
            "false_northing":                0.0,
            "earth_radius":                  6371229.0,
            "GRIB_earth_shape":              "spherical",
            "GRIB_earth_shape_code":         6,
        },
    )

    # CF \u00a75.6 requires variables with standard_name=projection_x_coordinate /
    # projection_y_coordinate when a Lambert Conformal grid_mapping is declared.
    # HRRR is a 3-km grid; the renamed x/y dimensions carry these as 1D dim
    # coordinates expressed in metres relative to the grid centre.
    HRRR_DX_M = 3000.0
    if "x" in ds.dims and "x" not in ds.coords:
        nx = ds.sizes["x"]
        x_vals = ((np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0) * HRRR_DX_M)
        ds = ds.assign_coords(
            x=("x", x_vals, CF_COORD_ATTRS["x"]),
        )
    if "y" in ds.dims and "y" not in ds.coords:
        ny = ds.sizes["y"]
        y_vals = ((np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0) * HRRR_DX_M)
        ds = ds.assign_coords(
            y=("y", y_vals, CF_COORD_ATTRS["y"]),
        )

    # Global attributes
    ds.attrs["Conventions"] = "CF-1.6"
    ds.attrs["title"] = "HRRRCast forecast output"
    ds.attrs["institution"] = "NOAA Global Systems Laboratory"
    ds.attrs["source"] = "HRRRCast"
    ds.attrs["history"] = (
        f"{datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}: "
        f"created by HRRRCast forecast pipeline"
    )
    ds.attrs["references"] = "https://github.com/NOAA-GSL/HRRRCast"
    if init_datetime is not None:
        ds.attrs["initialization_time"] = init_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
    return ds


def get_cf_encoding(ds: xr.Dataset, init_datetime: datetime) -> dict:
    """Build CF-compliant encoding dictionary for NetCDF output.

    Constructs encoding specifications that ensure CF-1.6 compliance when
    writing a dataset to NetCDF. Sets appropriate fill values for data
    variables and coordinates, forbids fill values on coordinate variables
    per CF §2.5.1, and uses CF-compatible time encoding.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to encode. Used to determine which coordinates are present.
    init_datetime : datetime.datetime
        Forecast initialization time (UTC). Used for time coordinate encoding.

    Returns
    -------
    dict
        Encoding dictionary suitable for passing to ``xr.Dataset.to_netcdf()``.
    """
    # Apply CF-compliance encoding
    encoding = {v: {"_FillValue": np.float32(-9999.0)}
                for v in ds.data_vars if v != "grid_mapping"}
    encoding["latitude"] = {"_FillValue": np.float32(-9999.0)}
    encoding["longitude"] = {"_FillValue": np.float32(-9999.0)}
    # CF-1.6: store time as float64 hours since the initialization time
    # (xarray would otherwise emit int64 nanoseconds, which is not a CF type).
    # CF §2.5.1 forbids _FillValue on coordinate variables.
    encoding["time"] = {
        "units": f"hours since {init_datetime.strftime('%Y-%m-%d %H:%M:%S')}",
        "calendar": "standard",
        "dtype": "float64",
        "_FillValue": None,
    }
    if "level" in ds.coords:
        encoding["level"] = {"dtype": "int32", "_FillValue": None}
    # CF §2.5.1: projection coordinate variables x, y must not have _FillValue.
    if "x" in ds.coords:
        encoding["x"] = {"_FillValue": None}
    if "y" in ds.coords:
        encoding["y"] = {"_FillValue": None}
    # CF §2.5.1: coordinate variables must not have _FillValue.
    if "forecast_reference_time" in ds.coords:
        encoding["forecast_reference_time"] = {
            "units": "hours since 1970-01-01 00:00:00",
            "calendar": "standard",
            "dtype": "float64",
            "_FillValue": None,
        }
    if "lead_time" in ds.coords:
        encoding["lead_time"] = {"dtype": "float32", "_FillValue": None}
    return encoding
