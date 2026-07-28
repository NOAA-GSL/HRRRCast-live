"""
CF (Climate and Forecast) metadata conventions for HRRRCast output.

This module centralizes everything related to CF-1.6 metadata: the
per-variable ``long_name``/``units`` table (:data:`CF_ATTRS`) and the
:func:`apply_cf_attributes` helper that annotates a HRRRCast dataset with
the grid mapping, projection coordinates, and global attributes required
by the CF conventions.
"""

from datetime import datetime

import numpy as np
import xarray as xr


# CF-compliant long_name and units for all model output and diagnostic variables
CF_ATTRS = {
    # Pressure-level variables
    "UGRD":             {"long_name": "U-component of wind",                           "units": "m s-1"},
    "VGRD":             {"long_name": "V-component of wind",                           "units": "m s-1"},
    "VVEL":             {"long_name": "Vertical velocity (pressure coordinate)",        "units": "Pa s-1"},
    "TMP":              {"long_name": "Temperature",                                    "units": "K"},
    "HGT":              {"long_name": "Geopotential height",                            "units": "m"},
    "SPFH":             {"long_name": "Specific humidity",                              "units": "kg kg-1"},
    # Surface variables
    "PRES":             {"long_name": "Surface pressure",                               "units": "Pa"},
    "MSLMA":            {"long_name": "Mean sea level pressure (MAPS reduction)",       "units": "Pa"},
    "REFC":             {"long_name": "Composite reflectivity",                         "units": "dBZ"},
    "T2M":              {"long_name": "2-meter temperature",                            "units": "K"},
    "UGRD10M":          {"long_name": "U-component of 10-meter wind",                  "units": "m s-1"},
    "VGRD10M":          {"long_name": "V-component of 10-meter wind",                  "units": "m s-1"},
    "UGRD80M":          {"long_name": "U-component of 80-meter wind",                  "units": "m s-1"},
    "VGRD80M":          {"long_name": "V-component of 80-meter wind",                  "units": "m s-1"},
    "D2M":              {"long_name": "2-meter dew point temperature",                  "units": "K"},
    "TCDC":             {"long_name": "Total cloud cover",                              "units": "%"},
    "LCDC":             {"long_name": "Low-level cloud cover",                          "units": "%"},
    "MCDC":             {"long_name": "Mid-level cloud cover",                          "units": "%"},
    "HCDC":             {"long_name": "High-level cloud cover",                         "units": "%"},
    "VIS":              {"long_name": "Visibility",                                     "units": "m"},
    "APCP":             {"long_name": "Total precipitation",                            "units": "kg m-2"},
    "HGTCC":            {"long_name": "Convective cloud top height",                    "units": "m"},
    "CAPE":             {"long_name": "Convective available potential energy",           "units": "J kg-1"},
    "CIN":              {"long_name": "Convective inhibition",                          "units": "J kg-1"},
    # Constant/static fields
    "LAND":             {"long_name": "Land-sea mask",                                  "units": "1"},
    "OROG":             {"long_name": "Surface orography",                              "units": "m"},
    # Surface diagnostics
    "R2M":              {"long_name": "Relative humidity at 2 m",                       "units": "%"},
    "SPFH2M":           {"long_name": "Specific humidity at 2 m",                       "units": "kg kg-1"},
    "POT2M":            {"long_name": "Potential temperature at 2 m",                   "units": "K"},
    # Column-integrated
    "PWAT":             {"long_name": "Precipitable water",                             "units": "kg m-2"},
    # Precipitation diagnostics
    "CRAIN":            {"long_name": "Conditional rain rate",                          "units": "kg m-2"},
    "RAIN_MASK":        {"long_name": "Rain occurrence mask",                           "units": "1"},
    "RAIN_FRACTION":    {"long_name": "Rain area fraction",                             "units": "1"},
    "CFRZR":            {"long_name": "Conditional freezing rain rate",                 "units": "kg m-2"},
    "FRZR_MASK":        {"long_name": "Freezing rain occurrence mask",                  "units": "1"},
    "FRZR_FRACTION":    {"long_name": "Freezing rain area fraction",                    "units": "1"},
    "WARM_LAYER_DEPTH": {"long_name": "Warm layer depth above surface",                 "units": "hPa"},
    "COLD_LAYER_DEPTH": {"long_name": "Cold layer depth near surface",                  "units": "hPa"},
    # Wind diagnostics
    "GUST":             {"long_name": "Wind gust speed",                                "units": "m s-1"},
    "GUST_FACTOR":      {"long_name": "Empirical gust factor estimate",                 "units": "m s-1"},
    "GUST_CONV":        {"long_name": "Convective gust estimate",                       "units": "m s-1"},
    "WIND_10M":         {"long_name": "10-meter wind speed",                            "units": "m s-1"},
    "WIND_MAX":         {"long_name": "Maximum wind speed in lower atmospheric column", "units": "m s-1"},
    # Convective diagnostics
    "VUCSH_0_1km":      {"long_name": "U-component wind shear rate 0-1 km AGL",        "units": "s-1"},
    "VVCSH_0_1km":      {"long_name": "V-component wind shear rate 0-1 km AGL",        "units": "s-1"},
    "VUCSH_0_6km":      {"long_name": "U-component wind shear rate 0-6 km AGL",        "units": "s-1"},
    "VVCSH_0_6km":      {"long_name": "V-component wind shear rate 0-6 km AGL",        "units": "s-1"},
    "RELV_max_0_1km":   {"long_name": "Maximum relative vorticity 0-1 km AGL",         "units": "s-1"},
    "RELV_max_0_2km":   {"long_name": "Maximum relative vorticity 0-2 km AGL",         "units": "s-1"},
    "USTM_0_6km":       {"long_name": "U-component of storm motion 0-6 km (Bunkers)",  "units": "m s-1"},
    "VSTM_0_6km":       {"long_name": "V-component of storm motion 0-6 km (Bunkers)",  "units": "m s-1"},
    "HLCY_0_1km":       {"long_name": "Storm-relative helicity 0-1 km AGL",            "units": "m2 s-2"},
    "HLCY_0_3km":       {"long_name": "Storm-relative helicity 0-3 km AGL",            "units": "m2 s-2"},
    # 0 degC isotherm diagnostics
    "HGT_0C":           {"long_name": "Height AGL at the 0 degC isotherm",             "units": "m"},
    "UGRD_0C":          {"long_name": "U-component of wind at 0 degC isotherm",        "units": "m s-1"},
    "VGRD_0C":          {"long_name": "V-component of wind at 0 degC isotherm",        "units": "m s-1"},
    "WIND_0C":          {"long_name": "Wind speed at 0 degC isotherm",                 "units": "m s-1"},
    "SPFH_0C":          {"long_name": "Specific humidity at 0 degC isotherm",          "units": "kg kg-1"},
    "DU_SFC_0C":        {"long_name": "U-component wind shear surface to 0 degC isotherm", "units": "m s-1"},
    "DV_SFC_0C":        {"long_name": "V-component wind shear surface to 0 degC isotherm", "units": "m s-1"},
    "SHEAR_SFC_0C":     {"long_name": "Wind shear magnitude surface to 0 degC isotherm", "units": "m s-1"},
    "RH_0C":            {"long_name": "Relative humidity at 0 degC isotherm",          "units": "%"},
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
            x=("x", x_vals, {
                "standard_name": "projection_x_coordinate",
                "long_name": "x coordinate of projection",
                "units": "m",
                "axis": "X",
            }),
        )
    if "y" in ds.dims and "y" not in ds.coords:
        ny = ds.sizes["y"]
        y_vals = ((np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0) * HRRR_DX_M)
        ds = ds.assign_coords(
            y=("y", y_vals, {
                "standard_name": "projection_y_coordinate",
                "long_name": "y coordinate of projection",
                "units": "m",
                "axis": "Y",
            }),
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
