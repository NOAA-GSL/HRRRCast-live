"""
Meteorological diagnostics computation module.

This module contains functions to compute various weather diagnostics
from model output or observational data.
"""

import numpy as np
import xarray as xr


def get_lcc_grid_params(ds: xr.Dataset, lat_dim: str = "latitude", lon_dim: str = "longitude"):
    """
    Compute Lambert Conformal Conic (LCC) projection grid parameters.

    This helper function calculates the map scale factor, physical grid spacing, 
    rotation angle, and cone constant for an LCC-projected grid (e.g., HRRR).

    Args:
        ds (xr.Dataset): Input dataset containing latitude and longitude coordinates
        lat_dim (str): Name of latitude dimension (default: "latitude")
        lon_dim (str): Name of longitude dimension (default: "longitude")

    Returns:
        tuple: (m_lats, dx, dy, gamma, n) where:
            - m_lats: Map scale factor (latitude-dependent)
            - dx: Physical grid spacing in x direction (m)
            - dy: Physical grid spacing in y direction (m)
            - gamma: Rotation angle between earth and grid coordinates (radians)
            - n: Cone constant (map convergence factor)

    Notes:
        - Assumes HRRR-standard LCC projection (phi0=38.5°N, lon0=97.5°W)
        - Model grid spacing is 3 km in both x and y directions
        - Map scale factor accounts for projection distortion with latitude
    """
    # LCC projection parameters (HRRR standard)
    phi0 = np.deg2rad(38.5)      # Standard latitude (projection center)
    lon0 = -97.5                  # Standard longitude (projection center)
    n = np.sin(phi0)              # Map convergence factor (cone constant)

    # Compute map scale factor for each latitude point using LCC formula:
    # m(phi) = cos(phi0)/cos(phi) * [tan(pi/4 + phi0/2) / tan(pi/4 + phi/2)]^n
    phi = np.deg2rad(ds[lat_dim])
    m_lats = (
        (np.cos(phi0) / np.cos(phi)) *
        (np.tan(np.pi/4 + phi0/2) / np.tan(np.pi/4 + phi/2)) ** n
    )

    # Model grid spacing (3 km in x, 3 km in y for standard HRRR)
    dx_model = 3000.0
    dy_model = 3000.0

    # Physical (true Earth) distance = grid distance / map scale factor
    # m > 1 near standard parallel, < 1 away from it
    dx = dx_model / m_lats
    dy = dy_model / m_lats
    
    # Rotation angle for coordinate transformation
    # Convert longitude from 0-360 to -180-180 convention
    lon_180 = ds[lon_dim].where(ds[lon_dim] <= 180, ds[lon_dim] - 360)
    gamma = n * np.deg2rad(lon_180 - lon0)

    return m_lats, dx, dy, gamma, n


def compute_r2m(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute relative humidity at 2m (R2M) from surface temperature and dew point.

    This function calculates relative humidity at 2 meters height using the Magnus 
    formula for saturation vapor pressure. Relative humidity is a key indicator of 
    atmospheric moisture content and is important for weather forecasting, especially 
    for predicting fog, condensation, and precipitation processes.

    Args:
        ds (xr.Dataset): Input dataset containing the following required fields:
            - T2M: Temperature at 2m height (K)
            - D2M: Dew point temperature at 2m height (K)

    Returns:
        xr.Dataset: Input dataset augmented with the following variable:
            - R2M: Relative humidity at 2m (%, range 0-100)

    Notes:
        - Uses the Magnus formula for saturation vapor pressure calculation
        - Temperatures are assumed to be in Kelvin and are converted to Celsius internally
        - Output is clipped to [0, 100]% to ensure physically valid values
        - The Magnus coefficients (17.67, 243.5, 6.112) are empirically derived constants
    """
    # Convert to Celsius
    T_c = ds["T2M"] - 273.15
    Td_c = ds["D2M"] - 273.15

    # Magnus formula for saturation vapor pressure (hPa)
    es = 6.112 * np.exp((17.67 * T_c) / (T_c + 243.5))
    e = 6.112 * np.exp((17.67 * Td_c) / (Td_c + 243.5))

    # Relative humidity (%)
    RH = 100.0 * (e / es)

    # Clip to [0, 100]
    ds["R2M"] = RH.clip(min=0.0, max=100.0)

    return ds


def compute_spfh2m(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute specific humidity at 2m (SPFH2M) using the hydrostatic approximation.
    
    This function estimates the pressure at 2m height using the hydrostatic equation,
    then calculates specific humidity from the dew point temperature and estimated 
    pressure. Specific humidity is the mass of water vapor per unit mass of moist air 
    and is essential for computing other thermodynamic quantities such as virtual 
    temperature and atmospheric stability.
    
    Args:
        ds (xr.Dataset): Input dataset containing the following required fields:
            - PRES: Surface pressure (Pa)
            - D2M: Dew point temperature at 2m height (K)
            - T2M: Temperature at 2m height (K)
    
    Returns:
        xr.Dataset: Input dataset augmented with the following variable:
            - SPFH2M: Specific humidity at 2m (kg/kg, range 0-1)

    Notes:
        - Pressure at 2m is estimated using the hydrostatic equation: 
          P(z) = P_sfc * exp(-g*z / (Rd*T))
        - Specific gas constant for dry air (Rd) = 287.0 J/(kg·K)
        - Gravitational acceleration (g) = 9.81 m/s²
        - Uses the Magnus formula to compute saturation vapor pressure
        - The drying ratio (epsilon) = 0.622 (Rd/Rv, ratio of gas constants)
        - Output is clipped to [0, 1] to ensure physically valid values
    """
    Rd = 287.0  # Specific gas constant for dry air (J/(kg·K))
    g = 9.81    # Gravitational acceleration (m/s²)
    
    # Extract variables
    P_sfc = ds["PRES"]  # Surface pressure (Pa)
    T2m = ds["T2M"]     # Temperature at 2m (K)
    Td2m = ds["D2M"]    # Dew point at 2m (K)
    
    # Hydrostatic assumption: estimate pressure at 2m
    # P(z) = P_sfc * exp(-g*z / (Rd*T_avg))
    T_avg = T2m  # Use 2m temperature as approximation
    z = 2.0  # Height above ground (meters)
    
    P2m = P_sfc * np.exp(-g * z / (Rd * T_avg))
    
    # Compute saturation vapor pressure at dew point (hPa) using Magnus formula
    Td_c = Td2m - 273.15  # Convert to Celsius
    e = 6.112 * np.exp((17.67 * Td_c) / (Td_c + 243.5))
    
    # Convert to Pa
    e_pa = e * 100.0
    
    # Specific humidity from vapor pressure
    # q = (0.622 * e) / (P - 0.378*e)
    epsilon = 0.622
    q = (epsilon * e_pa) / (P2m - (1.0 - epsilon) * e_pa)
    
    # Clip to valid range [0, 1]
    ds["SPFH2M"] = q.clip(min=0.0, max=1.0)
    
    return ds


def compute_pot2m(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute potential temperature at 2m (POT2M) from surface data.

    Potential temperature is the temperature a parcel of air would have if brought 
    adiabatically to a reference pressure (1000 hPa). It is conserved during dry 
    adiabatic processes and is useful for understanding atmospheric stability and 
    air mass properties.

    Args:
        ds (xr.Dataset): Input dataset containing the following required fields:
            - T2M: Temperature at 2m height (K)
            - PRES: Surface pressure (Pa)

    Returns:
        xr.Dataset: Input dataset augmented with the following variable:
            - POT2M: Potential temperature at 2m (K)

    Notes:
        - Uses the dry adiabatic formula: θ = T * (P_ref / P)^(R/cp)
        - P_ref = 100000 Pa (1000 hPa) is the reference pressure
        - R/cp ≈ 0.2857 (specific gas constant / specific heat at constant pressure)
        - The formula assumes dry air; moisture content is neglected for potential temperature
        - Potential temperature is always greater than or equal to actual temperature 
          (except at reference pressure where they're equal)
    """
    # Constants
    P_ref = 100000.0    # Reference pressure (Pa)
    R_cp = 0.2857       # Ratio of gas constant to specific heat at constant pressure (R/cp = 2/7)
    
    # Extract variables
    T2m = ds["T2M"]     # Temperature at 2m (K)
    P_sfc = ds["PRES"]  # Surface pressure (Pa)
    
    # Compute potential temperature: θ = T * (P_ref / P)^(R/cp)
    pot_temp = T2m * (P_ref / P_sfc) ** R_cp
    
    ds["POT2M"] = pot_temp
    
    return ds


def compute_pwat(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute Precipitable Water (PWAT) - vertically integrated water vapor.

    Precipitable water is the total amount of water vapor integrated from the surface 
    to the top of the atmosphere. It is a critical indicator of atmospheric moisture 
    content and is strongly related to severe weather potential, precipitation amount, 
    and atmospheric instability.

    Args:
        ds (xr.Dataset): Input dataset containing the following required fields:
            - SPFH: Specific humidity at all pressure levels (kg/kg)
            - level: Pressure levels (hPa)

            Dataset must have dimensions: level, latitude, longitude

    Returns:
        xr.Dataset: Input dataset augmented with the following variable:
            - PWAT: Precipitable water (kg/m²)

    Notes:
        - Computed using the hydrostatic formula: PWAT = (1/g) * ∫ q * dp
        - Numerically integrated using mid-layer average specific humidity between
          adjacent pressure levels to reduce discretization error
        - Gravitational acceleration (g) = 9.81 m/s²
        - Output is typically 0-70 kg/m² globally, with higher values in tropics
    """
    g = 9.81  # Gravitational acceleration (m/s²)

    # Extract specific humidity
    q = ds["SPFH"]  # (kg/kg)

    # Pressure levels (convert from hPa to Pa)
    p = xr.DataArray(ds.level.values * 100.0, dims=["level"])

    # Mid-layer average specific humidity between adjacent levels
    q_mid = (q + q.shift(level=1)).isel(level=slice(1, None)) / 2.0

    # Pressure thickness of each layer (Pa, positive)
    dp = np.abs(p.diff("level").astype(np.float32))

    # PWAT = (1/g) * integral(q * dp)
    pwat = ((q_mid * dp).sum("level", skipna=True) / np.float32(g)).astype(np.float32)

    ds["PWAT"] = pwat

    return ds


def compute_conditional_rain(ds: xr.Dataset, precip_threshold: float = 0.254) -> xr.Dataset:
    """
    Compute conditional rain rate - precipitation rate where precipitation occurs.

    This function identifies grid points where precipitation exceeds a threshold
    and computes the mean precipitation rate at those locations. This is useful
    for analyzing precipitation intensity distributions and separating rainy from
    non-rainy conditions.

    Args:
        ds (xr.Dataset): Input dataset containing the following required field:
            - APCP: Accumulated precipitation (mm or kg/m²)
        
        threshold (float, optional): Minimum precipitation threshold in mm to
            consider as "raining". Default is 0.254 mm (0.01 inches).

    Returns:
        xr.Dataset: Input dataset augmented with the following variables:
            - CRAIN: Conditional rain rate (mm where APCP > threshold, 0 elsewhere)
            - RAIN_MASK: Binary mask (1 where raining, 0 elsewhere)
            - RAIN_FRACTION: Fraction of grid points with precipitation > threshold

    Notes:
        - Conditional rain rate is APCP masked to locations where APCP > threshold
        - RAIN_MASK can be used for further conditional statistics
        - RAIN_FRACTION provides spatial coverage of precipitation
        - Default threshold of 0.254 mm (~0.01") is commonly used in meteorology
        - For hourly data, APCP represents the hourly accumulation
    """
    # Extract precipitation
    precip = ds["APCP"]
    
    # Create rain mask (1 where raining, 0 elsewhere)
    rain_mask = (precip > precip_threshold).astype(np.float32)

    # Conditional rain: precipitation where it exceeds threshold, 0 elsewhere
    conditional_rain = precip.where(precip > precip_threshold, 0).astype(np.float32)

    # Compute fraction of raining grid points
    rain_fraction = rain_mask.mean(dim=["latitude", "longitude"])
    
    # Add to dataset
    ds["CRAIN"] = conditional_rain
    ds["RAIN_MASK"] = rain_mask
    ds["RAIN_FRACTION"] = rain_fraction
    
    return ds


def compute_conditional_freezing_rain(ds: xr.Dataset, precip_threshold: float = 0.254) -> xr.Dataset:
    """
    Compute conditional freezing rain - precipitation occurring with freezing conditions.

    Freezing rain occurs when liquid precipitation falls through a shallow subfreezing
    layer near the surface and freezes upon contact. This function identifies conditions
    favorable for freezing rain using temperature profile analysis.

    Classic freezing rain profile:
    1. Surface temperature below freezing (T2M < 0°C)
    2. Precipitation occurring (APCP > threshold)
    3. Warm layer aloft where T > 0°C (melting layer)
    4. Shallow cold layer near surface (refreezing layer)

    Args:
        ds (xr.Dataset): Input dataset containing the following required fields:
            - APCP: Accumulated precipitation (mm or kg/m²)
            - T2M: 2-meter temperature (K)
            - TMP: 3D atmospheric temperature (K) on pressure levels
            - Optional: PRES (surface pressure) for better vertical analysis
        
        precip_threshold (float, optional): Minimum precipitation threshold in mm.
            Default is 0.254 mm (0.01 inches).

    Returns:
        xr.Dataset: Input dataset augmented with the following variables:
            - CFRZR: Conditional freezing rain rate (mm where conditions met, 0 elsewhere)
            - FRZR_MASK: Binary mask (1 where freezing rain conditions, 0 elsewhere)
            - FRZR_FRACTION: Fraction of grid points with freezing rain
            - WARM_LAYER_DEPTH: Depth of warm layer aloft in hPa (where T > 0°C)
            - COLD_LAYER_DEPTH: Depth of cold layer near surface in hPa (where T < 0°C)

    Notes:
        - Simplified detection: checks for surface T < 0°C, precipitation, and warm layer aloft
        - More sophisticated methods may use wet-bulb temperature or explicit melting/refreezing
        - WARM_LAYER_DEPTH = 0 indicates no warm layer (likely snow, not freezing rain)
        - Typical freezing rain: cold surface, 100-400 hPa warm layer, shallow cold layer
        - For hourly data, APCP represents the hourly accumulation
    """
    # Extract variables
    precip = ds["APCP"]
    t2m = ds["T2M"]
    temp_3d = ds["TMP"]  # Temperature on pressure levels
    
    # Temperature threshold (0°C in Kelvin)
    T_freeze = 273.15
    
    # Condition 1: Surface is below freezing
    cold_surface = t2m < T_freeze
    
    # Condition 2: Precipitation is occurring
    precip_mask = precip > precip_threshold
    
    # Condition 3: Check for warm layer aloft (T > 0°C above the surface)
    # Look at temperatures above 925 hPa (roughly 700-900m AGL) to avoid surface layer
    # Typical pressure levels in HRRR: [200, 300, ..., 925, 950, 975, 1000]
    
    # Find levels above 925 hPa (lower pressure values)
    # Select mid-to-upper levels (pressure < 900 hPa) for warm layer detection
    upper_levels = temp_3d.where(temp_3d.level < 900, drop=True)
    
    # Check if any level has T > freezing (indicates melting layer)
    has_warm_layer = (upper_levels > T_freeze).any(dim="level")
    
    # Compute warm layer depth: number of levels or pressure range with T > 0°C
    warm_levels = (upper_levels > T_freeze).sum(dim="level")
    
    # Find lowest and highest levels with T > freezing to get depth
    warm_mask_3d = (temp_3d > T_freeze)
    # Get pressure coordinates where warm
    if "level" in temp_3d.coords:
        level_vals = temp_3d.level
        # For each horizontal point, find the pressure range of warm layer
        # Simplified: count levels above freezing at upper levels
        warm_layer_depth = warm_levels * 50.0  # Approximate: 50 hPa per level spacing
    else:
        warm_layer_depth = warm_levels
    
    # Find cold layer depth near surface (levels below 925 hPa with T < 0°C)
    lower_levels = temp_3d.where(temp_3d.level >= 925, drop=True)
    cold_levels = (lower_levels < T_freeze).sum(dim="level")
    cold_layer_depth = cold_levels * 25.0  # Finer spacing near surface (~25 hPa)
    
    # Combined freezing rain mask: cold surface + precipitation + warm layer aloft
    frzr_mask = (cold_surface & precip_mask & has_warm_layer).astype(np.float32)
    
    # Conditional freezing rain: precipitation where conditions are met, 0 elsewhere
    conditional_frzr = precip.where(frzr_mask > 0, 0).astype(np.float32)
    
    # Compute fraction of freezing rain
    frzr_fraction = frzr_mask.mean(dim=["latitude", "longitude"])
    
    # Add to dataset
    ds["CFRZR"] = conditional_frzr
    ds["FRZR_MASK"] = frzr_mask
    ds["FRZR_FRACTION"] = frzr_fraction
    ds["WARM_LAYER_DEPTH"] = warm_layer_depth.fillna(0).astype(np.float32)
    ds["COLD_LAYER_DEPTH"] = cold_layer_depth.fillna(0).astype(np.float32)
    
    return ds


def compute_wind_gust(ds: xr.Dataset, gust_factor: float = 1.4) -> xr.Dataset:
    """
    Compute wind gust estimates from wind components and atmospheric variables.

    Wind gusts are brief (typically 3-5 second) increases in wind speed above the
    sustained wind. This function estimates wind gust speed using multiple methods:
    1. Gust factor method: applies empirical factor to mean wind speed
    2. Convective gust: enhanced by vertical velocity and boundary layer mixing
    3. Maximum wind in column: useful for elevated wind maxima (e.g., low-level jets)

    Args:
        ds (xr.Dataset): Input dataset containing the following required fields:
            - UGRD: U-component of wind (m/s) on pressure levels
            - VGRD: V-component of wind (m/s) on pressure levels
            - UGRD10M: U-component of 10-meter wind (m/s)
            - VGRD10M: V-component of 10-meter wind (m/s)
            
            Optional fields for enhanced gust estimation:
            - VVEL: Vertical velocity (Pa/s) for convective gusts
            - TMP: Temperature (K) for stability-based corrections
            - PRES: Surface pressure (Pa) for boundary layer analysis
        
        gust_factor (float, optional): Empirical gust factor applied to mean wind.
            Default is 1.4 (40% increase over sustained wind). Typical range: 1.3-1.6
            Higher values (~1.5-1.6) for unstable/convective conditions
            Lower values (~1.3) for stable conditions

    Returns:
        xr.Dataset: Input dataset augmented with the following variables:
            - GUST: Estimated wind gust speed (m/s) - maximum of all methods
            - GUST_FACTOR: Gust using empirical factor method (m/s)
            - GUST_CONV: Convective gust enhancement (m/s)
            - WIND_10M: 10-meter wind speed (m/s)
            - WIND_MAX: Maximum wind speed in atmospheric column (m/s)

    Notes:
        - GUST is the maximum of gust_factor method and column maximum
        - Convective enhancement added when strong updrafts present (VVEL < -1 Pa/s)
        - Typical observed gust factors: 1.3-1.4 (stable), 1.4-1.5 (neutral), 1.5-1.6 (unstable)
        - Function prioritizes surface-based gusts but captures elevated wind maxima
        - For severe convection, GUST_CONV can add 5-15 m/s to baseline gust estimate
    """
    # Extract 10-meter wind components
    u10 = ds["UGRD10M"]
    v10 = ds["VGRD10M"]
    
    # Compute 10-meter wind speed
    wind_10m = np.sqrt(u10**2 + v10**2).astype(np.float32)
    
    # Method 1: Gust factor method (empirical)
    gust_empirical = (np.float32(gust_factor) * wind_10m).astype(np.float32)
    
    # Method 2: Maximum wind speed in the column
    # Useful for elevated wind maxima (e.g., low-level jet)
    if "level" in ds["UGRD"].dims:
        u3d = ds["UGRD"]
        v3d = ds["VGRD"]
        wind_3d = np.sqrt(u3d**2 + v3d**2).astype(np.float32)
        
        # Find maximum wind in the column (typically in lowest 3-4 km)
        # Focus on lower atmosphere (pressure > 700 hPa) where gusts are most relevant
        wind_lower = wind_3d.where(wind_3d.level >= 700, drop=True)
        wind_max_column = wind_lower.max(dim="level", skipna=True).astype(np.float32)
    else:
        wind_max_column = wind_10m
    
    # Method 3: Convective gust enhancement
    # Strong updrafts can bring higher-momentum air to surface
    if "VVEL" in ds.data_vars and "level" in ds["VVEL"].dims:
        vvel = ds["VVEL"]
        
        # Find strong updraft regions (negative omega = upward motion)
        # Focus on mid-low levels (700-850 hPa) where momentum transport is effective
        vvel_lower = vvel.where((vvel.level >= 700) & (vvel.level <= 850), drop=True)
        max_updraft = vvel_lower.min(dim="level", skipna=True)  # Most negative = strongest updraft
        
        # Enhanced gust when updraft strength exceeds -1 Pa/s
        # Empirical: 0.5 m/s per -1 Pa/s of updraft strength
        updraft_enhancement = (np.maximum(np.float32(0), -max_updraft - np.float32(1.0)) * np.float32(0.5)).astype(np.float32)
        gust_convective = (gust_empirical + updraft_enhancement).astype(np.float32)
    else:
        gust_convective = gust_empirical
        updraft_enhancement = xr.zeros_like(wind_10m)
    
    # Final gust estimate: maximum of empirical factor and column maximum
    # Add convective enhancement when applicable
    gust_estimate = np.maximum(gust_empirical, wind_max_column)
    gust_final = np.maximum(gust_estimate, gust_convective).astype(np.float32)

    # Fill NaN values with 0 (no gust where data is missing)
    gust_final = gust_final.fillna(0.0).astype(np.float32)
    gust_empirical = gust_empirical.fillna(0.0).astype(np.float32)
    gust_convective = gust_convective.fillna(0.0).astype(np.float32)
    wind_10m = wind_10m.fillna(0.0).astype(np.float32)
    wind_max_column = wind_max_column.fillna(0.0).astype(np.float32)
    
    # Add to dataset
    ds["GUST"] = gust_final
    ds["GUST_FACTOR"] = gust_empirical
    ds["GUST_CONV"] = gust_convective
    ds["WIND_10M"] = wind_10m
    ds["WIND_MAX"] = wind_max_column
    
    return ds


def compute_vvel(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute vertical velocity (VVEL) in pressure coordinates (omega, Pa/s) from continuity equation.

    This function estimates vertical velocity using the continuity equation in pressure 
    coordinates: ∂u/∂x + ∂v/∂y + ∂ω/∂p = 0. The vertical velocity is computed by:
    1. Computing horizontal wind divergence accounting for Lambert Conformal Conic (LCC) projection
    2. Removing domain-mean divergence at each level to correct for AI model mass conservation errors
    3. Extrapolating divergence to 20 hPa and computing ω at the top data level (200 hPa)
       using ω=0 at 20 hPa as the upper boundary condition
    4. Integrating both upward from surface (ω=0) and downward from top (ω=omega_at_top)
    5. Blending the two solutions with logarithmic pressure-based weights

    The surface boundary condition is physically exact (air cannot flow through Earth's surface).
    The upper boundary condition assumes ω=0 at 20 hPa (~99% of atmospheric mass below),
    extrapolated from the top two data levels. Blending the two integrations ensures errors
    accumulate only to the midpoint from either boundary rather than across the full column.

    Args:
        ds (xr.Dataset): Input dataset containing the following required fields:
            - UGRD: U-component of wind at all pressure levels (m/s)
            - VGRD: V-component of wind at all pressure levels (m/s)
            - level: Pressure levels (hPa), must be sorted ascending (e.g. 200→1000 hPa)
            - latitude/lat: Latitude coordinate
            - longitude/lon: Longitude coordinate

            Dataset must have dimensions: (time, lead_time, level, latitude, longitude)
            Assumes Lambert Conformal Conic projection (e.g., HRRR grid)

    Returns:
        xr.Dataset: Input dataset augmented with the following variable:
            - VVEL: Vertical velocity (Pa/s). Positive values indicate downward motion,
                    negative values indicate upward motion.

    Notes:
        - Accounts for Lambert Conformal Conic projection distortion using map scale factor
        - Wind components are rotated to align with model grid (x, y) coordinates
        - Domain-mean divergence is removed at each level to suppress AI model mass conservation drift
        - Upper BC: ω=0 at 20 hPa, extrapolated from top two data levels
        - Lower BC: ω=0 at surface (exact physical constraint)
        - Uses trapezoidal rule for vertical integration
        - Blending weights are linear in pressure: near surface trusts bottom-up,
          near top trusts top-down, equal blend at midpoint
    """
    u = ds["UGRD"]
    v = ds["VGRD"]

    # ================================================================================
    # Get LCC projection parameters and compute divergence in geographic coordinates
    # ================================================================================
    m_lats, dx, dy, gamma, n = get_lcc_grid_params(ds)

    # Compute divergence directly in geographic coordinates (u, v)
    du_dx = (u.shift(longitude=-1) - u.shift(longitude=1)) / (2 * dx)
    dv_dy = (v.shift(latitude=-1)  - v.shift(latitude=1))  / (2 * dy)
    divergence = (du_dx + dv_dy)

    # Fill boundary NaNs (from finite difference stencil edges) with 0
    divergence = divergence.fillna(0.0).astype(np.float32)

    # ==========================================================================
    # Setup: pressure levels and integer coordinate replacement
    # ==========================================================================
    p_pa = ds.level.values * 100.0  # hPa → Pa, ascending order (low→high pressure)
    nlev = len(p_pa)

    assert (np.diff(p_pa) > 0).all(), "Pressure levels must be sorted ascending (e.g. 200→1000 hPa)"

    # Replace pressure coords with plain integers to prevent xarray alignment conflicts
    int_levels = np.arange(nlev)
    divergence = divergence.assign_coords(level=int_levels)

    # dp between adjacent data levels, positive (ascending pressure = downward)
    dp = np.diff(p_pa).astype(np.float32)  # shape: (nlev-1,)
    dp_da = xr.DataArray(dp, dims=["level"], coords={"level": int_levels[:-1]})

    # Trapezoidal average divergence between adjacent levels
    div_upper = divergence.isel(level=slice(None, -1))                              # coords 0..N-2
    div_lower = divergence.isel(level=slice(1, None)).assign_coords(level=int_levels[:-1])  # coords 0..N-2
    div_mid = 0.5 * (div_upper + div_lower)

    # Layer increments for trapezoidal integration: D_avg * dp
    increments = (div_mid * dp_da).assign_coords(level=int_levels[:-1])  # shape: (nlev-1, ...)

    # ==========================================================================
    # Upper boundary condition: extrapolate divergence to 20 hPa, compute
    # ω at top data level (200 hPa) by integrating down from ω=0 at 20 hPa
    # ==========================================================================
    p_assumed_top = 20.0 * 100.0  # 20 hPa in Pa

    div_lev0 = divergence.isel(level=0)                                    # divergence at 200 hPa
    div_lev1 = divergence.isel(level=1).assign_coords(level=0)             # divergence at next level

    # Linear extrapolation of divergence gradient to 50 hPa
    dD_dp = (div_lev1 - div_lev0) / (p_pa[1] - p_pa[0])
    div_ext = div_lev0 + dD_dp * (p_assumed_top - p_pa[0])                 # div at 50 hPa

    # One trapezoidal step from 20 hPa down to first data level (200 hPa)
    # ω=0 at 20 hPa, so: ω(200hPa) = 0 - 0.5*(D_20 + D_200) * (p_200 - p_20)
    dp_ext = np.float32(p_pa[0] - p_assumed_top)                           # positive, ~18000 Pa  (200-20 hPa = 180 hPa
    omega_at_top = -(0.5 * (div_ext + div_lev0) * dp_ext)                  # ω at 200 hPa

    # ==========================================================================
    # Bottom-up integration: ω=0 at surface, integrate upward
    # Flip to surface-first order, cumsum, flip back
    # ==========================================================================
    inc_flipped = increments.isel(level=slice(None, None, -1)).assign_coords(level=int_levels[:-1])
    omega_bot_flipped = inc_flipped.cumsum(dim="level")
    omega_bot_interfaces = omega_bot_flipped.isel(level=slice(None, None, -1)).assign_coords(level=int_levels[:-1])

    # Append surface BC: ω=0 at index nlev-1
    zero_sfc = xr.zeros_like(divergence.isel(level=-1)).expand_dims(dim={"level": [nlev - 1]}, axis=-3)
    omega_bot = xr.concat([omega_bot_interfaces, zero_sfc], dim="level").assign_coords(level=int_levels)

    # ==========================================================================
    # Top-down integration: ω=omega_at_top at level 0, integrate downward
    # cumsum in natural order (index 0=top → index nlev-1=surface)
    # ==========================================================================
    # Top-down cumsum gives ω at interfaces between levels (nlev-1 values)
    omega_top_interfaces = (omega_at_top - increments.cumsum(dim="level")).assign_coords(level=int_levels[1:])

    # Prepend top BC: ω=omega_at_top at index 0
    omega_at_top_expanded = omega_at_top.expand_dims(dim={"level": [0]}, axis=-3)
    omega_top = xr.concat([omega_at_top_expanded, omega_top_interfaces], dim="level").assign_coords(level=int_levels)

    # ==========================================================================
    # Blend: logarithmic weights based on pressure distance from each boundary
    # w_bot: 1 at surface, 0 at top  → trust bottom-up near surface
    # w_top: 0 at surface, 1 at top  → trust top-down near top
    # ==========================================================================
    log_p = np.log(p_pa)
    log_p_sfc = np.log(p_pa[-1])
    log_p_top = np.log(p_pa[0])

    w_bot = (log_p - log_p_top) / (log_p_sfc - log_p_top)  # 0 at top, 1 at surface
    w_top = 1.0 - w_bot

    w_bot_da = xr.DataArray(w_bot.astype(np.float32), dims=["level"], coords={"level": int_levels})
    w_top_da = xr.DataArray(w_top.astype(np.float32), dims=["level"], coords={"level": int_levels})

    vvel = w_bot_da * omega_bot + w_top_da * omega_top

    # Restore original pressure level coordinates
    vvel = vvel.assign_coords(level=ds.level)

    ds["VVEL"] = vvel.astype(np.float32)
    return ds


def compute_convective(ds):
    """
    This function calculates various convective parameters used in severe weather 
    forecasting from atmospheric model output, including shear rates, helicity, 
    vorticity, storm motion, and updraft/downdraft velocities.

    Args:
        ds (xr.Dataset): Input dataset containing HRRR variables with the following 
            required fields:
            - UGRD: U-component of wind (m/s)
            - VGRD: V-component of wind (m/s)
            - VVEL: Vertical velocity (omega, Pa/s)
            - TMP: Temperature (K)
            - SPFH: Specific humidity (kg/kg)
            - HGT: Geopotential height (m)
            - OROG: Orography/surface elevation (m)

            Dataset must have dimensions: level, latitude, longitude

    Returns:
        xr.Dataset: Input dataset augmented with the following convective diagnostics:
            - VUCSH_0_1km: U-component wind shear rate 0-1 km AGL (1/s)
            - VVCSH_0_1km: V-component wind shear rate 0-1 km AGL (1/s)
            - VUCSH_0_6km: U-component wind shear rate 0-6 km AGL (1/s)
            - VVCSH_0_6km: V-component wind shear rate 0-6 km AGL (1/s)
            - RELV_max_0_1km: Maximum relative vorticity 0-1 km AGL (1/s)
            - RELV_max_0_2km: Maximum relative vorticity 0-2 km AGL (1/s)
            - USTM_0_6km: U-component of storm motion (Bunkers right-moving, m/s)
            - VSTM_0_6km: V-component of storm motion (Bunkers right-moving, m/s)
            - HLCY_0_1km: Storm-relative helicity 0-1 km AGL (m²/s²)
            - HLCY_0_3km: Storm-relative helicity 0-3 km AGL (m²/s²)
            - MXUPHL_max_0_2km: Maximum updraft helicity 0-2 km AGL (m²/s²)
            - MNUPHL_min_0_2km: Minimum updraft helicity 0-2 km AGL (m²/s²)
            - MXUPHL_max_0_3km: Maximum updraft helicity 0-3 km AGL (m²/s²)
            - MNUPHL_min_0_3km: Minimum updraft helicity 0-3 km AGL (m²/s²)
            - MXUPHL_max_2_5km: Maximum updraft helicity 2-5 km AGL (m²/s²)
            - MNUPHL_min_2_5km: Minimum updraft helicity 2-5 km AGL (m²/s²)
            - MAXUVV_max_100_1000mb: Maximum upward vertical velocity 100-1000 mb (m/s)
            - MAXDVV_max_100_1000mb: Maximum downward vertical velocity 100-1000 mb (m/s)

    Notes:
        - Uses Lambert Conformal projection centered at 38.5°N, 97.5°W
        - Grid spacing assumed to be 6 km in both x and y directions
        - Tv is the virtual temperature, calculated as T * (1 + 0.61*q), which accounts 
          for the effect of moisture on air density
        - AGL = Above Ground Level
        - Storm motion uses Bunkers right-moving supercell method (mean wind 0-6 km + 
          7.5 m/s deviation perpendicular to shear vector)
        - WARNING: At 6km grid spacing, vorticity/shear calculations using 2-point 
          finite differences can be noisy. Consider smoothing if needed.
    """

    Rd = 287.0
    g  = 9.81

    u = ds["UGRD"]
    v = ds["VGRD"]
    omega = ds["VVEL"]
    T = ds["TMP"]
    q = ds["SPFH"]
    z = ds["HGT"]
    orog = ds["OROG"]

    # =====================================================
    # 1. Height above ground level (AGL)
    # =====================================================
    z_agl = z - orog
    mask1 = (z_agl>=0) & (z_agl<=1000)
    mask2 = (z_agl>=0) & (z_agl<=2000)
    mask6 = (z_agl>=0) & (z_agl<=6000)
    dz = -z_agl.diff("level")

    # fudge factor for maximum relative vorticity and max updraft helicity
    # to account for max in a 1h window instead of instantaneous value at each lead time
    FUDGE_FACTOR_VORTICITY = 6

    # =====================================================
    # 2. Convert omega -> w
    # =====================================================
    p = xr.DataArray(ds.level.values * 100.0, dims=["level"])

    Tv = T * (1 + 0.61*q)
    w = -omega * (Rd*Tv)/(g*p)

    # =====================================================
    # 3. Horizontal derivatives and vorticity
    # =====================================================
    # Get LCC projection parameters (handles map scale factor, grid spacing, rotation)
    m_lats, dx, dy, gamma, n = get_lcc_grid_params(ds)

    # Compute vorticity directly in geographic coordinates (u, v)
    dv_dx = (v.shift(longitude=-1) - v.shift(longitude=1)) / (2*dx)
    du_dy = (u.shift(latitude=-1) - u.shift(latitude=1)) / (2*dy)

    # Vorticity in geographic frame: zeta = dv/dx - du/dy
    zeta = dv_dx - du_dy

    # =================================================
    # Shear rate (1/s)
    # =================================================
    def shear_rate(top):

        u_bot = u.isel(level=-1)
        v_bot = v.isel(level=-1)

        # Select the level closest to (but not exceeding) the top height
        z_agl_above = z_agl.where(z_agl <= top)
        level_top_idx = z_agl_above.argmax(dim="level").compute()
        u_top = u.isel(level=level_top_idx)
        v_top = v.isel(level=level_top_idx)

        du = u_top - u_bot
        dv = v_top - v_bot

        depth = top  # meters

        return du / depth, dv / depth

    du1, dv1 = shear_rate(1000)
    du6, dv6 = shear_rate(6000)

    # HRRR variable names
    ds["VUCSH_0_1km"] = du1      # U shear rate
    ds["VVCSH_0_1km"] = dv1      # V shear rate

    ds["VUCSH_0_6km"] = du6
    ds["VVCSH_0_6km"] = dv6

    # =====================================================
    # 5. Relative vorticity maximum in lowest 1 km and 2 km AGL
    # =====================================================
    RELV_max = zeta * FUDGE_FACTOR_VORTICITY
    ds["RELV_max_0_1km"] = RELV_max.where(mask1).max("level", skipna=True).fillna(0).astype(np.float32)
    ds["RELV_max_0_2km"] = RELV_max.where(mask2).max("level", skipna=True).fillna(0).astype(np.float32)

    # =====================================================
    # 6. Storm motion (Bunkers)
    # =====================================================
    u06, v06 = du6 * 6000, dv6 * 6000

    shear_mag = np.sqrt(u06**2 + v06**2) + 1e-6

    right_u =  v06 / shear_mag
    right_v = -u06 / shear_mag

    mean_u = u.where(mask6).mean("level", skipna=True)
    mean_v = v.where(mask6).mean("level", skipna=True)

    cu = mean_u + 7.5*right_u
    cv = mean_v + 7.5*right_v

    ds["USTM_0_6km"] = cu
    ds["VSTM_0_6km"] = cv

    # =====================================================
    # 7. HLCY 0–1 km
    # =====================================================
    du_dz = u.diff("level") / dz
    dv_dz = v.diff("level") / dz

    # Mid-layer winds to match the N-1 staggered grid of the derivatives
    u_mid  = (u  + u.shift(level=1)).isel(level=slice(1, None)) / 2.0
    v_mid  = (v  + v.shift(level=1)).isel(level=slice(1, None)) / 2.0

    # Mid-layer masks
    z_agl_mid = (z_agl + z_agl.shift(level=1)).isel(level=slice(1, None)) / 2.0
    mask1_mid = (z_agl_mid >= 0) & (z_agl_mid <= 1000)
    mask3_mid = (z_agl_mid >= 0) & (z_agl_mid <= 3000)

    hlcy = ((u_mid - cu) * dv_dz - (v_mid - cv) * du_dz) * dz
    ds["HLCY_0_1km"] = hlcy.where(mask1_mid).sum("level", skipna=True).astype(np.float32)
    ds["HLCY_0_3km"] = hlcy.where(mask3_mid).sum("level", skipna=True).astype(np.float32)

    # =====================================================
    # 8. Updraft helicity 0-2 km, 0-3 km, 2–5 km
    # =====================================================
    w_mid    = (w    + w.shift(level=1)   ).isel(level=slice(1, None)) / 2.0
    zeta_mid = (zeta + zeta.shift(level=1)).isel(level=slice(1, None)) / 2.0

    mask2_mid   = (z_agl_mid >= 0)    & (z_agl_mid <= 2000)
    mask3_mid   = (z_agl_mid >= 0)    & (z_agl_mid <= 3000)
    mask2_5_mid = (z_agl_mid >= 2000) & (z_agl_mid <= 5000)

    uh_inst = w_mid * zeta_mid * dz
    uh_inst = uh_inst * FUDGE_FACTOR_VORTICITY

    # 0-2 km updraft helicity
    uh_inst_0_2 = uh_inst.where(mask2_mid)
    ds["MXUPHL_max_0_2km"] = uh_inst_0_2.clip(min=0).sum("level", skipna=True).fillna(0).astype(np.float32)
    ds["MNUPHL_min_0_2km"] = uh_inst_0_2.clip(max=0).sum("level", skipna=True).fillna(0).astype(np.float32)

    # 0-3 km updraft helicity
    uh_inst_0_3 = uh_inst.where(mask3_mid)
    ds["MXUPHL_max_0_3km"] = uh_inst_0_3.clip(min=0).sum("level", skipna=True).fillna(0).astype(np.float32)
    ds["MNUPHL_min_0_3km"] = uh_inst_0_3.clip(max=0).sum("level", skipna=True).fillna(0).astype(np.float32)

    # 2-5 km updraft helicity
    uh_inst_2_5 = uh_inst.where(mask2_5_mid)
    ds["MXUPHL_max_2_5km"] = uh_inst_2_5.clip(min=0).sum("level", skipna=True).fillna(0).astype(np.float32)
    ds["MNUPHL_min_2_5km"] = uh_inst_2_5.clip(max=0).sum("level", skipna=True).fillna(0).astype(np.float32)

    # =============================================================
    # 9. Maximum updraft velocity between 100-1000 mb
    # =============================================================
    level_hpa = w["level"]
    mask_v = (level_hpa >= 100) & (level_hpa <= 1000)
    w_layer = w.where(mask_v)

    # upward/downward vertical velocity maxima/minima in 100-1000 mb layer
    ds["MAXUVV_max_100_1000mb"] = w_layer.clip(min=0).max("level", skipna=True).fillna(0).astype(np.float32)
    ds["MAXDVV_max_100_1000mb"] = w_layer.clip(max=0).min("level", skipna=True).fillna(0).astype(np.float32)

    return ds


def compute_0C_isotherm(ds):
    """
    Compute wind, humidity, and height variables at the 0°C isotherm level from 
    atmospheric model output.

    This function identifies the pressure level closest to 0°C for each grid point and 
    time, then calculates meteorologically relevant diagnostics at that level. The 0°C 
    isotherm (freezing level) is critical for precipitation type forecasting and icing 
    hazard assessment.

    Args:
        ds (xr.Dataset): Input dataset containing the following required fields:
            - UGRD: U-component of wind (m/s)
            - VGRD: V-component of wind (m/s)
            - TMP: Temperature (K)
            - HGT: Geopotential height (m)
            - SPFH: Specific humidity (kg/kg)
            - OROG: Orography/surface elevation (m)

            Dataset must have dimensions: level, latitude, longitude

    Returns:
        xr.Dataset: Input dataset augmented with the following 0°C isotherm diagnostics:
            - HGT_0C: Height above ground level (AGL) at the 0°C isotherm (m)
            - UGRD_0C: U-component of wind at the 0°C isotherm (m/s)
            - VGRD_0C: V-component of wind at the 0°C isotherm (m/s)
            - WIND_0C: Wind speed at the 0°C isotherm (m/s)
            - SPFH_0C: Specific humidity at the 0°C isotherm (kg/kg)
            - DU_SFC_0C: U-component wind shear from surface to 0°C isotherm (m/s)
            - DV_SFC_0C: V-component wind shear from surface to 0°C isotherm (m/s)
            - SHEAR_SFC_0C: Wind shear magnitude from surface to 0°C isotherm (m/s)
            - RH_0C: Relative humidity at the 0°C isotherm (%)

    Notes:
        - The 0°C isotherm level is determined by finding the pressure level with 
          temperature closest to 0°C
        - Surface wind is derived from the lowest model level
        - Relative humidity is computed using the Magnus formula for saturation vapor 
          pressure and the saturation vapor pressure at 0°C (6.112 hPa)
        - Relative humidity is clipped to [0, 100]%
    """
    
    # Extract variables
    u = ds["UGRD"]
    v = ds["VGRD"]
    T = ds["TMP"]
    q = ds["SPFH"]
    z = ds["HGT"]
    orog = ds["OROG"]

    # =====================================================
    # 1. Height above ground level (AGL)
    # =====================================================
    z_agl = z - orog

    # =====================================================
    # 2. Find the 0°C isotherm level
    # =====================================================
    # Convert temperature to Celsius
    T_c = T - 273.15
    
    # Find the level closest to 0°C for each grid point and time 
    # Find where T_c is closest to 0
    T_c_abs = np.abs(T_c)
    level_0C_idx = T_c_abs.argmin(dim="level").compute()
    
    # Get the height at the 0°C level
    h_0C = z_agl.isel(level=level_0C_idx)
    ds["HGT_0C"] = h_0C.astype(np.float32)

    # =====================================================
    # 3. Interpolate variables to 0°C level
    # =====================================================
    
    # Get values at the 0°C level
    u_0C = u.isel(level=level_0C_idx)
    v_0C = v.isel(level=level_0C_idx)
    ds["UGRD_0C"] = u_0C
    ds["VGRD_0C"] = v_0C

    # Wind speed at 0°C level
    wind_speed_0C = np.sqrt(u_0C**2 + v_0C**2)
    ds["WIND_0C"] = wind_speed_0C

    # Specific humidity at 0°C level
    q_0C = q.isel(level=level_0C_idx)
    ds["SPFH_0C"] = q_0C

    # =====================================================
    # 4. Wind shear relative to surface
    # =====================================================
    # Get wind at the lowest level (surface)
    u_sfc = u.isel(level=-1)
    v_sfc = v.isel(level=-1)

    du_0C = u_0C - u_sfc
    dv_0C = v_0C - v_sfc
    ds["DU_SFC_0C"] = du_0C
    ds["DV_SFC_0C"] = dv_0C

    shear_mag_0C = np.sqrt(du_0C**2 + dv_0C**2)
    ds["SHEAR_SFC_0C"] = shear_mag_0C

    # =====================================================
    # 5. Relative humidity at 0°C level
    # =====================================================
    # Saturation vapor pressure at 0°C (hPa)
    es_0C = 6.112 * np.exp((17.67 * 0) / (0 + 243.5))
    
    # Pressure at 0°C level from level values
    p_levels = xr.DataArray(ds.level.values, dims=["level"])  # In hPa
    p_0C = p_levels.isel(level=level_0C_idx)
    
    # Vapor pressure at 0°C from specific humidity
    # q = (epsilon * e) / (p - (1-epsilon)*e), solve for e
    epsilon = 0.622
    e_0C = (q_0C * p_0C) / (epsilon + (1.0 - epsilon) * q_0C)
    
    # Relative humidity at 0°C
    rh_0C = 100.0 * (e_0C / es_0C)
    ds["RH_0C"] = rh_0C.clip(min=0.0, max=100.0).astype(np.float32)

    return ds


def compute_diagnostics(
    ds: xr.Dataset,
    include: list = None,
    exclude: list = None,
    skip_errors: bool = True,
    **kwargs
) -> xr.Dataset:
    """
    Compute all available meteorological diagnostics for HRRR model data.

    This is a master function that calls all individual diagnostic computation
    functions in a logical order. It provides flexible control over which
    diagnostics to compute and how to handle errors.

    Args:
        ds (xr.Dataset): Input dataset containing HRRR variables. Required variables
            depend on which diagnostics are computed. See individual function
            documentation for specific requirements.
        
        include (list, optional): List of diagnostic function names to compute.
            If None (default), all diagnostics are computed.
            Available: ['r2m', 'spfh2m', 'pot2m', 'pwat', 'conditional_rain',
                       'conditional_freezing_rain', 'wind_gust', 'convective',
                       '0C_isotherm']
        
        exclude (list, optional): List of diagnostic function names to skip.
            Takes precedence over 'include'. Default is None.
        
        skip_errors (bool, optional): If True, skip diagnostics that fail due to
            missing variables or errors and continue with others. If False, raise
            exceptions on errors. Default is True.
        
        **kwargs: Additional keyword arguments passed to specific diagnostic functions.
            Supported:
            - precip_threshold (float): For conditional_rain and conditional_freezing_rain
            - gust_factor (float): For wind_gust
            - Any other parameters for individual functions

    Returns:
        xr.Dataset: Input dataset augmented with all computed diagnostic variables.

    Notes:
        - Diagnostics are computed in the following order to minimize dependencies:
          1. Basic surface diagnostics: R2M, SPFH2M, POT2M
          2. Column-integrated: PWAT
          3. Precipitation diagnostics: CRAIN, CFRZR
          4. Wind diagnostics: GUST
          5. Complex convective diagnostics: convective parameters
          6. Vertical profile diagnostics: 0°C isotherm
        
        - If skip_errors=True, warnings are printed for failed diagnostics
        - Each diagnostic adds multiple variables - see individual function docs
        - Typical usage: compute_diagnostics(ds) to compute everything
        - Selective usage: compute_diagnostics(ds, include=['r2m', 'wind_gust'])

    Examples:
        >>> # Compute all diagnostics
        >>> ds_diag = compute_diagnostics(ds)
        
        >>> # Compute only surface and precipitation diagnostics
        >>> ds_diag = compute_diagnostics(ds, include=['r2m', 'spfh2m', 'conditional_rain'])
        
        >>> # Compute all except convective (which is computationally expensive)
        >>> ds_diag = compute_diagnostics(ds, exclude=['convective'])
        
        >>> # Compute with custom parameters
        >>> ds_diag = compute_diagnostics(ds, precip_threshold=0.5, gust_factor=1.5)
    """
    # Define all available diagnostics in execution order
    all_diagnostics = [
        ('r2m', compute_r2m),
        ('spfh2m', compute_spfh2m),
        ('pot2m', compute_pot2m),
        ('pwat', compute_pwat),
        ('vvel', compute_vvel),
        ('conditional_rain', compute_conditional_rain),
        ('conditional_freezing_rain', compute_conditional_freezing_rain),
        ('wind_gust', compute_wind_gust),
        ('convective', compute_convective),
        ('0C_isotherm', compute_0C_isotherm),
    ]
    
    # Determine which diagnostics to compute
    if include is not None:
        # Only compute specified diagnostics
        diagnostics_to_compute = [(name, func) for name, func in all_diagnostics if name in include]
    else:
        # Compute all diagnostics
        diagnostics_to_compute = all_diagnostics
    
    # Apply exclusions
    if exclude is not None:
        diagnostics_to_compute = [(name, func) for name, func in diagnostics_to_compute if name not in exclude]
    
    # Extract parameters for specific functions
    precip_threshold = kwargs.get('precip_threshold', 0.254)
    gust_factor = kwargs.get('gust_factor', 1.4)
    
    # Compute each diagnostic
    for name, func in diagnostics_to_compute:
        try:
            # Call function with appropriate parameters
            if name in ['conditional_rain', 'conditional_freezing_rain']:
                ds = func(ds, precip_threshold=precip_threshold)
            elif name == 'wind_gust':
                ds = func(ds, gust_factor=gust_factor)
            else:
                ds = func(ds)
            
            if not skip_errors:
                # Optionally print success message
                pass
        except Exception as e:
            if skip_errors:
                print(f"Warning: Failed to compute {name}: {str(e)}")
            else:
                raise
    
    return ds
