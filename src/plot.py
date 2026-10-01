#!/usr/bin/env python3
"""
Forecast Visualization Script

This script plots each variable from the forecast output and saves them as separate PNG files.
It handles both pressure level and surface variables from the HRRR forecast data.

Usage:
        python plot_forecast.py <init_time> <lead_hour> <member> [--forecast_dir DIR] [--output_dir DIR]
    
        Expects per-hour NetCDF files:
            - Member average (PMM/mean): hrrrcast_avg_fXX.nc
            - Ensemble spread:           hrrrcast_spr_fXX.nc
            - Ensemble probabilities:    hrrrcast_prob_fXX.nc
            - Individual members:        hrrrcast_mN_fXX.nc
"""

import argparse
import logging
import os
import sys
from datetime import timedelta
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import numpy as np
import xarray as xr
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

# Local imports
import utils
from utils import setup_logging
from cf_attributes import VARIABLE_METADATA

logger = None


PRODUCT_ALIASES = {
    "pmm": "avg",
    "mean": "avg",
    "avg": "avg",
    "spr": "spr",
    "spread": "spr",
    "std": "spr",
    "prob": "prob",
    "probs": "prob",
    "probability": "prob",
    "probabilities": "prob",
}


def _normalize_range(range_values: Optional[List[float]], range_name: str) -> Optional[tuple]:
    """Validate and normalize a two-value [min, max] numeric range."""
    if range_values is None:
        return None
    if len(range_values) != 2:
        raise ValueError(f"{range_name} must contain exactly 2 values: min max")
    low, high = float(range_values[0]), float(range_values[1])
    if low == high:
        raise ValueError(f"{range_name} min and max cannot be equal")
    return (min(low, high), max(low, high))


class ForecastPlotterConfig:
    """Configuration class for forecast plotting parameters."""
    
    def __init__(self):
        # Variable definitions matching the preprocessor
        self.pl_vars = ["UGRD", "VGRD", "VVEL", "TMP", "HGT", "SPFH"]
        # Updated surface variable list (matches preprocessing)
        self.sfc_vars = [
            "PRES", "MSLMA", "REFC", "T2M", "UGRD10M", "VGRD10M", "UGRD80M", "VGRD80M",
            "D2M", "R2M", "SPFH2M", "POT2M", "TCDC", "LCDC", "MCDC", "HCDC", "VIS", "APCP", "APCP_TOTAL",
            "HGTCC", "CAPE", "CIN", "PWAT", "CRAIN", "RAIN_MASK", "CFRZR", "FRZR_MASK", "WARM_LAYER_DEPTH", "COLD_LAYER_DEPTH",
            "GUST", "GUST_FACTOR", "GUST_CONV", "WIND_10M", "WIND_MAX",
            "VUCSH_0_1km", "VVCSH_0_1km", "VUCSH_0_6km", "VVCSH_0_6km",
            "RELV_max_0_1km", "RELV_max_0_2km", "USTM_0_6km", "VSTM_0_6km",
            "HLCY_0_1km", "HLCY_0_3km", "MXUPHL_max_0_2km", "MNUPHL_min_0_2km",
            "MXUPHL_max_0_3km", "MNUPHL_min_0_3km", "MXUPHL_max_2_5km", "MNUPHL_min_2_5km",
            "MAXUVV_max_100_1000mb", "MAXDVV_max_100_1000mb",
            "HGT_0C", "UGRD_0C", "VGRD_0C", "WIND_0C", "SPFH_0C", "RH_0C",
            "DU_SFC_0C", "DV_SFC_0C", "SHEAR_SFC_0C"
        ]
        
        # Pressure levels (hPa)
        self.levels = [200, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 825, 850, 875, 900, 925, 950, 975, 1000]

        # Plot settings
        self.figure_size = (12, 8)
        self.dpi = 300
        self.cmap_default = 'viridis'
        self.zoom_extent = None
        self.probability_fill_contours = False
        self.probability_label_fontsize = 6
        self.probability_label_outline_width = 1.25


class ForecastPlotter:
    """Handles forecast data visualization."""
    
    def __init__(self, config: ForecastPlotterConfig):
        self.config = config
        self.use_cartopy = CARTOPY_AVAILABLE
        if not self.use_cartopy:
            logger.warning("Cartopy not available, using simple plotting")
    
    def load_forecast_data(self, forecast_file: str) -> xr.Dataset:
        """Load forecast data from NetCDF file."""
        if not os.path.exists(forecast_file):
            raise FileNotFoundError(f"Forecast file not found: {forecast_file}")
        
        try:
            logger.info(f"Loading forecast data from {forecast_file}")
            ds = xr.open_dataset(forecast_file, decode_timedelta=True)
            return ds
        except Exception as e:
            logger.error(f"Error loading forecast data: {e}")
            raise
    
    @staticmethod
    def _sample_cmap(name, n):
        base = plt.get_cmap(name)
        return [mcolors.to_hex(base(i/(n-1))) for i in range(n)]


    @staticmethod
    def get_refc_cmap() -> tuple:
        """Return a colormap and normalization for reflectivity (REFC)."""
        reflectivity_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
        reflectivity_colors = [
            "#FFFFFF", "#00F9F9", "#0080FF", "#0004FF", "#00FF00", "#00C100", "#008000", "#F5FA00",
            "#FFBF00", "#FF8200", "#FF0400", "#BF0000", "#820000", "#FF00FF", "#9062CD",
        ]
        vmin, vmax = min(reflectivity_levels), max(reflectivity_levels)
        cmap = mcolors.ListedColormap(reflectivity_colors)
        norm = mcolors.BoundaryNorm(reflectivity_levels, cmap.N)
        return cmap, norm, vmin, vmax

    @staticmethod
    def get_apcp_cmap() -> tuple:
        """Return colormap + norm for accumulated precipitation (APCP)."""
        apcp_levels = [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 25, 35, 45, 60, 80, 100]
        apcp_colors = [
            "#FFFFFF", "#B0E2FF", "#7EC0EE", "#00FA9A", "#32CD32", "#FFFF00", "#FFD700",
            "#FFA500", "#FF4500", "#FF0000", "#8B0000", "#9400D3", "#8B008B", "#4B0082",
        ]
        vmin, vmax = min(apcp_levels), max(apcp_levels)
        cmap = mcolors.ListedColormap(apcp_colors)
        norm = mcolors.BoundaryNorm(apcp_levels, cmap.N)
        return cmap, norm, vmin, vmax

    @staticmethod
    def get_cape_cmap() -> tuple:
        """Colormap for CAPE (0-7000 J/kg) using 'inferno'."""
        levels = [0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000]
        colors = ForecastPlotter._sample_cmap("inferno", len(levels)-1)
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return cmap, norm, min(levels), max(levels)
    
    @staticmethod
    def get_cin_cmap() -> tuple:
        """Colormap for CIN (-2000 to 0 J/kg) using 'PuBuGn_r'."""
        levels = [-2000, -1500, -1000, -750, -500, -300, -200, -150, -100, -75, -50, -25, -10, -1, 0]
        colors = ForecastPlotter._sample_cmap("PuBuGn_r", len(levels)-1)
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return cmap, norm, min(levels), max(levels)
    
    @staticmethod
    def get_vis_cmap() -> tuple:
        """Colormap for VIS (0-100000 m) using 'YlOrBr_r' and log-ish spaced levels."""
        levels = [10, 50, 100, 200, 400, 800, 1500, 3000, 6000, 12000, 24000, 48000, 100000]
        colors = ForecastPlotter._sample_cmap("YlOrBr_r", len(levels)-1)
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return cmap, norm, min(levels), max(levels)
    
    @staticmethod
    def get_hgtcc_cmap() -> tuple:
        """Colormap for HGTCC (0-20000 m) using 'viridis'."""
        levels = [0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000, 20000]
        colors = ForecastPlotter._sample_cmap("viridis", len(levels)-1)
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return cmap, norm, min(levels), max(levels)

    @staticmethod
    def get_probability_cmap() -> tuple:
        """Colormap for probability fields (0-100%)."""
        levels = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        colors = ForecastPlotter._sample_cmap("YlOrRd", len(levels) - 1)
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
        return cmap, norm, min(levels), max(levels)

    @staticmethod
    def get_probability_linewidths(levels: List[float]) -> List[float]:
        """Return subtly thicker contour lines for higher absolute probabilities."""
        return [0.8 + 0.8 * (level / 100.0) for level in levels]

    @staticmethod
    def get_probability_contour_colors(levels: List[float]) -> List[tuple]:
        """Return contour colors sampled from the probability colormap."""
        cmap, norm, _, _ = ForecastPlotter.get_probability_cmap()
        return [mcolors.to_hex(cmap(norm(level))) for level in levels]

    @staticmethod
    def get_probability_label_positions(contour) -> List[tuple]:
        """Pick one label position per contour level from the longest path."""
        label_positions = []
        for level_segments in contour.allsegs:
            nonempty_segments = [segment for segment in level_segments if len(segment) > 0]
            if not nonempty_segments:
                continue
            longest_segment = max(nonempty_segments, key=len)
            midpoint = len(longest_segment) // 2
            label_positions.append(
                (float(longest_segment[midpoint][0]), float(longest_segment[midpoint][1]))
            )
        return label_positions

    
    def create_plot(self, data: np.ndarray, lats: np.ndarray, lons: np.ndarray, 
                   var_name: str, level: Optional[int] = None,
                   data_attrs: Optional[dict] = None,
                   title_suffix: str = "") -> plt.Figure:
        """Create a plot for a given variable."""
        data_attrs = data_attrs or {}
        is_probability = "probability_threshold" in data_attrs or "base_variable" in data_attrs
        base_var_name = str(data_attrs.get('base_variable', var_name))
        
        # Get variable configuration from VARIABLE_METADATA
        var_meta = VARIABLE_METADATA.get(base_var_name, VARIABLE_METADATA.get(var_name, {}))
        units = data_attrs.get('units', var_meta.get('units', ''))
        long_name = data_attrs.get('long_name', var_meta.get('long_name', var_name))
        
        # Special handling for categorical / thresholded fields
        norm = None
        if is_probability:
            cmap, norm, vmin, vmax = self.get_probability_cmap()
        elif var_name == 'REFC':
            cmap, norm, vmin, vmax = self.get_refc_cmap()
        elif var_name in ('APCP', 'APCP_TOTAL'):
            cmap, norm, vmin, vmax = self.get_apcp_cmap()
        elif var_name == 'CAPE':
            cmap, norm, vmin, vmax = self.get_cape_cmap()
        elif var_name == 'CIN':
            cmap, norm, vmin, vmax = self.get_cin_cmap()
        elif var_name == 'VIS':
            cmap, norm, vmin, vmax = self.get_vis_cmap()
        elif var_name == 'HGTCC':
            cmap, norm, vmin, vmax = self.get_hgtcc_cmap()
        else:
            cmap = var_meta.get('cmap', self.config.cmap_default)
            norm = None
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
        
        # Create figure
        if self.use_cartopy:
            fig = plt.figure(figsize=self.config.figure_size)
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.STATES, linewidth=0.3)
            if self.config.zoom_extent is not None:
                ax.set_extent(self.config.zoom_extent, crs=ccrs.PlateCarree())

            ax.gridlines(draw_labels=True)
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Create the plot
        if is_probability:
            contour_levels = [value for value in norm.boundaries[1:-1] if np.nanmin(data) <= value <= np.nanmax(data)]
            if not contour_levels:
                contour_levels = list(norm.boundaries[1:-1])
            contour_linewidths = self.get_probability_linewidths(contour_levels)
            contour_colors = self.get_probability_contour_colors(contour_levels)

            if self.config.probability_fill_contours:
                im = ax.contourf(
                    lons,
                    lats,
                    data,
                    levels=norm.boundaries,
                    cmap=cmap,
                    norm=norm,
                    extend='neither',
                )
                contour = ax.contour(
                    lons,
                    lats,
                    data,
                    levels=contour_levels,
                    colors=contour_colors,
                    linewidths=contour_linewidths,
                )
            else:
                contour = ax.contour(
                    lons,
                    lats,
                    data,
                    levels=contour_levels,
                    colors=contour_colors,
                    linewidths=contour_linewidths,
                )
                im = contour

            label_positions = self.get_probability_label_positions(contour)
            label_texts = ax.clabel(
                contour,
                contour.levels,
                inline=True,
                inline_spacing=2,
                fontsize=self.config.probability_label_fontsize,
                fmt=lambda value: f"{value:.0f}",
                manual=label_positions,
                colors=["black"],
                rightside_up=True,
                use_clabeltext=True,
            )
            for text in label_texts:
                text.set_path_effects([
                    pe.withStroke(
                        linewidth=self.config.probability_label_outline_width,
                        foreground="white",
                    )
                ])
        elif norm is not None:
            im = ax.contourf(lons, lats, data, levels=norm.boundaries, 
                           cmap=cmap, norm=norm, extend='both')
        else:
            im = ax.contourf(lons, lats, data, levels=20, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label(f'{long_name} ({units})', fontsize=10)
        
        # Set title
        level_str = f" at {level} hPa" if level is not None else ""
        title = f"{long_name}{level_str}{title_suffix}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Set labels
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        
        # Set grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig

    def _plot_variable_list(self, ds: xr.Dataset, variable_names: List[str], lead_hour: int,
                            output_dir: str, timestamp_str: str) -> None:
        """Plot a list of variables, handling both 2D and pressure-level fields."""
        lats = ds['latitude'].values
        lons = ds['longitude'].values
        title_suffix = f"\nForecast: {timestamp_str} + {lead_hour}h"

        for var_name in variable_names:
            if var_name == 'grid_mapping':
                continue
            if var_name not in ds.variables:
                logger.warning(f"Variable {var_name} not found in dataset")
                continue

            da = ds[var_name]
            if 'level' in da.dims:
                levels_in_ds = da['level'].values if 'level' in da.coords else self.config.levels
                for level_idx, level in enumerate(levels_in_ds):
                    try:
                        data = da.isel(time=0, lead_time=0, level=level_idx).values
                        logger.info(
                            f"{var_name} stats - mean: {np.nanmean(data):.2f}, std: {np.nanstd(data):.2f}, "
                            f"min: {np.nanmin(data):.2f}, max: {np.nanmax(data):.2f}"
                        )
                        fig = self.create_plot(data, lats, lons, var_name, level, dict(da.attrs), title_suffix)
                        filename = f"{var_name}_{level}hPa_lead{lead_hour:02d}h.png"
                        filepath = os.path.join(output_dir, filename)
                        fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
                        plt.close(fig)
                        logger.info(f"Saved: {filename}")
                    except Exception as e:
                        logger.error(f"Error plotting {var_name} at {level} hPa: {e}")
                        continue
            else:
                try:
                    data = da.isel(time=0, lead_time=0).values
                    logger.info(
                        f"{var_name} stats - mean: {np.nanmean(data):.2f}, std: {np.nanstd(data):.2f}, "
                        f"min: {np.nanmin(data):.2f}, max: {np.nanmax(data):.2f}"
                    )
                    fig = self.create_plot(data, lats, lons, var_name, None, dict(da.attrs), title_suffix)
                    filename = f"{var_name}_surface_lead{lead_hour:02d}h.png"
                    filepath = os.path.join(output_dir, filename)
                    fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"Saved: {filename}")
                except Exception as e:
                    logger.error(f"Error plotting surface variable {var_name}: {e}")
                    continue
    
    def plot_pressure_level_variables(self, ds: xr.Dataset, lead_hour: int, 
                                    output_dir: str, timestamp_str: str) -> None:
        """Plot all pressure level variables."""
        logger.info("Plotting pressure level variables...")
        self._plot_variable_list(ds, self.config.pl_vars, lead_hour, output_dir, timestamp_str)
    
    def plot_surface_variables(self, ds: xr.Dataset, lead_hour: int, 
                              output_dir: str, timestamp_str: str) -> None:
        """Plot surface variables."""
        logger.info("Plotting surface variables...")
        self._plot_variable_list(ds, self.config.sfc_vars, lead_hour, output_dir, timestamp_str)

    def plot_probability_variables(self, ds: xr.Dataset, lead_hour: int,
                                   output_dir: str, timestamp_str: str) -> None:
        """Plot all probability variables present in a probability dataset."""
        logger.info("Plotting probability variables...")
        prob_vars = [var_name for var_name in ds.data_vars if var_name != 'grid_mapping']
        self._plot_variable_list(ds, prob_vars, lead_hour, output_dir, timestamp_str)
    
    def create_summary_plot(self, ds: xr.Dataset, lead_hour: int, 
                           output_dir: str, timestamp_str: str) -> None:
        """Create a summary plot with key variables."""
        logger.info("Creating summary plot...")
        
        try:
            # Get coordinate data
            lats = ds['latitude'].values
            lons = ds['longitude'].values
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            if self.use_cartopy:
                # Recreate with cartopy if available
                fig = plt.figure(figsize=(16, 12))
                axes = []
                for i in range(4):
                    ax = plt.subplot(2, 2, i+1, projection=ccrs.PlateCarree())
                    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
                    ax.add_feature(cfeature.STATES, linewidth=0.3)
                    if self.config.zoom_extent is not None:
                        ax.set_extent(self.config.zoom_extent, crs=ccrs.PlateCarree())
                    axes.append(ax)
            else:
                axes = axes.flatten()
            
            # Plot key variables
            plots = [
                ('T2M', 'T2M', None, 'Temperature at 2m'),
                ('REFC', 'REFC', None, 'Composite Reflectivity'),
                ('TMP', 'TMP', 850, 'Temperature at 850 hPa'),
                ('UGRD', 'UGRD', 850, 'U-Wind at 850 hPa'),
            ]
            
            for i, (var_name, var_display, level, title) in enumerate(plots):
                if var_name not in ds.variables:
                    continue
                
                # Get data
                if level is not None:
                    # Find level index
                    level_idx = self.config.levels.index(level) if level in self.config.levels else 0
                    data = ds[var_name].isel(time=0, lead_time=0, level=level_idx).values
                else:
                    data = ds[var_name].isel(time=0, lead_time=0).values
                
                # Get colormap from VARIABLE_METADATA
                var_meta = VARIABLE_METADATA.get(var_display, {})
                cmap = var_meta.get('cmap', self.config.cmap_default)
                
                # Special handling for REFC/APCP
                if var_display == 'REFC':
                    cmap_refc, norm_refc, *_ = self.get_refc_cmap()
                    im = axes[i].contourf(
                        lons, lats, data, levels=norm_refc.boundaries,
                        cmap=cmap_refc, norm=norm_refc, extend='both'
                    )
                elif var_display in ('APCP', 'APCP_TOTAL'):
                    cmap_apcp, norm_apcp, *_ = self.get_apcp_cmap()
                    im = axes[i].contourf(
                        lons, lats, data, levels=norm_apcp.boundaries,
                        cmap=cmap_apcp, norm=norm_apcp, extend='both'
                    )
                else:
                    im = axes[i].contourf(lons, lats, data, levels=20, cmap=cmap, extend='both')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[i], shrink=0.4)
                
                # Set title
                axes[i].set_title(f"{title}\nForecast: {timestamp_str} + {lead_hour}h", 
                                fontsize=10, fontweight='bold')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save summary plot
            filename = f"summary_lead{lead_hour:02d}h.png"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved: {filename}")
            
        except Exception as e:
            logger.error(f"Error creating summary plot: {e}")


def plot_lead_hour(h, ds_path, init_datetime, init_year, init_month, init_day, init_hh,
                   output_dir, date_str, member, product_kind, config_dict):
    # Reconstruct config and plotter
    config = ForecastPlotterConfig()
    for k, v in config_dict.items():
        setattr(config, k, v)
    plotter = ForecastPlotter(config)
    ds = xr.open_dataset(ds_path, decode_timedelta=True)
    try:
        valid_datetime = init_datetime + timedelta(hours=h)
        timestamp_str = f"{init_year}-{init_month}-{init_day} {init_hh}:00 UTC"
        output_subdir = f"{output_dir}/{date_str}/mem{member}_lead{h:02d}h"
        utils.make_directory(output_subdir)
        logging.info(f"Plotting product={member} kind={product_kind} hour=f{h:02d} from {ds_path}")
        if product_kind == "prob":
            plotter.plot_probability_variables(ds, h, output_subdir, timestamp_str)
        else:
            plotter.plot_pressure_level_variables(ds, h, output_subdir, timestamp_str)
            plotter.plot_surface_variables(ds, h, output_subdir, timestamp_str)
            plotter.create_summary_plot(ds, h, output_subdir, timestamp_str)
        logging.info(f"Plots for lead hour {h} saved to: {output_subdir}")
    finally:
        ds.close()

def plot_forecast_data(datetime_str: str,
                      lead_hour: str, member: str,
                      zoom_extent: Optional[tuple] = None,
                      forecast_dir: str = "./", output_dir: str = "./",
                      probability_fill_contours: bool = False):
    """Main plotting function. Plots all hours from 1 to lead_hour (inclusive) in parallel."""
    try:
        # Validate inputs
        init_datetime, init_year, init_month, init_day, init_hh = utils.validate_datetime(datetime_str)
        date_str = f"{init_year}{init_month}{init_day}/{init_hh}"
        lead_hour_int = int(lead_hour)
        
        mem_str = str(member).strip().lower()
        mem_str = PRODUCT_ALIASES.get(mem_str, mem_str)
        product_kind = mem_str if mem_str in {"avg", "spr", "prob"} else "member"
        if product_kind == "member":
            mem_str = f"m{int(member):02d}"
        logger.info(f"Resolved plotting target {member} -> product_kind={product_kind}, file_token={mem_str}")

        # Initialize plotter config (for passing to subprocesses)
        config = ForecastPlotterConfig()
        config.zoom_extent = zoom_extent
        config.probability_fill_contours = probability_fill_contours
        config_dict = config.__dict__
        
        n_workers = lead_hour_int
        logger.info(f"Parallel plotting using {n_workers} workers (one per lead hour)")
        # Parallel plotting over lead hours
        args_list = []
        for h in range(1, lead_hour_int + 1):
            # Build per-hour file path
            ds_path = f"{forecast_dir}/{date_str}/hrrrcast_{mem_str}_f{h:02d}.nc"
            if not os.path.exists(ds_path):
                logger.warning(f"Skipping product={member} hour f{h:02d}: file not found {ds_path}")
                continue
            args_list.append((h, ds_path, init_datetime, init_year, init_month, init_day, init_hh, output_dir, date_str, mem_str, product_kind, config_dict))
        if not args_list:
            logger.warning(f"No plotting work queued for target {member} using token {mem_str}")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(plot_lead_hour, *args) for args in args_list]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in parallel plotting: {e}")
        logger.info(f"Plotting completed successfully for all hours 1 to {lead_hour_int}.")
        
    except Exception as e:
        logger.error(f"Plotting failed: {e}")
        raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot Forecast Variables",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('inittime',
                       help='Forecast initialization time in format YYYY-MM-DDTHH (e.g., "2024-05-06T23")')
    parser.add_argument("lead_hour", help="Lead hour for forecast (0, 1, 2, ...)")
    parser.add_argument("--members", nargs='+', required=True, help="List/range of member IDs or products (e.g., 0-2 4 avg spr prob)")
    parser.add_argument("--forecast_dir", default="./", help="Directory containing forecast files")
    parser.add_argument("--output_dir", default="./", help="Output directory for plots")
    parser.add_argument(
        "--lat-range",
        dest="lat_range",
        nargs=2,
        type=float,
        default=None,
        metavar=("LAT_MIN", "LAT_MAX"),
        help="Latitude zoom bounds for map extent (e.g., --lat-range 36 50)",
    )
    parser.add_argument(
        "--lon-range",
        dest="lon_range",
        nargs=2,
        type=float,
        default=None,
        metavar=("LON_MIN", "LON_MAX"),
        help="Longitude zoom bounds for map extent (e.g., --lon-range 259 272)",
    )
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument(
        "--prob-fill-contours",
        action="store_true",
        help="Fill color between probability contour lines; default is contour lines with labels only",
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    global logger
    args = parse_arguments()
    logger = setup_logging(args.log_level)

    try:
        try:
            lat_bounds = _normalize_range(args.lat_range, "lat_range")
            lon_bounds = _normalize_range(args.lon_range, "lon_range")
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)

        zoom_extent = None
        if lat_bounds is not None and lon_bounds is not None:
            zoom_extent = (lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1])
        elif lat_bounds is not None or lon_bounds is not None:
            logger.error("Both --lat-range and --lon-range must be provided together for zooming")
            sys.exit(1)

        def expand_member_arg(m):
            result = []
            for part in m.split(","):
                part = part.strip()
                if "-" in part and part.replace("-", "").isdigit():
                    start, end = part.split("-")
                    result.extend([str(i) for i in range(int(start), int(end) + 1)])
                elif part != "":
                    result.append(part)
            return result

        members = []
        for m in args.members:
            members.extend(expand_member_arg(m))
        members = sorted(set(members), key=lambda x: (not x.isdigit(), x))
        logger.info(f"Expanded plotting targets: {members}")

        for member in members:
            logger.info(f"Starting plotting target: {member}")
            plot_forecast_data(
                datetime_str=args.inittime,
                lead_hour=args.lead_hour,
                member=member,
                zoom_extent=zoom_extent,
                forecast_dir=args.forecast_dir,
                output_dir=args.output_dir,
                probability_fill_contours=args.prob_fill_contours,
            )
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
