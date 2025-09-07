#!/usr/bin/env python3
"""
Forecast Visualization Script

This script plots each variable from the forecast output and saves them as separate PNG files.
It handles both pressure level and surface variables from the HRRR forecast data.

Usage:
    python plot_forecast.py <init_time> <lead_hour> <member> [--forecast_dir DIR] [--output_dir DIR]
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from dateutil import parser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
from matplotlib.patches import Rectangle
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

# Local imports
import utils
from utils import setup_logging

logger = None


class ForecastPlotterConfig:
    """Configuration class for forecast plotting parameters."""
    
    def __init__(self):
        # Variable definitions matching the preprocessor
        self.pl_vars = ["UGRD", "VGRD", "VVEL", "TMP", "HGT", "SPFH"]
        # Updated surface variable list (matches preprocessing)
        self.sfc_vars = [
            "PRES", "MSLMA", "REFC", "T2M", "UGRD10M", "VGRD10M", "UGRD80M", "VGRD80M",
            "D2M", "R2M", "TCDC", "VIS", "APCP", "HGTCC", "CAPE", "CIN"
        ]
        
        # Pressure levels (hPa)
        self.levels = [200, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 825, 850, 875, 900, 925, 950, 975, 1000]

        # Plot settings
        self.figure_size = (12, 8)
        self.dpi = 300
        self.cmap_default = 'viridis'
        
        # Variable-specific plotting parameters
        # Expanded variable configs (surface + pressure)
        self.var_configs = {
            # Pressure-level
            'UGRD':   {'cmap': 'RdBu_r',    'units': 'm/s',   'long_name': 'U-component of Wind'},
            'VGRD':   {'cmap': 'RdBu_r',    'units': 'm/s',   'long_name': 'V-component of Wind'},
            'VVEL':   {'cmap': 'RdBu_r',    'units': 'Pa/s',  'long_name': 'Vertical Velocity'},
            'TMP':    {'cmap': 'coolwarm',  'units': 'K',     'long_name': 'Temperature'},
            'HGT':    {'cmap': 'terrain',   'units': 'm',     'long_name': 'Geopotential Height'},
            'SPFH':   {'cmap': 'Blues',     'units': 'kg/kg', 'long_name': 'Specific Humidity'},
            # Surface
            'PRES':    {'cmap': 'viridis',  'units': 'Pa',    'long_name': 'Surface Pressure'},
            'MSLMA':   {'cmap': 'viridis',  'units': 'Pa',    'long_name': 'Mean Sea Level Pressure'},
            'REFC':    {'cmap': 'pyart_NWSRef', 'units': 'dBZ','long_name': 'Composite Reflectivity'},
            'T2M':     {'cmap': 'coolwarm', 'units': 'K',     'long_name': '2m Temperature'},
            'UGRD10M': {'cmap': 'RdBu_r',   'units': 'm/s',   'long_name': '10m U Wind'},
            'VGRD10M': {'cmap': 'RdBu_r',   'units': 'm/s',   'long_name': '10m V Wind'},
            'UGRD80M': {'cmap': 'RdBu_r',   'units': 'm/s',   'long_name': '80m U Wind'},
            'VGRD80M': {'cmap': 'RdBu_r',   'units': 'm/s',   'long_name': '80m V Wind'},
            'D2M':     {'cmap': 'coolwarm', 'units': 'K',     'long_name': '2m Dewpoint'},
            'R2M':     {'cmap': 'YlGnBu',   'units': '%',     'long_name': '2m Relative Humidity'},
            'TCDC':    {'cmap': 'Greys',    'units': 'frac',  'long_name': 'Total Cloud Cover'},
            'VIS':     {'cmap': 'plasma_r', 'units': 'm',     'long_name': 'Visibility'},
            'APCP':    {'cmap': 'Blues',    'units': 'mm',    'long_name': 'Accumulated Precipitation'},
            'HGTCC':   {'cmap': 'cividis',  'units': 'm',     'long_name': 'Cloud Ceiling Height'},
            'CAPE':    {'cmap': 'Spectral_r','units': 'J/kg', 'long_name': 'CAPE'},
            'CIN':     {'cmap': 'PuOr',     'units': 'J/kg',  'long_name': 'CIN'},
        }


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
    
    def get_refc_cmap(self) -> tuple:
        """Return colormap + norm for composite reflectivity (REFC)."""
        reflectivity_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
        reflectivity_colors = [
            "#FFFFFF", "#B0E2FF", "#7EC0EE", "#00FA9A", "#32CD32", "#FFFF00", "#FFD700",
            "#FFA500", "#FF4500", "#FF0000", "#8B0000", "#9400D3", "#8B008B", "#4B0082",
        ]
        vmin, vmax = min(reflectivity_levels), max(reflectivity_levels)
        cmap = mcolors.ListedColormap(reflectivity_colors)
        norm = mcolors.BoundaryNorm(reflectivity_levels, cmap.N)
        return cmap, norm, vmin, vmax

    def get_apcp_cmap(self) -> tuple:
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
    
    def create_plot(self, data: np.ndarray, lats: np.ndarray, lons: np.ndarray, 
                   var_name: str, level: Optional[int] = None, 
                   title_suffix: str = "") -> plt.Figure:
        """Create a plot for a given variable."""
        
        # Get variable configuration
        var_config = self.config.var_configs.get(var_name, {})
        units = var_config.get('units', '')
        long_name = var_config.get('long_name', var_name)
        
        # Special handling for categorical / thresholded fields
        norm = None
        if var_name == 'REFC':
            cmap, norm, vmin, vmax = self.get_refc_cmap()
        elif var_name == 'APCP':
            cmap, norm, vmin, vmax = self.get_apcp_cmap()
        else:
            cmap = var_config.get('cmap', self.config.cmap_default)
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

            ax.gridlines(draw_labels=True)
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Create the plot
        if norm is not None:
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
    
    def plot_pressure_level_variables(self, ds: xr.Dataset, lead_hour: int, 
                                    output_dir: str, timestamp_str: str) -> None:
        """Plot all pressure level variables."""
        logger.info("Plotting pressure level variables...")
        
        # Get coordinate data
        lats = ds['latitude'].values
        lons = ds['longitude'].values
        
        # Create title suffix with forecast information
        title_suffix = f"\nForecast: {timestamp_str} + {lead_hour}h"
        
        # Plot each variable at each level
        levels_in_ds = ds['level'].values if 'level' in ds.dims or 'level' in ds.coords else self.config.levels
        for var_idx, var_name in enumerate(self.config.pl_vars):
            if var_name not in ds.variables:
                logger.warning(f"Variable {var_name} not found in dataset")
                continue

            for level_idx, level in enumerate(levels_in_ds):
                try:
                    # Extract data for this variable and level
                    # Use lead_time dimension and select first time step (time=0)
                    data = ds[var_name].isel(time=0, lead_time=lead_hour, level=level_idx).values
                    logger.info(f"{var_name} stats - mean: {np.nanmean(data):.2f}, std: {np.nanstd(data):.2f}, min: {np.nanmin(data):.2f}, max: {np.nanmax(data):.2f}")
                    
                    # Create plot
                    fig = self.create_plot(data, lats, lons, var_name, level, title_suffix)

                    # Save plot
                    filename = f"{var_name}_{level}hPa_lead{lead_hour:02d}h.png"
                    filepath = os.path.join(output_dir, filename)
                    fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
                    plt.close(fig)
                    
                    logger.info(f"Saved: {filename}")
                    
                except Exception as e:
                    logger.error(f"Error plotting {var_name} at {level} hPa: {e}")
                    continue
    
    def plot_surface_variables(self, ds: xr.Dataset, lead_hour: int, 
                              output_dir: str, timestamp_str: str) -> None:
        """Plot surface variables."""
        logger.info("Plotting surface variables...")
        
        # Get coordinate data
        lats = ds['latitude'].values
        lons = ds['longitude'].values
        
        # Create title suffix with forecast information
        title_suffix = f"\nForecast: {timestamp_str} + {lead_hour}h"
        
        # Plot each surface variable
        for var_name in self.config.sfc_vars:
            
            if var_name not in ds.variables:
                logger.warning(f"Variable {var_name} not found in dataset")
                continue
            
            try:
                # Extract data for this variable
                # Use lead_time dimension and select first time step (time=0)
                data = ds[var_name].isel(time=0, lead_time=lead_hour).values
                # log mean, std, min, max of data
                logger.info(f"{var_name} stats - mean: {np.nanmean(data):.2f}, std: {np.nanstd(data):.2f}, min: {np.nanmin(data):.2f}, max: {np.nanmax(data):.2f}")
                
                # Create plot
                fig = self.create_plot(data, lats, lons, var_name, None, title_suffix)
                
                # Save plot
                filename = f"{var_name}_surface_lead{lead_hour:02d}h.png"
                filepath = os.path.join(output_dir, filename)
                fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
                plt.close(fig)
                
                logger.info(f"Saved: {filename}")
                
            except Exception as e:
                logger.error(f"Error plotting surface variable {var_name}: {e}")
                continue
    
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
                    data = ds[var_name].isel(time=0, lead_time=lead_hour, level=level_idx).values
                else:
                    data = ds[var_name].isel(time=0, lead_time=lead_hour).values
                
                # Get colormap
                var_config = self.config.var_configs.get(var_display, {})
                cmap = var_config.get('cmap', self.config.cmap_default)
                
                # Special handling for REFC/APCP
                if var_display == 'REFC':
                    cmap_refc, norm_refc, *_ = self.get_refc_cmap()
                    im = axes[i].contourf(
                        lons, lats, data, levels=norm_refc.boundaries,
                        cmap=cmap_refc, norm=norm_refc, extend='both'
                    )
                elif var_display == 'APCP':
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


def plot_lead_hour(h, ds_path, init_datetime, init_year, init_month, init_day, init_hh, output_dir, date_str, member, config_dict):
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
        plotter.plot_pressure_level_variables(ds, h, output_subdir, timestamp_str)
        plotter.plot_surface_variables(ds, h, output_subdir, timestamp_str)
        plotter.create_summary_plot(ds, h, output_subdir, timestamp_str)
        logging.info(f"Plots for lead hour {h} saved to: {output_subdir}")
    finally:
        ds.close()

def plot_forecast_data(datetime_str: str,
                      lead_hour: str, member: str,
                      forecast_dir: str = "./", output_dir: str = "./"):
    """Main plotting function. Plots all hours from 1 to lead_hour (inclusive) in parallel."""
    try:
        # Validate inputs
        init_datetime, init_year, init_month, init_day, init_hh = utils.validate_datetime(datetime_str)
        date_str = f"{init_year}{init_month}{init_day}/{init_hh}"
        lead_hour_int = int(lead_hour)
        
        # Setup paths
        forecast_file = f"{forecast_dir}/{date_str}/hrrrcast_mem{member}.nc"
        logger.info(f"Forecast file: {forecast_file}")
        
        # Validate forecast file exists
        if not os.path.exists(forecast_file):
            raise FileNotFoundError(f"Forecast file not found: {forecast_file}")
        
        # Initialize plotter config (for passing to subprocesses)
        config = ForecastPlotterConfig()
        config_dict = config.__dict__
        
        # Load forecast data ONCE to check max lead
        ds = xr.open_dataset(forecast_file, decode_timedelta=True)
        max_lead = len(ds.lead_time) - 1
        ds.close()
        if lead_hour_int > max_lead:
            raise ValueError(f"Lead hour {lead_hour_int} not available in forecast data (max: {max_lead})")
        
        n_workers = lead_hour_int
        logger.info(f"Parallel plotting using {n_workers} workers (one per lead hour)")
        # Parallel plotting over lead hours
        args_list = [
            (h, forecast_file, init_datetime, init_year, init_month, init_day, init_hh, output_dir, date_str, member, config_dict)
            for h in range(1, lead_hour_int + 1)
        ]
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
    parser.add_argument("--members", nargs='+', required=True, help="List/range of member IDs (e.g., 0-2 4 6-7 pmm)")
    parser.add_argument("--forecast_dir", default="./", help="Directory containing forecast files")
    parser.add_argument("--output_dir", default="./", help="Output directory for plots")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    global logger
    args = parse_arguments()
    logger = setup_logging(args.log_level)

    try:
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

        for member in members:
            plot_forecast_data(
                datetime_str=args.inittime,
                lead_hour=args.lead_hour,
                member=member,
                forecast_dir=args.forecast_dir,
                output_dir=args.output_dir,
            )
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
