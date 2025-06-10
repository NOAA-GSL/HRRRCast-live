#!/usr/bin/env python3
"""
Forecast Visualization Script

This script plots each variable from the forecast output and saves them as separate PNG files.
It handles both pressure level and surface variables from the HRRR forecast data.

Usage:
    python plot_forecast.py <init_year> <init_month> <init_day> <init_hh> <lead_hour> [--forecast_dir DIR] [--output_dir DIR]
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ForecastPlotterConfig:
    """Configuration class for forecast plotting parameters."""
    
    def __init__(self):
        # Variable definitions matching the preprocessor
        self.pl_vars = ["UGRD", "VGRD", "VVEL", "TMP", "HGT", "SPFH"]
        self.sfc_vars = ["T2M", "REFC"]
        
        # Pressure levels (hPa)
        self.levels = [200, 300, 475, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
        
        # Plot settings
        self.figure_size = (12, 8)
        self.dpi = 300
        self.cmap_default = 'viridis'
        
        # Variable-specific plotting parameters
        self.var_configs = {
            'UGRD': {'cmap': 'RdBu_r', 'units': 'm/s', 'long_name': 'U-component of Wind'},
            'VGRD': {'cmap': 'RdBu_r', 'units': 'm/s', 'long_name': 'V-component of Wind'},
            'VVEL': {'cmap': 'RdBu_r', 'units': 'Pa/s', 'long_name': 'Vertical Velocity'},
            'TMP': {'cmap': 'coolwarm', 'units': 'K', 'long_name': 'Temperature'},
            'HGT': {'cmap': 'terrain', 'units': 'm', 'long_name': 'Geopotential Height'},
            'SPFH': {'cmap': 'Blues', 'units': 'kg/kg', 'long_name': 'Specific Humidity'},
            'T2M': {'cmap': 'coolwarm', 'units': 'K', 'long_name': '2m Temperature'},
            'REFC': {'cmap': 'pyart_NWSRef', 'units': 'dBZ', 'long_name': 'Composite Reflectivity'}
        }


class ForecastPlotter:
    """Handles forecast data visualization."""
    
    def __init__(self, config: ForecastPlotterConfig):
        self.config = config
        # Try to import cartopy, fall back to simple plotting if not available
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            self.use_cartopy = True
        except ImportError:
            logger.warning("Cartopy not available, using simple plotting")
            self.use_cartopy = False
    
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
    
    def get_reflectivity_colormap(self):
        """Create a custom colormap for radar reflectivity."""
        # Define reflectivity colors (similar to NWS radar) - expanded to match bounds
        colors = [
            "#FFFFFF",  # White (no echo)
            "#B0E2FF",  # Light Blue (weak echoes)
            "#7EC0EE",
            "#00FA9A",  # Green (light precipitation)
            "#32CD32",
            "#FFFF00",  # Yellow (moderate precipitation)
            "#FFD700",
            "#FFA500",  # Orange
            "#FF4500",  # Red (heavy rain)
            "#FF0000",
            "#8B0000",  # Dark red (intense storms)
            "#9400D3",  # Purple
            "#8B008B",
            "#4B0082",  # Deep purple (extreme storms)
        ]
        bounds = np.arange(0, 70, 5)
        norm = mcolors.BoundaryNorm(bounds, len(colors))
        cmap = mcolors.ListedColormap(colors)
        return cmap, norm
    
    def create_plot(self, data: np.ndarray, lats: np.ndarray, lons: np.ndarray, 
                   var_name: str, level: Optional[int] = None, 
                   title_suffix: str = "") -> plt.Figure:
        """Create a plot for a given variable."""
        
        # Get variable configuration
        var_config = self.config.var_configs.get(var_name, {})
        cmap = var_config.get('cmap', self.config.cmap_default)
        units = var_config.get('units', '')
        long_name = var_config.get('long_name', var_name)
        
        # Special handling for reflectivity
        if var_name == 'REFC':
            cmap, norm = self.get_reflectivity_colormap()
        else:
            norm = None
        
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
            im = ax.contourf(lons, lats, data, levels=20, cmap=cmap, extend='both')
        
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
        for var_idx, var_name in enumerate(self.config.pl_vars):
            
            if var_name not in ds.variables:
                logger.warning(f"Variable {var_name} not found in dataset")
                continue
            
            for level_idx, level in enumerate(self.config.levels):
                try:
                    # Extract data for this variable and level
                    # Use lead_time dimension and select first time step (time=0)
                    data = ds[var_name].isel(time=0, lead_time=lead_hour, level=level_idx).values
                    
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
                ('UGRD', 'UGRD', 850, 'U-Wind at 850 hPa')
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
                
                # Special handling for reflectivity
                if var_display == 'REFC':
                    cmap, norm = self.get_reflectivity_colormap()
                    im = axes[i].contourf(lons, lats, data, levels=norm.boundaries, 
                                        cmap=cmap, norm=norm, extend='both')
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


def plot_forecast_data(init_year: str, init_month: str, init_day: str, 
                      init_hh: str, lead_hour: str, member: int,
                      forecast_dir: str = "./", output_dir: str = "./"):
    """Main plotting function."""
    try:
        # Validate inputs
        date_str = f"{init_year}{init_month}{init_day}_{init_hh}"
        lead_hour_int = int(lead_hour)
        
        # Calculate forecast valid time
        init_datetime = datetime.strptime(date_str, "%Y%m%d_%H")
        valid_datetime = init_datetime + timedelta(hours=lead_hour_int)
        
        logger.info(f"Plotting forecast data for {init_datetime} + {lead_hour_int}h")
        logger.info(f"Valid time: {valid_datetime}")
        
        # Setup paths
        forecast_file = f"{forecast_dir}/{date_str}/hrrrcast_{date_str}_mem{member}.nc"
        
        # Create output directory
        timestamp_str = f"{init_year}-{init_month}-{init_day} {init_hh}:00 UTC"
        output_subdir = f"{output_dir}/{date_str}/{date_str}_mem{member}_lead{lead_hour_int:02d}h"
        Path(output_subdir).mkdir(parents=True, exist_ok=True)
        
        # Validate forecast file exists
        if not os.path.exists(forecast_file):
            raise FileNotFoundError(f"Forecast file not found: {forecast_file}")
        
        # Initialize plotter
        config = ForecastPlotterConfig()
        plotter = ForecastPlotter(config)
        
        # Load forecast data
        ds = plotter.load_forecast_data(forecast_file)
        
        # Validate lead hour exists in data
        if lead_hour_int >= len(ds.lead_time):
            raise ValueError(f"Lead hour {lead_hour_int} not available in forecast data (max: {len(ds.lead_time)-1})")
        
        # Plot variables
        plotter.plot_pressure_level_variables(ds, lead_hour_int, output_subdir, timestamp_str)
        plotter.plot_surface_variables(ds, lead_hour_int, output_subdir, timestamp_str)
        plotter.create_summary_plot(ds, lead_hour_int, output_subdir, timestamp_str)
        
        # Close dataset
        ds.close()
        
        logger.info(f"Plotting completed successfully. Plots saved to: {output_subdir}")
        
    except Exception as e:
        logger.error(f"Plotting failed: {e}")
        raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot Forecast Variables",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("init_year", help="Initialization year (YYYY)")
    parser.add_argument("init_month", help="Initialization month (MM)")
    parser.add_argument("init_day", help="Initialization day (DD)")
    parser.add_argument("init_hh", help="Initialization hour (HH)")
    parser.add_argument("lead_hour", help="Lead hour for forecast (0, 1, 2, ...)")
    parser.add_argument("member", type=int, default=0, help="Ensemble member ID (0...N)")
    parser.add_argument("--forecast_dir", default="./", help="Directory containing forecast files")
    parser.add_argument("--output_dir", default="./", help="Output directory for plots")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    try:
        args = parse_arguments()
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Run plotting
        plot_forecast_data(
            init_year=args.init_year,
            init_month=args.init_month,
            init_day=args.init_day,
            init_hh=args.init_hh,
            lead_hour=args.lead_hour,
            member=args.member,
            forecast_dir=args.forecast_dir,
            output_dir=args.output_dir
        )
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
