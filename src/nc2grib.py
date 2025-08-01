""" Utility for converting netcdf data to grib2.

    Histroy:
    07/14/2025: Linlin Cui (linlin.cui@noaa.gov), initial code
"""

import os
import logging
from datetime import datetime, timedelta
import glob
import subprocess
import numpy as np
import cf_units
import iris
from iris.coords import DimCoord
import iris_grib
import eccodes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Netcdf2Grib:
    def __init__(self):
        self.ATTR_MAPS = {
            'UGRD': [None, 'x_wind', 'm s**-1'],
            'VGRD': [None, 'y_wind', 'm s**-1'],
            'VVEL': [None, 'lagrangian_tendency_of_air_pressure', 'Pa s**-1'],
            'TMP': [None, 'air_temperature', 'K'],
            'HGT': [None, 'geopotential_height', 'm'],
            'SPFH': [None, 'specific_humidity', 'kg kg**-1'],
            'T2M': [2, 'air_temperature', 'K'],
            'REFC': [0, 'equivalent_reflectivity_factor', 'dBZ'],
        }

    def tweaked_messages(self, cube):
        """
        Adjust GRIB messages based on cube properties.
        """

        for cube, grib_message in iris_grib.save_pairs_from_cube(cube):

            eccodes.codes_set(grib_message, 'centre', 'kwbc')
            eccodes.codes_set(grib_message, 'localTablesVersion', 1)
            eccodes.codes_set(grib_message, 'latitudeOfFirstGridPointInDegrees', 21.138123)
            eccodes.codes_set(grib_message, 'longitudeOfFirstGridPointInDegrees', 237.280472)

            if cube.standard_name == 'equivalent_reflectivity_factor':
                eccodes.codes_set(grib_message, 'discipline', 0)
                eccodes.codes_set(grib_message, 'parameterCategory', 16)
                eccodes.codes_set(grib_message, 'parameterNumber', 196)
                eccodes.codes_set(grib_message, 'typeOfFirstFixedSurface', 10)

        yield grib_message

    def save_grib2(self, forecast_starttime, forecasts, member, outdir):
        """
        Convert netCDF file to GRIB2 format file.
            Args:
              forecast_starttime: datetime object for the model initialized time
              forecasts: xarray forecasts dataset
              member: int, member id
              outdir: output directory
        
            Returns:
              No return values, will save to grib2 file
        """
        forecasts = forecasts.isel(time=0, drop=True)
        #forecasts = forecasts.rename({'time': 'init_time'})
        forecasts = forecasts.rename({'lead_time': 'time'})

        #dx, dy
        ny, nx = forecasts.latitude.shape
        dx, dy = 6000, 6000
        x = (np.arange(nx) - nx // 2) * dx
        y = (np.arange(ny) - ny // 2) * dy

        forecasts = forecasts.rename_dims({'latitude': 'y', 'longitude': 'x'})
        forecasts = forecasts.assign_coords({"x":("x", x), "y": ("y", y)})

        # Update units
        forecasts['level'] = forecasts['level'] * 100
        forecasts['level'].attrs['long_name'] = 'pressure'
        forecasts['level'].attrs['units'] = 'Pa'
        #forecasts['HGT'] = forecasts['HGT'] / 9.80665

        forecasts['x'].attrs = {
            'long_name': 'x coordinate of projection',
            'standard_name': 'projection_x_coordinate',
            'units': 'm',
            'grid_spacing': 6000.,
        }
        forecasts['y'].attrs = {
            'long_name': 'y coordinate of projection',
            'standard_name': 'projection_y_coordinate',
            'units': 'm',
            'grid_spacing': 6000.,
        }
        forecasts['latitude'].attrs = {
            'units': 'degree_north',
            'standard_name': 'latitude',       
        }

        forecasts['longitude'].attrs = {
            'units': 'degree_east',
            'standard_name': 'longitude',       
        }

        if member == "avg":
            filename = os.path.join(outdir, f"hrrrcast_avg.nc")
        else:
            filename = os.path.join(outdir, f"hrrrcast_m{member:02d}.nc")
        # write to netCDF file
        forecasts.to_netcdf(filename)

        # Load cubes from netCDF file
        cubes = iris.load(filename)

        #add x, y coords
        y_dimco = DimCoord(
            y,
            standard_name='projection_y_coordinate',
            units='m',
        )
        x_dimco = DimCoord(
            x,
            standard_name='projection_x_coordinate',
            units='m',
        )

        times = cubes[0].coord('time').points
        cycle = forecast_starttime.hour
        logging.info(f'Forecast start time is {forecast_starttime}')

        datevectors = [forecast_starttime + timedelta(hours=int(t)) for t in times]

        time_fmt_str = '00:00:00'
        time_unit_str = f"Hours since {forecast_starttime.strftime('%Y-%m-%d %H:00:00')}"
        time_coord = cubes[0].coord('time')
        new_time_unit = cf_units.Unit(time_unit_str, calendar=cf_units.CALENDAR_STANDARD)
        new_time_points = [new_time_unit.date2num(dt) for dt in datevectors]
        new_time_coord = iris.coords.DimCoord(new_time_points, standard_name='time', units=new_time_unit)

        for idate, date in enumerate(datevectors):
            logging.info(f"Processing for time {date.strftime('%Y-%m-%d %H:00:00')}")
            hrs = int((date - forecast_starttime).total_seconds() // 3600)

            if member == "avg":
                outfile = os.path.join(outdir, f'hrrrcast.avg.t{cycle:02d}z.pgrb2.f{hrs:02d}')
            else:
                outfile = os.path.join(outdir, f'hrrrcast.m{member:02d}.t{cycle:02d}z.pgrb2.f{hrs:02d}')
            logging.info(f"grib2 file name: {outfile}")

            for cube in sorted(cubes, key=lambda cube: cube.name()):
                var_name = cube.name()

                # Adjust cube for different variables
                time_coord_dim = cube.coord_dims('time')
                cube.remove_coord('time')
                cube.add_dim_coord(new_time_coord, time_coord_dim)

                if idate == 0:
                    for idim, co in enumerate([y_dimco, x_dimco]):
                        if len(cube.data.shape) == 4:
                            cube.add_dim_coord(co, idim+2)
                        elif len(cube.data.shape) == 3:
                            cube.add_dim_coord(co, idim+1)


                hour_slice = iris.Constraint(time=iris.time.PartialDateTime(month=date.month, day=date.day, hour=date.hour))
                cube_slice = cube.extract(hour_slice)

                cube_slice.coord('projection_y_coordinate').coord_system = iris.coord_systems.LambertConformal(
                    central_lat=38.5, 
                    central_lon=262.5, 
                    false_easting=0.0, 
                    false_northing=0.0, 
                    secant_latitudes=(38.5, 38.5), 
                    ellipsoid=iris.coord_systems.GeogCS(6371229.0)
                )
                cube_slice.coord('projection_x_coordinate').coord_system = iris.coord_systems.LambertConformal(
                    central_lat=38.5, 
                    central_lon=262.5, 
                    false_easting=0.0, 
                    false_northing=0.0, 
                    secant_latitudes=(38.5, 38.5), 
                    ellipsoid=iris.coord_systems.GeogCS(6371229.0)
                )

                if len(cube_slice.data.shape) == 3:
                    levels = cube_slice.coord('pressure').points
                    for level in levels:
                        cube_slice_level = cube_slice.extract(iris.Constraint(pressure=level))
                        cube_slice_level.add_aux_coord(iris.coords.DimCoord(hrs, standard_name='forecast_period', units='hours'))
                        cube_slice_level.standard_name = self.ATTR_MAPS[var_name][1]
                        cube_slice_level.units = self.ATTR_MAPS[var_name][2]
                        iris_grib.save_messages(self.tweaked_messages(cube_slice_level), outfile, append=True)
                else:
                    cube_slice.add_aux_coord(iris.coords.DimCoord(hrs, standard_name='forecast_period', units='hours'))
                    cube_slice.standard_name = self.ATTR_MAPS[var_name][1]
                    cube_slice.units = self.ATTR_MAPS[var_name][2]

                    if var_name not in ['REFC']:
                        cube_slice.add_aux_coord(iris.coords.DimCoord(self.ATTR_MAPS[var_name][0], standard_name='height', units='m'))
                        iris_grib.save_messages(self.tweaked_messages(cube_slice), outfile, append=True)

                    elif var_name in ['REFC']:
                        cube_slice.add_aux_coord(iris.coords.DimCoord(self.ATTR_MAPS[var_name][0], standard_name='height', units='m'))
                        iris_grib.save_messages(self.tweaked_messages(cube_slice), outfile, append=True)

            # Use wgrib2 to generate index files
            output_idx_file = f"{outfile}.idx"
            
            # Construct the wgrib2 command
            wgrib2_command = ['wgrib2', '-s', outfile]
            
            try:
                # Open the output file for writing
                with open(output_idx_file, "w") as f_out:
                    # Execute the wgrib2 command and redirect stdout to the output file
                    subprocess.run(wgrib2_command, stdout=f_out, check=True)
            
                logging.info(f"Index file created successfully: {output_idx_file}")
            
            except subprocess.CalledProcessError as e:
                logging.info(f"Error running wgrib2 command: {e}")
    
        # Remove intermediate netCDF file
        if os.path.isfile(filename):
            logging.info(f'Deleting intermediate nc file {filename}: ')
            os.remove(filename)
