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

from utils import setup_logging

logger = setup_logging('INFO')


class Netcdf2Grib:
    def __init__(self):
        self.ATTR_MAPS = {
            "UGRD": [None, "x_wind", "m s**-1"],
            "VGRD": [None, "y_wind", "m s**-1"],
            "VVEL": [None, "lagrangian_tendency_of_air_pressure", "Pa s**-1"],
            "TMP": [None, "air_temperature", "K"],
            "HGT": [None, "geopotential_height", "m"],
            "SPFH": [None, "specific_humidity", "kg kg**-1"],
            "T2M": [2, "air_temperature", "K"],
            "REFC": [0, "equivalent_reflectivity_factor", "dBZ"],
            "LAND": [0, "land_binary_mask", "1"],
            "OROG": [0, "surface_altitude", "m"],
            # Expanded surface variables
            "PRES": [0, "surface_air_pressure", "Pa"],
            "MSLMA": [0, "air_pressure_at_mean_sea_level", "Pa"],
            "UGRD10M": [10, "x_wind", "m s**-1"],
            "VGRD10M": [10, "y_wind", "m s**-1"],
            "UGRD80M": [80, "x_wind", "m s**-1"],
            "VGRD80M": [80, "y_wind", "m s**-1"],
            "D2M": [2, "dew_point_temperature", "K"],
            "R2M": [2, "relative_humidity", "%"],
            "TCDC": [0, "cloud_area_fraction", "1"],
            "VIS": [0, "visibility_in_air", "m"],
            "APCP": [0, "lwe_thickness_of_precipitation_amount", "kg m**-2"],
            "HGTCC": [0, "geopotential_height_at_cloud_top", "m"],  # approximation
            "CAPE": [0, "atmosphere_convective_available_potential_energy", "J kg**-1"],
            "CIN": [0, "atmosphere_convective_inhibition", "J kg**-1"],
        }

        # GRIB2 parameter overrides: var_name -> (discipline, parameterCategory, parameterNumber, typeOfFirstFixedSurface)
        # NOTE: Codes chosen from WMO GRIB2 tables; some (HGTCC) are approximations and may need refinement.
        self.GRIB_PARAM_OVERRIDE = {
            "REFC":  (0, 16, 196, 10),   # already handled, retained for completeness
            "MSLMA": (0, 3, 1, 102),     # pressure reduced to MSL
            "TCDC":  (0, 6, 1, 10),      # total cloud cover, entire atmosphere
            "VIS":   (0, 19, 0, 1),      # visibility, surface
            "APCP":  (0, 1, 8, 1),       # total precipitation, surface (accum)
            "CAPE":  (0, 7, 6, 1),       # convective available potential energy, surface based
            "CIN":   (0, 7, 7, 1),       # convective inhibition, surface based
            "HGTCC": (0, 6, 13, 1),      # cloud ceiling height (approx: using phys atmos category)
            "OROG":  (0, 3, 5, 1),       # orography, surface
        }

    def tweaked_messages(self, cube):
        """
        Adjust GRIB messages based on cube properties.
        """

        for cube, grib_message in iris_grib.save_pairs_from_cube(cube):

            eccodes.codes_set(grib_message, "centre", "kwbc")
            eccodes.codes_set(grib_message, "localTablesVersion", 1)
            eccodes.codes_set(
                grib_message, "latitudeOfFirstGridPointInDegrees", 21.138123
            )
            eccodes.codes_set(
                grib_message, "longitudeOfFirstGridPointInDegrees", 237.280472
            )

            # Retrieve original variable name if stored
            orig_name = cube.attributes.get("orig_name", None)
            std_name = cube.standard_name if hasattr(cube, "standard_name") else None

            if std_name == "equivalent_reflectivity_factor" or orig_name == "REFC":
                eccodes.codes_set(grib_message, "discipline", 0)
                eccodes.codes_set(grib_message, "parameterCategory", 16)
                eccodes.codes_set(grib_message, "parameterNumber", 196)
                eccodes.codes_set(grib_message, "typeOfFirstFixedSurface", 10)
            else:
                key = orig_name if orig_name in self.GRIB_PARAM_OVERRIDE else None
                if key is not None:
                    disc, cat, num, surface = self.GRIB_PARAM_OVERRIDE[key]
                    eccodes.codes_set(grib_message, "discipline", disc)
                    eccodes.codes_set(grib_message, "parameterCategory", cat)
                    eccodes.codes_set(grib_message, "parameterNumber", num)
                    eccodes.codes_set(grib_message, "typeOfFirstFixedSurface", surface)

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
        # forecasts = forecasts.rename({'time': 'init_time'})
        forecasts = forecasts.rename({"lead_time": "time"})

        # dx, dy
        ny, nx = forecasts.latitude.shape
        dx, dy = 6000, 6000
        x = (np.arange(nx) - nx // 2) * dx
        y = (np.arange(ny) - ny // 2) * dy

        forecasts = forecasts.rename_dims({"latitude": "y", "longitude": "x"})
        forecasts = forecasts.assign_coords({"x": ("x", x), "y": ("y", y)})

        # Update units
        forecasts["level"] = forecasts["level"] * 100
        forecasts["level"].attrs["long_name"] = "pressure"
        forecasts["level"].attrs["units"] = "Pa"
        # forecasts['HGT'] = forecasts['HGT'] / 9.80665

        forecasts["x"].attrs = {
            "long_name": "x coordinate of projection",
            "standard_name": "projection_x_coordinate",
            "units": "m",
            "grid_spacing": 6000.0,
        }
        forecasts["y"].attrs = {
            "long_name": "y coordinate of projection",
            "standard_name": "projection_y_coordinate",
            "units": "m",
            "grid_spacing": 6000.0,
        }
        forecasts["latitude"].attrs = {
            "units": "degree_north",
            "standard_name": "latitude",
        }

        forecasts["longitude"].attrs = {
            "units": "degree_east",
            "standard_name": "longitude",
        }

        if member == "avg":
            filename = os.path.join(outdir, f"hrrrcast_avg.nc")
        else:
            filename = os.path.join(outdir, f"hrrrcast_m{member:02d}.nc")
        # write to netCDF file
        forecasts.to_netcdf(filename)

        # Load cubes from netCDF file
        cubes = iris.load(filename)

        # add x, y coords
        y_dimco = DimCoord(
            y,
            standard_name="projection_y_coordinate",
            units="m",
        )
        x_dimco = DimCoord(
            x,
            standard_name="projection_x_coordinate",
            units="m",
        )

        times = cubes[0].coord("time").points
        cycle = forecast_starttime.hour
        logger.info(f"Forecast start time is {forecast_starttime}")

        datevectors = [forecast_starttime + timedelta(hours=int(t)) for t in times]

        time_fmt_str = "00:00:00"
        time_unit_str = (
            f"Hours since {forecast_starttime.strftime('%Y-%m-%d %H:00:00')}"
        )
        time_coord = cubes[0].coord("time")
        new_time_unit = cf_units.Unit(
            time_unit_str, calendar=cf_units.CALENDAR_STANDARD
        )
        new_time_points = [new_time_unit.date2num(dt) for dt in datevectors]
        new_time_coord = iris.coords.DimCoord(
            new_time_points, standard_name="time", units=new_time_unit
        )

        for idate, date in enumerate(datevectors):
            logger.info(f"Processing for time {date.strftime('%Y-%m-%d %H:00:00')}")
            hrs = int((date - forecast_starttime).total_seconds() // 3600)

            if member == "avg":
                outfile = os.path.join(
                    outdir, f"hrrrcast.avg.t{cycle:02d}z.pgrb2.f{hrs:02d}"
                )
            else:
                outfile = os.path.join(
                    outdir, f"hrrrcast.m{member:02d}.t{cycle:02d}z.pgrb2.f{hrs:02d}"
                )
            logger.info(f"grib2 file name: {outfile}")

            for cube in sorted(cubes, key=lambda cube: cube.name()):
                var_name = cube.name()

                # Adjust cube for different variables
                time_coord_dim = cube.coord_dims("time")
                cube.remove_coord("time")
                cube.add_dim_coord(new_time_coord, time_coord_dim)

                if idate == 0:
                    for idim, co in enumerate([y_dimco, x_dimco]):
                        if len(cube.data.shape) == 4:
                            cube.add_dim_coord(co, idim + 2)
                        elif len(cube.data.shape) == 3:
                            cube.add_dim_coord(co, idim + 1)

                hour_slice = iris.Constraint(
                    time=iris.time.PartialDateTime(
                        month=date.month, day=date.day, hour=date.hour
                    )
                )
                cube_slice = cube.extract(hour_slice)

                cube_slice.coord(
                    "projection_y_coordinate"
                ).coord_system = iris.coord_systems.LambertConformal(
                    central_lat=38.5,
                    central_lon=262.5,
                    false_easting=0.0,
                    false_northing=0.0,
                    secant_latitudes=(38.5, 38.5),
                    ellipsoid=iris.coord_systems.GeogCS(6371229.0),
                )
                cube_slice.coord(
                    "projection_x_coordinate"
                ).coord_system = iris.coord_systems.LambertConformal(
                    central_lat=38.5,
                    central_lon=262.5,
                    false_easting=0.0,
                    false_northing=0.0,
                    secant_latitudes=(38.5, 38.5),
                    ellipsoid=iris.coord_systems.GeogCS(6371229.0),
                )

                if len(cube_slice.data.shape) == 3:
                    levels = cube_slice.coord("pressure").points
                    for level in levels:
                        cube_slice_level = cube_slice.extract(
                            iris.Constraint(pressure=level)
                        )
                        cube_slice_level.add_aux_coord(
                            iris.coords.DimCoord(
                                hrs, standard_name="forecast_period", units="hours"
                            )
                        )
                        cube_slice_level.attributes["orig_name"] = var_name
                        cube_slice_level.standard_name = self.ATTR_MAPS[var_name][1]
                        cube_slice_level.units = self.ATTR_MAPS[var_name][2]
                        iris_grib.save_messages(
                            self.tweaked_messages(cube_slice_level),
                            outfile,
                            append=True,
                        )
                else:
                    cube_slice.add_aux_coord(
                        iris.coords.DimCoord(
                            hrs, standard_name="forecast_period", units="hours"
                        )
                    )
                    if var_name in self.ATTR_MAPS:
                        cube_slice.attributes["orig_name"] = var_name
                        cube_slice.standard_name = self.ATTR_MAPS[var_name][1]
                        cube_slice.units = self.ATTR_MAPS[var_name][2]
                        height_val = self.ATTR_MAPS[var_name][0]
                        if height_val is not None and height_val > 0:
                            cube_slice.add_aux_coord(
                                iris.coords.DimCoord(
                                    height_val,
                                    standard_name="height",
                                    units="m",
                                )
                            )
                    else:
                        logger.warning(f"Variable {var_name} missing in ATTR_MAPS; using existing metadata")
                    iris_grib.save_messages(
                        self.tweaked_messages(cube_slice), outfile, append=True
                    )

            # Use wgrib2 to generate index files
            output_idx_file = f"{outfile}.idx"

            # Construct the wgrib2 command
            wgrib2_command = ["wgrib2", "-s", outfile]
            logger.info(f"Running wgrib2 command: {' '.join(wgrib2_command)} to generate index file {output_idx_file}")

            try:
                # Open the output file for writing
                with open(output_idx_file, "w") as f_out:
                    # Execute the wgrib2 command and redirect stdout to the output file
                    subprocess.run(wgrib2_command, stdout=f_out, check=True)

                logger.info(f"Index file created successfully: {output_idx_file}")

            except subprocess.CalledProcessError as e:
                logger.error(f"Error running wgrib2 command: {e}")

        # Remove intermediate netCDF file
        if os.path.isfile(filename):
            logger.info(f"Attempting to delete intermediate nc file {filename}")
            try:
                os.remove(filename)
                logger.info(f"Successfully deleted intermediate nc file {filename}")
            except Exception as e:
                logger.error(f"Failed to delete intermediate nc file {filename}: {e}")
