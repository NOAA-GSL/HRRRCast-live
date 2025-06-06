# -------------------------------
# Imports
# -------------------------------
import subprocess as sp
import sys


# -------------------------------
# Configuration
# -------------------------------
init_year = sys.argv[1]
init_month = sys.argv[2]
init_day = sys.argv[3]
init_hh = sys.argv[4]

base_dir = "./"

# -------------------------------
# Download GRIB2 data
# -------------------------------
def download_grib_files(year, month, day, hh):
    # Download HRRR pressure and surface GRIB2 files for initialization time
    pres_url = f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{year}{month}{day}/conus/hrrr.t{hh}z.wrfprsf00.grib2"
    sfc_url  = f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{year}{month}{day}/conus/hrrr.t{hh}z.wrfsfcf00.grib2"
    sp.call(f"wget -O {base_dir}/{year}{month}{day}_{hh}_pres {pres_url}", shell=True)
    sp.call(f"wget -O {base_dir}/{year}{month}{day}_{hh}_sfc  {sfc_url}", shell=True)

# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":
    # Download GRIB files
    download_grib_files(init_year, init_month, init_day, init_hh)
