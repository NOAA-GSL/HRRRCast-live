#!/bin/bash
#SBATCH --job-name=make_bcs
#SBATCH --output=logs/make_bcs_%j.out
#SBATCH --partition=u1-compute
#SBATCH --account=@[ACCNR]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=@[LEAD_HOUR]
#SBATCH --time=00:30:00
#SBATCH --exclusive

# conda
source etc/env.sh

# set vars
init_time="@[INIT_TIME]"
lead_hour=@[LEAD_HOUR]
year=`echo $init_time |cut -c1-4`
month=`echo $init_time |cut -c6-7`
day=`echo $init_time |cut -c9-10`
hour=`echo $init_time |cut -c12-13`
date_str="${year}${month}${day}_${hour}"

echo "In make_bcs, init_time=${init_time}, year/month/day/hour/,${year} ${month} ${day} ${hour}, lead_hour=${lead_hour}"
python3 src/make_bcs.py net-diffusion/normalize.nc ${init_time} ${lead_hour} --hrrr_grid_file "${date_str}/hrrr_${date_str}_surface.grib2"
