#!/bin/bash
#SBATCH --job-name=make_bcs
#SBATCH --output=logs/make_bcs_%j.out
#SBATCH --partition=u1-compute
#SBATCH --account=@[ACCNR]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00

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

python3 src/make_bcs.py net-diffusion/normalize.nc ${init_time} ${lead_hour} --hrrr_grid_file "${date_str}/hrrr_${date_str}_surface.grib2"
