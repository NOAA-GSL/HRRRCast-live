#!/bin/bash
#SBATCH --job-name=make_bcs
#SBATCH --output=logs/make_bcs_%j.out
#SBATCH --partition=u1-compute
#SBATCH --account=gsd-hpcs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00

# conda
source etc/env.sh

year=${1:-2024}
month=${2:-05}
day=${3:-06}
hour=${4:-23}
lead_hour=${5:-18}
init_time="${year} ${month} ${day} ${hour}"
date_str="${year}${month}${day}_${hour}"

python3 src/make_bcs.py net-diffusion/normalize.nc ${init_time} ${lead_hour} --hrrr_grid_file "${date_str}/hrrr_${date_str}_surface.grib2"
