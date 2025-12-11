#!/bin/bash
#SBATCH --job-name=make_bcs
#SBATCH --output=logs/make_bcs_%j.out
#SBATCH --partition=u1-compute
#SBATCH --account=@[ACCNR]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=@[LEAD_HOUR]
#SBATCH --time=@[MAKE_BCS_WALLTIME]
#SBATCH --exclusive

# set vars
INIT_TIME="@[INIT_TIME]"
LEAD_HOUR=@[LEAD_HOUR]
PACKAGEROOT=@[PACKAGEROOT]
DATAROOT=@[DATAROOT]

# extract date and hour
DATE=${INIT_TIME%%T*}
DATE=${DATE//-/}
HOUR=${INIT_TIME#*T}

# conda
source ${PACKAGEROOT}/etc/env.sh

# job
echo "In make_bcs, init_time=${INIT_TIME}, lead_hour=${LEAD_HOUR}"
python3 ${PACKAGEROOT}/src/make_bcs.py ${PACKAGEROOT}/net-diffusion/normalize-stats.nc ${INIT_TIME} ${LEAD_HOUR} --base_dir ${DATAROOT} --output_dir ${DATAROOT} --hrrr_grid_file "${DATE}/${HOUR}/hrrr_${DATE}_${HOUR}_surface.grib2"
