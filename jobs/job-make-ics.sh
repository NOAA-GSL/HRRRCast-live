#!/bin/bash
#SBATCH --job-name=make_ics
#SBATCH --output=logs/make_ics_%j.out
#SBATCH --partition=u1-compute
#SBATCH --account=@[ACCNR]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00

# set vars
INIT_TIME="@[INIT_TIME]"
PACKAGEROOT=@[PACKAGEROOT]
DATAROOT=@[DATAROOT]

# conda
source ${PACKAGEROOT}/etc/env.sh

# job
echo "In make_ics, init_time=${INIT_TIME}"
python3 ${PACKAGEROOT}/src/make_ics.py ${PACKAGEROOT}/net-diffusion/normalize.nc ${INIT_TIME} --base_dir ${DATAROOT} --output_dir ${DATAROOT}
