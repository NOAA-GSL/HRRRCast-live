#!/bin/bash
#SBATCH --job-name=get_ics
#SBATCH --output=logs/get_ics_%j.out
#SBATCH --partition=u1-service
#SBATCH --account=@[ACCNR]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

# set vars
init_time="@[INIT_TIME]"
PACKAGEROOT="@[PACKAGEROOT]"
DATAROOT="@[DATAROOT]"

# conda
source ${PACKAGEROOT}/etc/env.sh

echo "In get_ics, init_time=${init_time} "
python3 ${PACKAGEROOT}/src/get_ics.py ${init_time} --base_dir ${DATAROOT}
