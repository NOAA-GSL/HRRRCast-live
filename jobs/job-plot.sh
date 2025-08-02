#!/bin/bash
#SBATCH --job-name=plot
#SBATCH --output=logs/plot_%j.out
#SBATCH --partition=u1-compute
#SBATCH --account=@[ACCNR]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=@[LEAD_HOUR]
#SBATCH --time=00:30:00
#SBATCH --exclusive

# set vars
INIT_TIME="@[INIT_TIME]"
LEAD_HOUR=@[LEAD_HOUR]
MEMBER=@[MEMBER]
PACKAGEROOT=@[PACKAGEROOT]
DATAROOT=@[DATAROOT]

# conda
source ${PACKAGEROOT}/etc/env.sh

# job
echo "In plot, init_time=${INIT_TIME}, lead_hour=${LEAD_HOUR}, member=${MEMBER}"
python3 ${PACKAGEROOT}/src/plot.py ${INIT_TIME} ${LEAD_HOUR} --members ${MEMBER} --forecast_dir ${DATAROOT} --output_dir ${DATAROOT}
