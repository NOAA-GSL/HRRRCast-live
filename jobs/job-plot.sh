#!/bin/bash
#SBATCH --job-name=plot
#SBATCH --output=logs/plot_%j.out
#SBATCH --partition=u1-compute
#SBATCH --account=@[ACCNR]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=@[LEAD_HOUR]
#SBATCH --time=00:30:00
#SBATCH --exclusive

#set vars
init_time="@[INIT_TIME]"
lead_hour=@[LEAD_HOUR]
member=@[MEMBER]
PACKAGEROOT=@[PACKAGEROOT]
DATAROOT=@[DATAROOT]

# conda
source ${PACKAGEROOT}/etc/env.sh

echo "In plot, init_time=${init_time}, lead_hour=${lead_hour}, member=${member}"
python3 ${PACKAGEROOT}/src/plot.py ${init_time} ${lead_hour} --members ${member} --forecast_dir ${DATAROOT} --output_dir ${DATAROOT}
