#!/bin/bash
#SBATCH --job-name=get_bcs
#SBATCH --output=logs/get_bcs_%j.out
#SBATCH --partition=u1-service
#SBATCH --account=@[ACCNR]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

# set vars
init_time="@[INIT_TIME]"
lead_hour=@[LEAD_HOUR]
PACKAGEROOT=@[PACKAGEROOT]
DATAROOT=@[DATAROOT]
 
# conda
source ${PACKAGEROOT}/etc/env.sh

echo "In get_bcs, init_time=${init_time}, lead_hour=${lead_hour}"
python3 ${PACKAGEROOT}/src/get_bcs.py ${init_time} ${lead_hour} --base_dir ${DATAROOT}
