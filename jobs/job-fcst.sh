#!/bin/bash
#SBATCH --job-name=fcst
#SBATCH --output=logs/fcst_%j.out
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --gres=gpu:h100:1
#SBATCH --account=@[ACCNR]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:30:00

module use /contrib/spack-stack/spack-stack-1.9.1/envs/ue-oneapi-2024.2.1/install/modulefiles/Core/
module load stack-oneapi
module load wgrib2

# set vars
init_time="@[INIT_TIME]"
lead_hour=@[LEAD_HOUR]
member=@[MEMBER]
use_diffusion=@[USE_DIFFUSION]
PACKAGEROOT=@[PACKAGEROOT]
DATAROOT=@[DATAROOT]
ENVMODE=@[ENVMODE]

# conda
if [ "$ENVMODE" == "OPN" ]; then
    source ${PACKAGEROOT}/etc/env_emc.sh
else
    source ${PACKAGEROOT}/etc/env.sh
fi

echo "In fcst, init_time=${init_time}, lead_hour=${lead_hour}, member=${member}, use_diffusion=${use_diffusion}, base_dir=${DATAROOT}"
if [ $use_diffusion -eq 0 ]; then
    python ${PACKAGEROOT}/src/fcst.py $PACKAGEROOT/net-deterministic/model.keras ${init_time} ${lead_hour} --members ${member} --no_diffusion --base_dir ${DATAROOT} --output_dir ${DATAROOT}
else
    python ${PACKAGEROOT}/src/fcst.py $PACKAGEROOT/net-diffusion/model.keras ${init_time} ${lead_hour} --members ${member} --base_dir ${DATAROOT} --output_dir ${DATAROOT}
fi
