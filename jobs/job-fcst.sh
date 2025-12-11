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
#SBATCH --time=@[FCST_WALLTIME]

# load wgrib2 modules
module use /contrib/spack-stack/spack-stack-1.9.1/envs/ue-oneapi-2024.2.1/install/modulefiles/Core/
module load stack-oneapi
module load wgrib2

# set vars
INIT_TIME="@[INIT_TIME]"
LEAD_HOUR=@[LEAD_HOUR]
MEMBER=@[MEMBER]
PACKAGEROOT=@[PACKAGEROOT]
DATAROOT=@[DATAROOT]
ENVMODE=@[ENVMODE]

# conda
if [ "$ENVMODE" == "OPN" ]; then
    source ${PACKAGEROOT}/etc/env_emc.sh
else
    source ${PACKAGEROOT}/etc/env.sh
fi

# job
echo "In fcst, INIT_TIME=${INIT_TIME}, LEAD_HOUR=${LEAD_HOUR}, MEMBER=${MEMBER}, base_dir=${DATAROOT}"
python ${PACKAGEROOT}/src/fcst.py $PACKAGEROOT/net-diffusion/model.keras ${INIT_TIME} ${LEAD_HOUR} \
    --members ${MEMBER} --base_dir ${DATAROOT} --output_dir ${DATAROOT}
