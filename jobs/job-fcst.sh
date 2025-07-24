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
#SBATCH --time=00:10:00

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
year=`echo $init_time |cut -c1-4`
month=`echo $init_time |cut -c6-7`
day=`echo $init_time |cut -c9-10`
hour=`echo $init_time |cut -c12-13`

# conda
current_user=$(whoami)
if [[ "$current_user" == "Jun.Wang" || "$current_user" == "Linlin.Cui" ]]; then
    source ${PACKAGEROOT}/etc/env_emc.sh
else
    source ${PACKAGEROOT}/etc/env.sh
fi

echo "In fcst, init_time=${init_time}, year/month/day/hour/=${year} ${month} ${day} ${hour}, lead_hour=${lead_hour}, member=${member}, use_diffusion=${use_diffusion}, base_dir=${DATAROOT}"
if [ $use_diffusion -eq 0 ]; then
    python ${PACKAGEROOT}/src/fcst.py $PACKAGEROOT/net-deterministic/model.keras ${year} ${month} ${day} ${hour} ${lead_hour} ${member} --no_diffusion --base_dir ${DATAROOT} --output_dir ${DATAROOT}
else
    python ${PACKAGEROOT}/src/fcst.py $PACKAGEROOT/net-diffusion/model.keras ${year} ${month} ${day} ${hour} ${lead_hour} ${member} --base_dir ${DATAROOT} --output_dir ${DATAROOT}
fi
