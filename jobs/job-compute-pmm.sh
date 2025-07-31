#!/bin/bash
#SBATCH --job-name=compute_pmm
#SBATCH --output=logs/compute_pmm_%j.out
#SBATCH --partition=u1-compute
#SBATCH --account=@[ACCNR]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00

module use /contrib/spack-stack/spack-stack-1.9.1/envs/ue-oneapi-2024.2.1/install/modulefiles/Core/
module load stack-oneapi
module load wgrib2

#set vars
init_time="@[INIT_TIME]"
lead_hour=@[LEAD_HOUR]
member=@[MEMBER]
PACKAGEROOT=@[PACKAGEROOT]
DATAROOT=@[DATAROOT]

source ${PACKAGEROOT}/etc/env.sh

echo "In compute_pmm, init_time=${init_time}"
python ${PACKAGEROOT}/src/compute_pmm.py ${init_time} --forecast_dir ${DATAROOT} --output_dir ${DATAROOT}
