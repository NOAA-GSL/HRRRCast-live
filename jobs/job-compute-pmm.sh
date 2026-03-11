#!/bin/bash
#SBATCH --job-name=compute_pmm
#SBATCH --output=logs/compute_pmm_%j.out
#SBATCH --partition=u1-compute
#SBATCH --account=@[ACCNR]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=@[PMM_WALLTIME]
#SBATCH --mem=128G

# load wgrib2 modules
module use /contrib/spack-stack/spack-stack-1.9.1/envs/ue-oneapi-2024.2.1/install/modulefiles/Core/
module load stack-oneapi
module load wgrib2

# set vars
INIT_TIME="@[INIT_TIME]"
PACKAGEROOT=@[PACKAGEROOT]
DATAROOT=@[DATAROOT]
LEAD_HOUR=@[LEAD_HOUR]
N_ENSEMBLES=@[N_ENSEMBLES]

export PMM_POLL_SECONDS=@[PMM_POLL_SECONDS]
export PMM_MIN_AGE_SECONDS=@[PMM_MIN_AGE_SECONDS]
export NETCDF2GRIB_SECTION3=@[NETCDF2GRIB_SECTION3]
export WGRIB2=@[WGRIB2]

# conda
source ${PACKAGEROOT}/etc/env.sh

# job
echo "In compute_pmm, init_time=${INIT_TIME}, lead_hour=${LEAD_HOUR}, n_ensembles=${N_ENSEMBLES}"
python ${PACKAGEROOT}/src/compute_pmm.py ${INIT_TIME} ${LEAD_HOUR} --forecast_dir ${DATAROOT} --output_dir ${DATAROOT} --n_ensembles ${N_ENSEMBLES}
