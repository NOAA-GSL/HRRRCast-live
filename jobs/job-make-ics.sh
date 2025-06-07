#!/bin/bash
#SBATCH --job-name=make_ics
#SBATCH --output=logs/make_ics_%j.out
#SBATCH --partition=u1-compute
#SBATCH --account=gsd-hpcs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00

# virutal env
#source /scratch3/BMC/gsd-hpcs/Daniel.Abdi/hrrrcast-live/env/bin/activate

# conda
source /scratch3/BMC/gsd-hpcs/Daniel.Abdi/miniconda3/etc/profile.d/conda.sh
conda activate hrrrcast-live

python3 src/make_ics.py net-deterministic/normalize.nc 2024 05 06 23
