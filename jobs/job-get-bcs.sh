#!/bin/bash
#SBATCH --job-name=get_bcs
#SBATCH --output=logs/get_bcs_%j.out
#SBATCH --partition=u1-service
#SBATCH --account=gsd-hpcs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

# virutal env
#source /scratch3/BMC/gsd-hpcs/Daniel.Abdi/hrrrcast-live/env/bin/activate

# conda
source /scratch3/BMC/gsd-hpcs/Daniel.Abdi/miniconda3/etc/profile.d/conda.sh
conda activate hrrrcast-live

python3 src/get_bcs.py 2024 05 06 23 18
