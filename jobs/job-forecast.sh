#!/bin/bash
#SBATCH --job-name=forecast
#SBATCH --output=logs/forecast_%j.out
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --gres=gpu:h100:2
#SBATCH --account=gsd-hpcs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:10:00

# virutal env
#source /scratch3/BMC/gsd-hpcs/Daniel.Abdi/hrrrcast-live/env/bin/activate

# conda
source /scratch3/BMC/gsd-hpcs/Daniel.Abdi/miniconda3/etc/profile.d/conda.sh
conda activate hrrrcast-live

python3 src/forecast.py $PWD/nets/model_deterministic.keras 2024 05 06 23 18
