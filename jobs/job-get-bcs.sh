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

year=${1:-2024}
month=${2:-05}
day=${3:-06}
hour=${4:-23}
lead_hour=${5:-18}
init_time="${year} ${month} ${day} ${hour}"

python3 src/get_bcs.py ${init_time} ${lead_hour}
