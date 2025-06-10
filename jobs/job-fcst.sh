#!/bin/bash
#SBATCH --job-name=fcst
#SBATCH --output=logs/fcst_%j.out
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --gres=gpu:h100:1
#SBATCH --account=gsd-hpcs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:10:00

# conda
source etc/env.sh

year=${1:-2024}
month=${2:-05}
day=${3:-06}
hour=${4:-23}
lead_hour=${5:-18}
member=${6:-0}
init_time="${year} ${month} ${day} ${hour}"

python3 src/fcst.py $PWD/net-diffusion/model.keras ${init_time} ${lead_hour} ${member}
