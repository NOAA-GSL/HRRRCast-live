#!/bin/bash
#SBATCH --job-name=get_ics
#SBATCH --output=logs/get_ics_%j.out
#SBATCH --partition=u1-service
#SBATCH --account=gsd-hpcs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

# conda
source etc/env.sh

year=${1:-2024}
month=${2:-05}
day=${3:-06}
hour=${4:-23}
init_time="${year} ${month} ${day} ${hour}"

python3 src/get_ics.py ${init_time}
