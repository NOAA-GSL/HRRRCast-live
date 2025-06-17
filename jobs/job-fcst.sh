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

# conda
source etc/env.sh

# set vars
init_time="@[INIT_TIME]"
lead_hour=@[LEAD_HOUR]
member=@[MEMBER]
year=`echo $init_time |cut -c1-4`
month=`echo $init_time |cut -c6-7`
day=`echo $init_time |cut -c9-10`
hour=`echo $init_time |cut -c12-13`

echo "In fcst, init_time=${init_time}, year/month/day/hour/=${year} ${month} ${day} ${hour}, lead_hour=${lead_hour}, member=${member}"
python3 src/fcst.py $PWD/net-diffusion/model.keras ${init_time} ${lead_hour} ${member}
