#!/bin/bash
#SBATCH --job-name=make_ics
#SBATCH --output=logs/make_ics_%j.out
#SBATCH --partition=u1-compute
#SBATCH --account=@[ACCNR]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00

# conda
source etc/env.sh

# set vars
init_time="@[INIT_TIME]"
year=`echo $init_time |cut -c1-4`
month=`echo $init_time |cut -c6-7`
day=`echo $init_time |cut -c9-10`
hour=`echo $init_time |cut -c12-13`

# make ics
echo "In make_ics, init_time=${init_time}, year/month/day/hour/,${year} ${month} ${day} ${hour}"
python3 src/make_ics.py net-diffusion/normalize.nc ${init_time}
