#!/bin/bash
#SBATCH --job-name=get_ics
#SBATCH --output=logs/get_ics_%j.out
#SBATCH --partition=u1-service
#SBATCH --account=@[ACCNR]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

# conda
source etc/env.sh

# set vars
init_time="@[INIT_TIME]"
year=`echo $init_time |cut -c1-4`
month=`echo $init_time |cut -c6-7`
day=`echo $init_time |cut -c9-10`
hour=`echo $init_time |cut -c12-13`

echo "In get_ics, init_time=${init_time}, year/month/day/hour/,${year} ${month} ${day} ${hour}"
python3 src/get_ics.py ${init_time}
