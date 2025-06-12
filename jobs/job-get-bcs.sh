#!/bin/bash
#SBATCH --job-name=get_bcs
#SBATCH --output=logs/get_bcs_%j.out
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
lead_hour=@[LEAD_HOUR]
year=`echo $init_time |cut -c1-4`
month=`echo $init_time |cut -c6-7`
day=`echo $init_time |cut -c9-10`
hour=`echo $init_time |cut -c12-13`
 
echo "In get_bcs, init_time=$init_time, year/month/day/hour/,${year} ${month} ${day} ${hour}, lead_hour=$lead_hour"
python3 src/get_bcs.py ${init_time} ${lead_hour}
