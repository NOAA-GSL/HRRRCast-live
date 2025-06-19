#!/bin/bash
#SBATCH --job-name=compute_pmm
#SBATCH --output=logs/compute_pmm_%j.out
#SBATCH --partition=u1-compute
#SBATCH --account=@[ACCNR]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00

# conda
source etc/env.sh

#set vars
init_time="@[INIT_TIME]"
lead_hour=@[LEAD_HOUR]
member=@[MEMBER]
year=`echo $init_time |cut -c1-4`
month=`echo $init_time |cut -c6-7`
day=`echo $init_time |cut -c9-10`
hour=`echo $init_time |cut -c12-13`

echo "In compute_pmm, init_time=${init_time}, year/month/day/hour/=${year} ${month} ${day} ${hour}"
python3 src/compute_pmm.py ${init_time}
