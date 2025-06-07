#!/bin/bash

year=${1:-2024}
month=${2:-05}
day=${3:-06}
hour=${4:-23}
lead_hour=${5:-18}
init_time="${year} ${month} ${day} ${hour}"

submit_with_check() {
    local jobid
    jobid=$(eval "$@")
    if [[ $? -ne 0 || -z "$jobid" ]]; then
        echo "Failed to submit job: $*" >&2
        exit 1
    fi
    echo "$jobid"
}

jobid1=$(submit_with_check sbatch --parsable jobs/job-get-ics.sh ${init_time})
echo "Submitted job 1: $jobid1"

jobid2=$(submit_with_check sbatch --parsable jobs/job-get-bcs.sh ${init_time} ${lead_hour})
echo "Submitted job 2: $jobid2"

jobid3=$(submit_with_check sbatch --dependency=afterany:$jobid1 --parsable jobs/job-make-ics.sh ${init_time} ${lead_hour})
echo "Submitted job 3: $jobid3"

jobid4=$(submit_with_check sbatch --dependency=afterany:$jobid3 --parsable jobs/job-fcst.sh ${init_time} ${lead_hour})
echo "Submitted job 4: $jobid4"

jobid5=$(submit_with_check sbatch --dependency=afterany:$jobid4 --parsable jobs/job-plot.sh ${init_time} ${lead_hour})
echo "Submitted job 5: $jobid5"
