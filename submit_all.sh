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
echo "Submitted job: $jobid1"

jobid2=$(submit_with_check sbatch --parsable jobs/job-get-bcs.sh ${init_time} ${lead_hour})
echo "Submitted job: $jobid2"

jobid3=$(submit_with_check sbatch --dependency=afterany:$jobid1 --parsable jobs/job-make-ics.sh ${init_time})
echo "Submitted job: $jobid3"

jobid4=$(submit_with_check sbatch --dependency=afterany:$jobid2 --parsable jobs/job-make-bcs.sh ${init_time} ${lead_hour})
echo "Submitted job: $jobid4"

# run two ensemble members
for member in {0..1}; do
    jobid5=$(submit_with_check sbatch --dependency=afterany:$jobid3:$jobid4 --parsable jobs/job-fcst.sh ${init_time} ${lead_hour} ${member})
    echo "Submitted job: $jobid5"

    jobid6=$(submit_with_check sbatch --dependency=afterany:$jobid5 --parsable jobs/job-plot.sh ${init_time} ${lead_hour} ${member})
    echo "Submitted job: $jobid6"
done
