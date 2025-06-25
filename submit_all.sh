#!/bin/bash

INIT_TIME=${1:-"2024-05-06T23"}
LEAD_HOUR=${2:-18}
USE_DIFFUSION=${3:-1}
ACCNR=${ACCNR:-gsd-hpcs}

submit_with_check() {
    local jobid
    jobid=$(eval "$@")
    if [[ $? -ne 0 || -z "$jobid" ]]; then
        echo "Failed to submit job: $*" >&2
        exit 1
    fi
    echo "$jobid"
}

source ./atparse.bash

atparse < jobs/job-get-ics.sh > logs/job-get-ics.sh
jobid1=$(submit_with_check sbatch --parsable logs/job-get-ics.sh)
echo "Submitted job: $jobid1"

atparse < jobs/job-get-bcs.sh > logs/job-get-bcs.sh
jobid2=$(submit_with_check sbatch --parsable logs/job-get-bcs.sh)
echo "Submitted job: $jobid2"

atparse < jobs/job-make-ics.sh > logs/job-make-ics.sh
jobid3=$(submit_with_check sbatch --dependency=afterok:$jobid1 --parsable logs/job-make-ics.sh)
echo "Submitted job: $jobid3"

atparse < jobs/job-make-bcs.sh > logs/job-make-bcs.sh
jobid4=$(submit_with_check sbatch --dependency=afterok:$jobid2 --parsable logs/job-make-bcs.sh)
echo "Submitted job: $jobid4"

if [ $USE_DIFFUSION -eq 0 ]; then
    #deterministic forecast
    MEMBER=0
    atparse < jobs/job-fcst.sh > logs/job-fcst.sh
    jobid5=$(submit_with_check sbatch --dependency=afterok:$jobid3:$jobid4 --parsable logs/job-fcst.sh)
    echo "Submitted job: $jobid5"
    
    atparse < jobs/job-plot.sh > logs/job-plot.sh
    jobid6=$(submit_with_check sbatch --dependency=afterok:$jobid5 --parsable logs/job-plot.sh)
    echo "Submitted job: $jobid6"
else
    # run two ensemble members
    jobids=()
    for MEMBER in {0..2}; do
        atparse < jobs/job-fcst.sh > logs/job-fcst-${MEMBER}.sh
        jobid5=$(submit_with_check sbatch --dependency=afterok:$jobid3:$jobid4 --parsable logs/job-fcst-${MEMBER}.sh)
        jobids+=($jobid5)
        echo "Submitted job: $jobid5"
    
        atparse < jobs/job-plot.sh > logs/job-plot-${MEMBER}.sh
        jobid6=$(submit_with_check sbatch --dependency=afterok:$jobid5 --parsable logs/job-plot-${MEMBER}.sh)
        echo "Submitted job: $jobid6"
    done
    
    # ensemble PMM
    MEMBER="_mean"
    
    atparse < jobs/job-compute-pmm.sh > logs/job-compute-pmm.sh
    jobid7=$(submit_with_check sbatch --dependency=afterok:$(IFS=:; echo "${jobids[*]}") --parsable logs/job-compute-pmm.sh)
    echo "Submitted job: $jobid7"
    
    atparse < jobs/job-plot.sh > logs/job-plot-mean.sh
    jobid8=$(submit_with_check sbatch --dependency=afterok:$jobid7 --parsable logs/job-plot-mean.sh)
    echo "Submitted job: $jobid8"
fi
