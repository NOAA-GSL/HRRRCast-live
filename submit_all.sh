#!/bin/bash

set -x

INIT_TIME=${1:-"2024-07-17T23"}
LEAD_HOUR=${2:-18}
USE_DIFFUSION=${3:-1}
ACCNR=${ACCNR:-gsd-hpcs}
PACKAGEROOT=${4:-`pwd`}
DATAROOT=${5:-`pwd`}

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
if [ ! -d "$DIRECTORY" ]; then
    mkdir -p $DATAROOT/logs
fi
cd $DATAROOT

echo "PACKAGEROOT=$PACKAGEROOT,DATAROOT=$DATAROOT"

atparse < $PACKAGEROOT/jobs/job-get-ics.sh > $DATAROOT/logs/job-get-ics.sh
jobid1=$(submit_with_check sbatch --parsable $DATAROOT/logs/job-get-ics.sh)
echo "Submitted job: $jobid1"

atparse < $PACKAGEROOT/jobs/job-get-bcs.sh > $DATAROOT/logs/job-get-bcs.sh
jobid2=$(submit_with_check sbatch --parsable $DATAROOT/logs/job-get-bcs.sh)
echo "Submitted job: $jobid2"

atparse < $PACKAGEROOT/jobs/job-make-ics.sh > $DATAROOT/logs/job-make-ics.sh
jobid3=$(submit_with_check sbatch --dependency=afterok:$jobid1 --parsable $DATAROOT/logs/job-make-ics.sh)
echo "Submitted job: $jobid3"

atparse < $PACKAGEROOT/jobs/job-make-bcs.sh > $DATAROOT/logs/job-make-bcs.sh
jobid4=$(submit_with_check sbatch --dependency=afterok:$jobid2 --parsable $DATAROOT/logs/job-make-bcs.sh)
echo "Submitted job: $jobid4"

if [ $USE_DIFFUSION -eq 0 ]; then
    #deterministic forecast
    MEMBER=0
    atparse < $PACKAGEROOT/jobs/job-fcst.sh > $DATAROOT/logs/job-fcst.sh
    jobid5=$(submit_with_check sbatch --dependency=afterok:$jobid3:$jobid4 --parsable $DATAROOT/logs/job-fcst.sh)
    echo "Submitted job: $jobid5"
    
    atparse < $PACKAGEROOT/jobs/job-plot.sh > $DATAROOT/logs/job-plot.sh
    jobid6=$(submit_with_check sbatch --dependency=afterok:$jobid5 --parsable $DATAROOT/logs/job-plot.sh)
    echo "Submitted job: $jobid6"
else
    # run two ensemble members
    jobids=()
    for MEMBER in {0..2}; do
        atparse < $PACKAGEROOT/jobs/job-fcst.sh > $DATAROOT/logs/job-fcst-${MEMBER}.sh
        jobid5=$(submit_with_check sbatch --dependency=afterok:$jobid3:$jobid4 --parsable $DATAROOT/logs/job-fcst-${MEMBER}.sh)
        jobids+=($jobid5)
        echo "Submitted job: $jobid5"

        atparse < $PACKAGEROOT/jobs/job-plot.sh > $DATAROOT/logs/job-plot-${MEMBER}.sh
        jobid6=$(submit_with_check sbatch --dependency=afterok:$jobid5 --parsable $DATAROOT/logs/job-plot-${MEMBER}.sh)
        echo "Submitted job: $jobid6"
    done
    
    # ensemble PMM
    MEMBER="avg"
    
    atparse < $PACKAGEROOT/jobs/job-compute-pmm.sh > $DATAROOT/logs/job-compute-pmm.sh
    jobid7=$(submit_with_check sbatch --dependency=afterok:$(IFS=:; echo "${jobids[*]}") --parsable $DATAROOT/logs/job-compute-pmm.sh)
    echo "Submitted job: $jobid7"
    
    atparse < $PACKAGEROOT/jobs/job-plot.sh > $DATAROOT/logs/job-plot-mean.sh
    jobid8=$(submit_with_check sbatch --dependency=afterok:$jobid7 --parsable $DATAROOT/logs/job-plot-mean.sh)
    echo "Submitted job: $jobid8"
fi
