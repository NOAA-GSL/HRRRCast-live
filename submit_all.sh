#!/bin/bash

set -x

INIT_TIME=${1:-"2024-07-17T23"}
LEAD_HOUR=${2:-18}
USE_DIFFUSION=${3:-1}
N_ENSEMBLES=${4:-1}
N_GPUS=${5:-1}
ACCNR=${ACCNR:-gsd-hpcs}
PACKAGEROOT=${6:-`pwd`}
DATAROOT=${7:-`pwd`}
ENVMODE=${8:-``}

submit_with_check() {
    local jobid
    jobid=$(eval "$@")
    if [[ $? -ne 0 || -z "$jobid" ]]; then
        echo "Failed to submit job: $*" >&2
        exit 1
    fi
    echo "$jobid"
}

get_ranges() {
    local N=$1     # number of ensembles
    local Ng=$2    # number of GPUs

    local chunk=$(( N / Ng ))
    local rem=$(( N % Ng ))
    local start=0

    for (( i=0; i<Ng; i++ )); do
        local extra=0
        if (( i < rem )); then
            extra=1
        fi

        local end=$(( start + chunk + extra - 1 ))

        echo "$start-$end"

        start=$(( end + 1 ))
    done
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
    for MEMBER in $(get_ranges $N_ENSEMBLES $N_GPUS); do
        atparse < $PACKAGEROOT/jobs/job-fcst.sh > $DATAROOT/logs/job-fcst-${MEMBER}.sh
        jobid5=$(submit_with_check sbatch --dependency=afterok:$jobid3:$jobid4 --parsable $DATAROOT/logs/job-fcst-${MEMBER}.sh)
        jobids+=($jobid5)
        echo "Submitted job: $jobid5"

        atparse < $PACKAGEROOT/jobs/job-plot.sh > $DATAROOT/logs/job-plot-${MEMBER}.sh
        jobid6=$(submit_with_check sbatch --dependency=afterok:$jobid5 --parsable $DATAROOT/logs/job-plot-${MEMBER}.sh)
        echo "Submitted job: $jobid6"
    done
    
    # ensemble PMM
    if [ $N_ENSEMBLES -ge 2 ]; then
        MEMBER="avg"
    
        atparse < $PACKAGEROOT/jobs/job-compute-pmm.sh > $DATAROOT/logs/job-compute-pmm.sh
        jobid7=$(submit_with_check sbatch --dependency=afterok:$(IFS=:; echo "${jobids[*]}") --parsable $DATAROOT/logs/job-compute-pmm.sh)
        echo "Submitted job: $jobid7"
    
        atparse < $PACKAGEROOT/jobs/job-plot.sh > $DATAROOT/logs/job-plot-mean.sh
        jobid8=$(submit_with_check sbatch --dependency=afterok:$jobid7 --parsable $DATAROOT/logs/job-plot-mean.sh)
        echo "Submitted job: $jobid8"
    fi
fi
