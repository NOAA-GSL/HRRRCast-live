#!/bin/bash

set -x

INIT_TIME=${1:-"2024-07-17T23"}
LEAD_HOUR=${2:-18}
N_ENSEMBLES=${3:-1}
N_GPUS=${4:-1}
PACKAGEROOT=${5:-`pwd`}
DATAROOT=${6:-`pwd`}
RUNPLOT=${7:-"YES"}
ENVMODE=${8:-``}
ACCNR=${ACCNR:-gsd-hpcs}

# set wall clock time limits
hr=$(echo "$INIT_TIME" | grep -oP '\d{2}$')
if [[ "$hr" =~ ^(00|06|12|18)$ ]]; then
    FCST_WALLTIME="02:30:00"
    PMM_WALLTIME="02:50:00"
    GET_BCS_WALLTIME="00:30:00"
    MAKE_BCS_WALLTIME="01:00:00"
else
    FCST_WALLTIME="01:00:00"
    PMM_WALLTIME="01:20:00"
    GET_BCS_WALLTIME="00:15:00"
    MAKE_BCS_WALLTIME="00:30:00"
fi

GET_ICS_WALLTIME="00:10:00"
MAKE_ICS_WALLTIME="00:10:00"
PLOT_WALLTIME="00:30:00"

# set environment variables
PMM_POLL_SECONDS="60"
PMM_MIN_AGE_SECONDS="90"
NETCDF2GRIB_SECTION3=
WGRIB2="wgrib2"

# submit job and check for failures
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
if [ ! -d "$DATAROOT/logs" ]; then
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

# submit forecasts as a job array over GPU slots; member range computed in job-fcst.sh
atparse < $PACKAGEROOT/jobs/job-fcst.sh > $DATAROOT/logs/job-fcst.sh
jobid5=$(submit_with_check sbatch --dependency=afterok:$jobid3:$jobid4 --array=0-$((N_GPUS-1)) --wait-all-nodes=1 --parsable $DATAROOT/logs/job-fcst.sh)
echo "Submitted forecast job array: $jobid5"

# submit plots as job array
if [ "$RUNPLOT" == "YES" ]; then
    atparse < $PACKAGEROOT/jobs/job-plot.sh > $DATAROOT/logs/job-plot.sh
    jobid6=$(submit_with_check sbatch --dependency=afterok:$jobid5 --array=0-$((N_GPUS-1)) --wait-all-nodes=1 --parsable $DATAROOT/logs/job-plot.sh)
    echo "Submitted plot job array: $jobid6"
fi

# ensemble PMM
if [ $N_ENSEMBLES -ge 2 ]; then
    atparse < $PACKAGEROOT/jobs/job-compute-pmm.sh > $DATAROOT/logs/job-compute-pmm.sh
    jobid7=$(submit_with_check sbatch --dependency=after:$jobid5 --parsable $DATAROOT/logs/job-compute-pmm.sh)
    echo "Submitted job: $jobid7"

    if [ "$RUNPLOT" == "YES" ]; then
        atparse < $PACKAGEROOT/jobs/job-plot.sh > $DATAROOT/logs/job-plot-pmm.sh
        jobid8=$(submit_with_check sbatch --dependency=afterok:$jobid7 --parsable $DATAROOT/logs/job-plot-pmm.sh)
        echo "Submitted job: $jobid8"
    fi
fi
