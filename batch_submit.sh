#!/bin/bash
# =============================================================================
# batch_submit.sh — Submit HRRRCast forecast jobs over a date range, in batches
#
# Usage:
#   ./batch_submit.sh
#   (edit the CONFIGURATION block below before running)
#
# The script submits up to BATCH_SIZE forecast pipelines at a time.
# After all forecast jobs in a batch finish, it runs cleanup directly (rm)
# before submitting the next batch. This prevents storage overflow.
#
# Cleanup keeps only:  hrrrcast_mem*_f<LL>.nc  (LL = LEAD_HOURS, zero-padded)
# and removes everything else in each run directory.
# =============================================================================

# =============================================================================
#  CONFIGURATION — edit these settings before running
# =============================================================================

START_DATE="2025-12-01T00"   # First init time  (YYYY-MM-DDTHH, UTC)
END_DATE="2026-03-31T23"     # Last  init time  (YYYY-MM-DDTHH, UTC, inclusive)
INTERVAL_HOURS=6             # Hours between consecutive forecast init times

LEAD_HOURS=6                 # Forecast lead time (hours)
N_ENSEMBLES=1                # Number of ensemble members
N_GPUS=2                     # Number of GPUs (also controls job array size)

ACCNR="gpu-ghpcs"            # SLURM account name
PACKAGEROOT="$(cd "$(dirname "$0")" && pwd)"  # Repo root (auto-detected from script location)
DATAROOT="$(pwd)"                             # Data/output root (default: current directory)

CLEANUP="YES"                # YES = remove intermediate files after each batch
                             # NO  = keep all files

BATCH_SIZE=10                # Number of forecast runs to submit before waiting
                             # for the batch to complete (storage management)

POLL_INTERVAL=60             # Seconds between checks when waiting for a batch

# =============================================================================
#  DO NOT EDIT BELOW THIS LINE (unless you know what you're doing)
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

to_epoch() {
    local dt="${1/T/ }"
    date -u -d "${dt}:00:00 UTC" +%s
}

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }

# ---------------------------------------------------------------------------
# Validate configuration
# ---------------------------------------------------------------------------

if [ ! -f "$PACKAGEROOT/submit_all.sh" ]; then
    echo "ERROR: submit_all.sh not found under PACKAGEROOT=$PACKAGEROOT"
    exit 1
fi

mkdir -p "$DATAROOT/logs"

START_EPOCH=$(to_epoch "$START_DATE")
END_EPOCH=$(to_epoch "$END_DATE")

if [ "$START_EPOCH" -gt "$END_EPOCH" ]; then
    echo "ERROR: START_DATE ($START_DATE) is after END_DATE ($END_DATE)"
    exit 1
fi

# Last forecast file index to keep (e.g. LEAD_HOURS=6 → LAST_F=06)
# f00 = initial state, f01..fNN = forecast hours, so last file = fLEAD_HOURS
LAST_F=$(printf '%02d' $((LEAD_HOURS)))

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

log "=========================================="
log "HRRRCast batch submission"
log "  Start date   : $START_DATE"
log "  End date     : $END_DATE"
log "  Interval     : ${INTERVAL_HOURS}h"
log "  Lead hours   : ${LEAD_HOURS}h  (keeping hrrrcast_mem*_f${LAST_F}.nc)"
log "  Ensemble mem : $N_ENSEMBLES"
log "  GPUs         : $N_GPUS"
log "  SLURM account: $ACCNR"
log "  PACKAGEROOT  : $PACKAGEROOT"
log "  DATAROOT     : $DATAROOT"
log "  Cleanup      : $CLEANUP"
log "  Batch size   : $BATCH_SIZE runs"
log "  Poll interval: ${POLL_INTERVAL}s"
log "=========================================="
echo ""

# ---------------------------------------------------------------------------
# wait_for_batch: poll squeue until all given job IDs have left the queue.
# ---------------------------------------------------------------------------

wait_for_batch() {
    local job_ids=("$@")
    if [ ${#job_ids[@]} -eq 0 ]; then return; fi

    local dep
    dep=$(IFS=:; echo "${job_ids[*]}")
    log "Waiting for ${#job_ids[@]} forecast job(s) to finish: $dep"

    local remaining
    while true; do
        remaining=0
        for jid in "${job_ids[@]}"; do
            if squeue -j "$jid" -h 2>/dev/null | grep -q .; then
                ((remaining++)) || true
            fi
        done
        [ "$remaining" -eq 0 ] && break
        log "  Still waiting — $remaining job(s) active ..."
        sleep "$POLL_INTERVAL"
    done

    log "  All jobs finished."
}

# ---------------------------------------------------------------------------
# cleanup_batch: remove intermediate files for a list of run directories,
# keeping only hrrrcast_mem*_f${LAST_F}.nc in each.
# ---------------------------------------------------------------------------

cleanup_batch() {
    local run_dirs=("$@")
    if [ ${#run_dirs[@]} -eq 0 ] || [ "$CLEANUP" != "YES" ]; then return; fi

    log "Running cleanup for ${#run_dirs[@]} run director(ies) ..."
    for run_dir in "${run_dirs[@]}"; do
        if [ ! -d "$run_dir" ]; then
            log "  WARNING: directory not found, skipping: $run_dir"
            continue
        fi

        log "  Cleaning $run_dir (keeping hrrrcast_mem*_f${LAST_F}.nc) ..."

        # Remove everything except the final forecast NetCDF
        find "$run_dir" -maxdepth 1 -type f ! -name "hrrrcast_mem*_f${LAST_F}.nc" -print \
            | while IFS= read -r f; do
                rm -f "$f" && log "    removed: $(basename "$f")"
            done

        log "  Kept: $(ls "$run_dir")"
    done
    log "Cleanup done."
}

# ---------------------------------------------------------------------------
# submit_one_run: submit the full pipeline for a single INIT_TIME.
# Sets globals FCST_JOBID and RUN_DIR (empty on failure).
# ---------------------------------------------------------------------------

submit_one_run() {
    local init_time=$1
    local date_part hour_part
    date_part=$(date -u -d "@$(to_epoch "$init_time")" +"%Y%m%d")
    hour_part=$(date -u -d "@$(to_epoch "$init_time")" +"%H")

    FCST_JOBID=""
    RUN_DIR="${DATAROOT}/${date_part}/${hour_part}"

    log "Submitting $init_time ..."

    # Call submit_all.sh; capture stdout for job IDs, redirect stderr to log
    local submit_log="${DATAROOT}/logs/submit_${date_part}_${hour_part}.log"
    local submit_out
    submit_out=$(
        cd "$PACKAGEROOT"
        ACCNR="$ACCNR" bash submit_all.sh \
            "$init_time" "$LEAD_HOURS" "$N_ENSEMBLES" "$N_GPUS" \
            "$PACKAGEROOT" "$DATAROOT" \
            2>"$submit_log"
    ) || {
        log "  WARNING: submit_all.sh failed for $init_time — see $submit_log"
        return 1
    }

    FCST_JOBID=$(echo "$submit_out" | grep "Submitted forecast job array:" | awk '{print $NF}')

    if [ -z "$FCST_JOBID" ]; then
        log "  WARNING: could not parse forecast job ID for $init_time — see $submit_log"
        return 1
    fi

    log "  -> forecast job : $FCST_JOBID  (output dir: $RUN_DIR)"
    return 0
}

# ---------------------------------------------------------------------------
# Main loop — submit in batches of BATCH_SIZE
# ---------------------------------------------------------------------------

TOTAL_SUBMITTED=0
TOTAL_FAILED=0
BATCH_NUM=0
CURRENT=$START_EPOCH

while [ "$CURRENT" -le "$END_EPOCH" ]; do

    ((BATCH_NUM++)) || true
    log "--- Starting batch $BATCH_NUM ---"

    BATCH_FCST_IDS=()   # forecast job IDs to wait on
    BATCH_RUN_DIRS=()   # run directories to clean up after the batch
    BATCH_COUNT=0

    # Submit up to BATCH_SIZE runs
    while [ "$CURRENT" -le "$END_EPOCH" ] && [ "$BATCH_COUNT" -lt "$BATCH_SIZE" ]; do

        INIT_TIME=$(date -u -d "@$CURRENT" +"%Y-%m-%dT%H")

        if submit_one_run "$INIT_TIME"; then
            ((TOTAL_SUBMITTED++)) || true
            ((BATCH_COUNT++)) || true
            BATCH_FCST_IDS+=("$FCST_JOBID")
            BATCH_RUN_DIRS+=("$RUN_DIR")
        else
            ((TOTAL_FAILED++)) || true
        fi

        CURRENT=$((CURRENT + INTERVAL_HOURS * 3600))
    done

    log "Batch $BATCH_NUM: submitted $BATCH_COUNT run(s)."

    # Wait for all forecast jobs, then clean up — always, including the last batch
    if [ ${#BATCH_FCST_IDS[@]} -gt 0 ]; then
        wait_for_batch "${BATCH_FCST_IDS[@]}"
        cleanup_batch "${BATCH_RUN_DIRS[@]}"
    fi

    echo ""
done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

log "=========================================="
log "All batches complete"
log "  Total submitted : $TOTAL_SUBMITTED run(s) in $BATCH_NUM batch(es)"
log "  Total failed    : $TOTAL_FAILED run(s)"
log "  Log dir         : $DATAROOT/logs/"
log "=========================================="
