#!/bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=logs/cleanup_%j.out
#SBATCH --partition=u1-service
#SBATCH --account=@[ACCNR]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --mem=4G

# set vars
INIT_TIME="@[INIT_TIME]"
DATAROOT="@[DATAROOT]"
PRESERVE_PATTERN="hrrrcast.*.pgrb2*"

# Parse INIT_TIME to extract YYYYMMDD and HH
# Expected format: YYYY-MM-DDTHH (e.g., "2024-07-17T23")
YYYYMMDD=$(echo "$INIT_TIME" | cut -d'T' -f1 | tr -d '-')
HH=$(echo "$INIT_TIME" | cut -d'T' -f2)

# Construct cleanup directory
CLEANUP_DIR="${DATAROOT}/${YYYYMMDD}/${HH}"

echo "=================================================="
echo "Starting cleanup"
echo "INIT_TIME: ${INIT_TIME}"
echo "DATAROOT: ${DATAROOT}"
echo "Cleanup directory: ${CLEANUP_DIR}"
echo "Date: $(date)"
echo "=================================================="

# Check if cleanup directory exists
if [ ! -d "${CLEANUP_DIR}" ]; then
    echo "WARNING: Cleanup directory does not exist: ${CLEANUP_DIR}"
    echo "Nothing to clean up."
    exit 0
fi

# Safety check - ensure CLEANUP_DIR is not root or home
if [ "${CLEANUP_DIR}" == "/" ] || [ "${CLEANUP_DIR}" == "${HOME}" ]; then
    echo "ERROR: Cleanup directory cannot be root or home directory!"
    exit 1
fi

echo "Scanning directory: ${CLEANUP_DIR}"
echo ""

# Count files before cleanup
total_files=$(find "${CLEANUP_DIR}" -type f | wc -l)
preserve_files=$(find "${CLEANUP_DIR}" -type f -name "${PRESERVE_PATTERN}" | wc -l)
echo "Total files found: ${total_files}"
echo "Files to preserve (${PRESERVE_PATTERN}): ${preserve_files}"
echo "Files to delete: $((total_files - preserve_files))"
echo ""

echo "Starting cleanup..."
echo ""

# Find and delete all files except ${PRESERVE_PATTERN} files
deleted_count=0
error_count=0

while IFS= read -r -d '' file; do
    if rm -f "${file}"; then
        deleted_count=$((deleted_count + 1))
        if [ $((deleted_count % 100)) -eq 0 ]; then
            echo "Deleted ${deleted_count} files..."
        fi
    else
        echo "ERROR: Failed to delete ${file}"
        error_count=$((error_count + 1))
    fi
done < <(find "${CLEANUP_DIR}" -type f ! -name "${PRESERVE_PATTERN}" -print0)

echo ""
echo "=================================================="
echo "Cleanup completed!"
echo "Files deleted: ${deleted_count}"
echo "Errors encountered: ${error_count}"
echo "Files preserved: ${preserve_files}"
echo "Date: $(date)"
echo "=================================================="
