#!/bin/bash

# Script to process HDF5 files in parallel with limited concurrency
# Usage: ./process_hdf5_parallel.sh [OPTIONS]

set -e

# Default values
MAX_JOBS=20
DATA_FOLDER=""
RECOMPUTE_AERO=false
RECOMPUTE_TABLE=false
RECOMPUTE_RACKET=false
USE_GT200=false
EXPORT_CSV=false
ZERO_TOSS_SPIN=false
AGGREGATE_ONLY=false
CSV_FILE=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPDATE_SCRIPT="${SCRIPT_DIR}/update_hdf5.py"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    cat << EOF
Usage: $0 -d DATA_FOLDER [OPTIONS]

Process HDF5 files in parallel with limited concurrency.

Required Arguments:
    -d, --data-folder DIR     Root directory to search for data.h5 files

Optional Arguments:
    -j, --jobs N              Maximum number of parallel jobs (default: 20)
    -a, --recompute-aero      Recompute aerodynamics (default: false)
    -t, --recompute-table     Recompute table contacts (default: false)
    -r, --recompute-racket    Recompute racket contacts (default: false)
    -g, --use-gt200           Use gt200 positions instead of APS for aerodynamics (default: false)
    -e, --export-csv          Also write extracted.csv / extracted_TCM.csv (default: false)
    -z, --zero-toss-spin      Set pre-contact spin to 0 for ball tosses (default: false)
    -A, --aggregate-only      Only aggregate existing per-folder CSVs, skip all processing
    -c, --csv-file FILE       CSV file (location,date,experiment,...) to restrict which data.h5
                              files are processed (same format as copy_h5_to_local.py)
    -h, --help                Show this help message

Example:
    $0 -d data/samples/test_inference -j 10 -a
    $0 -d data/samples/test_inference -j 10 -a -g   # with gt200 positions
    $0 -d data/samples/test_inference -j 10 -a -r -e # recompute aero + racket, export CSVs
    $0 -d data/samples/test_inference -j 10 -a -c filter.csv  # process only CSV-listed experiments
    $0 -d data/samples/test_inference -A              # only merge existing extracted*.csv files
EOF
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-folder)
            DATA_FOLDER="$2"
            shift 2
            ;;
        -j|--jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        -a|--recompute-aero)
            RECOMPUTE_AERO=true
            shift
            ;;
        -t|--recompute-table)
            RECOMPUTE_TABLE=true
            shift
            ;;
        -r|--recompute-racket)
            RECOMPUTE_RACKET=true
            shift
            ;;
        -g|--use-gt200)
            USE_GT200=true
            shift
            ;;
        -e|--export-csv)
            EXPORT_CSV=true
            shift
            ;;
        -z|--zero-toss-spin)
            ZERO_TOSS_SPIN=true
            shift
            ;;
        -A|--aggregate-only)
            AGGREGATE_ONLY=true
            shift
            ;;
        -c|--csv-file)
            CSV_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATA_FOLDER" ]]; then
    echo -e "${RED}Error: Data folder is required${NC}"
    usage
fi

if [[ ! -d "$DATA_FOLDER" ]]; then
    echo -e "${RED}Error: Data folder does not exist: $DATA_FOLDER${NC}"
    exit 1
fi

if [[ "$AGGREGATE_ONLY" != "true" ]] && [[ ! -f "$UPDATE_SCRIPT" ]]; then
    echo -e "${RED}Error: Update script not found: $UPDATE_SCRIPT${NC}"
    exit 1
fi

# Convert to absolute path
DATA_FOLDER="$(cd "$DATA_FOLDER" && pwd)"

echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}  HDF5 Parallel Processing${NC}"
echo -e "${GREEN}=======================================${NC}"
echo "Data folder: $DATA_FOLDER"
echo "Max parallel jobs: $MAX_JOBS"
echo "Recompute aerodynamics: $RECOMPUTE_AERO"
echo "Recompute table contacts: $RECOMPUTE_TABLE"
echo "Recompute racket contacts: $RECOMPUTE_RACKET"
echo "Use gt200 positions: $USE_GT200"
echo "Export CSV: $EXPORT_CSV"
echo "Zero toss spin: $ZERO_TOSS_SPIN"
echo "Aggregate only: $AGGREGATE_ONLY"
echo "CSV filter: ${CSV_FILE:-none}"
echo -e "${GREEN}=======================================${NC}"
echo ""

# ── Aggregate-only mode: skip processing, just merge CSVs ────────────────
if [[ "$AGGREGATE_ONLY" == "true" ]]; then
    echo -e "${YELLOW}Aggregate-only mode: skipping HDF5 processing${NC}"
    echo ""
    for csv_name in extracted.csv extracted_TCM.csv; do
        mapfile -t CSV_FILES < <(find "$DATA_FOLDER" -mindepth 2 -name "$csv_name" -type f)
        if [[ ${#CSV_FILES[@]} -gt 0 ]]; then
            OUT_CSV="${DATA_FOLDER}/${csv_name}"
            head -n 1 "${CSV_FILES[0]}" > "$OUT_CSV"
            for csv_file in "${CSV_FILES[@]}"; do
                tail -n +2 "$csv_file" >> "$OUT_CSV"
            done
            TOTAL_ROWS=$(( $(wc -l < "$OUT_CSV") - 1 ))
            echo -e "${GREEN}  Merged ${#CSV_FILES[@]} ${csv_name} files (${TOTAL_ROWS} rows) → ${OUT_CSV}${NC}"
        else
            echo -e "${YELLOW}  No ${csv_name} files found under ${DATA_FOLDER}${NC}"
        fi
    done
    echo ""
    echo -e "${GREEN}Aggregation complete!${NC}"
    exit 0
fi

# Find data.h5 files (optionally filtered by CSV)
if [[ -n "$CSV_FILE" ]]; then
    if [[ ! -f "$CSV_FILE" ]]; then
        echo -e "${RED}Error: CSV file not found: $CSV_FILE${NC}"
        exit 1
    fi
    echo -e "${YELLOW}Filtering data.h5 files using CSV: $CSV_FILE${NC}"
    HDF5_FILES=()
    # Skip header, read location/date/experiment columns
    while IFS=, read -r location date experiment _rest; do
        exp_dir="${DATA_FOLDER}/${location}/${date}/${experiment}"
        if [[ -d "$exp_dir" ]]; then
            while IFS= read -r -d '' f; do
                HDF5_FILES+=("$f")
            done < <(find "$exp_dir" -name "data.h5" -type f -print0)
        else
            echo -e "${YELLOW}Warning: directory '$exp_dir' does not exist. Skipping.${NC}"
        fi
    done < <(tail -n +2 "$CSV_FILE")
else
    echo -e "${YELLOW}Searching for data.h5 files...${NC}"
    mapfile -t HDF5_FILES < <(find "$DATA_FOLDER" -name "data.h5" -type f)
fi

if [[ ${#HDF5_FILES[@]} -eq 0 ]]; then
    echo -e "${RED}Error: No data.h5 files found in $DATA_FOLDER${NC}"
    exit 1
fi

echo -e "${GREEN}Found ${#HDF5_FILES[@]} data.h5 files${NC}"
echo ""

# Function to process a single file
process_file() {
    local file_path="$1"
    local folder_path="$(dirname "$file_path")"
    local recompute_aero="$2"
    local recompute_table="$3"
    local recompute_racket="$4"
    local use_gt200="$5"
    local export_csv="$6"
    local zero_toss_spin="$7"
    
    echo -e "${YELLOW}[$(date +%H:%M:%S)] Processing: $folder_path${NC}"
    
    # Build the command
    local cmd="python \"$UPDATE_SCRIPT\" --data_folder \"$folder_path\""
    
    if [[ "$recompute_aero" == "true" ]]; then
        cmd="$cmd --recompute_aero True"
    fi
    
    if [[ "$recompute_table" == "true" ]]; then
        cmd="$cmd --recompute_table_contacts True"
    fi
    
    if [[ "$recompute_racket" == "true" ]]; then
        cmd="$cmd --recompute_racket_contacts True"
    fi
    
    if [[ "$use_gt200" == "true" ]]; then
        cmd="$cmd --use_gt200_positions"
    fi
    
    if [[ "$export_csv" == "true" ]]; then
        cmd="$cmd --export_csv"
    fi
    
    if [[ "$zero_toss_spin" == "true" ]]; then
        cmd="$cmd --zero_toss_spin"
    fi
    
    # Execute and capture output
    if eval "$cmd" > "/tmp/hdf5_process_$$.log" 2>&1; then
        echo -e "${GREEN}[$(date +%H:%M:%S)] ✓ Completed: $folder_path${NC}"
        rm -f "/tmp/hdf5_process_$$.log"
        return 0
    else
        echo -e "${RED}[$(date +%H:%M:%S)] ✗ Failed: $folder_path${NC}"
        echo -e "${RED}    See log: /tmp/hdf5_process_$$.log${NC}"
        return 1
    fi
}

export -f process_file
export UPDATE_SCRIPT
export RECOMPUTE_AERO
export RECOMPUTE_TABLE
export RECOMPUTE_RACKET
export USE_GT200
export EXPORT_CSV
export ZERO_TOSS_SPIN
export GREEN RED YELLOW NC

# Process files in parallel using GNU parallel or xargs
if command -v parallel &> /dev/null; then
    echo -e "${GREEN}Using GNU parallel for processing${NC}"
    echo ""
    printf "%s\n" "${HDF5_FILES[@]}" | \
        parallel -j "$MAX_JOBS" --bar \
        process_file {} "$RECOMPUTE_AERO" "$RECOMPUTE_TABLE" "$RECOMPUTE_RACKET" "$USE_GT200" "$EXPORT_CSV" "$ZERO_TOSS_SPIN"
else
    echo -e "${YELLOW}GNU parallel not found, using xargs (progress bar not available)${NC}"
    echo -e "${YELLOW}Install GNU parallel for better progress tracking: sudo apt-get install parallel${NC}"
    echo ""
    
    # Use xargs as fallback
    printf "%s\n" "${HDF5_FILES[@]}" | \
        xargs -I {} -P "$MAX_JOBS" bash -c 'process_file "$@"' _ {} "$RECOMPUTE_AERO" "$RECOMPUTE_TABLE" "$RECOMPUTE_RACKET" "$USE_GT200" "$EXPORT_CSV" "$ZERO_TOSS_SPIN"
fi

echo ""
echo -e "${GREEN}=======================================${NC}"
echo -e "${GREEN}  Processing Complete!${NC}"
echo -e "${GREEN}=======================================${NC}"
echo "Total files processed: ${#HDF5_FILES[@]}"
echo ""

# ── Aggregate per-folder CSVs into a single file at the top level ─────────
if [[ "$EXPORT_CSV" == "true" ]]; then
    echo -e "${YELLOW}Aggregating per-folder CSV files...${NC}"
    for csv_name in extracted.csv extracted_TCM.csv; do
        mapfile -t CSV_FILES < <(find "$DATA_FOLDER" -mindepth 2 -name "$csv_name" -type f)
        if [[ ${#CSV_FILES[@]} -gt 0 ]]; then
            OUT_CSV="${DATA_FOLDER}/${csv_name}"
            # Write header from the first file, then data rows from all files
            head -n 1 "${CSV_FILES[0]}" > "$OUT_CSV"
            for csv_file in "${CSV_FILES[@]}"; do
                tail -n +2 "$csv_file" >> "$OUT_CSV"
            done
            TOTAL_ROWS=$(( $(wc -l < "$OUT_CSV") - 1 ))
            echo -e "${GREEN}  Merged ${#CSV_FILES[@]} ${csv_name} files (${TOTAL_ROWS} rows) → ${OUT_CSV}${NC}"
        fi
    done
    echo ""
fi

# Check for any remaining error logs
ERROR_LOGS=$(find /tmp -name "hdf5_process_*.log" 2>/dev/null)
if [[ -n "$ERROR_LOGS" ]]; then
    echo -e "${YELLOW}Warning: Some processes failed. Check logs in /tmp/${NC}"
    echo "$ERROR_LOGS"
else
    echo -e "${GREEN}All files processed successfully!${NC}"
fi
