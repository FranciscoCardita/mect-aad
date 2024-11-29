#!/bin/bash

# WAIT_SECONDS=1800 # 30 minutes
WAIT_SECONDS=15

CUDA_ENABLED=false
OUTPUT_FILE=""

for ((i = 1; i <= $#; i++)); do
    case ${!i} in
        -c|--cuda)
            CUDA_ENABLED=true
            ;;
        -o|--output)
            ((i++))
            OUTPUT_FILE=${!i}
            ;;
    esac
done

if [ -n "$OUTPUT_FILE" ]; then
    > "$OUTPUT_FILE"
fi

run_command() {
    local cmd="$1"
    if [ -n "$OUTPUT_FILE" ]; then
        eval "$cmd" | tee -a "$OUTPUT_FILE"
    else
        eval "$cmd"
    fi
}

for X in 0 1 2; do
    run_command "./deti_coins_intel -s$X $WAIT_SECONDS 1"
    echo "--------------------------------------------------------------------"
done

if [ "$CUDA_ENABLED" == true ]; then
    run_command "./deti_coins_intel -s3 $WAIT_SECONDS 1"
else
    echo "Skipping -s3 (CUDA not enabled)"
fi

