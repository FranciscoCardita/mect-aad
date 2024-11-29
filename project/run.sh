#!/bin/bash

WAIT_SECONDS=1800 # 30 minutes
# WAIT_SECONDS=15

CUDA_ENABLED=false

for arg in "$@"; do
    if [ "$arg" == "--cuda" ]; then
        CUDA_ENABLED=true
        break
    fi
done

for X in 0 1 2; do
    ./deti_coins_intel -s"$X" "$WAIT_SECONDS" 1
    echo "--------------------------------------------------------------------"
done

if [ "$CUDA_ENABLED" == true ]; then
    ./deti_coins_intel -s3 "$WAIT_SECONDS" 1
else
    echo "Skipping -s3 (CUDA not enabled)"
fi

