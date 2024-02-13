#!/bin/bash
#F_BCID='21.7'
F_BCID='40.0'
#F_BCID='20.0'
export F_BCID=$F_BCID

H5_DIR="output_data/module_0/chip_0"
OUT_DIR="script_output/scan_bcid_variation_${F_BCID}MHz_long/data"

#put VRESET to 120 for more gaussian



function run_scan {
    export PULSE_START_CONFIG=$1
    echo "--------- step: $1 s-curve ---------"
    python scan_threshold.py
    
    mkdir -p "$OUT_DIR/PULSE_START_CONFIG_$1"
    mv $H5_DIR/*threshold_scan* $OUT_DIR/PULSE_START_CONFIG_$1/
}

# $(seq 6 1 160)
# $(seq 149 1 200) $(seq 202 1 1000)
for delay in $(seq 13 1 21); do
    run_scan $delay
done


