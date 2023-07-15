#!/bin/bash

# This script scans the IDEL over PULSE START CONFIG with injection
# 
# 

for IDEL in 40; do

    H5_DIR="output_data/module_0/chip_0"
    OUT_DIR="script_output/scan_idel_40MHz_150deltaV/data/IDEL_$IDEL"

    #put VRESET to 120 for more gaussian

    function run_scan {
        export PULSE_START_CONFIG=$1
        export IDEL=$IDEL
        echo "--------- step: $1 analog scan ---------"
        python scan_analog.py
        
        mkdir -p "$OUT_DIR/PULSE_START_CONFIG_$1"
        mv $H5_DIR/*analog_scan* $OUT_DIR/PULSE_START_CONFIG_$1/
    }

    export NUM_COLS=32
    export DELTA_TARGET=26

    #echo '--------- global thr ---------'
    #python tune_global_threshold.py
    #echo '--------- local thr ---------'
    #python tune_local_threshold.py

    # $(seq 0 1 63)
    for delay in $(seq 0 1 64); do # $(seq 0 1 63); do
        run_scan $delay
    done
    exit 0
    # 55: 63
    
done
