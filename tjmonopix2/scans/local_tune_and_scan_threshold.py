#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

'''
Performs the standard threshold scanning and tuning sequence:
1. scan_threshold with TDAC=4 for all pixels (untuned)
2. Shows S-curve to decide target threshold
3. tune_local_threshold with the given target threshold
4. scan_threshold with the tuned TDAC values
'''

import os
import subprocess
import sys

from scan_threshold import ThresholdScan
from tune_local_threshold import TDACTuning

scan_configuration = {
    'start_column': 213, # 216
    'stop_column': 223, #230
    'start_row': 120, #120
    'stop_row': 220, #220

    'n_injections': 100,
    'bcid_reset': False,  # BCID reset before injection
}

threshold_scan_configuration = {
    'VCAL_HIGH': 140,
    'VCAL_LOW_start': 139,
    'VCAL_LOW_stop': 65,
    'VCAL_LOW_step': -1,
    # load_tdac_from will be set later in the script
}

tdac_tuning_configuration = {
    'VCAL_LOW': 30
    # VCAL_HIGH will be set later depending on the target threshold you ask for
}

register_overrides = {
    'ITHR': 64,  # Default 64
    'IBIAS': 50,  # Default 50
    'VRESET': 110,  # Default 143
    'ICASN': 200,  # Default 0
    'VCASP': 93,  # Default 93
    "VCASC": 228,  # Default 228
    "IDB": 100,  # Default 100
    'ITUNE': 150,  # Default 53

    # Enable VL and VH measurement and override
    # 'MON_EN_VH': 0,
    # 'MON_EN_VL': 0,
    # 'OVR_EN_VH': 0,
    # 'OVR_EN_VL': 0,
    # Enable analog monitoring pixel
    'EN_PULSE_ANAMON_L': 1,
    'ANAMON_SFN_L': 0b0001,
    'ANAMON_SFP_L': 0b1000,
    'ANAMONIN_SFN1_L': 0b1000,
    'ANAMONIN_SFN2_L': 0b1000,
    'ANAMONIN_SFP_L': 0b1000,
    # Enable hitor
    'SEL_PULSE_EXT_CONF': 0,

    # set readout cycle timing as in TB
    'FREEZE_START_CONF': 1,  # Default 1, TB 41
    'READ_START_CONF': 3,  # Default 3, TB 81
    'READ_STOP_CONF': 5,  # Default 5, TB 85
    'LOAD_CONF': 7,  # Default 7, TB 119
    'FREEZE_STOP_CONF': 8,  # Default 8, TB 120
    'STOP_CONF': 8  # Default 8, TB 120
}

COLOR = "\x1b[36m"
RESET = "\x1b[0m"
SCRIPT_DIR = os.path.dirname(__file__)
PLOT_STD = os.path.join(SCRIPT_DIR, "plot_std_pisa.py")
PLOT_SCURVE = os.path.join(SCRIPT_DIR, "plot_scurve_pisa.py")

if __name__ == '__main__':
    print(f"{COLOR}Running scan_threshold with untuned TDAC{RESET}")
    threshold_scan_configuration['load_tdac_from'] = None
    with ThresholdScan(scan_config=scan_configuration|threshold_scan_configuration, register_overrides=register_overrides) as scan:
        scan.start()
        out = scan.output_filename + "_interpreted.h5"
    if not os.path.isfile(out):
        print(f"{COLOR}ERROR No _interpreted.h5 produced, stopping.{RESET}")
        sys.exit(1)


    print(f"{COLOR}Producing standard plots and s-curves{RESET}")
    subprocess.run([PLOT_STD, out], check=True)
    subprocess.run([PLOT_SCURVE, out], check=True)
    pdf = os.path.splitext(out)[0] + "_scurve.pdf"
    subprocess.run(["xdg-open", pdf])
    print(f"{COLOR}Produced PDF with s-curves {pdf}{RESET}")
    print(f"{COLOR}Look at it and choose the target threshold for the TDAC/local tuning{RESET}")
    while True:
        try:
            target_th = int(input(f"{COLOR}Target threshold [DAC] = {RESET}"))
            if 0 < target_th < 170:
                break
        except Exception:
            pass

    print(f"{COLOR}Running tune_local_threshold with target threshold = {target_th}{RESET}")
    tdac_tuning_configuration['VCAL_HIGH'] = tdac_tuning_configuration['VCAL_LOW'] + target_th
    with TDACTuning(scan_config=scan_configuration|tdac_tuning_configuration, register_overrides=register_overrides) as tuning:
        tuning.start()
        out = tuning.output_filename + "_interpreted.h5"
    if not os.path.isfile(out):
        print(f"{COLOR}ERROR No _interpreted.h5 produced, stopping.{RESET}")
        sys.exit(1)
    print(f"{COLOR}Produced H5 with tuned TDACs {out}{RESET}")

    print(f"{COLOR}Running scan_threshold with TDAC values from previous tuning{RESET}")
    threshold_scan_configuration['load_tdac_from'] = out
    with ThresholdScan(scan_config=scan_configuration|threshold_scan_configuration, register_overrides=register_overrides) as scan:
        scan.start()
        out = scan.output_filename + "_interpreted.h5"
    if not os.path.isfile(out):
        print(f"{COLOR}ERROR No _interpreted.h5 produced, stopping.{RESET}")
        sys.exit(1)
    print(f"{COLOR}Produced H5 with s-curves {out}{RESET}")
    print(f"{COLOR}Done.{RESET}")
