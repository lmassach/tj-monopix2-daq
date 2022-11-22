#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

from scan_source import SourceScan
import datetime
import tqdm


scan_configuration = {
    'start_column': 180,
    'stop_column': 223,
    'start_row': 150,
    'stop_row': 500,

    'scan_time': 30,  # seconds
    # File produced w/o BCID reset target=20 DAC psub/pwell=-6V cols=180-223 rows=150-500
    'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0_2022-11-17/chip_0/20221117_184249_local_threshold_tuning_interpreted.h5'
}

default_register_overrides = {
    'ITHR': 64,  # Default 64
    'IBIAS': 50,  # Default 50
    'VRESET': 110,  # Default 143
    'ICASN': 200,  # Default 0
    'VCASP': 93,  # Default 93
    "VCASC": 228,  # Default 228
    "IDB": 100,  # Default 100
    'ITUNE':  150,  # Default 53
    'VCLIP': 160,  # Default 255

    # set readout cycle timing as in TB/or as default in Pisa
    'FREEZE_START_CONF': 10,  # Default 1, TB 41
    'READ_START_CONF': 13,  # Default 3, TB 81
    'READ_STOP_CONF': 15,  # Default 5, TB 85
    'LOAD_CONF': 30,  # Default 7, TB 119
    'FREEZE_STOP_CONF': 31,  # Default 8, TB 120
    'STOP_CONF': 31  # Default 8, TB 120
}

sweeps = {  # REGISTER: (START, STOP, STEP)
    'VCLIP': (155, 100, -5),
    #'ITHR': (25, 75, 5),
    #'VRESET': (70, 255, 20),
    #'IBIAS': (20, 60, 5),
    #'VCASP': (100, 140, 5),
    #'ICASN': (0, 16, 1)
    #'ITHR': (20, 41, 20),
    #'ICASN': (5, 16, 5),
    #'IDB': (100, 250, 10)
}

if __name__ == "__main__":
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"output_data/scan_source_sweep_{date}.txt", "a") as ofs:
        for reg, sweep_range in sweeps.items():
            print(f"!!! Sweeping over {reg} in range {sweep_range}")
            for i in tqdm.tqdm(range(*sweep_range), unit=f"{reg} steps", smoothing=0):
                print(f"!!! Sweeping over {reg}: {i}")
                register_overrides = default_register_overrides.copy()
                register_overrides[reg] = i
                with SourceScan(scan_config=scan_configuration, register_overrides=register_overrides) as scan:
                    scan.start()
                    print(f"{scan.output_filename} {reg}={i}", file=ofs)
