#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

from scan_threshold import ThresholdScan
import datetime


scan_configuration = {
    'start_column': 225,
    'stop_column': 230,
    'start_row': 120,
    'stop_row': 420,

    'n_injections': 100,
    'VCAL_HIGH': 140,
    'VCAL_LOW_start': 139,
    'VCAL_LOW_stop': 1,
    'VCAL_LOW_step': -1
}

default_register_overrides = {
    "ITHR": 64,
    "IBIAS": 50,
    "ICASN": 0,
    "VCASP": 93,
    "VRESET": 143,
    "VCASC": 228,
    "IDB": 100,
    "ITUNE": 53
}

sweeps = {  # REGISTER: (START, STOP, STEP)
    'ITHR': (20, 41, 20),
    #'VRESET': (70, 255, 20),
    #'IBIAS': (20, 60, 5),
    #'VCASP': (3, 255, 10),
    #'ICASN': (0, 16, 1)
    #'ITHR': (20, 41, 20),
    #'ICASN': (5, 16, 5)
}

if __name__ == "__main__":
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"output_data/scan_threshold_sweep_{date}.txt", "a") as ofs:
        for reg, sweep_range in sweeps.items():
            print(f"!!! Sweeping over {reg} in range {sweep_range}")
            for i in range(*sweep_range):
                print(f"!!! Sweeping over {reg}: {i}")
                register_overrides = default_register_overrides.copy()
                register_overrides[reg] = i
                with ThresholdScan(scan_config=scan_configuration, register_overrides=register_overrides) as scan:
                    scan.start()
                    print(f"{scan.output_filename} {reg}={i}", file=ofs)
