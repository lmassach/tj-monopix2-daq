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
    'start_column': 0,
    'stop_column': 448,
    'start_row': 0,
    'stop_row': 512,

    'scan_time': 5,  # seconds
}

default_register_overrides = {
    'ITHR': 20,
    'IBIAS': 50,
    'VRESET': 143,
    'ICASN': 10,
    'VCASP': 93,
    'IDB': 200,
    # 'MON_EN_IDB': 1
}

sweeps = {  # REGISTER: (START, STOP, STEP)
    #'ITHR': (25, 75, 5),
    #'VRESET': (70, 255, 20),
    #'IBIAS': (20, 60, 5),
    #'VCASP': (100, 140, 5),
    #'ICASN': (0, 16, 1)
    #'ITHR': (20, 41, 20),
    #'ICASN': (5, 16, 5),
    'IDB': (100, 250, 10)
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
