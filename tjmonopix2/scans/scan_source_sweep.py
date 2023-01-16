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
    'stop_column': 447,
    'start_row': 0,
    'stop_row': 512,

    'scan_time': 60,  # seconds
    # File produced w/o BCID reset target=20 DAC psub/pwell=-6V cols=180-223 rows=150-500
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0_2022-11-17/chip_0/20221117_184249_local_threshold_tuning_interpreted.h5'
    # chipW15R12 File produced w BCID reset target=31 DAC psub ground pwell=-3V cols=0-223 rows=0-511
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0_2022-12-14/chip_0/20221214_114000_local_threshold_tuning.h5'
    #chipW8r13 File produced w BCID reset target=25 DAC psub/pwell=-6V cols=0-447 rows=0-512
    'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0/chip_0/20230108_084232_local_threshold_tuning_interpreted.h5'
}

default_register_overrides = {
  'ITHR':30,  # Default 64
    'IBIAS': 50,  # Default 50
    'VRESET': 110,  # Default 143, 110 for lower THR
    'ICASN': 0,  # Default TB 0 , 150 for -3V , 200 for -6V
    'VCASP': 93,  # Default 93
    "VCASC": 228,  # Default 228
    "IDB": 100,  # Default 100
    'ITUNE': 175,  # Default TB 53, 150 for lower THR tuning
    'VCLIP': 255,  # Default 255

    # set readout cycle timing as in TB/or as default in Pisa
    'FREEZE_START_CONF': 10,  # Default 1, TB 41
    'READ_START_CONF': 13,  # Default 3, TB 81
    'READ_STOP_CONF': 15,  # Default 5, TB 85
    'LOAD_CONF': 30,  # Default 7, TB 119
    'FREEZE_STOP_CONF': 31,  # Default 8, TB 120
    'STOP_CONF': 31  # Default 8, TB 120
}

sweeps = {  # REGISTER: (START, STOP, STEP)
    #'VCLIP': (155, 100, -5),
    #'ITHR': (60, 10, -10),
    #'VRESET': (70, 255, 20),
    #'IBIAS': (20, 60, 5),
    #'VCASP': (100, 140, 5),
    #'ICASN': (0, 16, 1)
    #'ITHR': (20, 41, 20),
    'ICASN': (0, 25, 5),
    #'IDB': (100, 50, -10)
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
