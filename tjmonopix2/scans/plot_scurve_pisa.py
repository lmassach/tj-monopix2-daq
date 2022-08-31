#!/usr/bin/env python3
"""Plots the results of scan_threshold (HistOcc and HistToT not required)."""
import argparse
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import tables as tb
from plot_utils_pisa import get_config_dict


def main(input_file):
    print("Plotting", input_file)
    input_file_name = os.path.basename(input_file)
    with tb.open_file(input_file) as f:
        cfg = get_config_dict(f)
        chip_serial_number = cfg["configuration_in.chip.settings.chip_sn"]
        # TODO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file", nargs="*",
        help="The _threshold_scan_interpreted.h5 file(s)."
             " If not given, looks in output_data/module_0/chip_0.")
    args = parser.parse_args()
    if args.input_file:  # If anything was given on the command line
        for pattern in args.input_file:
            for fp in glob.glob(pattern, recursive=True):
                main(fp)
    else:
        for fp in glob.glob("output_data/module_0/chip_0/*_threshold_scan_interpreted.h5"):
            main(fp)
