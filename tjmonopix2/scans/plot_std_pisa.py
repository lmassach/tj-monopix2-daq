#!/usr/bin/env python3
"""Standard plots like hitmap and ToT histogram (HistOcc and HistToT not required)."""
import argparse
import glob
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import tables as tb
from plot_utils_pisa import *


def main(input_file):
    print("Plotting", input_file)
    input_file_name = os.path.basename(input_file)
    output_file = os.path.splitext(input_file)[0] + ".pdf"
    with tb.open_file(input_file) as f, PdfPages(output_file) as pdf:
        cfg = get_config_dict(f)
        chip_serial_number = cfg["configuration_in.chip.settings.chip_sn"]
        plt.figure(figsize=(6.4, 4.8))
        plt.annotate(
            split_long_text(f"{os.path.abspath(input_file)}\n"
                            f"Chip {chip_serial_number}\n"
                            f"Version {get_commit()}"),
            (0.5, 0.5), ha='center', va='center')
        plt.gca().set_axis_off()
        pdf.savefig(); plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file", nargs="*",
        help="The _interpreted.h5 file(s). If not given, looks in output_data/module_0/chip_0.")
    args = parser.parse_args()
    if args.input_file:  # If anything was given on the command line
        for pattern in args.input_file:
            for fp in glob.glob(pattern, recursive=True):
                main(fp)
    else:
        for fp in glob.glob("output_data/module_0/chip_0/*_interpreted.h5"):
            main(fp)
