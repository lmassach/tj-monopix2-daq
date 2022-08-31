#!/usr/bin/env python3
"""Plots the results of scan_threshold (HistOcc and HistToT not required)."""
import argparse
import glob
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import tables as tb
from tqdm import tqdm
from plot_utils_pisa import *


def main(input_file):
    print("Plotting", input_file)
    output_file = os.path.splitext(input_file)[0] + ".pdf"
    with tb.open_file(input_file) as f, PdfPages(output_file) as pdf, tqdm(total=5) as bar:
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

        # Load hits
        hits = f.root.Dut[:]
        with np.errstate(all='ignore'):
            tot = (hits["te"] - hits["le"]) & 0x7f
        # Load information on injected charge and steps taken
        scan_params = f.root.configuration_in.scan.scan_params[:]
        vh = scan_params["vcal_high"][hits["scan_param_id"]]
        vl = scan_params["vcal_low"][hits["scan_param_id"]]
        charge_dac = vh - vl
        n_injections = int(cfg["configuration_in.scan.scan_config.n_injections"])
        charge_dac_values = [
            int(cfg["configuration_in.scan.scan_config.VCAL_HIGH"]) - x
            for x in range(
                int(cfg["configuration_in.scan.scan_config.VCAL_LOW_start"]),
                int(cfg["configuration_in.scan.scan_config.VCAL_LOW_stop"]),
                int(cfg["configuration_in.scan.scan_config.VCAL_LOW_step"]))]
        charge_dac_bins = len(charge_dac_values)
        charge_dac_range = [min(charge_dac_values) - 0.5, max(charge_dac_values) + 0.5]
        # TODO Count hits per pixel per injected charge value
        bar.update(1)

        plt.close()


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
