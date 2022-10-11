#!/usr/bin/env python3
"""Plots related to a Fe55 (or other) source peak (HistOcc and HistToT not required)."""
import argparse
import glob
import os
import traceback
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import tables as tb
from tqdm import tqdm
from plot_utils_pisa import *


def main(input_file, overwrite=False):
    output_file = os.path.splitext(input_file)[0] + "_peak.pdf"
    if os.path.isfile(output_file) and not overwrite:
        return
    print("Plotting", input_file)
    with tb.open_file(input_file) as f, PdfPages(output_file) as pdf:
        cfg = get_config_dict(f)
        plt.figure(figsize=(6.4, 4.8))

        draw_summary(input_file, cfg)
        pdf.savefig(); plt.clf()
        # print("Summary")

        # Prepare histogram
        counts = np.zeros((512, 512, 128))

        # Process 16M hits at a time
        csz = 2**24
        n_hits = f.root.Dut.shape[0]
        for i_first in tqdm(range(0, n_hits, csz), unit="chunk", disable=n_hits/csz<=1):
            i_last = min(i_first + csz, n_hits)
            hits = f.root.Dut[i_first:i_last]
            with np.errstate(all='ignore'):
                tmp, edges = np.histogramdd(
                    (hits["col"], hits["row"], (hits["te"] - hits["le"]) & 0x7f),
                    bins=[512, 512, 128], range=[[0, 512], [0, 512], [0, 128]])
                counts += tmp
                del tmp
        del hits

        # Find the position of the peak in each pixel
        max_tot = np.argmax(counts, axis=2)

        # Peak center distribution
        for fc, lc, name in FRONTENDS:
            plt.hist(max_tot[fc:lc+1,:].reshape(-1), bins=128, range=[0, 128], label=name, rasterized=True)
        plt.xlabel("ToT of max [25 ns]")
        plt.ylabel("Pixels / bin")
        plt.title("ToT bin with the most entries (approx. peak)")
        plt.legend()
        plt.grid()
        pdf.savefig(); plt.clf()

        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file", nargs="*",
        help="The _source_scan_interpreted.h5 file(s). If not given, looks in output_data/module_0/chip_0.")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Overwrite plots when already present.")
    args = parser.parse_args()

    files = []
    if args.input_file:  # If anything was given on the command line
        for pattern in args.input_file:
            files.extend(glob.glob(pattern, recursive=True))
    else:
        files.extend(glob.glob("output_data/module_0/chip_0/*_source_scan_interpreted.h5"))
    files.sort()

    for fp in tqdm(files):
        try:
            main(fp, args.overwrite)
        except Exception:
            print(traceback.format_exc())
