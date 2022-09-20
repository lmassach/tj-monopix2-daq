#!/usr/bin/env python3
"""Standard plots like hitmap and ToT histogram (HistOcc and HistToT not required)."""
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
    output_file = os.path.splitext(input_file)[0] + ".pdf"
    if os.path.isfile(output_file) and not overwrite:
        return
    print("Plotting", input_file)
    with tb.open_file(input_file) as f, PdfPages(output_file) as pdf:
        cfg = get_config_dict(f)
        plt.figure(figsize=(6.4, 4.8))

        draw_summary(input_file, cfg)
        pdf.savefig(); plt.clf()

        hits = f.root.Dut[:]
        counts2d, edges, _ = np.histogram2d(hits["col"], hits["row"], bins=[512, 512], range=[[0, 512], [0, 512]])
        with np.errstate(all='ignore'):
            tot = (hits["te"] - hits["le"]) & 0x7f
        fe_masks = [(hits["col"] >= fc) & (hits["col"] <= lc) for fc, lc, _ in FRONTENDS]

        # Histogram of hits per pixel
        m = np.quantile(counts2d[counts2d > 0], 0.99) * 1.2 if np.any(counts2d > 0) else 1
        bins = 100 if m > 200 else int(max(m, 5))
        for fc, lc, name in FRONTENDS:
            plt.hist(counts2d[fc:lc+1,:].reshape(-1), label=name, histtype='step',
                     bins=bins, range=[0.5, max(m, 5) + 0.5])
        plt.title("Hits per pixel")
        plt.xlabel("Number of hits")
        plt.ylabel("Pixels / bin")
        plt.grid(axis='y')
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        plt.legend()
        pdf.savefig(); plt.clf()

        # Histogram of ToT
        for (_, _, name), mask in zip(FRONTENDS, fe_masks):
            plt.hist(tot[mask], bins=128, range=[-0.5, 127.5], histtype='step', label=name)
        plt.title("ToT")
        plt.xlabel("ToT [25 ns]")
        plt.ylabel("Hits / bin")
        plt.grid(axis='y')
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        plt.legend()
        pdf.savefig(); plt.clf()

        # Hit map
        plt.pcolormesh(edges, edges, counts2d.transpose(), vmin=0, vmax=m,
                       rasterized=True)  # Necessary for quick save and view in PDF
        plt.title("Hit map")
        plt.xlabel("Col")
        plt.ylabel("Row")
        cb = integer_ticks_colorbar()
        cb.set_label("Hits / pixel")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        frontend_names_on_top()
        pdf.savefig(); plt.clf()

        # Map of the average ToT
        tot2d, _, _ = np.histogram2d(hits["col"], hits["row"], bins=[512, 512],
                                     range=[[0, 512], [0, 512]], weights=tot)
        with np.errstate(all='ignore'):
            totavg = tot2d /counts2d
        plt.pcolormesh(edges, edges, totavg.transpose(), vmin=-0.5, vmax=127.5,
                       rasterized=True)  # Necessary for quick save and view in PDF
        plt.title("Average ToT map")
        plt.xlabel("Col")
        plt.ylabel("Row")
        cb = integer_ticks_colorbar()
        cb.set_label("ToT [25 ns]")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        frontend_names_on_top()
        pdf.savefig(); plt.clf()

        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file", nargs="*",
        help="The _interpreted.h5 file(s). If not given, looks in output_data/module_0/chip_0.")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Overwrite plots when already present.")
    args = parser.parse_args()

    files = []
    if args.input_file:  # If anything was given on the command line
        for pattern in args.input_file:
            files.extend(glob.glob(pattern, recursive=True))
    else:
        files.extend(glob.glob("output_data/module_0/chip_0/*_interpreted.h5"))
    files.sort()

    for fp in tqdm(files):
        try:
            main(fp, args.overwrite)
        except Exception:
            print(traceback.format_exc())
