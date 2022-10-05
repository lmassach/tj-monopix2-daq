#!/usr/bin/env python3
"""Standard plots like hitmap and ToT histogram (HistOcc and HistToT not required)."""
import argparse
import glob
import os
import traceback
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import tables as tb
from tqdm import tqdm
from plot_utils_pisa import *


def main(input_file, overwrite=False, log_tot=False):
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
        if log_tot:
            plt.yscale('log')
            set_integer_ticks(plt.gca().xaxis)
        else:
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

        # Noisy pixels
        if cfg.get("configuration_in.scan.run_config.scan_id") == "source_scan":
            MAX_RATE = 1  # Above this rate [Hz] pixels are marked noisy
            scan_time = float(cfg["configuration_in.scan.scan_config.scan_time"])
            max_hits = scan_time * MAX_RATE
            mask = counts2d > max_hits
            plt.axes((0.125, 0.11, 0.775, 0.72))
            plt.pcolormesh(edges, edges, 1 * mask.transpose(), vmin=0, vmax=1,
                           rasterized=True)  # Necessary for quick save and view in PDF
            plt.suptitle("Noisy pixels in yellow (ignore this plot if source was used)")
            plt.title(f"Noisy means rate > {MAX_RATE:.3g} Hz")
            plt.xlabel("Col")
            plt.ylabel("Row")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            frontend_names_on_top()
            pdf.savefig(); plt.clf()

            if np.count_nonzero(mask):
                noisy_list = np.argwhere(mask)
                noisy_indices = np.nonzero(mask)
                srt = np.argsort(-counts2d[noisy_indices])
                noisy_indices = tuple(x[srt] for x in noisy_indices)
                noisy_list = noisy_list[srt]
                mi = min(len(noisy_list), 100)
                tmp = "\n".join(
                    ",    ".join(f"({a}, {b}) = {float(c)/scan_time:.3g}" for (a, b), c in g)
                    for g in groupwise(zip(noisy_list[:mi], counts2d[tuple(x[:mi] for x in noisy_indices)]), 4))
                plt.annotate(
                    split_long_text(
                        "Noisiest pixels (col, row) = rate [Hz]\n"
                        f"{tmp}"
                        f'{", ..." if len(noisy_list) > mi else ""}'
                        f"\nTotal = {len(noisy_list)} pixels"
                    ), (0.5, 0.5), ha='center', va='center')
            else:
                plt.annotate("No noisy pixel found.", (0.5, 0.5), ha='center', va='center')
            plt.gca().set_axis_off()
            pdf.savefig(); plt.clf()

        # Source positioning
        counts2d16, edges16, _ = np.histogram2d(hits["col"], hits["row"], bins=[32, 32], range=[[0, 512], [0, 512]])
        m = np.quantile(counts2d16[counts2d16 > 0], 0.99) * 1.2 if np.any(counts2d > 0) else 1
        cmap = matplotlib.cm.get_cmap("viridis").copy()
        cmap.set_over("r")
        plt.pcolormesh(edges16, edges16, counts2d16.transpose(), vmin=0, vmax=m,
                       cmap=cmap, rasterized=True)  # Necessary for quick save and view in PDF
        plt.title("Hit map in 16x16 regions for source positioning")
        plt.xlabel("Col")
        plt.ylabel("Row")
        cb = plt.colorbar()
        cb.set_label("Avg. hits / 16x16 region (red = out of scale)")
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
    parser.add_argument("--log-tot", action="store_true",
                        help="Use log scale for ToT.")
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
            main(fp, args.overwrite, args.log_tot)
        except Exception:
            print(traceback.format_exc())
