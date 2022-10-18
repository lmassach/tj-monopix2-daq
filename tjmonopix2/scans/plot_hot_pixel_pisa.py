#!/usr/bin/env python3
"""Plots for hot_pixel_study."""
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
    output_file = os.path.splitext(input_file)[0] + "_hp.pdf"
    if os.path.isfile(output_file) and not overwrite:
        return

    with tb.open_file(input_file) as f, PdfPages(output_file) as pdf:
        cfg = get_config_dict(f)
        plt.figure(figsize=(6.4, 4.8))
        draw_summary(input_file, cfg)
        pdf.savefig(); plt.clf()

        if f.root.Dut.shape[0] == 0:
            plt.annotate("No hits recorded!", (0.5, 0.5), ha='center', va='center')
            plt.gca().set_axis_off()
            pdf.savefig(); plt.clf()
            return

        inj_col = int(cfg["configuration_in.scan.scan_config.inj_col"])
        inj_row = int(cfg["configuration_in.scan.scan_config.inj_row"])

        # Distinguish the hits from the injected pixel, and those from other pixels
        hits = f.root.Dut[:]
        inj_mask = (hits["col"] == inj_col) & (hits["row"] == inj_row)
        print("Injected pixel:", (inj_col, inj_row))
        print("Other pixels:", np.unique(hits[~inj_mask][["col", "row"]]))

        # Compute the le, te and timestamp of the previous hit on the injected pixel
        inj_ts = np.full(hits.shape, np.nan, np.float64)  # Timestamp of the last injection for each hit
        inj_le = np.full(hits.shape, np.nan, np.float64)  # LE of the last injection for each hit
        inj_te = np.full(hits.shape, np.nan, np.float64)  # TE of the last injection for each hit
        unique_timestamps = np.unique(hits["timestamp"])
        last_ts, last_le, last_te = np.nan, np.nan, np.nan
        for ts in unique_timestamps:
            ts_mask = hits["timestamp"] == ts
            if np.count_nonzero(ts_mask & inj_mask):
                inj_hit_idx = np.argmax(ts_mask & inj_mask)
                last_ts, last_le, last_te = hits[inj_hit_idx][["timestamp", "le", "te"]]
            inj_ts[ts_mask] = last_ts
            inj_le[ts_mask] = last_le
            inj_te[ts_mask] = last_te
        del unique_timestamps, last_ts, last_te, last_le, ts_mask

        # Time from last injection
        delta_ts = hits["timestamp"] - inj_ts
        delta_le = (hits["le"] - inj_le) % 128
        delta_le_te = (hits["le"] - inj_te) % 128
        delta_te = (hits["te"] - inj_te) % 128

        # Histograms
        for mask, name in [(inj_mask, "Injected pixel"), (~inj_mask, "Other pixels")]:
            with np.errstate(all='ignore'):
                plt.hist((hits[mask]["te"] - hits[mask]["le"]) & 0x7f,
                         bins=128, range=[-0.5, 127.5], histtype='step', label=name)
        plt.title("ToT distribution")
        plt.xlabel("ToT [25 ns]")
        plt.ylabel("Hits / bin")
        plt.grid()
        plt.legend()
        pdf.savefig(); plt.clf()

        for mask, name in [(inj_mask, "Injected pixel"), (~inj_mask, "Other pixels")]:
            plt.hist(delta_ts[mask] / 640, bins=700, range=[0, 17.5], histtype='step', label=name)
        plt.title("$\\Delta$timestamp from last injection")
        plt.xlabel("$\\Delta$timestamp [us]")
        plt.xlim(0, delta_ts[delta_ts / 640 <= 17.5].max() / 640 + 0.5)
        plt.ylabel("Hits / bin")
        plt.grid()
        plt.legend()
        pdf.savefig(); plt.clf()

        for mask, name in [(inj_mask, "Injected pixel"), (~inj_mask, "Other pixels")]:
            plt.hist(delta_le[mask], bins=128, range=[-0.5, 127.5], histtype='step', label=name)
        plt.title("$\\Delta$LE from last injection")
        plt.xlabel("$\\Delta$LE [25 ns]")
        plt.xlim(delta_le.min() - 1, delta_le.max() + 1)
        plt.ylabel("Hits / bin")
        plt.grid()
        plt.legend()
        pdf.savefig(); plt.clf()

        for mask, name in [(inj_mask, "Injected pixel"), (~inj_mask, "Other pixels")]:
            plt.hist(delta_le_te[mask], bins=128, range=[-0.5, 127.5], histtype='step', label=name)
        plt.title("LE - TE of last injection")
        plt.xlabel("LE - TE$_{inj}$ [25 ns]")
        plt.xlim(delta_le_te.min() - 1, delta_le_te.max() + 1)
        plt.ylabel("Hits / bin")
        plt.grid()
        plt.legend()
        pdf.savefig(); plt.clf()

        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file", nargs="*",
        help="The hot_pixel_scan_interpreted.h5 file(s). If not given, looks in output_data/module_0/chip_0.")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Overwrite plots when already present.")
    args = parser.parse_args()

    files = []
    if args.input_file:  # If anything was given on the command line
        for pattern in args.input_file:
            files.extend(glob.glob(pattern, recursive=True))
    else:
        files.extend(glob.glob("output_data/module_0/chip_0/*hot_pixel_scan_interpreted.h5"))
    files.sort()

    for fp in tqdm(files):
        try:
            main(fp, args.overwrite)
        except Exception:
            print(traceback.format_exc())
