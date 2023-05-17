#!/usr/bin/env python3
"""Plots for hot_pixel_study."""
import argparse
import glob
from itertools import chain
import os
import traceback
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import tables as tb
from tqdm import tqdm
from plot_utils_pisa import *


def main(input_file, overwrite=False):
    output_file = os.path.splitext(input_file)[0] + "_delay.pdf"
    if os.path.isfile(output_file) and not overwrite:
        return

    print(f"Processing {input_file}")

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

        hits = f.root.Dut[:]

        # Event filters (event = multiple hits w same timestamp)
        timestamps, timestamp_idxs, timestamp_hits = np.unique(hits['timestamp'], return_inverse=True, return_counts=True)
        mask = timestamp_hits > 10  # Filter out noise events
        hits = hits[mask[timestamp_idxs]]
        with np.errstate(all='ignore'):
            tot = (hits['te'] - hits['le']) & 0x7f

        timestamps, timestamp_idxs, timestamp_hits = np.unique(hits['timestamp'], return_inverse=True, return_counts=True)
        # Find the LE of one pixel in row 0 for each timestamp
        timestamp_row0_le = np.full_like(timestamps, np.nan, np.float32)
        timestamp_row0_col = np.full_like(timestamps, -1, np.int16)
        mask_row0 = hits['row'] == 0
        for i, ts in tqdm(enumerate(timestamps), total=len(timestamps), unit='event', delay=2):
            mask_ts = hits['timestamp'] == ts
            mask_cut = tot > 10  # ADDITIONAL CUT ON ROW 0 HIT
            mask = mask_ts & mask_row0 & mask_cut
            if np.count_nonzero(mask):
                idx = np.argmax(mask)
                timestamp_row0_le[i] = hits['le'][idx]
                timestamp_row0_col[i] = hits['col'][idx]
        # Compute the difference LE_row0 - LE_rowi for each hit
        with np.errstate(all='ignore'):
            tmp = (timestamp_row0_le[timestamp_idxs] - hits['le']) % 128
            le_diff_to_row0 = np.where(tmp > 63, tmp - 128, tmp)
            del tmp

        # Print the first 1000 hits to a txt file with the same name as the output pdf
        with open(os.path.splitext(output_file)[0] + ".txt", "w") as ofs:
            print("COL ROW  LE  TE   TOT   DeltaTS TS", file=ofs)
            prev_ts = 0
            nhits = 0
            n = min(len(hits), 1000)
            for h, t in zip(hits[:n], tot[:n]):
                print(f"{h['col']:3d} {h['row']:3d} {h['le']:3d}  {h['te']:3d} {t:3d} {h['timestamp']-prev_ts:7d} {h['timestamp']}", file=ofs)
                prev_ts = h['timestamp']
                nhits += 1
            print("N hits = ", nhits, file=ofs)

        # Plots
        count = 0
        for i, ts in chain([(-1, None)], enumerate(timestamps)):
            if i != -1 and np.isnan(timestamp_row0_le[i]):  # Skip events w/o good hits in the row 0
                continue
            if count > 10:  # Only plot the first 10 events
                break
            count += 1

            if i == -1:
                mask_ts = np.full_like(hits, True, bool)
                subtitle = f"All {len(timestamps)} events ({len(hits):.4g} hits)"
            else:
                mask_ts = hits['timestamp'] == ts
                subtitle = f"Event {i} (TS = {ts}, {timestamp_hits[i]:.4g} hits)"

            ts_hits = hits[mask_ts]
            ts_tot = tot[mask_ts]

            # 2D histogram of LE vs row
            plt.hist2d(ts_hits['row'], ts_hits['le'], bins=[512, 128], range=[[0, 512], [0, 128]], rasterized=True)
            plt.xlabel("Row")
            plt.ylabel("LE [25 ns]")
            plt.suptitle("LE vs row")
            plt.title(subtitle)
            plt.colorbar().set_label("Number of hits")
            pdf.savefig(); plt.clf()

            # 2D histogram of ToT vs row
            plt.hist2d(ts_hits['row'], ts_tot, bins=[512, 128], range=[[0, 512], [0, 128]], rasterized=True)
            plt.xlabel("Row")
            plt.ylabel("ToT [25 ns]")
            plt.suptitle("ToT vs row")
            plt.title(subtitle)
            plt.colorbar().set_label("Number of hits")
            pdf.savefig(); plt.clf()

            # 2D histogram of LE vs ToT
            plt.hist2d(ts_tot, ts_hits['le'], bins=[128, 128], range=[[0, 128], [0, 128]], rasterized=True)
            plt.xlabel("ToT [25 ns]")
            plt.ylabel("LE [25 ns]")
            plt.suptitle("LE vs ToT")
            plt.title(subtitle)
            plt.colorbar().set_label("Number of hits")
            pdf.savefig(); plt.clf()

            # Delay map
            # Delay = BCID delay - compensation with IDEL
            # Computed as mean of (LE_row0-LE)*25ns for each pixel
            if i == -1:
                mask_cut = tot > 10  # ADDITIONAL CUT ON ROW i HITS
                mask = np.isfinite(le_diff_to_row0) & mask_cut
                hn, edges512, _ = np.histogram2d(
                    hits[mask]['col'], hits[mask]['row'], bins=[512, 512], range=[[0, 512], [0, 512]],
                    weights=le_diff_to_row0[mask])
                hd, _, _ = np.histogram2d(
                    hits[mask]['col'], hits[mask]['row'], bins=[512, 512], range=[[0, 512], [0, 512]])
                with np.errstate(all='ignore'):
                    delay_map = 25 * hn / hd
                m1 = np.nanquantile(delay_map.reshape(-1), 0.16)
                m2 = np.nanquantile(delay_map.reshape(-1), 0.84)
                ml = m2 - m1
                m1 -= 0.1 * ml
                m2 += 0.1 * ml
                plt.pcolormesh(edges512, edges512, delay_map.transpose(),
                               vmin=m1, vmax=m2, rasterized=True)
                plt.xlabel("Col")
                plt.ylabel("Row")
                plt.suptitle("Delay map - " + subtitle)
                plt.colorbar().set_label("Delay [25 ns]")
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                frontend_names_on_top()
                pdf.savefig(); plt.clf()

        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file", nargs="*",
        help="The bcid_delay_scan_interpreted.h5 file(s). If not given, looks in output_data/module_0/chip_0.")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Overwrite plots when already present.")
    args = parser.parse_args()

    files = []
    if args.input_file:  # If anything was given on the command line
        for pattern in args.input_file:
            files.extend(glob.glob(pattern, recursive=True))
    else:
        files.extend(glob.glob("output_data/module_0/chip_0/*bcid_delay_scan_interpreted.h5"))
    files.sort()

    for fp in tqdm(files):
        try:
            main(fp, args.overwrite)
        except Exception:
            print(traceback.format_exc())
