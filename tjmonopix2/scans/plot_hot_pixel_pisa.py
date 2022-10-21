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


def main(input_file, overwrite=False, verbose=False):
    output_file = os.path.splitext(input_file)[0] + "_hp.pdf"
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

        inj_col = int(cfg["configuration_in.scan.scan_config.inj_col"])
        inj_row = int(cfg["configuration_in.scan.scan_config.inj_row"])

        # Distinguish the hits from the injected pixel, and those from other pixels
        hits = f.root.Dut[:]
        inj_mask = (hits["col"] == inj_col) & (hits["row"] == inj_row)
        print("Injected pixel:", (inj_col, inj_row))
        print("Other pixels:", np.unique(hits[~inj_mask][["col", "row"]]))

        if verbose:
            TS_CLK = 40  # MHz  # Multiplying by 1.106 we get match between ΔTE and ΔTS: wrong/unsynchronized clocks?
            print(f"\x1b[1mAssuming timestamp clock = {TS_CLK:.2f} MHz\x1b[0m")
            print("\x1b[1mGreen = injected pixels\x1b[0m")
            print("\x1b[1mRow  Col   LE   TE  ΔLE  ΔTE   ΔTS[25ns]  TS[25ns]\x1b[0m")
            pu, pl, pt = np.nan, np.nan, np.nan
            for cnt, (r, c, l, t, u, i) in enumerate(zip(hits["row"], hits["col"], hits["le"], hits["te"], hits["timestamp"], inj_mask)):
                if cnt > 100:
                    break
                u = (u - hits["timestamp"][0]) / TS_CLK / 25e-3
                color = "\x1b[32m" if i else ""
                reset = "\x1b[0m" if i else ""
                with np.errstate(all='ignore'):
                    print(f"{color}{r:3d}  {c:3d}  {l:3d}  {t:3d}  {(l-pl)%128:3.0f}  {(t-pt)%128:3.0f}  {u-pu:10.4f}  {u:.4f}{reset}")
                pu, pl, pt = u, l, t

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
        del last_ts, last_te, last_le, ts_mask

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

        plt.axes((0.125, 0.11, 0.775, 0.72))
        plt.hist(np.diff(hits[inj_mask]["timestamp"]) / TS_CLK, bins=700, range=[0, 280], histtype='step')
        plt.title("$\\Delta$timestamp between injections")
        plt.xlabel("$\\Delta$timestamp [μs]")
        plt.ylabel("Hits / bin")
        plt.grid()
        # Axis above with time in 25 ns units
        xl, xu = plt.xlim()
        ax2 = plt.gca().twiny()
        ax2.set_xlim(xl * 40, xu * 40)
        ax2.set_xlabel("$\\Delta$timestamp [25 ns]")
        pdf.savefig(); plt.clf()

        plt.axes((0.125, 0.11, 0.775, 0.72))
        for mask, name in [(inj_mask, "Injected pixel"), (~inj_mask, "Other pixels")]:
            plt.hist(delta_ts[mask] / TS_CLK, bins=700, range=[0, 280], histtype='step', label=name)
        plt.title("$\\Delta$timestamp from last injection")
        plt.xlabel("$\\Delta$timestamp [μs]")
        plt.xlim(-0.025, delta_ts[delta_ts / TS_CLK <= 280].max() / TS_CLK + 0.025)
        plt.ylabel("Hits / bin")
        plt.grid()
        plt.legend()
        # Axis above with time in 25 ns units
        xl, xu = plt.xlim()
        ax2 = plt.gca().twiny()
        ax2.set_xlim(xl * 40, xu * 40)
        ax2.set_xlabel("$\\Delta$timestamp [25 ns]")
        pdf.savefig(); plt.clf()

        plt.axes((0.125, 0.11, 0.775, 0.72))
        for mask, name in [(inj_mask, "Injected pixel"), (~inj_mask, "Other pixels")]:
            plt.hist(delta_ts[mask] / TS_CLK, bins=80, range=[-0.0125, 2-0.0125], histtype='step', label=name)
        plt.title("$\\Delta$timestamp from last injection")
        plt.xlabel("$\\Delta$timestamp [μs]")
        plt.xlim(-0.025, delta_ts[delta_ts / TS_CLK <= 2-0.0125].max() / TS_CLK + 0.025)
        plt.ylabel("Hits / bin")
        plt.grid()
        plt.legend()
        # Axis above with time in 25 ns units
        xl, xu = plt.xlim()
        ax2 = plt.gca().twiny()
        ax2.set_xlim(xl * 40, xu * 40)
        ax2.set_xlabel("$\\Delta$timestamp [25 ns]")
        pdf.savefig(); plt.clf()

        for mask, name in [(inj_mask, "Injected pixel"), (~inj_mask, "Other pixels")]:
            plt.hist(delta_le[mask], bins=128, range=[-0.5, 127.5], histtype='step', label=name)
        plt.title("$\\Delta$LE from last injection")
        plt.xlabel("$\\Delta$LE [25 ns]")
        plt.xlim(delta_le[np.isfinite(delta_le)].min() - 1, delta_le[np.isfinite(delta_le)].max() + 1)
        plt.ylabel("Hits / bin")
        plt.grid()
        plt.legend()
        pdf.savefig(); plt.clf()

        for mask, name in [(inj_mask, "Injected pixel"), (~inj_mask, "Other pixels")]:
            plt.hist(delta_le_te[mask], bins=128, range=[-0.5, 127.5], histtype='step', label=name)
        plt.title("LE - TE of last injection")
        plt.xlabel("LE - TE$_{inj}$ [25 ns]")
        plt.xlim(delta_le_te[np.isfinite(delta_le_te)].min() - 1, delta_le_te[np.isfinite(delta_le_te)].max() + 1)
        plt.ylabel("Hits / bin")
        plt.grid()
        plt.legend()
        pdf.savefig(); plt.clf()

        h = np.zeros(128)
        for mask, name in [(inj_mask, "Injected pixel"), (~inj_mask, "Other pixels")]:
            with np.errstate(all='ignore'):
                ht, _, _ = plt.hist(
                    (hits["le"][1:][mask[1:]] - hits["te"][:-1][mask[1:]]) & 0x7f,
                    bins=128, range=[-0.5, 127.5], histtype='step', label=name)
                h += ht
        plt.title("LE - TE of previous hit")
        plt.xlabel("LE$_{i}$ - TE$_{i-1}$ [25 ns]")
        nz = np.nonzero(h > 0)[0]
        plt.xlim(nz[0] - 1, nz[-1] + 1)
        plt.ylabel("Hits / bin")
        plt.grid()
        plt.legend()
        pdf.savefig()
        plt.xlim(-1, 21)
        plt.ylim(0, (h[:21].max() + 1) * 1.2)
        pdf.savefig(); plt.clf()

        # Frames ("photos" of pixels that fire with the same timestamp, i.e. read at the same time)
        for i, ts in enumerate(unique_timestamps):
            if i > 15:
                break
            mask = hits["timestamp"] == ts
            mh = hits[mask]
            plt.hist2d(mh["col"], mh["row"], bins=[512, 512], range=[[0, 512], [0, 512]], rasterized=True)
            plt.xlabel("Column")
            plt.ylabel("Row")
            plt.xlim(200, 230)
            plt.ylim(120, 220)
            plt.colorbar().set_label("Number of hits")
            plt.title(f"Frame {i} @ timestamp = {(ts-unique_timestamps.min())/16} [25 ns]")
            pdf.savefig(); plt.clf()

        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file", nargs="*",
        help="The hot_pixel_scan_interpreted.h5 file(s). If not given, looks in output_data/module_0/chip_0.")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Overwrite plots when already present.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print a table with data from the first 100 hits.")
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
            main(fp, args.overwrite, args.verbose)
        except Exception:
            print(traceback.format_exc())
