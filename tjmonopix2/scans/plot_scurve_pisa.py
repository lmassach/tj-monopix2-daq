#!/usr/bin/env python3
"""Plots the results of scan_threshold (HistOcc and HistToT not required)."""
import argparse
import glob
from itertools import chain
import os
import traceback
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import tables as tb
from tqdm import tqdm
from uncertainties import ufloat
from plot_utils_pisa import *

COLOR_GRADIENTS = [
    # Blue fading to white
    [((0x1f + c*0xff/100) / 512, (0x77 + c*0xff/100) / 512, (0xb4 + c*0xff/100) / 512) for c in range(64)],
    # Orange fading to white
    [((0xff + c*0xff/100) / 512, (0x7f + c*0xff/100) / 512, (0x0e + c*0xff/100) / 512) for c in range(64)],
    # Green fading to white
    [((0x2c + c*0xff/100) / 512, (0xa0 + c*0xff/100) / 512, (0x2c + c*0xff/100) / 512) for c in range(64)],
    # Red fading to white
    [((0xd6 + c*0xff/100) / 512, (0x27 + c*0xff/100) / 512, (0x28 + c*0xff/100) / 512) for c in range(64)]]


@np.errstate(divide='ignore')
def average(a, axis=None, weights=1, invalid=np.NaN):
    """Like np.average, but returns `invalid` instead of crashing if the sum of weights is zero."""
    return np.nan_to_num(np.sum(a * weights, axis=axis).astype(float) / np.sum(weights, axis=axis).astype(float), nan=invalid)


def main(input_file, overwrite=False):
    output_file = os.path.splitext(input_file)[0] + "_scurve.pdf"
    if os.path.isfile(output_file) and not overwrite:
        return
    print("Plotting", input_file)
    with tb.open_file(input_file) as f, PdfPages(output_file) as pdf:
        cfg = get_config_dict(f)
        plt.figure(figsize=(6.4, 4.8))

        draw_summary(input_file, cfg)
        pdf.savefig(); plt.clf()

        # Load hits
        hits = f.root.Dut[:]
        with np.errstate(all='ignore'):
            tot = (hits["te"] - hits["le"]) & 0x7f
        fe_masks = [(hits["col"] >= fc) & (hits["col"] <= lc) for fc, lc, _ in FRONTENDS]

        # Load information on injected charge and steps taken
        sp = f.root.configuration_in.scan.scan_params[:]
        scan_params = np.zeros(sp["scan_param_id"].max() + 1, dtype=sp.dtype)
        for i in range(len(scan_params)):
            m = sp["scan_param_id"] == i
            if np.any(m):
                scan_params[i] = sp[m.argmax()]
            else:
                scan_params[i]["scan_param_id"] = i
        del sp
        vh = scan_params["vcal_high"][hits["scan_param_id"]]
        vl = scan_params["vcal_low"][hits["scan_param_id"]]
        del scan_params
        charge_dac = vh - vl
        n_injections = int(cfg["configuration_in.scan.scan_config.n_injections"])
        the_vh = int(cfg["configuration_in.scan.scan_config.VCAL_HIGH"])
        start_vl = int(cfg["configuration_in.scan.scan_config.VCAL_LOW_start"])
        stop_vl = int(cfg["configuration_in.scan.scan_config.VCAL_LOW_stop"])
        step_vl = int(cfg["configuration_in.scan.scan_config.VCAL_LOW_step"])
        charge_dac_values = [
            the_vh - x for x in range(start_vl, stop_vl, step_vl)]
        subtitle = f"VH = {the_vh}, VL = {start_vl}..{stop_vl} (step {step_vl})"
        charge_dac_bins = len(charge_dac_values)
        charge_dac_range = [min(charge_dac_values) - 0.5, max(charge_dac_values) + 0.5]
        # Count hits per pixel per injected charge value
        row_start = int(cfg["configuration_in.scan.scan_config.start_row"])
        row_stop = int(cfg["configuration_in.scan.scan_config.stop_row"])
        col_start = int(cfg["configuration_in.scan.scan_config.start_column"])
        col_stop = int(cfg["configuration_in.scan.scan_config.stop_column"])
        row_n, col_n = row_stop - row_start, col_stop - col_start
        occupancy, occupancy_edges = np.histogramdd(
            (hits["col"], hits["row"], charge_dac),
            bins=[col_n, row_n, charge_dac_bins],
            range=[[col_start, col_stop], [row_start, row_stop], charge_dac_range])
        occupancy /= n_injections

        # Look for the noisiest pixels
        top_left = np.array([[col_start, row_start]])
        max_occu = np.max(occupancy, axis=2)
        mask = max_occu > 1.05  # Allow a few extra hits
        noisy_list = np.argwhere(mask) + top_left
        noisy_indices = np.nonzero(mask)
        srt = np.argsort(-max_occu[noisy_indices])
        noisy_indices = tuple(x[srt] for x in noisy_indices)
        noisy_list = noisy_list[srt]
        if len(noisy_list):
            mi = min(len(noisy_list), 100)
            tmp = "\n".join(
                ",    ".join(f"({a}, {b}) = {float(c):.1f}" for (a, b), c in g)
                for g in groupwise(zip(noisy_list[:mi], max_occu[tuple(x[:mi] for x in noisy_indices)]), 4))
            plt.annotate(
                split_long_text(
                    "Noisiest pixels (col, row) = occupancy$_{max}$\n"
                    f"{tmp}"
                    f'{", ..." if len(noisy_list) > mi else ""}'
                    f"\nTotal = {len(noisy_list)} pixels ({len(noisy_list)/row_n/col_n:.1%})"
                ), (0.5, 0.5), ha='center', va='center')
        else:
            plt.annotate("No noisy pixel found.", (0.5, 0.5), ha='center', va='center')
        plt.gca().set_axis_off()
        pdf.savefig(); plt.clf()

        # S-Curve as 2D histogram
        occupancy_charges = occupancy_edges[2].astype(np.float32)
        occupancy_charges = (occupancy_charges[:-1] + occupancy_charges[1:]) / 2
        occupancy_charges = np.tile(occupancy_charges, (col_n, row_n, 1))
        for fc, lc, name in chain([(0, 511, 'All FEs')], FRONTENDS):
            if fc >= col_stop or lc < col_start:
                continue
            fc = max(0, fc - col_start)
            lc = min(col_n - 1, lc - col_start)
            plt.hist2d(occupancy_charges[fc:lc+1,:,:].reshape(-1),
                       occupancy[fc:lc+1,:,:].reshape(-1),
                       bins=[charge_dac_bins, 150], range=[charge_dac_range, [0, 1.5]],
                       cmin=1, rasterized=True)  # Necessary for quick save and view in PDF
            plt.title(subtitle)
            plt.suptitle(f"S-Curve ({name})")
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("Occupancy")
            set_integer_ticks(plt.gca().xaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Pixels / bin")
            pdf.savefig(); plt.clf()

        # ToT vs injected charge as 2D histogram
        m = 32 if tot.max() <= 32 else 128
        for (fc, lc, name), mask in zip(chain([(0, 511, 'All FEs')], FRONTENDS), chain([slice(-1)], fe_masks)):
            if fc >= col_stop or lc < col_start:
                continue
            plt.hist2d(charge_dac[mask], tot[mask], bins=[250, m],
                       range=[[-0.5, 249.5], [-0.5, m + 0.5]],
                       cmin=1, rasterized=True)  # Necessary for quick save and view in PDF
            plt.title(subtitle)
            plt.suptitle(f"ToT curve ({name})")
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("ToT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            pdf.savefig(); plt.clf()

        # Compute the threshold for each pixel as the weighted average
        # of the injected charge, where the weights are given by the
        # occupancy such that occu = 0.5 has weight 1, occu = 0,1 have
        # weight 0, and anything in between is linearly interpolated
        # Assuming the shape is an erf, this estimator is consistent
        w = np.maximum(0, 0.5 - np.abs(occupancy - 0.5))
        threshold_DAC = average(occupancy_charges, axis=2, weights=w, invalid=0)
        m1 = int(max(charge_dac_range[0], threshold_DAC.min() - 2))
        m2 = int(min(charge_dac_range[1], threshold_DAC.max() + 2))
        for i, (fc, lc, name) in enumerate(FRONTENDS):
            if fc >= col_stop or lc < col_start:
                continue
            fc = max(0, fc - col_start)
            lc = min(col_n - 1, lc - col_start)
            th = threshold_DAC[fc:lc+1,:]
            th_mean = ufloat(np.mean(th[th>0]), np.std(th[th>0], ddof=1))
            plt.hist(th.reshape(-1), bins=m2-m1, range=[m1, m2],
                     label=f"{name} ${th_mean:L}$", histtype='step', color=f"C{i}")
        plt.title(subtitle)
        plt.suptitle("Threshold distribution")
        plt.xlabel("Threshold [DAC]")
        plt.ylabel("Pixels / bin")
        set_integer_ticks(plt.gca().yaxis)
        plt.legend()
        plt.grid(axis='y')
        pdf.savefig(); plt.clf()

        # Threshold map
        plt.axes((0.125, 0.11, 0.775, 0.72))
        plt.pcolormesh(occupancy_edges[0], occupancy_edges[1], threshold_DAC.transpose(),
                       rasterized=True)  # Necessary for quick save and view in PDF
        plt.title(subtitle)
        plt.suptitle("Threshold map")
        plt.xlabel("Column")
        plt.ylabel("Row")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        cb = plt.colorbar()
        cb.set_label("Threshold [DAC]")
        frontend_names_on_top()
        pdf.savefig(); plt.clf()

        # Compute the noise (the width of the up-slope of the s-curve)
        # as a variance with the weights above
        noise_DAC = np.sqrt(average((occupancy_charges - np.expand_dims(threshold_DAC, -1))**2, axis=2, weights=w, invalid=0))
        m = int(np.ceil(noise_DAC.max(initial=0, where=np.isfinite(noise_DAC)))) + 1
        for i, (fc, lc, name) in enumerate(FRONTENDS):
            if fc >= col_stop or lc < col_start:
                continue
            fc = max(0, fc - col_start)
            lc = min(col_n - 1, lc - col_start)
            ns = noise_DAC[fc:lc+1,:]
            noise_mean = ufloat(np.mean(ns[ns>0]), np.std(ns[ns>0], ddof=1))
            plt.hist(ns.reshape(-1), bins=min(20*m, 100), range=[0, m],
                     label=f"{name} ${noise_mean:L}$", histtype='step', color=f"C{i}")
        plt.title(subtitle)
        plt.suptitle(f"Noise (width of s-curve slope) distribution")
        plt.xlabel("Noise [DAC]")
        plt.ylabel("Pixels / bin")
        set_integer_ticks(plt.gca().yaxis)
        plt.grid(axis='y')
        plt.legend()
        pdf.savefig(); plt.clf()

        # Noise map
        plt.axes((0.125, 0.11, 0.775, 0.72))
        plt.pcolormesh(occupancy_edges[0], occupancy_edges[1], noise_DAC.transpose(),
                       rasterized=True)  # Necessary for quick save and view in PDF
        plt.title(subtitle)
        plt.suptitle("Noise (width of s-curve slope) map")
        plt.xlabel("Column")
        plt.ylabel("Row")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        cb = plt.colorbar()
        cb.set_label("Noise [DAC]")
        frontend_names_on_top()
        pdf.savefig(); plt.clf()

        # Time since previous hit vs ToT
        m = 32 if tot.max() <= 32 else 128
        for (fc, lc, name), mask in zip(chain([(0, 511, 'All FEs')], FRONTENDS), chain([slice(-1)], fe_masks)):
            if fc >= col_stop or lc < col_start:
                continue
            plt.hist2d(tot[mask][1:], np.diff(hits["timestamp"][mask]) / 640.,
                       bins=[m, 479], range=[[-0.5, m + 0.5], [25e-3, 12]],
                       cmin=1, rasterized=True)  # Necessary for quick save and view in PDF
            plt.title(subtitle)
            plt.suptitle(f"Time between hits ({name})")
            plt.xlabel("ToT [25 ns]")
            plt.ylabel("$\\Delta t_{{token}}$ from previous hit [μs]")
            set_integer_ticks(plt.gca().xaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            pdf.savefig(); plt.clf()

        # Scan pattern
        # Find the index of the first hit on each pixel with an inj.
        # charge chosen way above threshold
        cols = np.tile(np.arange(col_start, col_stop), (row_n, 1))
        rows = np.tile(np.arange(row_start, row_stop).reshape(-1, 1), (1, col_n))
        tmp = hits[charge_dac == int(threshold_DAC.max()) + 5]
        first_hit_index = np.argmax(
            (tmp["col"].reshape((-1, 1, 1)) == cols.reshape(1, row_n, col_n))
            & (tmp["row"].reshape((-1, 1, 1)) == rows.reshape(1, row_n, col_n)),
            axis=0)
        del cols, rows
        fts = tmp["timestamp"][first_hit_index].astype(np.int64)  # In 1/(640 MHz) units
        del tmp
        fts -= fts.min()  # Time wrt first hit
        # Convert to units of minimum interval between different pixels (8.75 μs * n_injections)
        fts //= 1400 * 4 * n_injections  # Integer type => round to nearest smaller or equal integer
        # Convert to sequential values
        _, fts_unique = np.unique(fts, return_inverse=True)
        fts_unique = fts_unique.reshape(fts.shape)
        del fts
        plt.axes((0.125, 0.11, 0.775, 0.72))
        cm = ListedColormap(sum(COLOR_GRADIENTS[:min(row_n, 4)], []))
        plt.pcolormesh(
            np.arange(col_start, col_stop+1), np.arange(row_start, row_stop+1),
            fts_unique, cmap=cm,
            rasterized=True)  # Necessary for quick save and view in PDF
        plt.title(subtitle)
        plt.suptitle(f"Scan pattern (pixels injected at the same time)")
        plt.xlabel("Column")
        plt.ylabel("Row")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        cb = integer_ticks_colorbar()
        cb.set_label("Same value = injected together")
        frontend_names_on_top()
        pdf.savefig(); plt.clf()

        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file", nargs="*",
        help="The _threshold_scan_interpreted.h5 file(s)."
             " If not given, looks in output_data/module_0/chip_0.")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Overwrite plots when already present.")
    args = parser.parse_args()

    files = []
    if args.input_file:  # If anything was given on the command line
        for pattern in args.input_file:
            files.extend(glob.glob(pattern, recursive=True))
    else:
        files.extend(glob.glob("output_data/module_0/chip_0/*_threshold_scan_interpreted.h5"))
    files.sort()

    for fp in tqdm(files):
        try:
            main(fp, args.overwrite)
        except Exception:
            print(traceback.format_exc())
