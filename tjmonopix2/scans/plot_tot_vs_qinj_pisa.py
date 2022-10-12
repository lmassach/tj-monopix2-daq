#!/usr/bin/env python3
"""Per-pixel fit of ToT vs Qinj."""
import argparse
import glob
from itertools import product, combinations
import os
import traceback
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import tables as tb
from tqdm import tqdm
from uncertainties import ufloat
from plot_utils_pisa import *


@np.errstate(all='ignore')
def fit_func(x, a, b, c, t):
    return np.where(x < t, 0, np.maximum(0, a*x + b - c/(x-t)))


def main(input_file, overwrite=False, pixels=[0, 511, 0, 511]):
    output_file = os.path.splitext(input_file)[0] + "_ToTvsQinj.pdf"
    if os.path.isfile(output_file) and not overwrite:
        return
    print("Plotting", input_file)
    # Open file and fill histograms (actual plotting below)
    with tb.open_file(input_file) as f:
        cfg = get_config_dict(f)

        n_hits = f.root.Dut.shape[0]

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
        n_injections = int(cfg["configuration_in.scan.scan_config.n_injections"])
        the_vh = int(cfg["configuration_in.scan.scan_config.VCAL_HIGH"])
        start_vl = int(cfg["configuration_in.scan.scan_config.VCAL_LOW_start"])
        stop_vl = int(cfg["configuration_in.scan.scan_config.VCAL_LOW_stop"])
        step_vl = int(cfg["configuration_in.scan.scan_config.VCAL_LOW_step"])
        charge_dac_values = np.array([
            the_vh - x for x in range(start_vl, stop_vl, step_vl)])
        subtitle = f"VH = {the_vh}, VL = {start_vl}..{stop_vl} (step {step_vl})"
        charge_dac_bins = len(charge_dac_values)
        charge_dac_range = [min(charge_dac_values) - 0.5, max(charge_dac_values) + 0.5]
        row_start = int(cfg["configuration_in.scan.scan_config.start_row"])
        row_stop = int(cfg["configuration_in.scan.scan_config.stop_row"])
        col_start = int(cfg["configuration_in.scan.scan_config.start_column"])
        col_stop = int(cfg["configuration_in.scan.scan_config.stop_column"])
        row_n, col_n = row_stop - row_start, col_stop - col_start

        # Prepare histograms
        counts = np.zeros((col_n, row_n, charge_dac_bins))
        sum_tot = np.zeros((col_n, row_n, charge_dac_bins))

        # Process one chunk of data at a time
        csz = 2**24
        for i_first in tqdm(range(0, n_hits, csz), unit="chunk", disable=n_hits/csz<=1):
            i_last = min(n_hits, i_first + csz)

            # Load hits
            hits = f.root.Dut[i_first:i_last]
            with np.errstate(all='ignore'):
                tot = (hits["te"] - hits["le"]) & 0x7f

            # Determine injected charge for each hit
            vh = scan_params["vcal_high"][hits["scan_param_id"]]
            vl = scan_params["vcal_low"][hits["scan_param_id"]]
            charge_dac = vh - vl
            del vl, vh

            # Count hits per pixel per injected charge value
            counts_tmp, counts_edges = np.histogramdd(
                (hits["col"], hits["row"], charge_dac),
                bins=[col_n, row_n, charge_dac_bins],
                range=[[col_start, col_stop], [row_start, row_stop], charge_dac_range])
            counts += counts_tmp
            del counts_tmp

            # Sum the ToT of the hits (per pixel per Qinj value) to later compute the average
            tot_tmp, _ = np.histogramdd(
                (hits["col"], hits["row"], charge_dac),
                bins=[col_n, row_n, charge_dac_bins],
                range=[[col_start, col_stop], [row_start, row_stop], charge_dac_range],
                weights=tot)
            sum_tot += tot_tmp
            del tot_tmp

    # Compute the average ToT per pixel per Qinj value
    with np.errstate(all='ignore'):
        avg_tot = sum_tot / counts

    # Do the actual fitting and plotting
    with PdfPages(output_file) as pdf:
        plt.figure(figsize=(6.4, 4.8))

        draw_summary(input_file, cfg)
        pdf.savefig(); plt.clf()

        if n_hits == 0:
            plt.annotate("No hits recorded!", (0.5, 0.5), ha='center', va='center')
            plt.gca().set_axis_off()
            pdf.savefig(); plt.clf()
            return

        # Distribution of average pixel ToT vs Qinj
        plt.hist2d(
            np.tile(charge_dac_values, (col_n, row_n, 1)).reshape(-1), avg_tot.reshape(-1),
            bins=[charge_dac_bins, 128], range=[charge_dac_range, [0, 128]],
            cmin=1, rasterized=True)
        plt.title(subtitle)
        plt.suptitle("Distribution of average ToT per pixel vs injected charge")
        plt.xlabel("Injected charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        cb = integer_ticks_colorbar()
        cb.set_label("Pixels / bin")
        pdf.savefig(); plt.clf()

        # Per-pixel fitting
        np.nan_to_num(  # Modifies original array avg_tot, removing non-finites (curve_fit can't handle those)
            avg_tot, posinf=0.0, neginf=0.0, copy=False)
        # Rows and columns to run the fitting on
        col_first, col_last, row_first, row_last = pixels
        col_first = max(col_first, col_start)
        col_last = min(col_last, col_stop - 1)
        row_first = max(row_first, row_start)
        row_last = min(row_last, row_stop - 1)
        # Results storage
        a = np.full((512, 512), np.nan)
        b = np.full((512, 512), np.nan)
        c = np.full((512, 512), np.nan)
        t = np.full((512, 512), np.nan)
        n_pix = 1 + max(0, (col_last + 1 - col_first) * (row_last + 1 - row_first))
        n_failed_fit = 0
        n_broken_pix = 0
        for i, (col, row) in tqdm(
            enumerate(product(range(col_first, col_last+1), range(row_first, row_last+1))),
            unit="pixel", disable=n_pix<250, total=n_pix):
            tot = avg_tot[col-col_start, row-row_start]
            if not np.any(tot > 2):
                n_broken_pix += 1
                continue  # Skip broken pixels in last 16 cols
            try:
                th = charge_dac_values[np.argmax(tot > 0)]
                max_tot = np.max(tot)
                max_pos = charge_dac_values[np.argmax(tot)]
                a0 = max_tot / (max_pos - th)
                p0 = (a0, max_tot - a0 * max_pos + 10, 200, th)
                popt, pcov = curve_fit(fit_func, charge_dac_values, tot, p0=p0)
            except Exception:
                popt = np.full(4, np.nan)
                pcov = np.full((4, 4), np.nan)
                n_failed_fit += 1
            pstd = np.sqrt(pcov.diagonal())
            if i % max(2, n_pix//10) == 0 or (np.isnan(popt[0]) and n_failed_fit < 5):
                plt.plot(charge_dac_values, tot, '.', label='Data')
                if np.isnan(popt[0]):
                    fit_res = "\n".join(f"${n}={m:.3g}$" for m, n in zip(p0, "abct"))
                    plt.plot(charge_dac_values, fit_func(charge_dac_values, *p0),
                             label=f'Initial fit parameters\n{fit_res}')
                    ylim = plt.ylim()
                    plt.plot(charge_dac_values, p0[0]*charge_dac_values + p0[1], '--')
                    plt.ylim(*ylim)
                else:
                    fit_res = "\n".join(f"${n}={ufloat(m,s):L}$" for m, s, n in zip(popt, pstd, "abct"))
                    plt.plot(charge_dac_values, fit_func(charge_dac_values, *popt),
                             label=f'Fit\n{fit_res}')
                    ylim = plt.ylim()
                    plt.plot(charge_dac_values, popt[0]*charge_dac_values + popt[1], '--')
                    plt.ylim(*ylim)
                the_pixel = "all pixels" if col is None else str((col, row))
                plt.title(f"{the_pixel} (fit failed)" if np.isnan(popt[0]) else f"Fit to {the_pixel}")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("Average ToT [25 ns]")
                plt.legend()
                pdf.savefig(); plt.clf()
            a[col,row] = popt[0]
            b[col,row] = popt[1]
            c[col,row] = popt[2]
            t[col,row] = popt[3]

        # Parameters distribution
        bins_ranges = {
            'a': (50, [0, 0.3]),
            'b': (50, [-10, 25]),
            'c': (50, [0, 1000]),
            't': (50, [0, 50])}
        for name, unit, data in zip("abct", ["25 ns / DAC", "25 ns", "25 ns DAC", "DAC"], (a, b, c, t)):
            for i, (fc, lc, fe_name) in enumerate(FRONTENDS):
                if fc > col_last or lc < col_first:
                    continue
                fc = max(fc, col_first)
                lc = min(lc, col_last+1)
                bi, ra = bins_ranges[name]
                plt.hist(data[fc:lc,:].reshape(-1), bins=bi, range=ra,
                         color=f'C{i}', histtype='step', label=fe_name)
            plt.title(f"Fit failed for {n_failed_fit} pixels ({n_failed_fit/max(1,(n_pix-1)):.2%}). {n_broken_pix} pixels are broken ({n_broken_pix/max(1,(n_pix-1)):.2%}).")
            plt.suptitle(f"${name}$ distribution")
            plt.xlabel(f"${name}$ [{unit}]")
            plt.ylabel("Pixels / bin")
            plt.legend()
            plt.grid()
            pdf.savefig(); plt.clf()

        for (name1, data1), (name2, data2) in combinations(zip("abct", (a, b, c, t)), 2):
            b1, r1 = bins_ranges[name1]
            b2, r2 = bins_ranges[name2]
            plt.hist2d(
                data1[col_first:col_last+1,row_first:row_last+1].reshape(-1),
                data2[col_first:col_last+1,row_first:row_last+1].reshape(-1),
                bins=[b1, b2], range=[r1, r2], rasterized=True, cmin=0)
            plt.title(f"Combined distribution of ${name1}$ and ${name2}$")
            plt.xlabel(name1)
            plt.ylabel(name2)
            pdf.savefig(); plt.clf()

        plt.close()

        # Save fit results
        np.savez_compressed(
            os.path.splitext(output_file)[0] + ".npz",
            a=a, b=b, c=c, t=t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file", nargs="*",
        help="The _threshold_scan_interpreted.h5 file(s)."
             " If not given, looks in output_data/module_0/chip_0.")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Overwrite plots when already present.")
    parser.add_argument("--pixels", nargs=4, default=[0, 511, 0, 511], type=int,
                        metavar=("FIRST_COL", "LAST_COL", "FIRST_ROW", "LAST_ROW"),
                        help="Single pixels to fit (last row/col are included). Default: all.")
    args = parser.parse_args()

    files = []
    if args.input_file:  # If anything was given on the command line
        for pattern in args.input_file:
            files.extend(glob.glob(pattern, recursive=True))
    else:
        files.extend(glob.glob("output_data/module_0/chip_0/*_threshold_scan_interpreted.h5"))
    files.sort()

    for fp in tqdm(files, unit="file"):
        try:
            main(fp, args.overwrite, args.pixels)
        except Exception:
            print(traceback.format_exc())
