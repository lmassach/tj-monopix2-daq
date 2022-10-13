#!/usr/bin/env python3
"""Computes Cinj from the NPZ files produced by plot_peak_Fe55 and plot_tot_vs_qinj."""
import argparse
import glob
from itertools import product
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from plot_utils_pisa import *

VIRIDIS_WHITE_UNDER = matplotlib.cm.get_cmap('viridis').copy()
VIRIDIS_WHITE_UNDER.set_under('w')

NOMINAL_C_INJ = 10.1  # e- / DAC
FE55_CHARGE = 1616  # e-


@np.errstate(all='ignore')
def dac(tot, a, c, t):
    return (a*t + tot + np.sqrt(a*t*a*t - 2*a*t*tot * tot*tot + 4*a*c)) / (2*a)


def load_without_overwriting(input_data, output_array, name):
    overwritten = (~np.isnan(output_array)) & (~np.isnan(input_data[str(name)]))
    n_overwritten = np.count_nonzero(overwritten)
    if n_overwritten:
        print("WARNING Multiple values of threshold for the same pixel(s)")
        print(f"    count={n_overwritten}, file={fp}")
    output_array[:] = np.where(np.isnan(output_array), input_data[str(name)], output_array)


def map_plot(data, title, label, **kwargs):
    plt.axes((0.125, 0.11, 0.775, 0.72))
    edges = np.linspace(0, 512, 513, endpoint=True)
    plt.pcolormesh(edges, edges, data.transpose(), rasterized=True, **kwargs)
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    cb = plt.colorbar()
    cb.set_label(label)
    frontend_names_on_top()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_file", help="The output PDF.")
    parser.add_argument("input_file", nargs="+",
                        help="The _ToTvsQinj.npz and _Fe55.npz file(s).")
    parser.add_argument("--shift", type=float, default=0.0,
                        help="Threshold shift (due to Vinj saturation) to be"
                             " subtracted. Unit: DAC. Default: 0.")
    args = parser.parse_args()

    files = []
    for pattern in args.input_file:
        files.extend(glob.glob(pattern, recursive=True))
    files.sort()

    # Load results from NPZ files
    mus = np.full((512, 512), np.nan)  # Fe55 peak center and width
    # sigmas = np.full((512, 512), np.nan)
    a = np.full((512, 512), np.nan)  # ToT-DAC conversion function parameters
    c = np.full((512, 512), np.nan)
    t = np.full((512, 512), np.nan)
    for fp in tqdm(files, unit="file"):
        with np.load(fp) as data:
            try:
                load_without_overwriting(data, mus, 'mus')
                # load_without_overwriting(data, sigmas, 'sigmas')
            except Exception:
                try:
                    load_without_overwriting(data, a, 'a')
                    load_without_overwriting(data, c, 'c')
                    load_without_overwriting(data, t, 't')
                except Exception:
                    print("WARNING: neither a _ToTvsQinj.npz nor a _Fe55.npz:", fp)

    # Do the plotting
    output_file = args.output_file
    if not output_file.lower().endswith(".pdf"):
        output_file += ".pdf"
    with PdfPages(output_file) as pdf:
        plt.figure(figsize=(6.4, 4.8))

        plt.annotate(
            split_long_text(
                "This file was generated by joining the following\n\n"
                + "\n".join(files)
                ), (0.5, 0.5), ha='center', va='center')
        plt.gca().set_axis_off()
        pdf.savefig(); plt.clf()

        # Map of input parameters (sanity check)
        map_plot(mus, "Fe55 peak center", "Fe55 peak center [25 ns]")
        pdf.savefig(); plt.clf()
        map_plot(a, "$a$ parameter", "$a$")
        pdf.savefig(); plt.clf()
        map_plot(c, "$c$ parameter", "$c$")
        pdf.savefig(); plt.clf()
        map_plot(t, "$t$ parameter", "$t$")
        pdf.savefig(); plt.clf()

        c_inj = np.full((512, 512), np.nan)
        with np.errstate(all='ignore'):
            c_inj = FE55_CHARGE / (dac(mus, a, c, t) - args.shift)

        # Cinj histogram
        m1 = max(0, np.nan_to_num(c_inj, posinf=0, neginf=0).min() - 1)
        m2 = min(20, np.nan_to_num(c_inj, posinf=0, neginf=0).max() + 1)
        for i, (fc, lc, name) in enumerate(FRONTENDS):
            plt.hist(c_inj[fc:lc,:].reshape(-1), bins=50, range=[m1, m2],
                        histtype='step', color=f'C{i}', label=name)
        plt.axvline(NOMINAL_C_INJ, c='k', ls='--', label="Nominal $C_{inj}$")
        plt.title("$C_{inj}$ distribution")
        plt.xlabel("$C_{inj}$ [e- / DAC]")
        plt.ylabel("Pixels / bin")
        plt.legend()
        plt.grid()
        pdf.savefig(); plt.clf()

        # Cinj map
        map_plot(c_inj, "$C_{inj}$ distribution", "$C_{inj}$ [e- / DAC]")
        pdf.savefig(); plt.clf()

        # Charge loss histogram
        for i, (fc, lc, name) in enumerate(FRONTENDS):
            plt.hist(c_inj[fc:lc,:].reshape(-1) / NOMINAL_C_INJ, bins=50,
                        range=[m1 / NOMINAL_C_INJ, m2 / NOMINAL_C_INJ],
                        histtype='step', color=f'C{i}', label=name)
        plt.suptitle("Charge collection distribution")
        plt.title("$C_{inj}/C_{inj,nominal}$")
        plt.xlabel("Collected charge fraction")
        plt.ylabel("Pixels / bin")
        plt.legend()
        plt.grid()
        pdf.savefig(); plt.clf()

        # Cinj map
        map_plot(c_inj / NOMINAL_C_INJ, "Charge collection distribution", "Collected charge fraciton")
        pdf.savefig(); plt.clf()


        # TODO Compute Cinj = 1616 / dac_from_tot(Fe55_peak)
        # TODO Compute Cinj / 10.1 (deviation from known value, or charge loss for HV)
