#!/usr/bin/env python3
"""Standard plots like hitmap and ToT histogram (HistOcc and HistToT not required)."""
import argparse
import glob
import os
import traceback
from itertools import product
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import tables as tb
from tqdm import tqdm
from plot_utils_pisa import *


def main(input_files, overwrite=False, log_tot=False, output_file=None):
    if output_file is None:
        output_file = os.path.splitext(input_files[0])[0] + ".pdf"
    if os.path.isfile(output_file) and not overwrite:
        return

    # Prepare histograms
    counts2d = np.zeros((512, 512))
    tot1d = [np.zeros(128) for _ in range(len(FRONTENDS))]
    tot1d_single_hits = [np.zeros(128) for _ in range(len(FRONTENDS))]
    tot2d = np.zeros((512, 512))
    counts2d16 = np.zeros((32, 32))
    cfg = []
    tdac = []
    n_total_hits = 0

    for input_file in tqdm(input_files, disable=len(input_files)<2):
        print("Processing", input_file)
        with tb.open_file(input_file) as f:
            cfg.append(get_config_dict(f))
            tdac.append(f.root.configuration_out.chip.masks.tdac[:])

            try:
                n_hits = f.root.Dut.shape[0]
            except tb.NoSuchNodeError:
                continue
            n_total_hits += n_hits

            # Process one chunk of data at a time
            csz = 2**24
            for i_first in tqdm(range(0, n_hits, csz), unit="chunk", disable=n_hits/csz<=1):
                i_last = min(n_hits, i_first + csz)

                # Load hits
                hits = f.root.Dut[i_first:i_last]
                with np.errstate(all='ignore'):
                    tot = (hits["te"] - hits["le"]) & 0x7f
                fe_masks = [(hits["col"] >= fc) & (hits["col"] <= lc) for fc, lc, _ in FRONTENDS]
                single_hits_mask = is_single_hit_event(hits["timestamp"])

                counts2d_tmp, counts2d_edges, _ = np.histogram2d(
                    hits["col"], hits["row"], bins=[512, 512], range=[[0, 512], [0, 512]])
                counts2d += counts2d_tmp
                del counts2d_tmp

                for i, mask in enumerate(fe_masks):
                    tot1d_tmp, tot1d_edges = np.histogram(
                        tot[mask], bins=128, range=[-0.5, 127.5])
                    tot1d[i] += tot1d_tmp
                    tot1d_tmp, tot1d_edges = np.histogram(
                        tot[single_hits_mask & mask], bins=128, range=[-0.5, 127.5])
                    tot1d_single_hits[i] += tot1d_tmp
                    del tot1d_tmp

                tot2d_tmp, tot2d_edges, _  = np.histogram2d(
                    hits["col"], hits["row"], bins=[512, 512], range=[[0, 512], [0, 512]],
                    weights=tot)
                tot2d += tot2d_tmp
                del tot2d_tmp

                counts2d16_tmp, edges16, _ = np.histogram2d(
                    hits["col"], hits["row"], bins=[32, 32], range=[[0, 512], [0, 512]])
                counts2d16 += counts2d16_tmp
                del counts2d16_tmp

                del hits, tot, fe_masks

    with PdfPages(output_file) as pdf:
        plt.figure(figsize=(6.4, 4.8))

        if len(input_files) > 1:
            plt.annotate(
                split_long_text(
                    "This file was generated by joining the following\n\n"
                    + "\n".join(input_files)
                    ), (0.5, 0.5), ha='center', va='center')
            plt.gca().set_axis_off()
            pdf.savefig(); plt.clf()

        for input_file, c, t in zip(input_files, cfg, tdac):
            draw_summary(input_file, c)
            pdf.savefig(); plt.clf()

            # TDAC map
            plt.axes((0.125, 0.11, 0.775, 0.72))
            nzc, nzr = np.nonzero(t)
            if not nzc.size:
                fc, lc, fr, lr = 0, 512, 0, 512
            else:
                fc, lc = nzc.min(), nzc.max() + 1
                fr, lr = nzr.min(), nzr.max() + 1
            del nzc, nzr
            plt.pcolormesh(np.arange(fc, lc + 1), np.arange(fr, lr + 1),
                           t[fc:lc, fr:lr].transpose(),
                           vmin=-0.5, vmax=7.5, cmap=TDAC_CMAP, rasterized=True)
            plt.xlabel("Column")
            plt.ylabel("Row")
            plt.title("Map of TDAC values")
            frontend_names_on_top()
            integer_ticks_colorbar().set_label("TDAC")
            pdf.savefig(); plt.clf()
        # print("Summary")

        if n_total_hits == 0:
            plt.annotate("No hits recorded!", (0.5, 0.5), ha='center', va='center')
            plt.gca().set_axis_off()
            pdf.savefig(); plt.clf()
            return

        # Histogram of hits per pixel
        m = np.quantile(counts2d[counts2d > 0], 0.99) * 1.2 if np.any(counts2d > 0) else 1
        bins = 100 if m > 200 else int(max(m, 5))
        for fc, lc, name in FRONTENDS:
            plt.hist(counts2d[fc:lc+1,:].reshape(-1), label=name, histtype='step',
                     bins=bins, range=[0.5, max(m, 5) + 0.5])
        plt.title("Hits per pixel")
        plt.xlabel("Number of hits")
        plt.ylabel("Pixels / bin")
        # plt.yscale("log")
        plt.grid(axis='y')
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        plt.legend()
        pdf.savefig(); plt.clf()
        # print("Hits hist")

        # Histogram of ToT
        for (_, _, name), hist in zip(FRONTENDS, tot1d):
            plt.step((tot1d_edges[1:] + tot1d_edges[:-1]) / 2,
                     hist, where='mid', label=name)
        plt.title("ToT")
        plt.xlabel("ToT [25 ns]")
        plt.ylabel("Hits / bin")
        #plt.xlim(10, 120)
        #plt.xlim(10, 128)
        #plt.ylim(0, max(h[10:].max() for h in tot1d) * 1.2)
        plt.grid(axis='y')
        if log_tot:
            # plt.ylim(0.9, max(h[10:120].max() for h in tot1d) * 1.2)
            plt.ylim(0.9, max(h.max() for h in tot1d) * 1.2)
            plt.yscale('log')
            set_integer_ticks(plt.gca().xaxis)
        else:
            plt.ylim(0, max(h[10:120].max() for h in tot1d) * 1.2)
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        plt.legend()
        pdf.savefig(); plt.clf()
        # print("ToT Hist")

        # Histogram of ToT (single hits)
        for (_, _, name), hist in zip(FRONTENDS, tot1d_single_hits):
            plt.step((tot1d_edges[1:] + tot1d_edges[:-1]) / 2,
                     hist, where='mid', label=name)
        plt.title("ToT (isolated hits only)")
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
        # print("ToT Hist")

        # Hit map
        cmap = matplotlib.cm.get_cmap("viridis").copy()
        cmap.set_over("r")
        plt.pcolormesh(counts2d_edges, counts2d_edges, counts2d.transpose(),
                       vmin=0, vmax=m, cmap=cmap, rasterized=True)  # Necessary for quick save and view in PDF
        plt.title("Hit map")
        plt.xlabel("Col")
        plt.ylabel("Row")
        cb = integer_ticks_colorbar()
        cb.set_label("Hits / pixel")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        frontend_names_on_top()
        pdf.savefig(); plt.clf()

        ZOOM_STEP = 64
        for i, j in product(range(0, 512, ZOOM_STEP), range(0, 512, ZOOM_STEP)):
            i1 = min(i+ZOOM_STEP, 512)
            j1 = min(j+ZOOM_STEP, 512)
            if not np.any(counts2d[i:i1,j:j1]):
                continue
            plt.pcolormesh(counts2d_edges[i:i1+1], counts2d_edges[j:j1+1], counts2d[i:i1,j:j1].transpose(),
                           vmin=0, vmax=m, rasterized=True)  # Necessary for quick save and view in PDF
            plt.title("Hit map (subregion)")
            plt.xlabel("Col")
            plt.ylabel("Row")
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / pixel")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            frontend_names_on_top()
            plt.xlim(i, i1)
            plt.ylim(j, j1)
            pdf.savefig(); plt.clf()
        # print("Hitmap")

        # Map of the average ToT
        with np.errstate(all='ignore'):
            totavg = tot2d / counts2d
        plt.pcolormesh(tot2d_edges, tot2d_edges, totavg.transpose(),
                       vmin=-0.5, vmax=25.5, rasterized=True)  # Necessary for quick save and view in PDF
        plt.title("Average ToT map")
        plt.xlabel("Col")
        plt.ylabel("Row")
        cb = integer_ticks_colorbar()
        cb.set_label("ToT [25 ns]")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        frontend_names_on_top()
        pdf.savefig(); plt.clf()

        for i, j in product(range(0, 512, ZOOM_STEP), range(0, 512, ZOOM_STEP)):
            i1 = min(i+ZOOM_STEP, 512)
            j1 = min(j+ZOOM_STEP, 512)
            if not np.any(counts2d[i:i1,j:j1]):
                continue
            plt.pcolormesh(tot2d_edges[i:i1+1], tot2d_edges[j:j1+1], totavg[i:i1,j:j1].transpose(),
                           vmin=-0.5, vmax=25.5, rasterized=True)  # Necessary for quick save and view in PDF
            plt.title("Average ToT map (subregion)")
            plt.xlabel("Col")
            plt.ylabel("Row")
            cb = integer_ticks_colorbar()
            cb.set_label("ToT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            frontend_names_on_top()
            plt.xlim(i, i1)
            plt.ylim(j, j1)
            pdf.savefig(); plt.clf()
        # print("ToT map")

        # Noisy pixels
        if all(c.get("configuration_in.scan.run_config.scan_id") == "source_scan" for c in cfg):
            MAX_RATE = 1  # Above this rate [Hz] pixels are marked noisy
            scan_time = 0
            for i, c in enumerate(cfg):
                try:
                    scan_time += float(c["configuration_in.scan.scan_config.scan_time"])
                except Exception:
                    print(f"WARNING: could not determine scan time from {input_files[i]}")
            max_hits = scan_time * MAX_RATE
            mask = counts2d > max_hits
            plt.axes((0.125, 0.11, 0.775, 0.72))
            plt.pcolormesh(counts2d_edges, counts2d_edges, 1 * mask.transpose(),
                           vmin=0, vmax=1, rasterized=True)  # Necessary for quick save and view in PDF
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
                mi = min(len(noisy_list), 10)
                tmp = "\n".join(
                    ",    ".join(f"({a}, {b}) = {float(c)/scan_time:.3g}" for (a, b), c in g)
                    for g in groupwise(zip(noisy_list[:mi], counts2d[tuple(x[:mi] for x in noisy_indices)]), 1))
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
        # print("Noisy")

        # Source positioning
        m = np.quantile(counts2d16[counts2d16 > 0], 0.99) * 1.2 if np.any(counts2d > 0) else 1
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
        # print("Source")

        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file", nargs="*",
        help="The _interpreted.h5 file(s). If not given, looks in output_data/module_0/chip_0.")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Overwrite plots when already present.")
    parser.add_argument("-j", "--join", metavar="OUTPUT_FILE.PDF", default=None,
                        help="Join all input files and put results in the given PDF.")
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

    if args.join is None:
        for fp in tqdm(files):
            try:
                main([fp], args.overwrite, args.log_tot)
            except Exception:
                print(traceback.format_exc())
    else:
        main(files, args.overwrite, args.log_tot, args.join)
