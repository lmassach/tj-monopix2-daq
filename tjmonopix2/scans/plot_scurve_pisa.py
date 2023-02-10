#!/usr/bin/env python3
"""Plots the results of scan_threshold (HistOcc and HistToT not required)."""
import argparse
import glob
from itertools import chain
import os
import traceback
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
import tables as tb
from tqdm import tqdm
from uncertainties import ufloat
from plot_utils_pisa import *

VIRIDIS_WHITE_UNDER = matplotlib.cm.get_cmap('viridis').copy()
VIRIDIS_WHITE_UNDER.set_under('w')


def s_curve(x, mu, sigma):
    return 0.5 + 0.5 * erf((x - mu) / np.sqrt(2) / sigma)


@np.errstate(all='ignore')
def average(a, axis=None, weights=1, invalid=np.NaN):
    """Like np.average, but returns `invalid` instead of crashing if the sum of weights is zero."""
    return np.nan_to_num(np.sum(a * weights, axis=axis).astype(float) / np.sum(weights, axis=axis).astype(float), nan=invalid)


def main(input_file, overwrite=False):
    output_file = os.path.splitext(input_file)[0] + "_scurve.pdf"
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
        charge_dac_values = [
            the_vh - x for x in range(start_vl, stop_vl, step_vl)]
        subtitle = f"VH = {the_vh}, VL = {start_vl}..{stop_vl} (step {step_vl})"
        charge_dac_bins = len(charge_dac_values)
        charge_dac_range = [min(charge_dac_values) - 0.5, max(charge_dac_values) + 0.5]
        row_start = int(cfg["configuration_in.scan.scan_config.start_row"])
        row_stop = int(cfg["configuration_in.scan.scan_config.stop_row"])
        col_start = int(cfg["configuration_in.scan.scan_config.start_column"])
        col_stop = int(cfg["configuration_in.scan.scan_config.stop_column"])
        row_n, col_n = row_stop - row_start, col_stop - col_start

        # Prepare histograms
        occupancy = np.zeros((col_n, row_n, charge_dac_bins))
        tot_hist = [np.zeros((charge_dac_bins, 128)) for _ in range(len(FRONTENDS) + 1)]
        dt_tot_hist = [np.zeros((128, 479)) for _ in range(len(FRONTENDS) + 1)]
        dt_q_hist = [np.zeros((charge_dac_bins, 479)) for _ in range(len(FRONTENDS) + 1)]
        tot_per_pixel_hist = np.zeros((col_n, row_n, 128))

        # Process one chunk of data at a time
        csz = 2**24
        random_firing_pixels = set()
        for i_first in tqdm(range(0, n_hits, csz), unit="chunk", disable=n_hits/csz<=1):
            i_last = min(n_hits, i_first + csz)

            # Load hits
            hits = f.root.Dut[i_first:i_last]
            # Filter only the hits in the scan area (from col/row_start to col/row_stop)
            # Sometimes disabled pixels outside of the scan area still fire for some reason
            scan_area_mask = (hits["col"] >= col_start) & (hits["col"] < col_stop) & (hits["row"] >= row_start) & (hits["row"] < row_stop)
            hits = hits[scan_area_mask]
            del scan_area_mask

            # Filter hits coming from a random firing pixel (found with the code below)
            rnd_fire_mask = np.ones_like(hits, bool)
            for col, row in [(255, 344), (245, 181), (264, 425), (247, 198), (252, 378), (242, 457), (240, 496), (263, 498), (271, 511), (254, 440), (268, 457), (263, 507), (248, 479), (266, 414), (248, 497), (244, 475), (250, 415), (252, 497), (271, 477), (261, 454), (262, 508), (255, 478), (252, 472), (264, 446), (249, 473), (262, 492), (256, 478), (248, 483), (255, 498), (242, 418), (270, 507), (256, 462), (253, 445), (245, 462), (240, 475), (255, 473), (248, 467), (247, 499), (269, 499), (263, 495), (252, 458), (259, 473), (262, 496), (244, 490), (259, 509), (270, 509), (240, 468), (252, 451), (260, 434), (261, 399), (246, 426), (270, 475), (267, 467), (244, 474), (265, 387), (259, 502), (253, 431), (256, 448), (263, 481), (266, 498), (243, 475), (264, 427), (259, 477), (259, 367), (243, 511), (244, 476), (244, 485), (264, 463), (248, 446), (263, 465), (240, 463), (245, 459), (262, 509), (252, 446), (255, 479), (259, 461), (240, 490), (270, 470), (240, 499), (243, 495), (245, 425), (250, 510), (259, 369), (256, 443), (268, 417), (248, 457), (266, 493), (262, 459), (270, 472), (242, 410), (270, 490), (254, 503), (260, 406), (258, 473), (255, 456), (258, 491), (240, 485), (250, 487), (270, 465), (248, 416), (240, 460), (266, 506), (271, 432), (254, 480), (262, 463), (269, 511), (271, 441), (262, 472), (254, 507), (248, 436), (268, 423), (255, 469), (253, 499), (243, 476), (254, 473), (243, 485), (263, 421), (242, 398), (245, 305), (251, 483), (271, 461), (248, 438), (263, 457), (258, 470), (240, 473), (242, 492), (242, 501), (250, 484), (252, 447), (253, 501), (269, 497), (247, 497), (254, 475), (254, 493), (246, 489), (258, 454), (246, 498), (258, 463), (244, 418), (252, 440), (262, 442), (265, 477), (249, 499), (254, 495), (251, 435), (242, 496), (250, 479), (262, 462), (260, 501), (265, 497), (241, 497), (266, 471), (250, 454), (242, 480), (242, 498), (265, 463), (263, 411), (254, 481), (257, 498), (268, 498), (258, 442), (257, 507), (247, 442), (265, 499), (247, 460), (266, 473), (253, 473), (264, 503), (242, 482), (243, 459), (259, 443), (250, 483), (251, 457), (260, 487), (268, 500), (260, 496), (260, 505), (242, 466), (270, 427), (250, 458), (270, 436), (266, 432), (260, 489), (258, 446), (261, 463), (266, 459), (242, 459), (254, 442), (242, 486), (268, 477), (254, 469), (268, 486), (254, 478), (266, 443), (268, 495), (246, 483), (250, 444), (269, 457), (261, 483), (242, 479), (263, 511), (246, 440), (245, 395), (242, 497), (254, 462), (253, 497), (260, 475), (269, 423), (257, 497), (267, 499), (247, 441), (258, 459), (264, 502), (256, 498), (268, 463), (253, 490), (268, 481), (266, 438), (260, 486), (243, 433), (261, 469), (264, 486), (264, 495), (242, 474), (268, 474), (246, 462), (252, 496), (240, 421), (241, 475), (243, 417), (258, 436), (244, 501), (241, 484), (249, 497), (241, 493), (245, 475), (261, 471), (263, 371), (263, 499), (245, 493), (265, 459), (260, 472), (258, 429), (241, 477), (260, 499), (271, 496), (245, 477), (256, 477), (256, 486), (271, 505), (264, 499), (242, 478), (255, 497), (248, 491), (259, 387), (262, 511), (267, 507), (256, 470), (242, 462), (256, 488), (264, 501), (248, 502), (241, 463), (259, 499), (261, 441), (264, 458), (244, 489), (270, 508), (240, 494), (260, 442), (248, 486), (243, 499), (255, 501), (243, 508), (244, 482), (242, 430), (260, 478), (264, 460), (264, 469), (244, 500)]:
                rnd_fire_mask &= ~((hits["col"] == col) & (hits["row"] == row))
            hits = hits[rnd_fire_mask]
            del rnd_fire_mask

            with np.errstate(all='ignore'):
                tot = (hits["te"] - hits["le"]) & 0x7f

            # Determine injected charge for each hit
            vh = scan_params["vcal_high"][hits["scan_param_id"]]
            vl = scan_params["vcal_low"][hits["scan_param_id"]]
            charge_dac = vh - vl
            del vl, vh

            # # Filter hits with ToT < 8 at charge > 60 (from the plots I see they are random hits)
            # rnd_msk = (tot < 8) & (charge_dac > 60)
            # for col, row in np.unique(hits[rnd_msk][["col", "row"]]):
            #     random_firing_pixels.add((col, row))
            # hits = hits[~rnd_msk]
            # tot = tot[~rnd_msk]
            # charge_dac = charge_dac[~rnd_msk]
            # del rnd_msk

            fe_masks = [(hits["col"] >= fc) & (hits["col"] <= lc) for fc, lc, _ in FRONTENDS]

            # Count hits per pixel per injected charge value
            occupancy_tmp, occupancy_edges = np.histogramdd(
                (hits["col"], hits["row"], charge_dac),
                bins=[col_n, row_n, charge_dac_bins],
                range=[[col_start, col_stop], [row_start, row_stop], charge_dac_range])
            occupancy_tmp /= n_injections
            occupancy += occupancy_tmp
            del occupancy_tmp

            # Fill histogram with ToT distribution per pixel
            tot_per_pixel_tmp, tot_per_pixel_edges = np.histogramdd(
                (hits["col"], hits["row"], tot), bins=[col_n, row_n, 128],
                range=[[col_start, col_stop], [row_start, row_stop], [-0.5, 127.5]])
            tot_per_pixel_hist += tot_per_pixel_tmp
            del tot_per_pixel_tmp

            for i, ((fc, lc, _), mask) in enumerate(zip(chain([(0, 511, 'All FEs')], FRONTENDS), chain([slice(-1)], fe_masks))):
                if fc >= col_stop or lc < col_start:
                    continue

                # ToT vs injected charge as 2D histogram
                tot_hist[i] += np.histogram2d(
                    charge_dac[mask], tot[mask], bins=[charge_dac_bins, 128],
                    range=[charge_dac_range, [-0.5, 127.5]])[0]

                # Histograms of time since previous hit vs TOT and QINJ
                dt_tot_hist[i] += np.histogram2d(
                    tot[mask][1:], np.diff(hits["timestamp"][mask]) / 40.,
                    bins=[128, 479], range=[[-0.5, 127.5], [25e-3*16, 12*16]])[0]
                dt_q_hist[i] += np.histogram2d(
                    charge_dac[mask][1:], np.diff(hits["timestamp"][mask]) / 40.,
                    bins=[charge_dac_bins, 479], range=[charge_dac_range, [25e-3*16, 12*16]])[0]

            del charge_dac

        print("Random firing pixels:", random_firing_pixels)

    # Do the actual plotting
    with PdfPages(output_file) as pdf:
        plt.figure(figsize=(6.4, 4.8))

        draw_summary(input_file, cfg)
        pdf.savefig(); plt.clf()

        if n_hits == 0:
            plt.annotate("No hits recorded!", (0.5, 0.5), ha='center', va='center')
            plt.gca().set_axis_off()
            pdf.savefig(); plt.clf()
            return

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
            mi = min(len(noisy_list), 25)
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
            # print(f"Noisy pixels (n = {len(noisy_list)})")
            # print("[" + ", ".join(str((a, b)) for a, b in noisy_list) + "]")
            output_file_txt = os.path.splitext(input_file)[0]
            with open(output_file_txt + "_noisy_pixels_occu.txt", "w") as f1:
                print("[" + ", ".join(f'"{int(a)}, {int(b)}"' for a, b in noisy_list) + "]", file=f1)
        else:
            plt.annotate("No noisy pixel found.", (0.5, 0.5), ha='center', va='center')
        plt.gca().set_axis_off()
        pdf.savefig(); plt.clf()

        # Look for pixels with random ToT (those with hits with ToT ≥ 100)
        n_crazy_hits = np.sum(tot_per_pixel_hist[:,:,100:], axis=2)
        mask = n_crazy_hits > 0
        crazy_list = np.argwhere(mask) + top_left
        crazy_indices = np.nonzero(mask)
        srt = np.argsort(-n_crazy_hits[crazy_indices])
        crazy_indices = tuple(x[srt] for x in crazy_indices)
        crazy_list = crazy_list[srt]
        if len(crazy_list):
            mi = min(len(crazy_list), 50)
            tmp = "\n".join(
                ",    ".join(f"({a}, {b}) = {float(c):.0f}" for (a, b), c in g)
                for g in groupwise(zip(crazy_list[:mi], n_crazy_hits[tuple(x[:mi] for x in crazy_indices)]), 3))
            plt.annotate(
                split_long_text(
                    "Crazy pixels search (i.e. pixels with random ToT)\n"
                    "Pixels with ≥ 1 hit with ToT ≥ 100 (col, row)\n"
                    f"{tmp}"
                    f'{", ..." if len(crazy_list) > mi else ""}'
                    f"\nTotal = {len(crazy_list)} pixels ({len(crazy_list)/row_n/col_n:.1%})"
                ), (0.5, 0.5), ha='center', va='center')
            # print(f"Crazy pixels (n = {len(crazy_list)})")
            # print("\n".join(f"({a}, {b}) = {float(c):.0f}" for (a, b), c in zip(
                # crazy_list, n_crazy_hits[tuple(x for x in crazy_indices)])))
            # print("[" + ", ".join(str((a, b)) for a, b in crazy_list) + "]")
        else:
            plt.annotate("No crazy pixel found.", (0.5, 0.5), ha='center', va='center')
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
            plt.hist2d(occupancy_charges[fc:lc+1,:,:].reshape(-1) * 10,
                       occupancy[fc:lc+1,:,:].reshape(-1),
                       bins=[charge_dac_bins, 150], range=[[10*x for x in charge_dac_range], [0, 1.5]],
                       norm=LogNorm(), cmin=1, rasterized=True)  # Necessary for quick save and view in PDF
            plt.title(subtitle)
            plt.suptitle(f"S-Curve ({name})")
            plt.xlabel("Injected charge [$e^-$]")
            plt.ylabel("Occupancy")
            plt.xlim(0, 450)
            set_integer_ticks(plt.gca().xaxis)
            cb = plt.colorbar()
            cb.set_label("Pixels / bin")
            pdf.savefig(); plt.clf()

        # S-Curve for specific pixels
        for col, row in [
            (132, 200), (140, 300), (150, 400), (160, 500),
            (240, 200), (250, 300), (260, 400), (270, 500)
        ]:
            if not (col_start <= col < col_stop and row_start <= row < row_stop):
                continue
            plt.plot([x*10 for x in charge_dac_values], occupancy[col-col_start,row-row_start,:], '.-', label=str((col, row)))
        plt.title(subtitle)
        plt.suptitle(f"S-Curve of specific pixels")
        plt.xlabel("Injected charge [$e^-$]")
        plt.ylabel("Occupancy")
        plt.ylim(0, 1.1)
        plt.xlim(0, 450)
        plt.grid()
        plt.legend()
        set_integer_ticks(plt.gca().xaxis)
        pdf.savefig(); plt.clf()

        # ToT vs injected charge as 2D histogram
        for (fc, lc, name), hist in zip(chain([(0, 511, 'All FEs')], FRONTENDS), tot_hist):
            if fc >= col_stop or lc < col_start:
                continue
            plt.pcolormesh(
                occupancy_edges[2] * 10, np.linspace(-0.5, 127.5, 129, endpoint=True),
                hist.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF
            plt.title(subtitle)
            plt.suptitle(f"ToT curve ({name})")
            plt.xlabel("Injected charge [$e^-$]")
            plt.ylabel("ToT [25 ns]")
            plt.ylim(0, 64)
            plt.grid(axis='both',)
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            pdf.savefig(); plt.clf()

        # Compute the threshold and noise for each pixels by fitting
        # each s-curve with an error function
        threshold_DAC = np.zeros((col_n, row_n))
        noise_DAC = np.zeros((col_n, row_n))
        charge_dac_np = np.array(charge_dac_values)
        for c in tqdm(range(col_n), unit='col', desc='fit', delay=2):
            for r in range(row_n):
                o = np.clip(occupancy[c,r], 0, 1)
                if not (np.any(o == 0) and np.any(o == 1)):
                    continue
                thr = charge_dac_np[np.argmin(np.abs(o - 0.5))]
                popt, pcov = curve_fit(s_curve, charge_dac_np, o, p0=(thr, 1))
                if not np.all(np.isfinite(popt)) or not np.all(np.isfinite(pcov)):
                    print("Fit failed:", c + col_start, r + row_start, o)
                    continue
                threshold_DAC[c,r], noise_DAC[c,r] = popt

        #print("Pixels with THR  < 1")
        #for col, row in zip(*np.nonzero(threshold_DAC < 1)):
        #    print(f"    ({col+col_start:3d}, {row+row_start:3d}), THR = {threshold_DAC[col,row]}")
        # print("Pixels with 1 < THR < 25")
        # for col, row in zip(*np.nonzero((1 < threshold_DAC) & (threshold_DAC < 25))):
        #     print(f"    ({col+col_start:3d}, {row+row_start:3d}), THR = {threshold_DAC[col,row]}")
        # print("Pixels with THR > 50")
        # for col, row in zip(*np.nonzero(threshold_DAC > 50)):
        #     print(f"    ({col+col_start:3d}, {row+row_start:3d}), THR = {threshold_DAC[col,row]}")
        # print("First 10 pixels with 34 < THR < 36")
        # for i, (col, row) in enumerate(zip(*np.nonzero((34 < threshold_DAC) & (threshold_DAC < 36)))):
        #     if i >= 100:
        #         break
        #     print(f"    ({col+col_start:3d}, {row+row_start:3d}), THR = {threshold_DAC[col,row]}")

        # Threshold hist
        good_thr_msk = np.isfinite(threshold_DAC) & (threshold_DAC != 0)
        m1, m2 = np.quantile(threshold_DAC[good_thr_msk], [0.05, 0.95])
        m1, m2 = max(0.1, m1 - 2), m2 + 2
        for i, (fc, lc, name) in enumerate(FRONTENDS):
            if fc >= col_stop or lc < col_start:
                continue
            fc = max(0, fc - col_start)
            lc = min(col_n - 1, lc - col_start)
            th = threshold_DAC[fc:lc+1,:]
            th_mean = ufloat(np.mean(th[th>0]), np.std(th[th>0], ddof=1))
            plt.hist(th.reshape(-1) * 10, bins=100, range=[0, m2*10],
                     label=f"{name} ${th_mean*10:L}$", histtype='step', color=f"C{i}")
        plt.title(subtitle)
        plt.suptitle("Threshold distribution")
        plt.xlabel("Threshold [$e^-$]")
        plt.ylabel("Pixels / bin")
        set_integer_ticks(plt.gca().yaxis)
        plt.legend()
        plt.grid(axis='y')
        # plt.xlim(150, 300)
        # plt.yscale('log')
        pdf.savefig(); plt.clf()

        # Threshold map
        plt.axes((0.125, 0.11, 0.775, 0.72))
        plt.pcolormesh(occupancy_edges[0], occupancy_edges[1], threshold_DAC.transpose(), vmin=m1, vmax=m2,
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

        # Noise hist
        good_noise_msk = np.isfinite(noise_DAC) & (noise_DAC != 0)
        m1, m2 = np.quantile(noise_DAC[good_noise_msk], [0.05, 0.95])
        m1, m2 = max(0.01, m1 - 0.2), m2 + 0.2
        for i, (fc, lc, name) in enumerate(FRONTENDS):
            if fc >= col_stop or lc < col_start:
                continue
            fc = max(0, fc - col_start)
            lc = min(col_n - 1, lc - col_start)
            ns = noise_DAC[fc:lc+1,:]
            noise_mean = ufloat(np.mean(ns[ns>0]), np.std(ns[ns>0], ddof=1))
            plt.hist(ns.reshape(-1) * 10, bins=100, range=[0, m2*10],
                     label=f"{name} ${noise_mean*10:L}$", histtype='step', color=f"C{i}")
        plt.title(subtitle)
        plt.suptitle(f"Noise (width of s-curve slope) distribution")
        plt.xlabel("Noise (ENC) [$e^-$]")
        plt.ylabel("Pixels / bin")
        # plt.xlim(2, 10)
        set_integer_ticks(plt.gca().yaxis)
        plt.grid(axis='y')
        plt.legend()
        pdf.savefig(); plt.clf()

        # Noise map
        plt.axes((0.125, 0.11, 0.775, 0.72))
        plt.pcolormesh(occupancy_edges[0], occupancy_edges[1], noise_DAC.transpose(), vmin=m1, vmax=m2,
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
        for (fc, lc, name), hist in zip(chain([(0, 511, 'All FEs')], FRONTENDS), dt_tot_hist):
            if fc >= col_stop or lc < col_start:
                continue
            plt.pcolormesh(
                np.linspace(-0.5, 127.5, 129, endpoint=True),
                np.linspace(25e-3, 12, 480, endpoint=True),
                hist.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)
            plt.title(subtitle)
            plt.suptitle(f"Time between hits ({name})")
            plt.xlabel("ToT [25 ns]")
            plt.ylabel("$\\Delta t_{{token}}$ from previous hit [μs]")
            set_integer_ticks(plt.gca().xaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            pdf.savefig(); plt.clf()

        # Time since previous hit vs injected charge
        m = 32 if tot.max() <= 32 else 128
        for (fc, lc, name), hist in zip(chain([(0, 511, 'All FEs')], FRONTENDS), dt_q_hist):
            if fc >= col_stop or lc < col_start:
                continue
            plt.pcolormesh(
                occupancy_edges[2],
                np.linspace(25e-3, 12, 480, endpoint=True),
                hist.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)
            plt.title(subtitle)
            plt.suptitle(f"Time between hits ({name})")
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("$\\Delta t_{{token}}$ from previous hit [μs]")
            set_integer_ticks(plt.gca().xaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            pdf.savefig(); plt.clf()

        plt.close()

        # Save some data
        threshold_data = np.full((512, 512), np.nan)
        threshold_data[col_start:col_stop,row_start:row_stop] = threshold_DAC
        noise_data = np.full((512, 512), np.nan)
        noise_data[col_start:col_stop,row_start:row_stop] = noise_DAC
        np.savez_compressed(
            os.path.splitext(output_file)[0] + ".npz",
            thresholds=threshold_data,
            noise=noise_data)


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

    for fp in tqdm(files, unit="file"):
        try:
            main(fp, args.overwrite)
        except Exception:
            print(traceback.format_exc())
