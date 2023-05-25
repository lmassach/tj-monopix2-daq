#!/usr/bin/env python3
"""Plots for hot_pixel_study."""
import argparse
import glob
from itertools import chain
import os
import traceback
import matplotlib.cm
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
        idel = cfg["configuration_out.chip.registers.IDEL"]
        print("IDEL =", idel)

        if f.root.Dut.shape[0] == 0:
            plt.annotate("No hits recorded!", (0.5, 0.5), ha='center', va='center')
            plt.gca().set_axis_off()
            pdf.savefig(); plt.clf()
            return

        hits = f.root.Dut[:] # hits now is an array/list? with all hits (many with same TS, same evt)

        # Event filters (event = multiple hits w same timestamp)
        #  # counts unique TS and get their timestamps and how many hits in each timestamp_hits (these 2 array have dimensions of N of evts) timestamp_idxs is an array  with the dimensions of hits (all) with the index that can be used to get the hits ordered by the same TS.
        timestamps, timestamp_idxs, timestamp_hits = np.unique(hits['timestamp'], return_inverse=True, return_counts=True)
        # Debug plot of the number of hits per "event"
        plt.hist(timestamp_hits, bins=100)
        plt.title("Debug: number of hits per 'event'")
        plt.yscale('log')
        plt.grid()
        pdf.savefig(); plt.clf()

        mask = (timestamp_hits > 10) & (timestamp_hits < 300)  # Filter out noise events default set 10. mask is an array with true and false with dimensions of evts, not hits = ??
        hits = hits[mask[timestamp_idxs]]  # hits now are only the ones in events with more than 10 hits/evts mask
        with np.errstate(all='ignore'):
            print("calculate tot")
            tot = (hits['te'] - hits['le']) & 0x7f

        timestamps, timestamp_idxs, timestamp_hits = np.unique(hits['timestamp'], return_inverse=True, return_counts=True) # this is needed to consider only evts with evts with more than 10 hits....but I'm not sure now if the tot array defined above has the correct dimensions and if can be used to access the correct TOT of the hits in the new array filtered
        # Find the LE of one pixel in row 0 for each timestamp
        timestamp_row0_le = np.full_like(timestamps, np.nan, np.float32)
        timestamp_row0_col = np.full_like(timestamps, -1, np.int16)
        # define which row is used as reference for LE row0 is default but whe spot is not close to row0 need to change it to different value
        #mask_row0 = hits['row'] == 0 # default
        mask_row0 = hits['row'] ==16
        for i, ts in tqdm(enumerate(timestamps), total=len(timestamps), unit='event', delay=2):
            mask_ts = hits['timestamp'] == ts
            mask_cut = (tot > 20) & (tot < 25)  # ADDITIONAL CUT ON ROW 0 HIT: now select only  row0 TOT=45, evts where row0_TOT is different are still there but since the row0 used for reference LE is not selected they don't enter in the delay map, but they enter still in all other MAPS and histo
            #mask_cut = tot == 30  # ADDITIONAL CUT ON ROW 0 HIT
            mask = mask_ts & mask_row0 & mask_cut
#            print("event i with TS =", i, ts, len(hits[mask]))
            if np.count_nonzero(mask):
                idx = np.argmax(mask)
                timestamp_row0_le[i] = hits['le'][idx]
                timestamp_row0_col[i] = hits['col'][idx]
#                print("event i with TS, row0 chosen with row0 col le te tot=", i, ts, hits['row'][idx], timestamp_row0_col[i], timestamp_row0_le[i], hits['te'][idx], tot[idx])

       # Compute the difference LE_row0 - LE_rowi for each hit
        with np.errstate(all='ignore'):
            print("calculate delta LE if row0 hit found")
            tmp = (timestamp_row0_le[timestamp_idxs] - hits['le']) % 128
            le_diff_to_row0 = np.where(tmp > 63, tmp - 128, tmp)
            del tmp

        # Print the first 1000 hits to a txt file with the same name as the output pdf
        with open(os.path.splitext(output_file)[0] + ".txt", "w") as ofs:
            print("IDEL =", idel,file=ofs)
            print("COL ROW  LE  TE   TOT   DeltaTS TS")
            print("COL ROW  LE  TE   TOT   DeltaTS TS", file=ofs)
            prev_ts = 0
            nhits = 0
            n = min(len(hits), 2)
            for h, t in zip(hits[:n], tot[:n]):
                print(f"{h['col']:3d} {h['row']:3d} {h['le']:3d}  {h['te']:3d} {t:3d} {h['timestamp']-prev_ts:7d} {h['timestamp']}", file=ofs)
#                print(f"{h['col']:3d} {h['row']:3d} {h['le']:3d}  {h['te']:3d} {t:3d} {h['timestamp']-prev_ts:7d} {h['timestamp']}")
                prev_ts = h['timestamp']
                nhits += 1
            print("N hits = ", nhits, file=ofs)
            print("N hits = ", nhits)

        # Plots
        count = 0
        for i, ts in chain([(-1, None)], enumerate(timestamps)):
            if i != -1 and np.isnan(timestamp_row0_le[i]):  # Skip events w/o good hits in the row 0
                continue
            if count > 10:  # Only plot the first 10 events
                break
            count += 1

            if i == -1:
                mask_ts = np.full_like(hits, True, bool) # this to select all hits from all evt togheter
                print("events used first iteration", len(timestamps))
                print("hits used first iteration no cuts", len(hits))
                #subtitle = f"All {len(timestamps)} events ({len(hits):.4g} hits), IDEL = {idel}"
            else:
                mask_ts = hits['timestamp'] == ts  # this to select only hits belonging to a given ts/evts
#                print("event i in  single iteration with TS=", i ,ts)
#                print("hits used in this event no cut", timestamp_hits[i])

                #subtitle = f"Event {i} (TS = {ts}, {timestamp_hits[i]:.4g} hits), IDEL = {idel}"

            # add additional mask to remove hits with mask_cut and from evt where row0 ref was not found
            mask_cut = (tot > 20) & (tot < 25) # ADDITIONAL CUT ON ROW i HITS same as for row0 to avoid effect of time walk
            #mask_cut = tot == 20 # ADDITIONAL CUT ON ROW i HITS same as for row0 to avoid effect of time walk
            mask_ts = np.isfinite(le_diff_to_row0) & mask_cut & mask_ts  # in this way select only hits that have mask_cut and also have row0 found

            ts_hits = hits[mask_ts]
            ts_tot = tot[mask_ts]

            if i == -1:
                #mask_ts = np.full_like(hits, True, bool) # this to select all hits from all evt togheter
                print("events used second call", np.count_nonzero(np.isfinite(timestamp_row0_le)))
                print("hits used in all events (with cuts applied)", len(ts_hits))
                subtitle = f"All {np.count_nonzero(np.isfinite(timestamp_row0_le))} events ({len(ts_hits):.4g} hits), IDEL = {idel}"
            else:
                #mask_ts = hits['timestamp'] == ts  # this to select only hits belonging to a given ts/evts
#                print("second iteration event i in single iteration with TS =", i, ts)
#                print("second iteration hits used in this event (with cuts applied)", len(ts_hits))
                subtitle = f"Event {i} (TS = {ts}, {len(ts_hits):.4g} hits), IDEL = {idel}"


            # 2D histogram of LE vs row (use hits from all evts that passed initial cut timestamp_hits > 10, also hits where there is no row0 le available, that do not enter in the delay map)
            plt.hist2d(ts_hits['row'], ts_hits['le'], bins=[512, 128], range=[[0, 512], [0, 128]], rasterized=True)
            plt.xlabel("Row")
            plt.ylabel("LE [25 ns]")
            plt.suptitle("LE vs row")
            plt.title(subtitle)
            plt.colorbar().set_label("Number of hits")
            pdf.savefig(); plt.clf()

            # 2D histogram of ToT vs row (use hits from all evts that passed initial cut timestamp_hits > 10, also hits where there is no row0 le available, that do not enter in the delay map)
            plt.hist2d(ts_hits['row'], ts_tot, bins=[512, 128], range=[[0, 512], [0, 128]], rasterized=True)
            plt.xlabel("Row")
            plt.ylabel("ToT [25 ns]")
            plt.suptitle("ToT vs row")
            plt.title(subtitle)
            plt.colorbar().set_label("Number of hits")
            pdf.savefig(); plt.clf()

            # 2D histogram of LE vs ToT (use hits from all evts that passed initial cut timestamp_hits > 10, also hits where there is no row0 le available, that do not enter in the delay map)
            plt.hist2d(ts_tot, ts_hits['le'], bins=[128, 128], range=[[0, 128], [0, 128]], rasterized=True)
            plt.xlabel("ToT [25 ns]")
            plt.ylabel("LE [25 ns]")
            plt.suptitle("LE vs ToT")
            plt.title(subtitle)
            plt.colorbar().set_label("Number of hits")
            pdf.savefig(); plt.clf()

            # TOT map (use hits from all evts that passed initial cut timestamp_hits > 10, also hits where there is no row0 le available, that do not enter in the delay map, unless the following 2 masks are set to remove hits with TOT < cut and from evt where row0 ref not found is )
            #mask_cut = tot > 10  # ADDITIONAL CUT ON ROW i HITS same as for row0 to avoid effect of time walk
            #mask_ts = np.isfinite(le_diff_to_row0) & mask_cut & mask_ts  # in this way selects only hits that have mask cut and also in that evt the row0 was found
            tot2d, tot2d_edges, _  = np.histogram2d(
                hits[mask_ts]['col'], hits[mask_ts]['row'], bins=[512, 512], range=[[0, 512], [0, 512]],
                weights=tot[mask_ts])
            counts2d, counts2d_edges, _ = np.histogram2d(
                hits[mask_ts]['col'], hits[mask_ts]['row'], bins=[512, 512], range=[[0, 512], [0, 512]])
            with np.errstate(all='ignore'):
                tot2davg = tot2d / counts2d

            m1 = np.nanquantile(tot2davg.reshape(-1), 0.16)
            m2 = np.nanquantile(tot2davg.reshape(-1), 0.84)
            ml = m2 - m1
            m1 -= 0.1 * ml
            m2 += 0.1 * ml
#            plt.pcolormesh(np.arange(299,303), np.arange(0, 512), tot2davg[299:302,0:511].transpose(),
#                           vmin=9.5, vmax=128.5, rasterized=True)
            plt.pcolormesh(tot2d_edges, tot2d_edges, tot2davg.transpose(),
                           vmin=m1, vmax=m2, rasterized=True)
            plt.xlabel("Col")
            plt.ylabel("Row")
            plt.grid()
            plt.suptitle("TOT map, hits with cut, " + subtitle)
#            plt.title(subtitle)
            plt.colorbar().set_label("TOT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            frontend_names_on_top()
            pdf.savefig(); plt.clf()

            # Hit map plot
            cmap = matplotlib.cm.get_cmap("viridis").copy()
            cmap.set_over("r")
            m = np.quantile(counts2d[counts2d > 0], 0.99) * 1.2 if np.any(counts2d > 0) else 1
            plt.pcolormesh(counts2d_edges, counts2d_edges, counts2d.transpose(),
                           vmin=0, vmax=m, cmap=cmap, rasterized=True)  # Necessary for quick save and view in PDF
            plt.suptitle("Hit map, " + subtitle)
#            plt.title(subtitle)
            plt.xlabel("Col")
            plt.ylabel("Row")
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / pixel")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            frontend_names_on_top()
            pdf.savefig(); plt.clf()



            #TOT 2d around  col 300
            plt.pcolormesh(np.arange(299,303), np.arange(0, 512), tot2davg[299:302,0:511].transpose(),
                           vmin=m1, vmax=m2, rasterized=True)
#            plt.pcolormesh(tot2d_edges, tot2d_edges, tot2davg.transpose(),
#                           vmin=m1, vmax=m2, rasterized=True)
            plt.xlabel("Col")
            plt.ylabel("Row")
            plt.suptitle("TOT map, " + subtitle)
#            plt.title(subtitle)
            plt.colorbar().set_label("TOT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            frontend_names_on_top()
            pdf.savefig(); plt.clf()




            # Delay map
            # Delay = BCID delay - compensation with IDEL
            # Computed as mean of (LE_row0-LE)*25ns for each pixel
            if i == -1:
                # Colormap with red for points above maximum
                cmap_red_over = plt.cm.viridis.copy()
                cmap_red_over.set_over('r')


#                mask_cut = tot > 10  # ADDITIONAL CUT ON ROW i HITS
                #mask_cut = tot > 10  # ADDITIONAL CUT ON ROW i HITS same as for row0 to avoid effect of time walk
                #mask = np.isfinite(le_diff_to_row0) & mask_cut  # in this delay map enter only hits that have TOT = 45 and also in that evt the row0 had TOT = 55
                hn, edges512, _ = np.histogram2d(
                    hits[mask_ts]['col'], hits[mask_ts]['row'], bins=[512, 512], range=[[0, 512], [0, 512]],
                    weights=le_diff_to_row0[mask_ts])
                hd, _, _ = np.histogram2d(
                    hits[mask_ts]['col'], hits[mask_ts]['row'], bins=[512, 512], range=[[0, 512], [0, 512]])
                with np.errstate(all='ignore'):
                    delay_map = 25 * hn / hd
                m1 = np.nanquantile(delay_map.reshape(-1), 0.16)
                m2 = np.nanquantile(delay_map.reshape(-1), 0.84)
                ml = m2 - m1
                m1 -= 0.4 * ml
                m2 += 0.4 * ml
                plt.pcolormesh(edges512, edges512, delay_map.transpose(),
                               vmin=m1, vmax=m2, rasterized=True, cmap=cmap_red_over)
                plt.xlabel("Col")
                plt.ylabel("Row")
                plt.grid()
                plt.suptitle("Delay(=(LE_0-LE)*25ns) map, " + subtitle)
#                plt.title(subtitle)
                plt.colorbar().set_label("Delay [ns]")
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                frontend_names_on_top()
                pdf.savefig(); plt.clf()

                # Projection on y axis
                delay_vs_row = np.nanmean(delay_map, axis=0)
                plt.step(np.arange(512), delay_vs_row, where='mid')
                plt.xlabel("Row")
                plt.ylabel("Mean delay along columns [ns]")
                plt.title("Delay projection " + subtitle)
                plt.ylim(m1, m2)
                plt.grid()
                pdf.savefig(); plt.clf()

                # Projection with errorbars
                delay_std_vs_row = np.nanstd(delay_map, axis=0)
                plt.errorbar(np.arange(512), delay_vs_row, delay_std_vs_row, fmt='|')
                plt.xlabel("Row")
                plt.ylabel("Mean delay along columns [ns]")
                plt.title("Delay projection " + subtitle)
                plt.ylim(m1, m2)
                plt.grid()
                pdf.savefig(); plt.clf()


                # Delay map only for 3 col around col 300
                plt.pcolormesh(np.arange(299,303), np.arange(0, 512), delay_map[299:302,0:511].transpose(),
                               vmin=m1, vmax=m2, rasterized=True)
                plt.xlabel("Col")
                plt.ylabel("Row")
                # plt.suptitle("Delay map - " + subtitle)
                plt.title("Delay map for col 300, " + subtitle)
#                plt.title(subtitle)
                plt.colorbar().set_label("Delay [ns]")
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                frontend_names_on_top()
                pdf.savefig(); plt.clf()

                # TOT avg map
                #tot2d_tmp, tot2d_edges, _  = np.histogram2d(
                #    hits[mask]['col'], hits[mask]['row'], bins=[512, 512], range=[[0, 512], [0, 512]],
                #    weights=tot[mask])
                #tot2d += tot2d_tmp
                #del tot2d_tmp
                #m1 = np.nanquantile(tot2d.reshape(-1), 0.16)
                #m2 = np.nanquantile(tot2d.reshape(-1), 0.84)
                #ml = m2 - m1
                #m1 -= 0.1 * ml
                #m2 += 0.1 * ml
                #plt.pcolormesh(tot2d_edges, tot2d_edges512, tot2d.transpose(),
                #               vmin=m1, vmax=m2, rasterized=True)
                #plt.xlabel("Col")
                #plt.ylabel("Row")
                #plt.suptitle("TOT avg map, " + subtitle)
                #plt.colorbar().set_label("TOT [25 ns]")
                #set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                #frontend_names_on_top()
                #pdf.savefig(); plt.clf()



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
