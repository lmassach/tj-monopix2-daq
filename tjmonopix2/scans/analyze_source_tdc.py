import tables as tb
import argparse
import numba as nb
import os.path as path
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tjmonopix2.analysis import analysis



tot_cut_pos = 0
tot_cut_neg = -2

def analyze(file):
    file_interpreted = file.rsplit(".h5")[0] + "_interpreted_cluster.h5"

    if os.path.exists(file_interpreted):
        os.remove(file_interpreted)

    print('Analyzing file: ' + path.basename(file))
    with analysis.Analysis(raw_data_file=file, cluster_hits=True, analyzed_data_file=file_interpreted) as a:
        a.analyze_data()

    return file_interpreted


def table_to_dict(table_item, key_name='attribute', value_name='value'):
    ret = {}
    for row in table_item.iterrows():
        ret[row[key_name].decode('UTF-8')] = row[value_name].decode('UTF-8')
    return ret




def main(infile):
    out_prefix = os.path.splitext(infile)[0]
    chip_sn, idel, tdc_tdel_2dhist, tdel_row_2dhist, tdel_row_2dhist_tdc, ts_le_2dhist, ass_hitmap, tdc_tot_2dhist, lediff_row_2dhist = analyze_tdc(infile)

    with PdfPages(out_prefix + "_ana.pdf") as pdf:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)

        #tdc_tdel_2dhist[tdc_tdel_2dhist > 800] = np.nan
        #image = plt.imshow(hitor_del.transpose()[:,:32], aspect='auto', interpolation='none')
        image = plt.imshow(tdc_tdel_2dhist, extent=[0, 4096/0.640/25, 256/0.640, 0], aspect='auto', interpolation='none')
        cbar = plt.colorbar(image)
        cbar.set_label('Hits')
        #plt.clim(14,16)
        plt.gca().invert_yaxis()

        ax.set_xlim([0, 80])
        ax.set_ylim([30, 200])

        ax.set_ylabel('Trigger distance / ns')
        ax.set_xlabel('ToT / 25ns')
        ax.grid()
        ax.set_title(f'{chip_sn}: ToT vs Trigger distance')

        plt.tight_layout()
        # plt.savefig(f'{out_prefix}_{chip_sn}_source_tdc_tdel_2dhist.png')
        pdf.savefig()
        plt.close()

        # TDEL vs row
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)

        #image = plt.imshow(hitor_del.transpose()[:,:32], aspect='auto', interpolation='none')  #, extent=[0, 512, 256/0.640, 0]
        image = plt.imshow(tdel_row_2dhist, extent=[0, 511, 256/0.640, 0], aspect='auto', interpolation='none')
        cbar = plt.colorbar(image)
        cbar.set_label('Hits')
        #plt.clim(14,16)
        plt.gca().invert_yaxis()

        #ax.set_xlim([479, 496])
        ax.set_ylim([0, 150])

        ax.set_ylabel('Trigger distance / ns')
        ax.set_xlabel('Row')
        ax.grid()
        ax.set_title(f'{chip_sn}: Row vs Trigger distance')

        plt.tight_layout()
        # plt.savefig(f'{out_prefix}_{chip_sn}_source_tdel_row_2dhist_clust.png')
        pdf.savefig()
        plt.close()


        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)

        #image = plt.imshow(hitor_del.transpose()[:,:32], aspect='auto', interpolation='none')  #, extent=[0, 512, 256/0.640, 0]
        image = plt.imshow(tdel_row_2dhist_tdc, extent=[0, 511, 256/0.640, 0], aspect='auto', interpolation='none')
        cbar = plt.colorbar(image)
        cbar.set_label('Average ToT/25ns')
        #plt.clim(14,16)
        plt.gca().invert_yaxis()

        #ax.set_xlim([479, 496])
        ax.set_ylim([0, 150])

        ax.set_ylabel('Trigger distance / ns')
        ax.set_xlabel('Row')
        ax.grid()
        ax.set_title(f'{chip_sn}: Row vs Trigger distance')

        plt.tight_layout()
        # plt.savefig(f'{out_prefix}_{chip_sn}_source_tdel_row_2dhist_tdc_clust.png')
        pdf.savefig()
        plt.close()


        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)

        #image = plt.imshow(hitor_del.transpose()[:,:32], aspect='auto', interpolation='none')  #, extent=[0, 512, 256/0.640, 0]
        image = plt.imshow(ts_le_2dhist, extent=[0, 128, 64, 0], aspect='auto', interpolation='none')
        cbar = plt.colorbar(image)
        cbar.set_label('Hits')
        #plt.clim(14,16)
        plt.gca().invert_yaxis()

        #ax.set_xlim([479, 496])
        #ax.set_ylim([0, 100])

        ax.set_ylabel('Corrected Timestamp / 25ns')
        ax.set_xlabel('Le BCID')
        ax.grid()
        #ax.set_title(f'Row vs Trigger distance')

        #plt.tight_layout()
        # plt.savefig(f'{out_prefix}_{chip_sn}_source_ts_le_2dhist_clust.png')
        pdf.savefig()
        plt.close()


        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)

        # Subtract time offset
        time_offset = np.nanargmax(lediff_row_2dhist[:, 0])
        tmp = np.copy(lediff_row_2dhist)
        sz0 = lediff_row_2dhist.shape[0]
        for i in range(sz0):
            lediff_row_2dhist[(i+sz0//2)%sz0,:] = tmp[(i+time_offset)%sz0,:]
        time_offset = sz0//2
        # Normalize each row to 1
        lediff_row_2dhist = np.divide(lediff_row_2dhist, np.nansum(lediff_row_2dhist, axis=0))
        # Actual plotting
        image = plt.imshow(lediff_row_2dhist, aspect='auto', interpolation='none', extent=[0, 128, 64, 0])
        cbar = plt.colorbar(image)
        cbar.set_label('Hits (relative)')
        #plt.clim(14,16)
        plt.gca().invert_yaxis()

        #ax.set_xlim([479, 496])
        ax.set_ylim([time_offset/16-5, time_offset/16+5])

        ax.set_ylabel('Le Difference to TDC Timestamp / 25ns')
        ax.set_xlabel('Row')
        ax.grid()
        ax.set_title(f'Row vs Le distance (IDEL: {idel})')

        #plt.tight_layout()
        # plt.savefig(f'{out_prefix}_{chip_sn}_source_lediff_row_2dhist.png')
        pdf.savefig()
        plt.close()


        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)

        weights = np.nan_to_num(lediff_row_2dhist[time_offset-5*16:time_offset+5*16, :])
        sum_w = np.sum(weights, axis=0)
        sum_wv = np.sum((np.arange(time_offset-5*16, time_offset+5*16)).reshape((-1,1)) * weights, axis=0)
        lediff_row_mean = sum_wv / sum_w
        plt.plot(lediff_row_mean / 16)

        #ax.set_xlim([479, 496])
        ax.set_ylim([time_offset/16-2, time_offset/16+2])

        ax.set_ylabel('Le Difference to TDC Timestamp / 25ns')
        ax.set_xlabel('Row')
        ax.grid()
        ax.set_title(f'Row vs Le distance (IDEL: {idel})')

        #plt.tight_layout()
        # plt.savefig(f'{out_prefix}_{chip_sn}_source_lediff_row_avg.png')
        pdf.savefig()
        plt.close()



        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)

        # Rebin to groups of 16 rows
        lediff_row_2dhist_rebin = np.zeros((lediff_row_2dhist.shape[0], lediff_row_2dhist.shape[1] // 16), lediff_row_2dhist.dtype)
        for i in range(16):
            lediff_row_2dhist_rebin[:,:] += lediff_row_2dhist[:,i::16]
        # Actual plotting
        image = plt.imshow(lediff_row_2dhist_rebin, aspect='auto', interpolation='none', extent=[0, 128, 64, 0])
        cbar = plt.colorbar(image)
        cbar.set_label('Hits (relative)')
        #plt.clim(14,16)
        plt.gca().invert_yaxis()

        #ax.set_xlim([479, 496])
        ax.set_ylim([time_offset/16-5, time_offset/16+5])

        ax.set_ylabel('Le Difference to TDC Timestamp / 25ns')
        ax.set_xlabel('Row')
        ax.grid()
        ax.set_title(f'Row vs Le distance (IDEL: {idel})')

        #plt.tight_layout()
        # plt.savefig(f'{out_prefix}_{chip_sn}_source_lediff_row_2dhist_rebin.png')
        pdf.savefig()
        plt.close()


        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)

        weights = np.nan_to_num(lediff_row_2dhist_rebin[time_offset-5*16:time_offset+5*16, :])
        sum_w = np.sum(weights, axis=0)
        sum_wv = np.sum((np.arange(time_offset-5*16, time_offset+5*16)).reshape((-1,1)) * weights, axis=0)
        lediff_row_mean = sum_wv / sum_w
        plt.plot(np.arange(0, 512, 16) + 8, lediff_row_mean / 16)

        #ax.set_xlim([479, 496])
        ax.set_ylim([time_offset/16-2, time_offset/16+2])

        ax.set_ylabel('Le Difference to TDC Timestamp / 25ns')
        ax.set_xlabel('Row')
        ax.grid()
        ax.set_title(f'Row vs Le distance (IDEL: {idel})')

        #plt.tight_layout()
        # plt.savefig(f'{out_prefix}_{chip_sn}_source_lediff_row_avg_rebin.png')
        pdf.savefig()
        plt.close()



        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)

        image = plt.imshow(ass_hitmap.transpose(), aspect='auto', interpolation='none')
        cbar = plt.colorbar(image)
        cbar.set_label('Hits')
        #plt.clim(14,16)
        plt.gca().invert_yaxis()

        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.grid()
        #ax.legend()
        ax.set_title(f'{chip_sn}: TDC Associated seed pixel map')

        plt.tight_layout()
        # plt.savefig(f'{out_prefix}_{chip_sn}_source_TDC_ass_map.png')
        pdf.savefig()
        plt.close()


        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)

        #image = plt.imshow(hitor_del.transpose()[:,:32], aspect='auto', interpolation='none')
        image = plt.imshow(tdc_tot_2dhist, extent=[0, 127, 4096/16, 0], aspect='auto', interpolation='none')
        cbar = plt.colorbar(image)
        cbar.set_label('Hits')
        #plt.clim(14,16)
        plt.gca().invert_yaxis()

        #ax.set_xlim([0, 80])
        ax.set_ylim([0, 128])

        ax.set_ylabel('TDC Result / 25ns')
        ax.set_xlabel('ToT / 25ns')
        ax.grid()
        ax.set_title(f'{chip_sn}: ToT vs TDC value')

        plt.tight_layout()
        # plt.savefig(f'{out_prefix}_{chip_sn}_source_tdc_tot.png')
        pdf.savefig()
        plt.close()


@nb.njit(parallel=False)
def ana_clist(seed_col,
              seed_row,
              seed_le,     # TDC: Trigdist
              seed_te,
              token_id,    # TDC: Value
              event_no,    # TDC: Timestamp
              tdc_tdel_2dhist, tdel_col_2dhist, tdel_col_2dhist_tdc, ts_le_2dhist, ass_hitmap, tdc_tot_2dhist, lediff_row_2dhist):
    for i in nb.prange(len(seed_col)):
        if seed_col[i] == 1022:  # TDC hit
            rowi = int(seed_row[i])
            if rowi != 254:
                for dist in range(1, 10):
                    if i+dist < len(seed_col) and seed_col[i+dist] != 1022:
                        tot = (int(seed_te[i+dist]) - int(seed_le[i+dist])) % 128
                        diff = int(token_id[i])//16 - tot
                        if diff <= tot_cut_pos and diff >= tot_cut_neg:
                            tdc_tdel_2dhist[int(seed_le[i]), int(token_id[i])] += 1
                            tdel_col_2dhist[rowi, int(seed_row[i+dist])] += 1
                            tdel_col_2dhist_tdc[rowi, int(seed_row[i+dist])] += int(token_id[i])
                            ts_le_2dhist[((int(event_no[i])*16)-(rowi))%(64*16), int(seed_le[i+dist])] += 1
                            ass_hitmap[int(seed_col[i+dist]), int(seed_row[i+dist])] += 1
                            tdc_tot_2dhist[int(token_id[i]), tot] += 1
                            if tot > 80:
                                lediff_row_2dhist[((int(event_no[i])*16)-(rowi) - int(seed_le[i+dist]*16))%(64*16), int(seed_row[i+dist])] += 1
                    else:
                        break


def analyze_tdc(p):
    h5file = tb.open_file(p, mode="r", title='configuration_in')

    clist = h5file.root.Cluster
    settings = table_to_dict(h5file.root.configuration_in.chip.settings)
    regs =     table_to_dict(h5file.root.configuration_in.chip.registers, key_name='register')
    #scan_params = h5file.root.configuration_out.scan.scan_params

    tdc_tdel_2dhist = np.zeros([256, 4096])
    tdel_col_2dhist = np.zeros([256, 512])
    tdel_col_2dhist_tdc = np.zeros([256, 512])
    ts_le_2dhist = np.zeros([64*16, 128])
    lediff_row_2dhist = np.zeros([64*16, 512])
    ass_hitmap = np.zeros([512, 512])
    tdc_tot_2dhist = np.zeros([4096, 128])

    ana_clist(clist.col('seed_col'),
              clist.col('seed_row'),
              clist.col('seed_le'),
              clist.col('seed_te'),
              clist.col('seed_token_id'),  # TDC Value
              clist.col('event_number'),   # TDC Timestamp
              tdc_tdel_2dhist, tdel_col_2dhist, tdel_col_2dhist_tdc, ts_le_2dhist, ass_hitmap, tdc_tot_2dhist, lediff_row_2dhist)

    # ana_hlist(hitlist.col('col'),          # 1022
    #           hitlist.col('row'),          # Trigger Dist
    #           hitlist.col('le'),           # Trigger dist
    #           hitlist.col('token_id'),     # TDC
    #           tdc_tdel_2dhist)

    #unique, counts = np.unique(inj_tdc, return_counts=True)
    #print(dict(zip(unique, counts)))

    h5file.close()


    tdc_tdel_2dhist[tdc_tdel_2dhist == 0] = np.nan
    tdel_col_2dhist[tdel_col_2dhist == 0] = np.nan
    ts_le_2dhist[ts_le_2dhist == 0] = np.nan
    ass_hitmap[ass_hitmap == 0] = np.nan
    tdel_col_2dhist_tdc = np.divide(tdel_col_2dhist_tdc, tdel_col_2dhist)/16
    tdc_tot_2dhist[tdc_tot_2dhist == 0] = np.nan
    lediff_row_2dhist[lediff_row_2dhist == 0] = np.nan

    return settings['chip_sn'], int(regs['IDEL']), tdc_tdel_2dhist, tdel_col_2dhist, tdel_col_2dhist_tdc, ts_le_2dhist, ass_hitmap, tdc_tot_2dhist, lediff_row_2dhist



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", metavar="input_file", nargs="+",
                        help="The input _tdc_hitor.h5 file(s)")
    args = parser.parse_args()

    for p in args.input_files:
        if 'interpreted' not in p:
            p = analyze(p)
        main(p)
