import tables as tb
import argparse
import glob
import os.path as path
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tjmonopix2.analysis import analysis



tot_cut_pos = 2
tot_cut_neg = -4

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
    chip_sn, tdc_tdel_2dhist, tdel_row_2dhist, tdel_row_2dhist_tdc, ts_le_2dhist, ass_hitmap, tdc_tot_2dhist = analyze_tdc(infile)

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
    plt.savefig(f'{chip_sn}_source_tdc_tdel_2dhist.png')
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
    plt.savefig(f'{chip_sn}_source_tdel_row_2dhist_clust.png')
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
    plt.savefig(f'{chip_sn}_source_tdel_row_2dhist_tdc_clust.png')
    plt.close()


    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
    
    #image = plt.imshow(hitor_del.transpose()[:,:32], aspect='auto', interpolation='none')  #, extent=[0, 512, 256/0.640, 0]
    image = plt.imshow(ts_le_2dhist, aspect='auto', interpolation='none')
    cbar = plt.colorbar(image)
    cbar.set_label('Hits')
    #plt.clim(14,16)
    plt.gca().invert_yaxis()
    
    #ax.set_xlim([479, 496])
    #ax.set_ylim([0, 100])
    
    ax.set_ylabel('Corrected Timestamp / 25ns')
    ax.set_xlabel('Le BCID')
    #ax.grid()
    #ax.set_title(f'Row vs Trigger distance')

    #plt.tight_layout()
    plt.savefig(f'{chip_sn}_source_ts_le_2dhist_clust.png')
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
    plt.savefig(f'{chip_sn}_source_TDC_ass_map.png')
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
    plt.savefig(f'{chip_sn}_source_tdc_tot.png')
    plt.close()

    


import numba as nb

@nb.njit(parallel=False)
def ana_clist(seed_col, 
              seed_row, 
              seed_le,     # TDC: Trigdist
              seed_te,
              token_id,    # TDC: Value
              event_no,    # TDC: Timestamp
              tdc_tdel_2dhist, tdel_col_2dhist, tdel_col_2dhist_tdc, ts_le_2dhist, ass_hitmap, tdc_tot_2dhist):
    dist = 1
    for i in nb.prange(len(seed_col)):
        if seed_col[i] == 1022:  # TDC hit
            rowi = int(seed_row[i])
            
            if rowi != 254:
                if i+dist < len(seed_col) and seed_col[i+dist] != 1022:
                    tot = (int(seed_te[i+dist]) - int(seed_le[i+dist])) % 128
                    diff = int(token_id[i])//16 - tot
                    if diff <= tot_cut_pos and diff >= tot_cut_neg:
                        tdc_tdel_2dhist[int(seed_le[i]), int(token_id[i])] += 1
                        tdel_col_2dhist[rowi, int(seed_row[i+dist])] += 1
                        tdel_col_2dhist_tdc[rowi, int(seed_row[i+dist])] += int(token_id[i])
                        ts_le_2dhist[int(event_no[i])+16-rowi//16, int(seed_le[i+dist])] += 1
                        ass_hitmap[int(seed_col[i+dist]), int(seed_row[i+dist])] += 1
                        tdc_tot_2dhist[int(token_id[i]), tot] += 1
                        

def analyze_tdc(p):    
    h5file = tb.open_file(p, mode="r", title='configuration_in')

    clist = h5file.root.Cluster
    settings = table_to_dict(h5file.root.configuration_in.chip.settings)
    #scan_params = h5file.root.configuration_out.scan.scan_params
    
    tdc_tdel_2dhist = np.zeros([256, 4096])
    tdel_col_2dhist = np.zeros([256, 512])
    tdel_col_2dhist_tdc = np.zeros([256, 512])
    ts_le_2dhist = np.zeros([256+16, 128])
    ass_hitmap = np.zeros([512, 512])
    tdc_tot_2dhist = np.zeros([4096, 128])
    
    ana_clist(clist.col('seed_col'), 
              clist.col('seed_row'), 
              clist.col('seed_le'),
              clist.col('seed_te'),
              clist.col('seed_token_id'),  # TDC Value
              clist.col('event_number'),   # TDC Timestamp
              tdc_tdel_2dhist, tdel_col_2dhist, tdel_col_2dhist_tdc, ts_le_2dhist, ass_hitmap, tdc_tot_2dhist)
    
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
    
    return settings['chip_sn'], tdc_tdel_2dhist, tdel_col_2dhist, tdel_col_2dhist_tdc, ts_le_2dhist, ass_hitmap, tdc_tot_2dhist



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", metavar="input_file", nargs="+",
                        help="The input _tdc_hitor.h5 file(s)")
    args = parser.parse_args()
    
    for p in args.input_files:
        if not p.__contains__('interpreted_cluster'):
            p = analyze(p)
        main(p)
    
    
    
    
    
    
