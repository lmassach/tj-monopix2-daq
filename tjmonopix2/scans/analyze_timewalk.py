from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import tables as tb


def main(basepath_infiles, lim=0.0):
    out_path = os.path.dirname(basepath_infiles[0])
    tdcs = [analyze_tdc(f) for f in basepath_infiles]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
    for tdc, q, col in tdcs:
        ax.plot(np.linspace(0, len(tdc)/0.640/25, len(tdc)), tdc, '-+', label=f'Pixel [0, {col}]')
    
    ax.set_ylabel('Timewalk / ns')
    ax.set_xlabel('ToT / 25ns')
    if lim:
        ax.set_ylim(0.0, lim)
    ax.grid()
    ax.legend()
    ax.set_title(f'Timewalk vs ToT')

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'hitor_timewalk.png'))
    plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
    for tdc, q, col in tdcs:
        ax.plot(q, '-+', label=f'Pixel [0, {col}]')
    
    ax.set_xlabel('Injection Charge, deltaV / DAC')
    ax.set_ylabel('Timewalk / ns')
    if lim:
        ax.set_ylim(0.0, lim)
    ax.grid()
    ax.legend()
    ax.set_title(f'Timewalk vs Injected Charge')

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f'hitor_timewalk_q.png'))
    plt.close()


@nb.njit(parallel=False)
def optimize_hitlist(col, le, tdc, scan_param_id, vcal_low, vcal_high, tdc_sum_tdc, inj_tdc, tdc_sum_q, inj_q):
    for i in nb.prange(len(col)):
        if col[i] == 1022:  # TDC hit
            spi = scan_param_id[i]
            tdci = int(tdc[i])
            les = int(le[i])
            if les < 0:
                les += 256
            
            q = int(vcal_high[spi] - vcal_low[spi])
            
            tdc_sum_tdc[tdci] += les
            inj_tdc[tdci] += 1
            
            tdc_sum_q[q] += les
            inj_q[q] += 1
        

def analyze_tdc(p):
    start_col = 0
    with tb.open_file(p, mode="r", title='configuration_in') as h5file:
        hitlist = h5file.root.Dut
        scan_params = h5file.root.configuration_out.scan.scan_params
        
        tdc_sum_tdc = np.zeros([2048])
        inj_tdc = np.zeros([2048])
        
        tdc_sum_q = np.zeros([256])
        inj_q = np.zeros([256])
        
        optimize_hitlist(hitlist.col('col'), hitlist.col('le'), hitlist.col('token_id'), hitlist.col('scan_param_id'), scan_params.col('vcal_low'), scan_params.col('vcal_high'), tdc_sum_tdc, inj_tdc, tdc_sum_q, inj_q)
        
        unique, counts = np.unique(inj_tdc, return_counts=True)
        print(dict(zip(unique, counts)))
        
        for k, v in h5file.root.configuration_in.scan.scan_config[:]:
            if k == b'start_column':
                start_col = int(str(v, encoding='utf8'))
    
    tw_q = np.divide(tdc_sum_q, inj_q)
    tw_q -= np.nanmin(tw_q)
    tw_q /= 0.640
    
    tw_tdc = np.divide(tdc_sum_tdc, inj_tdc)
    tw_tdc -= np.nanmin(tw_tdc)
    tw_tdc /= 0.640
    
    return tw_tdc, tw_q, start_col
    

def plot_single(hitor_del):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    
    # les[les > 35] = float('nan')
    image = plt.imshow(hitor_del.transpose()[:,:32], aspect='auto', interpolation='none')
    cbar = plt.colorbar(image)
    cbar.set_label('average Delay')
    #plt.clim(14,16)
    plt.gca().invert_yaxis()

    #IDEL = int(registers[6][1])
    
    
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.title(f'Delay per pixel')

    plt.tight_layout()
    plt.savefig(f'hitor_delay.png')
    plt.close()
    # plt.show()
    
    #return avg_delay, stddev_del, IDEL


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_files", nargs="+", metavar="input_file",
                        help="The input _timewalk_scan_intepreted.h5 file(s)")
    parser.add_argument("--lim", type=float, default=0.0, help="Optional limit for y axis in ns.")
    args = parser.parse_args()
    main(args.input_files, args.lim)
