import argparse
import os
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import tables as tb


def main(basepath_infiles):
    out_dir = os.path.dirname(basepath_infiles[0])
    hitor_del, hitor_del_per_row = analyze_tdc(basepath_infiles)
    plot_single(hitor_del, out_dir)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), dpi=100)

    ax[0].plot(hitor_del_per_row, '-', label='all')
    print(f'Max delay: {np.max(hitor_del_per_row)}')

    ax[0].set_xlabel('Row')
    ax[0].set_ylabel('TDC Trigger Distance / ns')
    ax[0].grid()
    ax[0].set_title(f'Average TDC Trigger distance vs Row')

    ax[1].plot(np.nanmean(hitor_del[:, 0:4], axis=1), '-+', label='rows 0:4')
    ax[1].plot(np.nanmean(hitor_del[:, 0:16], axis=1), '-+', label='rows 0:16')

    ax[1].set_xlabel('Column')
    ax[1].set_ylabel('TDC Trigger Distance / ns')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_title(f'Average TDC Trigger distance vs Column')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'hitor_delay_gr.png'))


@nb.njit(parallel=False)
def optimize_hitlist(col, le, scan_param_id, px_row, px_col, trig_del_sum, injections):
    for i in nb.prange(len(col)):
        if col[i] == 1022:  # TDC hit
            spi = scan_param_id[i]
            coli = int(px_col[spi])
            rowi = int(px_row[spi])

            les = le[i]
            if les < 0:
                les += 256

            trig_del_sum[coli, rowi] += les
            injections[coli, rowi] += 1


def analyze_tdc(paths):
    trig_del_sum = np.zeros([512, 512])
    injections = np.zeros([512, 512])

    for p in paths:
        with tb.open_file(p, mode="r", title='configuration_in') as h5file:
            hitlist = h5file.root.Dut
            scan_params = h5file.root.configuration_out.scan.scan_params
            optimize_hitlist(hitlist.col('col'), hitlist.col('le'), hitlist.col('scan_param_id'), scan_params.col('row'), scan_params.col('col'), trig_del_sum, injections)

    unique, counts = np.unique(injections, return_counts=True)
    print(dict(zip(unique, counts)))

    hitor_del = np.divide(trig_del_sum, injections)
    hitor_del_per_row = np.divide(np.sum(trig_del_sum, axis=0), np.sum(injections,axis=0))

    hitor_offset = np.nanmean(hitor_del[:, 0:4], )

    hitor_del = (hitor_del - hitor_offset) / 0.640   # 640 MHz to get to ns
    hitor_del_per_row = (hitor_del_per_row - hitor_offset) / 0.640

    return hitor_del, hitor_del_per_row


def plot_single(hitor_del, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    # les[les > 35] = float('nan')
    image = plt.imshow(hitor_del.transpose()[:,:], aspect='auto', interpolation='none')
    cbar = plt.colorbar(image)
    cbar.set_label('HitOr Delay map')
    #plt.clim(14,16)
    plt.gca().invert_yaxis()

    #IDEL = int(registers[6][1])

    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.title(f'HitOr Delay')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'hitor_delay.png'))
    plt.close()
    # plt.show()

    #return avg_delay, stddev_del, IDEL


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", metavar="input_file", nargs="+",
                        help="The input _tdc_hitor_scan_interpreted.h5 file(s)")
    args = parser.parse_args()
    main(args.input_files)
