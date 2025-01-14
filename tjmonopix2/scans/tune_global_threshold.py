#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

'''
    Finds the optimum global threshold value for target threshold using binary search.
'''

import time

import numpy as np
import tables as tb
from tqdm import tqdm

from tjmonopix2.system.scan_base import ScanBase
from tjmonopix2.scans.shift_and_inject import shift_and_inject
from tjmonopix2.analysis import online as oa


scan_configuration = {
    'start_column': 300, # 216
    'stop_column': 301, #230
    'start_row': 0, #120
    'stop_row': 512, #220

    'n_injections': 100,

    # Target threshold
    'VCAL_LOW': 30,
    'VCAL_HIGH': 30+18,

    'bcid_reset': True,  # BCID reset before injection
    # chipW8R13 File produced w BCID reset target=25 ITHR=64 ICASN=80 settings psub pwell=-6V cols=224-448 rows=0-512
    # 'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0_2023-03-25/chip_0/20230325_182214_local_threshold_tuning_interpreted.h5',
    # chipW8R13 File produced w BCID reset target=27 ITHR=64 IBIAS=100 ICASN=2 settings psub pwell=-6V cols=224-448 rows=0-512
    'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0-2023-05-24/chip_0/20230524_192643_local_threshold_tuning_interpreted.h5',

    # This setting does not have to be changed, it only allows (slightly) faster retuning
    # E.g.: gdac_value_bits = [3, 2, 1, 0] uses the 4th, 3rd, 2nd, and 1st GDAC value bit.
    # GDAC is not an existing DAC, its value is mapped to ICASN currently
    'gdac_value_bits': range(6, -1, -1)
}

register_overrides = {
    # 'ITHR': 64,  # Default 64
    # 'IBIAS': 50,  # Default 50
    # 'VRESET': 110,  # Default 143
    # 'ICASN': 0,  # Default 0
    # 'VCASP': 93,  # Default 93
    # "VCASC": 228,  # Default 228
    # "IDB": 100,  # Default 100
    # 'ITUNE': 150,  # Default 53

     'ITHR':64,  # Default 64
     'IBIAS': 100,  # Default 50
     'VRESET': 110,  # Default TB 143, 110 for lower THR, Lars dec proposal 128
     'ICASN': 0,  # Lars proposed 54
     'VCASP': 93,  # Default 93
     "VCASC": 228,  # Lars proposed 150
     "IDB": 100,  # Default 100
     'ITUNE': 170,  # Default TB 53, 150 for lower THR tuning
     'VCLIP': 255,  # Default 255
     'IDEL':255,


    # Enable VL and VH measurement and override
    # 'MON_EN_VH': 0,
    # 'MON_EN_VL': 0,
    # 'OVR_EN_VH': 0,
    # 'OVR_EN_VL': 0,
    # Enable analog monitoring pixel
    'EN_PULSE_ANAMON_L': 1,
    'ANAMON_SFN_L': 0b0001,

    'ANAMON_SFP_L': 0b1000,
    'ANAMONIN_SFN1_L': 0b1000,
    'ANAMONIN_SFN2_L': 0b1000,
    'ANAMONIN_SFP_L': 0b1000,
    # Enable hitor
    'SEL_PULSE_EXT_CONF': 0,

    # set readout cycle timing as in TB
    # 'FREEZE_START_CONF': 1,  # Default 1, TB 41
    # 'READ_START_CONF': 3,  # Default 3, TB 81
    # 'READ_STOP_CONF': 5,  # Default 5, TB 85
    # 'LOAD_CONF': 7,  # Default 7, TB 119
    # 'FREEZE_STOP_CONF': 8,  # Default 8, TB 120
    # 'STOP_CONF': 8  # Default 8, TB 120
    # set readout cycle timing as in TB/or as default in Pisa/or as in hot pixel study
    'FREEZE_START_CONF': 10,  # Default 1, TB 41
    'READ_START_CONF': 13,  # Default 3, TB 81
    'READ_STOP_CONF': 15,  # Default 5, TB 85
    'LOAD_CONF': 30,  # Default 7, TB 119
    'FREEZE_STOP_CONF': 31,  # Default 8, TB 120
    'STOP_CONF': 31  # Default 8, TB 120
}

class GDACTuning(ScanBase):
    scan_id = 'global_threshold_tuning'

    def _configure(self, start_column=0, stop_column=512, start_row=0, stop_row=512, VCAL_LOW=30, VCAL_HIGH=60, load_tdac_from=None,**_):
        '''
        Parameters
        ----------
        start_column : int [0:512]
            First column to scan
        stop_column : int [0:512]
            Column to stop the scan. This column is excluded from the scan.
        start_row : int [0:512]
            First row to scan
        stop_row : int [0:512]
            Row to stop the scan. This row is excluded from the scan.

        VCAL_LOW : int
            Injection DAC low value.
        VCAL_HIGH : int
            Injection DAC high value.
        '''

        self.data.start_column, self.data.stop_column, self.data.start_row, self.data.stop_row = start_column, stop_column, start_row, stop_row
        self.chip.masks['enable'][:, :] = False
        self.chip.masks['injection'][:, :] = False
        self.chip.masks['enable'][start_column:stop_column, start_row:stop_row] = True
        self.chip.masks['injection'][start_column:stop_column, start_row:stop_row] = True
        self.chip.masks['tdac'][start_column:stop_column, start_row:stop_row] = 0b100

        # Load TDAC from h5 file (optional)
        if load_tdac_from:
            with tb.open_file(load_tdac_from) as f:
                file_tdac = f.root.configuration_out.chip.masks.tdac[:]
                file_tdac = file_tdac[start_column:stop_column, start_row:stop_row]
                # Do not replace TDAC values with zeros from the file, use the default for those pixels
                self.chip.masks['tdac'][start_column:stop_column, start_row:stop_row] = \
                    np.where(
                        file_tdac != 0, file_tdac,
                        self.chip.masks['tdac'][start_column:stop_column, start_row:stop_row])

        # Disable W8R13 bad/broken columns (25, 160, 161, 224, 274, 383-414 included, 447) and pixels
        self.chip.masks['enable'][25,:] = False  # Many pixels don't fire
        self.chip.masks['enable'][160:162,:] = False  # Wrong/random ToT
        self.chip.masks['enable'][224,:] = False  # Many pixels don't fire
        self.chip.masks['enable'][274,:] = False  # Many pixels don't fire
        self.chip.masks['enable'][383:415,:] = False  # Wrong/random ToT
        self.chip.masks['enable'][447,:] = False  # Many pixels don't fire
        self.chip.masks['enable'][75,159] = False
        self.chip.masks['enable'][163,219] = False
        self.chip.masks['enable'][427,259] = False
        self.chip.masks['enable'][219,161] = False # disable hottest pixel on chip
        self.chip.masks['enable'][214,88] = False
        self.chip.masks['enable'][215,101] = False
        self.chip.masks['enable'][191:223,:] = False  # cols 191-223 are broken since Nov/dec very low THR

        # # Noisy/hot W8R13 pixels
        # for col, row in [(219, 161), (222, 188), (219, 192), (219, 129), (221, 125), (219, 190), (220, 205), (220, 144), (220, 168), (219, 179), (221, 136), (222, 186), (219, 163), (221, 205), (226, 135), (222, 174), (221, 199), (222, 185), (221, 203), (225, 181), (220, 123), (222, 142), (223, 143), (220, 154), (221, 149), (221, 179), (222, 120), (219, 125)] \
        #         + [(227, 167), (222, 177), (221, 156), (221, 196), (219, 170), (221, 150), (222, 172), (221, 132), (219, 133), (226, 146), (220, 187), (219, 200), (227, 142), (220, 173), (229, 142), (219, 131), (219, 195), (221, 126), (220, 136), (222, 159), (221, 188), (227, 207), (219, 167), (219, 193), (220, 131), (228, 158), (219, 198), (221, 139), (219, 211), (221, 177), (223, 189), (219, 156), (222, 144), (222, 184), (228, 200), (219, 174), (229, 197), (220, 184), (222, 133), (222, 171), (219, 152), (222, 147), (222, 197), (219, 132), (223, 167), (221, 163), (223, 126), (227, 199), (229, 168), (229, 188), (221, 165), (222, 139), (219, 217), (222, 161), (225, 143), (219, 187), (221, 202), (223, 141), (222, 134), (220, 204), (221, 209), (220, 191), (221, 183), (223, 137), (223, 188), (221, 219), (222, 140), (223, 146), (219, 142), (222, 219), (229, 134), (221, 168), (227, 145), (222, 199), (227, 211), (220, 201), (220, 217), (221, 190), (221, 216), (223, 204), (221, 186), (227, 178), (222, 125), (221, 122), (220, 190), (227, 196), (222, 141), (220, 145), (229, 215)] \
        #         + [(221, 174), (219, 188)] \
        #         + [(228, 203), (219, 180), (227, 137), (220, 146), (219, 146), (221, 137), (221, 172)] \
        #         + [(228, 181), (221, 211), (228, 201), (221, 123), (229, 173), (220, 143), (228, 219), (220, 135), (226, 141), (222, 217), (228, 188), (228, 156), (222, 122), (229, 191), (222, 160), (225, 146), (227, 155), (220, 179), (220, 175), (219, 137), (220, 149), (227, 171), (226, 178), (221, 200), (220, 198), (222, 181), (221, 198), (228, 216), (227, 174), (228, 133), (219, 182), (229, 149), (219, 136), (229, 163), (223, 148), (221, 120), (221, 160), (222, 213), (219, 175), (225, 203), (222, 165), (221, 170), (225, 186), (228, 190), (221, 143), (219, 194), (222, 194), (219, 124), (226, 133), (223, 199), (221, 192), (219, 205), (219, 139), (225, 207), (225, 197), (221, 169), (219, 153), (219, 186), (221, 148), (221, 131), (219, 173), (228, 169), (228, 150), (219, 191), (221, 129), (222, 193), (222, 196), (220, 186), (220, 148), (221, 189), (221, 134), (219, 135), (220, 209), (221, 159), (220, 182), (220, 169), (222, 215), (221, 171), (220, 121), (228, 209), (219, 123), (220, 153), (222, 201), (220, 210), (221, 151), (219, 207), (222, 162), (227, 132), (220, 203), (228, 198), (228, 144), (221, 140), (222, 192), (223, 217), (219, 196), (226, 199), (223, 177), (225, 193), (219, 181), (222, 178), (219, 144), (221, 164), (219, 171), (219, 201), (220, 125), (219, 130), (222, 207)] \
        #         :
        #     self.chip.masks['enable'][col,row] = False

        # W8R13 pixels that fire even when disabled
        # For these ones, we disable the readout of the whole double-column
        reg_values = [0xffff] * 16
        col_HV = list(range(448, 512))
        col_bad = [85, 109, 131, 145, 157, 163, 204, 205, 279, 282, 295, 327, 335, 450]
        #col_bad_tmp = list(range(0, 255))
        col_bad_tmp = list(range(0, 0))
        #col_disabled = col_HV + col_bad + col_bad_tmp
        col_disabled = col_HV + col_bad
        for col in col_disabled:
            dcol = col // 2
            reg_values[dcol//16] &= ~(1 << (dcol % 16))
        for i, v in enumerate(reg_values):
            # EN_RO_CONF
            self.chip._write_register(155+i, v)
            # EN_BCID_CONF
            self.chip._write_register(171+i, v)
            # EN_RO_RST_CONF
            self.chip._write_register(187+i, v)
            # EN_FREEZE_CONF
            self.chip._write_register(203+i, v)

        self.chip.masks.apply_disable_mask()
        self.chip.masks.update(force=True)

        for r in self.register_overrides:
            self.chip.registers[r].write(self.register_overrides[r])

        # Only check pixel that can respond
        self.sel_pixel = np.zeros(shape=(512, 512), dtype=bool)
        self.sel_pixel[self.data.start_column:self.data.stop_column, self.data.start_row:self.data.stop_row] = \
            (self.chip.masks['tdac'][self.data.start_column:self.data.stop_column, self.data.start_row:self.data.stop_row] != 0) \
            & self.chip.masks['enable'][self.data.start_column:self.data.stop_column, self.data.start_row:self.data.stop_row]
        print(np.count_nonzero(self.sel_pixel), "pixels are enabled")

        print(f"Setting VH and VL to {VCAL_HIGH} and {VCAL_LOW}")
        self.chip.registers["VL"].write(VCAL_LOW)
        self.chip.registers["VH"].write(VCAL_HIGH)

        self.chip.registers["SEL_PULSE_EXT_CONF"].write(0)

        self.data.hist_occ = oa.OccupancyHistogramming()

    def _scan(self, n_injections=100, gdac_value_bits=range(6, -1, -1), bcid_reset=True, **_):
        '''
        Global threshold tuning main loop

        Parameters
        ----------
        n_injections : int
            Number of injections.
        gdac_value_bits : iterable
            Bits to toggle during tuning. Should be monotone.
        '''

        def update_best_gdacs(mean_occ, best_gdacs, best_gdac_offsets):
            if np.abs(mean_occ - n_injections / 2.) < best_gdac_offsets:
                best_gdac_offsets = np.abs(mean_occ - n_injections / 2.)
                best_gdacs = gdac_new
            return best_gdacs, best_gdac_offsets

        def write_gdac_registers(gdac):
            ''' Write new GDAC setting for enabled flavors '''
            print("Setting ICASN to ", repr(gdac))
            self.chip.registers['ICASN'].write(gdac)
            self.chip.configuration['registers']['ICASN'] = int(gdac)

        # Set GDACs to start value
        start_value = 2 ** gdac_value_bits[0]
        # FIXME: keep track of chip config here, since it is not provided by bdaq53 yet?
        gdac_new = start_value
        best_gdacs = start_value

        self.log.info('Searching optimal global threshold setting.')
        #self.data.pbar = tqdm(total=len(gdac_value_bits) * self.chip.masks.get_mask_steps() * 2, unit=' Mask steps', delay=0.1)
        self.data.pbar = tqdm(total=(len(gdac_value_bits) + 1) * self.chip.masks.get_mask_steps(), unit=' Mask steps')
        for scan_param_id in range(len(gdac_value_bits)):
            # Set the GDAC bit in all flavours
            gdac_bit = gdac_value_bits[scan_param_id]
            gdac_new = np.bitwise_or(gdac_new, 1 << gdac_bit)
            write_gdac_registers(gdac_new)

            # Calculate new GDAC from hit occupancies: median pixel hits < n_injections / 2 --> decrease global threshold
            hist_occ = self.get_occupancy(scan_param_id, n_injections, bcid_reset)
            for i, b in enumerate(np.histogram(hist_occ[self.sel_pixel], bins=10, range=[0, n_injections])[0]):
                print(f'\x1b[36m{i*n_injections/10:3.0f}', '█' * int(b * 40 / np.count_nonzero(self.sel_pixel)), end='\x1b[0m\n')
            mean_occ = np.mean(hist_occ[self.sel_pixel & (hist_occ <= n_injections)])
            # mean_occ = np.median(hist_occ[self.sel_pixel])
            self.log.info(f'Mean Occ of {mean_occ:.2f} at ICASN = {gdac_new}')

            # Binary search does not have to converge to best solution for not exact matches
            # Thus keep track of best solution and set at the end if needed
            if not scan_param_id:  # First iteration --> initialize best gdac settings
                best_gdac_offset = np.abs(mean_occ - n_injections / 2.)
            else:  # Update better settings
                best_gdacs, best_gdac_offset = update_best_gdacs(mean_occ, best_gdacs, best_gdac_offset)

            # Seedup by skipping remaining iterations if result for all selected flavors is already found
            if (mean_occ == n_injections / 2.) | np.isnan(mean_occ):
                self.log.info('Found best result, skip remaining iterations')
                break


            # Update GDACS from measured mean occupancy
            if not np.isnan(mean_occ) and mean_occ > n_injections / 2.:  # threshold too low
                gdac_new = np.bitwise_and(gdac_new, ~(1 << gdac_bit))  # decrease threshold

        else:  # Loop finished but last bit = 0 still has to be checked
            #self.data.pbar.close()
            scan_param_id += 1
            gdac_new = np.bitwise_and(gdac_new, ~(1 << gdac_bit))
            # Do not check if setting was already used before, safe time of one iteration
            if best_gdacs != gdac_new:
                write_gdac_registers(gdac_new)
                hist_occ = self.get_occupancy(scan_param_id, n_injections, bcid_reset)
                for i, b in enumerate(np.histogram(hist_occ[self.sel_pixel], bins=10, range=[0, n_injections])[0]):
                    print(f'\x1b[36m{i*n_injections/10:3.0f}', '█' * int(b * 40 / np.count_nonzero(self.sel_pixel)), end='\x1b[0m\n')
                mean_occ = np.mean(hist_occ[self.sel_pixel & (hist_occ <= n_injections)])
                # mean_occ = np.median(hist_occ[self.sel_pixel])
                self.log.info(f'Mean Occ of {mean_occ:.2f} at ICASN = {gdac_new}')
                best_gdacs, best_gdac_offset = update_best_gdacs(mean_occ, best_gdacs, best_gdac_offset)
        self.data.pbar.close()

        # Mean is the correct thing to do, not median, because pixels will mostly fire 0% or 100% of the times unless
        # they are exactly on-threshold (which is relatively unlikely), and the median will be naturally drawn to
        # either 0 or 100. But if we do the mean, we must cut the noisy pixels away (those with occupancy > 100%).
        self.log.success(f'Optimal ICASN value is {best_gdacs}')
        # self.log.success('Optimal ICASN value is {0:1.0f} with median occupancy {1:1.0f}'.format(best_gdacs, int(mean_occ)))

        # Set final result
        self.data.best_gdacs = best_gdacs
        write_gdac_registers(best_gdacs)
        self.data.hist_occ.close()  # stop analysis process

    def get_occupancy(self, scan_param_id, n_injections, bcid_reset):
        ''' Analog scan and stuck pixel scan '''
        # Set new TDACs
        # Inject target charge
        with self.readout(scan_param_id=scan_param_id, callback=self.analyze_data_online):
            shift_and_inject(scan=self, n_injections=n_injections, pbar=self.data.pbar, scan_param_id=scan_param_id, reset_bcid=bcid_reset)
        #self.update_pbar_with_word_rate(pbar)
        # Get hit occupancy using online analysis
        occupancy = self.data.hist_occ.get()

        return occupancy

    def analyze_data_online(self, data_tuple):
        raw_data = data_tuple[0]
        self.data.hist_occ.add(raw_data)
        super(GDACTuning, self).handle_data(data_tuple)

    def analyze_data_online_no_save(self, data_tuple):
        raw_data = data_tuple[0]
        self.data.hist_occ.add(raw_data)

    def _analyze(self):
        pass


if __name__ == '__main__':
    with GDACTuning(scan_config=scan_configuration, register_overrides=register_overrides) as tuning:
        tuning.start()
