#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

from tjmonopix2.analysis import analysis
from tjmonopix2.scans.shift_and_inject import get_scan_loop_mask_steps, shift_and_inject
from tjmonopix2.system.scan_base import ScanBase
from tqdm import tqdm
import tables as tb
import numpy as np

scan_configuration = {
    'start_column': 218, # 216
    'stop_column': 223, #230
    'start_row': 120, #120
    'stop_row': 220, #220

    'n_injections': 100,
    'VCAL_HIGH': 140,
    'VCAL_LOW_start': 139,
    'VCAL_LOW_stop': 65,
    'VCAL_LOW_step': -1,

    'reset_bcid': False,  # Reset BCID counter before every injection
    'load_tdac_from': None,  # Optional h5 file to load the TDAC values from
}

register_overrides = {
    'ITHR': 64,  # Default 64
    'IBIAS': 50,  # Default 50
    'VRESET': 143,  # Default 143
    'ICASN': 150,  # Default 0
    'VCASP': 93,  # Default 93
    "VCASC": 228,  # Default 228
    "IDB": 100,  # Default 100
    'ITUNE': 100,  # Default 53
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
    'FREEZE_START_CONF': 1,  # Default 1, TB 41
    'READ_START_CONF': 3,  # Default 3, TB 81
    'READ_STOP_CONF': 5,  # Default 5, TB 85
    'LOAD_CONF': 7,  # Default 7, TB 119
    'FREEZE_STOP_CONF': 8,  # Default 8, TB 120
    'STOP_CONF': 8  # Default 8, TB 120

}


class ThresholdScan(ScanBase):
    scan_id = 'threshold_scan'

    def _configure(self, start_column=0, stop_column=512, start_row=0, stop_row=512, load_tdac_from=None, **_):
        # Setting the enable mask to False is equivalent to setting tdac to 0 = 0b000
        # This prevents the discriminator from firing, but we are not sure whether it disables the analog FE or not
        self.chip.masks['enable'][:,:] = False
        self.chip.masks['injection'][:,:] = False
        self.chip.masks['enable'][start_column:stop_column, start_row:stop_row] = True
        self.chip.masks['injection'][start_column:stop_column, start_row:stop_row] = True
        # TDAC=4 for threshold tuning 0b100
        self.chip.masks['tdac'][start_column:stop_column, start_row:stop_row] = 0b100  # TDAC=4 (default)
        # self.chip.masks['tdac'][start_column:stop_column, start_row:stop_row] = 0b110  # TDAC=6
        # self.chip.masks['tdac'][start_column:stop_column, start_row:stop_row] = 7  # TDAC=7 DONT USE DECIMAL!!!!
        # self.chip.masks['tdac'][220:222,159] = 1  # TDAC=1 TDAC=1 DONT USE DECIMAL!!!!
        #self.chip.masks['tdac'][222,188] = 0b111  # TDAC=7 for hot pixels
        #self.chip.masks['tdac'][221,105] = 0b111  # TDAC=7 for hot pixels
        #self.chip.masks['tdac'][221,174] = 0b111  # TDAC=7 for hot pixels

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
        for col in [85, 109, 131, 145, 157, 163, 204, 205, 279, 282, 295, 327, 335]:
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
        # Enable HITOR (active high) on all columns, all rows
        for i in range(512//16):
            self.chip._write_register(18+i, 0xffff)
            self.chip._write_register(50+i, 0xffff)

        self.daq.rx_channels['rx0']['DATA_DELAY'] = 14

        self.chip.registers["SEL_PULSE_EXT_CONF"].write(0)

    def _scan(self, n_injections=100, VCAL_HIGH=80, VCAL_LOW_start=80, VCAL_LOW_stop=40, VCAL_LOW_step=-1, reset_bcid=False, **_):
        """
        Injects charges from VCAL_LOW_START to VCAL_LOW_STOP in steps of VCAL_LOW_STEP while keeping VCAL_HIGH constant.
        """

        self.chip.registers["VH"].write(VCAL_HIGH)
        vcal_low_range = range(VCAL_LOW_start, VCAL_LOW_stop, VCAL_LOW_step)

        pbar = tqdm(total=get_scan_loop_mask_steps(self) * len(vcal_low_range), unit='Mask steps', smoothing=0)
        for scan_param_id, vcal_low in enumerate(vcal_low_range):
            self.chip.registers["VL"].write(vcal_low)

            self.store_scan_par_values(scan_param_id=scan_param_id, vcal_high=VCAL_HIGH, vcal_low=vcal_low)
            with self.readout(scan_param_id=scan_param_id):
                shift_and_inject(scan=self, n_injections=n_injections, pbar=pbar, scan_param_id=scan_param_id, reset_bcid=reset_bcid)
        pbar.close()
        self.log.success('Scan finished')

    def _analyze(self):
        with analysis.Analysis(raw_data_file=self.output_filename + '.h5', **self.configuration['bench']['analysis']) as a:
            a.analyze_data()


if __name__ == "__main__":
    with ThresholdScan(scan_config=scan_configuration, register_overrides=register_overrides) as scan:
        scan.start()
