#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#
"""Enables some pixels, inject a fixed charge only on one."""

from tjmonopix2.analysis import analysis
from tjmonopix2.system.scan_base import ScanBase

scan_configuration = {
    # Pixels to enable
    'start_column': 222,
    'stop_column': 222,
    'start_row': 188,
    'stop_row': 188,

    #'start_column': 219,
    #'stop_column': 220,
    #'start_row': 161,
    #'stop_row': 162,


    # Pixel to inject
    'inj_col': 217, # 220
    'inj_row': 140, # 200

    'n_injections': 10000,
    'reset_bcid': False,
}

register_overrides = {
    'VL': 1,  # Inject VH-VL (fixed charge)
    'VH': 140,
    'ITHR': 64,  # Default 64
    'IBIAS': 50,  # Default 50
    'VRESET': 110,  # Default 143, 110 for lower THR
    'ICASN': 200,  # Default TB 0 , 150 for -3V , 200 for -6V
    'VCASP': 93,  # Default 93
    "VCASC": 228,  # Default 228
    "IDB": 35,  # Default 100
    'ITUNE': 150,  # Default TB 53, 150 for lower THR tuning
    'VCLIP': 255,  # Default 255

    # # set readout cycle timing as in TB
    # 'FREEZE_START_CONF': 41,  # Default 1, TB 41
    # 'READ_START_CONF': 81,  # Default 3, TB 81
    # 'READ_STOP_CONF': 85,  # Default 5, TB 85
    # 'LOAD_CONF': 119,  # Default 7, TB 119
    # 'FREEZE_STOP_CONF': 120,  # Default 8, TB 120
    # 'STOP_CONF': 120  # Default 8, TB 120

    'FREEZE_START_CONF': 10,  # Default 1
    'READ_START_CONF': 13,  # Default 3
    'READ_STOP_CONF': 15,  # Default 5
    'LOAD_CONF': 30,  # Default 7
    'FREEZE_STOP_CONF': 31,  # Default 8
    'STOP_CONF': 31  # Default 8
}


class HotPixelScan(ScanBase):
    scan_id = 'hot_pixel_scan'

    def _configure(self, start_column=0, stop_column=512, start_row=0, stop_row=512, inj_col=0, inj_row=0, **_):
        # Setting the enable mask to False is equivalent to setting tdac to 0 = 0b000
        # This prevents the discriminator from firing, but we are not sure whether it disables the analog FE or not
        self.chip.masks['enable'][:,:] = False
        self.chip.masks['injection'][:,:] = False
        self.chip.masks['enable'][start_column:stop_column, start_row:stop_row] = True
        self.chip.masks['tdac'][start_column:stop_column, start_row:stop_row] = 0b100  # TDAC=4 for threshold tuning

        # Injected pixel
        self.chip.masks['enable'][inj_col,inj_row] = True
        self.chip.masks['injection'][inj_col,inj_row] = True
        self.chip.masks['tdac'][inj_col,inj_row] = 6  # default should  be 4

        # self.chip.masks['enable'][222,188] = True # enable an hot pixel
        # self.chip.masks['tdac'][222,188] = 4
        # self.chip.masks['enable'][218,155] = True # enable an hot pixel
        # self.chip.masks['tdac'][218,155] = 4
        #self.chip.masks['enable'][219,192] = True # enable 3 hot pixel
        #self.chip.masks['tdac'][219,192] = 4
        # self.chip.masks['enable'][214,149] = True # enable an hot pixel
        # self.chip.masks['tdac'][214,149] = 1
        # self.chip.masks['enable'][1,140] = True # enable an hot pixel
        # self.chip.masks['tdac'][1,140] = 3
        self.chip.masks['enable'][220,140] = True # enable an hot pixel
        self.chip.masks['tdac'][220,140] = 3

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

        self.daq.rx_channels['rx0']['DATA_DELAY'] = 14

        self.chip.registers["SEL_PULSE_EXT_CONF"].write(0)

    def _scan(self, n_injections=100, reset_bcid=False, **_):
        """
        Injects charges from VCAL_LOW_START to VCAL_LOW_STOP in steps of VCAL_LOW_STEP while keeping VCAL_HIGH constant.
        """

        with self.readout():
            self.chip.inject(repetitions=n_injections, reset_bcid=reset_bcid)

        self.log.success('Scan finished')

    def _analyze(self):
        with analysis.Analysis(raw_data_file=self.output_filename + '.h5', **self.configuration['bench']['analysis']) as a:
            a.analyze_data()


if __name__ == "__main__":
    with HotPixelScan(scan_config=scan_configuration, register_overrides=register_overrides) as scan:
        scan.start()
