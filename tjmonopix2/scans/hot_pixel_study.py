#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#
"""Enables some pixels, inject a fixed charge only on one."""

from tjmonopix2.analysis import analysis
from tjmonopix2.scans.shift_and_inject import get_scan_loop_mask_steps, shift_and_inject
from tjmonopix2.system.scan_base import ScanBase
from tqdm import tqdm
from plotting_scurves import Plotting

scan_configuration = {
    # Pixels to enable
    'start_column': 216,
    'stop_column': 222,
    'start_row': 120,
    'stop_row': 220,

    # Pixel to inject
    'inj_col': 219,
    'inj_row': 140,

    'n_injections': 100,
}

register_overrides = {
    'VL': 1,  # Inject VH-VL (fixed charge)
    'VH': 140,
    'ITHR': 64,  # Default 64
    'IBIAS': 50,  # Default 50
    'VRESET': 143,  # Default 143
    'ICASN': 90,  # Default 0
    'VCASP': 93,  # Default 93
    "VCASC": 228,  # Default 228
    "IDB": 100,  # Default 100
    'ITUNE': 53,  # Default 53
}


class HotPixelScan(ScanBase):
    scan_id = 'hot_pixel_scan'

    def _configure(self, start_column=0, stop_column=512, start_row=0, stop_row=512, inj_col=0, inj_row=0, **_):
        # Setting the enable mask to False is equivalent to setting tdac to 0 = 0b000
        # This prevents the discriminator from firing, but we are not sure whether it disables the analog FE or not
        self.chip.masks['enable'][:,:] = False
        self.chip.masks['injection'][:,:] = False
        self.chip.masks['enable'][start_column:stop_column, start_row:stop_row] = True
        self.chip.masks['injection'][inj_col,inj_row] = True
        self.chip.masks['tdac'][start_column:stop_column, start_row:stop_row] = 0b100  # TDAC=4 for threshold tuning

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

        self.chip.masks.apply_disable_mask()
        self.chip.masks.update(force=True)

        for r in self.register_overrides:
            self.chip.registers[r].write(self.register_overrides[r])

        self.daq.rx_channels['rx0']['DATA_DELAY'] = 14

        self.chip.registers["SEL_PULSE_EXT_CONF"].write(0)

    def _scan(self, n_injections=100, **_):
        """
        Injects charges from VCAL_LOW_START to VCAL_LOW_STOP in steps of VCAL_LOW_STEP while keeping VCAL_HIGH constant.
        """

        with self.readout():
            self.chip.inject(repetitions=n_injections)

        self.log.success('Scan finished')

    def _analyze(self):
        with analysis.Analysis(raw_data_file=self.output_filename + '.h5', **self.configuration['bench']['analysis']) as a:
            a.analyze_data()


if __name__ == "__main__":
    with HotPixelScan(scan_config=scan_configuration, register_overrides=register_overrides) as scan:
        scan.start()
