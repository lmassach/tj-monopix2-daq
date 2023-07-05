#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# Hephy, Austrian Academy of Sciences, Vienna
# ------------------------------------------------------------
#

from tjmonopix2.analysis import analysis, plotting
from tjmonopix2.scans.shift_and_inject import (get_scan_loop_mask_steps,
                                               shift_and_inject)
from tjmonopix2.system.scan_base import ScanBase
from tqdm import tqdm

import os
COLUMN = int(os.environ.get('COLUMN', 0))

# Only enable one pixel for this scan
scan_configuration = {
    'start_column': COLUMN,
    'stop_column': COLUMN+1,
    'start_row': 0,
    'stop_row': 1,

    'n_injections': 100,
    'VCAL_HIGH': 140,
    'VCAL_LOW_start': 120,
    'VCAL_LOW_stop': 0,
    'VCAL_LOW_step': -1
}


class TimewalkScan(ScanBase):
    scan_id = 'timewalk_scan'

    def _configure(self, start_column=0, stop_column=512, start_row=0, stop_row=512, **_):
        self.chip.masks['enable'][start_column:stop_column, start_row:stop_row] = True
        self.chip.masks['injection'][start_column:stop_column, start_row:stop_row] = True
        self.chip.masks['hitor'][start_column:stop_column, start_row:stop_row] = True

        # Enable readout and bcid/freeze distribution only to columns we actually use
        dcols_enable = [0] * 16
        for c in range(start_column, stop_column):
            dcols_enable[c // 32] |= (1 << ((c >> 1) & 15))
        for c in []:  # List of disabled columns
            dcols_enable[c // 32] &= ~(1 << ((c >> 1) & 15))
        for i, v in enumerate(dcols_enable):
            self.chip._write_register(155 + i, v)  # EN_RO_CONF
            self.chip._write_register(171 + i, v)  # EN_BCID_CONF
            self.chip._write_register(187 + i, v)  # EN_RO_RST_CONF
            self.chip._write_register(203 + i, v)  # EN_FREEZE_CONF

        self.chip.registers["ITHR"].write(30)
        self.chip.registers["IBIAS"].write(60)
        self.chip.registers["VRESET"].write(50)
        self.chip.registers["VCASP"].write(40)
        self.chip.registers["VCASC"].write(140)
        self.chip.registers["IDB"].write(150)
        self.chip.registers["ITUNE"].write(200)

        self.chip.masks.apply_disable_mask()
        self.chip.masks.update(force=True)

        self.chip.registers["SEL_PULSE_EXT_CONF"].write(0)
        self.chip.registers["CMOS_TX_EN_CONF"].write(1)

        # TDC config
        self.daq['tdc'].EN_WRITE_TIMESTAMP = 0
        self.daq['tdc'].EN_TRIGGER_DIST = 1
        self.daq['tdc'].EN_NO_WRITE_TRIG_ERR = 1
        self.daq.configure_tdc_module()
        self.daq.enable_tdc_module()

        # Pulsing config
        self.daq.set_LEMO_MUX('LEMO_MUX_TX0', 1)
        self.daq.configure_cmd_loop_start_pulse(width=200, delay=270)

    def _scan(self, n_injections=100, VCAL_HIGH=80, VCAL_LOW_start=80, VCAL_LOW_stop=40, VCAL_LOW_step=-1, **_):
        """
        Injects charges from VCAL_LOW_START to VCAL_LOW_STOP in steps of VCAL_LOW_STEP while keeping VCAL_HIGH constant.
        """

        self.chip.registers["VH"].write(VCAL_HIGH)
        vcal_low_range = range(VCAL_LOW_start, VCAL_LOW_stop, VCAL_LOW_step)

        pbar = tqdm(total=get_scan_loop_mask_steps(self.chip) * len(vcal_low_range), unit='Mask steps')
        for scan_param_id, vcal_low in enumerate(vcal_low_range):
            self.chip.registers["VL"].write(vcal_low)

            self.store_scan_par_values(scan_param_id=scan_param_id, vcal_high=VCAL_HIGH, vcal_low=vcal_low)
            with self.readout(scan_param_id=scan_param_id):
                shift_and_inject(chip=self.chip, n_injections=n_injections, pbar=pbar, scan_param_id=scan_param_id)
        pbar.close()
        self.log.success('Scan finished')

    def _analyze(self):
        with analysis.Analysis(raw_data_file=self.output_filename + '.h5', **self.configuration['bench']['analysis']) as a:
            a.analyze_data(low_ram=True)

        if self.configuration['bench']['analysis']['create_pdf']:
            with plotting.Plotting(analyzed_data_file=a.analyzed_data_file) as p:
                p.create_standard_plots()


if __name__ == "__main__":
    with TimewalkScan(scan_config=scan_configuration) as scan:
        scan.start()
