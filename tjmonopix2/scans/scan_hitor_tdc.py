#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

from tjmonopix2.analysis import analysis, plotting
from tjmonopix2.scans.shift_and_inject import (get_scan_loop_mask_steps,
                                               shift_and_inject)
from tjmonopix2.system.scan_base import ScanBase
from tqdm import tqdm

scan_configuration = {
    'start_column': 295,
    'stop_column': 295+16,
    'start_row': 0,
    'stop_row': 512,
}


class AnalogScan(ScanBase):
    scan_id = 'tdc_hitor_scan'

    def _configure(self, start_column=0, stop_column=512, start_row=0, stop_row=512, **_):
        self.chip.masks['enable'][start_column, start_row]  = True
        self.chip.masks['injection'][start_column, start_row] = True
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

        self.chip.masks.apply_disable_mask()
        self.chip.masks.update(force=True)

        self.chip.registers["VL"].write(10)
        self.chip.registers["VH"].write(150)
        self.chip.registers["CMOS_TX_EN_CONF"].write(1)
        self.chip.registers["SEL_PULSE_EXT_CONF"].write(0)

        # TDC config
        self.daq['tdc'].EN_WRITE_TIMESTAMP = 1
        self.daq['tdc'].EN_TRIGGER_DIST = 1
        self.daq['tdc'].EN_NO_WRITE_TRIG_ERR = 1
        self.daq.configure_tdc_module()
        self.daq.enable_tdc_module()

        # Pulsing config
        self.daq.set_LEMO_MUX('LEMO_MUX_TX0', 1)
        self.daq.configure_cmd_loop_start_pulse(width=200, delay=270)

        #temp = self.daq.get_temperature_NTC()
        #print(f'Temperature: {temp}C')

        # Configure graceful exit on CTRL+C
        self.enable_graceful_exit()

    def _scan(self, n_injections=100, start_column=0, stop_column=16, **_):
        no_cols = stop_column - start_column
        pixel_range = range(0, no_cols*512, 1)

        pbar = tqdm(total=get_scan_loop_mask_steps(self.chip) * len(pixel_range), unit='Mask steps')
        for scan_param_id, pixel_id in enumerate(pixel_range):
            row = (pixel_id)  % 512
            col = (pixel_id) // 512 + start_column
            # self.log.info(f'Step: col: {col}, row {row}')

            self.chip.masks['enable'][col, row] = True
            self.chip.masks['injection'][col, row] = True
            self.chip.masks.apply_disable_mask()
            self.chip.masks.update()

            self.store_scan_par_values(scan_param_id=scan_param_id, row=row, col=col)
            with self.readout(scan_param_id=scan_param_id):
                shift_and_inject(chip=self.chip, n_injections=n_injections, pbar=pbar, scan_param_id=scan_param_id)

            self.chip.masks['enable'][col, row] = False
            self.chip.masks['injection'][col, row] = False

            if self.should_exit_gracefully:
                break

        pbar.close()
        self.log.success('Scan finished')

    def _analyze(self):
        with analysis.Analysis(raw_data_file=self.output_filename + '.h5', **self.configuration['bench']['analysis'], ) as a:
            a.analyze_data(low_ram=True)

        #if self.configuration['bench']['analysis']['create_pdf']:
        #    with plotting.Plotting(analyzed_data_file=a.analyzed_data_file) as p:
        #        p.create_standard_plots()


if __name__ == "__main__":
    with AnalogScan(scan_config=scan_configuration) as scan:
        scan.start()
