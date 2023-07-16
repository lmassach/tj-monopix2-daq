#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

import time
from math import inf
from tqdm import tqdm

from tjmonopix2.analysis import analysis, plotting
from tjmonopix2.system.scan_base import ScanBase

scan_configuration = {
    'start_column': 0,
    'stop_column': 224,
    'start_row': 0,
    'stop_row': 512,

    'scan_timeout': 60,  # Timeout for scan after which the scan will be stopped, in seconds; if False no limit on scan time
    'max_triggers': False,  # Number of maximum received triggers after stopping readout, if False no limit on received trigger
    # NOTE you can set no limit on either of the above, and the scan will just run until CTRL+C is used to stop it
}


class SimpleScan(ScanBase):
    scan_id = 'simple_scan'

    def _configure(self, scan_timeout=10, max_triggers=False, start_column=0, stop_column=512, start_row=0, stop_row=512, **_):
        if scan_timeout and max_triggers:
            self.log.warning('You should only use one of the stop conditions at a time.')

        self.chip.masks['enable'][start_column:stop_column, start_row:stop_row] = True
        self.chip.masks.apply_disable_mask()
        self.chip.masks.update()

        if max_triggers:
            self.daq.configure_tlu_veto_pulse(veto_length=500)
            self.daq.configure_tlu_module(max_triggers=max_triggers)

        # Configure graceful exit on CTRL+C
        self.enable_graceful_exit()

    def _scan(self, scan_timeout=10, max_triggers=False, **_):
        start_time = time.time()
        if scan_timeout:
            self.pbar = tqdm(total=scan_timeout, unit=' s')
            end_time = start_time + scan_timeout
        elif max_triggers:
            self.pbar = tqdm(total=max_triggers, unit=' Triggers')
            end_time = inf
        else:
            self.pbar = tqdm(unit=' s')
            end_time = inf

        with self.readout():
            if max_triggers:
                self.daq.enable_tlu_module()
                triggers = 0

            while not (self.should_exit_gracefully or time.time() > end_time or (max_triggers and triggers >= max_triggers)):
                if max_triggers:
                    triggers = self.daq.get_trigger_counter()
                else:
                    now = time.time()

                time.sleep(1)

                # Update progress bar
                try:
                    if max_triggers:
                        self.pbar.update(self.daq.get_trigger_counter() - triggers)
                    else:
                        self.pbar.update(time.time() - now)
                except ValueError:
                    pass

        self.pbar.close()
        if max_triggers:
            self.daq.disable_tlu_module()

        self.log.success('Scan finished')

    def _analyze(self):
        with analysis.Analysis(raw_data_file=self.output_filename + '.h5', **self.configuration['bench']['analysis']) as a:
            a.analyze_data()

        if self.configuration['bench']['analysis']['create_pdf']:
            with plotting.Plotting(analyzed_data_file=a.analyzed_data_file) as p:
                p.create_standard_plots()


if __name__ == "__main__":
    with SimpleScan(scan_config=scan_configuration) as scan:
        scan.start()
