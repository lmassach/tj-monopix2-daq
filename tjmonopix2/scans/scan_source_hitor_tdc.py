#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

import time

from tqdm import tqdm

from tjmonopix2.system.scan_base import ScanBase
from tjmonopix2.analysis import analysis

scan_configuration = {
    'start_column': 0,
    'stop_column': 224,
    'start_row': 0,
    'stop_row': 512,
}


registers = ['IBIAS', 'ICASN', 'IDB', 'ITUNE', 'ITHR', 'ICOMP', 'IDEL', 'VRESET', 'VCASP', 'VH', 'VL', 'VCLIP', 'VCASC', 'IRAM']


class SourceScan(ScanBase):
    scan_id = 'source_scan_hitor_tdc'

    def _configure(self, start_column=0, stop_column=512, start_row=0, stop_row=512, **_):
        self.chip.masks['enable'][start_column:stop_column, start_row:stop_row] = True
        self.chip.masks['injection'][start_column:stop_column, start_row:stop_row] = False
        self.chip.masks['tdac'][start_column:stop_column, start_row:stop_row] = 0b100
        self.chip.masks['hitor'][start_column:stop_column, start_row:stop_row] = True

        self.chip.masks.apply_disable_mask()
        self.chip.masks.update(force=True)

        self.chip.registers["SEL_PULSE_EXT_CONF"].write(0)
        self.chip.registers["CMOS_TX_EN_CONF"].write(1)
        
        # configure TDC in FPGA
        self.daq['tdc'].EN_WRITE_TIMESTAMP = 1
        self.daq['tdc'].EN_TRIGGER_DIST = 1
        self.daq['tdc'].EN_NO_WRITE_TRIG_ERR = 1
        self.daq.configure_tdc_module()
        self.daq.enable_tdc_module()


    def _scan(self, scan_time=900, **_):

        pbar = tqdm(total=int(scan_time), unit='s')
        now = time.time()
        end_time = now + scan_time
        with self.readout(scan_param_id=0):
            while now < end_time:
                sleep_time = min(1, end_time - now)
                time.sleep(sleep_time)
                last_time = now
                now = time.time()
                pbar.update(int(round(now - last_time)))
        pbar.close()

        ret = {}
        for r in registers:
            ret[r] = self.chip.registers[r].read()
        self.scan_registers = ret

        self.log.success('Scan finished')

    def _analyze(self):
        self.configuration['bench']['analysis']['cluster_hits'] = True
        with analysis.Analysis(raw_data_file=self.output_filename + '.h5', **self.configuration['bench']['analysis']) as a:
            a.analyze_data()


if __name__ == "__main__":
    with SourceScan(scan_config=scan_configuration) as scan:
        scan.start()
