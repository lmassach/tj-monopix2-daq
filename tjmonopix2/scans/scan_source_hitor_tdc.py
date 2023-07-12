#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

import time
import signal

from tqdm import tqdm

from tjmonopix2.system.scan_base import ScanBase
from tjmonopix2.analysis import analysis
import os


STOP_RUNNING = False
def sig_handler(sig, _):
    global STOP_RUNNING
    STOP_RUNNING = True
signal.signal(signal.SIGINT, sig_handler)


IDEL = int(os.environ.get('IDEL', 88))

scan_configuration = {
    'start_column': 448,
    'stop_column': 496,
    'start_row': 0,
    'stop_row': 512,
    'scan_time': 3600,
}


registers = ['IBIAS', 'ICASN', 'IDB', 'ITUNE', 'ITHR', 'ICOMP', 'IDEL', 'VRESET', 'VCASP', 'VH', 'VL', 'VCLIP', 'VCASC', 'IRAM']


class SourceScan(ScanBase):
    scan_id = 'source_scan_hitor_tdc'

    def _configure(self, start_column=0, stop_column=512, start_row=0, stop_row=512, **_):
        self.chip.masks['enable'][start_column:stop_column, start_row:stop_row] = True
        self.chip.masks['injection'][start_column:stop_column, start_row:stop_row] = False
        self.chip.masks['hitor'][:, :] = False
        self.chip.masks['hitor'][start_column:stop_column, start_row:stop_row] = True
        
        self.chip.masks['enable'][477, 0:256] = False
        self.chip.masks['hitor'][477,  0:256] = False

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

        self.chip.registers["SEL_PULSE_EXT_CONF"].write(0)
        self.chip.registers["CMOS_TX_EN_CONF"].write(1)

        # self.chip.registers["ITHR"].write(35)
        # self.chip.registers["VRESET"].write(100)
        # self.chip.registers["VCASP"].write(40)
        # self.chip.registers["IBIAS"].write(60)
        # self.chip.registers["ICASN"].write(8)
        self.chip.registers["IDEL"].write(IDEL)
        
        # configure TDC in FPGA
        self.daq['tdc'].EN_WRITE_TIMESTAMP = 1
        self.daq['tdc'].EN_TRIGGER_DIST = 1
        self.daq['tdc'].EN_NO_WRITE_TRIG_ERR = 1
        self.daq.configure_tdc_module()
        self.daq.enable_tdc_module()


    def _scan(self, scan_time=120, **_):
        
        temp =  self.daq.get_temperature_NTC(connector=7)
        self.log.info(f'Chip temperature: {temp}C')

        pbar = tqdm(total=int(scan_time), unit='s')
        now = time.time()
        end_time = now + scan_time
        with self.readout(scan_param_id=0):
            while now < end_time and not STOP_RUNNING:
                sleep_time = min(1, end_time - now)
                time.sleep(sleep_time)
                last_time = now
                now = time.time()
                pbar.update(int(round(now - last_time)))
        if STOP_RUNNING:
            print("\x1b[36mTrying to stop gracefully, use CTRL+\\ if you want to kill now\x1b[0m")
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
