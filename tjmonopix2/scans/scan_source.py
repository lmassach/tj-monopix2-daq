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
    'start_column':0,
    'stop_column': 382,
    'start_row': 0,
    'stop_row': 512,

    'scan_time': 60,  # seconds
}

register_overrides = {
    'ITHR': 64,
    'IBIAS': 50,
    'VRESET': 143,
    'ICASN': 0,
    'VCASP': 93,
    'IDB': 100,
    # 'MON_EN_IDB': 1
}

registers = ['IBIAS', 'ICASN', 'IDB', 'ITUNE', 'ITHR', 'ICOMP', 'IDEL', 'VRESET', 'VCASP', 'VH', 'VL', 'VCLIP', 'VCASC', 'IRAM']


class SourceScan(ScanBase):
    scan_id = 'source_scan'

    def _configure(self, start_column=0, stop_column=512, start_row=0, stop_row=512, **_):
        self.chip.masks['enable'][:,:] = False
        self.chip.masks['injection'][:,:] = False
        self.chip.masks['enable'][start_column:stop_column, start_row:stop_row] = True
        self.chip.masks['injection'][start_column:stop_column, start_row:stop_row] = False
        self.chip.masks['tdac'][start_column:stop_column, start_row:stop_row] = 0b100

        #Masking noisy pixels (W14R12 chip, HV FEs)
        #self.chip.masks['tdac'][483, 379] = 0b000
        #self.chip.masks['tdac'][485, 352] = 0b000
        #self.chip.masks['tdac'][452, 248] = 0b000
        #self.chip.masks['tdac'][480, 317] = 0b000
        #self.chip.masks['tdac'][465, 302] = 0b000
        #self.chip.masks['tdac'][461, 126] = 0b000
        #self.chip.masks['tdac'][489, 510] = 0b000
        #self.chip.masks['tdac'][483, 340] = 0b000
        #self.chip.masks['tdac'][491, 450] = 0b000
        #self.chip.masks['tdac'][494, 463] = 0b000
        #self.chip.masks['tdac'][486, 498] = 0b000
        #self.chip.masks['tdac'][476, 24] = 0b000
        #self.chip.masks['tdac'][493, 242] = 0b000
        #self.chip.masks['tdac'][474, 53] = 0b000
        #self.chip.masks['tdac'][476, 211] = 0b000
        #self.chip.masks['tdac'][487, 16] = 0b000

        #self.chip.masks['hitor'][0, 0] = True

        self.chip.masks.apply_disable_mask()
        self.chip.masks.update(force=True)

        self.chip.registers["SEL_PULSE_EXT_CONF"].write(0)

        for r in self.register_overrides:
            if r != 'scan_time':
                self.chip.registers[r].write(self.register_overrides[r])
            #print("Write: ", r, " to ", self.register_overrides[r])

        self.daq.rx_channels['rx0']['DATA_DELAY'] = 14

    def _scan(self, scan_time=60, **_):
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
        # for r in registers:
        #     ret[r] = self.chip.registers[r].read()
        self.scan_registers = ret

        self.log.success('Scan finished')

    def _analyze(self):
        self.hist_occ = 0
        self.hist_tot = 0
        with analysis.Analysis(raw_data_file=self.output_filename + '.h5', **self.configuration['bench']['analysis']) as a:
            a.analyze_data()
            #self.hist_occ = a.hist_occ
            #self.hist_tot = a.hist_tot


if __name__ == "__main__":
    with SourceScan(scan_config=scan_configuration, register_overrides=register_overrides) as scan:
        scan.start()
