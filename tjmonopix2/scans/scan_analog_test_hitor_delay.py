#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

from tqdm import tqdm
import time
import tables as tb
import numpy as np

from tjmonopix2.system.scan_base import ScanBase
from tjmonopix2.scans.shift_and_inject import shift_and_inject, get_scan_loop_mask_steps
from tjmonopix2.analysis import analysis

scan_configuration = {
    # 'start_column': 448,
    # 'stop_column': 464,
    'start_column': 300,
    'stop_column': 301,
    'start_row': 0,
    'stop_row': 512,
    'reset_bcid': True,  # Reset BCID counter before every injection
    'inj_pulse_start_delay': 1,  # Delay between BCID reset and inj pulse in 320 MHz clock cycles (there is also an offset of about 80 cycles)

    #load_tdac_from': None,  # Optional h5 file to load the TDAC values from



    # chipW8R13 File produced w BCID reset target=27 ITHR=30 ICASN=0 settings psub pwell=-6V cols=224-448 rows=0-512
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0/chip_0/20230324_191453_local_threshold_tuning_interpreted.h5'
    # chipW8R13 File produced w BCID reset target=25 ITHR=20 ICASN=0 settings psub pwell=-6V cols=224-448 rows=0-512
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0/chip_0/20230325_153148_local_threshold_tuning_interpreted.h5'
    # chipW8R13 File produced w BCID reset target=25 ITHR=64 ICASN=80 settings psub pwell=-6V cols=224-448 rows=0-512
    'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0_2023-03-25/chip_0/20230325_182214_local_threshold_tuning_interpreted.h5'
}

register_overrides = {
    'n_injections' : 10,
    "CMOS_TX_EN_CONF": 1,
    'VL': 1,
    'VH':101,


    #'ITHR': 64,
    #'IBIAS': 50,
    #'VRESET': 143,
    #'ICASN': 0,

    #'VCASP': 93,
    #'VCASC': 228,  # Default
    #'IDB': 100,

    # 'ITHR': 64,  # Default 64
    # 'IBIAS': 50,  # Default 50
    # 'VRESET': 110,  # Default 143, 110 for lower THR
    # 'ICASN': 0,  # Default TB 0 , 150 for -3V , 200 for -6V
    # 'VCASP': 93,  # Default 93
    # "VCASC": 228,  # Default 228
    # "IDB": 100,  # Default 100
    # 'ITUNE': 53,  # Default TB 53, 150 for lower THR tuning
    # 'VCLIP': 255,  # Default 255

     'ITHR':64,  # Default 64
     'IBIAS': 50,  # Default 50
     'VRESET': 110,  # Default TB 143, 110 for lower THR, Lars dec proposal 128
     'ICASN': 80,  # Lars proposed 54
     'VCASP': 93,  # Default 93
     "VCASC": 228,  # Lars proposed 150
     "IDB": 60,  # Default 100
     'ITUNE': 220,  # Default TB 53, 150 for lower THR tuning
     'VCLIP': 255,  # Default 255
     'IDEL': 255,



    # set readout cycle timing as in TB/or as default in Pisa
    'FREEZE_START_CONF': 10,  # Default 1, TB 41
    'READ_START_CONF': 13,  # Default 3, TB 81
    'READ_STOP_CONF': 15,  # Default 5, TB 85
    'LOAD_CONF': 30,  # Default 7, TB 119
    'FREEZE_STOP_CONF': 31,  # Default 8, TB 120
    'STOP_CONF': 31  # Default 8, TB 120}

    # With the delayed FREEZE etc are cutting the occupancy to 50% in first rows 0-132
    # because it takes longer to read everything than the injection period
    # 'FREEZE_START_CONF': 40,  # Default 1, TB 41
    # 'READ_START_CONF': 43,  # Default 3, TB 81
    # 'READ_STOP_CONF': 45,  # Default 5, TB 85
    # 'LOAD_CONF': 110,  # Default 7, TB 119
    # 'FREEZE_STOP_CONF': 111,  # Default 8, TB 120
    # 'STOP_CONF': 111  # Default 8, TB 120
}

registers = ['IBIAS', 'ICASN', 'IDB', 'ITUNE', 'ITHR', 'ICOMP', 'IDEL', 'VRESET', 'VCASP', 'VH', 'VL', 'VCLIP', 'VCASC', 'IRAM']

class AnalogScan(ScanBase):
    scan_id = 'analog_scan'

    def _configure(self, start_column=0, stop_column=512, start_row=0, stop_row=512, load_tdac_from=None, **_):
        self.chip.masks['enable'][:,:] = False
        self.chip.masks['injection'][:,:] = False
        self.chip.masks['enable'][start_column:stop_column, start_row:stop_row] = True
        self.chip.masks['injection'][start_column:stop_column, start_row:stop_row] = True
        self.chip.masks['tdac'][start_column:stop_column, start_row:stop_row] = 0b100
        #self.chip.masks['hitor'][0, 0] = True

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

        # # W8R13 pixels that fire even when disabled
        # # For these ones, we disable the readout of the whole double-column
        # reg_values = [0xffff] * 16
        # for col in [85, 109, 131, 145, 157, 163, 204, 205, 279, 282, 295, 327, 335, 450]:
        # #for col in [511]:
        #     dcol = col // 2
        #     reg_values[dcol//16] &= ~(1 << (dcol % 16))
        # for i, v in enumerate(reg_values):
        #     # EN_RO_CONF
        #     self.chip._write_register(155+i, v)
        #     # EN_BCID_CONF (to disable BCID distribution, use 0 instead of v)
        #     self.chip._write_register(171+i, v)
        #     # EN_RO_RST_CONF
        #     self.chip._write_register(187+i, v)
        #     # EN_FREEZE_CONF
        #     self.chip._write_register(203+i, v)

        self.chip.masks.apply_disable_mask()
        self.chip.masks.update(force=True)
        #self.daq.set_LEMO_MUX('LEMO_MUX_TX0', 1)

        for r in self.register_overrides:
            if r != 'n_injections':
                self.chip.registers[r].write(self.register_overrides[r])
            #print("Write: ", r, " to ", self.register_overrides[r])
            #print('n_injections in Analogscans',n_injections)
        # Enable hitor is in _scan()

        # Enable injection (active high) on all col and all row  // I think the inj is done only on pixel that have mask set to injection before (tested)
        # for i in range(512//16):
        #     scan.chip._write_register(82+i, 0xffff)
        #     scan.chip._write_register(114+i, 0xffff)
            # scan.chip._write_register(82+18, 2**(300%16))
            # scan.chip._write_register(114+0, 0b1100)
        # Enable injection (active high) only on row 509 (the analog pixel) and no column (no matrix pixel)
            #scan.chip._write_register(82+i, 0xffff)  # inj col ?
            #scan.chip._write_register(114+1, 8192) # inj row ?
            #scan.chip._write_register(114+31, 8192)

        # # Enable analog monitoring on HVFE
        # self.chip.registers["EN_PULSE_ANAMON_R"].write(1)
        # Enable analog monitoring on Normal FE
        self.chip.registers["EN_PULSE_ANAMON_L"].write(1)

        # Asked to do this by Lars Schall
        self.chip.registers['ANAMON_SFN_L'].write(0b0001)
        self.chip.registers['ANAMON_SFP_L'].write(0b1000)
        self.chip.registers['ANAMONIN_SFN1_L'].write(0b1000)
        self.chip.registers['ANAMONIN_SFN2_L'].write(0b1000)
        self.chip.registers['ANAMONIN_SFP_L'].write(0b1000)

        self.daq.rx_channels['rx0']['DATA_DELAY'] = 14



    def _scan(self, n_injections=1, inj_pulse_start_delay=1, **_):
        n_injections=self.register_overrides.get("n_injections", 1)
        #print('n_injections in scans',"n_injections")

        with self.readout(scan_param_id=0):
            # Enable HITOR general output (active low)
            self.chip.registers["SEL_PULSE_EXT_CONF"].write(0)
            # Disable HITOR (active high) on all columns, all rows - needed to reset this for next step
            for i in range(512//16):
                self.chip._write_register(18+i, 0)
                self.chip._write_register(50+i, 0)
            # Enable HITOR (active high) on all columns, all rows
            # for i in range(512//16):
            #     self.chip._write_register(18+i, 0xffff)
            #     self.chip._write_register(50+i, 0xffff)
            # Enable HITOR (active high) on col 300 (18+ int 300//16=18+18 , 2**(300%16) and row 2 (50+2//16=50+0, 2**(2%16) )
            # or row x (50+(x//16), 2**(x%16)
            for i in range(512//16):
                # self.chip._write_register(18+i,  0xffff)
                # self.chip._write_register(50+i, 0xffff)
                self.chip._write_register(18+(300//16), 2**(300%16))
                self.chip._write_register(50+(509//16), 2**(509%16))
            # # Enable HITOR (active high) on col 300 (18+ int 300/16=18+18 , 2**(300%16) and row 2 (50+2/16=50+0, 2**(2%16) )
            # self.chip._write_register(18+18, 2**(300%16))
            # self.chip._write_register(50, 2**(2%16))

            self.chip.inject(PulseStartCnfg=inj_pulse_start_delay, PulseStopCnfg=1500, repetitions=1000, latency=1400, reset_bcid=True)

        ret = {}
        for r in registers:
            ret[r] = self.chip.registers[r].read()
        self.scan_registers = ret

        self.log.success('Scan finished')

    def _analyze(self):
        self.hist_occ = 0
        self.hist_tot = 0
        with analysis.Analysis(raw_data_file=self.output_filename + '.h5', **self.configuration['bench']['analysis']) as a:
            a.analyze_data()
            # self.hist_occ = a.hist_occ
            # self.hist_tot = a.hist_tot


if __name__ == "__main__":
    with AnalogScan(scan_config=scan_configuration, register_overrides=register_overrides) as scan:
        scan.start()
