#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

import time

import numpy as np
import tables as tb
from tqdm import tqdm

from tjmonopix2.system.scan_base import ScanBase
from tjmonopix2.analysis import analysis

scan_configuration = {
    'start_column': 224,
    'stop_column': 448,
    'start_row': 0,
    'stop_row': 512,

    'scan_time': 10,  # seconds
    #'load_tdac_from': None,  # Optional h5 file to load the TDAC values from
    #'load_tdac_from': "/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0_2022-10-27/chip_0/20221027_175016_local_threshold_tuning_interpreted.h5",  # Optional h5 file to load the TDAC values from
    # file produced w/o BCID target THR=20 DAC and pwell/psub=-6V
    #'load_tdac_from': "/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0_2022-11-02/chip_0/20221102_125629_local_threshold_tuning.h5",  # Optional h5 file to load the TDAC values from
    # File produced w/o BCID reset target=20 DAC psub/pwell=-6V cols=213-223 rows=120-220
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0/chip_0/20221111_162731_local_threshold_tuning_interpreted.h5'
    # File produced w/o BCID reset target=20 DAC psub/pwell=-6V cols=180-223 rows=120-220
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0/chip_0/20221112_121058_local_threshold_tuning_interpreted.h5'
    # File produced w/o BCID reset target=20 DAC psub/pwell=-6V cols=213-223 rows=120-220
    # 'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0_2022-11-15/chip_0/20221115_091130_local_threshold_tuning_interpreted.h5'
    # File produced w/o BCID reset target=20 DAC psub/pwell=-6V cols=180-223 rows=150-500
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0_2022-11-17/chip_0/20221117_184249_local_threshold_tuning_interpreted.h5'
    # File produced w BCID reset target=21 DAC psub/pwell=-3V cols=0-223 rows=0-511 ITUNE=175
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0_2022-12-11/chip_0/20221211_113327_local_threshold_tuning_interpreted.h5'
    # File produced w BCID reset target=21 DAC psub/pwell=-3V cols=0-223 rows=0-511 ITUNE=175 redone disabling bad col 192-223
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0_2022-12-13/chip_0/20221213_152510_local_threshold_tuning_interpreted.h5'
    # chipW15R12 File produced w BCID reset target=31 DAC psub ground pwell=-3V cols=0-223 rows=0-511
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0_2022-12-14/chip_0/20221214_114000_local_threshold_tuning.h5'
    # chipW15R12 File produced w BCID reset target=34 DAC psub ground pwell=-3V cols=0-224 rows=0-512
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0/chip_0/20221229_133824_local_threshold_tuning.h5'
    #chipW8r13 File produced w BCID reset target=22 DAC psub/pwell=-6V cols=0-224 rows=0-512
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0/chip_0/20230102_172341_local_threshold_tuning_interpreted.h5'
    #chipW8r13 File produced w BCID reset target=25 DAC psub/pwell=-6V cols=0-224 rows=0-512
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0_2023-01-07/chip_0/20230107_105116_local_threshold_tuning_interpreted.h5'
    #chipW8r13 File produced w BCID reset target=25 DAC psub/pwell=-6V cols=224-447 rows=0-512
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0/chip_0/20230108_081926_local_threshold_tuning_interpreted.h5'
    #chipW8r13 File produced w BCID reset target=25 DAC psub/pwell=-6V cols=0-447 rows=0-512
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0_2023-01-08/chip_0/20230108_084232_local_threshold_tuning_interpreted.h5'
    # chipW8R13 File produced w BCID reset target=32 VCASC=228 DAC psub pwell=-6V cols=0-447 rows=150-512
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0_2023-01-15/chip_0/20230115_183553_local_threshold_tuning_interpreted.h5'
    # chipW8R13 File produced w BCID reset target=22 VCASC=228 ICASN=54 DAC psub pwell=-6V cols=0-447 rows=150-512
    # 'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0/chip_0/20230116_130625_local_threshold_tuning_interpreted.h5'
    # chipW8R13 File produced w BCID reset target=40 TB settings psub pwell=-6V cols=224-447 rows=150-512
    # 'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0/chip_0/20230227_123341_local_threshold_tuning_interpreted.h5'
    # chipW8R13 File produced w BCID reset target=40 Lars setting but ICASN=0 settings psub pwell=-6V cols=224-448 rows=150-512
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0/chip_0/20230227_171449_local_threshold_tuning_interpreted.h5'
    # chipW8R13 File produced w BCID reset target=24 Lars setting but ICASN=150 settings psub pwell=-6V cols=224-448 rows=150-512
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0/chip_0/20230227_181812_local_threshold_tuning_interpreted.h5'
    # chipW8R13 File produced w BCID reset target=43 TB settings psub pwell=-6V cols=225-230 rows=150-512
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0/chip_0/20230301_153124_local_threshold_tuning_interpreted.h5'
    # chipW8R13 File produced w BCID reset target=23 Lars setting psub pwell=-6V cols=225-448 rows=150-512
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0/chip_0/20230306_115601_local_threshold_tuning_interpreted.h5'
    # chipW8R13 File produced w BCID reset target=25 ITHR=20 ICASN=0 settings psub pwell=-6V cols=224-448 rows=0-512
    #'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0/chip_0/20230325_153148_local_threshold_tuning_interpreted.h5'
     # chipW8R13 File produced w BCID reset target=25 ITHR=64 ICASN=80 settings psub pwell=-6V cols=224-448 rows=0-512
    'load_tdac_from': '/home/labb2/tj-monopix2-daq/tjmonopix2/scans/output_data/module_0/chip_0/20230325_182214_local_threshold_tuning_interpreted.h5'
 }

register_overrides = {
    # 'ITHR':10,  # Default 64
    # 'IBIAS': 50,  # Default 50
    # 'VRESET': 110,  # Default 143, 110 for lower THR
    # 'ICASN': 0,  # Default TB 0 , 150 for -3V , 200 for -6V
    # 'VCASP': 93,  # Default 93
    # "VCASC": 228,  # Default 228
    # "IDB": 100,  # Default 100
    # 'ITUNE': 220,  # Default TB 53, 150 for lower THR tuning
    # 'VCLIP': 255,  # Default 255

    #'ITHR':30,  # Default 64
    #'IBIAS': 50,  # Default 50
    #'VRESET': 110,  # Default 143, 110 for lower THR
    #'ICASN': 0,  # Default TB 0 , 150 for -3V , 200 for -6V
    #'VCASP': 93,  # Default 93
    #"VCASC": 228,  # Default 228, lars proposed 150
    #"IDB": 100,  # Default 100
    #'ITUNE': 175,  # Default TB 53, 150 for lower THR tuning
    #'VCLIP': 255,  # Default 255

# Lars proposed tuning with target ~ 23 but in this chip seems ITHR=30
    #  'ITHR':64,  # Default 64
    #  'IBIAS': 100,  # Default 50
    #  'VRESET': 128,  # Default TB 143, 110 for lower THR, Lars dec proposal 128
    #  'ICASN': 54,  # Lars proposed 54
    #  'VCASP': 93,  # Default 93
    #  "VCASC": 150,  # Lars proposed 150
    #  "IDB": 100,  # Default 100
    #  'ITUNE': 175,  # Default TB 53, 150 for lower THR tuning
    #  'VCLIP': 255,  # Default 255
     'ITHR':64,  # Default 64
     'IBIAS': 50,  # Default 50
     'VRESET': 110,  # Default TB 143, 110 for lower THR, Lars dec proposal 128
     'ICASN': 80,  # Lars proposed 54
     'VCASP': 93,  # Default 93
     "VCASC": 228,  # Lars proposed 150
     "IDB": 50,  # Default 100
     'ITUNE': 220,  # Default TB 53, 150 for lower THR tuning
     'VCLIP': 255,  # Default 255


    # HV  settings
    # 'ITHR':30,  # Default 64
    # 'IBIAS': 60,  # Default 50
    # 'VRESET': 50,  # Default 143, 110 for lower THR
    # 'ICASN': 0,  # Default TB 0 , 150 for -3V , 200 for -6V
    # 'VCASP': 40,  # Default 93
    # "VCASC": 228,  # Default 228
    # "IDB": 150,  # Default 100
    # 'ITUNE': 150,  # Default TB 53, 150 for lower THR tuning
    # 'VCLIP': 255,  # Default 255

    # 'ITHR':30,  # Default 64
    # 'IBIAS': 60,  # Default 50
    # 'VRESET': 100,  # Default 143, 110 for lower THR
    # 'ICASN': 8,  # Default TB 0 , 150 for -3V , 200 for -6V
    # 'VCASP': 40,  # Default 93
    # "VCASC": 228,  # Default 228
    # "IDB": 100,  # Default 100
    # 'ITUNE': 53,  # Default TB 53, 150 for lower THR tuning
    # 'VCLIP': 255,  # Default 255

    # TB settings
    # 'ITHR':64,  # Default 64
    # 'IBIAS': 50,  # Default 50
    # 'VRESET': 143,  # Default 143, 110 for lower THR
    # 'ICASN': 0,  # Default TB 0 , 150 for -3V , 200 for -6V
    # 'VCASP': 93,  # Default 93
    # "VCASC": 228,  # Default 228
    # "IDB": 100,  # Default 100
    # 'ITUNE': 53,  # Default TB 53, 150 for lower THR tuning
    # 'VCLIP': 255,  # Default 255

    # set readout cycle timing as in TB/or as default in Pisa
    #'FREEZE_START_CONF': 10,  # Default 1, TB 41
    #'READ_START_CONF': 13,  # Default 3, TB 81
    #'READ_STOP_CONF': 15,  # Default 5, TB 85
    #'LOAD_CONF': 30,  # Default 7, TB 119
    #'FREEZE_STOP_CONF': 31,  # Default 8, TB 120
    #'STOP_CONF': 31,  # Default 8, TB 120

    # set readout cycle timing as in TB/or as default in Pisa
    'FREEZE_START_CONF': 10,  # Default 1, TB 41
    'READ_START_CONF': 13,  # Default 3, TB 81
    'READ_STOP_CONF': 15,  # Default 5, TB 85
    'LOAD_CONF': 30,  # Default 7, TB 119
    'FREEZE_STOP_CONF': 31,  # Default 8, TB 120
    'STOP_CONF': 31,  # Default 8, TB 120
    # 'FREEZE_START_CONF': 60,  # Default 1, TB 41
    # 'READ_START_CONF': 63,  # Default 3, TB 81
    # 'READ_STOP_CONF': 65,  # Default 5, TB 85
    # 'LOAD_CONF': 110,  # Default 7, TB 119
    # 'FREEZE_STOP_CONF': 111,  # Default 8, TB 120
    # 'STOP_CONF': 111,  # Default 8, TB 120

   # Enable analog monitoring pixel
    'EN_PULSE_ANAMON_L': 1,
    'ANAMON_SFN_L': 0b0001,
    'ANAMON_SFP_L': 0b1000,
    'ANAMONIN_SFN1_L': 0b1000,
    'ANAMONIN_SFN2_L': 0b1000,
    'ANAMONIN_SFP_L': 0b1000,
    # Enable hitor Enable HITOR general output (active low)
    'SEL_PULSE_EXT_CONF': 0,
}

registers = ['IBIAS', 'ICASN', 'IDB', 'ITUNE', 'ITHR', 'ICOMP', 'IDEL', 'VRESET', 'VCASP', 'VH', 'VL', 'VCLIP', 'VCASC', 'IRAM']


class SourceScan(ScanBase):
    scan_id = 'source_scan'

    def _configure(self, start_column=0, stop_column=512, start_row=0, stop_row=512, load_tdac_from=None, **_):
        self.chip.masks['enable'][:,:] = False
        self.chip.masks['injection'][:,:] = False
        self.chip.masks['enable'][start_column:stop_column, start_row:stop_row] = True
        self.chip.masks['tdac'][start_column:stop_column, start_row:stop_row] = 0b100

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
        self.chip.masks['enable'][214,88] = False
        self.chip.masks['enable'][215,101] = False
        self.chip.masks['enable'][191:223,:] = False  # cols 191-223 are broken since Nov/dec very low THR


        # # Noisy/hot W8R13 pixels
        # for col, row in [(219, 161), (222, 188), (219, 192), (219, 129), (221, 125), (219, 190), (220, 205), (220, 144), (220, 168), (219, 179), (221, 136), (222, 186), (219, 163), (221, 205), (226, 135), (222, 174), (221, 199), (222, 185), (221, 203), (225, 181), (220, 123), (222, 142), (223, 143), (220, 154), (221, 149), (221, 179), (222, 120), (219, 125)] \
        #         + [(227, 167), (222, 177), (221, 156), (221, 196), (219, 170), (221, 150), (222, 172), (221, 132), (219, 133), (226, 146), (220, 187), (219, 200), (227, 142), (220, 173), (229, 142), (219, 131), (219, 195), (221, 126), (220, 136), (222, 159), (221, 188), (227, 207), (219, 167), (219, 193), (220, 131), (228, 158), (219, 198), (221, 139), (219, 211), (221, 177), (223, 189), (219, 156), (222, 144), (222, 184), (228, 200), (219, 174), (229, 197), (220, 184), (222, 133), (222, 171), (219, 152), (222, 147), (222, 197), (219, 132), (223, 167), (221, 163), (223, 126), (227, 199), (229, 168), (229, 188), (221, 165), (222, 139), (219, 217), (222, 161), (225, 143), (219, 187), (221, 202), (223, 141), (222, 134), (220, 204), (221, 209), (220, 191), (221, 183), (223, 137), (223, 188), (221, 219), (222, 140), (223, 146), (219, 142), (222, 219), (229, 134), (221, 168), (227, 145), (222, 199), (227, 211), (220, 201), (220, 217), (221, 190), (221, 216), (223, 204), (221, 186), (227, 178), (222, 125), (221, 122), (220, 190), (227, 196), (222, 141), (220, 145), (229, 215)] \
        #         + [(221, 174), (219, 188)] \
        #         + [(228, 203), (219, 180), (227, 137), (220, 146), (219, 146), (221, 137), (221, 172)] \
        #         + [(228, 181), (221, 211), (228, 201), (221, 123), (229, 173), (220, 143), (228, 219), (220, 135), (226, 141), (222, 217), (228, 188), (228, 156), (222, 122), (229, 191), (222, 160), (225, 146), (227, 155), (220, 179), (220, 175), (219, 137), (220, 149), (227, 171), (226, 178), (221, 200), (220, 198), (222, 181), (221, 198), (228, 216), (227, 174), (228, 133), (219, 182), (229, 149), (219, 136), (229, 163), (223, 148), (221, 120), (221, 160), (222, 213), (219, 175), (225, 203), (222, 165), (221, 170), (225, 186), (228, 190), (221, 143), (219, 194), (222, 194), (219, 124), (226, 133), (223, 199), (221, 192), (219, 205), (219, 139), (225, 207), (225, 197), (221, 169), (219, 153), (219, 186), (221, 148), (221, 131), (219, 173), (228, 169), (228, 150), (219, 191), (221, 129), (222, 193), (222, 196), (220, 186), (220, 148), (221, 189), (221, 134), (219, 135), (220, 209), (221, 159), (220, 182), (220, 169), (222, 215), (221, 171), (220, 121), (228, 209), (219, 123), (220, 153), (222, 201), (220, 210), (221, 151), (219, 207), (222, 162), (227, 132), (220, 203), (228, 198), (228, 144), (221, 140), (222, 192), (223, 217), (219, 196), (226, 199), (223, 177), (225, 193), (219, 181), (222, 178), (219, 144), (221, 164), (219, 171), (219, 201), (220, 125), (219, 130), (222, 207)] \
        #         :
        #     self.chip.masks['enable'][col,row] = False

        # # Noisy/hot W8R13 pixels from Jan 7-8 source scan with Fe55
        for col, row in [(7,126), (16,75), (10,362), (30, 34), (12, 453), (6, 348), (6, 348), (20, 404), (11, 379), (30, 271)] \
               + [(30, 164), (24, 411),  (24, 65),  (10, 65),  (18, 341), (10, 290), (10, 176), (12, 329), (28, 438), (26, 439), (30, 155)] \
               + [(191, 5),(26, 437),(9, 381),(20, 239),(26, 429),(28, 280),(22, 273),(18, 260),(18, 323),(16, 464),(30, 227)] \
               + [(27, 189),(18, 292),(28, 237), (18, 231), (8, 511), (28, 409), (20, 314), (17, 294), (12, 151), (18, 332), (7, 146), (26, 479)]\
               + [(30, 285) , (4, 205) , (2, 145) , (28, 246) , (28, 89) , (6, 370) , (10, 441), (8,269), (20,208),(18,295), (8,296), (26,199)  ] \
               + [(240, 181), (241, 412), (240, 390), (240, 305)] \
               + [(358, 412), (303, 37), (229, 282)] \
               + [(419, 143), (425, 193), (450,0), (0,0)] \
                :
             self.chip.masks['enable'][col,row] = False

        # # Noisy/hot W5R24 pixels
        #for col, row in [ (240, 228) (239,451), (240,233)] \
        #         + [(239,188)] \
        #         :
        #     self.chip.masks['enable'][col,row] = False

        # W8R13 pixels that fire even when disabled
        # For these ones, we disable the readout of the whole double-column
        reg_values = [0xffff] * 16
        for col in [85, 109, 131, 145, 157, 163, 204, 205, 279, 282, 295, 327, 335, 450]: #on chip w8r24
        #for col in [511]:
            dcol = col // 2
            reg_values[dcol//16] &= ~(1 << (dcol % 16))
        for i, v in enumerate(reg_values):
            # EN_RO_CONF
            self.chip._write_register(155+i, v)
            # EN_BCID_CONF (to disable BCID distribution, use 0 instead of v)
            # EN_BCID_CONF
            self.chip._write_register(171+i, v)
            # EN_RO_RST_CONF
            self.chip._write_register(187+i, v)
            # EN_FREEZE_CONF
            self.chip._write_register(203+i, v)

        self.chip.masks.apply_disable_mask()
        self.chip.masks.update(force=True)

        self.chip.registers["SEL_PULSE_EXT_CONF"].write(0)

        self.scan_registers = {}
        for r in self.register_overrides:
            if r != 'scan_time':
                self.chip.registers[r].write(self.register_overrides[r])
                self.scan_registers[r] = self.register_overrides[r]

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
                self.update_pbar_with_word_rate(pbar)
        pbar.close()

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
