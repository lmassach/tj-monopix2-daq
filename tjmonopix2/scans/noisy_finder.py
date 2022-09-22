#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#
"""Finds noisy pixels by disabling them few at a time, reports them on terminal."""

import time
import numpy as np
from tjmonopix2.system.scan_base import ScanBase
from tjmonopix2.analysis import analysis

scan_configuration = {
    'start_column': 0,
    'stop_column': 448,
    'start_row': 0,
    'stop_row': 512,

    'max_iterations': 50,  # Acquire-and-mask retries
    'acquisition_time': 1,  # Acquisition time per each iteration [s]
    'max_rate': 0.1,  # Above this rate a pixel is marked noisy [Hz]
    'inject': True,  # Enables injecting on the top pixel of each column
}

register_overrides = {
    'ITHR': 40,
    'IBIAS': 50,
    'VRESET': 143,
    'ICASN': 0,
    'VCASP': 93,
    'IDB': 100,
    'MON_EN_IDB': 1,
    'VH': 255,
    'VL': 1
}


def interpret_data_short(raw_data):
    """Copied from TJMonopix2 class, then removed stuff we don't need."""
    hit_dtype = np.dtype([("col", "<u2"), ("row", "<u2")])
    r = raw_data[(raw_data & 0xF8000000) == 0x40000000]
    r0 = (r & 0x7FC0000) >> 18
    r1 = (r & 0x003FE00) >> 9
    r2 = (r & 0x00001FF)
    rx_data = np.reshape(np.vstack((r0, r1, r2)), -1, order="F")
    hit = np.empty(len(rx_data) // 4 + 10, dtype=hit_dtype)
    h_i = 0
    r_i = 0
    idx = 0
    flg = 0
    broken_count = 0
    while idx < len(rx_data):
        if rx_data[idx] == 0x1fc:
            if len(rx_data) > idx + 5 and rx_data[idx + 4] == 0x15c: # reg data
                r_i = r_i + 1
                idx = idx + 5
            else:
                # Broken register data
                broken_count += 1
                idx = idx +1
        elif rx_data[idx] == 0x1bc:  # sof
            idx=idx+1
            if flg!=0:
                # EOF missing
                broken_count += 1
            flg = 1
        elif rx_data[idx] == 0x17c:  # eof
            if flg!=1:
                # EOF before SOF
                broken_count += 1
            flg = 0
            idx = idx + 1
        elif rx_data[idx] == 0x13c: ## idle (dummy data)
            idx = idx + 1
        else:
            if flg != 1:
                # SOF missing
                broken_count += 1
            if len(rx_data) < idx + 4 :
                # Incomplete data at end
                break
            hit[h_i]['row'] = ((rx_data[idx+2] & 0x1) << 8) | (rx_data[idx+3] & 0xFF)
            hit[h_i]['col'] = ((rx_data[idx] & 0xFF) << 1) + ((rx_data[idx+2] & 0x2) >> 1)
            idx = idx+4
            h_i = h_i+1
    hit = hit[:h_i]
    return hit, broken_count


class NoisyPixelScan(ScanBase):
    scan_id = 'noisy_pixel_scan'

    def _configure(self, start_column=0, stop_column=512, start_row=0, stop_row=512, **_):
        self.chip.masks['enable'][start_column:stop_column, start_row:stop_row] = True
        self.chip.masks['injection'][start_column:stop_column, start_row:stop_row] = False
        self.chip.masks['tdac'][start_column:stop_column, start_row:stop_row] = 0b100
        #self.chip.masks['hitor'][0, 0] = True

        self.chip.masks.apply_disable_mask()
        self.chip.masks.update(force=True)

        self.chip.registers["SEL_PULSE_EXT_CONF"].write(0)

        for r in self.register_overrides:
            if r != 'scan_time':
                self.chip.registers[r].write(self.register_overrides[r])
            #print("Write: ", r, " to ", self.register_overrides[r])

        self.daq.rx_channels['rx0']['DATA_DELAY'] = 14

    def _scan(self, max_iterations=50, acquisition_time=1, max_rate=1, inject=True, start_column=0, stop_column=512, start_row=0, **_):
        data = []
        def data_callback(data_tuple):
            nonlocal data
            data.append(data_tuple)
        # Data will contain tuples of (array_of_raw_words_uint32, start_time, stop_time, n_errors)

        N_INJ = 100
        all_noisy_pixels = set()
        for it in range(max_iterations):
            # Acquire data for acquisition_time seconds
            self.log.info('Iteration %d', it)
            with self.readout(callback=data_callback):
                time.sleep(acquisition_time/2)
                if inject:
                    for i in range(min(4, stop_column - start_column)):
                        self.chip.masks['injection'][start_column:stop_column, start_row] = False
                        self.chip.masks['injection'][start_column+i:stop_column:4, start_row] = True
                        self.chip.masks.update(force=True)
                        self.chip.inject(PulseStartCnfg=1, PulseStopCnfg=512, repetitions=N_INJ, latency=1400)
                time.sleep(acquisition_time/2)
            # Decipher the data just acquired
            delta_t = data[-1][2] - data[0][1]  # Seconds
            n_errors = sum(x[3] for x in data)
            raw_data = np.concatenate([x[0] for x in data])
            if n_errors:
                self.log.warning("Got %d readout errors (whatever it means)", n_errors)
            # Decipher raw data
            self.log.info("Got %d raw words", len(raw_data))
            hits, n_broken = interpret_data_short(raw_data)
            if n_broken:
                self.log.warning("Got %d broken hits", n_broken)
            pixels, counts = np.unique(hits, return_counts=True)
            counts[pixels["row"] == start_row] = np.maximum(0, counts[pixels["row"] == start_row] - N_INJ)
            rate = counts.astype(np.float32) / delta_t
            noisy_pixels = pixels[rate > max_rate]
            if len(noisy_pixels) == 0:
                self.log.success('No more noisy pixels found')
                break
            # Disable noisy pixels and list them
            for col, row in noisy_pixels:
                self.chip.masks['enable'][col, row] = False
                if (col, row) in all_noisy_pixels:
                    self.log.warning("Pixel (%d, %d) fired despite being masked", col, row)
                all_noisy_pixels.add((col, row))
            self.chip.masks.update(force=True)
            time.sleep(0.1)
            # Reset for next iteration
            data = []

        self.log.success('Scan finished')
        self.log.success('Found %d noisy pixels', len(all_noisy_pixels))
        print(all_noisy_pixels)

    def _analyze(self):
        pass


if __name__ == "__main__":
    with NoisyPixelScan(scan_config=scan_configuration, register_overrides=register_overrides) as scan:
        scan.start()
