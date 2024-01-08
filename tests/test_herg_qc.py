import logging
import os
import string
import unittest

import numpy as np

from pcpostprocess.hergQC import hERGQC
from pcpostprocess.trace import Trace


class TestHergQC(unittest.TestCase):
    def setUp(self):
        filepath = os.path.join('tests', 'test_data', '13112023_MW2_FF',
                                'staircaseramp (2)_2kHz_15.01.07')

        self.all_wells = [
            lab + str(i).zfill(2) for lab in string.ascii_uppercase[:16]
            for i in range(1, 25)]

        filepath2 = os.path.join('tests', 'test_data', '13112023_MW2_FF',
                                 'staircaseramp (2)_2kHz_15.11.33')

        json_file = "staircaseramp (2)_2kHz_15.01.07"
        json_file2 = "staircaseramp (2)_2kHz_15.11.33"

        self.output_dir = os.path.join('test_output', 'test_herg_qc')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.test_trace_before = Trace(filepath, json_file)
        self.test_trace_after = Trace(filepath2, json_file2)

    def test_run_qc(self):
        tr_before = self.test_trace_before
        tr_after = self.test_trace_after

        v = tr_before.get_voltage()

        ts = tr_after.get_times()

        self.assertTrue(np.all(np.isfinite(v)))
        self.assertTrue(np.all(np.isfinite(ts)))

        # Calculate sampling rate in (use kHz)
        sampling_rate = int(1.0 / (ts[1] - ts[0]))

        plot_dir = os.path.join(self.output_dir,
                                'test_run_qc')

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        hergqc = hERGQC(sampling_rate=sampling_rate,
                        plot_dir=plot_dir,
                        voltage=v)

        sweeps = [0, 1]
        before = tr_before.get_trace_sweeps(sweeps)
        after = tr_after.get_trace_sweeps(sweeps)
        qc_vals_before = tr_before.get_onboard_QC_values(sweeps=sweeps)
        qc_vals_after = tr_before.get_onboard_QC_values(sweeps=sweeps)

        res = {}

        # Spot check a few wells
        # We could check all of the wells but it's time consuming

        test_wells = ['A01', 'A02', 'A03', 'A04', 'A05', 'D01']

        for well in test_wells:
            # Take values from the first sweep only
            qc_vals_before_well = np.array(qc_vals_before[well])[0, :]
            qc_vals_after_well = np.array(qc_vals_after[well])[0, :]

            before_well = np.array(before[well])
            after_well = np.array(after[well])

            passed, qcs = hergqc.run_qc(before_well, after_well,
                                        qc_vals_before_well,
                                        qc_vals_after_well,
                                        n_sweeps=2)

            logging.debug(well, passed)
            res[well] = passed

        self.assertTrue(res['A01'])
        self.assertTrue(res['A02'])
        self.assertTrue(res['A03'])
        self.assertTrue(res['A04'])
        self.assertFalse(res['A05'])
        self.assertFalse(res['D01'])
