#!/usr/bin/env python
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from syncropatch_export.trace import Trace
from syncropatch_export.voltage_protocols import VoltageProtocol


class TestTraceClass(unittest.TestCase):
    """
    Tests both the Trace and VoltageProtocol classes.
    """

    def setUp(self):
        f = 'staircaseramp (2)_2kHz_15.01.07'
        self.trace = Trace(
            os.path.join('tests', 'test_data', '13112023_MW2_FF', f), f)

    def test_protocol_descriptions(self):
        voltages = self.trace.get_voltage()
        times = self.trace.get_times()

        protocol_from_json = self.trace.get_voltage_protocol()
        holding_potential = protocol_from_json.get_holding_potential()
        protocol_desc = VoltageProtocol.from_voltage_trace(voltages, times,
                                                           holding_potential)

        sections1 = protocol_from_json.get_all_sections()
        sections2 = protocol_desc.get_all_sections()

        v_error = np.max(np.abs((sections1 - sections2))[:, 2:])

        # There may be some extra time on the end of the protocol in the json
        t_error = np.max(np.abs((sections1 - sections2))[:-1, :2])

        self.assertLess(t_error, 1e-2)
        self.assertLess(v_error, 1e-4)

    def test_get_protocol_description(self):
        a = np.array(self.trace.get_protocol_description())
        b = np.array(self.trace.get_voltage_protocol().get_all_sections())
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.all(a == b))

    def test_protocol_export(self):
        with tempfile.TemporaryDirectory() as d:
            protocol = self.trace.get_voltage_protocol()
            protocol.export_txt(os.path.join(d, 'protocol.txt'))
            json_protocol = self.trace.get_voltage_protocol_json()
            with open(os.path.join(d, 'protocol.json'), 'w') as fin:
                json.dump(json_protocol, fin)

    def test_protocol_timeseries(self):
        voltages = self.trace.get_voltage()
        times = self.trace.get_times()
        voltage_protocol = self.trace.get_voltage_protocol()

        def voltage_func(t):
            for tstart, tend, vstart, vend in voltage_protocol.get_all_sections():
                if t >= tstart and t < tend:
                    if vstart != vend:
                        return vstart + (vend - vstart) * (t - tstart) / (tend - tstart)
                    else:
                        return vstart
            return voltage_protocol.get_holding_potential()  # pragma: no-cover

        for t, v in zip(times, voltages):
            self.assertLess(voltage_func(t) - v, 1e-3)

    def test_protocol_get_step_start_times(self):
        a = list(self.trace.get_voltage_protocol().get_step_start_times())
        b = [0, 250, 300, 696, 896, 1896, 2396, 3396, 3896, 4396, 4896, 5396,
             5896, 6396, 6896, 7396, 7896, 8396, 8896, 9396, 9896, 10396,
             10896, 11396, 11896, 12396, 12896, 13896, 14396, 14406, 14502,
             14892]
        self.assertEqual(a, b)

    def test_protocol_get_ramps(self):
        a = np.array(self.trace.get_voltage_protocol().get_ramps())
        b = np.array([[300, 696, -120, -80], [14406, 14502, -70, -110]])
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.all(a == b))

    def test_get_QC(self):
        QC_values = self.trace.get_onboard_QC_values()
        self.assertGreater(len(QC_values), 0)
        df = self.trace.get_onboard_QC_df()

        self.assertGreater(df.shape[0], 0)
        self.assertGreater(df.shape[1], 0)

    def test_get_traces(self):
        v = self.trace.get_voltage()
        ts = self.trace.get_times()
        all_traces = self.trace.get_all_traces(leakcorrect=True)
        all_traces = self.trace.get_all_traces()
        # TODO: Check the output, numerically, by comparing a few points

        self.assertTrue(np.all(np.isfinite(v)))
        self.assertTrue(np.all(np.isfinite(ts)))

        for well, trace in all_traces.items():
            self.assertTrue(np.all(np.isfinite(trace)))

        # Test complex sweep selection
        a = self.trace.get_trace_sweeps([-1, -2])
        b = self.trace.get_trace_sweeps([1, 0])
        self.assertEqual(len(a), len(b))
        self.assertTrue(np.all(a['A01'] == b['A01']))

        # Test asking for non-existent sweeps
        self.assertRaisesRegex(ValueError, 'Invalid sweep selection',
                               self.trace.get_trace_sweeps, [2])
        self.assertRaisesRegex(ValueError, 'Invalid sweep selection',
                               self.trace.get_trace_sweeps, [-3])

        '''
        # plot test output
        if False:
            d = 'test_output'
            if not os.path.exists(d):
                os.makedirs(d)

            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.set_title('Example Sweeps')
            some_sweeps = self.trace.get_trace_sweeps([0])['A01']

            ax1.plot(ts, np.transpose(some_sweeps), color='grey', alpha=0.5)
            ax1.set_ylabel('Current')
            ax1.set_xlabel('Time')
            ax2.set_title('Voltage Protocol')
            ax2.plot(ts, v)
            ax2.set_ylabel('Voltage')
            ax2.set_xlabel('Time')
            plt.tight_layout()
            plt.savefig(os.path.join(d, 'example_trace'))
            plt.close(fig)
        '''

    def test_qc_df(self):
        dfs = [self.trace.get_onboard_QC_df(sweeps=[0]),
               self.trace.get_onboard_QC_df(sweeps=None)]
        for res in dfs:
            # Check res is a pd.DataFrame
            self.assertIsInstance(res, pd.DataFrame)

            # Check it contains data (number of rows>0)
            self.assertGreater(res.shape[0], 0)

            # Check it contains all quality control parameters
            for qcParam in ['Rseal', 'Cm', 'Rseries', 'well', 'sweep']:
                self.assertIn(qcParam, res)

        # Check restricting number of sweeps returns less data
        self.assertLess(dfs[0].shape[0], dfs[1].shape[0])


if __name__ == '__main__':
    unittest.main()
