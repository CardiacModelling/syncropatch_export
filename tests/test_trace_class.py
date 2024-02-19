import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

from syncropatch_export.trace import Trace as tr
from syncropatch_export.voltage_protocols import VoltageProtocol


class TestTraceClass(unittest.TestCase):
    def setUp(self):
        filepath = os.path.join('tests', 'test_data', '13112023_MW2_FF',
                                'staircaseramp (2)_2kHz_15.01.07')
        json_file = "staircaseramp (2)_2kHz_15.01.07"

        self.output_dir = os.path.join('test_output', 'test_trace_class')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.test_trace = tr(filepath, json_file)

    def test_protocol_descriptions(self):
        voltages = self.test_trace.get_voltage()
        times = self.test_trace.get_times()

        protocol_from_json = self.test_trace.get_voltage_protocol()
        protocol_desc = VoltageProtocol.from_voltage_trace(voltages, times)

        sections1 = protocol_from_json.get_all_sections()
        sections2 = protocol_desc.get_all_sections()

        t_error = np.max(np.abs((sections1[:-2, :2] - sections2[:-2, :2])))
        v_error = np.max(np.abs((sections1[:, 2:] - sections2[:, 2:])))

        self.assertLess(t_error, 1e-2)
        self.assertLess(v_error, 1e-4)

    def test_protocol_timeseries(self):
        voltages = self.test_trace.get_voltage()
        times = self.test_trace.get_times()

        protocol_desc = self.test_trace.get_voltage_protocol()

        def voltage_func(t):
            for tstart, tend, vstart, vend in protocol_desc.get_all_sections():
                if t >= tstart and t < tend:
                    if vstart != vend:
                        return vstart + (vend - vstart) * (t - tstart)/(tend - tstart)
                    else:
                        return vstart

            return vend

        for t, v in zip(times, voltages):
            self.assertLess(voltage_func(t) - v, 1e-3)

    def test_get_traces(self):
        tr = self.test_trace
        v = tr.get_voltage()
        ts = tr.get_times()
        all_traces = tr.get_all_traces()

        self.assertTrue(np.all(np.isfinite(v)))
        self.assertTrue(np.all(np.isfinite(ts)))

        for well, trace in all_traces.items():
            self.assertTrue(np.all(np.isfinite(trace)))

        if self.output_dir:
            # plot test output
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.set_title("Example Sweeps")
            some_sweeps = tr.get_trace_sweeps([0])['A01']

            ax1.plot(ts, np.transpose(some_sweeps), color='grey', alpha=0.5)
            ax1.set_ylabel('Current')
            ax1.set_xlabel('Time')
            ax2.set_title("Voltage Protocol")
            ax2.plot(ts, v)
            ax2.set_ylabel('Voltage')
            ax2.set_xlabel('Time')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir,
                                     'example_trace'))
            plt.close(fig)
