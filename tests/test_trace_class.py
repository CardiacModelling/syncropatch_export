import numpy as np
import unittest
import os
from matplotlib import pyplot as plt
from methods.trace import Trace as tr
import sys
from pathlib import Path
p = Path(__file__).parents[1]
sys.path.insert(0, str(p))


class TestTraceClass(unittest.TestCase):
    def setUp(self):
        filepath = os.path.join('tests', 'test_files', 'cisapride')
        json_file = "cisapride_protocol_11.13.12"

        self.output_dir = os.path.join('test_output', 'test_trace_class')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.test_trace = tr(filepath, json_file)

    def test_get_traces(self):
        tr = self.test_trace
        v = tr.get_voltage()
        ts = tr.get_times()
        all_traces = tr.get_all_traces()

        self.assertTrue(np.all(np.isfinite(v)))
        self.assertTrue(np.all(np.isfinite(ts)))

        for trace in all_traces:
            self.assertTrue(np.all(np.isfinite(trace)))

        if self.output_dir:
            # plot test output
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.set_title("Example Sweeps")
            some_sweeps = tr.get_trace_sweep(self.test_trace,
                                             [0, 1, 2])['A01']

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
            fig.close()

    def get_voltage(self, trace):
        return trace.voltage()

    def test_times(self, trace):
        return trace.times()

    def test_all_trace(self, trace):
        return trace.all_traces()

    def test_get_trace_sweep(self, trace, sweep):
        return trace.get_trace_sweep(sweep)

    def test_QC(self, trace):
        return trace.QC()

