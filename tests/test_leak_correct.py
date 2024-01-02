import unittest
from pcpostprocess.trace import Trace as tr
from pcpostprocess import leak_correct
import os


class TestLeakCorrect(unittest.TestCase):
    def setUp(self):
        test_data_dir = os.path.join('tests', 'test_data', '13112023_MW2_FF',
                                     "staircaseramp (2)_2kHz_15.01.07")
        json_file = "staircaseramp (2)_2kHz_15.01.07.json"

        self.output_dir = os.path.join('test_output', 'test_trace_class')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.ramp_bounds = [1700, 2500]
        self.test_trace = tr(test_data_dir, json_file)

        # get currents and QC from trace object
        self.currents = self.test_trace.get_all_traces(leakcorrect=False)
        self.currents['times'] = self.test_trace.get_times()
        self.currents['voltages'] = self.test_trace.get_voltage()
        self.QC = self.test_trace.get_onboard_QC_values()

    def test_get_QC_dict(self):
        QC = self.test_trace.get_onboard_QC_values()
        return leak_correct.get_QC_dict(QC)

    def test_plot_leak_fit(self):
        well = 'A01'
        sweep = 0

        leak_correct.fit_linear_leak(self.test_trace, well, sweep,
                                     self.ramp_bounds, plot=True,
                                     output_dir=self.output_dir)

    def test_get_leak_correct(self):
        trace = self.test_trace
        currents = self.currents
        QC_filt = leak_correct.get_QC_dict(self.QC)

        # test getting leak corrected data
        _ = leak_correct.get_leak_corrected(
            trace, currents, QC_filt, self.ramp_bounds)

        return leak_correct.get_leak_corrected(trace, currents, QC_filt,
                                               self.ramp_bounds)


if __name__ == "__main__":
    pass
