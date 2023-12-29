from matplotlib import pyplot as plt
import numpy as np
from methods.trace import Trace as tr
from methods import leak_correct as lc
import sys
from pathlib import Path
p = Path(__file__).parents[1]
sys.path.insert(0, str(p))


def test_get_QC_dict(QC):
    return lc.get_QC_dict(QC)


def test_plot_leak_fit(currents, QC_filt, well, sweep, ramp_bounds):
    lc.plot_leak_fit(currents, QC_filt, well, sweep, ramp_bounds)


def test_get_leak_correct(trace, currents, QC_filt, ramp_bounds):
    return lc.get_leak_corrected(trace, currents, QC_filt, ramp_bounds)


if __name__ == "__main__":
    filepath = "test/test_files/terfenadine/"
    json_file = "terfenadine_prot_01_12_23_12.13.23"
    trace = tr(filepath, json_file)

    # get currents and QC from trace object
    currents = trace.all_traces(leakcorrect=False)
    currents['times'] = trace.times()
    currents['voltages'] = trace.voltage()
    QC = trace.QC()

    # get filtered QC dict
    QC_filt = test_get_QC_dict(QC)

    # test getting leak corrected data
    leak_corrected = test_get_leak_correct(
        trace, currents, QC_filt, ramp_bounds=[1700, 2500])
    print('All tests passed')

    # test plotting leak fit
    test_plot_leak_fit(currents, QC_filt, well='D15',
                       sweep=9, ramp_bounds=[48300, 49100])
