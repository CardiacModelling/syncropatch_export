import sys
from pathlib import Path
p = Path(__file__).parents[1]
sys.path.insert(0, str(p))
from methods.trace import Trace as tr
from matplotlib import pyplot as plt
import numpy as np

def test_voltage(trace):
    return trace.voltage()

def test_times(trace):
    return trace.times()

def test_all_trace(trace):
    return trace.all_traces()

def test_get_trace_sweep(trace, sweep):
    return trace.get_trace_sweep(sweep)

def test_QC(trace):
    return trace.QC()

if __name__ == "__main__":
    filepath = "test/test_files/cisapride/"
    json_file = "cisapride_protocol_11.13.12"
    test_trace = tr(filepath, json_file)

    ### test trace class attributes
    v = test_voltage(test_trace) # get voltage
    ts = test_times(test_trace) # get times  
    all_trace = test_all_trace(test_trace) # get all traces
    some_sweeps = test_get_trace_sweep(test_trace, [0, 1, 2])['A01'] # get some sweeps
    QC = test_QC(test_trace) # get QC values
    print("All test passed")
    
    ### plot test output
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title("Example Sweeps")
    ax1.plot(ts, np.transpose(some_sweeps), color = 'grey', alpha = 0.5)
    ax1.set_ylabel('Current')
    ax1.set_xlabel('Time')
    ax2.set_title("Voltage Protocol")
    ax2.plot(ts, v)
    ax2.set_ylabel('Voltage')  
    ax2.set_xlabel('Time')  
    plt.tight_layout()
    plt.show()

