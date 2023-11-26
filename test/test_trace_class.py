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

if __name__ == "__main__":
    filepath = "test/test_files/"
    json_file = "cisapride_protocol_11.13.12.json"
    test_trace = tr(filepath, json_file)
    
    v = test_voltage(test_trace)
    ts = test_times(test_trace)   
    all_trace = test_all_trace(test_trace)
    some_sweeps = test_get_trace_sweep(test_trace, [0, 1, 2])['A01']
    print("All test passed")

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title("Example Sweeps")
    ax1.plot(ts, np.transpose(some_sweeps), color = 'grey', alpha = 0.5)
    ax2.set_title("Voltage Protocol")
    ax2.plot(ts, v)
    ax1.set_ylabel('Current')
    ax1.set_xlabel('Time')
    ax2.set_ylabel('Voltage')  
    ax2.set_xlabel('Time')  
    plt.tight_layout()
    plt.show()

