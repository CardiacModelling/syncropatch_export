import sys
from pathlib import Path
p = Path(__file__).parents[1]
sys.path.insert(0, str(p))
from methods.trace import Trace as tr
from matplotlib import pyplot as plt

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
    #plt.plot(ts, v)
    #plt.show()    

    #all_t = test_all_trace(test_trace)
    A01_sweep1 = test_get_trace_sweep(test_trace, [0])['A01'][0]
    
    #plt.plot(ts, test_get_trace_sweep(test_trace, [0])['A01'][0])
    #plt.show()


