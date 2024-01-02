import numpy as np


class VoltageProtocol():
    def __init__(self, voltage_trace, times, holding_potential=-80.0):

        threshold = 1e-5

        self.voltage_trace = voltage_trace
        self.times = times

        # Find gradient changes
        diff2 = np.abs(np.diff(voltage_trace, n=2))
        # diff1 = np.abs(np.diff(voltages, n=1))

        windows = np.argwhere(diff2 > threshold).flatten()
        window_locs = np.unique(windows)
        window_locs = np.array([val for val in window_locs if val + 1\
                                not in window_locs]) + 1

        windows = zip([0] + list(window_locs), list(window_locs) \
                      + [len(voltage_trace) - 1])

        lst = []
        for start, end in windows:
            start_t = times[start]
            end_t = times[end-1]

            v_start = voltage_trace[start]
            v_end = voltage_trace[end-1]

            lst.append(np.array([start_t, end_t, v_start, v_end]))

        lst.append(np.array([end_t, np.inf, holding_potential,
                             holding_potential]))
        self._desc = lst

    def get_step_start_times(self):
        return [line[0] for line in self._desc]

    def get_ramps(self):
        return [line for line in self._desc if line[2] != line[3]]
