import numpy as np


class VoltageProtocol():
    def __init__(self, voltage_trace, times, holding_potential=-80.0):

        threshold = 1e-3

        # Convert to mV
        voltage_trace = voltage_trace * 1e3
        self.voltage_trace = voltage_trace

        # convert to ms
        times = times * 1e3
        self.times = times

        # Find gradient changes
        diff2 = np.abs(np.diff(voltage_trace, n=2))

        windows = np.argwhere(diff2 > threshold).flatten()
        window_locs = np.unique(windows)
        window_locs = np.array([val for val in window_locs if val + 1
                                not in window_locs]) + 1

        windows = zip([0] + list(window_locs), list(window_locs)
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
        self._desc = np.vstack(lst)

    def get_step_start_times(self):
        return [line[0] for line in self._desc]

    def get_ramps(self):
        return [line for line in self._desc if line[2] != line[3]]

    def get_all_sections(self):
        """ Return a np.array describing the protocol.

        returns: an np.array where the ith row is the start-time,
        end-time, start-voltage and end-voltage for the ith section of the protocol

        """
        return np.array(self._desc)
