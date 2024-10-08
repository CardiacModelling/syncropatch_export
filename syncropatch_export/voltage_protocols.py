import numpy as np


class VoltageProtocol():
    def from_json(json_protocol, holding_potential):
        """ Converts a protocol (from the json file) into a np.array

        """

        output_sections = []
        for section in json_protocol:
            tstart = float(section['SegmentStart_ms'])
            tdur = float(section['Duration ms'])
            vstart = float(section['VoltageStart'])
            vend = float(section['VoltageEnd'])

            output_sections.append(np.array((tstart, tstart + tdur,
                                             vstart, vend)))

        return VoltageProtocol(np.array(output_sections),
                               holding_potential=holding_potential)

    def from_voltage_trace(voltage_trace, times, holding_potential=-80.0):
        threshold = 1e-3

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
            end_t = times[end]

            ramp = voltage_trace[end - 1] != voltage_trace[start]

            v_start = voltage_trace[start]

            if ramp:
                grad = (voltage_trace[end - 1] - voltage_trace[start]) / \
                    (times[end - 1] - times[start])
                v_end = v_start + grad * (end_t - start_t)
            else:
                v_end = voltage_trace[end - 1]

            lst.append(np.array([start_t, end_t, v_start, v_end]))

        desc = np.vstack(lst)
        return VoltageProtocol(desc, holding_potential)

    def __init__(self, desc, holding_potential):
        self._desc = desc
        self.holding_potential = holding_potential

    def get_holding_potential(self):
        return self.holding_potential

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

    def export_txt(self, fname):
        output_lines = ['Type \t Voltage \t Duration']

        desc = self.get_all_sections()

        for (tstart, tend, vstart, vend) in desc:
            dur = tend - tstart

            if vstart == vend:
                _type = 'Set'
            else:
                _type = 'Ramp'

            if round:
                vend = np.round(vend)

            output_lines.append(f"{_type}\t{vend}\t{dur}")

        with open(fname, 'w') as fout:
            for line in output_lines:
                fout.write(line)
                fout.write('\n')

