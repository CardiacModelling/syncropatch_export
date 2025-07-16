import numpy as np


class VoltageProtocol:
    """
    Represent a voltage step and ramp protocol.

    Each protocol is represented as

    1. A list of segment starts, ends, initial voltages, and final voltages
    2. A holding potential

    To create a :class:`VoltageProtocol`, use either
    :meth:`VoltageProtocol.from_json` or
    `meth:`VoltageProtocol.from_voltage_trace`.
    """

    def __init__(self, desc, holding_potential, copy_data=True):
        self._desc = np.copy(desc) if copy_data else desc
        self.holding_potential = holding_potential

    @classmethod
    def from_json(cls, json_protocol, holding_potential):
        """
        Reads a protocol from a JSON file.

        Args:
            json_protocol (list): A list or other sequence containing the
                ``VoltageProtocol`` section from the JSON file.
            holding_potential (float): The holding potential

        """
        output_sections = []
        for section in json_protocol:
            tstart = float(section['SegmentStart_ms'])
            tdur = float(section['Duration ms'])
            vstart = float(section['VoltageStart'])
            vend = float(section['VoltageEnd'])
            output_sections.append((tstart, tstart + tdur, vstart, vend))
        return cls(np.array(output_sections), holding_potential, False)

    @classmethod
    def from_voltage_trace(cls, voltage_trace, times, holding_potential=-80.0):
        """
        Creates an approximate voltage protocol from a time series ``(times,
        voltage_trace)``.
        """
        threshold = 1e-3

        # Find gradient changes
        diff2 = np.abs(np.diff(voltage_trace, n=2))
        window_locs = np.unique(np.argwhere(diff2 > threshold).flatten())
        window_locs = 1 + np.array([
            val for val in window_locs if val + 1 not in window_locs])
        windows = zip([0] + list(window_locs),
                      list(window_locs) + [len(voltage_trace) - 1])

        lst = []
        for start, end in windows:
            start_t = times[start]
            end_t = times[end]
            v_start = voltage_trace[start]
            if voltage_trace[end - 1] != voltage_trace[start]:
                # Ramp
                grad = (voltage_trace[end - 1] - voltage_trace[start]) / \
                    (times[end - 1] - times[start])
                v_end = v_start + grad * (end_t - start_t)
            else:
                # Step
                v_end = voltage_trace[end - 1]

            lst.append(np.array([start_t, end_t, v_start, v_end]))

        return cls(np.vstack(lst), holding_potential, False)

    def get_holding_potential(self):
        """ Returns this protocol's holding potential. """
        return self.holding_potential

    def get_step_start_times(self):
        """ Returns a list of all segment start times. """
        return [line[0] for line in self._desc]

    def get_ramps(self):
        """
        Returns all segments that are ramps.

        Each segment is represented as ``(start time, end time, start voltage,
        end voltage)``.
        """
        return [line for line in self._desc if line[2] != line[3]]

    def get_all_sections(self):
        """
        Return an ``np.array`` describing the protocol, where each row in the
        array contains the the start time, end time, initial voltage and final
        voltage for a segment of the protocol.
        """
        return np.array(self._desc)

    def export_txt(self, fname):
        """
        Writes a partial textual representation of this protocol to a file.

        The created file will have a header line, followed by one line per
        segment. Segments are represented as "Type" (Set or Ramp), "Voltage"
        (the final voltage of a segment), and "Duration".
        """

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

            output_lines.append(f'{_type}\t{vend}\t{dur}')

        with open(fname, 'w') as fout:
            for line in output_lines:
                fout.write(line)
                fout.write('\n')

