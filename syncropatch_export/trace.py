import json
import os
import string

import numpy as np
import pandas as pd

from .voltage_protocols import VoltageProtocol


class Trace:
    """ Defines a Trace object from the output of a Nanion experiment.

    @params
    filepath: path pointing to folder containing .json and .dat files (str)
    json_file: specific filename of json file (str)
    """

    def __init__(self, filepath, json_file: str):
        # store file paths
        self.filepath = filepath
        if json_file[-5:] == '.json':
            self.json_file = json_file
        else:
            self.json_file = json_file + ".json"

        # load json file
        with open(os.path.join(self.filepath, self.json_file)) as f:
            self.meta = json.load(f)

        # store necessary header info
        TraceHeader = self.meta['TraceHeader']
        try:
            self.TimeScaling = TraceHeader['TimeScaling']
        except KeyError:
            self.TimeScaling = TraceHeader['TimeScalingIV']

        times = self.get_times()
        self.sampling_rate = int(1 / (times[1] - times[0]))

        self.MeasurementLayout = TraceHeader['MeasurementLayout']
        self.FileInformation = TraceHeader['FileInformation']

        self.WELL_ID = np.array([
            [lab + str(i).zfill(2) for lab in string.ascii_uppercase[:16]]
            for i in range(1, 25)])

        self.NofSweeps = self.MeasurementLayout['NofSweeps']
        self.WP_nRows = TraceHeader['Chiplayout']['WP_nRows']
        self.WP_nCols = TraceHeader['Chiplayout']['WP_nCols']
        self.nCols = self.MeasurementLayout['nCols']
        self.NofSamples = self.MeasurementLayout['NofSamples']
        self.Leakdata = self.MeasurementLayout['Leakdata']
        self.SweepsPerFile = self.FileInformation['SweepsPerFile']
        self.I2DScale = self.TimeScaling['I2DScale']
        self.ColsMeasured = self.MeasurementLayout['ColsMeasured']
        self.FileList = self.FileInformation['FileList']
        self.FileList.sort()

        self.voltage_protocol = self.get_voltage_protocol()

    def get_voltage_protocol(self, holding_potential=-80.0):
        """Extract information about the voltage protocol from the json file

        returns: a VoltageProtocol object

        """

        voltage_protocol = VoltageProtocol.from_json(
            self.meta['ExperimentConditions']['VoltageProtocol'],
            self.meta['ExperimentConditions']['VMembrane_mV']
        )

        return voltage_protocol

    def get_voltage_protocol_json(self):
        """
        Returns the voltage protocol as a JSON object
        """
        return self.meta['ExperimentConditions']['VoltageProtocol'][0]

    def get_protocol_description(self, holding_potential=-80.0):
        """Get the protocol as a numpy array describing the voltages and
        durations for each section

        returns: np.array where each row contains the start time, end time,
        initial voltage, and final voltage

        """
        return self.get_voltage_protocol().get_all_sections()

    def get_voltage(self):
        '''
        Returns the voltage stimulus from Nanion .json file
        '''
        return np.array(self.TimeScaling['Stimulus']).astype(np.float64)\
            * 1e3

    def get_times(self):
        '''
        Returns the time steps from Nanion .json file
        '''
        return np.array(self.TimeScaling['TR_Time']) * 1e3

    def get_all_traces(self, leakcorrect=False):
        '''

        Params:
        leakcorrect: Bool. Set to true if using onboard leak correction

        Returns: all raw current traces from .dat files

        '''
        return self.get_trace_sweeps(leakcorrect=leakcorrect)

    def get_trace_file(self, sweeps):
        '''
        Returns the trace file index of the file for a given set of sweeps
        '''
        OUT_file_idx = []
        OUT_idx_i = []
        for actSweep in sweeps:
            # work out trace file idx
            ActFile = int(actSweep/self.SweepsPerFile)
            OUT_file_idx.append(ActFile)

            # work out idx_i
            DataPerWell = self.NofSamples * self.Leakdata
            ColOffset = DataPerWell * self.WP_nRows
            SweepOffset = ColOffset * self.nCols
            ReadOffset = (actSweep % self.SweepsPerFile) * SweepOffset
            OUT_idx_i.append(ReadOffset)
        return OUT_file_idx, OUT_idx_i

    def get_trace_sweeps(self, sweeps=None, leakcorrect=False):
        '''
        Returns a subset of sweeps defined by the input 'sweeps'
        '''

        # initialise output
        out_dict = {}
        for iCol in self.WELL_ID:
            for ijWell in iCol:
                out_dict[ijWell] = []

        if sweeps is None:
            #  Sometimes NofSweeps seems to be incorrect
            sweeps = list(range(self.NofSweeps))

        # check `getsweep` input is something sensible
        if len(sweeps) > self.NofSweeps:
            raise ValueError('Required #sweeps > total #sweeps.')

        # convert negative values to positive
        for i, sweep in enumerate(sweeps):
            if sweep < 0:
                sweeps[i] = self.NofSweeps + sweep

        trace_file_idxs, idx_is = self.get_trace_file(sweeps)

        for trace_file_idx, idx_i in zip(trace_file_idxs, idx_is):
            if trace_file_idx < len(self.FileList) - 1:
                totalSweep = self.SweepsPerFile
            else:
                totalSweep = self.NofSweeps % self.SweepsPerFile

            # get trace data
            trace_file = self.FileList[trace_file_idx]
            with open(os.path.join(self.filepath, trace_file), 'r') as f:
                trace = np.fromfile(f, dtype=np.int16)

            # convert to numpy array
            trace = np.asarray(trace)

            # check loaded traces have the expected length
            # assert len(trace) == self.Leakdata * self.NofSamples\
            #     * self.WP_nRows * self.nCols * totalSweep

            if len(trace) == 0:
                trace = np.full(self.Leakdata * self.NofSamples * self.WP_nRows
                                * self.nCols * totalSweep, np.nan)

            #  Iterate over wells
            for i, iCol in enumerate(self.ColsMeasured):
                if iCol == -1:
                    continue  # -1 not measured (need to doublecheck this)
                idx_f = idx_i + self.Leakdata * self.NofSamples * self.WP_nRows

                # convert to double in pA
                iColTraces = trace[idx_i:idx_f] * self.I2DScale[i] * 1e12
                iColWells = self.WELL_ID[i]

                for j, ijWell in enumerate(iColWells):
                    if leakcorrect:
                        leakoffset = 1
                    else:
                        leakoffset = 0
                    start = j * self.Leakdata * self.NofSamples \
                        + leakoffset * self.NofSamples
                    end = j * self.Leakdata * self.NofSamples \
                        + (leakoffset + 1) * self.NofSamples
                    well_trace = np.array(iColTraces[start:end])
                    out_dict[ijWell].append(well_trace)

                # update idx_i
                idx_i = idx_f
            del trace

        for key in out_dict:
            # If shapes agree, convert them into a single array
            list_arrs = out_dict[key]
            shapes = [arr.shape for arr in list_arrs]
            if len(set(shapes)) != 1:
                # logging.warning(f"Mismatched sweep dimensions (maybe one is empty?)")
                traces = [t for t in out_dict[key] if len(t) > 0]
            else:
                traces = out_dict[key]

            out_dict[key] = np.vstack(traces)

        return out_dict

    def get_onboard_QC_values(self, sweeps=None):
        '''Read quality control values Rseal, Cslow (Cm), and Rseries from a Nanion .json file

        returns: A dictionary where the keys are the well e.g. 'A01' and the
        values are the values used for onboard QC i.e., the seal resistance,
        cell capacitance and the series resistance.

        '''

        # load QC values
        RSeal = np.array(self.meta['QCData']['RSeal'])
        Capacitance = np.array(self.meta['QCData']['Capacitance'])
        Rseries = np.array(self.meta['QCData']['Rseries'])

        # initialise output
        out_dict = {}
        for iCol in self.WELL_ID:
            for ijWell in iCol:
                out_dict[ijWell] = []

        if sweeps is None:
            sweeps = list(range(RSeal.shape[0]))

        for k in sweeps:
            for i in range(self.WP_nCols):
                for j in range(self.WP_nRows):
                    out_dict[self.WELL_ID[i][j]].append((RSeal[k, i, j],
                                                         Capacitance[k, i, j],
                                                         Rseries[k, i, j]))

        # Convert values to np arrays taking care to remove handle None values
        for well in out_dict:
            vals = out_dict[well]
            if vals:
                shape = (len(vals), len(vals[0]))
                vals = [x if x is not None else np.nan for x in vals]
                vals = np.vstack(vals).reshape(shape).astype(np.float64)
                out_dict[well] = vals

        return out_dict

    def get_onboard_QC_df(self, sweeps=None):
        """Create a Pandas DataFrame which lists the Rseries, memebrane
        capacitance and Rseries for each well and sweep.

        @Returns A pandas.DataFrame describing the onboard QC estimates for
        each well, sweep

        """

        QC_dict = self.get_onboard_QC_values(sweeps)

        if sweeps is None:
            sweeps = list(range(self.NofSweeps))

        df_rows = []
        for sweep in sweeps:
            for well in self.WELL_ID.flatten():
                Rseal, Capacitance, Rseries = QC_dict[well][sweep]
                df_row = {'Rseal': Rseal,
                          'Cm': Capacitance,
                          'Rseries': Rseries,
                          'well': well,
                          'sweep': sweep
                          }
                df_rows.append(df_row)

        return pd.DataFrame.from_records(df_rows)
