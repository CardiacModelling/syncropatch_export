import json
import numpy as np
import string


class Trace:
    '''
    Defines a Trace object from the output of a Nanion experiment.
    Required as input:
    filepath - path pointing to folder containing .json and .dat files (str)
    json_file - specific filename of json file (str)
    '''

    def __init__(self, filepath, json_file):
        # store file paths
        self.filepath = filepath
        if ".json" in json_file:
            self.json_file = json_file
        else:
            self.json_file = json_file + ".json"

        # load json file
        with open(self.filepath + self.json_file) as f:
            self.meta = json.load(f)

        # store necessary header info
        TraceHeader = self.meta['TraceHeader']
        try:
            self.TimeScaling = TraceHeader['TimeScaling']
        except KeyError:
            self.TimeScaling = TraceHeader['TimeScalingIV']
        self.MeasurementLayout = TraceHeader['MeasurementLayout']
        self.FileInformation = TraceHeader['FileInformation']
        self.WELL_ID = [
            [l+str(i).zfill(2) for l in string.ascii_uppercase[:16]] for i in range(1, 25)]
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

    def get_voltage(self):
        '''
        Returns the voltage stimulus from Nanion .json file
        '''
        return self.TimeScaling['Stimulus']

    def get_times(self):
        '''
        Returns the time steps from Nanion .json file
        '''
        return self.TimeScaling['TR_Time']

    def get_all_traces(self, leakcorrect=False):
        '''
        Returns all raw current traces from .dat files
        '''

        # initialise output
        OUT = {}
        for iCol in self.WELL_ID:
            for ijWell in iCol:
                OUT[ijWell] = []

        currentSweep = 0
        for trace_file in self.FileList:
            # work out the total number of sweeps in the current trace file
            if currentSweep + self.SweepsPerFile > self.NofSweeps:
                totalSweep = self.NofSweeps - currentSweep
            else:
                totalSweep = self.SweepsPerFile
            currentSweep += self.SweepsPerFile
            assert totalSweep > 0

            # get trace data
            with open(self.filepath + trace_file, 'r') as f:
                trace = np.fromfile(f, dtype=np.int16)

            # convert to numpy array
            trace = np.asarray(trace)

            # check loaded traces have the expected length
            assert len(trace) == self.Leakdata * self.NofSamples \
                * self.WP_nRows * self.nCols * totalSweep
            idx_i = 0

            # loop through sweeps and columns
            for kSweep in range(totalSweep):
                for i, iCol in enumerate(self.ColsMeasured):
                    if iCol == -1:
                        # -1 not measured (TODO doublecheck this)
                        continue

                    idx_f = idx_i + self.Leakdata * \
                        self.NofSamples * self.WP_nRows
                    assert idx_f <= len(trace)

                    # convert to double in pA
                    iColTraces = np.array(
                        trace[idx_i:idx_f]) * self.I2DScale[i] * 1e12

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
                        OUT[ijWell].append(iColTraces[start:end])
                    del iColTraces
                    # update idx_i
                    idx_i = idx_f
            del trace
        return OUT

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

    def get_trace_sweep(self, sweeps, leakcorrect=False):
        '''
        Returns a subset of sweeps defined by the input 'sweeps'
        '''

        # initialise output
        OUT = {}
        for iCol in self.WELL_ID:
            for ijWell in iCol:
                OUT[ijWell] = []

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
            with open(self.filepath + trace_file, 'r') as f:
                trace = np.fromfile(f, dtype=np.int16)

            # convert to numpy array
            trace = np.asarray(trace)

            # check loaded traces have the expected length
            assert len(trace) == self.Leakdata * self.NofSamples\
                * self.WP_nRows * self.nCols * totalSweep

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
                    OUT[ijWell].append(iColTraces[start:end])

                # update idx_i
                idx_i = idx_f
            del trace
        return OUT

    def get_onboard_QC_values(self, sweep=None):
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
        OUT = {}
        for iCol in self.WELL_ID:
            for ijWell in iCol:
                OUT[ijWell] = []

        if sweep is None:
            sweep = range(self.NofSweeps)

        for k in sweep:
            for i in range(self.WP_nCols):
                for j in range(self.WP_nRows):
                    OUT[self.WELL_ID[i][j]].append((RSeal[k][i][j],
                                                    Capacitance[k][i][j],
                                                    Rseries[k][i][j]))

        return OUT
