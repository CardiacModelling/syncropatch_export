import json
import numpy as np
import string

class Trace:
    def __init__(self, filepath, json_file):
        self.filepath = filepath
        self.json_file = json_file
        with open(self.filepath + self.json_file) as f:
            self.meta = json.load(f)
        TraceHeader = self.meta['TraceHeader']
        try:
            self.TimeScaling = TraceHeader['TimeScaling']
        except KeyError:
            self.TimeScaling = TraceHeader['TimeScalingIV']
        self.MeasurementLayout = TraceHeader['MeasurementLayout']
        self.FileInformation = TraceHeader['FileInformation']
        self.WELL_ID = [[l+str(i).zfill(2) for l in string.ascii_uppercase[:16]] for i in range(1,25)]
        self.NofSweeps = self.MeasurementLayout['NofSweeps']
        self.WP_nRows = TraceHeader['Chiplayout']['WP_nRows']
        self.nCols = self.MeasurementLayout['nCols'] 
        self.NofSamples = self.MeasurementLayout['NofSamples']
        self.Leakdata = self.MeasurementLayout['Leakdata']
        self.SweepsPerFile = self.FileInformation['SweepsPerFile']
        self.I2DScale = self.TimeScaling['I2DScale']
        self.ColsMeasured = self.MeasurementLayout['ColsMeasured']
        self.FileList = self.FileInformation['FileList']
        self.FileList.sort()

    def voltage(self):
        return self.TimeScaling['Stimulus']
        
    def times(self):
        return self.TimeScaling['TR_Time']
        
    def all_traces(self, leakcorrect=False):
        OUT = {}
        for iCol in self.WELL_ID:
            for ijWell in iCol:
                OUT[ijWell] = []

        currentSweep = 0
        for trace_file in self.FileList:
            if currentSweep + self.SweepsPerFile > self.NofSweeps:
                totalSweep = self.NofSweeps - currentSweep
            else:
                totalSweep = self.SweepsPerFile
            currentSweep += self.SweepsPerFile
            assert(totalSweep > 0)
            with open(self.filepath + trace_file, 'r') as f:
                trace = np.fromfile(f, dtype=np.int16)
            trace = np.asarray(trace)
            assert(len(trace) == self.Leakdata * self.NofSamples * self.WP_nRows
                                * self.nCols * totalSweep)
            idx_i = 0
            for kSweep in range(totalSweep):
                for i, iCol in enumerate(self.ColsMeasured):
                    if iCol == -1:
                        continue
                    idx_f = idx_i + self.Leakdata * self.NofSamples * self.WP_nRows
                    assert(idx_f <= len(trace))
                    iColTraces = np.array(trace[idx_i:idx_f]) * self.I2DScale[i] * 1e12
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
                    del(iColTraces)
                    idx_i = idx_f
            del(trace)
        return OUT
  
    def get_trace_file(self, sweeps):
        OUT_file_idx = []
        OUT_idx_i = []
        for actSweep in sweeps:
            ActFile = int(actSweep/self.SweepsPerFile)
            OUT_file_idx.append(ActFile)
            DataPerWell = self.NofSamples * self.Leakdata
            ColOffset = DataPerWell * self.WP_nRows
            SweepOffset = ColOffset * self.nCols
            ReadOffset = (actSweep % self.SweepsPerFile) * SweepOffset
            OUT_idx_i.append(ReadOffset)
        return OUT_file_idx, OUT_idx_i

    def get_trace_sweep(self, sweeps, leakcorrect=False):
        OUT = {}
        for iCol in self.WELL_ID:
            for ijWell in iCol:
                OUT[ijWell] = []
        
        if len(sweeps) > self.NofSweeps:
            raise ValueError('Required #sweeps > total #sweeps.')
        
        for i, sweep in enumerate(sweeps):
            if sweep < 0:
                sweeps[i] = self.NofSweeps + sweep
        
        trace_file_idxs, idx_is = self.get_trace_file(sweeps)
        
        for trace_file_idx, idx_i in zip(trace_file_idxs, idx_is):
            if trace_file_idx < len(self.FileList) - 1:
                totalSweep = self.SweepsPerFile
            else:
                totalSweep = self.NofSweeps % self.SweepsPerFile
            
            trace_file = self.FileList[trace_file_idx]
            
            with open(self.filepath + trace_file, 'r') as f:
                trace = np.fromfile(f, dtype=np.int16)
            
            trace = np.asarray(trace)
            
            assert(len(trace) == self.Leakdata * self.NofSamples * self.WP_nRows
                                * self.nCols * totalSweep)
            
            for i, iCol in enumerate(self.ColsMeasured):
                if iCol == -1:
                    continue
                idx_f = idx_i + self.Leakdata * self.NofSamples * self.WP_nRows
                # convert to double in pA!
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

                idx_i = idx_f
            del(trace)
        return OUT