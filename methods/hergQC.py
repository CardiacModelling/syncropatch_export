#!/usr/bin/env python2
import numpy as np
import scipy.stats
import os

import matplotlib.pyplot as plt


class hERGQC(object):

    QCnames = ['qc1.rseal', 'qc1.cm', 'qc1.rseries',
               'qc2.raw', 'qc2.subtracted',
               'qc3.raw', 'qc3.E4031', 'qc3.subtracted',
               'qc4.rseal', 'qc4.cm', 'qc4.rseries',
               'qc5.staircase', 'qc5.1.staircase',
               'qc6.subtracted', 'qc6.1.subtracted', 'qc6.2.subtracted']

    def __init__(self, sampling_rate=5, plot_dir=None, voltage=np.array([])):
        '''
        #
        # \item[\textbf{qc1.rseal}] check rseal within [1e8, 1e12] \gary{Units on all these...}
        # \item[\textbf{qc1.cm}] check cm within [1e-12, 1e-10]
        # \item[\textbf{qc1.rseries}] check rseries within [1e6, 2.5e7]
        # \item[\textbf{qc2.raw}] check raw trace recording SNR > 5 (SNR defined as std(trace) / std(noise)
        # \item[\textbf{qc2.subtracted}] check subtracted trace SNR > 5
        # \item[\textbf{qc3.raw}] check 2 sweeps of raw trace recording are similar by comparing the RMSD of the two sweeps < mean(RMSD to zero of the two sweeps) * 0.2
        # \item[\textbf{qc3.E4031}] check 2 sweeps of E4031 trace recording are similar (same comparison as qc3.raw)
        # \item[\textbf{qc3.subtracted}] check 2 sweeps of subtracted trace recording are similar (same comparison as qc3.raw)
        # \item[\textbf{qc4...}] check rseal, cm, rseries, respectively, before and after E4031 change (defined as std/mean) < 0.5
        # \item[\textbf{qc5.staircase}] check current during which hERG current could peak change by at least 75\% of the raw trace after E4031 addition
        # \item[\textbf{qc5.1.staircase}] check RMSD to zero (RMSD\_0) of staircase protocol change by at least 50\% of the raw trace after E4031 addition
        # \item[\textbf{qc6.subtracted}]: check the first big step up to +40 mV in the subtracted trace is bigger than -2 * estimated noise level
        # \item[\textbf{qc6.1.subtracted}] same as qc6.subtracted and applied at the first +40 mV during the staircase
        # \item[\textbf{qc6.2.subtracted}] same as qc6.subtracted and applied at the second +40 mV during the staircase
        #
        # Total number of QCs
        '''

        if plot_dir is not None:
            self.plot_dir = plot_dir

        self._n_qc = 16

        self.voltage = np.array(voltage)

        # Define all threshold here
        # qc1
        self.rsealc = [1e8, 1e12]  # in Ohm # TODO double check values
        self.cmc = [1e-12, 1e-10]  # in F
        self.rseriesc = [1e6, 2.5e7]  # in Ohm
        # qc2
        self.snrc = 25
        # qc3
        self.rmsd0c = 0.2
        # qc4
        self.rsealsc = 0.5
        self.cmsc = 0.5
        self.rseriessc = 0.5
        # qc5
        self.max_diffc = 0.75
        # self.qc5_win = [3275, 5750]  # indices with hERG screening peak!
        # indices where hERG could peak (for different temperatures)
        self.qc5_win = np.array([8550 + 400, 10950 + 400]) * sampling_rate
        # qc5_1
        self.rmsd0_diffc = 0.5
        # qc6
        self.negative_tolc = -2
        ''' # These are for `whole` (just staircase) protocol at 5kHz
        self.qc6_win = [3000, 7000]  # indices for first +40 mV
        self.qc6_1_win = [35250, 37250]  # indices for second +40 mV
        self.qc6_2_win = [40250, 42250]  # indices for third +40 mV
        '''
        # These are for `staircaseramp` protocol

        # Firstly, indices for 1st +40 mV
        self.qc6_win = np.array([1000, 1800]) * sampling_rate
        # indices for 2nd +40 mV
        self.qc6_1_win = np.array([7450, 7850]) * sampling_rate
        # indices for 3rd +40 mV
        self.qc6_2_win = np.array([8450, 8850]) * sampling_rate

        # Ensure these indices are integers
        self.qc5_win = self.qc5_win.astype(int)
        self.qc6_win = self.qc6_win.astype(int)
        self.qc6_1_win = self.qc6_1_win.astype(int)
        self.qc6_2_win = self.qc6_2_win.astype(int)

        self._debug = False
        self.fcap = None

    def get_qc_names(self):
        return self.QCnames

    def set_fcap(self, fcap):
        self.fcap = fcap

    def set_trace(self, before, after, qc_before, qc_after, n_sweeps):
        self._before = before
        self._qc_before = qc_before
        self._after = after
        self._qc_after = qc_after
        self._n_sweeps = n_sweeps

    def set_debug(self, debug):
        self._debug = debug

    def _run_qc(self):
        """Run each QC criteria on a single (before-trace, after-trace) pair.

        This requires self._before, and self._after should only be called by
        self.run_hergqc.
        """

        before = self._before
        qc_before = self._qc_before[0]
        after = self._after
        qc_after = self._qc_after[0]
        if (None in qc_before) or (None in qc_after):
            return False, [False] * 3 + [None] * 13

        # Filter off capacitive spikes
        if self.fcap is not None:
            for i in range(len(before)):
                before[i] = before[i] * self.fcap
                after[i] = after[i] * self.fcap

        qc1_1 = self.qc1(*qc_before)
        qc1_2 = self.qc1(*qc_after)
        qc1 = [i and j for i, j in zip(qc1_1, qc1_2)]

        qc2_1 = True
        qc2_2 = True
        for i in range(self._n_sweeps):
            qc2_1 = qc2_1 and self.qc2(before[i])
            qc2_2 = qc2_2 and self.qc2(before[i] - after[i])

        qc3_1 = self.qc3(before[0], before[1])
        qc3_2 = self.qc3(after[0], after[1])
        qc3_3 = self.qc3(before[0] - after[0],
                         before[1] - after[1])

        rseals = [qc_before[0], qc_after[0]]
        cms = [qc_before[1], qc_after[1]]
        rseriess = [qc_before[2], qc_after[2]]
        qc4 = self.qc4(rseals, cms, rseriess)

        qc5 = self.qc5(before[0], after[0],
                       self.qc5_win)  # indices where hERG peaks!
        qc5_1 = self.qc5_1(before[0], after[0], label='1')
        # Should be indices with +40 mV step up excluding first/last 100 ms

        # Ensure that the windows are correct by checking the voltage trace
        assert np.all(
            np.abs(self.voltage[self.qc6_win[0]: self.qc6_win[1]] - 40.0)) < 1e-8
        assert np.all(
            np.abs(self.voltage[self.qc6_1_win[0]: self.qc6_1_win[1]] - 40.0)) < 1e-8
        assert np.all(
            np.abs(self.voltage[self.qc6_2_win[0]: self.qc6_2_win[1]] - 40.0)) < 1e-8

        qc6 = self.qc6((before[0] - after[0]),
                       self.qc6_win, label='0')
        qc6_1 = self.qc6((before[0] - after[0]),
                         self.qc6_1_win, label='1')
        qc6_2 = self.qc6((before[0] - after[0]),
                         self.qc6_2_win, label='2')

        if self._debug:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(8, 5))
            fig.plot(before[0] - after[0])
            for l1, l2, l3, l4 in zip(self.qc5_win, self.qc6_win,
                                      self.qc6_1_win, self.qc6_2_win):
                plt.axvline(l1, c='#7f7f7f', label='qc5')
                plt.axvline(l2, c='#ff7f0e', ls='--', label='qc6')
                plt.axvline(l3, c='#2ca02c', ls='-.', label='qc6_1')
                plt.axvline(l4, c='#9467bd', ls=':', label='qc6_2')
            plt.xlabel('Time index (sample)')
            plt.ylabel('Current [pA]')

            # https://stackoverflow.com/a/13589144
            from collections import OrderedDict  # fix legend labels
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            fig.legend(by_label.values(), by_label.keys())
            fig.savefig('debug.png')
            plt.close(fig)

        QC = qc1 + [qc2_1, qc2_2, qc3_1, qc3_2, qc3_3] \
            + qc4 + [qc5, qc5_1, qc6, qc6_1, qc6_2]

        if all(QC):
            return True, QC
        else:
            return False, QC

    def qc1(self, rseal, cm, rseries):
        # Check R_seal, C_m, R_series within desired range
        if rseal < self.rsealc[0] or rseal > self.rsealc[1] \
                or not np.isfinite(rseal):
            print('rseal: ', rseal)
            qc11 = False
        else:
            qc11 = True
        if cm < self.cmc[0] or cm > self.cmc[1] or not np.isfinite(cm):
            print('cm: ', cm)
            qc12 = False
        else:
            qc12 = True
        if rseries < self.rseriesc[0] or rseries > self.rseriesc[1] \
                or not np.isfinite(rseries):
            print('rseries: ', rseries)
            qc13 = False
        else:
            qc13 = True
        return [qc11, qc12, qc13]

    def qc2(self, recording, method=3):
        # Check SNR is good
        if method == 1:
            # Not sure if this is good...
            snr = scipy.stats.signaltonoise(recording)
        elif method == 2:
            noise = np.std(recording[:200])
            snr = (np.max(recording) - np.min(recording) - 2 * noise) / noise
        elif method == 3:
            noise = np.std(recording[:200])
            snr = (np.std(recording) / noise) ** 2
        if snr < self.snrc or not np.isfinite(snr):
            print('snr: ', snr)
            return False
        return True

    def qc3(self, recording1, recording2, method=3):
        # Check 2 sweeps similar
        if method == 1:
            rmsdc = 2  # A/F * F
        elif method == 2:
            noise_1 = np.std(recording1[:200])
            peak_1 = (np.max(recording1) - noise_1)
            noise_2 = np.std(recording2[:200])
            peak_2 = (np.max(recording2) - noise_2)
            rmsdc = max(np.mean([peak_1, peak_2]) * 0.1,
                        np.mean([noise_1, noise_2]) * 5)
        elif method == 3:
            noise_1 = np.std(recording1[:200])
            noise_2 = np.std(recording2[:200])
            rmsd0_1 = np.sqrt(np.mean((recording1) ** 2))
            rmsd0_2 = np.sqrt(np.mean((recording2) ** 2))
            rmsdc = max(np.mean([rmsd0_1, rmsd0_2]) * self.rmsd0c,
                        np.mean([noise_1, noise_2]) * 6)
        rmsd = np.sqrt(np.mean((recording1 - recording2) ** 2))
        if rmsd > rmsdc or not (np.isfinite(rmsd) and np.isfinite(rmsdc)):
            print('rmsd: ', rmsd, 'rmsdc: ', rmsdc)
            return False
        return True

    def qc4(self, rseals, cms, rseriess):
        # Check R_seal, C_m, R_series stability
        # Require std/mean < x%
        if np.std(rseals) / np.mean(rseals) > self.rsealsc or not (
                np.isfinite(np.mean(rseals)) and np.isfinite(np.std(rseals))):
            print('d_rseal: ', np.std(rseals) / np.mean(rseals))
            qc41 = False
        else:
            qc41 = True
        if np.std(cms) / np.mean(cms) > self.cmsc or not (
                np.isfinite(np.mean(cms)) and np.isfinite(np.std(cms))):
            print('d_cm: ', np.std(cms) / np.mean(cms))
            qc42 = False
        else:
            qc42 = True
        if np.std(rseriess) / np.mean(rseriess) > self.rseriessc or not (
                np.isfinite(np.mean(rseriess))
                and np.isfinite(np.std(rseriess))):
            print('d_rseries: ', np.std(rseriess) / np.mean(rseriess))
            qc43 = False
        else:
            qc43 = True
        return [qc41, qc42, qc43]

    def qc5(self, recording1, recording2, win=None, label=''):
        # Check pharma peak value drops after E-4031 application
        # Require subtracted peak > 70% of the original peak
        if win is not None:
            i, f = win
        else:
            i, f = 0, -1
        # only look for peak before E4031

        if self.plot_dir:
            plt.axvspan(win[0], win[1], color='grey', alpha=.1)
            plt.plot(recording1, label='recording1')
            plt.plot(recording2, label='recording2')
            plt.savefig(os.path.join(self.plot_dir, f"qc5_{label}"))
            plt.clf()

        wherepeak = np.argmax(recording1[i:f])
        max_diff = recording1[i:f][wherepeak] - recording2[i:f][wherepeak]
        max_diffc = self.max_diffc * recording1[i:f][wherepeak]
        if (max_diff < max_diffc) or not (np.isfinite(max_diff)
                                          and np.isfinite(max_diffc)):
            print('max_diff: ', max_diff, 'max_diffc: ', max_diffc)
            return False
        return True

    def qc5_1(self, recording1, recording2, win=None, label=''):
        # Check RMSD_0 drops after E-4031 application
        # Require RMSD_0 (after E-4031 / before) diff > 50% of RMSD_0 before
        if win is not None:
            i, f = win
        else:
            i, f = 0, -1

        if self.plot_dir:
            if win:
                plt.axvspan(win[0], win[1], color='grey', alpha=.1)
            else:
                plt.plot(recording1, label='recording1')
                plt.plot(recording2, label='recording2')
                plt.savefig(os.path.join(self.plot_dir, f"qc5_{label}"))
                plt.clf()

        rmsd0_diff = np.sqrt(np.mean((recording1[i:f]) ** 2)) \
            - np.sqrt(np.mean((recording2[i:f]) ** 2))

        rmsd0_diffc = self.rmsd0_diffc *\
            np.sqrt(np.mean((recording1[i:f]) ** 2))

        if (rmsd0_diff < rmsd0_diffc) or not (np.isfinite(rmsd0_diff)
                                              and np.isfinite(rmsd0_diffc)):
            print('rmsd0_diff: ', rmsd0_diff, 'rmsd0c: ', rmsd0_diffc)
            return False
        return True

    def qc6(self, recording1, win=None, label=''):
        # Check subtracted staircase +40mV step up is non negative
        if win is not None:
            i, f = win
        else:
            i, f = 0, -1
        val = np.mean(recording1[i:f])

        if self.plot_dir:
            plt.axvspan(win[0], win[1], color='grey', alpha=.1)
            plt.plot(recording1, label='recording1')
            plt.savefig(os.path.join(self.plot_dir, f"qc6_{label}"))
            plt.clf()

        # valc = -0.005 * np.abs(np.sqrt(np.mean((recording1) ** 2)))  # or just 0
        valc = self.negative_tolc * np.std(recording1[:200])
        if (val < valc) or not (np.isfinite(val)
                                and np.isfinite(valc)):
            print('val: ', val, 'valc: ', valc)
            return False
        return True


def run_hergqc(before, after, qc_before, qc_after, n_sweeps, debug=False,
               fcap=None, sampling_rate=5, plot_dir=None, voltage=np.array([])):
    """ Run hERG QC using default setting

    """

    QC = hERGQC(sampling_rate=sampling_rate, plot_dir=plot_dir,
                voltage=voltage)
    QC.set_trace(before, after, qc_before, qc_after, n_sweeps)
    QC.set_fcap(fcap)
    QC.set_debug(debug)
    return QC.run()


def visual_hergqc(SELECTION, ID, QCs, saveas):
    '''
    Plot the number of cells selected by each of the QC critieron
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import string
    WELL_ID = [label + str(i).zfill(2) for label in string.ascii_uppercase[:16]
               for i in range(1, 25)]

    chip_qc = []
    n_no_cell = 0
    for cell in WELL_ID:
        try:
            selection = SELECTION[ID + cell]
            if type(selection) is tuple:
                # Can remove this later, my previous bug in run-qc.py
                selection = selection[1]
            if not any(selection[:3]) and \
                    selection[3:] == [None] * len(selection[3:]):
                n_no_cell += 1
            else:
                chip_qc.append(selection)
        except KeyError:
            print('No cell: ' + ID + cell)
            n_no_cell += 1
    chip_qc = np.asarray(chip_qc)
    n_cells = chip_qc.shape[0]
    assert n_cells + n_no_cell == 384

    # Plot 1
    fig = plt.figure(figsize=(6, 3.5))
    # Add one more column for simply no cell
    bars = np.append(n_no_cell, n_cells - np.sum(chip_qc, axis=0))
    fig.bar(range(len(QCs) + 1), bars, color='#4271ff')
    fig.xticks(range(len(QCs) + 1), ['no cell'] + QCs, rotation=90)
    fig.ylabel('Filtered number', fontsize=12)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.savefig(saveas + '-qc.png', bbox_inch='tight', dpi=300)
    plt.close(fig)

    # Plot matrix
    matrix = np.zeros((len(QCs), len(QCs)))
    for i in range(len(QCs)):
        for j in range(len(QCs)):
            matrix[i, j] = np.sum(np.bitwise_not(
                chip_qc[:, i] | chip_qc[:, j]))
    fig, ax = plt.subplots(figsize=(8, 8))
    # vmin, vmax here is a bit arbitrary...
    vmin = 0
    vmax = np.max(matrix)
    # .T is needed for the ordering i,j below
    im = ax.matshow(matrix.T, cmap=plt.cm.Blues, vmin=vmin, vmax=vmax)
    # do some tricks with the colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    # Only do integer ticks
    from matplotlib.ticker import MaxNLocator
    cbar.locator = MaxNLocator(integer=True)
    cbar.ax.set_ylabel('Filtered number', fontsize=16)
    # change the current axis back to ax
    fig.sca(ax)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            c = matrix[i, j]
            ax.text(i, j, '%d' % c, va='center', ha='center')
    fig.yticks(range(len(QCs)), QCs)
    fig.xticks(range(len(QCs)), QCs, rotation=90)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.savefig(saveas + '-matrix.png', bbox_inch='tight', dpi=300)
    plt.close(fig)
