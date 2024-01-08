import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


class hERGQC(object):

    QCnames = ['qc1.rseal', 'qc1.cm', 'qc1.rseries',
               'qc2.raw', 'qc2.subtracted',
               'qc3.raw', 'qc3.E4031', 'qc3.subtracted',
               'qc4.rseal', 'qc4.cm', 'qc4.rseries',
               'qc5.staircase', 'qc5.1.staircase',
               'qc6.subtracted', 'qc6.1.subtracted', 'qc6.2.subtracted']

    def __init__(self, sampling_rate=5, plot_dir=None, voltage=np.array([]),
                 n_sweeps=None):
        # TODO docstring

        if plot_dir is not None:
            self.plot_dir = plot_dir

        self._n_qc = 16

        self.output_dir = 'output'
        self.voltage = np.array(voltage)

        # Define all thresholds

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

        self._debug = True
        self.fcap = None

        self.qc_labels = ['qc1.rseal', 'qc1.cm', 'qc1.rseries', 'qc2.raw',
                          'qc2.subtracted', 'qc3.raw', 'qc3.E4031',
                          'qc3.subtracted', 'qc4.rseal', 'qc4.cm',
                          'qc4.rseries', 'qc5.staircase', 'qc5.1.staircase',
                          'qc6.subtracted', 'qc6.1.subtracted',
                          'qc6.2.subtracted']

    def get_qc_names(self):
        return self.QCnames

    def set_fcap(self, fcap):
        self.fcap = fcap

    def set_trace(self, before, after, qc_vals_before,
                  qc_vals_after, n_sweeps):
        self._before = before
        self._qc_vals_before = qc_vals_before
        self._after = after
        self._qc_vals_after = qc_vals_after

    def set_debug(self, debug):
        self._debug = debug

    def run_qc(self, before=None, after=None, qc_vals_before=None,
               qc_vals_after=None, n_sweeps=None):
        """Run each QC criteria on a single (before-trace, after-trace) pair.

        This requires self._before, and self._after should only be called by
        self.run_hergqc.
        """

        if before is None:
            before = self.before

        if not n_sweeps:
            n_sweeps = len(before)

        if after is None:
            after = self.after

        before = np.array(before)
        after = np.array(after)

        if len(before) == 0 or len(after) == 0:
            return False, [False for lab in self.qc_labels]

        if qc_vals_before is None:
            qc_vals_before = self.qc_vals_before

        if qc_vals_after is None:
            qc_vals_after = self.qc_vals_after

        if (None in qc_vals_before) or (None in qc_vals_after):
            return False, [False] * 3 + [None] * 13

        # Filter off capacitive spikes
        if self.fcap is not None:
            for i in range(len(before)):
                before[i] = before[i] * self.fcap
                after[i] = after[i] * self.fcap

        qc1_1 = self.qc1(*qc_vals_before)
        qc1_2 = self.qc1(*qc_vals_after)
        qc1 = [i and j for i, j in zip(qc1_1, qc1_2)]

        qc2_1 = True
        qc2_2 = True
        for i in range(n_sweeps):
            qc2_1 = qc2_1 and self.qc2(before[i])
            qc2_2 = qc2_2 and self.qc2(before[i] - after[i])

        qc3_1 = self.qc3(before[0, :], before[1, :])
        qc3_2 = self.qc3(after[0, :], after[1, :])
        qc3_3 = self.qc3(before[0, :] - after[0, :],
                         before[1, :] - after[1, :])

        rseals = [qc_vals_before[0], qc_vals_after[0]]
        cms = [qc_vals_before[1], qc_vals_after[1]]
        rseriess = [qc_vals_before[2], qc_vals_after[2]]
        qc4 = self.qc4(rseals, cms, rseriess)

        qc5 = self.qc5(before[0, :], after[0, :],
                       self.qc5_win)  # indices where hERG peaks!
        qc5_1 = self.qc5_1(before[0, :], after[0, :], label='1')
        # Should be indices with +40 mV step up excluding first/last 100 ms

        # Ensure that the windows are correct by checking the voltage trace
        assert np.all(
            np.abs(self.voltage[self.qc6_win[0]: self.qc6_win[1]] - 40.0))\
            < 1e-8
        assert np.all(
            np.abs(self.voltage[self.qc6_1_win[0]: self.qc6_1_win[1]] - 40.0))\
            < 1e-8
        assert np.all(
            np.abs(self.voltage[self.qc6_2_win[0]: self.qc6_2_win[1]] - 40.0))\
            < 1e-8

        qc6 = self.qc6((before[0, :] - after[0, :]),
                       self.qc6_win, label='0')
        qc6_1 = self.qc6((before[0, :] - after[0, :]),
                         self.qc6_1_win, label='1')
        qc6_2 = self.qc6((before[0, :] - after[0, :]),
                         self.qc6_2_win, label='2')

        if self._debug:
            fig = plt.figure(figsize=(8, 5))
            ax = fig.subplots()
            ax.plot(before[0, :] - after[0, :])
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

            fig.savefig(os.path.join(self.output_dir, 'debug.png'))
            plt.close(fig)

        # Make a flat list of all QC criteria (pass/fail bool)
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
            logging.debug('rseal: ', rseal)
            qc11 = False
        else:
            qc11 = True
        if cm < self.cmc[0] or cm > self.cmc[1] or not np.isfinite(cm):
            logging.debug('cm: ', cm)
            qc12 = False
        else:
            qc12 = True
        if rseries < self.rseriesc[0] or rseries > self.rseriesc[1] \
                or not np.isfinite(rseries):
            logging.debug('rseries: ', rseries)
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
            logging.debug('snr: ', snr)
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
            logging.debug('rmsd: ', rmsd, 'rmsdc: ', rmsdc)
            return False
        return True

    def qc4(self, rseals, cms, rseriess):
        # Check R_seal, C_m, R_series stability
        # Require std/mean < x%
        if np.std(rseals) / np.mean(rseals) > self.rsealsc or not (
                np.isfinite(np.mean(rseals)) and np.isfinite(np.std(rseals))):
            logging.debug('d_rseal: ', np.std(rseals) / np.mean(rseals))
            qc41 = False
        else:
            qc41 = True
        if np.std(cms) / np.mean(cms) > self.cmsc or not (
                np.isfinite(np.mean(cms)) and np.isfinite(np.std(cms))):
            logging.debug('d_cm: ', np.std(cms) / np.mean(cms))
            qc42 = False
        else:
            qc42 = True
        if np.std(rseriess) / np.mean(rseriess) > self.rseriessc or not (
                np.isfinite(np.mean(rseriess))
                and np.isfinite(np.std(rseriess))):
            logging.debug('d_rseries: ', np.std(rseriess) / np.mean(rseriess))
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

        if self.plot_dir and self._debug:
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
            logging.debug('max_diff: ', max_diff, 'max_diffc: ', max_diffc)
            return False
        return True

    def qc5_1(self, recording1, recording2, win=None, label=''):
        # Check RMSD_0 drops after E-4031 application
        # Require RMSD_0 (after E-4031 / before) diff > 50% of RMSD_0 before
        if win is not None:
            i, f = win
        else:
            i, f = 0, -1

        if self.plot_dir and self._debug:
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
            logging.debug('rmsd0_diff: ', rmsd0_diff, 'rmsd0c: ', rmsd0_diffc)
            return False
        return True

    def qc6(self, recording1, win=None, label=''):
        # Check subtracted staircase +40mV step up is non negative
        if win is not None:
            i, f = win
        else:
            i, f = 0, -1
        val = np.mean(recording1[i:f])

        if self.plot_dir and self._debug:
            plt.axvspan(win[0], win[1], color='grey', alpha=.1)
            plt.plot(recording1, label='recording1')
            plt.savefig(os.path.join(self.plot_dir, f"qc6_{label}"))
            plt.clf()

        # valc = -0.005 * np.abs(np.sqrt(np.mean((recording1) ** 2)))  # or just 0
        valc = self.negative_tolc * np.std(recording1[:200])
        if (val < valc) or not (np.isfinite(val)
                                and np.isfinite(valc)):
            logging.debug('val: ', val, 'valc: ', valc)
            return False
        return True
