import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly

from .trace import Trace


def infer_reversal_potential(trace: Trace, sweep: int, well: str, ax=None,
                             output_path=None, plot=None, known_Erev=None,
                             current=None):

    if output_path:
        dirname = os.path.dirname(output_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    if (ax or output_path) and plot is not False:
        plot = True

    times = trace.get_times()
    # convert to ms
    times = times

    voltages = trace.get_voltage()
    # convert to mV
    voltages = voltages

    # Find indices of observations during the reversal ramp
    prot_description = trace.get_protocol_description()
    ramps = [line for line in prot_description.get_ramps()
             if line[2] != line[3]]

    # Assume the last ramp is the reversal ramp (convert to ms)
    tstart, tend = np.array(ramps)[-1, :2]

    istart = np.argmax(times > tstart)
    iend = np.argmax(times > tend)

    # print(tstart, tend)
    # print(istart, iend)
    # print(times)

    if current is None:
        current = trace.get_trace_sweeps([sweep])[well][0, :].flatten()

    times = times[istart:iend]
    current = current[istart:iend]
    voltages = voltages[istart:iend]

    try:
        fitted_poly = poly.Polynomial.fit(voltages, current, 4)
    except ValueError as exc:
        logging.warning(str(exc))
        return np.nan

    try:
        roots = np.unique([np.real(root) for root in fitted_poly.roots()
                           if root > np.min(voltages) and root < np.max(voltages)])
    except np.linalg.LinAlgError as exc:
        logging.warning(str(exc))
        return np.nan

    # Take the last root (greatest voltage). This should be the first time that
    # the current crosses 0 and where the ion-channel kinetics are too slow to
    # play a role

    if len(roots) == 0:
        return np.nan

    if plot:
        created_fig = False
        if ax is None and output_path is not None:

            created_fig = True
            fig = plt.figure()
            ax = fig.subplots()

        ax.set_xlabel('voltage mV')
        ax.set_ylabel('current nA')
        # Now plot current vs voltage
        ax.plot(voltages, current, 'x', markersize=2, color='grey', alpha=.5)
        ax.axvline(roots[-1], linestyle='--', color='grey', label="$E_{Kr}$")
        if known_Erev:
            ax.axvline(known_Erev, linestyle='--', color='yellow', label="known $E_{Kr}$")
        ax.axhline(0, linestyle='--', color='grey')
        ax.plot(*fitted_poly.linspace())
        ax.legend()

        if output_path is not None:
            fig = ax.figure
            fig.savefig(output_path)

        if created_fig:
            plt.close(fig)

    return roots[-1]
