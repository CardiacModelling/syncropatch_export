import numpy as np
from matplotlib import pyplot as plt


def linear_reg(V, I):
    # number of observations/points
    n = np.size(V)

    # mean of V and I vector
    m_V = np.mean(V)
    m_I = np.mean(I)

    # calculating cross-deviation and deviation about V
    SS_VI = np.sum(I*V) - n*m_I*m_V
    SS_VV = np.sum(V*V) - n*m_V*m_V

    # calculating regression coefficients
    b_1 = SS_VI / SS_VV
    b_0 = m_I - b_1*m_V

    return (b_0, b_1)


def get_QC_dict(QC, bounds={'Rseal': (10e8, 10e12), 'Cm': (1e-12, 1e-10), 'Rseries': (1e6, 2.5e7)}):
    '''
    inputs: 
    QC  - QC trace attribute
    bounds - method of filtering
    '''
    QC_dict = {}
    for well in QC:
        for qc in QC[well]:
            if all(qc):
                if (bounds['Rseal'][0] < qc[0] < bounds['Rseal'][1]) & (bounds['Cm'][0] < qc[1] < bounds['Cm'][1]) & (bounds['Rseries'][0] < qc[2] < bounds['Rseries'][1]):
                    if well in QC_dict:
                        QC_dict[well] = QC_dict[well] + [qc]
                    else:
                        QC_dict[well] = [qc]

    max_swp = max(len(QC_dict[well]) for well in QC_dict)
    QC_copy = QC_dict.copy()
    for well in QC_copy:
        if len(QC_dict[well]) != max_swp:
            QC_dict.pop(well)
    return QC_dict


def get_leak_corrected(trace, currents, QC_filt, ramp_bounds):
    leak_corrected = {}
    V = 1000*np.array(currents['voltages'])  # mV
    for row in trace.WELL_ID:
        for well in row:
            if well in QC_filt.keys():
                leak_corrected[well] = {}
                for sweep in range(trace.NofSweeps):
                    I = currents[well][sweep]  # pA
                    b_0, b_1 = linear_reg(
                        V[ramp_bounds[0]:ramp_bounds[1]+1], I[ramp_bounds[0]:ramp_bounds[1]+1])
                    I_leak = b_1*V + b_0
                    leak_corrected[well][sweep] = I - I_leak
    return leak_corrected


def plot_leak_fit(currents, QC_filt, well, sweep, ramp_bounds, save=False):
    if save is False:
        print(f'QC_pass: {well in QC_filt.keys()}')
    V = 1000*np.array(currents['voltages'])  # mV
    I = currents[well][sweep]  # pA
    b_0, b_1 = linear_reg(V[ramp_bounds[0]:ramp_bounds[1]+1],
                          I[ramp_bounds[0]:ramp_bounds[1]+1])
    I_leak = b_1*V + b_0

    # fit to leak ramp
    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(7.5, 6))

    ax1.set_title('current vs time')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('current (pA)')
    ax1.plot(currents['times'], I)
    ax1.axvline(ramp_bounds[0]*0.0005, linestyle='--', color='k', alpha=0.5)
    ax1.axvline(ramp_bounds[1]*0.0005, linestyle='--', color='k', alpha=0.5)
    ax1.set_xlim(left=ramp_bounds[0]*0.0005 - 1,
                 right=ramp_bounds[1]*0.0005 + 1)

    ax2.set_title('voltage vs time')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('voltage (mV)')
    ax2.plot(currents['times'], V)
    ax2.axvline(ramp_bounds[0]*0.0005, linestyle='--', color='k', alpha=0.5)
    ax2.axvline(ramp_bounds[1]*0.0005, linestyle='--', color='k', alpha=0.5)
    ax2.set_xlim(left=ramp_bounds[0]*0.0005 - 1,
                 right=ramp_bounds[1]*0.0005 + 1)

    ax3.set_title('current vs voltage')
    ax3.set_xlabel('voltage (mV)')
    ax3.set_ylabel('current (pA)')
    ax3.plot(V[ramp_bounds[0]:ramp_bounds[1]+1],
             I[ramp_bounds[0]:ramp_bounds[1]+1], 'x')
    ax3.plot(V[ramp_bounds[0]:ramp_bounds[1]+1],
             I_leak[ramp_bounds[0]:ramp_bounds[1]+1], '--')

    ax4.set_title(
        f'current vs. time (gleak: {np.round(b_1,1)}, Eleak: {np.round(b_0/b_1,1)})')
    ax4.set_xlabel('time (s)')
    ax4.set_ylabel('current (pA)')
    ax4.plot(currents['times'], I, label='I_obs')
    ax4.plot(currents['times'], I_leak, linestyle='--', label='I_leak')
    ax4.plot(currents['times'], I - I_leak,
             linestyle='--', alpha=0.5, label='Ikr')
    ax4.legend()

    plt.tight_layout()
    if save:
        plt.savefig(f"{well}_{sweep}.png")
    else:
        plt.show()
        print(f'gleak: {b_1}, Eleak: {b_0/b_1}')
