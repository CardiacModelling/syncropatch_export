import argparse
import importlib.util
import logging
import multiprocessing
import matplotlib
import os
import string
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regex as re
from matplotlib.gridspec import GridSpec

from pcpostprocess.hergQC import hERGQC
from pcpostprocess.infer_reversal import infer_reversal_potential
from pcpostprocess.leak_correct import detect_ramp_bounds, fit_linear_leak, get_leak_corrected
from pcpostprocess.trace import Trace
from pcpostprocess.voltage_protocols import VoltageProtocol

matplotlib.rc('font', size='9')

pool_kws = {'maxtasksperchild': 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory')
    parser.add_argument('-c', '--no_cpus', default=1, type=int)
    parser.add_argument('--output_dir')
    parser.add_argument('-w', '--wells', nargs='+')
    parser.add_argument('--protocols', nargs='+')
    parser.add_argument('--export_failed', action='store_true')
    parser.add_argument('--selection_file')
    parser.add_argument('--subtracted_only', action='store_true')
    parser.add_argument('--figsize', nargs=2, type=int, default=[5, 8])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_level', default='INFO')
    parser.add_argument('--Erev', default=-90.71, type=float)

    global args
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level)

    if args.output_dir is None:
        args.output_dir = os.path.join('output', 'hergqc')
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    spec = importlib.util.spec_from_file_location(
        'export_config',
        os.path.join(args.data_directory,
                     'export_config.py'))

    if args.wells is None:
        args.wells = [row + str(i).zfill(2) for row in string.ascii_uppercase[:16]
                      for i in range(1, 25)]
        wells = args.wells

    else:
        wells = args.wells

    # Import and exec config file
    global export_config
    export_config = importlib.util.module_from_spec(spec)
    sys.modules['export_config'] = export_config
    spec.loader.exec_module(export_config)

    protocols_regex = \
        r'([a-z|A-Z|_|0-9| |\(|\)]+)_([0-9][0-9]\.[0-9][0-9]\.[0-9][0-9])'

    protocols_regex = re.compile(protocols_regex)

    res_dict = {}
    for dirname in os.listdir(args.data_directory):
        dirname = os.path.basename(dirname)
        match = protocols_regex.match(dirname)

        if match is None:
            continue

        protocol_name = match.group(1)

        if protocol_name not in export_config.D2S\
           and protocol_name not in export_config.D2S_QC:
            continue

        # map name to new name using export_config
        # savename = export_config.D2S[protocol_name]
        time = match.group(2)

        if protocol_name not in res_dict:
            res_dict[protocol_name] = []

        res_dict[protocol_name].append(time)

    readnames, savenames, times_list = [], [], []

    combined_dict = {**export_config.D2S, **export_config.D2S_QC}

    # Select QC protocols and times
    for protocol in res_dict:

        if protocol not in export_config.D2S_QC:
            continue

        times = sorted(res_dict[protocol])

        savename = export_config.D2S_QC[protocol]

        if len(times) == 2:
            savenames.append(savename)
            readnames.append(protocol)
            times_list.append(times)

        elif len(times) == 4:
            savenames.append(savename)
            readnames.append(protocol)
            times_list.append([times[0], times[2]])

            # Make seperate savename for protocol repeat
            savename = combined_dict[protocol] + '_2'
            assert savename not in export_config.D2S.values()
            savenames.append(savename)
            times_list.append([times[1], times[3]])
            readnames.append(protocol)

    with multiprocessing.Pool(min(args.no_cpus, len(readnames)),
                              **pool_kws) as pool:
        well_selections, qc_dfs = \
            list(zip(*pool.starmap(run_qc_for_protocol, zip(readnames,
                                                            savenames,
                                                            times_list))))

    qc_df = pd.concat(qc_dfs, ignore_index=True)

    # Do QC which requires both repeats
    # qc3.bookend check very first and very last staircases are similar
    protocol, savename = list(export_config.D2S_QC.items())[0]
    if len(times) == 4:
        qc3_bookend_dict = qc3_bookend(protocol, savename, times)
    else:
        qc3_bookend_dict = {well: True for well in qc_df.well.unique()}

    qc_df['qc3.bookend'] = [qc3_bookend_dict[well] for well in qc_df.well]

    savedir = os.path.join(args.output_dir, export_config.savedir)
    saveID = export_config.saveID

    if not os.path.exists(os.path.join(args.output_dir, savedir)):
        os.makedirs(os.path.join(args.output_dir, savedir))

    #  qc_df will be updated and saved again, but it's useful to save them here for debugging
    # Write qc_df to file
    qc_df.to_csv(os.path.join(savedir, 'QC-%s.csv' % saveID))

    # Write data to JSON file
    qc_df.to_json(os.path.join(savedir, 'QC-%s.json' % saveID),
                  orient='records')

    # Overwrite old files
    for protocol in list(export_config.D2S_QC.values()):
        fname = os.path.join(savedir, 'selected-%s-%s.txt' % (saveID, protocol))
        with open(fname, 'w') as fout:
            pass

    overall_selection = []
    for well in qc_df.well.unique():
        failed = False
        for well_selection, protocol in zip(well_selections,
                                            list(savenames)):

            logging.debug(f"{well_selection} selected from protocol {protocol}")
            fname = os.path.join(savedir, 'selected-%s-%s.txt' %
                                 (saveID, protocol))
            if well not in well_selection:
                failed = True
            else:
                with open(fname, 'a') as fout:
                    fout.write(well)
                    fout.write('\n')

        # well in every selection
        if not failed:
            overall_selection.append(well)

    selectedfile = os.path.join(savedir, 'selected-%s.txt' % saveID)
    with open(selectedfile, 'w') as fout:
        for well in overall_selection:
            fout.write(well)
            fout.write('\n')

    logfile = os.path.join(savedir, 'table-%s.txt' % saveID)
    with open(logfile, 'a') as f:
        f.write('\\end{table}\n')

    # Export all protocols
    savenames, readnames, times_list = [], [], []
    for protocol in res_dict:

        if args.protocols:
            if savename not in args.protocols:
                continue

        # Sort into chronological order
        times = sorted(res_dict[protocol])
        savename = combined_dict[protocol]

        readnames.append(protocol)

        if len(times) == 2:
            savenames.append(savename)
            times_list.append(times)

        elif len(times) == 4:
            savenames.append(savename)
            times_list.append(times[::2])

            # Make seperate savename for protocol repeat
            savename = combined_dict[protocol] + '_2'
            assert savename not in combined_dict.values()
            savenames.append(savename)
            times_list.append(times[1::2])
            readnames.append(protocol)

    wells_to_export = wells if args.export_failed else overall_selection

    logger.info(f"exporting wells {wells}")

    no_protocols = len(res_dict)

    args_list = list(zip(readnames, savenames, times_list, [wells_to_export] *
                         len(savenames)))

    with multiprocessing.Pool(min(args.no_cpus, no_protocols),
                              **pool_kws) as pool:
        dfs = list(pool.starmap(extract_protocol, args_list))

    extract_df = pd.concat(dfs, ignore_index=True)
    extract_df['selected'] = extract_df['well'].isin(overall_selection)

    logger.info(f"extract_df: {extract_df}")

    qc_erev_spread = {}
    erev_spreads = {}
    passed_qc_dict = {}
    for well in extract_df.well.unique():
        logger.info(f"Checking QC for well {well}")
        # Select only this well
        sub_df = extract_df[extract_df.well == well]
        sub_qc_df = qc_df[qc_df.well == well]

        passed_qc3_bookend = np.all(sub_qc_df['qc3.bookend'].values)
        logger.info(f"passed_QC3_bookend_all {passed_qc3_bookend}")
        passed_QC_Erev_all = np.all(sub_df['QC.Erev'].values)
        passed_QC1_all = np.all(sub_df.QC1.values)
        logger.info(f"passed_QC1_all {passed_QC1_all}")

        passed_QC4_all = np.all(sub_df.QC4.values)
        logger.info(f"passed_QC4_all {passed_QC4_all}")
        passed_QC6_all = np.all(sub_df.QC6.values)
        logger.info(f"passed_QC6_all {passed_QC1_all}")

        E_revs = sub_df['E_rev'].values.flatten().astype(np.float64)
        E_rev_spread = E_revs.max() - E_revs.min()
        # QC Erev spread: check spread in reversal potential isn't too large
        passed_QC_Erev_spread = E_rev_spread <= 5.0
        logger.info(f"passed_QC_Erev_spread {passed_QC_Erev_spread}")

        qc_erev_spread[well] = passed_QC_Erev_spread
        erev_spreads[well] = E_rev_spread

        passed_QC_Erev_all = np.all(sub_df['QC.Erev'].values)
        logger.info(f"passed_QC_Erev_all {passed_QC_Erev_all}")

        was_selected = np.all(sub_df['selected'].values)

        passed_qc = passed_qc3_bookend and was_selected\
            and passed_QC_Erev_all and passed_QC6_all\
            and passed_QC_Erev_spread and passed_QC1_all\
            and passed_QC4_all

        passed_qc_dict[well] = passed_qc

    extract_df['passed QC'] = [passed_qc_dict[well] for well in extract_df.well]
    extract_df['QC.Erev.spread'] = [qc_erev_spread[well] for well in extract_df.well]
    extract_df['Erev_spread'] = [erev_spreads[well] for well in extract_df.well]

    #  Update qc_df
    update_cols = []
    for index, vals in qc_df.iterrows():
        append_dict = {}

        sub_df = extract_df[(extract_df.well == well) &
                            (extract_df.protocol == protocol)]

        append_dict['QC.Erev.all_protocols'] =\
            np.all(sub_df['QC.Erev'])

        append_dict['QC.Erev.spread'] =\
            np.all(sub_df['QC.Erev.spread'])

        append_dict['QC1.all_protocols'] =\
            np.all(sub_df['QC1'])

        append_dict['QC4.all_protocols'] =\
            np.all(sub_df['QC4'])

        append_dict['QC6.all_protocols'] =\
            np.all(sub_df['QC6'])

        update_cols.append(append_dict)

    for key, val in append_dict.items():
        qc_df[key] = [row[key] for row in update_cols]

    # Save in csv format
    qc_df.to_csv(os.path.join(savedir, 'QC-%s.csv' % saveID))

    # Write data to JSON file
    qc_df.to_json(os.path.join(savedir, 'QC-%s.json' % saveID),
                  orient='records')

    #  Load only QC vals. TODO use a new variabile name to avoid confusion
    qc_df['drug'] = 'before'
    qc_df = extract_df[['well', 'sweep', 'protocol', 'Rseal', 'Cm', 'Rseries']]
    qc_df.to_csv(os.path.join(args.output_dir, 'qc_vals_df.csv'))

    extract_df.to_csv(os.path.join(args.output_dir, 'subtraction_qc.csv'))

    with open(os.path.join(args.output_dir, 'passed_wells.txt'), 'w') as fout:
        for well, passed in passed_qc_dict.items():
            if passed:
                fout.write(well)
                fout.write('\n')


def extract_protocol(readname, savename, time_strs, selected_wells):
    logger.info(f"extracting {savename}")
    savedir = os.path.join(args.output_dir, export_config.savedir)

    saveID = export_config.saveID
    traces_dir = os.path.join(savedir, 'traces')

    if not os.path.exists(traces_dir):
        os.makedirs(traces_dir)

    row_dict = {}

    subtraction_plots_dir = os.path.join(savedir, 'subtraction_plots')

    if not os.path.isdir(subtraction_plots_dir):
        os.makedirs(subtraction_plots_dir)

    logger.info(f"Exporting {readname} as {savename}")

    filepath_before = os.path.join(args.data_directory,
                                   f"{readname}_{time_strs[0]}")
    filepath_after = os.path.join(args.data_directory,
                                  f"{readname}_{time_strs[1]}")
    json_file_before = f"{readname}_{time_strs[0]}"
    json_file_after = f"{readname}_{time_strs[1]}"
    before_trace = Trace(filepath_before,
                         json_file_before)
    after_trace = Trace(filepath_after,
                        json_file_after)

    voltage_protocol = before_trace.get_voltage_protocol()
    t = before_trace.get_times()
    tstart, tend = voltage_protocol.get_ramps()[0][:2]
    ramp_bounds = [np.argmax(t > tstart), np.argmax(t > tend)]

    nsweeps_before = before_trace.NofSweeps = 2
    nsweeps_after = after_trace.NofSweeps = 2

    assert nsweeps_before == nsweeps_after

    # Time points
    times_before = before_trace.get_times()
    times_after = after_trace.get_times()

    try:
        assert all(np.abs(times_before - times_after) < 1e-8)
    except Exception as exc:
        logger.warning(f"Exception thrown when handling {savename}: ", str(exc))
        return

    header = "\"current\""

    qc_before = before_trace.get_onboard_QC_values()
    qc_after = after_trace.get_onboard_QC_values()
    qc_vals_all = before_trace.get_onboard_QC_values()

    for i_well, well in enumerate(selected_wells):  # Go through all wells
        if i_well % 24 == 0:
            logger.info('row ' + well[0])

        if args.selection_file:
            if well not in selected_wells:
                continue

        if None in qc_before[well] or None in qc_after[well]:
            continue

        # Save 'before drug' trace as .csv
        for sweep in range(nsweeps_before):
            out = before_trace.get_trace_sweeps([sweep])[well][0]
            save_fname = os.path.join(traces_dir, f"{saveID}-{savename}-"
                                      f"{well}-before-sweep{sweep}")

            np.savetxt(save_fname, out, delimiter=',',
                       header=header)

        # Save 'after drug' trace as .csv
        for sweep in range(nsweeps_after):
            save_fname = os.path.join(traces_dir, f"{saveID}-{savename}-"
                                      f"{well}-after-sweep{sweep}")
            out = after_trace.get_trace_sweeps([sweep])[well][0]
            if len(out) > 0:
                np.savetxt(save_fname, out,
                           delimiter=',', comments='', header=header)

    voltage_before = before_trace.get_voltage()
    voltage_after = after_trace.get_voltage()

    assert len(voltage_before) == len(voltage_after)
    assert len(voltage_before) == len(times_before)
    assert len(voltage_after) == len(times_after)
    voltage = voltage_before

    voltage_df = pd.DataFrame(np.vstack((times_before.flatten(),
                                         voltage.flatten())).T,
                              columns=['time', 'voltage'])

    if not os.path.exists(os.path.join(traces_dir,
                                       f"{saveID}-{savename}-voltages.csv")):
        voltage_df.to_csv(os.path.join(traces_dir,
                                       f"{saveID}-{savename}-voltages.csv"))

    np.savetxt(os.path.join(traces_dir, f"{saveID}-{savename}-times.txt"),
               times_before)

    # plot subtraction
    fig = plt.figure(figsize=args.figsize, layout='constrained')

    reversal_plot_dir = os.path.join(savedir, 'reversal_plots')

    rows = []

    before_leak_current_dict = {}
    after_leak_current_dict = {}

    for well in selected_wells:
        before_current = before_trace.get_trace_sweeps()[well]
        after_current = after_trace.get_trace_sweeps()[well]

        before_leak_currents = []
        after_leak_currents = []

        out_dir = os.path.join(savedir,
                               f"{saveID}-{savename}-leak_fit-before")

        for sweep in range(before_current.shape[0]):
            row_dict = {
                'well': well,
                'sweep': sweep,
                'protocol': savename
            }

            qc_vals = qc_vals_all[well][sweep]
            if qc_vals is None:
                continue
            if len(qc_vals) == 0:
                continue

            row_dict['Rseal'] = qc_vals[0]
            row_dict['Cm'] = qc_vals[1]
            row_dict['Rseries'] = qc_vals[2]

            before_params, before_leak = fit_linear_leak(before_trace,
                                                         well, sweep,
                                                         ramp_bounds,
                                                         plot=True,
                                                         output_dir=out_dir)

            before_leak_currents.append(before_leak)

            out_dir = os.path.join(savedir,
                                   f"{saveID}-{savename}-leak_fit-after")
            # Convert linear regression parameters into conductance and reversal
            row_dict['gleak_before'] = before_params[1]
            row_dict['E_leak_before'] = -before_params[0] / before_params[1]

            after_params, after_leak = fit_linear_leak(before_trace,
                                                       well, sweep,
                                                       ramp_bounds,
                                                       plot=True,
                                                       output_dir=out_dir)

            after_leak_currents.append(after_leak)

            # Convert linear regression parameters into conductance and reversal
            row_dict['gleak_after'] = after_params[1]
            row_dict['E_leak_after'] = -after_params[0] / after_params[1]

            subtracted_trace = before_current[sweep, :] - before_leak\
                - (after_current[sweep, :] - after_leak)
            out_fname = os.path.join(savedir,
                                     f"{saveID}-{savename}-{well}-sweep{sweep}-subtracted.csv")
            after_corrected = after_current[sweep, :] - after_leak
            before_corrected = before_current[sweep, :] - before_leak

            E_rev_before = infer_reversal_potential(before_trace, sweep, well,
                                                    plot=True,
                                                    output_path=os.path.join(reversal_plot_dir,
                                                                             f"{well}_{savename}_sweep{sweep}_before"),
                                                    known_Erev=args.Erev)

            E_rev_after = infer_reversal_potential(after_trace, sweep, well,
                                                   plot=True,
                                                   output_path=os.path.join(reversal_plot_dir,
                                                                            f"{well}_{savename}_sweep{sweep}_after"),
                                                   known_Erev=args.Erev)

            E_rev = infer_reversal_potential(before_trace, sweep, well,
                                             plot=True,
                                             output_path=os.path.join(reversal_plot_dir,
                                                                      f"{well}_{savename}_sweep{sweep}_subtracted"),
                                             known_Erev=args.Erev,
                                             current=subtracted_trace)

            row_dict['R_leftover'] =\
                np.sqrt(np.sum(after_corrected**2)/(np.sum(before_corrected**2)))

            row_dict['E_rev'] = E_rev
            row_dict['E_rev_before'] = E_rev_before
            row_dict['E_rev_after'] = E_rev_after

            row_dict['QC.Erev'] = E_rev < -50 and E_rev > -120

            # Check QC6 for each protocol (not just the staircase)
            plot_dir = os.path.join(savedir, 'debug')

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            hergqc = hERGQC(sampling_rate=before_trace.sampling_rate,
                            plot_dir=plot_dir,
                            n_sweeps=before_trace.NofSweeps)

            row_dict['QC6'] = hergqc.qc6(subtracted_trace,
                                         win=hergqc.qc6_win,
                                         label='0')

            #  Assume there is only one sweep for all non-QC protocols
            rseal_before, cm_before, rseries_before = qc_before[well][0]
            rseal_after, cm_after, rseries_after = qc_after[well][0]

            row_dict['QC1'] = all(list(hergqc.qc1(rseal_before, cm_before, rseries_before)) +
                                  list(hergqc.qc1(rseal_after, cm_after, rseries_after)))

            row_dict['QC4'] = all(hergqc.qc4([rseal_before, rseal_after],
                                             [cm_before, cm_after],
                                             [rseries_before, rseries_after]))

            np.savetxt(out_fname, subtracted_trace.flatten())
            rows.append(row_dict)

        before_leak_current_dict[well] = np.vstack(before_leak_currents)
        after_leak_current_dict[well] = np.vstack(after_leak_currents)

    extract_df = pd.DataFrame.from_dict(rows)
    logger.debug(extract_df)

    times = before_trace.get_times()
    voltages = before_trace.get_voltage()

    before_current_all = before_trace.get_trace_sweeps()
    after_current_all = after_trace.get_trace_sweeps()

    # Convert everything to nA...
    before_current_all = {key: value * 1e-3 for key, value in before_current_all.items()}
    after_current_all = {key: value * 1e-3 for key, value in after_current_all.items()}

    before_leak_current_dict = {key: value * 1e-3 for key, value in before_leak_current_dict.items()}
    after_leak_current_dict = {key: value * 1e-3 for key, value in after_leak_current_dict.items()}

    # TODO Put this code in a seperate function so we can easily plot individual subtractions
    for well in selected_wells:
        before_current = before_current_all[well]
        after_current = after_current_all[well]

        before_leak_currents = before_leak_current_dict[well]
        after_leak_currents = after_leak_current_dict[well]

        nsweeps = before_current_all[well].shape[0]

        axs = setup_subtraction_grid(fig, nsweeps)
        protocol_axs, before_axs, after_axs, \
            corrected_axs, subtracted_ax, \
            long_protocol_ax = axs

        sub_df = extract_df[extract_df.well == well]
        sweeps = sorted(list(sub_df.sweep.unique()))
        sub_df = sub_df.set_index('sweep')
        logger.debug(sub_df)

        axs = protocol_axs + before_axs + after_axs + corrected_axs + \
            [subtracted_ax, long_protocol_ax]

        times = before_trace.get_times() * 1e-3

        for ax in protocol_axs:
            ax.plot(times, voltages, color='black')
            ax.set_xlabel('time (s)')
            ax.set_ylabel(r'$V_\text{command}$ (mV)')

        for i, (sweep, ax) in enumerate(zip(sweeps, before_axs)):
            gleak, Eleak = sub_df.loc[sweep][['gleak_before', 'E_leak_before']]
            ax.plot(times, before_current[i, :], label=f"pre-drug raw, sweep {sweep}")
            ax.plot(times, before_leak_currents[i, :],
                    label=r'$I_\text{leak}$.' f"g={gleak:1E}, E={Eleak:.1e}")
            # ax.legend()

            if ax.get_legend():
                ax.get_legend().remove()
            ax.set_xlabel('time (s)')
            ax.set_ylabel(r'pre-drug trace')
            # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            # ax.tick_params(axis='y', rotation=90)

        for i, (sweep, ax) in enumerate(zip(sweeps, after_axs)):
            gleak, Eleak = sub_df.loc[sweep][['gleak_after', 'E_leak_after']]
            ax.plot(times, after_current[i, :], label=f"post-drug raw, sweep {sweep}")
            ax.plot(times, after_leak_currents[i, :],
                    label=r"$I_\text{leak}$." f"g={gleak:1E}, E={Eleak:.1e}")
            # ax.legend()
            if ax.get_legend():
                ax.get_legend().remove()
            ax.set_xlabel('time (s)')
            ax.set_ylabel(r'post-drug trace')
            # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            # ax.tick_params(axis='y', rotation=90)

        for i, (sweep, ax) in enumerate(zip(sweeps, corrected_axs)):
            corrected_current = before_current[i, :] - before_leak_currents[i, :]
            corrected_after_current = after_current[i, :] - after_leak_currents[i, :]
            ax.plot(times, corrected_current,
                    label=f"leak corrected before drug trace, sweep {sweep}")
            ax.plot(times, corrected_after_current,
                    label=f"leak corrected after drug trace, sweep {sweep}")
            ax.set_xlabel('time (s)')
            ax.set_ylabel(r'leak corrected traces')
            # ax.tick_params(axis='y', rotation=90)
            # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

        ax = subtracted_ax
        for i, sweep in enumerate(sweeps):
            before_params, before_leak = fit_linear_leak(before_trace,
                                                         well, sweep,
                                                         ramp_bounds)
            after_params, after_leak = fit_linear_leak(after_trace,
                                                       well, sweep,
                                                       ramp_bounds)

            subtracted_current = before_current[i, :] - before_leak_currents[i, :] - \
                (after_current[i, :] - after_leak_currents[i, :])
            ax.plot(times, subtracted_current, label=f"sweep {sweep}")
            ax.set_ylabel(r'$I_\text{obs, subtracted}$ (mV)')
            ax.set_xlabel('time (s)')
            # ax.tick_params(axis='x', rotation=90)

        long_protocol_ax.plot(times, voltages, color='black')
        long_protocol_ax.set_xlabel('time (s)')
        long_protocol_ax.set_ylabel(r'$V_\text{command}$ (mV)')
        long_protocol_ax.tick_params(axis='y', rotation=90)

        fig.savefig(os.path.join(subtraction_plots_dir,
                                 f"{saveID}-{savename}-{well}-sweep{sweep}-subtraction"))
        fig.clf()

    plt.close(fig)

    # extract protocol
    before_trace.get_voltage_protocol().export_txt(os.path.join(savedir,
                                                                f"{saveID}-{savename}.txt"))

    return extract_df


def run_qc_for_protocol(readname, savename, time_strs):
    df_rows = []

    assert len(time_strs) == 2

    filepath_before = os.path.join(args.data_directory,
                                   f"{readname}_{time_strs[0]}")
    json_file_before = f"{readname}_{time_strs[0]}"

    filepath_after = os.path.join(args.data_directory,
                                  f"{readname}_{time_strs[1]}")
    json_file_after = f"{readname}_{time_strs[1]}"

    logging.debug(f"loading {json_file_after} and {json_file_before}")

    before_trace = Trace(filepath_before,
                         json_file_before)

    after_trace = Trace(filepath_after,
                        json_file_after)

    assert before_trace.sampling_rate == after_trace.sampling_rate

    # Convert to s
    sampling_rate = before_trace.sampling_rate

    savedir = os.path.join(args.output_dir, export_config.savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    before_voltage = before_trace.get_voltage()
    after_voltage = after_trace.get_voltage()

    # Assert that protocols are exactly the same
    assert np.all(before_voltage == after_voltage)

    voltage = before_voltage

    sweeps = [0, 1]
    raw_before_all = before_trace.get_trace_sweeps(sweeps)
    raw_after_all = after_trace.get_trace_sweeps(sweeps)

    selected_wells = []
    for well in args.wells:

        plot_dir = os.path.join(savedir, "debug", f"debug_{well}_{savename}")

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Setup QC instance. We could probably just do this inside the loop
        hergqc = hERGQC(sampling_rate=sampling_rate,
                        plot_dir=plot_dir,
                        voltage=before_voltage)

        qc_before = before_trace.get_onboard_QC_values()
        qc_after = after_trace.get_onboard_QC_values()

        # Check if any cell first!
        if (None in qc_before[well][0]) or (None in qc_after[well][0]):
            # no_cell = True
            continue

        else:
            # no_cell = False
            pass

        nsweeps = before_trace.NofSweeps
        assert after_trace.NofSweeps == nsweeps

        before_currents = np.empty((nsweeps, before_trace.NofSamples))
        after_currents = np.empty((nsweeps, after_trace.NofSamples))

        # Get ramp times from protocol description
        voltage_protocol = VoltageProtocol.from_voltage_trace(voltage,
                                                              before_trace.get_times())

        # Get first ramp
        tstart, tend = voltage_protocol.get_ramps()[0][:2]
        t = before_trace.get_times()

        ramp_bounds = [np.argmax(t > tstart), np.argmax(t > tend)]

        assert after_trace.NofSamples == before_trace.NofSamples

        for sweep in range(nsweeps):
            before_params1, before_leak = fit_linear_leak(before_trace, well,
                                                          sweep, ramp_bounds,
                                                          plot=True,
                                                          label=f"{savename}-before",
                                                          output_dir=savedir)

            after_params1, after_leak = fit_linear_leak(after_trace, well,
                                                        sweep, ramp_bounds,
                                                        plot=True,
                                                        label=f"{savename}-after",
                                                        output_dir=savedir)

            before_raw = np.array(raw_before_all[well])[sweep, :]
            after_raw = np.array(raw_after_all[well])[sweep, :]

            before_currents[sweep, :] = before_raw - before_leak
            after_currents[sweep, :] = after_raw - after_leak

        # TODO Note: only run this for whole/staircaseramp for now...
        logger.info(f"{well} {savename}\n----------")
        logger.info(f"sampling_rate is {sampling_rate}")

        # Run QC with leak subtracted currents
        selected, QC = hergqc.run_qc(before_currents,
                                     after_currents,
                                     np.array(qc_before[well])[0, :],
                                     np.array(qc_after[well])[0, :],
                                     nsweeps)

        df_rows.append([well] + list(QC))

        if selected:
            selected_wells.append(well)

        # Save subtracted current in csv file
        header = "\"current\""

        for i in range(nsweeps):

            savepath = os.path.join(savedir,
                                    f"{export_config.saveID}-{savename}-{well}-sweep{i}.csv")
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            subtracted_current = before_currents[0, :] - after_currents[0, :]
            np.savetxt(savepath, subtracted_current, delimiter=',',
                       comments='', header=header)

    column_labels = ['well', 'qc1.rseal', 'qc1.cm', 'qc1.rseries', 'qc2.raw',
                     'qc2.subtracted', 'qc3.raw', 'qc3.E4031', 'qc3.subtracted',
                     'qc4.rseal', 'qc4.cm', 'qc4.rseries', 'qc5.staircase',
                     'qc5.1.staircase', 'qc6.subtracted', 'qc6.1.subtracted',
                     'qc6.2.subtracted']

    df = pd.DataFrame(np.array(df_rows), columns=column_labels)

    # Add onboard qc to dataframe
    for well in args.wells:
        if well not in df['well'].values:
            onboard_qc_df = pd.DataFrame([[well] + [False for col in
                                                    list(df)[1:]]],
                                         columns=list(df))
            df = pd.concat([df, onboard_qc_df], ignore_index=True)

    df['protocol'] = savename

    return selected_wells, df


def qc3_bookend(readname, savename, time_strs):
    #  TODO Run this with subtracted traces
    plot_dir = os.path.join(args.output_dir, export_config.savedir,
                            f"{export_config.saveID}-{savename}-qc3-bookend")

    filepath_first_before = os.path.join(args.data_directory,
                                         f"{readname}_{time_strs[0]}")
    filepath_last_before = os.path.join(args.data_directory,
                                        f"{readname}_{time_strs[1]}")
    json_file_first_before = f"{readname}_{time_strs[0]}"
    json_file_last_before = f"{readname}_{time_strs[1]}"

    first_before_trace = Trace(filepath_first_before,
                               json_file_first_before)
    last_before_trace = Trace(filepath_last_before,
                              json_file_last_before)

    voltage_protocol = first_before_trace.get_voltage_protocol()
    ramp_bounds = detect_ramp_bounds(first_before_trace,
                                     voltage_protocol)

    before_traces_first = get_leak_corrected(first_before_trace,
                                             ramp_bounds)
    before_traces_last = get_leak_corrected(last_before_trace,
                                            ramp_bounds)

    filepath_first_after = os.path.join(args.data_directory,
                                        f"{readname}_{time_strs[2]}")
    filepath_last_after = os.path.join(args.data_directory,
                                       f"{readname}_{time_strs[3]}")
    json_file_first_after = f"{readname}_{time_strs[2]}"
    json_file_last_after = f"{readname}_{time_strs[3]}"

    first_after_trace = Trace(filepath_first_after,
                              json_file_first_after)
    last_after_trace = Trace(filepath_last_after,
                             json_file_last_after)

    after_traces_first = get_leak_corrected(first_after_trace,
                                            ramp_bounds)
    after_traces_last = get_leak_corrected(last_after_trace,
                                           ramp_bounds)

    assert np.all(first_before_trace.get_voltage() == last_before_trace.get_voltage())
    assert np.all(first_after_trace.get_voltage() == last_after_trace.get_voltage())
    assert np.all(first_before_trace.get_voltage() == first_after_trace.get_voltage())

    first_processed = {well: before_traces_first[well] - after_traces_first[well]
                       for well in before_traces_first}

    last_processed = {well: before_traces_last[well] - after_traces_last[well]
                      for well in after_traces_first}

    assert np.all(first_before_trace.get_voltage() == last_before_trace.get_voltage())

    voltage = first_before_trace.get_voltage()

    hergqc = hERGQC(sampling_rate=first_before_trace.sampling_rate,
                    plot_dir=plot_dir,
                    voltage=voltage)

    assert first_before_trace.NofSweeps == last_before_trace.NofSweeps

    # first_trace_sweeps = first_before_trace.get_trace_sweeps()
    # last_trace_sweeps = last_before_trace.get_trace_sweeps()
    res_dict = {}
    for well in args.wells:
        passed = hergqc.qc3(first_processed[well][0, :],
                            last_processed[well][-1, :])
        res_dict[well] = passed

    return res_dict


def setup_subtraction_grid(fig, nsweeps):
    # Use 5 x 2 grid
    gs = GridSpec(6, nsweeps, figure=fig)

    # plot protocol at the top
    protocol_axs = [fig.add_subplot(gs[0, i]) for i in range(nsweeps)]

    # Plot before drug traces
    before_axs = [fig.add_subplot(gs[1, i]) for i in range(nsweeps)]

    # Plot after traces
    after_axs = [fig.add_subplot(gs[2, i]) for i in range(nsweeps)]

    # Leak corrected traces
    corrected_axs = [fig.add_subplot(gs[3, i]) for i in range(nsweeps)]

    # Subtracted traces on one axis
    subtracted_ax = fig.add_subplot(gs[4, :])

    # Long axis for protocol on the bottom (full width)
    long_protocol_ax = fig.add_subplot(gs[5, :])

    return protocol_axs, before_axs, after_axs, corrected_axs, subtracted_ax, long_protocol_ax


if __name__ == '__main__':
    main()
