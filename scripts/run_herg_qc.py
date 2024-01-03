import argparse
import importlib.util
import logging
import multiprocessing
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
from pcpostprocess.leak_correct import fit_linear_leak
from pcpostprocess.trace import Trace
from pcpostprocess.voltage_protocols import VoltageProtocol

global wells
wells = [row + str(i).zfill(2) for row in string.ascii_uppercase[:16] for i in
         range(1, 25)]

pool_kws = {'maxtasksperchild': 16}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory')
    parser.add_argument('-c', '--no_cpus', default=1, type=int)
    parser.add_argument('--output_dir')
    parser.add_argument('-w', '--wells', nargs='+')
    parser.add_argument('--export_failed', action='store_true')
    parser.add_argument('--selection_file')
    parser.add_argument('--subtracted_only', action='store_true')
    parser.add_argument('--figsize', nargs=2, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--Erev', default=-90.71, type=float)

    global args
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig()
        logging.getLogger().setLevel(logging.INFO)

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

    # Select QC protocols and times
    for protocol in res_dict:

        if protocol not in export_config.D2S_QC:
            continue

        times = res_dict[protocol]
        savename = export_config.D2S_QC[protocol]

        readnames.append(protocol)

        if len(times) == 2:
            savenames.append(savename)
            times_list.append(times)

        elif len(times) == 4:
            savenames.append(savename)
            times_list.append(times[:2])

            # Make seperate savename for protocol repeat
            savename = export_config.D2S[protocol] + '_2'
            assert savename not in export_config.D2S.values()
            savenames.append(savename)
            times_list.append(times[2:])
            readnames.append(protocol)

    with multiprocessing.Pool(min(args.no_cpus, len(export_config.D2S_QC)),
                              **pool_kws) as pool:
        well_selections, qc_dfs = \
            list(zip(*pool.starmap(run_qc_for_protocol, zip(readnames,
                                                            savenames,
                                                            times_list))))

    # Do QC which requires both repeats
    # qc3.bookend check very first and very last staircases are similar
    qc3_bookend_res = []
    for protocol, times in res_dict.items():
        if protocol in export_config.D2S_QC:
            savename = export_config.D2S_QC[protocol]
        else:
            continue

        if len(times) == 4:
            res = qc3_bookend(protocol, savename, times)
        else:
            res = True
        qc3_bookend_res.append(res)

    qc3_df = pd.concat(qc3_bookend_res, ignore_index=True)

    qc_df = pd.concat(qc_dfs, ignore_index=True)

    qc_df = qc_df.set_index(['well', 'protocol'])
    qc_df = qc_df.join(qc3_df.set_index(['well', 'protocol']))
    qc_df = qc_df.reset_index()

    savedir = os.path.join(args.output_dir, export_config.savedir)
    saveID = export_config.saveID

    if not os.path.exists(os.path.join(args.output_dir, savedir)):
        os.makedirs(os.path.join(args.output_dir, savedir))

    # Write qc_df to file
    qc_df.to_csv(os.path.join(args.output_dir,
                              savedir, 'QC-%s.csv' % saveID))

    # Write data to JSON file
    qc_df.to_json(os.path.join(args.output_dir, savedir, 'QC-%s.json' % saveID),
                  orient='records')

    # Overwrite old files
    for protocol in list(export_config.D2S_QC.values()):
        fname = os.path.join(savedir, 'selected-%s-%s.txt' % (saveID, protocol))
        with open(fname, 'w') as fout:
            pass

    overall_selection = []
    for well in args.wells:
        failed = False
        for well_selection, protocol in zip(well_selections,
                                            list(export_config.D2S_QC.values())):

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
        fout.write('# Based on QC on protocol \'whole\' for each cell.\n')
        for well in overall_selection:
            fout.write(well)
            fout.write('\n')

    logfile = os.path.join(savedir, 'table-%s.txt' % saveID)
    with open(logfile, 'a') as f:
        f.write('\\end{table}\n')

    # Export all protocols
    savenames, readnames, times_list = [], [], []
    for protocol in res_dict:
        times = res_dict[protocol]
        savename = export_config.D2S[protocol]

        readnames.append(protocol)

        if len(times) == 2:
            savenames.append(savename)
            times_list.append(times)

        elif len(times) == 4:
            savenames.append(savename)
            times_list.append(times[:2])

            # Make seperate savename for protocol repeat
            savename = export_config.D2S[protocol] + '_2'
            assert savename not in export_config.D2S.values()
            savenames.append(savename)
            times_list.append(times[2:])
            readnames.append(protocol)

    wells_to_export = wells if args.export_failed else overall_selection

    with multiprocessing.Pool(min(args.no_cpus, len(export_config.D2S_QC)),
                              **pool_kws) as pool:
        dfs = list(pool.starmap(extract_protocol, zip(readnames,
                                                      savenames,
                                                      times_list,
                                                      [wells_to_export]
                                                      * len(savenames)))
                   )
    extract_df = pd.concat(dfs, ignore_index=True)

    passed_qc_dict = {}
    for well in extract_df.well.unique():
        sub_df = extract_df[extract_df.well == well]
        passed_qc3_bookend = \
            np.all(qc_df.set_index(['well']).loc[well]['qc3.bookend'])
        passed_QC_Erev_all = np.all(sub_df['QC.Erev'].values)
        passed_QC6_all = np.all(sub_df.QC6.values)

        # QC Erev spread: check spread in reversal potential isn't too large
        passed_QC_Erev_spread = (sub_df['E_rev'].values.max() -
                                 sub_df['E_rev'].values.min()) <= 5

        passed_QC_Erev_all = np.all(sub_df['QC.Erev'].values)

        was_selected = well in overall_selection

        passed_qc = passed_qc3_bookend and was_selected\
            and passed_QC_Erev_all and passed_QC6_all\
            and passed_QC_Erev_spread

        passed_qc_dict[well] = passed_qc

    extract_df['passed QC'] = [passed_qc_dict[well] for well in extract_df.well]
    extract_df.to_csv(os.path.join(args.output_dir, 'subtraction_qc.csv'))


def extract_protocol(readname, savename, time_strs, selected_wells):
    savedir = os.path.join(args.output_dir, export_config.savedir)

    saveID = export_config.saveID

    row_dict = {}

    plot_dir = os.path.join(args.output_dir, savedir,
                            "{saveID}-{savename}-plot")

    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    logging.info(f"Exporting {readname} as {savename}")

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

    voltage_protocol = before_trace.get_protocol_description()
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
        logging.warning(f"Exception thrown when handling {savename}: ", str(exc))
        return

    header = "\"current\""

    qc_before = before_trace.get_onboard_QC_values()
    qc_after = after_trace.get_onboard_QC_values()

    qc_vals_all = before_trace.get_onboard_QC_values()

    for i_well, well in enumerate(selected_wells):  # Go through all wells
        if i_well % 24 == 0:
            logging.info('row ' + well[0])

        if args.selection_file:
            if well not in selected_wells:
                continue

        if None in qc_before[well] or None in qc_after[well]:
            continue

        # Save 'before drug' trace as .csv
        for sweep in range(nsweeps_before):
            out = before_trace.get_trace_sweeps([sweep])[well][0]
            save_fname = os.path.join(savedir, f"{saveID}-{savename}-"
                                      f"{well}-before-sweep{sweep}")

            np.savetxt(save_fname, out, delimiter=',',
                       header=header)

        # Save 'after drug' trace as .csv
        for sweep in range(nsweeps_after):
            save_fname = os.path.join(savedir, f"{saveID}-{savename}-"
                                      f"{well}-after-sweep{sweep}")
            out = after_trace.get_trace_sweeps([sweep])[well][0]
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

    if not os.path.exists(os.path.join(savedir,
                                       f"{saveID}-{savename}-voltages.csv")):
        voltage_df.to_csv(os.path.join(savedir,
                                       f"{saveID}-{savename}-voltages.csv"))

    np.savetxt(os.path.join(savedir, f"{saveID}-{savename}-times.txt"),
               times_before)

    # plot subtraction
    fig = plt.figure(figsize=args.figsize)

    reversal_plot_dir = os.path.join(savedir, 'reversal_plots')

    rows = []
    for well in selected_wells:
        before_current = before_trace.get_trace_sweeps()[well]
        after_current = after_trace.get_trace_sweeps()[well]
        out_dir = os.path.join(savedir,
                               f"{saveID}-{savename}-leak_fit-before")

        for sweep in list(range(before_trace.NofSweeps)):
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
                                                                      f"{well}_{savename}_sweep{sweep}_after"),
                                             known_Erev=args.Erev,
                                             current=subtracted_trace)

            row_dict['R_leftover'] =\
                np.sqrt(np.sum(after_corrected**2)/(np.sum(before_corrected**2)))

            row_dict['E_rev'] = E_rev
            row_dict['E_rev_before'] = E_rev_before
            row_dict['E_rev_after'] = E_rev_after

            row_dict['QC.Erev'] = E_rev < -50 and E_rev > -120

            # Get indices of first step up to +40mV
            protocol_description = before_trace.get_protocol_description()
            steps_40 = [step for step in protocol_description.get_all_sections()
                        if step[2] == 40.0 and step[3] == 40.0]
            # print(protocol_description.get_all_sections())
            # print(steps_40)
            assert len(steps_40) >= 2
            tstart, tend = steps_40[0][:2]
            times = before_trace.get_times()
            istart = np.argmax(times > tstart)
            iend = np.argmax(times > tstart)

            # Check QC6 for each protocol (not just the staircase)
            plot_dir = os.path.join(savedir, 'debug')

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            hergqc = hERGQC(sampling_rate=before_trace.sampling_rate,
                            plot_dir=plot_dir,
                            n_sweeps=before_trace.NofSweeps)

            row_dict['QC6'] = hergqc.qc6(subtracted_trace,
                                         win=[istart, iend])

            np.savetxt(out_fname, subtracted_trace.flatten())
            rows.append(row_dict)

    extract_df = pd.DataFrame.from_dict(rows)
    print(extract_df)

    nsweeps = before_trace.NofSweeps

    axs = setup_subtraction_grid(fig, nsweeps)
    protocol_axs, before_axs, after_axs, \
        corrected_axs, subtracted_ax, \
        long_protocol_ax = axs

    axs = protocol_axs + before_axs + after_axs + corrected_axs + \
        [subtracted_ax, long_protocol_ax]

    times = before_trace.get_times()
    voltages = before_trace.get_voltage()

    for well in selected_wells:
        for ax in protocol_axs:
            ax.plot(times, voltages, color='black')
            ax.set_xlabel('time (ms)')
            ax.set_ylabel(r'$V_\text{command}$ (mV)')

        for ax in before_axs:
            for sweep in range(nsweeps):
                ax.plot(times, before_current[sweep, :], label=f"sweep {sweep}")
                ax.set_xlabel('time (ms)')
                ax.set_ylabel(r'$I_\text{obs, before}$ (mV)')

        for ax in after_axs:
            for sweep in range(nsweeps):
                ax.plot(times, after_current[sweep, :], label=f"sweep {sweep}")
                ax.set_xlabel('time (ms)')
                ax.set_ylabel(r'$I_\text{obs, after}$ (mV)')

        for ax in corrected_axs:
            for sweep in range(nsweeps):
                before_params, before_leak = fit_linear_leak(before_trace,
                                                             well, sweep,
                                                             ramp_bounds)
                corrected_current = before_current[sweep, :] - before_leak
                ax.plot(times, corrected_current, label=f"sweep {sweep}")
                ax.set_xlabel('time (ms)')
                ax.set_ylabel(r'$I_\text{obs, corrected}$ (mV)')

        ax = subtracted_ax
        for sweep in range(nsweeps):
            before_params, before_leak = fit_linear_leak(before_trace,
                                                         well, sweep,
                                                         ramp_bounds)
            after_params, after_leak = fit_linear_leak(after_trace,
                                                       well, sweep,
                                                       ramp_bounds)

            subtracted_current = before_current[sweep, :] - before_leak - \
                (after_current[sweep, :])
            ax.plot(times, subtracted_current, label=f"sweep {sweep}")
            ax.set_ylabel(r'$I_\text{obs, subtracted}$ (mV)')
            ax.set_xlabel('time (ms)')

        long_protocol_ax.plot(times, voltages, color='black')
        long_protocol_ax.set_xlabel('time (ms)')
        long_protocol_ax.set_ylabel(r'$V_\text{command}$ (mV)')

        fig.savefig(os.path.join(savedir,
                                 f"{saveID}-{savename}-{well}-sweep{sweep}-subtraction"))
        for ax in axs:
            ax.cla()

    plt.close(fig)
    return extract_df


def run_qc_for_protocol(readname, savename, time_strs):
    selected_cells = []
    df_rows = []

    assert len(time_strs) == 2

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

    assert before_trace.sampling_rate == after_trace.sampling_rate

    # Convert to s
    sampling_rate = before_trace.sampling_rate * 1e-3

    plot_dir = args.output_dir
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    before_voltage = before_trace.get_voltage()
    after_voltage = after_trace.get_voltage()

    # Assert that protocols are exactly the same
    assert np.all(before_voltage == after_voltage)

    voltage = before_voltage

    # Setup QC instance. We could probably just do this inside the loop
    hergqc = hERGQC(sampling_rate=sampling_rate,
                    plot_dir=plot_dir,
                    voltage=before_voltage)

    sweeps = [0, 1]
    raw_before_all = before_trace.get_trace_sweeps(sweeps)
    raw_after_all = before_trace.get_trace_sweeps(sweeps)

    # Iterate over cells. Reuse QC instance
    for well in args.wells:
        qc_before = before_trace.get_onboard_QC_values()
        qc_after = after_trace.get_onboard_QC_values()

        # Check if any cell first!
        if (None in qc_before[well][0]) or (None in qc_after[well][0]):
            no_cell = True
            # continue

        else:
            no_cell = False

        nsweeps = before_trace.NofSweeps
        assert after_trace.NofSweeps == nsweeps

        before_currents = np.empty((nsweeps, before_trace.NofSamples))
        after_currents = np.empty((nsweeps, after_trace.NofSamples))

        # Get ramp times from protocol description
        voltage_protocol = VoltageProtocol(voltage, before_trace.get_times())

        # Get first ramp
        tstart, tend = voltage_protocol.get_ramps()[0][:2]
        t = before_trace.get_times()
        ramp_bounds = [np.argmax(t > tstart), np.argmax(t > tend)]

        assert after_trace.NofSamples == before_trace.NofSamples

        selected_wells = []
        for sweep in range(nsweeps):
            before_params1, before_leak = fit_linear_leak(before_trace, well,
                                                          sweep, ramp_bounds,
                                                          plot=True,
                                                          label=savename,
                                                          output_dir=args.output_dir)

            after_params1, after_leak = fit_linear_leak(after_trace, well,
                                                        sweep, ramp_bounds,
                                                        plot=True,
                                                        label=savename,
                                                        output_dir=args.output_dir)

            before_raw = np.array(raw_before_all[well])[sweep, :]
            after_raw = np.array(raw_after_all[well])[sweep, :]

            before_currents[sweep, :] = before_raw - before_leak
            after_currents[sweep, :] = after_raw - after_leak

        # TODO Note: only run this for whole/staircaseramp for now...
        logging.info(savename + ' ' + well + ' ' + savename + '\n----------')
        logging.info(f"sampling_rate is {sampling_rate}")

        plot_dir = os.path.join(args.output_dir, f"debug_{export_config.saveID}",
                                f"{well}-{savename}")

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        hERGQC.plot_dir = plot_dir

        # Run QC with leak subtracted currents
        selected, QC = hergqc.run_qc(before_currents,
                                     after_currents,
                                     np.array(qc_before[well])[0, :],
                                     np.array(qc_after[well])[0, :],
                                     nsweeps)

        df_rows.append([well] + QC)

        if selected:
            selected_wells.append(well)

        # Save subtracted current in csv file
        header = "\"current\""
        savepath = os.path.join(args.output_dir, export_config.savedir,
                                f"{export_config.saveID}-{savename}-{well}")
        if not os.path.exists(os.path.join(args.output_dir,
                                           export_config.savedir)):
            os.makedirs(os.path.join(args.output_dir, export_config.savedir))

        for i in range(nsweeps):
            subtracted_current = before_currents[0, :] - after_currents[0, :]
            np.savetxt(savepath, subtracted_current, delimiter=',',
                       comments='', header=header)

    column_labels = ['well', 'qc1.rseal', 'qc1.cm', 'qc1.rseries', 'qc2.raw',
                     'qc2.subtracted', 'qc3.raw', 'qc3.E4031', 'qc3.subtracted',
                     'qc4.rseal', 'qc4.cm', 'qc4.rseries', 'qc5.staircase',
                     'qc5.1.staircase', 'qc6.subtracted', 'qc6.1.subtracted',
                     'qc6.2.subtracted']

    df = pd.DataFrame(np.array(df_rows), columns=column_labels)
    df['no_cell'] = no_cell

    # Add onboard qc to dataframe
    for well in args.wells:
        if well not in df['well'].values:
            onboard_qc_df = pd.DataFrame([[well] + [False for col in
                                                    list(df)[1:]]],
                                         columns=list(df))
            df = pd.concat([df, onboard_qc_df], ignore_index=True)

    df['protocol'] = savename

    return selected_cells, df


def qc3_bookend(readname, savename, time_strs):

    plot_dir = os.path.join(args.output_dir, export_config.savedir,
                            "{saveID}-{savename}-plot")

    filepath_first = os.path.join(args.data_directory,
                                  f"{readname}_{time_strs[0]}")
    filepath_last = os.path.join(args.data_directory,
                                 f"{readname}_{time_strs[3]}")
    json_file_first = f"{readname}_{time_strs[0]}"
    json_file_last = f"{readname}_{time_strs[3]}"

    first_trace = Trace(filepath_first,
                        json_file_first)
    last_trace = Trace(filepath_last,
                       json_file_last)

    assert np.all(first_trace.get_voltage() == last_trace.get_voltage())

    voltage = first_trace.get_voltage()

    hergqc = hERGQC(sampling_rate=first_trace.sampling_rate,
                    plot_dir=plot_dir,
                    voltage=voltage)

    assert first_trace.NofSweeps == last_trace.NofSweeps

    first_trace_sweeps = first_trace.get_trace_sweeps()
    last_trace_sweeps = last_trace.get_trace_sweeps()
    rows = []
    for well in args.wells:
        qc3_bookend = hergqc.qc3(first_trace_sweeps[well][0, :],
                                 last_trace_sweeps[well][1, :])
        rows.append([qc3_bookend, well])

    df = pd.DataFrame(rows, columns=['qc3.bookend', 'well'])
    df['protocol'] = savename
    return df


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
