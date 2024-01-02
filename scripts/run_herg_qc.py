import numpy as np
import argparse
import logging
import glob
import os
import sys
import importlib.util
import pandas as pd
import regex as re
import multiprocessing

from pcpostprocess.trace import Trace
from pcpostprocess.hergQC import hERGQC
from pcpostprocess.leak_correct import fit_linear_leak
from pcpostprocess.voltage_protocols import VoltageProtocol

import string


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

    global args
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join('output', 'hergqc')
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    spec = importlib.util.spec_from_file_location(
        'export_config',
        os.path.join(args.data_directory,
                     'export_config.py'))

    if args.wells is None:
        args.wells = wells

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
    qc_df = pd.concat(qc_dfs, ignore_index=True)

    savedir = os.path.join(args.output_dir, export_config.savedir)
    saveID = export_config.saveID

    if not os.path.exists(os.path.join(args.output_dir, savedir)):
        os.makedirs(os.path.join(args.output_dir, savedir))

    # Write qc_df to file
    qc_df.to_csv(os.path.join(args.output_dir,
                              savedir, 'QC-%s.csv' % saveID))

    # Write data to JSON file
    qc_df.to_json(os.path.join(args.output_dir,savedir, 'QC-%s.json' % saveID),
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

    print(savenames, readnames, times_list)
    with multiprocessing.Pool(min(args.no_cpus, len(export_config.D2S_QC)),
                              **pool_kws) as pool:
        pool.starmap(extract_protocol, zip(readnames,
                                           savenames,
                                           times_list,
                                           [wells_to_export]
                                           * len(savenames)))


def extract_protocol(readname, savename, time_strs, selected_wells):
    savedir = os.path.join(args.output_dir, export_config.savedir)
    saveID  = export_config.saveID

    plot_dir = os.path.join(args.output_dir, savedir,
                            "{saveID}-{savename}-plot")

    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    logging.debug(f"Exporting {readname} as {savename}")

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

    nsweeps_before = before_trace.NofSweeps = 2
    nsweeps_after = after_trace.NofSweeps = 2

    assert nsweeps_before == nsweeps_after

    # Time points
    times_before = before_trace.get_times()
    times_after = after_trace.get_times()

    try:
        assert all(np.abs(times_before - times_after) < 1e-8)
    except Exception as exc:
        print(f"Exception thrown when handling {savename}: ", str(exc))
        return

    header = "\"current\""

    qc_before = before_trace.get_onboard_QC_values()
    qc_after = after_trace.get_onboard_QC_values()

    for i_well, well in enumerate(selected_wells):  # Go through all wells
        if i_well % 24 == 0:
            print('row ' + well[0])

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
    voltage = voltage_before * 1e3

    voltage_df = pd.DataFrame(np.vstack((times_before.flatten(),
                                         voltage.flatten())).T,
                              columns=['time', 'voltage'])

    voltage_df.to_csv(os.path.join(f"{saveID}-{savename}-voltages.csv"))
    np.savetxt(os.path.join(savedir, f"{saveID}-{savename}-times.txt"),
               times_before)


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
        logging.debug(savename + ' ' + well + ' ' + savename + '\n----------')
        logging.debug(f"sampling_rate is {sampling_rate}")

        plot_dir = os.path.join(args.output_dir, f"debug_{export_config.saveID}",
                                f"{well}-{savename}")

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        hERGQC.plot_dir = plot_dir

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
        savepath = os.path.join(export_config.savedir,
                                f"{export_config.saveID}-{savename}-{well}")
        if not os.path.exists(export_config.savedir):
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


if __name__ == '__main__':
    main()
