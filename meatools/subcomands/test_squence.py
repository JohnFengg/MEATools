#!/usr/bin/env python
"""Test sequence and polarization analysis."""

import os
import json
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from glob import glob
import pandas as pd

from ..utils.serialization import NumpyEncoder
from ..utils.plot_style import apply_unicode_font

apply_unicode_font()

"""
When I wrote this code, only God and I know it.
Now ...
ONLY God knows...
"""


def find_csv_files(root_path):
    """Find all CSV files recursively with relative paths."""
    return [os.path.join(root, f)
            for root, _, files in os.walk(root_path)
            for f in files if f.lower().endswith('.csv')]


def find_csv_pol_files(root_path, search_key):
    """Find polarization CSV files matching search_key."""
    files = glob(os.path.join(root_path, '**', search_key), recursive=True)
    return files


def extract_start_times(csv_files, log=None):
    """Extract start times from files with validation.

    Args:
        csv_files: List of CSV file paths.
        log: Optional file-like object for logging.

    Returns:
        Sorted list of dicts with path, time, time_str, line.
    """
    results = []
    for filepath in csv_files:
        try:
            with open(filepath, 'r', encoding='ISO-8859-1') as f:
                lines = [line.strip() for line in f.readlines()]
                for line_num in [5, 6, 7]:
                    if line_num >= len(lines):
                        continue
                    parts = lines[line_num].split(',') or lines[line_num].split('\\t')
                    if len(parts) >= 2 and parts[0].strip() == 'Start time':
                        time_str = parts[1].strip()
                        try:
                            time_obj = datetime.strptime(time_str, '%m/%d/%y %H:%M:%S')
                            results.append({
                                'path': filepath,
                                'time': time_obj,
                                'time_str': time_str,
                                'line': line_num + 1
                            })
                            break
                        except ValueError:
                            continue
        except Exception as e:
            if log:
                log.write(f"\nError processing {os.path.basename(filepath)}: {e}\n")
    return sorted(results, key=lambda x: x['time'])


def extract_data_from_file(filepath, log=None):
    """Extract data columns from a CSV file.

    Args:
        filepath: Path to the CSV file.
        log: Optional file-like object for logging.

    Returns:
        Dict with file_info, data dict, and columns_found list.
    """
    with open(filepath, 'r', encoding='ISO-8859-1') as f:
        lines = [line.strip() for line in f.readlines()]

    data_start = None
    for i, line in enumerate(lines):
        if line.lower().startswith('time stamp'):
            data_start = i
            break

    if data_start is None:
        if log:
            log.write(f"\nNo 'Time stamp' line found in {filepath}\n")
        return None

    data = np.genfromtxt(filepath,
                         delimiter=',',
                         skip_header=data_start + 1,
                         invalid_raise=False,
                         encoding='ISO-8859-1')

    header = lines[data_start].lower().split(',')
    col_indices = {}
    targets = [
        'elapsed time',
        'current',
        'current_set',
        'cell_voltage_001',
        lambda x: x.endswith('.resistance'),
        'temp_coolant_inlet',
        'temp_cathode_dewpoint_gas',
        'temp_anode_dewpoint_gas',
        'pressure_cathode_inlet',
        'cell_active_area'
    ]

    for i, col in enumerate(header):
        col = col.strip()
        for target in targets:
            if (callable(target) and target(col)) or (col == target):
                col_name = 'resistance' if callable(target) else col.replace(' ', '_')
                col_indices[col_name] = i
                break

    extracted = {}
    for name, idx in col_indices.items():
        extracted[name] = data[:, idx]

    return {
        'file_info': filepath,
        'data': extracted,
        'columns_found': list(col_indices.keys())
    }


def _filter_plottable(all_results, required, plot_name, log):
    """Drop files that lack columns needed for a plot (with a log line).

    Generic CSV scans pick up pressure-holding / voltage-only / logger
    files that have no current (or temperature) columns; plotting must
    not die on them (B1).

    Returns:
        List of (key, result) entries that have all required columns.
    """
    plottable = []
    for key, result in all_results.items():
        data = result['data']
        missing = [c for c in required if c not in data]
        if missing:
            if log:
                log.write(f"\nSkipping {plot_name} for "
                          f"{os.path.basename(result['file_info'])}: "
                          f"missing column(s) {', '.join(missing)}\n")
        else:
            plottable.append((key, result))
    return plottable


def plot_voltages(all_results, log=None):
    """Generate subplots for each file's current vs elapsed time."""
    plottable = _filter_plottable(all_results,
                                  ('elapsed_time', 'current'),
                                  'voltage plot', log)
    num_files = len(plottable)
    if num_files == 0:
        return
    cols = 3
    rows = (num_files + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if num_files > 1 else [axes]

    for idx, (key, result) in enumerate(plottable):
        ax = axes[idx]
        data = result['data']
        ax.plot(data['elapsed_time'] / 60, data['current'], 'b-', label='Current')
        ax.set_title(f"{key}: {os.path.basename(result['file_info'])}")
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Current (A)')
        ax.grid(True)
        ax.legend()

    for idx in range(num_files, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig('results/test_sequence/voltage_plots.png')


def plot_Tcells(all_results, log=None):
    """Generate temperature subplots for each file."""
    plottable = _filter_plottable(all_results, ('elapsed_time', 'temp_coolant_inlet'),
                                  'temperature plot', log)
    num_files = len(plottable)
    if num_files == 0:
        return
    cols = 3
    rows = (num_files + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if num_files > 1 else [axes]

    for idx, (key, result) in enumerate(plottable):
        ax = axes[idx]
        data = result['data']
        ax.plot(data['elapsed_time'] / 60, data['temp_coolant_inlet'], 'r-', label='Coolant In')
        if 'temp_cathode_dewpoint_gas' in data:
            ax.plot(data['elapsed_time'] / 60, data['temp_cathode_dewpoint_gas'], 'b-', label='Cathode Dew')
        if 'temp_anode_dewpoint_gas' in data:
            ax.plot(data['elapsed_time'] / 60, data['temp_anode_dewpoint_gas'], 'g-', label='Anode Dew')
        ax.set_title(f"{key}: Temps")
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Temperature (C)')
        ax.grid(True)
        ax.legend()

    for idx in range(num_files, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig('results/test_sequence/temperature_plots.png')


def plot_step_timeline(steps_data, title="Test sequence", log=None):
    """Plot a Gantt-style timeline of test steps."""
    if not steps_data:
        # B5: no CSV had a parseable start time - there is nothing to
        # draw. Must not crash, or the plots and the whole polarization
        # half of ttseq would never run.
        if log:
            log.write("\nNo steps with valid start times; "
                      "skipping step timeline plot\n")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    base_time = datetime.strptime(steps_data[0]['start'], '%Y-%m-%d %H:%M:%S')

    for i, step in enumerate(steps_data):
        start = datetime.strptime(step['start'], '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime(step['end'], '%Y-%m-%d %H:%M:%S')
        start_hours = (start - base_time).total_seconds() / 3600
        duration_hours = (end - start).total_seconds() / 3600
        ax.barh(i, duration_hours, left=start_hours, height=0.4, label=step['name'])

    ax.set_yticks(range(len(steps_data)))
    ax.set_yticklabels([f"Step {s['step']}" for s in steps_data])
    ax.set_xlabel('Hours from start')
    ax.set_title(title)
    ax.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig('results/test_sequence/test_timeline.png')


def plot_Pol(all_pol_results, sampleSize, log=None):
    """Plot polarization curves."""
    plt.figure(figsize=(10, 6))
    for key, result in all_pol_results.items():
        data = result['data']
        if 'current' not in data or 'cell_voltage_001' not in data:
            if log:
                log.write(f"\nSkipping polarization curve for "
                          f"{os.path.basename(result['file_info'])}: "
                          "missing 'current' or 'cell_voltage_001' column\n")
            continue
        voltage = data['cell_voltage_001']
        current = data['current']
        plt.plot(current / sampleSize, voltage, 'o-', label=key)
    plt.xlabel('Current Density (A/cm2)')
    plt.ylabel('Voltage (V)')
    plt.title('Polarization Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/polarization/polarization_curves.png')


def areaDetermine(all_pol_results):
    """Determine sample area from polarization data."""
    areas = []
    for key, result in all_pol_results.items():
        data = result['data']
        if 'cell_active_area' in data:
            areas.append(np.mean(data['cell_active_area']))
    return np.mean(areas) if areas else 5.0


def analyze_pol_steps(result, sampleSize, pol_report_avg, log=None):
    """Per-step polarization analysis for one polarization CSV file.

    Restores the pre-refactor (v0.3.5) ``polarization_results.json``
    entry: at the end of each long current step, average the last
    ``pol_report_avg`` seconds of current / voltage / voltage+IR, and
    convert current to current density using the sample area.

    Args:
        result: Dict from ``extract_data_from_file()``.
        sampleSize: Cell active area in cm2.
        pol_report_avg: Reporting window in seconds.
        log: Optional file-like object for logging.

    Returns:
        Dict with the OCV and per-step arrays (6 keys), or None when the
        file lacks required columns or no qualifying step ends exist.
    """
    data = result['data']
    required = ('elapsed_time', 'current', 'cell_voltage_001',
                'resistance', 'current_set')
    missing = [c for c in required if c not in data]
    if missing:
        if log:
            log.write(f"\nSkipping pol step analysis for "
                      f"{result['file_info']}: missing columns "
                      f"{', '.join(missing)}\n")
        return None

    time = data['elapsed_time']
    voltage = data['cell_voltage_001']
    current = data['current']
    HFR = data['resistance']
    currentSet = data['current_set']
    voltageIR = voltage + HFR * current / 1000
    ocv = float(voltage[-1])

    # Ends of current steps: where the set current drops by > 0.95 A.
    positions = np.where(np.diff(currentSet) < -0.95)[0] + 1
    if len(positions) >= 2:
        diffs = np.diff(positions)
        split_indices = np.where(diffs > 50)[0] + 1
    else:
        split_indices = np.array([], dtype=int)
    segs = positions[split_indices]
    if len(segs) == 0:
        if log:
            log.write(f"\nNo qualifying current-step ends found in "
                      f"{result['file_info']}\n")
        return None

    segsVol = np.zeros((len(segs), 5))
    for s, steps in enumerate(segs):
        mask = (time < time[steps]) & (time > time[steps] - pol_report_avg)
        with np.errstate(all='ignore'):
            segsVol[s, 0] = float(np.average(current[mask])) if np.any(mask) else np.nan
            segsVol[s, 1] = float(np.average(voltage[mask])) if np.any(mask) else np.nan
            segsVol[s, 2] = float(np.average(voltageIR[mask])) if np.any(mask) else np.nan
        segsVol[s, 3] = segsVol[s, 2] - segsVol[s, 1]
    segsVol[:, 4] = segsVol[:, 0] / sampleSize

    return {
        "OCV (V)": ocv,
        "current (A)": segsVol[:, 0].tolist(),
        "voltage (V)": segsVol[:, 1].tolist(),
        "voltage+IR (V)": segsVol[:, 2].tolist(),
        "IR (V)": segsVol[:, 3].tolist(),
        "current density (A cm^(-2))": segsVol[:, 4].tolist()
    }


def main(search_key, pol_report_avg, log=None):
    """Main entry point for test sequence and polarization analysis.

    Args:
        search_key: Glob pattern for polarization CSV files.
        pol_report_avg: Reporting average in seconds.
        log: Optional file-like object for logging.
    """
    script_dir = os.getcwd()

    ##############################################################
    ## detect all of the csv files and find start time from them ##
    ##############################################################
    csv_files = find_csv_files(script_dir)
    if not csv_files:
        if log:
            log.write("\nNo CSV files found in directory tree\n")
    else:
        if log:
            log.write(f"\nFound {len(csv_files)} CSV files\n")

    results = extract_start_times(csv_files, log=log)
    if not results:
        if log:
            log.write("No valid start times found in lines 6-8\n")
    else:
        if log:
            log.write(f"Found {len(results)} files with valid start times:\n")
            log.write("=" * 80)

    all_results = {}
    steps_data = []
    for i, res in enumerate(results, 1):
        rel_path = os.path.relpath(res['path'], script_dir)
        result = extract_data_from_file(rel_path, log=log)
        if result is None:
            continue
        data = result['data']
        if 'elapsed_time' in data:
            duration = np.max(data['elapsed_time']) / 60
        else:
            duration = float('nan')
        if not np.isfinite(duration):
            # B6: all-NaN (or missing) elapsed time -> timedelta(minutes=NaN)
            # used to crash the whole run before the timeline JSON was
            # written. Skip the file instead.
            if log:
                log.write(f"\nSkipping {rel_path}: elapsed time is all "
                          f"NaN (no valid duration)\n")
            continue
        all_results[f"A{i}"] = result

        if log:
            log.write(f"\n{i}. {rel_path}")
            log.write(f"\tParsed: {res['time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"\tDuration: {duration:.1f} mins\n")
            log.write(f"\tFound columns: {', '.join(result['columns_found'])}\n")
            log.write("-" * 80)

        steps_data.append({
            "step": i,
            "start": res['time'].strftime('%Y-%m-%d %H:%M:%S'),
            "duration": float(f"{duration:.1f}"),
            "name": f"Step {i}: {rel_path}"
        })

    with open('results/test_sequence/all_csv_results_in_timeline.json', 'w') as all_json:
        json.dump(all_results, all_json, indent=2, cls=NumpyEncoder)

    time_line = {}
    total_time = 0
    for step in steps_data:
        starttime = datetime.strptime(step['start'], '%Y-%m-%d %H:%M:%S')
        endtime = starttime + timedelta(minutes=step['duration'])
        endtime_str = endtime.strftime('%Y-%m-%d %H:%M:%S')
        step["end"] = endtime_str
        time_line[str(step["step"])] = {
            "start_time": step["start"],
            "end_time": step["end"],
            "duration (mins)": step["duration"],
            "file_path": step["name"]
        }
        total_time += step["duration"]
    time_line["total_time (mins)"] = total_time

    with open('results/test_sequence/test_order_in_timeline.json', 'w') as timeline:
        json.dump(time_line, timeline, indent=2, cls=NumpyEncoder)

    plot_step_timeline(steps_data, title="Test sequence", log=log)
    plot_voltages(all_results, log=log)
    plot_Tcells(all_results, log=log)

    ##########################################################
    ############### polarization analyzer #####################
    ##########################################################
    csv_files = find_csv_pol_files(script_dir, search_key)
    if log:
        log.write('\n' + "**" * 80)
        log.write(f"\nFound {len(csv_files)} Pol CSV files")
    results = extract_start_times(csv_files, log=log)

    all_pol_results = {}
    step_pol_data = {}
    if log:
        log.write(f"Found {len(results)} Pol files with valid start times:\n")
        log.write("=" * 80)
    for i, res in enumerate(results, 1):
        rel_path = os.path.relpath(res['path'], script_dir)
        result = extract_data_from_file(rel_path, log=log)
        if result is None:
            continue
        data = result['data']
        if 'elapsed_time' in data:
            duration = np.max(data['elapsed_time']) / 60
        else:
            duration = float('nan')
        if not np.isfinite(duration):
            # B6: all-NaN elapsed time -> timedelta(minutes=NaN) crash
            if log:
                log.write(f"\nSkipping polarization file {rel_path}: "
                          f"elapsed time is all NaN\n")
            continue
        all_pol_results[f"A{i}"] = result
        if log:
            log.write(f"\n{i}. {rel_path}")
            log.write(f"\tLine {res['line']}: Start time, {res['time_str']}\n")
            log.write(f"\tParsed: {res['time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"\tFound columns: {', '.join(result['columns_found'])}\n")
            log.write("-" * 80)
        endtime = res['time'] + timedelta(minutes=duration)
        endtime_str = endtime.strftime('%Y-%m-%d %H:%M:%S')
        step_data = {
            "start_time": res['time'].strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": endtime_str,
            "duration (mins)": float(f"{duration:.1f}"),
            "file_path": f"Step {i}: {rel_path}"
        }
        step_pol_data[str(i)] = step_data

    sampleSize = areaDetermine(all_pol_results)
    step_pol_data["sampleSize (cm2)"] = sampleSize

    if log:
        log.write(f"\nConfirmed the sample size is {sampleSize} cm2")
    with open('results/polarization/all_pol_results.json', 'w') as all_pol_json:
        json.dump(all_pol_results, all_pol_json, indent=2, cls=NumpyEncoder)

    with open('results/polarization/polarization.json', 'w') as pol:
        json.dump(step_pol_data, pol, indent=2)

    # Per-step polarization analysis (consumed by `mea conclude` and the
    # Polarization section of results.json / the HTML report).
    pol_results = {}
    for key, result in all_pol_results.items():
        filename = os.path.basename(result['file_info'])
        step = analyze_pol_steps(result, sampleSize, pol_report_avg, log=log)
        if step is not None:
            pol_results[filename] = step

    with open('results/polarization/polarization_results.json', 'w') as polres:
        json.dump(pol_results, polres, indent=2, cls=NumpyEncoder)

    plot_Pol(all_pol_results, sampleSize, log=log)


if __name__ == '__main__':
    searchKey = '*Pol*-*.csv'
    polReportAVG = 30
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results/test_sequence', exist_ok=True)
    os.makedirs('results/polarization', exist_ok=True)
    log = open('logs/test_sequence.log', 'w')
    main(searchKey, polReportAVG, log=log)
    log.close()
