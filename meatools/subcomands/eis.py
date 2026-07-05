#!/usr/bin/env python
"""EIS analysis: electrochemical impedance spectroscopy processing."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score

from ..utils.serialization import NumpyEncoder
from ..utils.file_utils import find_and_sort_dta_files_by_candidates


def read_dta_data(file_path):
    """Read EIS data from a DTA file.

    Args:
        file_path: Path to the DTA file.

    Returns:
        Numpy array of EIS data.
    """
    data = []
    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        txt_line = f.readlines()
        i = 0
        while i < len(txt_line):
            if txt_line[i].startswith('ZCURVE'):
                i += 3
                for j in range(i, len(txt_line), 1):
                    line = txt_line[j].strip()
                    values = np.array([float(x) for x in line.split('\t') if x])
                    data.append(values)
                break
            else:
                i += 1
    return np.array(data)


def EIS_calc(data, index, file_path):
    """Calculate EIS metrics and generate plots.

    Args:
        data: Numpy array of EIS data.
        index: File index for plot naming.
        file_path: Original file path for titles.

    Returns:
        Tuple of (hfr, r_ion, r_ion_std, sample_num).
    """
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.plot(data[:, 3], -data[:, 4], 'o')
    plt.xlim(0, 0.1)
    plt.ylim(0, 0.2)
    plt.subplot(2, 3, 2)
    plt.plot(data[:, 2], data[:, 3], 'o')
    plt.plot(data[:, 2], -data[:, 4], 'x')
    plt.ylim(0, 0.2)
    plt.xscale('log')
    plt.grid(True, which='major', linestyle='-', alpha=0.8)
    plt.grid(True, which='minor', linestyle=':', alpha=0.4)
    plt.title(file_path)

    freq, zreal, zimag = data[:, 2], data[:, 3], data[:, 4]

    # HFR calculation with validation
    positive_imag = zimag[zimag > 0]
    if len(positive_imag) == 0:
        raise ValueError(f"No positive imaginary values in {file_path}")
    hfr_idx_pos = np.argmin(positive_imag)
    hfr_idx_pos = np.where(zimag > 0)[0][hfr_idx_pos]

    negative_imag = zimag[zimag < 0]
    if len(negative_imag) > 0:
        hfr_idx_neg = np.argmin(np.abs(negative_imag))
        hfr_idx_neg = np.where(zimag < 0)[0][hfr_idx_neg]
    else:
        hfr_idx_neg = hfr_idx_pos

    hfr = (zreal[hfr_idx_pos] + zreal[hfr_idx_neg]) / 2

    num_p = 5
    p1_1 = np.zeros((5, len(data[:, 0]) - num_p))
    for q in range(len(data[:, 0]) - num_p):
        seq = slice(q, q + num_p + 1)
        x_data = data[seq, 3].reshape(-1, 1)
        y_data = data[seq, 4]
        model = HuberRegressor()
        model.fit(x_data, y_data)
        p1_1[0, q] = model.intercept_
        p1_1[1, q] = model.coef_[0]
        p1_1[4, q] = np.min(data[seq, 2])
        y_pred = model.predict(x_data)
        p1_1[2, q] = r2_score(y_data, y_pred)
        p1_1[3, q] = -model.intercept_ / model.coef_[0]

    mask0 = p1_1[2, :] >= 0.999
    mask1 = p1_1[2, :] >= np.quantile(p1_1[2, :], 0.9)
    mask2 = p1_1[4, :] <= 20
    mask = (mask1 | mask0) & mask2

    plt.subplot(2, 3, 4)
    plt.plot(p1_1[4, :], p1_1[0, :], 'o')
    plt.xscale('log')
    plt.subplot(2, 3, 5)
    plt.plot(p1_1[4, :], p1_1[3, :], 'o', alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(p1_1[4, mask], p1_1[3, mask], 'o')
    plt.grid(True, which='major', linestyle='-', alpha=0.8)
    plt.grid(True, which='minor', linestyle=':', alpha=0.4)
    plt.subplot(2, 3, 6)
    plt.plot(p1_1[4, :], p1_1[2, :], 'o')
    plt.xscale('log')
    plt.savefig(f'results/eis/curve{index}.png')

    median = np.median(p1_1[3, mask])
    std = np.std(p1_1[3, mask])
    r_ion = (median - hfr) * 3
    r_ion_std = std * 3
    sample_num = len(p1_1[2, mask])

    return hfr, r_ion, r_ion_std, sample_num


def main():
    """Main entry point for EIS analysis."""
    os.makedirs('results/eis/', exist_ok=True)
    results = {}
    file_info = find_and_sort_dta_files_by_candidates('.', candidates=('EIS', 'PEIS'))
    for i, (filetime, filepath) in enumerate(file_info):
        readable_time = datetime.fromtimestamp(filetime).strftime('%Y-%m-%d %H:%M:%S')
        data = read_dta_data(filepath)
        hfr, r_ion, r_ion_std, sample_num = EIS_calc(data, i, filepath)
        data_dump = {
            "filename": filepath,
            "filetime": readable_time,
            "HFR (ohm)": hfr,
            "R_ion (ohm)": r_ion,
            "R_ion_std": r_ion_std,
            "sample_number": sample_num
        }
        results[f"file_{i + 1}"] = data_dump
    with open('results/eis/eis_results.json', 'w') as results_file:
        json.dump(results, results_file, ensure_ascii=False, indent=2, cls=NumpyEncoder)


if __name__ == "__main__":
    main()
