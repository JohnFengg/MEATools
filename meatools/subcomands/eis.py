#!/usr/bin/env python
"""EIS analysis: electrochemical impedance spectroscopy processing."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from ..utils.serialization import NumpyEncoder
from ..utils.file_utils import find_and_sort_dta_files_by_candidates


def read_dta_data(file_path):
    """Read EIS data from a DTA file.

    Args:
        file_path: Path to the DTA file.

    Returns:
        2-D numpy array with shape (n, >= 5): the ZCURVE data rows.

    Raises:
        ValueError: when no usable data rows can be parsed (empty file,
            no ZCURVE block, binary/AppleDouble junk, malformed rows).
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
                if not line:
                    break
                try:
                    values = np.array([float(x) for x in line.split('\t') if x])
                except ValueError:
                    # Instrument text after the data body (e.g. the
                    # 'EXPERIMENTABORTED' abort marker) or a malformed
                    # line: stop at the end of the numeric data.
                    break
                data.append(values)
            break
        else:
            i += 1

    if not data:
        raise ValueError(f"no EIS data rows found in {file_path}")
    try:
        arr = np.array(data)
    except ValueError as exc:
        raise ValueError(
            f"inconsistent EIS data rows in {file_path}: {exc}") from exc
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError(
            f"unexpected EIS data shape {arr.shape} in {file_path}")
    return arr


def _vectorized_sliding_regression(data, num_p=5):
    """Vectorized sliding window linear regression.

    Args:
        data: Numpy array of EIS data.
        num_p: Window size - 1.

    Returns:
        Array of shape (5, m) with regression results.
    """
    n = len(data[:, 0])
    m = n - num_p

    # Extract sliding windows using advanced indexing
    indices = np.arange(num_p + 1) + np.arange(m)[:, None]
    x_windows = data[indices, 3]
    y_windows = data[indices, 4]
    freq_windows = data[indices, 2]

    # Build design matrices: X shape (m, num_p+1, 2) = [x, 1]
    X = np.stack([x_windows, np.ones_like(x_windows)], axis=2)

    # Normal equations: (X^T X) beta = X^T y
    XtX = np.einsum('mij,mik->mjk', X, X)
    Xty = np.einsum('mij,mi->mj', X, y_windows)

    # Solve for each window (lstsq tolerates singular / constant windows)
    beta = np.zeros((m, 2))
    for i in range(m):
        beta[i] = np.linalg.lstsq(XtX[i], Xty[i], rcond=None)[0]

    a = beta[:, 0]
    b = beta[:, 1]

    # R² calculation
    y_pred = a[:, None] * x_windows + b[:, None]
    ss_res = np.sum((y_windows - y_pred) ** 2, axis=1)
    y_mean = np.mean(y_windows, axis=1)
    ss_tot = np.sum((y_windows - y_mean[:, None]) ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1 - ss_res / ss_tot
        r2 = np.where(np.isfinite(r2), r2, 0.0)

    p1_1 = np.zeros((5, m))
    p1_1[0, :] = b
    p1_1[1, :] = a
    p1_1[2, :] = r2
    with np.errstate(divide="ignore", invalid="ignore"):
        p1_1[3, :] = -b / a
    p1_1[4, :] = np.min(freq_windows, axis=1)

    return p1_1


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

    # Vectorized sliding window regression
    p1_1 = _vectorized_sliding_regression(data, num_p=5)

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
    os.makedirs('results/eis', exist_ok=True)
    plt.savefig(f'results/eis/curve{index}.png')

    selected = p1_1[3, mask]
    selected = selected[np.isfinite(selected)]
    if len(selected) == 0:
        median = float("nan")
        std = float("nan")
        sample_num = 0
    else:
        median = float(np.median(selected))
        std = float(np.std(selected))
        sample_num = len(selected)
    r_ion = (median - hfr) * 3
    r_ion_std = std * 3

    return hfr, r_ion, r_ion_std, sample_num


def main():
    """Main entry point for EIS analysis.

    Each DTA file is processed independently: a bad file (no positive
    imaginary values, 1-D/empty parse result, abort markers, ...) is
    recorded as ``{"error": ...}`` and the remaining files are still
    processed, so one bad file cannot kill the whole case (B3).
    """
    os.makedirs('results/eis/', exist_ok=True)
    results = {}
    file_info = find_and_sort_dta_files_by_candidates('.', candidates=('EIS', 'PEIS'))
    for i, (filetime, filepath) in enumerate(file_info):
        readable_time = datetime.fromtimestamp(filetime).strftime('%Y-%m-%d %H:%M:%S')
        entry = {
            "filename": filepath,
            "filetime": readable_time,
        }
        try:
            data = read_dta_data(filepath)
            hfr, r_ion, r_ion_std, sample_num = EIS_calc(data, i, filepath)
        except Exception as exc:
            plt.close('all')
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[eis] skipping {filepath}: {entry['error']}")
        else:
            entry.update({
                "HFR (ohm)": hfr,
                "R_ion (ohm)": r_ion,
                "R_ion_std": r_ion_std,
                "sample_number": sample_num,
            })
        results[f"file_{i + 1}"] = entry
    with open('results/eis/eis_results.json', 'w') as results_file:
        json.dump(results, results_file, ensure_ascii=False, indent=2, cls=NumpyEncoder)


if __name__ == "__main__":
    main()
