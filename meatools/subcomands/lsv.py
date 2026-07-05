#!/usr/bin/env python
"""LSV analysis: linear sweep voltammetry processing."""

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from datetime import datetime

from ..utils.serialization import NumpyEncoder
from ..utils.file_utils import find_and_sort_load_dta_files, find_cv_subfolders

# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def lsv_calc(u_subfolders, search_key, ECAcutoff, log=None):
    """Process LSV subfolders and compute metrics.

    Args:
        u_subfolders: List of folder paths.
        search_key: Glob pattern for DTA files.
        ECAcutoff: Lower voltage cutoff (retained for API consistency).
        log: Optional file-like object for logging.

    Returns:
        Dict with results per directory and file.
    """
    results = {}
    for jk in range(len(u_subfolders)):
        file_path = u_subfolders[jk]
        file_info = find_and_sort_load_dta_files(file_path, search_key=search_key)
        plt.figure(jk + 1)
        results[f"dir_{jk}"] = {}
        for i, (mtime, filepath) in enumerate(file_info, 1):
            results[f"dir_{jk}"][str(i)] = {}
            readable_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            results[f"dir_{jk}"][str(i)]["file"] = filepath
            results[f"dir_{jk}"][str(i)]["time_stamp"] = readable_time

            with open(filepath, 'r') as f:
                for _ in range(59):
                    next(f)
                content_from_line65 = f.read()

            u = re.split('CURVE', content_from_line65)
            plt.subplot(2, 3, i)
            dump = {}
            for j in range(min(5, len(u))):
                temp_path = os.path.join(os.getcwd(), 'temp')
                with open(temp_path, 'w') as fileID:
                    fileID.write(u[j])
                A = np.loadtxt(temp_path, skiprows=2, usecols=range(8))
                plt.plot(A[:, 2], A[:, 3])
                if j > 0:
                    scanRate = np.median(np.abs(np.diff(A[:, 2]) / np.diff(A[:, 1])))
                    CVall = A[:, [2, 3]]
                    updData = CVall[:, :2]
                    mask1 = (updData[:, 0] > 0.3) & (updData[:, 0] < 0.6) & \
                            (np.concatenate(([0], np.diff(updData[:, 0]))) > 0)
                    double1 = updData[mask1, :]

                    vol = updData[:, 0]
                    index = np.argmin(abs(vol - 0.4))
                    H2cx3 = updData[:, 1][index]

                    x_new = np.arange(0.35, 0.55, 0.001)
                    f1 = interpolate.interp1d(double1[:, 0], double1[:, 1], bounds_error=False)
                    double1_interp = f1(x_new)
                    H2cx = np.quantile(double1_interp, 0.99)

                    n = len(double1[:, 0])
                    m = (n * np.sum(double1[:, 0] * double1[:, 1]) - np.sum(double1[:, 0]) * np.sum(double1[:, 1])) / \
                        (n * np.sum(double1[:, 0] * double1[:, 0]) - np.sum(double1[:, 0]) ** 2)
                    b = (np.sum(double1[:, 1]) - m * np.sum(double1[:, 0])) / n
                    H2cx2 = m * 0.8 + b

                    voltage, cur = A[:, 2], A[:, 3]
                    lsv_membrane = cur[np.argmin(np.abs(voltage - 0.4))]

                    dump[f"curve_{j}"] = {
                        "rate (V/s)": scanRate,
                        "H2cx99%": H2cx,
                        "slopeReg": m,
                        "H2cx_800mV": H2cx2,
                        "H2cx_0mV": b,
                        "H2cx_400mV": H2cx3,
                        "lsv_membrane*area": lsv_membrane
                    }
                else:
                    plt.title(f"{i}. {filepath}")

            results[f"dir_{jk}"][str(i)]["data"] = dump

    plt.savefig('results/lsv/lsv_results.png')
    return results


if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results/lsv/', exist_ok=True)
    log = open('logs/lsv.log', 'w')

    searchKey = 'LSV/**/*.DTA'
    ECAcutoff = 0.08
    u_subfolders = find_cv_subfolders('.', searchKey, log=log)
    results = lsv_calc(u_subfolders, searchKey, ECAcutoff, log=log)

    with open('results/lsv/lsv_results.json', 'w') as results_file:
        json.dump(results, results_file, indent=2, cls=NumpyEncoder)
