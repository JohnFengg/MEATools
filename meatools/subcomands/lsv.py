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
from ..utils.io_utils import loadtxt_from_text, open_text
from ..utils.plot_style import apply_unicode_font

apply_unicode_font()


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

            try:
                with open_text(filepath) as f:
                    lines = f.readlines()
                # LSV exports use a slightly shorter header than full CV DTA files.
                skip = 59 if len(lines) > 59 else 0
                content_from_line65 = "".join(lines[skip:])

                u = re.split('CURVE', content_from_line65)
                plt.subplot(2, 3, i)
                dump = {}
                for j in range(min(5, len(u))):
                    try:
                        A = loadtxt_from_text(u[j])
                    except Exception:
                        continue
                    if A.ndim == 1:
                        A = A.reshape(1, -1)
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
            except Exception as exc:
                # Per-file isolation: one bad LSV file must not prevent
                # lsv_results.json from being written (B8).
                if log:
                    log.write(f"\nFile processing failed for {filepath}: "
                              f"{type(exc).__name__}: {exc}\n")
                results[f"dir_{jk}"][str(i)]["data"] = {
                    "error": f"{type(exc).__name__}: {exc}"}

    plt.savefig('results/lsv/lsv_results.png')
    return results


if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results/lsv/', exist_ok=True)
    log = open('logs/lsv.log', 'w')

    searchKey = 'LSV/**/*.DTA'
    ECAcutoff = 0.08
    u_subfolders = find_cv_subfolders('.', searchKey, log=log)
    # The subfolders above already sit under the LSV tree, so the inner
    # search must not repeat the 'LSV/**' prefix (it used to, which made
    # lsv find zero files on every real case and write empty results).
    results = lsv_calc(u_subfolders, '*.DTA', ECAcutoff, log=log)

    with open('results/lsv/lsv_results.json', 'w') as results_file:
        json.dump(results, results_file, indent=2, cls=NumpyEncoder)
