#!/usr/bin/env python
"""ECSA normal mode: process CV files and compute ECSA."""

import os
import json
import re
import matplotlib.pyplot as plt
from datetime import datetime

from ..utils.serialization import NumpyEncoder
from ..utils.file_utils import find_and_sort_load_dta_files, find_cv_subfolders
from ..parsers.dta_parser import parse_dta_auto
from ..cv_processor import process_curve_data


def _make_process_callback(ECAcutoff):
    """Create a callback that processes a single curve with the given cutoff."""
    def callback(A, label):
        plt.subplot(2, 3, int(label))
        plt.plot(A[:, 2], A[:, 3])
        return process_curve_data(A, ECAcutoff)
    return callback


if __name__ == "__main__":
    ECAcutoff = 0.08
    searchKey = '*cv*.DTA'
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results/ecsa_normal/', exist_ok=True)
    log = open('logs/ecsa_normal.log', 'w')
    u_subfolders = find_cv_subfolders('./ECSA/', searchKey, log=log)
    results = {}

    for jk, file_path in enumerate(u_subfolders, start=1):
        file_info = find_and_sort_load_dta_files(file_path, search_key=searchKey)
        plt.figure(jk)
        results[f"dir_{jk}"] = {}

        for i, (mtime, filepath) in enumerate(file_info, 1):
            process_cb = _make_process_callback(ECAcutoff)
            data_dump = parse_dta_auto(filepath, process_cb, log=log)

            readable_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            results_jk_i = {
                "time_stamp": readable_time,
                "data": data_dump
            }
            results[f"dir_{jk}"][f"file_{i}"] = results_jk_i

        plt.savefig(f'results/ecsa_normal/ECSA_{jk}.png')

    with open('results/ecsa_normal/ecsa_results.json', 'w') as results_file:
        json.dump(results, results_file, indent=2, cls=NumpyEncoder)
    log.close()
