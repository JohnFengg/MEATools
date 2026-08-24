#!/usr/bin/env python
"""ECSA dry mode: CO stripping CV analysis."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from ..utils.serialization import NumpyEncoder
from ..utils.file_utils import find_and_sort_load_dta_files, find_cv_subfolders
from ..cv_processor import plot_COtripping


if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results/ecsa_dry/', exist_ok=True)
    log = open('logs/ecsa_dry.log', 'w')
    index = 1
    results = {}

    # Nested pattern selects only "Cathode CO CV" folders (not plain Cathode CV).
    # Once those folders are found, load files with a simple filename pattern.
    # Re-using the nested pattern inside the folder looks for
    # <folder>/**/Cathode CO*/*cv*.DTA and matches nothing (regression after
    # the searchKey-parameter refactor).
    folder_search_key = 'Cathode CO*/*cv*.DTA'
    file_search_key = '*cv*.DTA'
    ECAcutoff = 0.08
    x_CO = np.arange(0.5, 0.95, 0.001)
    u_subfolders = find_cv_subfolders('.', folder_search_key, log=log)

    for i, folderName in enumerate(u_subfolders, start=1):
        results[f"dir_{i}"] = {}

        file_info = find_and_sort_load_dta_files(folderName, search_key=file_search_key)
        if not file_info:
            msg = f"No DTA files matching {file_search_key!r} in {folderName}\n"
            log.write(msg)
            print(msg, end="")
            continue

        plt.figure(index)
        index += 1
        try:
            oldUpper, COdesorb, data = plot_COtripping(
                file_info, ECAcutoff, x_CO, log=log)
        except Exception as exc:
            # Per-folder isolation: one bad folder must not prevent
            # ecsa_results.json from being written (B8).
            msg = (f"CO stripping processing failed in {folderName}: "
                   f"{type(exc).__name__}: {exc}\n")
            log.write(msg)
            print(msg, end="")
            results[f"dir_{i}"]["data"] = None
            results[f"dir_{i}"]["COECA"] = None
            continue
        plt.savefig(f'results/ecsa_dry/ECSA_Dry_{i}-1.png')

        if COdesorb is None:
            msg = f"CO desorb curve not found in {folderName}; skipping COECA plot\n"
            log.write(msg)
            print(msg, end="")
            results[f"dir_{i}"]["data"] = data
            results[f"dir_{i}"]["COECA"] = None
            continue

        plt.figure(index)
        index += 1
        plt.plot(x_CO, oldUpper, 'r--')
        plt.plot(x_CO, COdesorb, 'b-')
        COECA = np.nansum(np.mean(np.diff(x_CO)) * (COdesorb - oldUpper))
        plt.savefig(f'results/ecsa_dry/ECSA_Dry_{i}-2.png')

        results[f"dir_{i}"]["data"] = data
        results[f"dir_{i}"]["COECA"] = COECA
        with open('results/ecsa_dry/ecsa_results.json', 'w') as results_file:
            json.dump(results, results_file, indent=2, cls=NumpyEncoder)

    # Persist partial results even if some folders were skipped
    with open('results/ecsa_dry/ecsa_results.json', 'w') as results_file:
        json.dump(results, results_file, indent=2, cls=NumpyEncoder)

    log.close()
