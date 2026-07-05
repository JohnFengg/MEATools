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
    log = open('logs/ecsa_normal.log', 'w')
    index = 1
    results = {}

    searchKey = 'Cathode CO*/*cv*.DTA'
    ECAcutoff = 0.08
    x_CO = np.arange(0.5, 0.95, 0.001)
    u_subfolders = find_cv_subfolders('.', searchKey, log=log)

    for i, folderName in enumerate(u_subfolders, start=1):
        results[f"dir_{i}"] = {}

        plt.figure(index)
        index += 1
        file_info = find_and_sort_load_dta_files(u_subfolders[i - 1], search_key=searchKey)
        oldUpper, COdesorb, data = plot_COtripping(file_info, ECAcutoff, x_CO, log=log)
        plt.savefig(f'results/ecsa_dry/ECSA_Dry_{i}-1.png')

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

    log.close()
