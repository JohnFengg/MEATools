#!/usr/bin/env python
"""Unified CV curve processor with normal and dry modes."""

import os
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import mstats


def process_curve_data(A, ECAcutoff):
    """Process a single CV curve and compute ECSA metrics.

    Args:
        A: Numpy array with CV data (columns 1=time, 2=voltage, 3=current).
        ECAcutoff: Lower voltage cutoff for UPD integration.

    Returns:
        Dict with scan rate, Vmin, double layer mean, and ECSA.
    """
    scanRate = np.median(np.abs(np.diff(A[:, 2]) / np.diff(A[:, 1])))
    CVall = A[:, [2, 3]]
    updData = CVall[:, :2]

    # extract double1 & double2
    mask1 = (updData[:, 0] > 0.3) & (updData[:, 0] < 0.6) & \
            (np.concatenate(([0], np.diff(updData[:, 0]))) > 0)
    double1 = updData[mask1, :]

    mask2 = (updData[:, 0] > 0.3) & (updData[:, 0] < 0.6) & \
            (np.concatenate(([0], np.diff(updData[:, 0]))) < 0)
    double2 = updData[mask2, :]

    # ranking and deduplication
    double1 = double1[np.argsort(double1[:, 0])]
    double2 = double2[np.argsort(double2[:, 0])]
    double1 = double1[np.concatenate(([True], np.diff(double1[:, 0]) != 0)), :]
    double2 = double2[np.concatenate(([True], np.diff(double2[:, 0]) != 0)), :]

    # Interpolation
    x_new = np.arange(0.35, 0.451, 0.001)
    f1 = interpolate.interp1d(double1[:, 0], double1[:, 1], bounds_error=False)
    double1_interp = f1(x_new)
    f2 = interpolate.interp1d(double2[:, 0], double2[:, 1], bounds_error=False)
    double2_interp = f2(x_new)

    ddouble = np.abs(double1_interp - double2_interp)
    ddouble = ddouble[~np.isnan(ddouble)]
    doubleMean = np.median(ddouble)

    # calculating ECSA
    updData = updData[np.concatenate(([True], np.diff(updData[:, 0]) > 0)), :]
    base = mstats.mquantiles(updData[(updData[:, 0] > 0.4) & (updData[:, 0] < 0.6), 1], 0.25)
    updData[:, 1] -= base
    updData = updData[updData[:, 1] > 0, :]
    updData = updData[(updData[:, 0] <= 0.4) & (updData[:, 0] > ECAcutoff), :]
    updData = updData[np.argsort(updData[:, 0]), :]

    area = np.sum(np.diff(updData[:, 0]) * updData[:-1, 1])  # in mAV
    QH = area / scanRate
    ECA = QH / 2.1e-4

    return {
        "rate (V/s)": scanRate,
        "Vmin (V)": np.min(A[:, 2]),
        "dd": doubleMean,
        "ECA": ECA
    }


def plot_COtripping(file_info, ECAcutoff, x_CO, log=None):
    """Process CO stripping CV files (dry mode).

    Args:
        file_info: List of (mtime, filepath) tuples.
        ECAcutoff: Lower voltage cutoff for UPD integration.
        x_CO: Voltage array for CO stripping interpolation.
        log: Optional file-like object for logging.

    Returns:
        Tuple of (oldUpper, COdesorb, data_dump).
    """
    data_dump = {}
    oldUpper = x_CO * 0
    COdesorb = None

    for i, (mtime, filepath) in enumerate(file_info, 1):
        dump = {}
        readable_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        dump["time_stamp"] = readable_time
        dump["file"] = filepath

        with open(filepath, 'r') as f:
            for _ in range(65):
                next(f)
            content_from_line65 = f.read()

        u = re.split('CURVE', content_from_line65)

        for j in range(len(u)):
            with open('temp', 'w') as fileID:
                fileID.write(u[j])
            A = np.loadtxt('temp', skiprows=2, usecols=range(8))
            plt.subplot(2, 3, j + 1)
            plt.plot(A[:, 2], A[:, 3])
            if j > 0 and j < (len(u) - 1):
                scanRate = np.median(np.abs(np.diff(A[:, 2]) / np.diff(A[:, 1])))
                CVall = A[:, [2, 3]]
                updData = CVall[:, :2]

                mask1 = (updData[:, 0] > 0.3) & (updData[:, 0] < 0.6) & \
                        (np.concatenate(([0], np.diff(updData[:, 0]))) > 0)
                double1 = updData[mask1, :]
                mask2 = (updData[:, 0] > 0.3) & (updData[:, 0] < 0.6) & \
                        (np.concatenate(([0], np.diff(updData[:, 0]))) < 0)
                double2 = updData[mask2, :]

                double1 = double1[np.argsort(double1[:, 0])]
                double2 = double2[np.argsort(double2[:, 0])]
                double1 = double1[np.concatenate(([True], np.diff(double1[:, 0]) != 0)), :]
                double2 = double2[np.concatenate(([True], np.diff(double2[:, 0]) != 0)), :]

                # CO stripping area
                maskCO = (updData[:, 0] > 0.5) & (np.concatenate(([0], np.diff(updData[:, 0]))) > 0)
                topLimits = updData[maskCO, :]
                fCO = interpolate.interp1d(topLimits[:, 0], topLimits[:, 1], bounds_error=False)
                newUpper = fCO(x_CO)
                plt.subplot(2, 3, 1)
                plt.plot(x_CO, newUpper)
                oldUpper = np.maximum(oldUpper, newUpper)

                # Interpolation for double layer
                x_new = np.arange(0.35, 0.451, 0.001)
                f1 = interpolate.interp1d(double1[:, 0], double1[:, 1], bounds_error=False)
                double1_interp = f1(x_new)
                f2 = interpolate.interp1d(double2[:, 0], double2[:, 1], bounds_error=False)
                double2_interp = f2(x_new)

                ddouble = np.abs(double1_interp - double2_interp)
                ddouble = ddouble[~np.isnan(ddouble)]
                doubleMean = np.median(ddouble)

                # Process updData
                updData = updData[np.concatenate(([True], np.diff(updData[:, 0]) > 0)), :]
                base = mstats.mquantiles(updData[(updData[:, 0] > 0.4) & (updData[:, 0] < 0.6), 1], 0.25)
                updData[:, 1] = updData[:, 1] - base
                updData = updData[updData[:, 1] > 0, :]
                updData = updData[(updData[:, 0] <= 0.4) & (updData[:, 0] > ECAcutoff), :]
                updData = updData[np.argsort(updData[:, 0]), :]

                area = np.sum(np.diff(updData[:, 0]) * updData[:-1, 1])
                QH = area / scanRate
                ECA = QH / 2.1e-4

                dump[f"curve_{j}"] = {
                    "rate (V/s)": scanRate,
                    "Vmin (V)": np.min(A[:, 2]),
                    "dd": doubleMean,
                    "ECA": ECA
                }

            elif j == 0:
                CVall = A[:, [2, 3]]
                updData = CVall[:, :2]
                maskCO = (updData[:, 0] > 0.5) & (np.concatenate(([0], np.diff(updData[:, 0]))) > 0)
                topLimits = updData[maskCO, :]
                fCO = interpolate.interp1d(topLimits[:, 0], topLimits[:, 1], bounds_error=False)
                COdesorb = fCO(x_CO)
            if os.path.exists('temp'):
                os.remove('temp')

        data_dump[f"file_{i}"] = dump

    return oldUpper, COdesorb, data_dump
