#!/usr/bin/env python
"""Unified CV curve processor with normal and dry modes."""

import os
import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import mstats

from .utils.io_utils import loadtxt_from_text, open_text


def _dedupe_by_voltage(arr):
    """Sort by voltage and drop consecutive duplicate voltages."""
    if arr.size == 0:
        return arr
    arr = arr[np.argsort(arr[:, 0])]
    if len(arr) == 1:
        return arr
    return arr[np.concatenate(([True], np.diff(arr[:, 0]) != 0)), :]


def process_curve_data(A, ECAcutoff):
    """Process a single CV curve and compute ECSA metrics.

    Args:
        A: Numpy array with CV data (columns 1=time, 2=voltage, 3=current).
        ECAcutoff: Lower voltage cutoff for UPD integration.

    Returns:
        Dict with scan rate, Vmin, double layer mean, and ECSA.
    """
    if A is None or len(A) == 0:
        raise ValueError("Empty CV array")
    if A.ndim != 2 or A.shape[1] < 4:
        raise ValueError("CV array must have at least 4 columns")

    with np.errstate(divide="ignore", invalid="ignore"):
        dV = np.diff(A[:, 2])
        dt = np.diff(A[:, 1])
        valid = dt != 0
        rates = np.abs(np.divide(dV, dt, where=valid))
        rates = rates[valid & np.isfinite(rates)]
    if len(rates) == 0 or np.median(rates) == 0:
        raise ValueError("Cannot determine scan rate from CV data")
    scanRate = float(np.median(rates))

    CVall = A[:, [2, 3]]
    updData = CVall[:, :2]

    # extract double1 & double2
    dV = np.concatenate(([0], np.diff(updData[:, 0])))
    mask1 = (updData[:, 0] > 0.3) & (updData[:, 0] < 0.6) & (dV > 0)
    double1 = updData[mask1, :]

    mask2 = (updData[:, 0] > 0.3) & (updData[:, 0] < 0.6) & (dV < 0)
    double2 = updData[mask2, :]

    double1 = _dedupe_by_voltage(double1)
    double2 = _dedupe_by_voltage(double2)
    if len(double1) < 2 or len(double2) < 2:
        raise ValueError(
            "Insufficient anodic/cathodic double-layer segments for ECSA"
        )

    # Interpolation
    x_new = np.arange(0.35, 0.451, 0.001)
    f1 = interpolate.interp1d(double1[:, 0], double1[:, 1], bounds_error=False)
    double1_interp = f1(x_new)
    f2 = interpolate.interp1d(double2[:, 0], double2[:, 1], bounds_error=False)
    double2_interp = f2(x_new)

    ddouble = np.abs(double1_interp - double2_interp)
    ddouble = ddouble[~np.isnan(ddouble)]
    if len(ddouble) == 0:
        raise ValueError("Double-layer interpolation produced no valid points")
    doubleMean = np.median(ddouble)

    # calculating ECSA
    updData = updData[np.concatenate(([True], np.diff(updData[:, 0]) > 0)), :]
    dl_mask = (updData[:, 0] > 0.4) & (updData[:, 0] < 0.6)
    if not np.any(dl_mask):
        raise ValueError("No double-layer region (0.4–0.6 V) in CV data")
    base = mstats.mquantiles(updData[dl_mask, 1], 0.25)
    updData[:, 1] -= base
    updData = updData[updData[:, 1] > 0, :]
    updData = updData[(updData[:, 0] <= 0.4) & (updData[:, 0] > ECAcutoff), :]
    updData = updData[np.argsort(updData[:, 0]), :]
    if len(updData) < 2:
        raise ValueError("Insufficient UPD points after filtering")

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

        with open_text(filepath) as f:
            lines = f.readlines()
        content_from_line65 = "".join(lines[65:]) if len(lines) > 65 else "".join(lines)

        u = re.split('CURVE', content_from_line65)

        for j in range(len(u)):
            try:
                A = loadtxt_from_text(u[j])
            except Exception:
                continue
            if A.ndim == 1:
                A = A.reshape(1, -1)
            plt.subplot(2, 3, j + 1)
            plt.plot(A[:, 2], A[:, 3])
            if j > 0 and j < (len(u) - 1):
                scanRate = np.median(np.abs(np.diff(A[:, 2]) / np.diff(A[:, 1])))
                CVall = A[:, [2, 3]]
                updData = CVall[:, :2]

                dV = np.concatenate(([0], np.diff(updData[:, 0])))
                mask1 = (updData[:, 0] > 0.3) & (updData[:, 0] < 0.6) & (dV > 0)
                double1 = _dedupe_by_voltage(updData[mask1, :])
                mask2 = (updData[:, 0] > 0.3) & (updData[:, 0] < 0.6) & (dV < 0)
                double2 = _dedupe_by_voltage(updData[mask2, :])

                # CO stripping area
                maskCO = (updData[:, 0] > 0.5) & (dV > 0)
                topLimits = updData[maskCO, :]
                if len(topLimits) >= 2:
                    fCO = interpolate.interp1d(topLimits[:, 0], topLimits[:, 1], bounds_error=False)
                    newUpper = fCO(x_CO)
                    plt.subplot(2, 3, 1)
                    plt.plot(x_CO, newUpper)
                    oldUpper = np.maximum(oldUpper, newUpper)

                # Interpolation for double layer
                if len(double1) >= 2 and len(double2) >= 2:
                    x_new = np.arange(0.35, 0.451, 0.001)
                    f1 = interpolate.interp1d(double1[:, 0], double1[:, 1], bounds_error=False)
                    double1_interp = f1(x_new)
                    f2 = interpolate.interp1d(double2[:, 0], double2[:, 1], bounds_error=False)
                    double2_interp = f2(x_new)

                    ddouble = np.abs(double1_interp - double2_interp)
                    ddouble = ddouble[~np.isnan(ddouble)]
                    doubleMean = float(np.median(ddouble)) if len(ddouble) else float("nan")
                else:
                    doubleMean = float("nan")

                # Process updData
                updData = updData[np.concatenate(([True], np.diff(updData[:, 0]) > 0)), :]
                dl_mask = (updData[:, 0] > 0.4) & (updData[:, 0] < 0.6)
                if np.any(dl_mask) and len(updData) >= 2:
                    base = mstats.mquantiles(updData[dl_mask, 1], 0.25)
                    updData[:, 1] = updData[:, 1] - base
                    updData = updData[updData[:, 1] > 0, :]
                    updData = updData[(updData[:, 0] <= 0.4) & (updData[:, 0] > ECAcutoff), :]
                    updData = updData[np.argsort(updData[:, 0]), :]

                    if len(updData) >= 2 and np.isfinite(scanRate) and scanRate != 0:
                        area = np.sum(np.diff(updData[:, 0]) * updData[:-1, 1])
                        QH = area / scanRate
                        ECA = QH / 2.1e-4
                    else:
                        ECA = float("nan")
                else:
                    ECA = float("nan")

                dump[f"curve_{j}"] = {
                    "rate (V/s)": scanRate,
                    "Vmin (V)": np.min(A[:, 2]),
                    "dd": doubleMean,
                    "ECA": ECA
                }

            elif j == 0:
                CVall = A[:, [2, 3]]
                updData = CVall[:, :2]
                dV = np.concatenate(([0], np.diff(updData[:, 0])))
                maskCO = (updData[:, 0] > 0.5) & (dV > 0)
                topLimits = updData[maskCO, :]
                if len(topLimits) >= 2:
                    fCO = interpolate.interp1d(topLimits[:, 0], topLimits[:, 1], bounds_error=False)
                    COdesorb = fCO(x_CO)

        data_dump[f"file_{i}"] = dump

    return oldUpper, COdesorb, data_dump
