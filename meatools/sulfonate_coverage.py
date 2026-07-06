#!/usr/bin/env python
"""Sulfonate group coverage (磺酸根覆盖度) calculation.

The coverage is computed from two charge measurements:

    SO3_cover% = 2 * Q_CO-displace / Q_CO-stripping * 100%

where:

* Q_CO-displace  : charge of the CO displacement peak, averaged over the
                   2nd and 3rd parallel runs (units: C).
* Q_CO-stripping : CO stripping charge from the 1st CV cycle in the dry
                   proton accessibility test at 100% RH (units: C).

Data sources:

* CO displacement   : HRL CSV files under
                      <case>/磺酸根覆盖度/{1,2,3}/CO displace/*.csv
* CO stripping      : Gamry DTA files under
                      <case>/干质子可及率/100%RH/Cathode CO CV/*.DTA
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# HRL CSV helpers
# ---------------------------------------------------------------------------

def _find_data_start(df):
    """Locate the row that contains column headers in an HRL CSV.

    The HRL CSV files start with ~10 metadata rows, then two unit/label rows,
    then the real header row. We identify the header row by the presence of
    ``Elapsed time`` in column B.
    """
    for idx in range(df.shape[0]):
        row = df.iloc[idx]
        if len(row) > 1 and str(row.iloc[1]).strip().lower() == "elapsed time":
            return idx
    raise ValueError("Could not locate 'Elapsed time' header row in CSV")


def read_co_displace_csv(csv_path):
    """Read a CO displacement CSV and return time (s) and current (A)."""
    df = pd.read_csv(csv_path, header=None, encoding="latin1", low_memory=False)
    header_idx = _find_data_start(df)

    # Column B = Elapsed time, column BZ = Gamry-01.current (Excel BZ = 78)
    time_col = 1
    current_col = 77

    data = df.iloc[header_idx + 1 :].copy()
    data = data.dropna(how="all")
    time = pd.to_numeric(data.iloc[:, time_col], errors="coerce").dropna().values
    current = pd.to_numeric(data.iloc[:, current_col], errors="coerce").dropna().values

    if len(time) != len(current):
        min_len = min(len(time), len(current))
        time = time[:min_len]
        current = current[:min_len]

    return time.astype(float), current.astype(float)


def integrate_co_displace_peak(
    time,
    current,
    baseline_window_before=60.0,
    baseline_window_skip=10.0,
    peak_pre=13.0,
    peak_post=6.0,
):
    """Integrate the CO displacement peak with baseline subtraction.

    The peak is located by the most negative current value.  A flat baseline
    is estimated from a stable region immediately before the peak, and the
    integration window is ``[t_min - peak_pre, t_min + peak_post]``.

    Parameters
    ----------
    time, current : array-like
        Elapsed time (s) and Gamry current (A).
    baseline_window_before : float
        Length of the pre-peak region used for baseline estimation (s).
    baseline_window_skip : float
        Skip this many seconds immediately before the peak to avoid the
        transition region.
    peak_pre, peak_post : float
        Integration window extends ``peak_pre`` seconds before the peak
        minimum and ``peak_post`` seconds after it.

    Returns
    -------
    dict
        Charge (C), baseline (A), peak minimum time (s), and integration
        window edges (s).
    """
    time = np.asarray(time, dtype=float)
    current = np.asarray(current, dtype=float)

    if len(time) == 0 or len(current) == 0:
        raise ValueError("Empty time or current array")

    min_idx = int(np.argmin(current))
    t_min = time[min_idx]

    # Baseline from a stable region before the peak
    baseline_mask = (time >= t_min - baseline_window_before - baseline_window_skip) & (
        time <= t_min - baseline_window_skip
    )
    if not baseline_mask.any():
        baseline_mask = time <= t_min
    baseline = float(np.mean(current[baseline_mask]))

    left = t_min - peak_pre
    right = t_min + peak_post
    window_mask = (time >= left) & (time <= right)

    if not window_mask.any():
        raise ValueError(f"No data points in integration window [{left}, {right}]")

    charge = float(np.trapezoid(current[window_mask] - baseline, time[window_mask]))

    return {
        "charge": abs(charge),
        "baseline": baseline,
        "t_min": t_min,
        "t_left": left,
        "t_right": right,
    }


# ---------------------------------------------------------------------------
# Gamry DTA helpers
# ---------------------------------------------------------------------------

def read_dta_curves(dta_path):
    """Read all CURVE tables from a Gamry DTA file.

    Returns a list of pandas DataFrames, one per curve, with columns:
    Pt, T, Vf, Im, Vu, Sig, Ach, IERange, Over, Cycle.
    """
    curves = []
    current = None

    with open(dta_path, "r", encoding="latin1") as fh:
        for line in fh:
            if re.match(r"CURVE\d+\tTABLE", line):
                current = []
                curves.append(current)
                continue
            if current is None:
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 10 and parts[0].strip().isdigit():
                current.append(parts[:10])

    dataframes = []
    for idx, rows in enumerate(curves, start=1):
        if not rows:
            continue
        df = pd.DataFrame(
            rows,
            columns=["Pt", "T", "Vf", "Im", "Vu", "Sig", "Ach", "IERange", "Over", "Cycle"],
        )
        for col in ["Pt", "T", "Vf", "Im", "Cycle"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Curve"] = idx
        dataframes.append(df)

    return dataframes


def extract_upper_half(curve_df):
    """Return the anodic (upper) half of a CV cycle.

    The upper half is defined as the segment from the minimum voltage to the
    maximum voltage, following the data order (i.e. the upward voltage scan).
    """
    v = curve_df["Vf"].values
    min_idx = int(np.argmin(v))
    # After the minimum the voltage increases again until the end of the curve
    return curve_df.iloc[min_idx:].copy().reset_index(drop=True)


def integrate_co_stripping(dta_path, scan_rate=0.04, v_start=0.5):
    """Integrate the CO stripping peak from a Gamry CV DTA file.

    Uses the 2nd CV cycle's upper half as the baseline, subtracts it from the
    1st cycle's upper half, and integrates the resulting CO oxidation peak
    from ``v_start`` to the top of the 1st cycle.

    Parameters
    ----------
    dta_path : str or Path
        Path to the Gamry DTA file.
    scan_rate : float
        Voltage scan rate in V/s (default 0.04 V/s as used in the protocol).
    v_start : float
        Lower voltage bound of the integration (default 0.5 V).

    Returns
    -------
    dict
        Charge (C), integral I·V (A·V), and metadata about the cycles.
    """
    curves = read_dta_curves(dta_path)
    if len(curves) < 2:
        raise ValueError(f"Expected at least 2 CV cycles, found {len(curves)}")

    cycle1 = curves[0].reset_index(drop=True)
    cycle2 = curves[1].reset_index(drop=True)

    upper1 = extract_upper_half(cycle1)
    upper2 = extract_upper_half(cycle2)

    if upper1.empty or upper2.empty:
        raise ValueError("Could not extract upper half of a CV cycle")

    v1 = upper1["Vf"].values
    i1 = upper1["Im"].values
    v2 = upper2["Vf"].values
    i2 = upper2["Im"].values

    # Interpolate cycle-2 baseline onto cycle-1 voltage grid
    baseline = interp1d(
        v2, i2, kind="linear", bounds_error=False, fill_value="extrapolate"
    )(v1)

    mask = v1 >= v_start
    if not mask.any():
        raise ValueError(f"No data points above {v_start} V")

    iv = float(np.trapezoid((i1 - baseline)[mask], v1[mask]))
    charge = iv / scan_rate

    return {
        "charge": charge,
        "iv_integral": iv,
        "scan_rate": scan_rate,
        "v_start": v_start,
        "v1_range": (float(v1.min()), float(v1.max())),
        "v2_range": (float(v2.min()), float(v2.max())),
    }


# ---------------------------------------------------------------------------
# High-level case processing
# ---------------------------------------------------------------------------

def find_case_files(case_dir):
    """Locate the CO displacement CSVs and CO stripping DTA for a case.

    Returns a dict with keys ``co_displace`` (list of 3 file paths, runs 1-3)
    and ``co_stripping`` (single DTA path).
    """
    case_dir = Path(case_dir)
    sulfonate_dir = case_dir / "磺酸根覆盖度"
    dry_dir = case_dir / "干质子可及率" / "100%RH" / "Cathode CO CV"

    co_displace = []
    for run in ["1", "2", "3"]:
        run_dir = sulfonate_dir / run / "CO displace"
        files = sorted(run_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CO displacement CSV found in {run_dir}")
        co_displace.append(files[0])

    dta_files = sorted(dry_dir.glob("*.DTA"))
    if not dta_files:
        # Fallback to lowercase extension
        dta_files = sorted(dry_dir.glob("*.dta"))
    if not dta_files:
        raise FileNotFoundError(f"No CO stripping DTA found in {dry_dir}")

    return {"co_displace": co_displace, "co_stripping": dta_files[0]}


def process_case(case_dir, co_displace_kwargs=None):
    """Compute sulfonate coverage for one case folder.

    Returns a dict with intermediate values and the final coverage percentage.
    """
    files = find_case_files(case_dir)
    co_displace_kwargs = co_displace_kwargs or {}

    # Average the 2nd and 3rd parallel runs as described in the protocol
    displace_charges = []
    details = []
    for idx, csv_path in enumerate(files["co_displace"], start=1):
        time, current = read_co_displace_csv(csv_path)
        result = integrate_co_displace_peak(time, current, **co_displace_kwargs)
        displace_charges.append(result["charge"])
        details.append({"run": idx, "file": str(csv_path), **result})

    q_displace_2 = displace_charges[1]
    q_displace_3 = displace_charges[2]
    q_displace_avg = (q_displace_2 + q_displace_3) / 2.0

    strip_result = integrate_co_stripping(files["co_stripping"])
    q_stripping = strip_result["charge"]

    coverage = 2.0 * q_displace_avg / q_stripping * 100.0

    return {
        "case": Path(case_dir).name,
        "q_co_displace": {
            "run_1": displace_charges[0],
            "run_2": q_displace_2,
            "run_3": q_displace_3,
            "average_of_2_and_3": q_displace_avg,
        },
        "q_co_stripping": q_stripping,
        "so3_coverage_percent": coverage,
        "details": details,
        "co_stripping_detail": strip_result,
    }
