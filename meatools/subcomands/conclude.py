#!/usr/bin/env python
import json
import os
import re
import warnings

import numpy as np

# Candidate paths for sulfonate coverage (relative to case cwd).
# Interactive sulf-cvrg writes results/sulf-cvrg/; older runs used case root.
SULF_CANDIDATES = (
    "results/sulf-cvrg/sulfonate_coverage.json",
    "results/sulfonate_coverage.json",
    "sulfonate_coverage.json",
)


def read_json_path(path, index=None):
    """Load a JSON file as UTF-8.

    Sulfonate (and other) outputs may contain non-ASCII paths written with
    ensure_ascii=False. On Windows the default locale is often GBK/cp936;
    decoding those files as GBK raises UnicodeDecodeError.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if index is not None:
            data = data[index]
        return data
    except FileNotFoundError:
        return {}
    except Exception as exc:
        warnings.warn(
            f"Failed to read {path}: {type(exc).__name__}: {exc}",
            UserWarning,
        )
        return {}


def read_json(filename, index=None):
    """Load a JSON file from results/ (UTF-8)."""
    return read_json_path(f"results/{filename}", index=index)


def read_sulfonate_coverage():
    """Load sulfonate coverage from known output locations."""
    for path in SULF_CANDIDATES:
        if not os.path.isfile(path):
            continue
        data = read_json_path(path)
        if data:
            print(f"[conclude] Sulfonate_Coverage loaded from {path}")
            return data
        warnings.warn(f"Sulfonate file exists but is empty/unreadable: {path}", UserWarning)
    print(
        "[conclude] Sulfonate_Coverage not found. Looked for:\n  - "
        + "\n  - ".join(SULF_CANDIDATES)
    )
    return {}


def extract_row(data, key, target):
    arr = np.array(data[key])
    idx = np.argmin(np.abs(arr - target))
    for k, v in data.items():
        if isinstance(v, list):
            data[k] = v[idx]
    return data


def strip_filename(filename: str) -> str:
    """
    sample:
    'HRL_D048_05_Polarisation02_80C_H2_60%O2_2_5_250kpa - 20250816 1528.csv'
    -> 'HRL_D048_05_Polarisation02_80C_H2_60%O2_2_5_250kpa'
    """
    return re.split(r" - \d{8} \d{4}", filename)[0]


def pairwise_average(big_dict):
    items = list(big_dict.items())
    length = len(items)
    results = {}

    for i in range(0, length, 2):
        if i + 1 >= length:
            name, data = items[i]
            results[strip_filename(name)] = data
            raise Warning(
                "Odd files number is detected, please check the test results of polarization"
            )

        name1, dict1 = items[i]
        name2, dict2 = items[i + 1]
        new_name = strip_filename(name1)

        avg_dict = {}
        for key in dict1.keys():
            v1, v2 = dict1[key], dict2[key]
            if isinstance(v1, list):
                arr = np.array([v1, v2])
                avg_dict[key] = arr.mean(axis=0).tolist()
            else:
                avg_dict[key] = float((v1 + v2) / 2)

        results[new_name] = avg_dict

    return results


def _safe_avg_ecsa(esca):
    for _dir, results in esca.items():
        if not isinstance(results, dict):
            continue
        for _file, info in results.items():
            if not isinstance(info, dict) or "data" not in info:
                continue
            eca, dd = [], []
            for _curve, value in info["data"].items():
                if isinstance(value, dict) and "ECA" in value:
                    eca.append(value["ECA"])
                    dd.append(value["dd"])
            if eca:
                info["avg_ECA"] = float(np.mean(eca))
                info["avg_dd"] = float(np.mean(dd))


def _safe_avg_lsv(lsv):
    if not lsv:
        return
    slopereg = []
    last_results = None
    for _dir, results in lsv.items():
        last_results = results
        if not isinstance(results, dict):
            continue
        for _file, info in results.items():
            if not isinstance(info, dict) or "data" not in info:
                continue
            for _curve, value in info["data"].items():
                if isinstance(value, dict) and "slopeReg" in value:
                    slopereg.append(value["slopeReg"])
    if slopereg and isinstance(last_results, dict):
        last_results["avg_slopereg"] = float(np.mean(slopereg))


def _safe_avg_ecsa_dry(ecsa_dry):
    for _dir, results in ecsa_dry.items():
        if not isinstance(results, dict) or "data" not in results:
            continue
        for _file, info in results["data"].items():
            if not isinstance(info, dict):
                continue
            eca, dd = [], []
            for _curve, value in info.items():
                if isinstance(value, dict) and "ECA" in value:
                    eca.append(value["ECA"])
                    dd.append(value["dd"])
            if eca:
                info["avg_ECA"] = float(np.mean(eca))
                info["avg_dd"] = float(np.mean(dd))


if __name__ == "__main__":
    result_tol = {}
    sample_name = os.path.basename(os.getcwd())

    match = re.search(r"\((.*?)\)", sample_name)
    station = match.group(1) if match else None

    sample_areas = []
    for _file, data in read_json("test_sequence/all_csv_results_in_timeline.json").items():
        try:
            sa = float(data["data"]["cell_active_area"][0])
            sample_areas.append(sa)
        except Exception:
            continue
    if not sample_areas:
        warnings.warn(
            "No cell_active_area found in test sequence; sample_area set to None",
            UserWarning,
        )
        sample_area = None
    else:
        if len(set(sample_areas)) != 1:
            warnings.warn(
                "Inconsistent values of cell_active_area are detected, please check!",
                UserWarning,
            )
        sample_area = sample_areas[0]

    test_seq = read_json("test_sequence/test_order_in_timeline.json")

    esca = read_json("ecsa_normal/ecsa_results.json")
    try:
        _safe_avg_ecsa(esca)
    except Exception as exc:
        warnings.warn(f"ECSA post-process failed: {exc}", UserWarning)

    lsv = read_json("lsv/lsv_results.json")
    try:
        _safe_avg_lsv(lsv)
    except Exception as exc:
        warnings.warn(f"LSV post-process failed: {exc}", UserWarning)

    pol = read_json("polarization/polarization_results.json")
    for _file, data in list(pol.items()):
        try:
            extract_row(data, "current density (A cm^(-2))", 1)
        except Exception:
            pass
    try:
        pol = pairwise_average(pol)
    except Exception:
        pass

    otr = read_json("impedence/final_results.json")

    ecsa_dry = read_json("ecsa_dry/ecsa_results.json")
    try:
        _safe_avg_ecsa_dry(ecsa_dry)
    except Exception as exc:
        warnings.warn(f"ECSA_Dry post-process failed: {exc}", UserWarning)

    eis = read_json("eis/eis_results.json")

    # Load sulf LAST and independently so earlier section failures cannot skip it.
    sulf_cvrg = read_sulfonate_coverage()

    result_tol["sample"] = sample_name
    result_tol["station_num."] = station
    result_tol["sample_area (cm^2)"] = sample_area
    result_tol["Test_Sequence"] = test_seq
    result_tol["ECSA"] = esca
    result_tol["ECSA_Dry"] = ecsa_dry
    result_tol["LSV"] = lsv
    result_tol["Polarization"] = pol
    result_tol["O_Transfer_Resistance"] = otr
    result_tol["EIS"] = eis
    result_tol["Sulfonate_Coverage"] = sulf_cvrg

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(result_tol, f, indent=2, ensure_ascii=False)

    if sulf_cvrg and "so3_coverage_percent" in sulf_cvrg:
        print(
            f"[conclude] Wrote results.json with SO3 coverage = "
            f"{sulf_cvrg['so3_coverage_percent']:.4f}%"
        )
    else:
        print("[conclude] Wrote results.json; Sulfonate_Coverage is empty")
