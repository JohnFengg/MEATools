#!/usr/bin/env python
import numpy as np
import os
import re
import sys
import json
from collections import defaultdict
import cantera as ct
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

if __name__ == "__main__" and __package__ is None:
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _pkg_root = os.path.abspath(os.path.join(_here, '..', '..'))
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    from meatools.subcomands.test_squence import extract_data_from_file as edf
else:
    from .test_squence import extract_data_from_file as edf


class r_total_calc():
    def __init__(self, root_path, temp=80,
                 exclude_o2_fractions=None,
                 exclude_pressures=None,
                 label="original"):
        self.root = os.path.abspath(root_path)
        self.temp = temp + 273.15
        self.exclude_o2_fractions = set(exclude_o2_fractions or [])
        # Pressures are parsed as strings from filenames.
        self.exclude_pressures = set(str(p) for p in (exclude_pressures or []))
        self.label = label

    def parse_data_title(self):
        results = defaultdict(lambda: defaultdict(dict))
        for o2_dir in os.listdir(self.root):
            if o2_dir.startswith('.'):
                continue
            o2_path = os.path.join(self.root, o2_dir)
            if not os.path.isdir(o2_path):
                continue
            match = re.search(r"(\d+(?:\.\d+)?)%O2", o2_dir, re.IGNORECASE)
            if not match:
                continue
            o2_fraction = float(match.group(1)) / 100
            if o2_fraction in self.exclude_o2_fractions:
                continue
            for file in os.listdir(o2_path):
                if file.startswith('.') or not file.endswith('.csv'):
                    continue
                match = re.search(r"(\d+)\s*kPa[a]?", file, re.IGNORECASE)
                if not match:
                    continue
                pressure = match.group(1)
                if pressure in self.exclude_pressures:
                    continue
                filepath = os.path.join(o2_path, file)
                results[o2_fraction][pressure]['data'] = {}
                results[o2_fraction][pressure]['file_path'] = filepath

        if not results:
            # Flat layout (18/505 real cases): CSVs sit directly in OTR/
            # with the O2 fraction and pressure in the filename, e.g.
            # '..._1%O2_85C_80%RH_150kPaa - 20250618 0047 - part_0.csv'.
            # The nested scan above matches nothing here, which used to
            # leave every fit empty and crash curve_fit (B4).
            for file in os.listdir(self.root):
                if file.startswith('.') or not file.endswith('.csv'):
                    continue
                match = re.search(r"(\d+(?:\.\d+)?)%O2", file, re.IGNORECASE)
                if not match:
                    continue
                o2_fraction = float(match.group(1)) / 100
                if o2_fraction in self.exclude_o2_fractions:
                    continue
                match = re.search(r"(\d+)\s*kPa[a]?", file, re.IGNORECASE)
                if not match:
                    continue
                pressure = match.group(1)
                if pressure in self.exclude_pressures:
                    continue
                filepath = os.path.join(self.root, file)
                results[o2_fraction][pressure]['data'] = {}
                results[o2_fraction][pressure]['file_path'] = filepath

        self.init_results = self.recursive_to_dict(results)
        return self.init_results

    def parse_data(self, long_out=False):
        for conc, info in self.init_results.items():
            for pressure, data in info.items():
                file_path = data['file_path']
                try:
                    results = edf(file_path)
                    data['data']['current'] = list(results['data']['current'])
                    data['data']['current_density'] = list(
                        np.array(results['data']['current']) /
                        np.array(results['data']['cell_active_area'])
                    )
                    o_conc, dry = self.concentration_calc(
                        self.temp, float(pressure), float(conc))
                    data['data']['o_concentration'] = o_conc
                    data['data']['dry_pressure'] = dry
                except Exception as exc:
                    # One unreadable/incomplete CSV must not kill the
                    # whole OTR run (B4).
                    data['data']['error'] = f"{type(exc).__name__}: {exc}"
        if long_out:
            with open('results/impedence/raw_data.json', 'w') as f:
                json.dump(self.init_results, f, indent=2)

        return self.init_results

    def r_calc(self, fit_plot=False):
        results = defaultdict(lambda: defaultdict(list))
        for conc, info in self.init_results.items():
            for pressure, data in info.items():
                if 'current_density' not in data['data']:
                    continue  # file failed to parse (error recorded)
                cd = np.array(data['data']['current_density'])
                cut_off = int(len(cd) * 0.8)
                cd_avg = np.mean(cd[:cut_off])
                results[pressure]['current_density'].append(cd_avg)
                results[pressure]['o_concentration'].append(
                    data['data']['o_concentration'])
                results[pressure]['dry_pressure'].append(
                    data['data']['dry_pressure'])

        for pressure, v in results.items():
            o_conc = v['o_concentration']
            cur_d = v['current_density']
            try:
                fitted = self.fitting(o_conc, cur_d,
                                      prefix=f"{self.label}_{pressure}",
                                      plot=fit_plot)
            except Exception as exc:
                # Per-group isolation: one bad pressure group must not
                # kill the whole OTR run (B4).
                fitted = {"error": f"{type(exc).__name__}: {exc}"}
            else:
                r_total = 4 * 96485 / 1000 / fitted['a']
                results[pressure]['r_total'].append(r_total)
            results[pressure]['fit_stats'] = fitted

        self.fitted_results = self.recursive_to_dict(results)
        return self.fitted_results

    def run_calc(self, long_out=False, fit_plot=False):
        self.parse_data_title()
        self.parse_data(long_out)
        self.r_calc(fit_plot)
        pressures, rs_total = [], []
        for pressure, data in self.fitted_results.items():
            if not data.get('r_total'):
                continue  # this pressure group failed to fit
            pressures.append(data['dry_pressure'][0])
            rs_total.extend(data['r_total'])
        try:
            results = self.fitting(pressures, rs_total,
                                   prefix=f"{self.label}_final", plot=fit_plot)
        except Exception as exc:
            # No fittable pressure groups (e.g. nothing matched): record
            # the error instead of crashing the whole run (B4).
            results = {"error": f"{type(exc).__name__}: {exc}"}
        if 'error' in results:
            results["r_diff (s m^-1)"] = None
            results["r_other (s m^-1)"] = None
        else:
            r_diff = (101 * results['a'] + results['b']) * 100
            r_other = 101 * results['b']
            results["r_diff (s m^-1)"] = r_diff
            results["r_other (s m^-1)"] = r_other
        results['label'] = self.label
        results = self.recursive_to_dict(results)
        self.final_results = results
        return self.fitted_results, self.final_results

    @staticmethod
    def recursive_to_dict(d):
        if isinstance(d, defaultdict):
            d = {k: r_total_calc.recursive_to_dict(v)
                 for k, v in d.items()}
        elif isinstance(d, dict):
            d = {k: r_total_calc.recursive_to_dict(v)
                 for k, v in d.items()}
        return d

    @staticmethod
    def concentration_calc(temp, pressure, ratio):
        water_pressure = 57875         # unit-> Pa 85 C
        dry_pressure = pressure * 1000 - water_pressure  # unit-> Pa
        gas = ct.Solution('air.yaml')
        gas.TPX = temp, dry_pressure, {'O2': ratio, 'N2': 1 - ratio}
        # unit-> mol/L
        conc = gas.concentrations[gas.species_index('O2')]
        return conc, dry_pressure / 1000

    @staticmethod
    def fitting(x, y, prefix, plot=False):
        linear_func = lambda x, a, b: a * x + b
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.size == 0 or y.size == 0:
            raise ValueError("no data points to fit")
        if x.size < 2:
            raise ValueError(
                f"insufficient data points for a 2-parameter fit (n={x.size})")
        popt, pcov = curve_fit(linear_func, x, y)
        a, b = popt
        y_pred = linear_func(x, a, b)
        residuals = y - y_pred
        mse = np.mean(residuals ** 2)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        results = {'function': 'a*x+b',
                   'a': a,
                   'b': b,
                   'a_err': pcov[0][0],
                   'b_err': pcov[1][1],
                   'mse': mse,
                   'r2': r2}
        if plot:
            plt.figure()
            plt.scatter(x, y, label='Data')
            plt.plot(x, y_pred, 'r-', label=f'Fit:y={a:.3f}x+{b:.3f}')
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"Linear Fit (MSE={mse:.4g},R2={r2:.4f})")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'results/impedence/{prefix}_fitting.png')

        return results


def run_all_otr_groups(root_path='OTR/', temp=80,
                       long_out=False, fit_plot=False):
    """
    Run OTR analysis for three data groups:
      - original: use all valid data
      - exclude_1pct_o2_and_150kpa: exclude 1% O2 and 150 kPa
      - exclude_300kpa: exclude 300 kPa
    """
    configs = [
        {"label": "original",
         "exclude_o2_fractions": [],
         "exclude_pressures": []},
        {"label": "exclude_1pct_o2_and_150kpa",
         "exclude_o2_fractions": [0.01],
         "exclude_pressures": ["150"]},
        {"label": "exclude_300kpa",
         "exclude_o2_fractions": [],
         "exclude_pressures": ["300"]},
    ]

    all_fitted = {}
    all_final = {}
    if not os.path.isdir(root_path):
        # B14: manual 'mea otr' on a case without OTR/ used to die with a
        # raw FileNotFoundError from os.listdir. Write a friendly empty
        # result instead.
        print(f"[otr] Warning: no OTR folder at {root_path}; nothing to "
              f"analyze.", file=sys.stderr)
        for config in configs:
            note = {"note": f"OTR folder not found at {root_path}",
                    "r_diff (s m^-1)": None,
                    "r_other (s m^-1)": None,
                    "label": config["label"]}
            all_fitted[config["label"]] = {"note": note["note"]}
            all_final[config["label"]] = note
        os.makedirs('results/impedence', exist_ok=True)
        with open('results/impedence/fitted_r_total.json', 'w') as f:
            json.dump(all_fitted, f, indent=2)
        with open('results/impedence/final_results.json', 'w') as f:
            json.dump(all_final, f, indent=2)
        return all_fitted, all_final

    os.makedirs('results/impedence', exist_ok=True)
    for config in configs:
        calc = r_total_calc(
            root_path=root_path,
            temp=temp,
            exclude_o2_fractions=config["exclude_o2_fractions"],
            exclude_pressures=config["exclude_pressures"],
            label=config["label"]
        )
        try:
            fitted, final = calc.run_calc(long_out=long_out, fit_plot=fit_plot)
        except Exception as exc:
            # Per-config isolation (B4): one failing config must not
            # prevent the other configs and the results JSONs.
            fitted = {"error": f"{type(exc).__name__}: {exc}"}
            final = {"error": f"{type(exc).__name__}: {exc}"}
        all_fitted[config["label"]] = fitted
        all_final[config["label"]] = final

    os.makedirs('results/impedence', exist_ok=True)
    with open('results/impedence/fitted_r_total.json', 'w') as f:
        json.dump(all_fitted, f, indent=2)
    with open('results/impedence/final_results.json', 'w') as f:
        json.dump(all_final, f, indent=2)

    return all_fitted, all_final


if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results/impedence', exist_ok=True)
    root_path = 'OTR/'
    run_all_otr_groups(root_path=root_path, long_out=True, fit_plot=True)
