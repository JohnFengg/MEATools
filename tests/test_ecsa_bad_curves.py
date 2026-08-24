#!/usr/bin/env python
"""B8: one bad CV curve/file must not abort the whole ecsa/lsv case.

Real-data failure: 3/505 ecsa cases crashed with
``ValueError: Insufficient UPD points after filtering`` (cv_processor)
raised from inside the per-curve callback, so ecsa_results.json was
never written. Same pattern of unguarded loops exists in ecsadry and
lsv (crash-free on the 505-case corpus but one bad file away).

After the fix: bad curves/files/folders are recorded as
``{"error": ...}`` and the remaining data is still processed, and the
results JSON is always written.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from meatools.cv_processor import process_curve_data
from meatools.parsers.dta_parser import parse_dta_auto, parse_dta_format2

PY = sys.executable


# ----------------------------------------------------------------------
# synthetic CV data
# ----------------------------------------------------------------------

def _cv_axes(n=200):
    up = np.linspace(0.05, 0.6, n // 2)
    down = np.linspace(0.6, 0.05, n // 2)[1:]
    v = np.concatenate([up, down])
    t = np.arange(len(v)) * 0.05
    return t, v


def _good_current(t, v):
    """Double layer + anodic UPD peak at 0.35 V on the anodic branch."""
    dv = np.diff(v, prepend=v[0])
    peak = 0.02 * np.exp(-((v - 0.35) / 0.04) ** 2) * (dv > 0)
    return 1e-4 + peak


def _bad_current(t, v):
    """Double layer only: low-voltage region strictly below the 0.4-0.6
    V baseline window -> no UPD points after filtering."""
    return np.where((v > 0.4) & (v <= 0.6), 1e-4, 0.5e-4)


def write_cv_dta(path, n=200):
    """Format-1 DTA: curves 0..4, with curve 2 lacking a UPD peak.

    parse_dta_format1 processes values v with 0 < v < n_values-1,
    i.e. curves 1, 2, 3 here (mirrors real files with values 0..4).
    """
    t, v = _cv_axes(n)
    currents = {
        '0': _good_current(t, v),
        '1': _good_current(t, v),
        '2': _bad_current(t, v),
        '3': _good_current(t, v),
        '4': _good_current(t, v),
    }
    rows = ['CURVE', '', 'curve data below']
    for value in ('0', '1', '2', '3', '4'):
        for k in range(len(t)):
            rows.append(f'{k} {t[k]:.4f} {v[k]:.5f} '
                        f'{currents[value][k]:.8f} 0 0 0 0 0 {value}')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows) + '\n')


class TestParserCurveIsolation:
    """dta_parser: per-curve exception isolation."""

    def test_format1_bad_curve_recorded_others_processed(self, tmp_path):
        p = tmp_path / 'cv.DTA'
        write_cv_dta(p)

        data_dump = parse_dta_auto(str(p), lambda A, label: process_curve_data(A, 0.08))

        assert 'ECA' in data_dump['curve_1']
        assert data_dump['curve_1']['ECA'] > 0
        assert data_dump['curve_2'] == {
            'error': 'ValueError: Insufficient UPD points after filtering'}
        assert 'ECA' in data_dump['curve_3']
        assert data_dump['file_path'] == str(p)

    def test_format2_bad_curve_recorded(self, tmp_path):
        p = tmp_path / 'cv2.DTA'
        pad = [f'PADLINE{i}' for i in range(66)]
        # loadtxt_from_text defaults: skiprows=2, usecols=range(8);
        # the newline after CURVE is the first skipped line
        block = 'x1\n1 2 3 4 5 6 7 8\n9 10 11 12 13 14 15 16\n'
        with open(p, 'w', encoding='utf-8') as f:
            f.write('\n'.join(pad) + '\n')
            f.write('CURVE\n' + block + 'CURVE\n' + block)

        def boom(A, j):
            raise ValueError('boom')

        data_dump = parse_dta_format2(str(p), boom)
        assert data_dump['curve_1'] == {'error': 'ValueError: boom'}


class TestPipelineIsolation:
    """Subcommand pipelines: results JSON written despite bad input."""

    def test_ecsa_normal_writes_results_with_bad_curve(self, tmp_path):
        folder = tmp_path / 'ECSA' / 'ECSA1'
        folder.mkdir(parents=True)
        write_cv_dta(folder / 'G17-0532-CV-01 Plot-cv data-250918 113044.DTA')

        env = {**os.environ, 'MPLBACKEND': 'Agg'}
        r = subprocess.run(
            [PY, '-m', 'meatools.subcomands.ecsa_normal'],
            cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, r.stderr[-800:]

        results = json.loads(
            (tmp_path / 'results' / 'ecsa_normal' / 'ecsa_results.json')
            .read_text(encoding='utf-8'))
        data = results['dir_1']['file_1']['data']
        assert 'ECA' in data['curve_1']
        assert data['curve_2']['error'] == (
            'ValueError: Insufficient UPD points after filtering')
        assert 'ECA' in data['curve_3']

    def test_lsv_writes_results_with_bad_file(self, tmp_path):
        # one data row with voltage outside 0.3-0.6 V -> empty double1
        # -> scipy interp1d ValueError in lsv_calc
        folder = tmp_path / 'LSV' / 'sweep1'
        folder.mkdir(parents=True)
        (folder / 'test-lsv.DTA').write_text(
            'CURVE\nx1\n0 0.1 0.2 5 6 7 8 9\n', encoding='utf-8')

        env = {**os.environ, 'MPLBACKEND': 'Agg'}
        r = subprocess.run(
            [PY, '-m', 'meatools.subcomands.lsv'],
            cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, r.stderr[-800:]

        results = json.loads(
            (tmp_path / 'results' / 'lsv' / 'lsv_results.json')
            .read_text(encoding='utf-8'))
        entry = results['dir_0']['1']
        assert 'error' in entry['data']
        assert 'ValueError' in entry['data']['error']

    def test_ecsa_dry_writes_results_with_bad_folder(self, tmp_path):
        # Real layout: 干质子可及率/<RH>/Cathode CO CV/*cv*.DTA
        # 3 CURVE blocks; each block's 2 lines are consumed by skiprows=2
        # -> (0, 8) middle array -> boolean-index IndexError in
        # plot_COtripping (the leading newline after CURVE is line 1)
        folder = tmp_path / '干质子可及率' / '100%RH' / 'Cathode CO CV'
        folder.mkdir(parents=True)
        block = 'x1\n'
        (folder / 'G17-test-cv data-1.DTA').write_text(
            'CURVE\n' + block + 'CURVE\n' + block, encoding='utf-8')

        env = {**os.environ, 'MPLBACKEND': 'Agg'}
        r = subprocess.run(
            [PY, '-m', 'meatools.subcomands.ecsa_dry'],
            cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, r.stderr[-800:]

        results = json.loads(
            (tmp_path / 'results' / 'ecsa_dry' / 'ecsa_results.json')
            .read_text(encoding='utf-8'))
        assert results['dir_1']['data'] is None
        assert results['dir_1']['COECA'] is None