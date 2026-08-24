#!/usr/bin/env python
"""B6: ttseq must not crash on a CSV with all-NaN elapsed time (4/505).

Before the fix, ``duration = np.max(elapsed_time)/60 = NaN`` reached
``timedelta(minutes=step['duration'])`` ->
``ValueError: cannot convert float NaN to integer`` *before*
test_order_in_timeline.json was written (in both the main timeline and
the polarization loop).

After the fix such files are skipped with a log line; the rest of the
case is processed and all outputs are written.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from tests.testdata_utils import write_hrl_csv
from tests.test_polarization_results import _pol_profile


def _run_ttseq(case_dir):
    env = dict(os.environ, MPLBACKEND='Agg')
    return subprocess.run(
        [sys.executable, '-m', 'meatools.subcomands.test_squence'],
        cwd=str(case_dir), env=env,
        capture_output=True, text=True, timeout=600)


def _good_csv(path, start):
    n = 120
    cols = {'Elapsed time': np.arange(1, n + 1, dtype=float),
            'cell_active_area': np.full(n, 5.0),
            'current_set': np.full(n, 1.0),
            'current': np.full(n, 1.0),
            'cell_voltage_001': np.full(n, 0.95),
            'HIOKI-01.resistance': np.full(n, 8.0),
            'temp_coolant_inlet': np.full(n, 80.0)}
    write_hrl_csv(path, start, list(cols), cols)


def _nan_elapsed_csv(path, start, name='HRL_Pol_nanElapsed-1.csv'):
    """CSV with a parseable start time but an all-NaN Elapsed time."""
    n = 120
    cols = {'Elapsed time': [float('nan')] * n,
            'cell_active_area': [5.0] * n,
            'current_set': [1.0] * n,
            'current': [1.0] * n,
            'cell_voltage_001': [0.95] * n,
            'HIOKI-01.resistance': [8.0] * n,
            'temp_coolant_inlet': [80.0] * n}
    write_hrl_csv(path / name, start, list(cols), cols)


class TestTtseqWithNanElapsed:

    def test_nan_file_skipped_good_file_processed(self, tmp_path):
        from datetime import datetime
        # two good files (a single plottable file would trip the
        # separate, still-open B7 single-file axes bug)
        _good_csv(tmp_path / 'HRL_Activation_80C - 20250915 0900.csv',
                  datetime(2025, 9, 15, 9, 0, 0))
        _good_csv(tmp_path / 'HRL_Activation2_80C - 20250915 1000.csv',
                  datetime(2025, 9, 15, 11, 0, 0))
        _nan_elapsed_csv(tmp_path, datetime(2025, 9, 15, 10, 0, 0))

        proc = _run_ttseq(tmp_path)
        assert proc.returncode == 0, (
            f"ttseq crashed on all-NaN elapsed time (B6):\n"
            f"{proc.stderr[-2000:]}")
        assert 'Traceback' not in proc.stderr

        # timeline JSON written (previously missing), only the good files
        # (the NaN file sat between them in time order, so the steps are
        # numbered 1 and 3)
        order = json.loads(
            (tmp_path / 'results' / 'test_sequence'
             / 'test_order_in_timeline.json').read_text(encoding='utf-8'))
        steps = [k for k in order if k != 'total_time (mins)']
        assert steps == ['1', '3']

        log_text = (tmp_path / 'logs' / 'test_sequence.log'
                    ).read_text(encoding='utf-8', errors='replace')
        assert 'all NaN' in log_text

    def test_all_files_nan_elapsed(self, tmp_path):
        from datetime import datetime
        _nan_elapsed_csv(tmp_path, datetime(2025, 9, 15, 9, 0, 0),
                         name='HRL_Pol_a-1.csv')
        _nan_elapsed_csv(tmp_path, datetime(2025, 9, 15, 10, 0, 0),
                         name='HRL_Pol_b-2.csv')

        proc = _run_ttseq(tmp_path)
        assert proc.returncode == 0, (
            f"ttseq crashed (B6):\n{proc.stderr[-2000:]}")
        assert 'Traceback' not in proc.stderr

        order = json.loads(
            (tmp_path / 'results' / 'test_sequence'
             / 'test_order_in_timeline.json').read_text(encoding='utf-8'))
        assert set(order.keys()) == {'total_time (mins)'}
        # polarization half still ran
        assert (tmp_path / 'results' / 'polarization'
                / 'polarization_results.json').exists()

    def test_nan_pol_file_skipped_in_pol_half(self, tmp_path):
        from datetime import datetime
        pol = tmp_path / '极化'
        pol.mkdir()
        prof = _pol_profile(0.90)
        write_hrl_csv(pol / 'HRL_Pol_good-1.csv',
                      datetime(2025, 9, 15, 9, 0, 0),
                      list(prof), prof)
        # a non-pol good file so the main half has >= 2 plottable files
        # (single plottable file trips the separate B7 axes bug)
        _good_csv(tmp_path / 'HRL_Activation_80C - 20250915 0800.csv',
                  datetime(2025, 9, 15, 8, 0, 0))
        _nan_elapsed_csv(pol, datetime(2025, 9, 15, 10, 0, 0))

        proc = _run_ttseq(tmp_path)
        assert proc.returncode == 0, proc.stderr[-2000:]

        pol_results = json.loads(
            (tmp_path / 'results' / 'polarization'
             / 'polarization_results.json').read_text(encoding='utf-8'))
        # only the good pol file produced analysis
        assert len(pol_results) == 1
        assert 'HRL_Pol_good-1.csv' in pol_results