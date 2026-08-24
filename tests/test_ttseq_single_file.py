#!/usr/bin/env python
"""B7: ttseq must not crash when exactly ONE CSV parses successfully
(3/505 real cases).

Before the fix, ``plt.subplots(1, 3)`` returns a bare ndarray for a
single file and the old ``axes = [axes]`` wrap made ``axes[0]`` the
whole 3-element array -> ``AttributeError: 'numpy.ndarray' object has
no attribute 'plot'`` in plot_voltages (and the same pattern in
plot_Tcells).

After the fix: ``np.atleast_1d(axes).flatten()`` handles 1 and N files.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from meatools.subcomands.test_squence import plot_Tcells, plot_voltages
from tests.testdata_utils import write_hrl_csv


class _Log:
    def __init__(self):
        self.text = []

    def write(self, s):
        self.text.append(s)


def _result(name, columns):
    n = 30
    data = {c: np.full(n, 1.0) for c in columns}
    data['elapsed_time'] = np.arange(n, dtype=float)
    return {'file_info': f'case/{name}', 'data': data,
            'columns_found': list(columns)}


class TestSingleFileAxes:

    def test_plot_voltages_one_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'results' / 'test_sequence').mkdir(parents=True)
        plot_voltages({'A1': _result('only.csv',
                                     ['current', 'temp_coolant_inlet'])},
                      log=_Log())  # must not raise
        assert (tmp_path / 'results' / 'test_sequence'
                / 'voltage_plots.png').exists()

    def test_plot_tcells_one_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'results' / 'test_sequence').mkdir(parents=True)
        plot_Tcells({'A1': _result('only.csv',
                                   ['current', 'temp_coolant_inlet'])},
                    log=_Log())  # must not raise
        assert (tmp_path / 'results' / 'test_sequence'
                / 'temperature_plots.png').exists()

    def test_plot_voltages_two_files_still_works(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'results' / 'test_sequence').mkdir(parents=True)
        plot_voltages({'A1': _result('a.csv',
                                     ['current', 'temp_coolant_inlet']),
                       'A2': _result('b.csv',
                                     ['current', 'temp_coolant_inlet'])},
                      log=_Log())
        assert (tmp_path / 'results' / 'test_sequence'
                / 'voltage_plots.png').exists()

    def test_plot_voltages_six_files_still_works(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'results' / 'test_sequence').mkdir(parents=True)
        results = {f'A{i}': _result(f'f{i}.csv',
                                    ['current', 'temp_coolant_inlet'])
                   for i in range(1, 7)}
        plot_voltages(results, log=_Log())
        assert (tmp_path / 'results' / 'test_sequence'
                / 'voltage_plots.png').exists()


class TestTtseqSingleCsv:
    """Pipeline: the B7 repro shape - exactly one parseable CSV."""

    def test_ttseq_completes_with_one_csv(self, tmp_path):
        from datetime import datetime
        n = 120
        cols = {'Elapsed time': np.arange(1, n + 1, dtype=float),
                'cell_active_area': np.full(n, 5.0),
                'current_set': np.full(n, 1.0),
                'current': np.full(n, 1.0),
                'cell_voltage_001': np.full(n, 0.95),
                'HIOKI-01.resistance': np.full(n, 8.0),
                'temp_coolant_inlet': np.full(n, 80.0)}
        write_hrl_csv(tmp_path / 'HRL_Activation_80C - 20250915 0900.csv',
                      datetime(2025, 9, 15, 9, 0, 0), list(cols), cols)

        env = dict(os.environ, MPLBACKEND='Agg')
        proc = subprocess.run(
            [sys.executable, '-m', 'meatools.subcomands.test_squence'],
            cwd=str(tmp_path), env=env,
            capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, (
            f"ttseq crashed with a single CSV (B7):\n{proc.stderr[-2000:]}")
        assert 'Traceback' not in proc.stderr
        for rel in (
            'results/test_sequence/voltage_plots.png',
            'results/test_sequence/temperature_plots.png',
            'results/test_sequence/test_timeline.png',
            'results/polarization/polarization_results.json',
        ):
            assert (tmp_path / rel).exists(), f"missing output {rel}"