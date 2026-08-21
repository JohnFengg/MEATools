#!/usr/bin/env python
"""B1: ttseq must not crash on CSVs without a 'current' column.

Real datasets contain pressure-holding CSVs (elapsed_time +
pressure_cathode_inlet only), voltage-only CSVs and elapsed-only logger
CSVs. The generic CSV scan picks all of them up, and the plotting
functions used to index data['current'] / data['temp_coolant_inlet'] /
data['cell_voltage_001'] unconditionally -> KeyError killed the whole
ttseq run (348/505 cases) before any polarization output was written.
"""

import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pytest

from meatools.subcomands.test_squence import plot_Pol, plot_Tcells, plot_voltages
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


def _run_module(module, case_dir):
    env = dict(os.environ, MPLBACKEND='Agg')
    return subprocess.run([sys.executable, '-m', module], cwd=str(case_dir),
                          env=env, capture_output=True, text=True, timeout=600)


class TestPlotGuards:
    """Unit tests: plots skip files lacking columns instead of crashing."""

    def test_plot_voltages_skips_files_without_current(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'results' / 'test_sequence').mkdir(parents=True)
        all_results = {
            'A1': _result('good1.csv',
                          ['current', 'temp_coolant_inlet']),
            'A2': _result('good2.csv',
                          ['current', 'temp_coolant_inlet']),
            'A3': _result('pressure.csv', ['pressure_cathode_inlet']),
        }
        log = _Log()
        plot_voltages(all_results, log=log)  # must not raise
        assert any('pressure.csv' in s and 'current' in s for s in log.text)
        assert (tmp_path / 'results' / 'test_sequence'
                / 'voltage_plots.png').exists()

    def test_plot_voltages_all_missing_returns_quietly(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        all_results = {'A1': _result('pressure.csv', ['pressure_cathode_inlet'])}
        plot_voltages(all_results, log=_Log())  # must not raise

    def test_plot_tcells_skips_files_without_temperature(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'results' / 'test_sequence').mkdir(parents=True)
        all_results = {
            'A1': _result('good1.csv', ['current', 'temp_coolant_inlet']),
            'A2': _result('good2.csv', ['current', 'temp_coolant_inlet']),
            'A3': _result('pressure.csv', ['pressure_cathode_inlet']),
        }
        log = _Log()
        plot_Tcells(all_results, log=log)  # must not raise
        assert any('pressure.csv' in s for s in log.text)

    def test_plot_pol_skips_files_without_voltage_or_current(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'results' / 'polarization').mkdir(parents=True)
        all_pol = {
            'A1': _result('good.csv', ['current', 'cell_voltage_001']),
            'A2': _result('pressure.csv', ['pressure_cathode_inlet']),
        }
        log = _Log()
        plot_Pol(all_pol, 5.0, log=log)  # must not raise
        assert any('pressure.csv' in s for s in log.text)
        assert (tmp_path / 'results' / 'polarization'
                / 'polarization_curves.png').exists()


class TestTtseqWithColumnPoorCsvs:
    """Pipeline: mixed case (good + pressure/voltage-only/elapsed-only CSVs)."""

    @pytest.fixture
    def mixed_case(self, tmp_path):
        case = tmp_path
        (case / 'pressure').mkdir()

        def full_cols(n=120):
            return {
                'Elapsed time': np.arange(1, n + 1, dtype=float),
                'cell_active_area': np.full(n, 5.0),
                'current_set': np.full(n, 1.0),
                'current': np.full(n, 1.0),
                'cell_voltage_001': np.full(n, 0.95),
                'HIOKI-01.resistance': np.full(n, 8.0),
                'temp_coolant_inlet': np.full(n, 80.0),
            }

        write_hrl_csv(case / 'HRL_Activation_80C_100%RH - 20250915 0900.csv',
                      datetime(2025, 9, 15, 9, 0, 0),
                      list(full_cols().keys()), full_cols())
        write_hrl_csv(case / 'HRL_Activation2_80C_100%RH - 20250915 1000.csv',
                      datetime(2025, 9, 15, 10, 0, 0),
                      list(full_cols().keys()), full_cols())

        # The three B1 crash classes:
        write_hrl_csv(
            case / 'pressure' / 'HRL_pressure holding - 250915 092755 - part_0.csv',
            datetime(2025, 9, 15, 9, 27, 55),
            ['Elapsed time', 'pressure_cathode_inlet'],
            {'Elapsed time': np.arange(1, 61, dtype=float),
             'pressure_cathode_inlet': np.full(60, 150.0)})
        write_hrl_csv(
            case / 'HRL_voltage_only - 250915 110000.csv',
            datetime(2025, 9, 15, 11, 0, 0),
            ['Elapsed time', 'cell_voltage_001'],
            {'Elapsed time': np.arange(1, 61, dtype=float),
             'cell_voltage_001': np.full(60, 0.95)})
        write_hrl_csv(
            case / 'HRL_elapsed_only - 250915 120000.csv',
            datetime(2025, 9, 15, 12, 0, 0),
            ['Elapsed time'],
            {'Elapsed time': np.arange(1, 61, dtype=float)})
        return case

    def test_ttseq_completes_and_writes_all_outputs(self, mixed_case):
        proc = _run_module('meatools.subcomands.test_squence', mixed_case)
        assert proc.returncode == 0, (
            f"ttseq crashed on column-poor CSVs (B1):\n{proc.stderr[-2000:]}")
        assert 'Traceback' not in proc.stderr

        for rel in (
            'results/test_sequence/all_csv_results_in_timeline.json',
            'results/test_sequence/test_order_in_timeline.json',
            'results/test_sequence/voltage_plots.png',
            'results/test_sequence/temperature_plots.png',
            'results/test_sequence/test_timeline.png',
            'results/polarization/all_pol_results.json',
            'results/polarization/polarization.json',
            'results/polarization/polarization_results.json',
            'results/polarization/polarization_curves.png',
        ):
            assert (mixed_case / rel).exists(), f"missing output {rel}"

        # All 5 CSVs (good and column-poor) stay in the timeline JSON;
        # only the plottable ones are drawn.
        timeline = json.loads(
            (mixed_case / 'results' / 'test_sequence'
             / 'all_csv_results_in_timeline.json').read_text(encoding='utf-8'))
        assert len(timeline) == 5