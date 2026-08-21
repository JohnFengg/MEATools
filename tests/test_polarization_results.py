#!/usr/bin/env python
"""B2: results.json Polarization section must not be empty.

The pre-refactor ttseq (v0.3.5) wrote
``results/polarization/polarization_results.json`` with per-step
polarization data; the refactor dropped that write and ``mea conclude``
kept reading it, so the Polarization section of results.json was empty
for 100% of real cases.

These tests pin the restored contract:
  * ``analyze_pol_steps`` computes the per-step OCV/current/voltage/IR/
    current-density entries (v0.3.5 semantics);
  * ``mea ttseq`` writes ``polarization_results.json``;
  * ``mea conclude`` fills a non-empty, numerically correct
    ``Polarization`` section in results.json.
"""

import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pytest

from meatools.subcomands.test_squence import analyze_pol_steps
from tests.testdata_utils import write_hrl_csv


def _pol_profile(v5):
    """360-sample @1 Hz polarization profile.

    5A(60 s) OCV(60 s) 5A(60 s) OCV(60 s) 10A(60 s) OCV(60 s)

    Step ends with a >0.95 A set-current drop: indices 60, 180, 300.
    Qualifying segment ends (segment length > 50): 180 and 300, i.e. the
    second 5 A step and the 10 A step -> densities 1.0 / 2.0 A cm^-2
    for a 5 cm2 cell.
    """
    cs = np.concatenate([np.full(60, 5.0), np.zeros(60), np.full(60, 5.0),
                         np.zeros(60), np.full(60, 10.0), np.zeros(60)])
    n = len(cs)
    voltage = np.where(cs == 0.0, 0.95, np.where(cs == 5.0, v5, 0.85))
    return {
        'Elapsed time': np.arange(1, n + 1, dtype=float),
        'cell_active_area': np.full(n, 5.0),
        'current_set': cs,
        'current': cs.copy(),
        'cell_voltage_001': voltage,
        'HIOKI-01.resistance': np.full(n, 8.0),
        'temp_coolant_inlet': np.full(n, 80.0),
    }


def _pol_data(v5=0.90):
    """extract_data_from_file() style dict for the profile above."""
    p = _pol_profile(v5)
    return {
        'elapsed_time': p['Elapsed time'],
        'cell_active_area': p['cell_active_area'],
        'current_set': p['current_set'],
        'current': p['current'],
        'cell_voltage_001': p['cell_voltage_001'],
        'resistance': p['HIOKI-01.resistance'],
        'temp_coolant_inlet': p['temp_coolant_inlet'],
    }


def _make_result(data, path='polarisation/x.csv'):
    return {'file_info': path, 'data': data,
            'columns_found': list(data.keys())}


def _run_module(module, case_dir):
    env = dict(os.environ, MPLBACKEND='Agg')
    return subprocess.run([sys.executable, '-m', module], cwd=str(case_dir),
                          env=env, capture_output=True, text=True, timeout=600)


class TestAnalyzePolSteps:
    """Unit tests for the restored per-step polarization analysis."""

    def test_step_values(self):
        out = analyze_pol_steps(_make_result(_pol_data(0.90)), 5.0, 30)
        assert out is not None
        assert out['OCV (V)'] == pytest.approx(0.95)
        assert out['current (A)'] == pytest.approx([5.0, 10.0])
        assert out['voltage (V)'] == pytest.approx([0.90, 0.85])
        # voltage+IR = voltage + HFR[mOhm] * I[A] / 1000
        assert out['voltage+IR (V)'] == pytest.approx([0.94, 0.93])
        assert out['IR (V)'] == pytest.approx([0.04, 0.08])
        assert out['current density (A cm^(-2))'] == pytest.approx([1.0, 2.0])

    def test_missing_columns_returns_none_with_log(self):
        data = _pol_data(0.90)
        data.pop('resistance')
        logs = []

        class _Log:
            def write(self, s):
                logs.append(s)

        out = analyze_pol_steps(_make_result(data), 5.0, 30, log=_Log())
        assert out is None
        assert any('resistance' in s for s in logs)

    def test_no_step_drops_returns_none(self):
        data = _pol_data(0.90)
        data['current_set'] = np.linspace(0.0, 5.0, 360)  # monotonic, no drops
        out = analyze_pol_steps(_make_result(data), 5.0, 30)
        assert out is None


class TestPolResultsPipeline:
    """End-to-end: ttseq writes the file, conclude consumes it."""

    @pytest.fixture
    def pol_case(self, tmp_path):
        """Synthetic case: 1 activation CSV + 2 polarization CSVs."""
        case = tmp_path
        (case / 'polarisation').mkdir()

        n = 120
        act = {
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
                      list(act.keys()), act)

        base = 'HRL_Pol_Test_80C_100%RH_H2_O2_0.5_1_150kpa'
        for suffix, start, v5 in (
            ('1602', datetime(2025, 9, 15, 16, 2, 20), 0.90),
            ('1700', datetime(2025, 9, 15, 17, 0, 0), 0.91),
        ):
            write_hrl_csv(
                case / 'polarisation' / f'{base} - 20250915 {suffix}.csv',
                start, list(_pol_profile(v5).keys()), _pol_profile(v5))
        return case

    def test_ttseq_writes_polarization_results(self, pol_case):
        proc = _run_module('meatools.subcomands.test_squence', pol_case)
        assert proc.returncode == 0, proc.stderr
        path = pol_case / 'results' / 'polarization' / 'polarization_results.json'
        assert path.is_file(), 'ttseq did not write polarization_results.json'
        pol = json.loads(path.read_text(encoding='utf-8'))
        assert len(pol) == 2, 'both polarization runs must be analyzed'
        for name, entry in pol.items():
            assert set(entry) == {
                'OCV (V)', 'current (A)', 'voltage (V)', 'voltage+IR (V)',
                'IR (V)', 'current density (A cm^(-2))'}
            assert entry['current density (A cm^(-2))'] == pytest.approx([1.0, 2.0])
            assert entry['OCV (V)'] == pytest.approx(0.95)

    def test_conclude_polarization_section_not_empty(self, pol_case):
        tt = _run_module('meatools.subcomands.test_squence', pol_case)
        assert tt.returncode == 0, tt.stderr
        cl = _run_module('meatools.subcomands.conclude', pol_case)
        assert cl.returncode == 0, cl.stderr

        results = json.loads(
            (pol_case / 'results.json').read_text(encoding='utf-8'))
        pol = results['Polarization']
        assert pol, 'Polarization section of results.json is empty (B2)'

        # Two runs of the same test are averaged; the row closest to
        # 1 A/cm2 is reported (extract_row semantics in conclude).
        entry = pol['HRL_Pol_Test_80C_100%RH_H2_O2_0.5_1_150kpa']
        assert entry['current density (A cm^(-2))'] == pytest.approx(1.0)
        assert entry['current (A)'] == pytest.approx(5.0)
        assert entry['voltage (V)'] == pytest.approx(0.905)  # avg(0.90, 0.91)
        assert entry['voltage+IR (V)'] == pytest.approx(0.945)
        assert entry['IR (V)'] == pytest.approx(0.04)
        assert entry['OCV (V)'] == pytest.approx(0.95)