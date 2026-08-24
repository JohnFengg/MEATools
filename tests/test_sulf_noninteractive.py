#!/usr/bin/env python
"""B12: sulf-cvrg must have a non-interactive mode; mea all must be able
to skip it (--no-sulf).

Before the fix: 'mea all' unconditionally launched the interactive
browser UI whenever sulf coverage folders were present (28/505 real
cases) and hung waiting for a human; 'mea sulf-cvrg' itself was also
broken (its args were never forwarded - the subcommand's own parser saw
the word 'sulf-cvrg' as a case directory).

After the fix:
- 'mea sulf-cvrg --non-interactive' applies the default peak boundaries
  (exactly what the UI's Skip button does) and saves the result JSON
- 'mea all' runs that non-interactive mode in batch context
- 'mea all --no-sulf' skips the step entirely
- CLI args after 'sulf-cvrg' are forwarded to the subcommand
"""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import meatools.cli as cli
import meatools.subcomands.mea_proccess as mp


# ---------------------------------------------------------------------------
# synthetic sulf case
# ---------------------------------------------------------------------------

def make_sulfonate_case(case_dir):
    """Build a minimal case with 3 CO-displace CSVs + 1 stripping DTA."""
    sulf = case_dir / '磺酸根覆盖度'
    dry = case_dir / '干质子可及率' / '100%RH' / 'Cathode CO CV'

    n = 300
    time = np.arange(n, dtype=float)
    current = 0.5 - 2.0 * np.exp(-((time - 200.0) / 8.0) ** 2)

    for run in ('1', '2', '3'):
        run_dir = sulf / run / 'CO displace'
        run_dir.mkdir(parents=True)
        rows = []
        rows.append(','.join(['Test name', 'synthetic'] + [''] * 76))
        rows.append(','.join(['', 'elapsed time'] + [''] * 76))
        for i in range(n):
            cells = ['2025-09-15 00:00:00', str(time[i])] + ['0'] * 76
            cells[77] = str(current[i])
            rows.append(','.join(cells))
        (run_dir / f'run{run}.csv').write_text(
            '\n'.join(rows) + '\n', encoding='latin-1')

    dry.mkdir(parents=True)
    v = np.linspace(0.05, 0.9, 90)
    dta_lines = []
    for cycle, amp in ((1, 0.5), (2, 0.0)):
        dta_lines.append(f'CURVE{cycle}\tTABLE')
        dta_lines.append('HEADER\tROW')
        for pt, vv in enumerate(v, 1):
            im = 0.001 + amp * np.exp(-((vv - 0.65) / 0.05) ** 2)
            dta_lines.append(
                f'{pt}\t0.0\t{vv:.4f}\t{im:.6f}\t{vv:.4f}\t0\t0\t0\t0\t{cycle}')
    (dry / 'strip.DTA').write_text(
        '\n'.join(dta_lines) + '\n', encoding='latin-1')


class TestNonInteractiveMode:

    def test_non_interactive_writes_result(self, tmp_path):
        make_sulfonate_case(tmp_path)
        env = dict(os.environ, MPLBACKEND='Agg')
        proc = subprocess.run(
            [sys.executable, '-m', 'meatools.subcomands.sulfonate_coverage',
             '--non-interactive'],
            cwd=str(tmp_path), env=env,
            capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-1500:]
        assert 'Interactive' not in proc.stdout

        out = tmp_path / 'results' / 'sulf-cvrg' / 'sulfonate_coverage.json'
        assert out.exists()
        result = json.loads(out.read_text(encoding='utf-8'))
        assert result['so3_coverage_percent'] > 0
        assert result['q_co_stripping'] > 0
        assert result['q_co_displace']['average_of_2_and_3'] > 0


class TestCliForwarding:

    def test_sulf_cvrg_args_forwarded_to_child(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv',
                            ['mea', 'sulf-cvrg', '--non-interactive',
                             '--port', '1234'])
        seen = {}

        def fake_run(cmd, **kw):
            seen['cmd'] = cmd
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(mp.subprocess, 'run', fake_run)
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
        tail = seen['cmd'][3:]
        assert tail == ['--non-interactive', '--port', '1234']

    def test_all_parses_no_sulf(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['mea', 'all', '--no-sulf'])
        captured = {}

        def fake_run_all(args):
            captured['args'] = args
            return 0

        monkeypatch.setattr(mp, 'run_all', fake_run_all)
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
        assert captured['args'].no_sulf is True


class TestRunAllSulfStep:

    @pytest.fixture
    def sulf_case(self, tmp_path, monkeypatch):
        make_sulfonate_case(tmp_path)
        monkeypatch.chdir(tmp_path)
        calls = []

        def make_fake(name):
            def fake(args=None):
                calls.append(name)
                return 0
            return fake

        for name in ['run_test_sequence', 'run_otr', 'run_ecsa',
                     'run_ecsa_dry', 'run_lsv', 'run_eis',
                     'run_conclude', 'run_render']:
            monkeypatch.setattr(mp, name, make_fake(name))
        monkeypatch.setattr(mp, 'has_sulfonate_coverage_files',
                            lambda path: True)
        self_calls = []
        monkeypatch.setattr(
            mp, 'run_sulfonate_coverage',
            lambda args=None: (self_calls.append(args), 0)[1])
        return calls, self_calls

    def test_all_runs_sulf_non_interactively(self, sulf_case):
        calls, self_calls = sulf_case
        assert mp.run_all(SimpleNamespace(no_sulf=False)) == 0
        assert len(self_calls) == 1
        assert self_calls[0].sulf_args == ['--non-interactive']
        assert 'run_conclude' in calls

    def test_all_no_sulf_skips_sulf(self, sulf_case):
        calls, self_calls = sulf_case
        assert mp.run_all(SimpleNamespace(no_sulf=True)) == 0
        assert self_calls == []
        assert 'run_conclude' in calls and 'run_render' in calls

    def test_all_without_sulf_folders_never_runs_sulf(self, tmp_path,
                                                      monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = []

        def make_fake(name):
            def fake(args=None):
                calls.append(name)
                return 0
            return fake

        for name in ['run_test_sequence', 'run_otr', 'run_ecsa',
                     'run_ecsa_dry', 'run_lsv', 'run_eis',
                     'run_conclude', 'run_render']:
            monkeypatch.setattr(mp, name, make_fake(name))
        monkeypatch.setattr(mp, 'has_sulfonate_coverage_files',
                            lambda path: False)
        self_calls = []
        monkeypatch.setattr(
            mp, 'run_sulfonate_coverage',
            lambda args=None: (self_calls.append(args), 0)[1])

        assert mp.run_all(SimpleNamespace(no_sulf=False)) == 0
        assert self_calls == []