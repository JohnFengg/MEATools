#!/usr/bin/env python
"""B13: sulf-cvrg folder detection must accept both real spellings of the
100%RH CO stripping folder and warn (not silently skip) when the layout
isn't recognized.

Dataset reality (35 sulf cases):
- 28 use 干质子可及率/100%RH/Cathode CO CV
- 5 use 干质子可及率/100%RH/Cathode CV CO (e.g. HRL-B092...-EOL)
- 5 have 磺酸根覆盖度/ but no 干质子可及率/ at all (e.g. HRL-B012)
The plain 'Cathode CV' folder (no CO) is a different measurement and
must never be used for the CO stripping charge.
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import meatools.subcomands.mea_proccess as mp
from meatools.sulfonate_coverage import (
    find_case_files,
    has_sulfonate_coverage_files,
    process_case,
)
from tests.test_sulf_noninteractive import make_sulfonate_case


class TestFolderDetection:

    def test_accepts_cathode_co_cv(self, tmp_path):
        make_sulfonate_case(tmp_path, dry_folder='Cathode CO CV')
        assert has_sulfonate_coverage_files(tmp_path)

    def test_accepts_cathode_cv_co(self, tmp_path):
        make_sulfonate_case(tmp_path, dry_folder='Cathode CV CO')
        assert has_sulfonate_coverage_files(tmp_path)

    def test_plain_cathode_cv_not_accepted(self, tmp_path):
        (tmp_path / '磺酸根覆盖度' / '1' / 'CO displace').mkdir(parents=True)
        (tmp_path / '干质子可及率' / '100%RH' / 'Cathode CV').mkdir(
            parents=True)
        assert not has_sulfonate_coverage_files(tmp_path)

    def test_missing_dry_tree_not_accepted(self, tmp_path):
        (tmp_path / '磺酸根覆盖度' / '1' / 'CO displace').mkdir(parents=True)
        assert not has_sulfonate_coverage_files(tmp_path)

    def test_missing_sulf_tree_not_accepted(self, tmp_path):
        (tmp_path / '干质子可及率' / '100%RH' / 'Cathode CO CV').mkdir(
            parents=True)
        assert not has_sulfonate_coverage_files(tmp_path)


class TestFindCaseFilesAltSpelling:

    def test_dta_found_in_cathode_cv_co(self, tmp_path):
        make_sulfonate_case(tmp_path, dry_folder='Cathode CV CO')
        files = find_case_files(tmp_path)
        assert 'Cathode CV CO' in str(files['co_stripping'])

    def test_process_case_computes_with_alt_spelling(self, tmp_path):
        make_sulfonate_case(tmp_path, dry_folder='Cathode CV CO')
        result = process_case(tmp_path, co_displace_kwargs={1: {}, 2: {}, 3: {}})
        assert result['so3_coverage_percent'] > 0

    def test_plain_cathode_cv_dta_not_used(self, tmp_path):
        make_sulfonate_case(tmp_path, dry_folder='Cathode CV CO')
        # a decoy plain-CV folder with a DTA must be ignored
        decoy = tmp_path / '干质子可及率' / '100%RH' / 'Cathode CV'
        decoy.mkdir()
        (decoy / 'decoy.DTA').write_text('CURVE1\tTABLE\n',
                                         encoding='latin-1')
        files = find_case_files(tmp_path)
        assert 'Cathode CV CO' in str(files['co_stripping'])


class TestWarnings:

    def test_run_all_warns_on_unrecognized_layout(self, tmp_path, monkeypatch,
                                                 capsys):
        (tmp_path / '磺酸根覆盖度' / '1' / 'CO displace').mkdir(parents=True)
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
        # real detector: sulf tree yes, dry tree no -> not recognized
        sulf_calls = []
        monkeypatch.setattr(
            mp, 'run_sulfonate_coverage',
            lambda args=None: (sulf_calls.append(args), 0)[1])

        assert mp.run_all(SimpleNamespace(no_sulf=False)) == 0
        err = capsys.readouterr().err
        assert 'Warning' in err and '磺酸根覆盖度' in err
        assert sulf_calls == []
        assert 'run_conclude' in calls

    def test_sulf_cvrg_cli_warns_on_unrecognized_layout(self, tmp_path):
        (tmp_path / '磺酸根覆盖度' / '1' / 'CO displace').mkdir(parents=True)
        env = dict(os.environ, MPLBACKEND='Agg')
        proc = subprocess.run(
            [sys.executable, '-m', 'meatools.subcomands.sulfonate_coverage',
             '--non-interactive'],
            cwd=str(tmp_path), env=env,
            capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0
        assert 'Warning' in proc.stderr and '磺酸根覆盖度' in proc.stderr

    def test_sulf_cvrg_cli_quiet_skip_without_sulf_tree(self, tmp_path):
        env = dict(os.environ, MPLBACKEND='Agg')
        proc = subprocess.run(
            [sys.executable, '-m', 'meatools.subcomands.sulfonate_coverage',
             '--non-interactive'],
            cwd=str(tmp_path), env=env,
            capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0
        assert 'Warning' not in proc.stderr
        assert 'no coverage data folders' in proc.stderr