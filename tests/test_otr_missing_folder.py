#!/usr/bin/env python
"""B14: mea otr on a case without OTR/ must not crash.

Before the fix, a manual `mea otr` in any of the 242 non-OTR cases died
with a raw FileNotFoundError from os.listdir in parse_data_title
(run_all guarded it with 'if "OTR" in dirs', but the direct command did
not).

After the fix: a friendly message and empty (note-marked) results JSONs.
"""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import meatools.subcomands.mea_proccess as mp
from meatools.subcomands.impedence_calc import run_all_otr_groups

GROUPS = ("original", "exclude_1pct_o2_and_150kpa", "exclude_300kpa")


class TestOtrWithoutFolder:

    def test_run_all_otr_groups_missing_dir(self, tmp_path, monkeypatch,
                                            capsys):
        monkeypatch.chdir(tmp_path)
        fitted, final = run_all_otr_groups(root_path='OTR/',
                                           long_out=False, fit_plot=False)
        assert set(fitted.keys()) == set(GROUPS)
        assert all('note' in fitted[g] for g in GROUPS)
        assert all('note' in final[g] for g in GROUPS)
        assert all(final[g]['r_diff (s m^-1)'] is None for g in GROUPS)
        assert 'no OTR folder' in capsys.readouterr().err

        on_disk = json.loads(
            (tmp_path / 'results' / 'impedence'
             / 'final_results.json').read_text(encoding='utf-8'))
        assert set(on_disk.keys()) == set(GROUPS)

    def test_mea_otr_command_missing_dir(self, tmp_path):
        env = dict(os.environ, MPLBACKEND='Agg')
        proc = subprocess.run(
            [sys.executable, '-m', 'meatools.subcomands.impedence_calc'],
            cwd=str(tmp_path), env=env,
            capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-1500:]
        assert 'Traceback' not in proc.stderr
        assert 'no OTR folder' in proc.stderr
        final = json.loads(
            (tmp_path / 'results' / 'impedence'
             / 'final_results.json').read_text(encoding='utf-8'))
        assert all('note' in final[g] for g in GROUPS)

    def test_run_all_skips_otr_without_folder(self, tmp_path, monkeypatch):
        # run_all already guarded this ('if "OTR" in dirs') - keep it.
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

        assert mp.run_all(SimpleNamespace(no_sulf=False)) == 0
        assert 'run_otr' not in calls
        assert 'run_conclude' in calls