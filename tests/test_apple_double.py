#!/usr/bin/env python
"""B15: finders must skip macOS AppleDouble (._*) junk files.

Real data trees contain 1668 ``._*.DTA`` / ``._*.csv`` binary AppleDouble
files that match the same glob patterns as real files (the ``._*.DTA``
matches were the trigger of the eis class-2 crashes in B3). ``._name``
sorts before ``name``, so an unfiltered ``sorted(glob(...))[0]`` can
return the junk file.

After the fix every finder filters ``._*`` entries.
"""

import json
import os
import subprocess
import sys

import pytest

from meatools.sulfonate_coverage import find_case_files
from meatools.subcomands.test_squence import (
    find_csv_files,
    find_csv_pol_files,
)
from meatools.utils.file_utils import (
    find_and_sort_dta_files_by_candidates,
    find_and_sort_load_dta_files,
    find_cv_subfolders,
    is_apple_double,
)

JUNK = b'\x00\x05\x16\x07\x00\x00\x00\x00' * 512  # AppleDouble-ish bytes


class TestPredicate:

    def test_predicate(self):
        assert is_apple_double('._x.DTA')
        assert not is_apple_double('x.DTA')
        assert not is_apple_double('.hidden')


class TestFileUtilsFinders:

    def test_load_dta_files_skips_junk(self, tmp_path):
        folder = tmp_path / 'ECSA' / 'ECSA1'
        folder.mkdir(parents=True)
        (folder / 'good-cv.DTA').write_text('x', encoding='utf-8')
        (folder / '._good-cv.DTA').write_bytes(JUNK)

        files = find_and_sort_load_dta_files(str(tmp_path), '*cv*.DTA')
        assert [os.path.basename(f) for _, f in files] == ['good-cv.DTA']

    def test_cv_subfolders_ignore_junk_only_folders(self, tmp_path):
        good = tmp_path / 'ECSA' / 'real'
        good.mkdir(parents=True)
        (good / 'a-cv.DTA').write_text('x', encoding='utf-8')
        junk_dir = tmp_path / 'ECSA' / 'junk'
        junk_dir.mkdir()
        (junk_dir / '._only-cv.DTA').write_bytes(JUNK)

        folders = find_cv_subfolders(str(tmp_path), '*cv*.DTA')
        assert folders == [str(good)]

    def test_eis_candidates_skip_junk(self, tmp_path):
        eis = tmp_path / 'EIS'
        eis.mkdir()
        (eis / 'good.DTA').write_text('x' * 100, encoding='utf-8')
        (eis / '._good.DTA').write_bytes(JUNK)

        files = find_and_sort_dta_files_by_candidates(str(tmp_path),
                                                      candidates=('EIS',))
        assert [os.path.basename(f) for _, f in files] == ['good.DTA']


class TestTtseqFinders:

    def test_find_csv_files_skips_junk(self, tmp_path):
        (tmp_path / 'real.csv').write_text('x', encoding='utf-8')
        (tmp_path / '._real.csv').write_bytes(JUNK)

        files = find_csv_files(str(tmp_path))
        assert [os.path.basename(f) for f in files] == ['real.csv']

    def test_find_csv_pol_files_skips_junk(self, tmp_path):
        pol = tmp_path / '极化'
        pol.mkdir()
        (pol / 'HRL_Pol_a-1.csv').write_text('x', encoding='utf-8')
        (pol / '._HRL_Pol_a-1.csv').write_bytes(JUNK)

        files = find_csv_pol_files(str(tmp_path), '*Pol*-*.csv')
        assert [os.path.basename(f) for f in files] == ['HRL_Pol_a-1.csv']


class TestSulfFinder:

    def test_find_case_files_prefers_real_over_junk(self, tmp_path):
        from tests.test_sulf_noninteractive import make_sulfonate_case
        make_sulfonate_case(tmp_path)
        # '._' sorts before real names: without the filter, files[0]
        # would be the junk in every run dir and DTA dir.
        for run in ('1', '2', '3'):
            run_dir = tmp_path / '磺酸根覆盖度' / run / 'CO displace'
            (run_dir / '._run.csv').write_bytes(JUNK)
        dry = tmp_path / '干质子可及率' / '100%RH' / 'Cathode CO CV'
        (dry / '._strip.DTA').write_bytes(JUNK)

        files = find_case_files(tmp_path)
        for p in files['co_displace']:
            assert not os.path.basename(str(p)).startswith('._')
        assert not os.path.basename(str(files['co_stripping'])).startswith(
            '._')


class TestEisPipelineWithJunk:
    """End-to-end: the B3 class-2 mechanism at finder level."""

    def test_eis_ignores_apple_double_files(self, tmp_path):
        # reuse the B3 synthetic EIS DTA builder
        from tests.test_eis_bad_files import _good_eis_columns, write_dta
        eis = tmp_path / 'EIS'
        eis.mkdir()
        write_dta(eis / 'good.DTA', _good_eis_columns())
        (eis / '._good.DTA').write_bytes(JUNK)

        env = dict(os.environ, MPLBACKEND='Agg')
        proc = subprocess.run(
            [sys.executable, '-m', 'meatools.subcomands.eis'],
            cwd=str(tmp_path), env=env,
            capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-1500:]

        results = json.loads(
            (tmp_path / 'results' / 'eis' / 'eis_results.json')
            .read_text(encoding='utf-8'))
        # exactly one entry (the junk file is not seen at all)
        assert len(results) == 1
        assert 'HFR (ohm)' in results['file_1']