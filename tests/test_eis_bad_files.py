#!/usr/bin/env python
"""B3: one bad EIS/PEIS DTA file must not abort the whole case.

Real-data failure classes (70/505 cases crashed, eis_results.json never
written):
  1. files whose imaginary axis has no positive values
     -> ValueError "No positive imaginary values"
  2. AppleDouble ``._*.DTA`` junk / empty ZCURVE bodies
     -> 1-D array -> IndexError "array is 1-dimensional"
  3. Gamry 'EXPERIMENTABORTED' marker text inside/after the data body
     -> ValueError "could not convert string to float"

After the fix: each file is processed independently, failures are
recorded as ``{"error": ...}`` entries, good files are still analyzed,
and eis_results.json is always written.
"""

import json
import os

import numpy as np
import pytest

from meatools.subcomands import eis as eis_mod
from meatools.subcomands.eis import read_dta_data


def _good_eis_columns(n=60):
    """Plausible impedance arc: positive imag at high f, negative at low f."""
    freq = np.logspace(1, 3, n)[::-1]          # high -> low frequency
    zreal = 0.05 + 0.02 * (1 - (np.log10(freq) - 1) / 2)
    zimag = np.linspace(0.03, -0.03, n)        # crosses zero
    idx = np.arange(n)
    return np.column_stack([idx, idx, freq, zreal, zimag])


def write_dta(path, columns, abort_marker=False, zcurve=True):
    """Write a minimal Gamry-style DTA file (ZCURVE at line 1)."""
    lines = []
    if zcurve:
        lines.append('ZCURVE\tTABLE')
        lines.append('')
        lines.append('\t#\ts\tHz\tV\tohm')
        for i in range(len(columns)):
            lines.append('\t'.join(str(v) for v in columns[i]))
        if abort_marker:
            lines.append('EXPERIMENTABORTED\tTOGGLE\tT\tExperiment Aborted')
    else:
        lines.append('GAMRY DTA FILE')
        lines.append('no curve here')
    with open(path, 'w', encoding='ISO-8859-1') as f:
        f.write('\n'.join(lines) + '\n')


class TestReadDtaDataValidation:
    """read_dta_data: clear errors instead of silent 1-D arrays."""

    def test_appledouble_junk_raises_value_error(self, tmp_path):
        p = tmp_path / '._bad.DTA'
        p.write_bytes(b'\x00\x05\x16\x07\x00\x00\x00\x00' * 512)
        with pytest.raises(ValueError):
            read_dta_data(str(p))

    def test_no_zcurve_raises_value_error(self, tmp_path):
        p = tmp_path / 'empty_body.DTA'
        write_dta(p, np.zeros((0, 5)), zcurve=False)
        with pytest.raises(ValueError, match='no EIS data rows'):
            read_dta_data(str(p))

    def test_abort_marker_truncates_but_keeps_valid_rows(self, tmp_path):
        p = tmp_path / 'aborted.DTA'
        write_dta(p, _good_eis_columns(20), abort_marker=True)
        arr = read_dta_data(str(p))
        assert arr.shape == (20, 5)

    def test_valid_file_shape(self, tmp_path):
        p = tmp_path / 'good.DTA'
        write_dta(p, _good_eis_columns(30))
        arr = read_dta_data(str(p))
        assert arr.ndim == 2 and arr.shape[1] >= 5


class TestEisMainIsolation:
    """main(): per-file error isolation, eis_results.json always written."""

    @pytest.fixture
    def eis_case(self, tmp_path, monkeypatch):
        (tmp_path / 'EIS').mkdir()
        write_dta(tmp_path / 'EIS' / 'good1.DTA', _good_eis_columns())
        write_dta(tmp_path / 'EIS' / 'good2.DTA', _good_eis_columns(50))
        # class 1: no positive imaginary values
        bad1 = _good_eis_columns()
        bad1[:, 4] = -abs(bad1[:, 4]) - 0.001
        write_dta(tmp_path / 'EIS' / 'bad_no_pos_imag.DTA', bad1)
        # class 2: AppleDouble junk
        (tmp_path / 'EIS' / '._bad_junk.DTA').write_bytes(b'\x00\x01\x02' * 100)
        # class 3: abort marker (valid data must still parse)
        write_dta(tmp_path / 'EIS' / 'aborted.DTA',
                  _good_eis_columns(25), abort_marker=True)
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def test_bad_files_do_not_abort_case(self, eis_case):
        eis_mod.main()  # must not raise

        out = eis_case / 'results' / 'eis' / 'eis_results.json'
        assert out.is_file(), 'eis_results.json was not written'
        results = json.loads(out.read_text(encoding='utf-8'))
        assert len(results) == 5

        with_error = [e for e in results.values() if 'error' in e]
        with_hfr = [e for e in results.values() if 'HFR (ohm)' in e]
        assert len(with_error) == 2          # junk + no-positive-imag
        assert len(with_hfr) == 3            # good1, good2, aborted
        err_msgs = ' '.join(e['error'] for e in with_error)
        assert 'no EIS data rows' in err_msgs
        assert 'No positive imaginary' in err_msgs
        for e in with_hfr:
            assert e['sample_number'] >= 0

    def test_all_bad_files_still_write_results(self, tmp_path, monkeypatch):
        (tmp_path / 'PEIS').mkdir()
        bad = _good_eis_columns()
        bad[:, 4] = -1.0
        write_dta(tmp_path / 'PEIS' / 'bad1.DTA', bad)
        write_dta(tmp_path / 'PEIS' / 'bad2.DTA', bad)
        monkeypatch.chdir(tmp_path)

        eis_mod.main()  # must not raise

        results = json.loads(
            (tmp_path / 'results' / 'eis' / 'eis_results.json')
            .read_text(encoding='utf-8'))
        assert len(results) == 2
        assert all('error' in e for e in results.values())
        assert all('No positive imaginary' in e['error']
                   for e in results.values())

    def test_error_entry_keeps_file_identity(self, tmp_path, monkeypatch):
        (tmp_path / 'EIS').mkdir()
        (tmp_path / 'EIS' / 'junk.DTA').write_bytes(b'\x00\xff' * 50)
        monkeypatch.chdir(tmp_path)

        eis_mod.main()
        entry = json.loads(
            (tmp_path / 'results' / 'eis' / 'eis_results.json')
            .read_text(encoding='utf-8'))['file_1']
        assert entry['filename'].endswith('junk.DTA')
        assert 'filetime' in entry
        assert 'error' in entry