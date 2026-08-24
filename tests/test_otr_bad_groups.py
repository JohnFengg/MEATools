#!/usr/bin/env python
"""B4: OTR analysis must not crash on real data (28/505 cases).

Real-data failure classes:
  1. flat OTR layout (18/28 cases): CSVs sit directly in OTR/ with the
     O2 fraction and pressure in the filename; the nested-only parser
     found nothing -> empty fits -> curve_fit '`ydata` must not be
     empty!'
  2. single-point pressure groups (1/28): 2-parameter linear fit on
     one point -> scipy TypeError.
  Any failing group/config used to kill the whole `mea otr` run before
  fitted_r_total.json / final_results.json were written.

After the fix: flat layout is parsed, fits validate their inputs, bad
groups record {'error': ...}, and the results JSONs are always written.
"""

import json
import os
from datetime import datetime

import pytest

from meatools.subcomands.impedence_calc import r_total_calc, run_all_otr_groups
from tests.testdata_utils import write_hrl_csv

EXPECTED_GROUPS = {
    "original",
    "exclude_1pct_o2_and_150kpa",
    "exclude_300kpa",
}


def make_otr_csv(path, amp):
    """Constant-current OTR run (current = amp A, area = 5 cm^2)."""
    n = 60
    write_hrl_csv(
        path, datetime(2025, 9, 15),
        ['elapsed time', 'current', 'cell_active_area'],
        {'elapsed time': list(range(n)),
         'current': [amp] * n,
         'cell_active_area': [5.0] * n},
    )


def make_nested_tree(otr_dir, spec):
    """spec: {(o2_label, pressure_label): current_amp}."""
    for (o2, pressure), amp in spec.items():
        folder = os.path.join(otr_dir, o2)
        os.makedirs(folder, exist_ok=True)
        make_otr_csv(os.path.join(folder, f'test_{pressure}kPa.csv'), amp)


class TestFittingValidation:
    """fitting(): explicit errors for unusable inputs."""

    def test_empty_raises(self):
        with pytest.raises(ValueError, match='no data points'):
            r_total_calc.fitting([], [], prefix='t', plot=False)

    def test_single_point_raises(self):
        with pytest.raises(ValueError, match='insufficient data points'):
            r_total_calc.fitting([1.0], [2.0], prefix='t', plot=False)


class TestFlatLayout:
    """Flat OTR layout: CSVs directly under OTR/."""

    def test_flat_files_discovered(self, tmp_path):
        otr = tmp_path / 'OTR'
        otr.mkdir()
        make_otr_csv(otr / 'x_1%O2_85C_150kPaa - 1.csv', 0.1)
        make_otr_csv(otr / 'x_1%O2_85C_200kPaa - 1.csv', 0.12)
        make_otr_csv(otr / 'x_2%O2_85C_150kPaa - 1.csv', 0.15)

        calc = r_total_calc(root_path=str(otr), label='original')
        result = calc.parse_data_title()
        assert set(result.keys()) == {0.01, 0.02}
        assert set(result[0.01].keys()) == {'150', '200'}
        assert set(result[0.02].keys()) == {'150'}

    def test_flat_respects_exclusions(self, tmp_path):
        otr = tmp_path / 'OTR'
        otr.mkdir()
        make_otr_csv(otr / 'x_1%O2_85C_150kPaa - 1.csv', 0.1)
        make_otr_csv(otr / 'x_2%O2_85C_150kPaa - 1.csv', 0.15)

        calc = r_total_calc(root_path=str(otr), label='original',
                            exclude_o2_fractions=[0.01],
                            exclude_pressures=['150'])
        result = calc.parse_data_title()
        assert result == {}

    def test_nested_layout_takes_precedence(self, tmp_path):
        otr = tmp_path / 'OTR'
        make_nested_tree(str(otr), {('1%O2', '150'): 0.1,
                                    ('1%O2', '200'): 0.12})
        # stray flat file must be ignored when nested dirs exist
        make_otr_csv(otr / 'stray_4%O2_150kPaa.csv', 0.9)

        calc = r_total_calc(root_path=str(otr), label='original')
        result = calc.parse_data_title()
        assert set(result.keys()) == {0.01}


class TestGroupIsolation:
    """Bad groups/configs record errors; JSONs are always written."""

    def test_single_point_group_isolated(self, tmp_path, monkeypatch):
        otr = tmp_path / 'OTR'
        # 150 kPa: one file (1 point -> unfittable)
        # 200 kPa: two files (2 points -> fits)
        make_nested_tree(str(otr), {('1%O2', '150'): 0.10,
                                    ('1%O2', '200'): 0.12,
                                    ('2%O2', '200'): 0.14})
        monkeypatch.chdir(tmp_path)

        fitted, final = run_all_otr_groups(
            root_path='OTR/', long_out=False, fit_plot=False)

        assert set(fitted.keys()) == EXPECTED_GROUPS
        assert 'error' in fitted['original']['150']['fit_stats']
        assert 'r2' in fitted['original']['200']['fit_stats']
        # final fit has only the 200 kPa point -> recorded error
        assert 'error' in final['original']
        assert final['original']['r_diff (s m^-1)'] is None

        on_disk = json.loads(
            (tmp_path / 'results' / 'impedence' / 'final_results.json')
            .read_text(encoding='utf-8'))
        assert set(on_disk.keys()) == EXPECTED_GROUPS

    def test_empty_otr_tree_writes_error_results(self, tmp_path, monkeypatch):
        (tmp_path / 'OTR').mkdir()
        monkeypatch.chdir(tmp_path)

        fitted, final = run_all_otr_groups(
            root_path='OTR/', long_out=False, fit_plot=False)

        assert set(fitted.keys()) == EXPECTED_GROUPS
        assert all('error' in final[g] for g in EXPECTED_GROUPS)
        assert (tmp_path / 'results' / 'impedence'
                / 'fitted_r_total.json').is_file()
        assert (tmp_path / 'results' / 'impedence'
                / 'final_results.json').is_file()

    def test_bad_csv_file_isolated(self, tmp_path, monkeypatch):
        otr = tmp_path / 'OTR'
        folder = otr / '1%O2'
        folder.mkdir(parents=True)
        make_otr_csv(folder / 'good_150kPa.csv', 0.10)
        (folder / 'bad_200kPa.csv').write_text('garbage, no header\n',
                                               encoding='utf-8')
        os.makedirs(otr / '2%O2', exist_ok=True)
        make_otr_csv(otr / '2%O2' / 'good2_200kPa.csv', 0.14)
        monkeypatch.chdir(tmp_path)

        fitted, final = run_all_otr_groups(
            root_path='OTR/', long_out=False, fit_plot=False)

        original = fitted['original']
        # 150 kPa has the good file only; 200 kPa group: bad file skipped,
        # good2 remains -> 1 point -> fit error, not a crash
        assert original['150']['fit_stats'].get('error') or \
            'r2' in original['150']['fit_stats']
        on_disk = json.loads(
            (tmp_path / 'results' / 'impedence' / 'fitted_r_total.json')
            .read_text(encoding='utf-8'))
        assert 'original' in on_disk