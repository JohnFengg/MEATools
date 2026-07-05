#!/usr/bin/env python
"""Tests for OTR (oxygen transfer resistance) calculation."""

import json
import os
import shutil
import tempfile
import numpy as np
import pytest

from meatools.subcomands.impedence_calc import r_total_calc, run_all_otr_groups


class TestFitting:
    """Test linear fitting helper."""

    def test_fitting_returns_r2(self):
        """Test that fitting() returns r2 close to 1 for perfect linear data."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0
        result = r_total_calc.fitting(x, y, prefix="test", plot=False)
        assert "r2" in result
        assert result["r2"] == pytest.approx(1.0, abs=1e-9)

    def test_fitting_noisy_data(self):
        """Test that fitting() returns reasonable r2 for noisy linear data."""
        rng = np.random.default_rng(42)
        x = np.linspace(1, 5, 20)
        y = 2.0 * x + 1.0 + rng.normal(0, 0.1, size=x.shape)
        result = r_total_calc.fitting(x, y, prefix="test_noisy", plot=False)
        assert 0 < result["r2"] <= 1.0


class TestConcentrationCalc:
    """Test O2 concentration calculation."""

    def test_concentration_calc_returns_positive(self):
        """Test concentration calculation returns positive O2 concentration."""
        conc, dry = r_total_calc.concentration_calc(353.15, 200, 0.21)
        assert conc > 0
        assert dry > 0


class TestParseDataTitleFiltering:
    """Test data discovery and filtering."""

    def _make_otr_tree(self, base_dir):
        """Create a minimal OTR directory tree."""
        otr_dir = os.path.join(base_dir, "OTR")
        for o2 in ["1%O2", "2%O2", "4%O2"]:
            os.makedirs(os.path.join(otr_dir, o2), exist_ok=True)
            for p in ["150", "200", "300"]:
                open(os.path.join(otr_dir, o2, f"test_{p}kPa.csv"), "w").close()
        return otr_dir

    def test_original_includes_all_files(self, temp_dir):
        """Test that original group includes all valid CSV files."""
        otr_dir = self._make_otr_tree(temp_dir)
        calc = r_total_calc(root_path=otr_dir, label="original")
        result = calc.parse_data_title()
        assert len(result) == 3
        assert set(result.keys()) == {0.01, 0.02, 0.04}
        for frac in result.values():
            assert set(frac.keys()) == {"150", "200", "300"}

    def test_exclude_1pct_and_150kpa(self, temp_dir):
        """Test exclusion of 1% O2 and 150 kPa data."""
        otr_dir = self._make_otr_tree(temp_dir)
        calc = r_total_calc(
            root_path=otr_dir,
            exclude_o2_fractions=[0.01],
            exclude_pressures=["150"],
            label="exclude"
        )
        result = calc.parse_data_title()
        assert set(result.keys()) == {0.02, 0.04}
        for frac in result.values():
            assert set(frac.keys()) == {"200", "300"}

    def test_exclude_300kpa(self, temp_dir):
        """Test exclusion of 300 kPa data."""
        otr_dir = self._make_otr_tree(temp_dir)
        calc = r_total_calc(
            root_path=otr_dir,
            exclude_pressures=["300"],
            label="exclude"
        )
        result = calc.parse_data_title()
        assert set(result.keys()) == {0.01, 0.02, 0.04}
        for frac in result.values():
            assert set(frac.keys()) == {"150", "200"}


class TestRunAllOtrGroups:
    """Integration test running all OTR analysis groups."""

    def test_output_structure(self, monkeypatch):
        """Test that run_all_otr_groups produces grouped output files."""
        case_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "case-tt",
            "HRL-D048_WT2025-1287-RD-0728(7-1#1)"
        )
        otr_dir = os.path.join(case_dir, "OTR")
        if not os.path.isdir(otr_dir):
            pytest.skip("case-tt OTR data not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copytree(otr_dir, os.path.join(tmpdir, "OTR"))
            monkeypatch.chdir(tmpdir)
            run_all_otr_groups(root_path="OTR/", long_out=False, fit_plot=False)

            with open("results/impedence/final_results.json") as f:
                final = json.load(f)
            with open("results/impedence/fitted_r_total.json") as f:
                fitted = json.load(f)

            expected_groups = {
                "original",
                "exclude_1pct_o2_and_150kpa",
                "exclude_300kpa"
            }
            assert set(final.keys()) == expected_groups
            assert set(fitted.keys()) == expected_groups

            for group in expected_groups:
                assert "r2" in final[group]
                assert "r_diff (s m^-1)" in final[group]
                assert "r_other (s m^-1)" in final[group]
                assert final[group]["label"] == group

                for pressure, pdata in fitted[group].items():
                    assert "fit_stats" in pdata
                    assert "r2" in pdata["fit_stats"]
                    assert isinstance(pdata["fit_stats"]["r2"], float)

    def test_original_matches_baseline(self, monkeypatch):
        """Test that original group reproduces the baseline OTR result."""
        case_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "case-tt",
            "HRL-D048_WT2025-1287-RD-0728(7-1#1)"
        )
        baseline_path = os.path.join(
            case_dir, "results", "impedence", "final_results.json"
        )
        otr_dir = os.path.join(case_dir, "OTR")
        if not os.path.isdir(otr_dir) or not os.path.isfile(baseline_path):
            pytest.skip("case-tt OTR baseline not available")

        with open(baseline_path) as f:
            baseline = json.load(f)

        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copytree(otr_dir, os.path.join(tmpdir, "OTR"))
            monkeypatch.chdir(tmpdir)
            fitted, final = run_all_otr_groups(
                root_path="OTR/", long_out=False, fit_plot=False
            )

            # Baseline may be old flat format or new grouped format.
            if "original" in baseline:
                baseline = baseline["original"]

            original = final["original"]
            assert original["r_diff (s m^-1)"] == pytest.approx(
                baseline["r_diff (s m^-1)"], rel=1e-6
            )
            assert original["r_other (s m^-1)"] == pytest.approx(
                baseline["r_other (s m^-1)"], rel=1e-6
            )
