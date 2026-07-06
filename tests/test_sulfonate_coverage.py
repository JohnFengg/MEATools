#!/usr/bin/env python
"""Tests for sulfonate group coverage calculation."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from meatools.sulfonate_coverage import (
    integrate_co_displace_peak,
    integrate_co_stripping,
    process_case,
    read_co_displace_csv,
    read_dta_curves,
)


DATA_ROOT = Path("/Users/toussaint/Work/Projects/MEA/case-coverage")


class TestReadCoDisplaceCsv:
    """Test reading HRL CO displacement CSV files."""

    def test_reads_elapsed_time_and_current(self):
        """Smoke test: ensure time/current arrays are returned."""
        case_dir = DATA_ROOT / "121-BOL" / "磺酸根覆盖度" / "2" / "CO displace"
        csv_files = list(case_dir.glob("*.csv"))
        assert csv_files, f"No CSV found in {case_dir}"

        time, current = read_co_displace_csv(csv_files[0])
        assert len(time) > 0
        assert len(time) == len(current)
        assert np.all(np.isfinite(time))
        assert np.all(np.isfinite(current))


class TestIntegrateCoDisplacePeak:
    """Test CO displacement peak integration."""

    def test_integration_returns_positive_charge(self):
        """Integration should return a positive charge value."""
        case_dir = DATA_ROOT / "121-BOL" / "磺酸根覆盖度" / "2" / "CO displace"
        csv_files = list(case_dir.glob("*.csv"))
        time, current = read_co_displace_csv(csv_files[0])

        result = integrate_co_displace_peak(time, current)
        assert result["charge"] > 0
        assert result["baseline"] > 0
        assert result["t_left"] < result["t_min"] < result["t_right"]

    def test_peak_window_parameters_change_result(self):
        """A wider integration window should produce a larger charge."""
        case_dir = DATA_ROOT / "121-BOL" / "磺酸根覆盖度" / "2" / "CO displace"
        csv_files = list(case_dir.glob("*.csv"))
        time, current = read_co_displace_csv(csv_files[0])

        narrow = integrate_co_displace_peak(time, current, peak_pre=10, peak_post=4)
        wide = integrate_co_displace_peak(time, current, peak_pre=16, peak_post=8)
        assert wide["charge"] > narrow["charge"]


class TestReadDtaCurves:
    """Test Gamry DTA parsing."""

    def test_reads_multiple_curves(self):
        """The CO stripping DTA should contain at least two CV cycles."""
        dta_dir = DATA_ROOT / "121-BOL" / "干质子可及率" / "100%RH" / "Cathode CO CV"
        dta_files = list(dta_dir.glob("*.DTA"))
        assert dta_files, f"No DTA found in {dta_dir}"

        curves = read_dta_curves(dta_files[0])
        assert len(curves) >= 2
        for curve in curves:
            assert {"Vf", "Im", "T"}.issubset(curve.columns)


class TestIntegrateCoStripping:
    """Test CO stripping charge integration."""

    def test_co_stripping_charge_positive(self):
        """CO stripping charge should be positive."""
        dta_dir = DATA_ROOT / "121-BOL" / "干质子可及率" / "100%RH" / "Cathode CO CV"
        dta_files = list(dta_dir.glob("*.DTA"))

        result = integrate_co_stripping(dta_files[0])
        assert result["charge"] > 0
        assert result["iv_integral"] > 0
        assert result["scan_rate"] == pytest.approx(0.04, abs=1e-6)


class TestProcessCase:
    """End-to-end verification against documented reference values."""

    EXPECTED = {
        "114-BOL": 7.18,
        "114-EOL": 11.56,
        "121-BOL": 4.91,
    }

    TOLERANCE = 0.50  # percentage points; manual peak integration is subjective

    @pytest.mark.parametrize("case_name", ["114-BOL", "114-EOL", "121-BOL"])
    def test_coverage_matches_documented_value(self, case_name):
        """Computed coverage should be within tolerance of the Word document."""
        case_dir = DATA_ROOT / case_name
        result = process_case(case_dir)

        computed = result["so3_coverage_percent"]
        expected = self.EXPECTED[case_name]

        assert computed == pytest.approx(expected, abs=self.TOLERANCE), (
            f"{case_name}: computed {computed:.2f}% vs expected {expected:.2f}%"
        )

    def test_runs_1_excluded_from_average(self):
        """The protocol averages runs 2 and 3, not all three."""
        result = process_case(DATA_ROOT / "121-BOL")
        avg = result["q_co_displace"]["average_of_2_and_3"]
        mean_of_all = np.mean(
            [
                result["q_co_displace"]["run_1"],
                result["q_co_displace"]["run_2"],
                result["q_co_displace"]["run_3"],
            ]
        )
        # Run 1 for 121-BOL is much larger, so the two averages must differ
        assert avg != pytest.approx(mean_of_all, abs=1e-6)

    def test_q_co_stripping_close_to_reference(self):
        """For 121-BOL the documented Q_CO-stripping is 5.199 C."""
        result = process_case(DATA_ROOT / "121-BOL")
        assert result["q_co_stripping"] == pytest.approx(5.199, abs=0.05)
