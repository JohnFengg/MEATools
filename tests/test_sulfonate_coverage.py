#!/usr/bin/env python
"""Tests for sulfonate group coverage calculation."""

import json
import os
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from meatools.sulfonate_coverage import (
    has_sulfonate_coverage_files,
    integrate_co_displace_peak,
    integrate_co_stripping,
    process_case,
    read_co_displace_csv,
    read_dta_curves,
)
from meatools.sulfonate_coverage_interactive import launch_interactive


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


class TestHasCoverageFiles:
    """Test auto-detection of coverage data folders."""

    def test_detects_coverage_folders(self):
        """Should return True for a case directory with the expected folders."""
        assert has_sulfonate_coverage_files(DATA_ROOT / "121-BOL") is True

    def test_rejects_non_coverage_directory(self, tmp_path):
        """Should return False for an unrelated directory."""
        assert has_sulfonate_coverage_files(tmp_path) is False


class TestPerRunBoundaries:
    """Test that per-run integration boundaries are respected."""

    def test_wider_window_increases_charge(self):
        """Increasing the integration window for run 2 should increase its charge."""
        default = process_case(DATA_ROOT / "121-BOL")
        wider = process_case(
            DATA_ROOT / "121-BOL",
            co_displace_kwargs={2: {"peak_pre": 18.0, "peak_post": 8.0}},
        )
        assert wider["q_co_displace"]["run_2"] > default["q_co_displace"]["run_2"]


class TestInteractiveServer:
    """Test the interactive peak-boundary selection server."""

    def test_skip_uses_defaults(self, tmp_path):
        """The /skip endpoint should compute with default boundaries."""
        output_path = tmp_path / "coverage_skip.json"
        port = 18765

        def run_server():
            launch_interactive(
                DATA_ROOT / "121-BOL",
                output_path,
                port=port,
                open_browser=False,
            )

        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        time.sleep(2)

        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/data") as resp:
                data = json.loads(resp.read().decode())

            payload = json.dumps({"boundaries": data["defaults"]}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/skip",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode())

            assert result["ok"] is True
            assert "coverage" in result
        finally:
            thread.join(timeout=10)

        assert output_path.exists()
        with open(output_path, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
        assert saved["so3_coverage_percent"] == pytest.approx(result["coverage"], abs=1e-6)
