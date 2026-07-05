#!/usr/bin/env python
"""Tests for the render subcommand."""

import json
import os
import pytest

from meatools.subcomands.render import render_results


@pytest.fixture
def sample_results():
    """Return a sample results dictionary."""
    return {
        "sample": "HRL-D048",
        "station_num.": 7,
        "sample_area (cm^2)": 5.0,
        "O_Transfer_Resistance": {
            "R_total (Ohm*cm^2)": 0.123,
            "groups": [
                {"condition": "100kPa_80RH", "value": 0.1},
                {"condition": "150kPa_80RH", "value": 0.15},
            ],
        },
        "ECSA": {"value": 45.6, "unit": "m^2/g"},
        "Test_Sequence": ["step1", "step2", "step3"],
    }


class TestRenderResults:
    """Test render_results function."""

    def test_renders_html(self, sample_results, temp_dir):
        """Basic JSON -> HTML rendering produces expected output."""
        input_path = os.path.join(temp_dir, "results.json")
        output_path = os.path.join(temp_dir, "results.html")

        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(sample_results, f)

        render_results(input_path, output_path)

        assert os.path.isfile(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            html = f.read()

        assert "<!DOCTYPE html>" in html
        assert "MEA Analysis Report" in html
        assert sample_results["sample"] in html
        assert "Oxygen Transfer Resistance" in html
        assert "ECSA" in html
        assert "Test Sequence" in html
        assert "<table>" in html

    def test_missing_input_file(self, temp_dir):
        """Missing input file raises FileNotFoundError."""
        input_path = os.path.join(temp_dir, "missing.json")
        output_path = os.path.join(temp_dir, "results.html")

        with pytest.raises(FileNotFoundError):
            render_results(input_path, output_path)

    def test_empty_dict(self, temp_dir):
        """Empty results dict still renders."""
        input_path = os.path.join(temp_dir, "results.json")
        output_path = os.path.join(temp_dir, "results.html")

        with open(input_path, "w", encoding="utf-8") as f:
            json.dump({}, f)

        render_results(input_path, output_path)

        assert os.path.isfile(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            html = f.read()

        assert "Unknown" in html
        assert "<!DOCTYPE html>" in html

    def test_list_of_dicts_rendered_as_table(self, sample_results, temp_dir):
        """List of dicts is rendered as HTML table with union of keys."""
        input_path = os.path.join(temp_dir, "results.json")
        output_path = os.path.join(temp_dir, "results.html")

        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(sample_results, f)

        render_results(input_path, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            html = f.read()

        assert "<thead>" in html
        assert "100kPa_80RH" in html
        assert "150kPa_80RH" in html


class TestRenderModuleMain:
    """Test render module CLI entry point."""

    def test_module_main(self, sample_results, temp_dir, monkeypatch):
        """Module main renders using command-line arguments."""
        input_path = os.path.join(temp_dir, "results.json")
        output_path = os.path.join(temp_dir, "report.html")

        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(sample_results, f)

        monkeypatch.setattr("sys.argv", ["render", input_path, output_path])
        import meatools.subcomands.render as render_module

        render_module.main()

        assert os.path.isfile(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            html = f.read()
        assert sample_results["sample"] in html
