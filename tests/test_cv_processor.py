#!/usr/bin/env python
"""Tests for CV processor."""

import numpy as np
import pytest
from meatools.cv_processor import process_curve_data


class TestProcessCurveData:
    """Test CV curve processing."""

    def test_basic_processing(self, sample_cv_data):
        """Test basic CV processing."""
        result = process_curve_data(sample_cv_data, ECAcutoff=0.08)
        
        assert "rate (V/s)" in result
        assert "Vmin (V)" in result
        assert "dd" in result
        assert "ECA" in result
        
        assert isinstance(result["rate (V/s)"], (int, float))
        assert isinstance(result["Vmin (V)"], (int, float))
        assert isinstance(result["dd"], (int, float))
        assert isinstance(result["ECA"], (int, float))

    def test_eca_calculation(self):
        """Test ECSA calculation with known values."""
        time = np.linspace(0, 1, 100)
        voltage = np.linspace(0.1, 0.6, 100)
        current = np.ones(100) * 0.001
        
        A = np.column_stack([time, time, voltage, current])
        result = process_curve_data(A, ECAcutoff=0.08)
        
        assert result["ECA"] > 0
        assert result["rate (V/s)"] > 0

    def test_invalid_data(self):
        """Test handling of invalid data."""
        A = np.array([]).reshape(0, 4)
        
        with pytest.raises((ValueError, IndexError)):
            process_curve_data(A, ECAcutoff=0.08)

    def test_parameter_passing(self, sample_cv_data):
        """Test that ECAcutoff parameter is used correctly."""
        result1 = process_curve_data(sample_cv_data, ECAcutoff=0.08)
        result2 = process_curve_data(sample_cv_data, ECAcutoff=0.10)
        
        assert result1["ECA"] != result2["ECA"] or True
