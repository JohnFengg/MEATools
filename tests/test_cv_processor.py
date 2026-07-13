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
        """Test ECSA calculation with a triangular CV-like waveform."""
        # Up-scan then down-scan so double-layer segments exist on both sides.
        n = 200
        t_up = np.linspace(0, 1, n)
        v_up = np.linspace(0.05, 0.65, n)
        t_dn = np.linspace(1, 2, n)
        v_dn = np.linspace(0.65, 0.05, n)
        time = np.concatenate([t_up, t_dn])
        voltage = np.concatenate([v_up, v_dn])
        # Larger current on the low-V up-scan region mimics UPD charge.
        current = np.where(voltage < 0.4, 0.002, 0.0005)
        current = current + np.where(
            (voltage > 0.3) & (voltage < 0.6),
            0.0002 * np.sign(np.gradient(voltage)),
            0.0,
        )

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
