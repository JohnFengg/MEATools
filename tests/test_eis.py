#!/usr/bin/env python
"""Tests for EIS analysis."""

import os
import numpy as np
import pytest
from meatools.subcomands.eis import read_dta_data, EIS_calc


class TestReadDtaData:
    """Test EIS DTA data reading."""

    def test_read_valid_data(self, temp_dir):
        """Test reading valid EIS data."""
        filepath = os.path.join(temp_dir, "test_eis.DTA")
        with open(filepath, "w", encoding="ISO-8859-1") as f:
            f.write("HEADER\n")
            f.write("ZCURVE\n")
            f.write("1 2 3 4 5\n")
            for i in range(10):
                freq = 10**i
                zreal = 0.05 + i * 0.001
                zimag = 0.01 * np.sin(i)
                f.write(f"{i}\t{freq}\t{zreal}\t{zimag}\t0\n")
        
        result = read_dta_data(filepath)
        assert isinstance(result, np.ndarray)
        assert result.shape[1] >= 3

    def test_empty_file(self, temp_dir):
        """Test reading empty file."""
        filepath = os.path.join(temp_dir, "empty.DTA")
        with open(filepath, "w") as f:
            f.write("")
        
        result = read_dta_data(filepath)
        assert len(result) == 0


class TestEISCalc:
    """Test EIS calculation."""

    def test_basic_calculation(self, sample_eis_data):
        """Test basic EIS calculation."""
        hfr, r_ion, r_ion_std, sample_num = EIS_calc(sample_eis_data, 0, "test")
        
        assert isinstance(hfr, (int, float))
        assert isinstance(r_ion, (int, float))
        assert isinstance(r_ion_std, (int, float))
        assert isinstance(sample_num, int)

    def test_positive_imaginary_validation(self):
        """Test validation when no positive imaginary values exist."""
        data = np.ones((100, 5))
        data[:, 4] = -1
        
        with pytest.raises(ValueError):
            EIS_calc(data, 0, "test")

    def test_mixed_imaginary(self):
        """Test with mixed positive/negative imaginary values."""
        data = np.ones((100, 5))
        data[:50, 4] = 1
        data[50:, 4] = -1
        
        hfr, r_ion, r_ion_std, sample_num = EIS_calc(data, 0, "test")
        assert hfr > 0
