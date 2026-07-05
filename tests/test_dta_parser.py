#!/usr/bin/env python
"""Tests for DTA parser."""

import os
import tempfile
import numpy as np
import pytest
from meatools.parsers.dta_parser import (
    detect_dta_format,
    parse_dta_auto,
    parse_dta_format1,
    parse_dta_format2
)


class TestDetectDtaFormat:
    """Test DTA format detection."""

    def test_format1_detection(self, temp_dir):
        """Test detecting format 1 (single CURVE)."""
        filepath = os.path.join(temp_dir, "test_format1.DTA")
        with open(filepath, "w") as f:
            f.write("HEADER\n")
            f.write("CURVE1\n")
            f.write("1 2 3 4 5 6 7 8\n")
            for i in range(10):
                f.write(f"{i} {i*0.1} 0.4 0.001 0 0 0 1\n")
        
        result = detect_dta_format(filepath)
        assert result == 1

    def test_format2_detection(self, temp_dir):
        """Test detecting format 2 (multiple CURVEs)."""
        filepath = os.path.join(temp_dir, "test_format2.DTA")
        with open(filepath, "w") as f:
            f.write("HEADER\n")
            for j in range(3):
                f.write(f"CURVE{j}\n")
                f.write("1 2 3 4 5 6 7 8\n")
                for i in range(10):
                    f.write(f"{i} {i*0.1} 0.4 0.001 0 0 0 1\n")
        
        result = detect_dta_format(filepath)
        assert result == 2


class TestParseDtaAuto:
    """Test automatic DTA parsing."""

    def test_parse_format1(self, temp_dir):
        """Test parsing format 1 automatically."""
        filepath = os.path.join(temp_dir, "test_format1.DTA")
        with open(filepath, "w") as f:
            f.write("HEADER\n")
            f.write("CURVE1\n")
            f.write("1 2 3 4 5 6 7 8\n")
            for i in range(10):
                f.write(f"{i} {i*0.1} 0.4 0.001 0 0 0 1\n")
        
        def callback(A, label):
            return {"shape": A.shape}
        
        result = parse_dta_auto(filepath, callback)
        assert "file_path" in result

    def test_parse_format2(self, temp_dir):
        """Test parsing format 2 automatically."""
        filepath = os.path.join(temp_dir, "test_format2.DTA")
        with open(filepath, "w") as f:
            f.write("HEADER\n")
            for j in range(3):
                f.write(f"CURVE{j}\n")
                f.write("1 2 3 4 5 6 7 8\n")
                for i in range(10):
                    f.write(f"{i} {i*0.1} 0.4 0.001 0 0 0 1\n")
        
        def callback(A, label):
            return {"shape": A.shape}
        
        result = parse_dta_auto(filepath, callback)
        assert "file_path" in result
