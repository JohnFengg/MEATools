#!/usr/bin/env python
"""Integration tests for meatools."""

import json
import numpy as np


class TestEndToEnd:
    """End-to-end tests for the complete workflow."""

    def test_imports(self):
        """Test that all modules can be imported."""
        from meatools.utils.serialization import NumpyEncoder
        from meatools.utils.file_utils import find_and_sort_load_dta_files
        from meatools.parsers.dta_parser import parse_dta_auto
        from meatools.cv_processor import process_curve_data
        from meatools.subcomands.eis import read_dta_data, EIS_calc
        
        assert True

    def test_serialization_roundtrip(self):
        """Test JSON serialization roundtrip."""
        from meatools.utils.serialization import NumpyEncoder
        
        data = {
            "array": np.array([1, 2, 3]),
            "float": np.float64(3.14),
            "int": np.int32(42)
        }
        
        serialized = json.dumps(data, cls=NumpyEncoder)
        deserialized = json.loads(serialized)
        
        assert deserialized["array"] == [1, 2, 3]
        assert deserialized["float"] == 3.14
        assert deserialized["int"] == 42
