#!/usr/bin/env python
"""Tests for serialization utilities."""

import json
import numpy as np
import pytest
from meatools.utils.serialization import NumpyEncoder


class TestNumpyEncoder:
    """Test NumpyEncoder class."""

    def test_encode_numpy_array(self):
        """Test encoding numpy arrays."""
        arr = np.array([1, 2, 3])
        result = json.dumps({"data": arr}, cls=NumpyEncoder)
        decoded = json.loads(result)
        assert decoded["data"] == [1, 2, 3]

    def test_encode_numpy_scalar(self):
        """Test encoding numpy scalars."""
        scalar = np.int64(42)
        result = json.dumps({"value": scalar}, cls=NumpyEncoder)
        decoded = json.loads(result)
        assert decoded["value"] == 42

    def test_encode_nested_structure(self):
        """Test encoding nested structures with numpy."""
        data = {
            "array": np.array([1.0, 2.0]),
            "scalar": np.float64(3.14),
            "nested": {
                "arr": np.array([4, 5])
            }
        }
        result = json.dumps(data, cls=NumpyEncoder)
        decoded = json.loads(result)
        assert decoded["array"] == [1.0, 2.0]
        assert decoded["scalar"] == 3.14
        assert decoded["nested"]["arr"] == [4, 5]

    def test_encode_regular_types(self):
        """Test that regular types still work."""
        data = {"str": "hello", "int": 42, "float": 3.14, "list": [1, 2, 3]}
        result = json.dumps(data, cls=NumpyEncoder)
        decoded = json.loads(result)
        assert decoded == data
