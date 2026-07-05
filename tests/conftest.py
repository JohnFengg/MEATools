#!/usr/bin/env python
"""Pytest configuration and fixtures for meatools tests."""

import pytest
import numpy as np
import os
import tempfile


@pytest.fixture
def sample_cv_data():
    """Generate synthetic CV data for testing."""
    time = np.linspace(0, 10, 1000)
    voltage = np.sin(time) * 0.3 + 0.4
    current = np.random.randn(1000) * 0.01
    return np.column_stack([time, time, voltage, current])


@pytest.fixture
def sample_eis_data():
    """Generate synthetic EIS data for testing."""
    freq = np.logspace(-2, 4, 100)
    zreal = np.ones(100) * 0.05 + np.random.randn(100) * 0.001
    zimag = np.sin(np.log10(freq)) * 0.01 + np.random.randn(100) * 0.001
    return np.column_stack([np.arange(100), np.arange(100), freq, zreal, zimag])


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
