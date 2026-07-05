#!/usr/bin/env python
"""Unified DTA parser with automatic format detection."""

import os
import re
from collections import Counter
import numpy as np


def _read_from_line65(filepath):
    """Read file content starting from line 65."""
    with open(filepath, 'r') as f:
        for _ in range(65):
            next(f)
        return f.read()


def _write_temp(content, temp_path='temp'):
    """Write content to a temporary file."""
    with open(temp_path, 'w') as f:
        f.write(content)


def _load_temp_array(temp_path='temp', skiprows=2, usecols=range(8)):
    """Load numpy array from temporary file."""
    return np.loadtxt(temp_path, skiprows=skiprows, usecols=usecols)


def parse_dta_format1(filepath, process_callback, log=None, temp_path='temp'):
    """Parse DTA format 1 (single CURVE block, value-based splitting).

    Args:
        filepath: Path to the DTA file.
        process_callback: Callable(A, label) -> dict to process each curve.
        log: Optional file-like object for logging.
        temp_path: Path for temporary file.

    Returns:
        Dict with curve data and file_path.
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()

    curve_start = next((i for i, line in enumerate(lines) if line.strip().startswith('CURVE')), None)
    data_lines = lines[curve_start + 3:]
    data = [line.split()[-1] for line in data_lines if line.strip() and len(line.split()) >= 10]

    value_counts = Counter(data)
    if log:
        log.write("\nValue counts in the last column:\n")
        for value, count in value_counts.items():
            log.write(f"Value {value}: {count} occurrences\n")

    data_dump = {}
    for value in value_counts.keys():
        try:
            val_float = float(value)
        except ValueError:
            continue
        if 0 < val_float < len(value_counts) - 1:
            with open(temp_path, 'w') as f:
                for line in data_lines:
                    if line.strip() and len(line.split()) >= 10 and line.split()[-1] == value:
                        f.write(line.strip() + '\n')

            A = _load_temp_array(temp_path)
            dump = process_callback(A, value)
            data_dump[f"curve_{value}"] = dump
            if os.path.exists(temp_path):
                os.remove(temp_path)

    data_dump["file_path"] = filepath
    return data_dump


def parse_dta_format2(filepath, process_callback, temp_path='temp'):
    """Parse DTA format 2 (multiple CURVE blocks).

    Args:
        filepath: Path to the DTA file.
        process_callback: Callable(A, label) -> dict to process each curve.
        temp_path: Path for temporary file.

    Returns:
        Dict with curve data and file_path.
    """
    content_from_line65 = _read_from_line65(filepath)
    u = re.split('CURVE', content_from_line65)

    data_dump = {}
    for j in range(len(u)):
        _write_temp(u[j], temp_path)
        A = _load_temp_array(temp_path)
        if 0 < j < len(u) - 1:
            dump = process_callback(A, j)
            data_dump[f"curve_{j}"] = dump
        if os.path.exists(temp_path):
            os.remove(temp_path)

    data_dump["file_path"] = filepath
    return data_dump


def detect_dta_format(filepath):
    """Detect DTA format by counting CURVE occurrences after line 65.

    Returns:
        1 for format1 (fewer than 2 CURVE blocks), 2 for format2.
    """
    content_from_line65 = _read_from_line65(filepath)
    num_curves = len(re.split('CURVE', content_from_line65))
    return 1 if num_curves < 2 else 2


def parse_dta_auto(filepath, process_callback, log=None, temp_path='temp'):
    """Automatically detect format and parse DTA file.

    Args:
        filepath: Path to the DTA file.
        process_callback: Callable(A, label) -> dict to process each curve.
        log: Optional file-like object for logging.
        temp_path: Path for temporary file.

    Returns:
        Dict with curve data and file_path.
    """
    fmt = detect_dta_format(filepath)
    if fmt == 1:
        if log:
            log.write(f'different format (detected {fmt})\n')
        return parse_dta_format1(filepath, process_callback, log=log, temp_path=temp_path)
    else:
        return parse_dta_format2(filepath, process_callback, temp_path=temp_path)
