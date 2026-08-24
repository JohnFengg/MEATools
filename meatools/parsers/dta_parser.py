#!/usr/bin/env python
"""Unified DTA parser with automatic format detection."""

import re
from collections import Counter

from ..utils.io_utils import loadtxt_from_text, open_text


def _read_from_line65(filepath):
    """Read file content starting from line 65 (0-based index 65).

    Real Gamry DTA files have a long header. Short files (tests / truncated
    exports) fall back to the full content so format detection still works.
    """
    with open_text(filepath) as f:
        lines = f.readlines()
    if len(lines) > 65:
        return "".join(lines[65:])
    return "".join(lines)


def parse_dta_format1(filepath, process_callback, log=None, temp_path=None):
    """Parse DTA format 1 (single CURVE block, value-based splitting).

    Args:
        filepath: Path to the DTA file.
        process_callback: Callable(A, label) -> dict to process each curve.
        log: Optional file-like object for logging.
        temp_path: Unused; kept for API compatibility.

    Returns:
        Dict with curve data and file_path.
    """
    with open_text(filepath) as file:
        lines = file.readlines()

    curve_start = next(
        (i for i, line in enumerate(lines) if line.strip().startswith("CURVE")),
        None,
    )
    if curve_start is None:
        return {"file_path": filepath}

    data_lines = lines[curve_start + 3 :]
    data = [
        line.split()[-1]
        for line in data_lines
        if line.strip() and len(line.split()) >= 10
    ]

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
            block = "\n".join(
                line.strip()
                for line in data_lines
                if line.strip()
                and len(line.split()) >= 10
                and line.split()[-1] == value
            )
            if not block:
                continue
            A = loadtxt_from_text(block, skiprows=0, usecols=range(8))
            if A.ndim == 1:
                A = A.reshape(1, -1)
            try:
                dump = process_callback(A, value)
            except Exception as exc:
                # One bad curve (e.g. no UPD region) must not kill the
                # whole file/case: record the error and keep going (B8).
                if log:
                    log.write(f"\nCurve {value} processing failed in "
                              f"{filepath}: {type(exc).__name__}: {exc}\n")
                data_dump[f"curve_{value}"] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }
                continue
            data_dump[f"curve_{value}"] = dump

    data_dump["file_path"] = filepath
    return data_dump


def parse_dta_format2(filepath, process_callback, temp_path=None):
    """Parse DTA format 2 (multiple CURVE blocks).

    Args:
        filepath: Path to the DTA file.
        process_callback: Callable(A, label) -> dict to process each curve.
        temp_path: Unused; kept for API compatibility.

    Returns:
        Dict with curve data and file_path.
    """
    content_from_line65 = _read_from_line65(filepath)
    u = re.split("CURVE", content_from_line65)

    data_dump = {}
    for j in range(len(u)):
        try:
            A = loadtxt_from_text(u[j])
        except Exception:
            continue
        if A.ndim == 1:
            A = A.reshape(1, -1)
        if 0 < j < len(u) - 1:
            try:
                dump = process_callback(A, j)
            except Exception as exc:
                # One bad curve must not kill the whole file/case (B8).
                dump = {"error": f"{type(exc).__name__}: {exc}"}
            data_dump[f"curve_{j}"] = dump

    data_dump["file_path"] = filepath
    return data_dump


def detect_dta_format(filepath):
    """Detect DTA format by counting CURVE blocks in the body.

    Returns:
        1 for format1 (single CURVE block), 2 for format2 (multiple).
    """
    content = _read_from_line65(filepath)
    # Count complete blocks demarcated by a line starting with CURVE
    blocks = re.split(r"(?m)^\s*CURVE", content)
    # blocks[0] is whatever is before the first CURVE; ignore empty trailing blocks
    non_empty = [b for b in blocks[1:] if b.strip()]
    return 1 if len(non_empty) < 2 else 2


def parse_dta_auto(filepath, process_callback, log=None, temp_path=None):
    """Automatically detect format and parse DTA file.

    Args:
        filepath: Path to the DTA file.
        process_callback: Callable(A, label) -> dict to process each curve.
        log: Optional file-like object for logging.
        temp_path: Unused; kept for API compatibility.

    Returns:
        Dict with curve data and file_path.
    """
    fmt = detect_dta_format(filepath)
    if fmt == 1:
        if log:
            log.write(f"different format (detected {fmt})\n")
        return parse_dta_format1(
            filepath, process_callback, log=log, temp_path=temp_path
        )
    return parse_dta_format2(filepath, process_callback, temp_path=temp_path)
