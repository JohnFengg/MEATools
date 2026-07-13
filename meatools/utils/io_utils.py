#!/usr/bin/env python
"""Cross-platform I/O helpers for instrument text files."""

from io import StringIO

import numpy as np

# Instrument exports (Gamry DTA, HRL CSV) are effectively Latin-1 / ISO-8859-1.
# Explicit encoding avoids Windows cp1252 / locale decode errors.
TEXT_ENCODING = "latin-1"


def open_text(path, mode="r"):
    """Open a text file with a stable encoding for instrument data."""
    return open(path, mode, encoding=TEXT_ENCODING)


def loadtxt_from_text(text, skiprows=2, usecols=range(8)):
    """Load a numeric array from an in-memory text block (no temp file)."""
    return np.loadtxt(StringIO(text), skiprows=skiprows, usecols=usecols)
