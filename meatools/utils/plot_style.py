#!/usr/bin/env python
"""Matplotlib defaults that work on Linux, macOS, and Windows."""

import matplotlib.pyplot as plt


def apply_unicode_font():
    """Prefer fonts that can show Chinese labels when available."""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
        "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False
