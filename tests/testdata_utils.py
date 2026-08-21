#!/usr/bin/env python
"""Helpers to build synthetic HRL-format case files for tests.

The HRL test-stand CSV layout (see real files under mea_data/):

    lines 0-5   Test name / User name / Comment / Profile name /
                Initial log rate / Number of tags
    line 6      Start time,<mm/dd/yy HH:MM:SS>
    line 7      separator
    lines 8-9   two free-text header lines (names / units)
    line 10     Time stamp,<col1>,<col2>,...
    line 11+    data rows
"""

import datetime as _dt


def write_hrl_csv(path, start_time, columns, data):
    """Write a synthetic HRL test CSV file.

    Args:
        path: Destination file path.
        start_time: ``datetime`` written into the 'Start time' header line.
        columns: Column names after 'Time stamp' (list of str).
        data: Mapping column name -> sequence of values (all same length).
    """
    width = 1 + len(columns)

    def pad(cells):
        cells = list(cells) + [''] * (width - len(cells))
        return ','.join(cells[:width])

    n = len(next(iter(data.values())))
    lines = [
        pad(["Test name", "synthetic"]),
        pad(["User name", "test"]),
        pad(["Comment", "test"]),
        pad(["Profile name", "test.ini"]),
        pad(["Initial log rate", "1000 ms"]),
        pad(["Number of tags", str(width)]),
        pad(["Start time", start_time.strftime('%m/%d/%y %H:%M:%S')]),
        pad(["~-~-~"] * 2),
        pad([""] + list(columns)),
        pad([""] + ["units"] * len(columns)),
        pad(["Time stamp"] + list(columns)),
    ]
    for i in range(n):
        ts = _dt.datetime(2025, 9, 15, 0, i // 60, i % 60)
        cells = [ts.strftime('%Y-%m-%d %H:%M:%S'),
                 *[str(data[c][i]) for c in columns]]
        lines.append(','.join(cells))
    with open(path, 'w', encoding='ISO-8859-1') as f:
        f.write('\n'.join(lines) + '\n')