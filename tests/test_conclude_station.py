#!/usr/bin/env python
"""B10: conclude station extraction must handle full-width parentheses
(75/505 = 15% of cases had station_num. = null).

Before the fix only ASCII ``()`` matched; 51 cases use full-width
``（…）`` and 24 have no parens. Names mixing both styles captured the
first ASCII group, which can be a wrong value (``…（7-8#NEW） (1)``
-> ``1``).

After the fix both styles are matched and the group containing the
station pattern is preferred.
"""

import json
import os
import subprocess
import sys

import pytest

from meatools.subcomands.conclude import extract_station


class TestExtractStation:

    @pytest.mark.parametrize("name, expected", [
        # ASCII parens (previously worked)
        ("HRL-2#_WT2025-1696-RD-1057 (7-4_SN24)", "7-4_SN24"),
        ("HRL-1#_WT2025-1657-RD-1027 (7-4_SN24) 10000圈", "7-4_SN24"),
        # full-width parens (B10: 51/75 crashed-to-null cases)
        ("HRL-ALD-25-C11_WT2025-1480-RD-0888（7-9_SN23）", "7-9_SN23"),
        ("HRL-ALD-25-C10_WT2025-1479-RD-0887（7-9_SN23）", "7-9_SN23"),
        # mixed styles: prefer the station-looking group
        ("HRL-D101_WT2025-1401-RD-0820（7-8#NEW） (1)", "7-8#NEW"),
        ("HRL-X (1) (7-1#1)", "7-1#1"),
        # parens without a station pattern: fall back to first group
        ("HRL-B003 (retest)", "retest"),
        ("HRL-B003", None),
        ("", None),
    ])
    def test_station_extraction(self, name, expected):
        assert extract_station(name) == expected


class TestConcludePipeline:
    """mea conclude on a full-width-paren case folder."""

    def _run_conclude(self, case_dir):
        env = dict(os.environ, MPLBACKEND='Agg')
        return subprocess.run(
            [sys.executable, '-m', 'meatools.subcomands.conclude'],
            cwd=str(case_dir), env=env,
            capture_output=True, text=True, timeout=300)

    def test_fullwidth_paren_folder(self, tmp_path):
        case = tmp_path / 'HRL-ALD-25-C11_WT2025-1480-RD-0888（7-9_SN23）'
        case.mkdir()
        (case / 'results').mkdir()
        proc = self._run_conclude(case)
        assert proc.returncode == 0, proc.stderr[-1500:]

        result = json.loads(
            (case / 'results.json').read_text(encoding='utf-8'))
        assert result['station_num.'] == '7-9_SN23'

    def test_mixed_style_folder_prefers_station(self, tmp_path):
        case = tmp_path / 'HRL-D101_WT2025-1401-RD-0820（7-8#NEW） (1)'
        case.mkdir()
        (case / 'results').mkdir()
        proc = self._run_conclude(case)
        assert proc.returncode == 0, proc.stderr[-1500:]

        result = json.loads(
            (case / 'results.json').read_text(encoding='utf-8'))
        assert result['station_num.'] == '7-8#NEW'

    def test_no_paren_folder_stays_null(self, tmp_path):
        case = tmp_path / 'HRL-B003'
        case.mkdir()
        (case / 'results').mkdir()
        proc = self._run_conclude(case)
        assert proc.returncode == 0, proc.stderr[-1500:]

        result = json.loads(
            (case / 'results.json').read_text(encoding='utf-8'))
        assert result['station_num.'] is None