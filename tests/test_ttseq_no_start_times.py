#!/usr/bin/env python
"""B5: ttseq must not crash when NO CSV has a parseable start time
(11/505 real cases).

Before the fix, ``plot_step_timeline`` indexed ``steps_data[0]['start']``
on the empty list -> IndexError, killing the plots and the entire
polarization half of ttseq (the JSONs were written as {} and nothing
after).

After the fix the empty timeline is skipped with a log line and the
polarization half still runs (writing its JSONs, empty where applicable).
"""

import json
import os
import subprocess
import sys

import pytest

from meatools.subcomands.test_squence import plot_step_timeline
from tests.testdata_utils import write_hrl_csv
import numpy as np
from datetime import datetime


class _Log:
    def __init__(self):
        self.text = []

    def write(self, s):
        self.text.append(s)


def _no_start_time_csv(path, n=120):
    """HRL-style CSV whose header block has NO 'Start time' line
    (lines 6-8 carry no parseable start time)."""
    cols = ['elapsed time', 'current', 'cell_active_area']
    width = 1 + len(cols)

    def pad(cells):
        cells = list(cells) + [''] * (width - len(cells))
        return ','.join(cells)

    lines = [
        pad(['Test name', 'synthetic']),
        pad(['User name', 'test']),
        pad(['Comment', 'test']),
        pad(['Profile name', 'test.ini']),
        pad(['Initial log rate', '1000 ms']),
        pad(['Number of tags', str(width)]),
        pad(['No start time here', 'nothing']),  # B5 trigger
        pad(['~-~-~'] * 2),
        pad([''] + cols),
        pad([''] + ['units'] * len(cols)),
        pad(['Time stamp'] + cols),
    ]
    for i in range(n):
        lines.append(','.join([f'2025-09-15 00:00:{i % 60:02d}',
                               str(i + 1), str(1.0), str(5.0)]))
    with open(path, 'w', encoding='ISO-8859-1') as f:
        f.write('\n'.join(lines) + '\n')


class TestPlotStepTimelineGuard:

    def test_empty_steps_no_crash(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'results' / 'test_sequence').mkdir(parents=True)
        log = _Log()
        plot_step_timeline([], log=log)  # must not raise
        assert not (tmp_path / 'results' / 'test_sequence'
                    / 'test_timeline.png').exists()
        assert any('No steps' in s for s in log.text)

    def test_single_step_still_plots(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'results' / 'test_sequence').mkdir(parents=True)
        steps = [{'step': 1,
                  'start': '2025-09-15 09:00:00',
                  'end': '2025-09-15 09:10:00',
                  'duration': 10.0,
                  'name': 'Step 1: x.csv'}]
        plot_step_timeline(steps, log=_Log())
        assert (tmp_path / 'results' / 'test_sequence'
                / 'test_timeline.png').exists()


class TestTtseqWithoutStartTimes:

    def test_ttseq_completes_and_runs_pol_half(self, tmp_path):
        pol_dir = tmp_path / '极化'
        pol_dir.mkdir()
        # one CSV without any start time in lines 6-8
        _no_start_time_csv(pol_dir / 'HRL_Pol_noStartTime-1.csv')

        env = dict(os.environ, MPLBACKEND='Agg')
        proc = subprocess.run(
            [sys.executable, '-m', 'meatools.subcomands.test_squence'],
            cwd=str(tmp_path), env=env,
            capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, (
            f"ttseq crashed:\n{proc.stderr[-2000:]}")
        assert 'Traceback' not in proc.stderr

        # The polarization half ran: its JSONs exist (not skipped).
        for rel in (
            'results/test_sequence/all_csv_results_in_timeline.json',
            'results/test_sequence/test_order_in_timeline.json',
            'results/polarization/all_pol_results.json',
            'results/polarization/polarization.json',
            'results/polarization/polarization_results.json',
            'results/polarization/polarization_curves.png',
        ):
            assert (tmp_path / rel).exists(), f"missing output {rel}"

    def test_ttseq_all_files_without_start_time(self, tmp_path):
        # The B5 repro shape: every CSV lacks a parseable start time.
        _no_start_time_csv(tmp_path / 'HRL_Pol_a-1.csv')
        _no_start_time_csv(tmp_path / 'HRL_Pol_b-2.csv')

        env = dict(os.environ, MPLBACKEND='Agg')
        proc = subprocess.run(
            [sys.executable, '-m', 'meatools.subcomands.test_squence'],
            cwd=str(tmp_path), env=env,
            capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, (
            f"ttseq crashed on missing start times (B5):\n"
            f"{proc.stderr[-2000:]}")
        assert 'Traceback' not in proc.stderr

        timeline = json.loads(
            (tmp_path / 'results' / 'test_sequence'
             / 'all_csv_results_in_timeline.json')
            .read_text(encoding='utf-8'))
        assert timeline == {}
        # timeline JSON still written, no steps
        order = json.loads(
            (tmp_path / 'results' / 'test_sequence'
             / 'test_order_in_timeline.json')
            .read_text(encoding='utf-8'))
        assert set(order.keys()) == {'total_time (mins)'}
        assert order['total_time (mins)'] == 0
        # no Gantt chart for an empty timeline
        assert not (tmp_path / 'results' / 'test_sequence'
                    / 'test_timeline.png').exists()