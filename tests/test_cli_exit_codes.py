#!/usr/bin/env python
"""B9: failures must propagate as non-zero exit codes.

Before the fix, every `mea ...` invocation exited 0 even when the
subcommand crashed (273/505 real cases had tracebacks yet rc=0),
because mea_proccess swallowed subprocess return codes and cli.main()
never called sys.exit.

After the fix: each run_* returns the child's rc, run_all fails fast on
the first failing step, and the CLI exits with that code.
"""

import sys
from types import SimpleNamespace

import pytest

import meatools.cli as cli
import meatools.subcomands.mea_proccess as mp


def _fake_proc(rc):
    return SimpleNamespace(returncode=rc)


class TestRunWrappersReturnRc:
    """Each run_* returns the subprocess return code."""

    @pytest.mark.parametrize("runner", [
        mp.run_test_sequence, mp.run_otr, mp.run_ecsa, mp.run_ecsa_dry,
        mp.run_lsv, mp.run_conclude, mp.run_eis, mp.run_render,
    ])
    def test_returns_child_rc(self, monkeypatch, runner):
        seen = {}

        def fake_run(cmd, **kw):
            seen['cmd'] = cmd
            return _fake_proc(3)

        monkeypatch.setattr(mp.subprocess, 'run', fake_run)
        assert runner() == 3
        assert '-m' in seen['cmd']

    def test_success_returns_zero(self, monkeypatch):
        monkeypatch.setattr(mp.subprocess, 'run',
                            lambda cmd, **kw: _fake_proc(0))
        assert mp.run_test_sequence() == 0


class TestRunAllFailFast:
    """run_all: stops at the first failing step and returns its rc."""

    def test_stops_and_returns_first_failure(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = []

        def make_fake(name):
            def fake(args=None):
                calls.append(name)
                return 0 if name != 'run_ecsa' else 7
            return fake

        for name in ['run_test_sequence', 'run_otr', 'run_ecsa',
                     'run_ecsa_dry', 'run_lsv', 'run_eis',
                     'run_conclude', 'run_render']:
            monkeypatch.setattr(mp, name, make_fake(name))
        monkeypatch.setattr(mp, 'has_sulfonate_coverage_files',
                            lambda path: False)

        assert mp.run_all() == 7
        # ttseq ran, ecsa failed, nothing after ecsa ran
        assert calls == ['run_test_sequence', 'run_ecsa']

    def test_success_returns_zero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = []

        def make_fake(name):
            def fake(args=None):
                calls.append(name)
                return 0
            return fake

        for name in ['run_test_sequence', 'run_otr', 'run_ecsa',
                     'run_ecsa_dry', 'run_lsv', 'run_eis',
                     'run_conclude', 'run_render']:
            monkeypatch.setattr(mp, name, make_fake(name))
        monkeypatch.setattr(mp, 'has_sulfonate_coverage_files',
                            lambda path: False)

        assert mp.run_all() == 0
        assert 'run_conclude' in calls and 'run_render' in calls


class TestCliExitCode:
    """The CLI process exits with the step's return code."""

    def test_cli_exits_nonzero_on_failure(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, 'argv', ['mea', 'ttseq'])
        monkeypatch.setattr(mp.subprocess, 'run',
                            lambda cmd, **kw: _fake_proc(2))
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 2

    def test_cli_exits_zero_on_success(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['mea', 'ttseq'])
        monkeypatch.setattr(mp.subprocess, 'run',
                            lambda cmd, **kw: _fake_proc(0))
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0