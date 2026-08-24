#!/usr/bin/env python
"""B11: mea ecsadry must not overwrite ecsa's log file.

ecsa_dry.py opened 'logs/ecsa_normal.log' (copy-paste name), so in the
`mea all` order (ecsa -> ecsadry) the dry run clobbered the normal
ECSA log. It must write 'logs/ecsa_dry.log'.
"""

import os
import subprocess
import sys


def _run(module, case_dir):
    env = dict(os.environ, MPLBACKEND='Agg')
    return subprocess.run(
        [sys.executable, '-m', module], cwd=str(case_dir),
        env=env, capture_output=True, text=True, timeout=300)


class TestEcsaDryLogName:

    def test_ecsa_dry_writes_own_log(self, tmp_path):
        (tmp_path / 'logs').mkdir()
        proc = _run('meatools.subcomands.ecsa_dry', tmp_path)
        assert proc.returncode == 0, proc.stderr[-1500:]
        assert (tmp_path / 'logs' / 'ecsa_dry.log').exists()
        assert not (tmp_path / 'logs' / 'ecsa_normal.log').exists()

    def test_ecsa_then_ecsadry_keeps_both_logs(self, tmp_path):
        # `mea all` order: ecsa first, ecsadry second. The dry run used
        # to clobber the normal log.
        (tmp_path / 'ECSA' / 'ECSA1').mkdir(parents=True)
        (tmp_path / '干质子可及率' / '100%RH' / 'Cathode CO CV').mkdir(
            parents=True)

        proc = _run('meatools.subcomands.ecsa_normal', tmp_path)
        assert proc.returncode == 0, proc.stderr[-1500:]
        assert (tmp_path / 'logs' / 'ecsa_normal.log').exists()
        normal_content = (tmp_path / 'logs' / 'ecsa_normal.log') \
            .read_text(encoding='utf-8', errors='replace')
        assert 'Subfolders containing CV DTA files' in normal_content

        proc = _run('meatools.subcomands.ecsa_dry', tmp_path)
        assert proc.returncode == 0, proc.stderr[-1500:]

        # both logs exist and the normal log survived
        assert (tmp_path / 'logs' / 'ecsa_dry.log').exists()
        assert (tmp_path / 'logs' / 'ecsa_normal.log').exists()
        assert (tmp_path / 'logs' / 'ecsa_normal.log') \
            .read_text(encoding='utf-8', errors='replace') == normal_content