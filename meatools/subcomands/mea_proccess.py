import subprocess
import sys
import os
from types import SimpleNamespace

from meatools.sulfonate_coverage import (
    has_sulfonate_coverage_files,
    resolve_case_dir,
)


def _run_module(module):
    """Run a subcommand module and return its exit code (B9)."""
    proc = subprocess.run([sys.executable, "-m", module])
    return proc.returncode


def run_test_sequence(args=None):
    return _run_module("meatools.subcomands.test_squence")


def run_otr(args=None):
    return _run_module("meatools.subcomands.impedence_calc")


def run_ecsa(args=None):
    return _run_module("meatools.subcomands.ecsa_normal")


def run_ecsa_dry(args=None):
    return _run_module("meatools.subcomands.ecsa_dry")


def run_lsv(args=None):
    return _run_module("meatools.subcomands.lsv")


def run_conclude(args=None):
    return _run_module("meatools.subcomands.conclude")


def run_eis(args=None):
    return _run_module("meatools.subcomands.eis")


def _run_sulf_interactive(args=None):
    """Pipeline-context sulf-cvrg: interactive boundary selection.

    Blocks until the user submits or skips in the browser UI.  Under
    ``mea web`` the page is embedded in the front-end instead of opening
    a new browser window (MEATOOLS_SULF_UI_NO_OPEN).
    """
    return run_sulfonate_coverage(SimpleNamespace(sulf_args=[]))


def _run_sulf_auto(args=None):
    """Pipeline-context sulf-cvrg without interaction (default boundaries)."""
    return run_sulfonate_coverage(
        SimpleNamespace(sulf_args=['--non-interactive']))


def run_all(args=None):
    """Run the full pipeline, failing fast on the first non-zero step (B9)."""
    dirs = [dir for dir in os.listdir() if os.path.isdir(dir)]
    steps = [run_test_sequence]
    if "OTR" in dirs:
        steps.append(run_otr)
    steps += [run_ecsa, run_ecsa_dry, run_lsv, run_eis]
    no_sulf = bool(getattr(args, 'no_sulf', False))
    sulf_auto = bool(getattr(args, 'sulf_auto', False))
    # resolve_case_dir covers web uploads nested one level deep (the folder
    # picker keeps the top-level folder name); the subcommand resolves again
    # at run time, here we only need it for detection.
    if has_sulfonate_coverage_files(resolve_case_dir(".")):
        if no_sulf:
            print("[mea all] --no-sulf: skipping sulf-cvrg")
        elif sulf_auto:
            print("[mea all] --sulf-auto: sulf-cvrg with default peak "
                  "boundaries (no interaction)")
            steps.append(_run_sulf_auto)
        else:
            # Interactive by default: open the boundary-selection UI and
            # wait for Submit / Skip (B12 reverted to interactive).
            print("Detected sulf-cvrg data folders; opening interactive "
                  "peak-boundary selection...")
            steps.append(_run_sulf_interactive)
    elif not no_sulf and os.path.isdir('磺酸根覆盖度'):
        # B13: coverage data started but the layout isn't recognized -
        # warn instead of skipping silently.
        print("[mea all] Warning: 磺酸根覆盖度 folder present but no "
              "recognized 干质子可及率/100%RH/Cathode CO CV (or Cathode CV "
              "CO) layout; skipping sulf-cvrg.", file=sys.stderr)
    steps += [run_conclude, run_render]

    for step in steps:
        rc = step()
        if rc:
            print(f"[mea all] {step.__name__} failed with exit code {rc}; "
                  f"aborting pipeline", file=sys.stderr)
            return rc
    return 0


def run_render(args=None):
    return _run_module("meatools.subcomands.render")


def run_sulfonate_coverage(args=None):
    """Run the sulf-cvrg subcommand, forwarding extra CLI args (B12).

    ``args.sulf_args`` (captured with argparse.REMAINDER by the CLI) is
    appended to the child command, e.g. ['--non-interactive'] when called
    from ``mea all``.
    """
    extra = []
    if args is not None:
        extra = [str(a) for a in (getattr(args, 'sulf_args', None) or [])]
    cmd = [sys.executable, "-m", "meatools.subcomands.sulfonate_coverage"] + extra
    proc = subprocess.run(cmd)
    return proc.returncode