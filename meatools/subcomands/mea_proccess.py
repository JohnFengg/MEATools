import subprocess
import sys
import os
from types import SimpleNamespace

from meatools.sulfonate_coverage import has_sulfonate_coverage_files


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


def _run_sulf_noninteractive(args=None):
    """Batch-context sulf-cvrg: default boundaries, no browser (B12)."""
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
    if has_sulfonate_coverage_files("."):
        if no_sulf:
            print("[mea all] --no-sulf: skipping sulf-cvrg")
        else:
            # Batch context: run the non-interactive default-boundary mode
            # so 'mea all' never hangs waiting at the browser (B12).
            print("Detected sulf-cvrg data folders; running "
                  "non-interactive peak selection...")
            steps.append(_run_sulf_noninteractive)
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