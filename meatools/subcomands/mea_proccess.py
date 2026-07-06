import subprocess
import sys
import os


def run_test_sequence(args=None):
    subprocess.run([sys.executable, "-m", "meatools.subcomands.test_squence"])


def run_otr(args=None):
    subprocess.run([sys.executable, "-m", "meatools.subcomands.impedence_calc"])


def run_ecsa(args=None):
    subprocess.run([sys.executable, "-m", "meatools.subcomands.ecsa_normal"])


def run_ecsa_dry(args=None):
    subprocess.run([sys.executable, "-m", "meatools.subcomands.ecsa_dry"])


def run_lsv(args=None):
    subprocess.run([sys.executable, "-m", "meatools.subcomands.lsv"])


def run_conclude(args=None):
    subprocess.run([sys.executable, "-m", "meatools.subcomands.conclude"])


def run_eis(args=None):
    subprocess.run([sys.executable, "-m", "meatools.subcomands.eis"])


def run_all(args=None):
    dirs = [dir for dir in os.listdir() if os.path.isdir(dir)]
    run_test_sequence()
    if "OTR" in dirs:
        run_otr()
    run_ecsa()
    run_ecsa_dry()
    run_lsv()
    run_eis()
    run_conclude()


def run_render(args=None):
    subprocess.run([sys.executable, "-m", "meatools.subcomands.render"])


def run_sulfonate_coverage(args=None):
    subprocess.run([sys.executable, "-m", "meatools.subcomands.sulfonate_coverage"])
