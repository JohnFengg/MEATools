#!/usr/bin/env python
"""CLI subcommand for sulfonate group coverage calculation.

sulf-cvrg always launches the interactive browser UI so the user can adjust
peak boundaries.  Inside the UI the user may Submit custom boundaries or Skip
to use the default boundaries.
"""

import argparse
import json
import sys
from pathlib import Path

from meatools.sulfonate_coverage import has_sulfonate_coverage_files
from meatools.sulfonate_coverage_interactive import launch_interactive


def _default_output_path(case_dir):
    """Default JSON output path when running inside a case directory."""
    case_path = Path(case_dir).resolve()
    return case_path / "results" / "sulf-cvrg" / "sulfonate_coverage.json"


def run_sulfonate_coverage(args=None):
    """Run sulfonate coverage analysis for one or more case folders.

    Always opens the interactive peak-boundary UI.  The result is saved to
    ``<case>/sulfonate_coverage.json`` by default.
    """
    parser = argparse.ArgumentParser(
        prog="mea sulf-cvrg",
        description=(
            "Calculate sulfonate group coverage from CO displacement and "
            "CO stripping data using an interactive browser UI."
        ),
    )
    parser.add_argument(
        "case_dirs",
        nargs="*",
        default=["."],
        help="Case directories (default: current directory)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional JSON output file for results (default: <case>/sulfonate_coverage.json)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port for the interactive server (default: auto-select)",
    )

    if args is None:
        parsed = parser.parse_args()
    else:
        parsed = parser.parse_args(args)

    results = []
    for case_dir in parsed.case_dirs:
        if not Path(case_dir).is_dir():
            print(f"Error: not a directory: {case_dir}", file=sys.stderr)
            sys.exit(1)

        output_path = (
            Path(parsed.output)
            if parsed.output
            else _default_output_path(case_dir)
        )

        if not has_sulfonate_coverage_files(case_dir):
            print(
                f"Skipping {case_dir}: no coverage data folders found.",
                file=sys.stderr,
            )
            continue

        result = launch_interactive(case_dir, output_path, port=parsed.port)
        if result is None:
            print(
                f"No result saved for {case_dir} (interactive session closed without submit/skip).",
                file=sys.stderr,
            )
            continue

        results.append(result)
        line = (
            f"{result['case']}: SO3 coverage = {result['so3_coverage_percent']:.2f}%  "
            f"(Qd_avg={result['q_co_displace']['average_of_2_and_3']:.6f} C, "
            f"Qs={result['q_co_stripping']:.6f} C)"
        )
        print(line)

    output = {
        "summary": [
            f"{r['case']}: SO3 coverage = {r['so3_coverage_percent']:.2f}%"
            for r in results
        ],
        "results": results,
    }
    return output


if __name__ == "__main__":
    run_sulfonate_coverage()
