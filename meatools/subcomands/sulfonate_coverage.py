#!/usr/bin/env python
"""CLI subcommand for sulfonate group coverage calculation."""

import argparse
import json
import sys
from pathlib import Path

from meatools.sulfonate_coverage import has_sulfonate_coverage_files, process_case
from meatools.sulfonate_coverage_interactive import launch_interactive


def _default_output_path(case_dir):
    """Default JSON output path when running inside a case directory."""
    case_path = Path(case_dir).resolve()
    return case_path / "sulfonate_coverage.json"


def run_sulfonate_coverage(args=None):
    """Run sulfonate coverage analysis for one or more case folders."""
    parser = argparse.ArgumentParser(
        prog="mea sulf-cvrg",
        description="Calculate sulfonate group coverage from CO displacement and CO stripping data.",
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
        "--peak-pre",
        type=float,
        default=13.0,
        help="Seconds before the CO displacement peak minimum to start integration (default: 13)",
    )
    parser.add_argument(
        "--peak-post",
        type=float,
        default=6.0,
        help="Seconds after the CO displacement peak minimum to end integration (default: 6)",
    )
    parser.add_argument(
        "--v-start",
        type=float,
        default=0.5,
        help="Lower voltage bound for CO stripping integration (default: 0.5 V)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch interactive browser UI to adjust peak boundaries",
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

    co_displace_kwargs = {
        "peak_pre": parsed.peak_pre,
        "peak_post": parsed.peak_post,
    }

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

        if parsed.interactive:
            if not has_sulfonate_coverage_files(case_dir):
                print(
                    f"Skipping interactive mode for {case_dir}: no coverage data folders found.",
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
        else:
            result = process_case(case_dir, co_displace_kwargs=co_displace_kwargs)
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, ensure_ascii=False)
            print(f"Results written to {output_path}")

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
