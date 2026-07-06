#!/usr/bin/env python
"""CLI subcommand for sulfonate group coverage calculation."""

import argparse
import json
import sys
from pathlib import Path

from meatools.sulfonate_coverage import process_case


def run_sulfonate_coverage(args=None):
    """Run sulfonate coverage analysis for one or more case folders."""
    parser = argparse.ArgumentParser(
        prog="mea sulfonate-coverage",
        description="Calculate sulfonate group coverage from CO displacement and CO stripping data.",
    )
    parser.add_argument(
        "case_dirs",
        nargs="+",
        help="Case directories (e.g. 114-BOL 114-EOL 121-BOL)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional JSON output file for results",
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
        result = process_case(case_dir, co_displace_kwargs=co_displace_kwargs)
        results.append(result)

    summary_lines = []
    for r in results:
        line = (
            f"{r['case']}: SO3 coverage = {r['so3_coverage_percent']:.2f}%  "
            f"(Qd_avg={r['q_co_displace']['average_of_2_and_3']:.6f} C, "
            f"Qs={r['q_co_stripping']:.6f} C)"
        )
        summary_lines.append(line)
        print(line)

    output = {
        "summary": summary_lines,
        "results": results,
    }

    if parsed.output:
        with open(parsed.output, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)
        print(f"\nResults written to {parsed.output}")

    return output


if __name__ == "__main__":
    run_sulfonate_coverage()
