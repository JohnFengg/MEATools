#!/usr/bin/env python
"""Render results.json into a readable HTML report."""

import json
import os
import sys


def _render_scalar(value):
    """Render a scalar value as HTML."""
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "<em>N/A</em>"
    return str(value)


def _render_json(value, level=0):
    """Recursively render JSON value to HTML."""
    if isinstance(value, dict):
        return _render_dict(value, level)
    if isinstance(value, list):
        return _render_list(value, level)
    return f"<code>{_render_scalar(value)}</code>"


def _render_dict(data, level=0):
    """Render a dictionary as an HTML table."""
    if not data:
        return "<em>empty</em>"
    rows = []
    for key, value in data.items():
        rows.append(
            f"<tr><th>{key}</th><td>{_render_json(value, level + 1)}</td></tr>"
        )
    return f"<table>{''.join(rows)}</table>"


def _render_list(data, level=0):
    """Render a list as an HTML list or table."""
    if not data:
        return "<em>empty</em>"

    # List of scalars -> bullet list
    if all(not isinstance(x, (dict, list)) for x in data):
        items = "".join(f"<li>{_render_json(x, level + 1)}</li>" for x in data)
        return f"<ul>{items}</ul>"

    # List of dicts -> table with union of keys as columns
    if all(isinstance(x, dict) for x in data):
        keys = []
        seen = set()
        for item in data:
            for key in item.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        header = "".join(f"<th>{key}</th>" for key in keys)
        rows = []
        for item in data:
            row = "".join(
                f"<td>{_render_json(item.get(key, ''), level + 1)}</td>"
                for key in keys
            )
            rows.append(f"<tr>{row}</tr>")
        return (
            f"<table><thead><tr>{header}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    # Mixed list -> bullet list
    items = "".join(f"<li>{_render_json(x, level + 1)}</li>" for x in data)
    return f"<ul>{items}</ul>"


def render_results(input_path="results.json", output_path="results.html"):
    """Render a results.json file to HTML.

    Args:
        input_path: Path to results.json.
        output_path: Path for the generated HTML file.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Results file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    sample = results.get("sample", "Unknown")
    station = results.get("station_num.", "N/A")
    area = results.get("sample_area (cm^2)", "N/A")

    sulf = results.get("Sulfonate_Coverage") or {}
    so3 = sulf.get("so3_coverage_percent") if isinstance(sulf, dict) else None
    so3_summary = (
        f"{so3:.4g} %"
        if isinstance(so3, (int, float))
        else ("empty — re-run mea conclude after sulf-cvrg" if sulf == {} else "N/A")
    )

    # Sulfonate first so it is easy to find (was buried mid-page).
    section_order = [
        ("Sulfonate_Coverage", "Sulfonate Coverage"),
        ("O_Transfer_Resistance", "Oxygen Transfer Resistance"),
        ("ECSA", "ECSA"),
        ("ECSA_Dry", "ECSA Dry"),
        ("LSV", "LSV"),
        ("Polarization", "Polarization"),
        ("EIS", "EIS"),
        ("Test_Sequence", "Test Sequence"),
    ]

    sections = []
    for key, title in section_order:
        if key not in results:
            continue
        sections.append(
            f"<section><h2>{title}</h2>{_render_json(results[key])}</section>"
        )

    # Add any remaining keys at the end
    handled = {k for k, _ in section_order}
    for key in sorted(results.keys()):
        if key in handled:
            continue
        sections.append(
            f"<section><h2>{key}</h2>{_render_json(results[key])}</section>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>MEA Analysis Report - {sample}</title>
    <style>
        :root {{
            --primary: #2e7d32;
            --primary-light: #e8f5e9;
            --border: #ddd;
            --bg: #f5f5f5;
            --card: #ffffff;
            --text: #333333;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                         Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 0;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px;
        }}
        header {{
            background: var(--card);
            padding: 25px 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 25px;
        }}
        h1 {{
            margin: 0 0 10px;
            color: var(--primary);
            font-size: 1.8rem;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .summary-item {{
            background: var(--primary-light);
            padding: 12px 15px;
            border-radius: 6px;
        }}
        .summary-item strong {{
            display: block;
            color: var(--primary);
            font-size: 0.85rem;
            margin-bottom: 4px;
        }}
        section {{
            background: var(--card);
            padding: 25px 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 25px;
        }}
        h2 {{
            margin: 0 0 20px;
            color: var(--primary);
            font-size: 1.4rem;
            border-bottom: 2px solid var(--primary-light);
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 0.95rem;
        }}
        th, td {{
            border: 1px solid var(--border);
            padding: 10px 12px;
            text-align: left;
            vertical-align: top;
        }}
        th {{
            background: #f0f0f0;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background: #fafafa;
        }}
        ul {{
            margin: 5px 0;
            padding-left: 25px;
        }}
        li {{
            margin: 4px 0;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: "SF Mono", Monaco, Consolas, monospace;
            font-size: 0.9em;
        }}
        footer {{
            text-align: center;
            color: #888;
            font-size: 0.85rem;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>MEA Analysis Report</h1>
            <div class="summary">
                <div class="summary-item">
                    <strong>Sample</strong>
                    {sample}
                </div>
                <div class="summary-item">
                    <strong>Station</strong>
                    {station}
                </div>
                <div class="summary-item">
                    <strong>Sample Area</strong>
                    {area} cm²
                </div>
                <div class="summary-item">
                    <strong>SO₃ Coverage</strong>
                    {so3_summary}
                </div>
            </div>
        </header>
        {''.join(sections)}
        <footer>
            Generated by meatools render
        </footer>
    </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Rendered report: {os.path.abspath(output_path)}")


def main():
    """CLI entry point."""
    input_path = "results.json"
    output_path = "results.html"
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    render_results(input_path, output_path)


if __name__ == "__main__":
    main()
