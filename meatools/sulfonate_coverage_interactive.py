#!/usr/bin/env python
"""Interactive peak-boundary selection for sulfonate coverage.

Launches a local HTTP server and opens a browser page where the user can drag
integration boundaries on the CO-displacement peaks for runs 2 and 3.
"""

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from meatools.sulfonate_coverage import (
    find_case_files,
    integrate_co_displace_peak,
    integrate_co_stripping,
    read_co_displace_csv,
)


HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>磺酸根覆盖度 — 峰边界选择</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 1rem; background: #f8f9fa; }
  h1 { font-size: 1.4rem; margin-bottom: .5rem; }
  h2 { font-size: 1.1rem; margin: 1.5rem 0 .5rem; }
  .card { background: #fff; border: 1px solid #dee2e6; border-radius: .5rem; padding: 1rem; margin-bottom: 1rem; }
  canvas { width: 100%; height: 260px; background: #fff; border: 1px solid #dee2e6; border-radius: .25rem; cursor: col-resize; }
  .info { display: flex; gap: 2rem; flex-wrap: wrap; margin: .5rem 0; font-size: .95rem; }
  .info span { display: inline-block; }
  .metric { font-weight: 600; color: #0d6efd; }
  .btn { padding: .6rem 1.2rem; border: none; border-radius: .4rem; cursor: pointer; font-size: 1rem; margin-right: .75rem; }
  .btn-primary { background: #0d6efd; color: #fff; }
  .btn-secondary { background: #6c757d; color: #fff; }
  .result { font-size: 1.2rem; font-weight: 600; margin: 1rem 0; }
  .note { color: #6c757d; font-size: .9rem; margin-top: .5rem; }
  .dragging { cursor: ew-resize !important; }
</style>
</head>
<body>
  <h1>磺酸根覆盖度 — 交互式峰边界选择</h1>
  <div class="card">
    <p>拖动红色竖线调整 CO 置换峰（Run 2 / Run 3）的积分左右边界；覆盖率会实时更新。满意后点击“提交”，若不想调整可点击“Skip 使用默认值”。</p>
    <div class="result">SO₃ 覆盖度：<span id="coverage" class="metric">--</span> %</div>
    <button id="submitBtn" class="btn btn-primary">提交边界</button>
    <button id="skipBtn" class="btn btn-secondary">Skip 使用默认值</button>
    <div id="status" class="note"></div>
  </div>

  <div id="plots"></div>

<script>
const state = { runs: [], defaults: {}, boundaries: {} };

async function init() {
  const res = await fetch('/data');
  const payload = await res.json();
  state.runs = payload.runs;
  state.defaults = payload.defaults;
  state.boundaries = JSON.parse(JSON.stringify(payload.defaults));
  renderPlots();
  updateCoverage();
}

function renderPlots() {
  const container = document.getElementById('plots');
  container.innerHTML = '';
  state.runs.forEach((run, idx) => {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `<h2>Run ${run.run} — ${run.file}</h2>
      <div class="info">
        <span>基线：<span class="metric" id="baseline-${run.run}">${run.baseline.toFixed(6)}</span> A</span>
        <span>Qd：<span class="metric" id="qd-${run.run}">--</span> C</span>
        <span>左边界：<span class="metric" id="left-${run.run}">--</span> s</span>
        <span>右边界：<span class="metric" id="right-${run.run}">--</span> s</span>
      </div>
      <canvas id="canvas-${run.run}"></canvas>`;
    container.appendChild(card);
    drawRun(run);
  });
}

function drawRun(run) {
  const canvas = document.getElementById(`canvas-${run.run}`);
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height;

  const pad = { top: 20, right: 30, bottom: 40, left: 60 };
  const gw = w - pad.left - pad.right;
  const gh = h - pad.top - pad.bottom;

  const t = run.time, i = run.current;
  const tMin = Math.min(...t), tMax = Math.max(...t);
  const iMin = Math.min(...i), iMax = Math.max(...i);
  const vPad = (iMax - iMin) * 0.1 || 0.001;

  const x = v => pad.left + (v - tMin) / (tMax - tMin) * gw;
  const y = a => pad.top + gh - (a - (iMin - vPad)) / ((iMax + vPad) - (iMin - vPad)) * gh;

  ctx.clearRect(0, 0, w, h);

  // grid
  ctx.strokeStyle = '#e9ecef';
  ctx.lineWidth = 1;
  for (let k = 0; k <= 5; k++) {
    const yy = pad.top + gh * k / 5;
    ctx.beginPath(); ctx.moveTo(pad.left, yy); ctx.lineTo(pad.left + gw, yy); ctx.stroke();
    const val = (iMax + vPad) - ((iMax + vPad) - (iMin - vPad)) * k / 5;
    ctx.fillStyle = '#6c757d'; ctx.font = '11px sans-serif';
    ctx.fillText(val.toFixed(3), pad.left - 50, yy + 3);
  }
  for (let k = 0; k <= 5; k++) {
    const xx = pad.left + gw * k / 5;
    ctx.beginPath(); ctx.moveTo(xx, pad.top); ctx.lineTo(xx, pad.top + gh); ctx.stroke();
    const val = tMin + (tMax - tMin) * k / 5;
    ctx.fillStyle = '#6c757d';
    ctx.fillText(val.toFixed(0), xx - 10, pad.top + gh + 20);
  }

  // current curve
  ctx.strokeStyle = '#0d6efd'; ctx.lineWidth = 1.5; ctx.beginPath();
  for (let k = 0; k < t.length; k++) {
    if (k === 0) ctx.moveTo(x(t[k]), y(i[k])); else ctx.lineTo(x(t[k]), y(i[k]));
  }
  ctx.stroke();

  // baseline
  ctx.strokeStyle = '#198754'; ctx.setLineDash([5, 5]); ctx.beginPath();
  ctx.moveTo(x(tMin), y(run.baseline)); ctx.lineTo(x(tMax), y(run.baseline)); ctx.stroke();
  ctx.setLineDash([]);

  // boundaries
  const b = state.boundaries[run.run];
  const leftX = x(b.left), rightX = x(b.right);
  [leftX, rightX].forEach((xx, idx) => {
    ctx.strokeStyle = '#dc3545'; ctx.lineWidth = 2; ctx.beginPath();
    ctx.moveTo(xx, pad.top); ctx.lineTo(xx, pad.top + gh); ctx.stroke();
    ctx.fillStyle = '#dc3545'; ctx.font = '12px sans-serif';
    ctx.fillText(idx === 0 ? 'L' : 'R', xx + 3, pad.top + 12);
  });

  // labels
  ctx.fillStyle = '#212529'; ctx.font = '12px sans-serif';
  ctx.fillText('Time (s)', pad.left + gw / 2 - 20, h - 5);
  ctx.save(); ctx.translate(15, h / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText('Current (A)', -30, 0); ctx.restore();

  // interaction
  let dragging = null;
  const threshold = 8;
  function boundaryFromX(mx) {
    const leftDist = Math.abs(mx - leftX);
    const rightDist = Math.abs(mx - rightX);
    if (leftDist < rightDist && leftDist < threshold) return 'left';
    if (rightDist <= leftDist && rightDist < threshold) return 'right';
    return null;
  }
  canvas.onmousedown = e => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    dragging = boundaryFromX(mx);
    if (dragging) canvas.classList.add('dragging');
  };
  canvas.onmousemove = e => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    if (dragging) {
      let tv = tMin + (mx - pad.left) / gw * (tMax - tMin);
      tv = Math.max(tMin, Math.min(tMax, tv));
      state.boundaries[run.run][dragging] = tv;
      drawRun(run);
      updateCoverage();
    } else {
      canvas.style.cursor = boundaryFromX(mx) ? 'ew-resize' : 'col-resize';
    }
  };
  window.addEventListener('mouseup', () => { dragging = null; canvas.classList.remove('dragging'); });
}

async function updateCoverage() {
  const payload = { boundaries: state.boundaries };
  const res = await fetch('/compute', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  const data = await res.json();
  document.getElementById('coverage').textContent = data.coverage.toFixed(2);
  state.runs.forEach(run => {
    const r = data.run_results[run.run];
    document.getElementById(`qd-${run.run}`).textContent = r.charge.toFixed(6);
    document.getElementById(`left-${run.run}`).textContent = r.left.toFixed(2);
    document.getElementById(`right-${run.run}`).textContent = r.right.toFixed(2);
  });
}

async function finalize(endpoint) {
  document.getElementById('status').textContent = '正在计算并保存结果...';
  const res = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ boundaries: state.boundaries }) });
  const data = await res.json();
  if (data.ok) {
    document.getElementById('status').textContent = `已保存：SO₃ 覆盖度 = ${data.coverage.toFixed(2)}%，结果写入 ${data.output}`;
    document.getElementById('submitBtn').disabled = true;
    document.getElementById('skipBtn').disabled = true;
    await fetch('/shutdown', { method: 'POST' });
  } else {
    document.getElementById('status').textContent = '保存失败：' + (data.error || '未知错误');
  }
}

document.getElementById('submitBtn').onclick = () => finalize('/save');
document.getElementById('skipBtn').onclick = () => finalize('/skip');

init();
</script>
</body>
</html>
"""


class _InteractiveHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the interactive selection page."""

    def log_message(self, format, *args):
        # suppress default request logging
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html, status=200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML_PAGE)
        elif parsed.path == "/data":
            self._handle_data()
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "Invalid JSON"}, 400)
            return

        if parsed.path == "/compute":
            self._handle_compute(payload)
        elif parsed.path == "/save":
            self._handle_save(payload, use_selected=True)
        elif parsed.path == "/skip":
            self._handle_save(payload, use_selected=False)
        elif parsed.path == "/shutdown":
            self._send_json({"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self.send_error(404)

    def _load_run_data(self):
        """Load runs 2 and 3 with default boundaries and baseline."""
        runs = []
        defaults = {}
        files = self.server.case_files["co_displace"]
        for run_idx in [1, 2]:
            csv_path = files[run_idx]
            time, current = read_co_displace_csv(csv_path)
            default = integrate_co_displace_peak(time, current)
            defaults[run_idx + 1] = {
                "left": default["t_left"],
                "right": default["t_right"],
            }
            # Downsample for browser performance if very long
            n = len(time)
            step = max(1, n // 2000)
            runs.append({
                "run": run_idx + 1,
                "file": Path(csv_path).name,
                "time": time[::step].tolist(),
                "current": current[::step].tolist(),
                "baseline": default["baseline"],
                "t_min": default["t_min"],
            })
        return runs, defaults

    def _handle_data(self):
        runs, defaults = self._load_run_data()
        self._send_json({"runs": runs, "defaults": defaults})

    def _boundaries_to_kwargs(self, payload, use_selected):
        boundaries = payload.get("boundaries", {})
        kwargs = {}
        for run_idx in [2, 3]:
            if use_selected and str(run_idx) in boundaries:
                b = boundaries[str(run_idx)]
                left = float(b["left"])
                right = float(b["right"])
                t_min = self.server.peak_info[run_idx]["t_min"]
                kwargs[run_idx] = {
                    "peak_pre": t_min - left,
                    "peak_post": right - t_min,
                }
        return kwargs

    def _compute_coverage(self, payload, use_selected):
        kwargs = self._boundaries_to_kwargs(payload, use_selected)
        result = self.server.compute_with_boundaries(kwargs)
        run_results = {}
        for run_idx in [2, 3]:
            pre = kwargs.get(run_idx, {}).get("peak_pre", 13.0)
            post = kwargs.get(run_idx, {}).get("peak_post", 6.0)
            t_min = self.server.peak_info[run_idx]["t_min"]
            run_results[run_idx] = {
                "charge": result["q_co_displace"][f"run_{run_idx}"],
                "left": t_min - pre,
                "right": t_min + post,
            }
        return {
            "coverage": result["so3_coverage_percent"],
            "run_results": run_results,
        }

    def _handle_compute(self, payload):
        self._send_json(self._compute_coverage(payload, use_selected=True))

    def _handle_save(self, payload, use_selected):
        try:
            result = self.server.compute_with_boundaries(
                self._boundaries_to_kwargs(payload, use_selected)
            )
            output_path = self.server.output_path
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, ensure_ascii=False)
            self._send_json({
                "ok": True,
                "coverage": result["so3_coverage_percent"],
                "output": str(output_path),
            })
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, 500)


class _InteractiveServer(HTTPServer):
    """Simple server that holds shared computation state."""

    def __init__(self, server_address, case_dir, output_path):
        super().__init__(server_address, _InteractiveHandler)
        self.case_dir = Path(case_dir)
        self.output_path = Path(output_path)
        self.case_files = find_case_files(self.case_dir)
        self.peak_info = {}
        for run_idx in [1, 2, 3]:
            time, current = read_co_displace_csv(self.case_files["co_displace"][run_idx - 1])
            self.peak_info[run_idx] = integrate_co_displace_peak(time, current)
        self._compute_func = None

    def set_compute_func(self, func):
        self._compute_func = func

    def compute_with_boundaries(self, boundaries_kwargs):
        return self._compute_func(self.case_dir, boundaries_kwargs)


def launch_interactive(case_dir, output_path, port=0, open_browser=True):
    """Launch the interactive boundary-selection UI.

    Returns a dict with the final coverage result, or None if the server was
    shut down before a result was saved.
    """
    output_path = Path(output_path)
    server = _InteractiveServer(("127.0.0.1", port), case_dir, output_path)
    server.set_compute_func(_compute_with_boundaries)

    if port == 0:
        port = server.server_address[1]

    url = f"http://127.0.0.1:{port}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    if open_browser:
        webbrowser.open(url)

    print(f"Interactive sulfonate coverage UI running at {url}")
    print("Adjust peak boundaries in your browser, then click Submit or Skip.")

    # Wait until server is shut down by user action
    thread.join()

    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _compute_with_boundaries(case_dir, boundaries_kwargs):
    """Helper used by the interactive server to recompute coverage."""
    from meatools.sulfonate_coverage import process_case

    co_displace_kwargs = {run_idx: {} for run_idx in [1, 2, 3]}
    for run_idx, kw in boundaries_kwargs.items():
        co_displace_kwargs[run_idx] = kw

    return process_case(case_dir, co_displace_kwargs=co_displace_kwargs)
