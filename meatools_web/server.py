#!/usr/bin/env python
"""Local web front-end for the MEATOOLs CLI.

Runs a stdlib-only HTTP server on localhost. Users upload a complete MEA
case folder (or a zip of it) through the browser; the files are saved under
a task directory named ``YYYYMMDD-HHMMSS-<hash8>`` inside the task root
(default: /home/hrl/work/mea/mea_web). MEA subcommands are then executed as
background subprocesses with cwd = task directory, and the generated
results (PNG / CSV / JSON, results.json, results.html) are browsed in the
front page. Upload & run history is persisted across restarts.

Usage:
    python -m meatools_web.server [--port 8710] [--host 127.0.0.1]
                                  [--root /home/hrl/work/mea/mea_web]
"""

import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

DEFAULT_ROOT = "/home/hrl/work/mea/mea_web"
DEFAULT_PORT = 8710

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
INDEX_FILE = ".mea_web_index.json"

JOB_DIR_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{8}$")

# command -> (label, description, extra-args-caller)
COMMANDS = {
    "all": ("All (full pipeline)",
            "ttseq → otr → ecsa → ecsadry → lsv → eis → sulf-cvrg → "
            "conclude → render. Produces results.json + results.html."),
    "ttseq": ("Test sequence", "Sequence timeline, voltage/temperature "
                               "plots, polarization curves."),
    "otr": ("OTR / impedance", "Oxygen transfer resistance fitting."),
    "ecsa": ("ECSA (wet)", "Electrochemically active surface area from CV."),
    "ecsadry": ("ECSA (dry)", "ECSA on dry MEA."),
    "lsv": ("LSV", "Linear sweep voltammetry curves."),
    "eis": ("EIS", "Electrochemical impedance spectroscopy."),
    "sulf-cvrg": ("Sulfonate coverage", "Sulfonate group coverage, "
                                  "non-interactive peak selection."),
    "conclude": ("Conclude", "Merge results/* into results.json."),
    "render": ("Render", "Render results.json → results.html."),
}

CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".csv": "text/csv; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".log": "text/plain; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".zip": "application/zip",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _now():
    return time.time()


def _human_size(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


class JobStore:
    """In-memory job registry backed by ROOT/.mea_web_index.json."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._lock = threading.RLock()
        self._procs = {}  # (job_id, run_id) -> Popen
        self._index_path = os.path.join(self.root, INDEX_FILE)
        self.jobs = {}
        self._load_index()
        self._adopt_orphans()
        self._rescan_disk()

    def _adopt_orphans(self):
        """Runs whose process died without a reaper (server restart) are
        marked interrupted instead of 'running' forever."""
        changed = False
        for job in self.jobs.values():
            for r in job["runs"]:
                if r.get("rc") is None:
                    r["rc"] = 130
                    r["interrupted"] = True
                    changed = True
        if changed:
            self._persist()

    # ---------- persistence ----------

    def _load_index(self):
        if os.path.isfile(self._index_path):
            try:
                with open(self._index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.jobs = data.get("jobs", {})
                return
            except Exception:
                print(f"[mea-web] index unreadable, starting fresh: "
                      f"{traceback.format_exc()}", file=sys.stderr)
        self.jobs = {}

    def _persist(self):
        tmp = self._index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"jobs": self.jobs}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self._index_path)

    def _rescan_disk(self):
        """Adopt task dirs on disk that are missing from the index."""
        added = 0
        for name in os.listdir(self.root):
            if not JOB_DIR_RE.match(name):
                continue
            if name in self.jobs:
                continue
            count, size = self._walk_size(name)
            self.jobs[name] = {
                "id": name,
                "name": name,
                "created": os.path.getmtime(os.path.join(self.root, name)),
                "hash": name.rsplit("-", 1)[-1],
                "files": count,
                "size": size,
                "runs": [],
            }
            added += 1
        if added:
            self._persist()
            print(f"[mea-web] adopted {added} existing task dir(s) from disk")

    # ---------- helpers ----------

    def job_dir(self, job_id):
        return os.path.join(self.root, job_id)

    def _walk_size(self, job_name):
        base = self.job_dir(job_name)
        count = 0
        size = 0
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                if f.startswith("."):
                    continue
                count += 1
                try:
                    size += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return count, size

    def get(self, job_id):
        with self._lock:
            job = self.jobs.get(job_id)
            return json.loads(json.dumps(job)) if job else None

    def list_jobs(self):
        with self._lock:
            out = []
            for job in self.jobs.values():
                item = dict(job)
                item["running"] = self._running_run(job["id"]) is not None
                item["last_run"] = None
                if job["runs"]:
                    last = dict(job["runs"][-1])
                    last["status"] = self._run_status(job["id"], last)
                    item["last_run"] = last
                out.append(item)
            out.sort(key=lambda j: j["created"], reverse=True)
            return out

    def _running_run(self, job_id):
        for (jid, rid), info in self._procs.items():
            if jid == job_id and info["proc"].poll() is None:
                return rid
        return None

    def _run_status(self, job_id, run):
        if (job_id, run["id"]) in self._procs:
            return "running"
        if run.get("rc") is None:
            return "interrupted"  # process lost (server restarted mid-run)
        return "failed" if run["rc"] != 0 else "done"

    def public_run(self, job_id, run):
        out = dict(run)
        out["status"] = self._run_status(job_id, run)
        return out

    # ---------- job lifecycle ----------

    def create_job(self, name, hash_full, file_count, total_size):
        short = re.sub(r"[^0-9a-f]", "", (hash_full or ""))[:8]
        if len(short) < 8:
            short = (short + "0" * 8)[:8]
        with self._lock:
            for job in self.jobs.values():
                if job.get("hash") == short:
                    return job["id"], True
            ts = time.strftime("%Y%m%d-%H%M%S")
            job_id = f"{ts}-{short}"
            while os.path.exists(self.job_dir(job_id)):
                time.sleep(1.05)
                ts = time.strftime("%Y%m%d-%H%M%S")
                job_id = f"{ts}-{short}"
            self.jobs[job_id] = {
                "id": job_id,
                "name": (name or job_id)[:200],
                "created": _now(),
                "hash": short,
                "files": file_count,
                "size": total_size,
                "runs": [],
            }
            os.makedirs(self.job_dir(job_id), exist_ok=True)
            self._persist()
            return job_id, False

    def delete_job(self, job_id):
        with self._lock:
            job = self.jobs.pop(job_id, None)
            if job is None:
                return False
            for (jid, rid), info in list(self._procs.items()):
                if jid == job_id:
                    try:
                        os.killpg(os.getpgid(info["proc"].pid),
                                  signal.SIGTERM)
                    except Exception:
                        try:
                            info["proc"].kill()
                        except Exception:
                            pass
                    self._procs.pop((jid, rid), None)
            self._persist()
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        return True

    # ---------- file uploads ----------

    def safe_path(self, job_id, rel):
        """Resolve rel inside the job dir; return None if unsafe."""
        if not rel or rel.startswith("/") or "\x00" in rel:
            return None
        norm = os.path.normpath(rel)
        if norm.startswith("..") or os.path.isabs(norm):
            return None
        base = os.path.realpath(self.job_dir(job_id))
        target = os.path.realpath(os.path.join(base, norm))
        if target != base and not target.startswith(base + os.sep):
            return None
        return target

    def open_upload(self, job_id, rel):
        """Resolve + create the destination for a raw-body upload."""
        target = self.safe_path(job_id, rel)
        if target is None:
            return None
        os.makedirs(os.path.dirname(target), exist_ok=True)
        return target

    def extract_zip(self, job_id, data):
        """Extract zip bytes into the job dir; returns (nfiles, total_size)."""
        n = 0
        total = 0
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                rel = info.filename
                # zip slip guard
                norm = os.path.normpath(rel.replace("\\", "/"))
                if norm.startswith("..") or os.path.isabs(norm):
                    continue
                target = self.safe_path(job_id, norm)
                if target is None:
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                n += 1
                total += info.file_size
        return n, total

    # ---------- runs ----------

    def start_run(self, job_id, command, extra_args=None):
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if command not in COMMANDS:
                raise ValueError(f"unknown command: {command}")
            if self._running_run(job_id) is not None:
                raise RuntimeError("a run is already in progress for this task")
            run_id = (max((r["id"] for r in job["runs"]), default=0) + 1)
            args = list(extra_args or [])
            if command == "sulf-cvrg" and "--non-interactive" not in args:
                args.append("--non-interactive")
            log_rel = f"logs/web_run_{run_id}_{command}.log"
            job_dir = self.job_dir(job_id)
            os.makedirs(os.path.join(job_dir, "logs"), exist_ok=True)
            argv = [sys.executable, "-m", "meatools", command] + args
            env = dict(os.environ)
            env.setdefault("MPLBACKEND", "Agg")
            logf = open(os.path.join(job_dir, log_rel), "wb")
            logf.write(
                f"# mea {command} {' '.join(args)}\n"
                f"# {sys.executable} -m meatools {command} {' '.join(args)}\n"
                f"# cwd={job_dir} started={time.strftime('%F %T')}\n"
                .encode("utf-8"))
            logf.flush()
            proc = subprocess.Popen(
                argv, cwd=job_dir, env=env,
                stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True)
            run = {
                "id": run_id,
                "command": command,
                "args": args,
                "argv": argv,
                "started": _now(),
                "finished": None,
                "rc": None,
                "log": log_rel,
            }
            job["runs"].append(run)
            self._procs[(job_id, run_id)] = {"proc": proc, "logf": logf}
            self._persist()

        def reap(jid=job_id, rid=run_id, p=proc, lf=logf):
            rc = p.wait()
            lf.close()
            with self._lock:
                self._procs.pop((jid, rid), None)
                j = self.jobs.get(jid)
                if j:
                    for r in j["runs"]:
                        if r["id"] == rid:
                            r["rc"] = rc
                            r["finished"] = _now()
                self._persist()

        threading.Thread(target=reap, daemon=True).start()
        return self.public_run(job_id, run)

    def stop_run(self, job_id, run_id=None):
        with self._lock:
            for (jid, rid), info in list(self._procs.items()):
                if jid == job_id and (run_id is None or rid == run_id):
                    try:
                        os.killpg(os.getpgid(info["proc"].pid),
                                  signal.SIGTERM)
                    except Exception:
                        try:
                            info["proc"].terminate()
                        except Exception:
                            pass
                    return True
        return False

    # ---------- disk views ----------

    def file_tree(self, job_id, limit=8000):
        base = self.job_dir(job_id)
        if not os.path.isdir(base):
            return []
        out = []
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs
                       if not d.startswith(".") and d != "__pycache__"]
            rel_root = os.path.relpath(root, base)
            for f in files:
                if f.startswith(".") or f == "__pycache__":
                    continue
                path = f if rel_root == "." else os.path.join(rel_root, f)
                full = os.path.join(root, f)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                out.append({
                    "path": path,
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                    "size_h": _human_size(st.st_size),
                })
                if len(out) >= limit:
                    return out
        out.sort(key=lambda x: x["path"].lower())
        return out

    def result_zip(self, job_id):
        """Build a zip of results/, results.json/html, logs/. Returns path."""
        base = self.job_dir(job_id)
        tmp = tempfile.NamedTemporaryFile(
            dir=self.root, prefix=".dl-", suffix=".zip", delete=False)
        tmp.close()
        included = 0
        with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in ("results.json", "results.html"):
                full = os.path.join(base, name)
                if os.path.isfile(full):
                    zf.write(full, name)
                    included += 1
            for top in ("results", "logs"):
                full = os.path.join(base, top)
                if not os.path.isdir(full):
                    continue
                for root, dirs, files in os.walk(full):
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                    for f in files:
                        p = os.path.join(root, f)
                        zf.write(p, os.path.relpath(p, base))
                        included += 1
        return tmp.name, included


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "MeaWeb/0.1"
    store: JobStore = None  # set in main()

    # -- plumbing -----------------------------------------------------------

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code, body=b"", ctype="application/octet-stream",
              headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _err(self, code, msg):
        self._json(code, {"error": msg})

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _json_body(self):
        raw = self._body()
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _stream_body(self, length, on_chunk):
        written = 0
        while written < length:
            chunk = self.rfile.read(min(1024 * 1024, length - written))
            if not chunk:
                break
            written += len(chunk)
            on_chunk(chunk)
        return written

    def _serve_static(self, rel):
        path = os.path.realpath(os.path.join(WEB_DIR, rel))
        if not path.startswith(os.path.realpath(WEB_DIR) + os.sep) \
                or not os.path.isfile(path):
            self._err(404, "not found")
            return
        ctype = CONTENT_TYPES.get(os.path.splitext(path)[1].lower(),
                                  "application/octet-stream")
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    # -- routes -------------------------------------------------------------

    def do_GET(self):
        try:
            self._route_get()
        except BrokenPipeError:
            pass
        except Exception:
            self._err(500, traceback.format_exc(limit=5))

    def do_POST(self):
        try:
            self._route_post()
        except BrokenPipeError:
            pass
        except Exception:
            self._err(500, traceback.format_exc(limit=5))

    def do_DELETE(self):
        try:
            u = urlparse(self.path)
            m = re.match(r"^/api/jobs/([^/]+)$", u.path)
            if not m:
                self._err(404, "not found")
                return
            ok = self.store.delete_job(m.group(1))
            if ok:
                self._json(200, {"deleted": m.group(1)})
            else:
                self._err(404, "unknown job")
        except Exception:
            self._err(500, traceback.format_exc(limit=5))

    def _route_get(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path

        if p in ("/", "/index.html"):
            return self._serve_static("index.html")
        if p.startswith("/static/"):
            return self._serve_static(p[len("/static/"):])

        if p == "/api/health":
            return self._json(200, {
                "ok": True, "root": self.store.root,
                "jobs": len(self.store.jobs), "version": "0.1.0"})

        if p == "/api/commands":
            return self._json(200, {
                "commands": [
                    {"id": k, "label": v[0], "description": v[1]}
                    for k, v in COMMANDS.items()
                ]})

        if p == "/api/jobs":
            return self._json(200, {"jobs": self.store.list_jobs()})

        m = re.match(r"^/api/jobs/([^/]+)$", p)
        if m:
            job = self.store.get(m.group(1))
            if not job:
                return self._err(404, "unknown job")
            job["running"] = self.store._running_run(job["id"]) is not None
            for r in job["runs"]:
                r["status"] = self.store._run_status(job["id"], r)
            return self._json(200, {"job": job})

        m = re.match(r"^/api/jobs/([^/]+)/files$", p)
        if m:
            if not self.store.get(m.group(1)):
                return self._err(404, "unknown job")
            return self._json(200, {
                "files": self.store.file_tree(m.group(1))})

        m = re.match(r"^/api/jobs/([^/]+)/file$", p)
        if m:
            job_id = m.group(1)
            if not self.store.get(job_id):
                return self._err(404, "unknown job")
            rel = (q.get("path") or [""])[0]
            full = self.store.safe_path(job_id, rel)
            if not full or not os.path.isfile(full):
                return self._err(404, "file not found")
            ext = os.path.splitext(full)[1].lower()
            ctype = CONTENT_TYPES.get(ext, "application/octet-stream")
            disposition = None
            if ext not in (".png", ".jpg", ".jpeg", ".gif", ".svg",
                           ".html", ".htm", ".json", ".csv", ".log",
                           ".txt", ".zip"):
                disposition = 'attachment'
            headers = {}
            if disposition:
                headers["Content-Disposition"] = (
                    f'attachment; filename="{os.path.basename(full)}"')
            with open(full, "rb") as f:
                self._send(200, f.read(), ctype, headers)
            return

        m = re.match(r"^/api/jobs/([^/]+)/log$", p)
        if m:
            job_id = m.group(1)
            job = self.store.get(job_id)
            if not job:
                return self._err(404, "unknown job")
            run = None
            if "run" in q:
                try:
                    rid = int(q["run"][0])
                except ValueError:
                    return self._err(400, "bad run id")
                run = next((r for r in job["runs"] if r["id"] == rid), None)
            else:
                run = job["runs"][-1] if job["runs"] else None
            if not run:
                return self._json(200, {"text": "", "size": 0,
                                        "exists": False})
            full = os.path.join(self.store.job_dir(job_id), run["log"])
            if not os.path.isfile(full):
                return self._json(200, {"text": "", "size": 0,
                                        "exists": False})
            size = os.path.getsize(full)
            with open(full, "rb") as f:
                if size > 65536:
                    f.seek(size - 65536)
                raw = f.read()
            return self._json(200, {
                "text": raw.decode("utf-8", "replace"),
                "size": size, "truncated": size > 65536,
                "exists": True,
                "status": self.store._run_status(job_id, run)})

        m = re.match(r"^/api/jobs/([^/]+)/zip$", p)
        if m:
            job_id = m.group(1)
            if not self.store.get(job_id):
                return self._err(404, "unknown job")
            path, n = self.store.result_zip(job_id)
            try:
                with open(path, "rb") as f:
                    data = f.read()
                self._send(200, data, "application/zip", {
                    "Content-Disposition":
                        f'attachment; filename="{job_id}-results.zip"'})
            finally:
                os.unlink(path)
            return

        self._err(404, "not found")

    def _route_post(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path

        if p == "/api/jobs":
            body = self._json_body()
            name = (body.get("name") or "").strip()
            h = (body.get("hash") or "").lower()
            n_files = int(body.get("files") or 0)
            size = int(body.get("size") or 0)
            job_id, reused = self.store.create_job(name, h, n_files, size)
            return self._json(200, {"job": self.store.get(job_id),
                                    "reused": reused})

        m = re.match(r"^/api/jobs/([^/]+)/file$", p)
        if m:
            job_id = m.group(1)
            if not self.store.get(job_id):
                return self._err(404, "unknown job")
            rel = (q.get("path") or [""])[0]
            target = self.store.open_upload(job_id, rel)
            if target is None:
                return self._err(400, "unsafe path")
            length = int(self.headers.get("Content-Length") or 0)
            with open(target, "wb") as f:
                self._stream_body(length, lambda c: f.write(c))
            return self._json(200, {
                "saved": os.path.relpath(target, self.store.job_dir(job_id))})

        m = re.match(r"^/api/jobs/([^/]+)/zip$", p)
        if m:
            job_id = m.group(1)
            if not self.store.get(job_id):
                return self._err(404, "unknown job")
            length = int(self.headers.get("Content-Length") or 0)
            buf = io.BytesIO()
            self._stream_body(length, buf.write)
            n, total = self.store.extract_zip(job_id, buf.getvalue())
            if n == 0:
                return self._err(400, "no files extracted from zip")
            return self._json(200, {"extracted": n, "size": total})

        m = re.match(r"^/api/jobs/([^/]+)/run$", p)
        if m:
            job_id = m.group(1)
            body = self._json_body()
            command = (body.get("command") or "").strip()
            extra = body.get("args") or []
            if not isinstance(extra, list):
                return self._err(400, "args must be a list")
            extra = [str(a) for a in extra]
            try:
                run = self.store.start_run(job_id, command, extra)
            except KeyError:
                return self._err(404, "unknown job")
            except ValueError as e:
                return self._err(400, str(e))
            except RuntimeError as e:
                return self._err(409, str(e))
            return self._json(200, {"run": self.store.public_run(job_id, run)})

        m = re.match(r"^/api/jobs/([^/]+)/stop$", p)
        if m:
            ok = self.store.stop_run(m.group(1))
            if ok:
                return self._json(200, {"stopped": True})
            return self._err(404, "no running process for this job")

        self._err(404, "not found")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="mea-web",
                                 description="Local web front-end for MEATOOLs")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--root", default=os.environ.get(
        "MEATOOLS_WEB_ROOT", DEFAULT_ROOT),
        help="task root directory (default: %(default)s)")
    args = ap.parse_args(argv)

    store = JobStore(args.root)
    Handler.store = store
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[mea-web] MEATOOLs web front-end")
    print(f"[mea-web]   url:  http://{args.host}:{args.port}")
    print(f"[mea-web]   root: {args.root}  ({len(store.jobs)} task(s) in history)")
    print(f"[mea-web]   python: {sys.executable}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[mea-web] shutting down")
        httpd.server_close()


if __name__ == "__main__":
    main()