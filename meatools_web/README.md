# MEATOOLs Web

A local, stdlib-only web front-end for the MEATOOLs CLI. Start it, open
`http://127.0.0.1:8710`, upload a complete MEA case folder, click a
function, and browse the results — all in the browser. No third-party
Python dependencies, no public exposure (binds to `127.0.0.1`).

## Start

```bash
mea web                     # or: python -m meatools_web.server
mea web --port 9000         # different port
mea web --root /other/path  # different task root (env: MEATOOLS_WEB_ROOT)
```

Open **http://127.0.0.1:8710**.

## What it does

1. **Upload** — drag & drop a complete case folder (or pick a folder /
   a `.zip`) in the *New task* panel. Files are streamed to a task
   directory under the task root:

   ```
   /home/hrl/work/mea/mea_web/20260722-153000-ab12cd34/   ← <timestamp>_<hash>
   ```

   The 8-hex suffix is the SHA-256 of the uploaded file manifest
   (path + size + per-file content hash). Re-uploading an identical
   folder reuses the existing task instead of duplicating files.

2. **Run** — the function grid calls the same subcommands as the CLI,
   each as a background process with `cwd =` the task directory
   (`MPLBACKEND=Agg`):

   | button     | runs                      | notes                                  |
   |------------|---------------------------|----------------------------------------|
   | `all`      | `mea all`                 | full pipeline → results.json + results.html; "skip sulf-cvrg" checkbox = `--no-sulf` |
   | `ttseq`    | `mea ttseq`               | sequence timeline, voltage/temp plots, polarization |
   | `otr`      | `mea otr`                 | OTR / impedance fitting                |
   | `ecsa`     | `mea ecssa`               | wet ECSA from CV                       |
   | `ecsadry`  | `mea ecsadry`             | dry ECSA                               |
   | `lsv`      | `mea lsv`                 | LSV curves                             |
   | `eis`      | `mea eis`                 | EIS                                    |
   | `sulf-cvrg`| `mea sulf-cvrg`           | interactive boundary picker embedded in the page (Submit / Skip = default boundaries) |
   | `conclude` | `mea conclude`            | merge `results/*` → results.json       |
   | `render`   | `mea render`              | results.json → results.html            |

   One run at a time per task; a **stop** button kills the whole
   process group. Live log tail (last 64 KB) updates every 2.5 s;
   full logs are in `logs/web_run_<n>_<cmd>.log` and viewable from the
   runs table.

3. **Results** —
   * single-function runs: generated PNGs open in a preview modal,
     CSV/JSON/log open as text, other types download;
   * `all` / `render`: the rendered report (`results.html`, produced by
     `mea render` from `results.json`) is embedded directly in an
     iframe and can be opened in a new tab;
   * **⬇ results.zip** bundles `results/`, `results.json`,
     `results.html` and `logs/` for download.

4. **History** — the sidebar lists every task (name, id, creation time,
   file count, last-run status dot) and every run per task. The index
   is persisted at `<root>/.mea_web_index.json`, and any task directory
   found on disk is adopted on startup, so history survives restarts.
   🗑 deletes a task (kills any running process first).

5. **Issues** — the sidebar *Issues* panel is a small notebook for
   user-reported bugs/observations. Each entry is stored as one
   Markdown file `<title>-<YYYYMMDD-HHMMSS>.md` under
   `<root>/issues/` (unsafe file-name characters in the title are
   stripped). Entries can be viewed (click) and deleted (hover trash).
   When a sulf-cvrg run is waiting for input, the interactive
   boundary-selection page is embedded in the run panel; the server
   publishes its URL at `logs/sulf_ui.json` inside the task dir.

## API (used by the front page, also handy with curl)

```
GET  /api/health
GET  /api/commands
GET  /api/jobs                                  # history + last-run status
POST /api/jobs            {"name","hash","files","size"}  → {"job","reused"}
POST /api/jobs/<id>/file?path=<rel>             # raw body = file bytes
POST /api/jobs/<id>/zip                         # raw body = zip bytes
POST /api/jobs/<id>/run    {"command","args":[]}
POST /api/jobs/<id>/stop
DELETE /api/jobs/<id>
GET  /api/jobs/<id>                             # job + runs (status)
GET  /api/jobs/<id>/files                       # file tree
GET  /api/jobs/<id>/file?path=<rel>             # file bytes
GET  /api/jobs/<id>/log?run=<n>                 # log tail (64 KB) + status
GET  /api/jobs/<id>/zip                         # results zip download
GET  /api/jobs/<id>/sulf-ui                     # {"active","url"} of the live boundary picker
GET  /api/issues                                # issue list (title/preview/time)
POST /api/issues          {"title","content"}  # writes issues/<title>-<ts>.md
GET  /api/issues/<name>                         # full markdown text
DELETE /api/issues/<name>
```

## Notes

* Runs execute `python -m meatools <cmd>` with the **same interpreter**
  that started the server, so the server must be started from the
  environment where MEATOOLs is installed (the one providing `mea`).
* Task directories are plain folders — you can still `cd` into them and
  use the CLI as usual.
* Uploads are not size-capped by the server; the browser streams files
  (folder uploads run with 4 parallel file streams).