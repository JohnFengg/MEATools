/* MEATOOLs web — front-end logic (vanilla JS, no dependencies)
 *
 * Layout contract (index.html):
 *   sidebar  → #dropzone #folderInput #zipInput #stage(#stageName #stageStats
 *              #stageList #btnCreate #btnCancel #progressWrap #stageProgress
 *              #progressText)  #jobList #jobCount #noJobs
 *   detail   → #emptyDetail(#rootPath)  #jobDetail
 *   overlays → #modal(#modalTitle #modalBody)  #toast  #topMeta
 */
"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  commands: [],          // [{id,label,description}]
  jobs: [],              // list from /api/jobs
  selectedId: null,
  job: null,             // full job object for the selection
  files: [],             // file tree of the selection
  staging: null,         // {mode:'folder'|'zip', name, files:[{file,rel}], size}
  uploading: false,
  tick: 0,
  lastRunSig: null,      // signature of last observed run state
  fileTab: "results",    // results | logs | data
  noSulf: false,         // remembered "skip sulf-cvrg" preference
  logOpen: null,         // remembered open-state of the live-log <details>
  sulfUiUrl: null,       // live URL of the interactive sulf-cvrg page
  sulfHidden: false,     // user collapsed the embedded sulf-cvrg selector
  issues: [],            // list from /api/issues
};

/* ---------------- inline SVG icons (stroke style) ---------------- */

const ICONS = {
  play: '<path d="M7 4.5v15l13-7.5-13-7.5z"/>',
  refresh: '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>'
    + '<path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>',
  download: '<path d="M12 3v12m0 0 4-4m-4 4-4-4"/><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/>',
  trash: '<path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6h14zM10 11v6M14 11v6"/>',
  external: '<path d="M14 4h6v6M20 4l-9 9"/><path d="M19 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h6"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="1.5"/>',
  check: '<path d="M4 12.5l5 5L20 6.5"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3.5 2.5"/>',
  zap: '<path d="M13 2 4 14h6l-1 8 9-12h-6l1-8z"/>',
  file: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-6-6z"/><path d="M14 3v6h6"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="1.6"/><path d="m21 16-4.5-4.5L7 21"/>',
  table: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M3 15h18M10 4v16M16 4v16"/>',
  braces: '<path d="M8 3H7a2 2 0 0 0-2 2v4a2 2 0 0 1-2 2 2 2 0 0 1 2 2v4a2 2 0 0 0 2 2h1M16 3h1a2 2 0 0 1 2 2v4a2 2 0 0 0 2 2 2 2 0 0 0-2 2v4a2 2 0 0 1-2 2h-1"/>',
  text: '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-6-6z"/><path d="M14 3v6h6M9 13h6M9 17h6"/>',
  terminal: '<polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/>',
  archive: '<rect x="3" y="4" width="18" height="5" rx="1"/><path d="M5 9v10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V9M10 13h4"/>',
  flask: '<path d="M10 3v6L4.5 18a2 2 0 0 0 1.8 3h11.4a2 2 0 0 0 1.8-3L14 9V3M8.5 3h7"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  note: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>',
};

function icon(name, size = 14) {
  const p = ICONS[name] || ICONS.file;
  return `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none"
    stroke="currentColor" stroke-width="1.9" stroke-linecap="round"
    stroke-linejoin="round" aria-hidden="true">${p}</svg>`;
}

/* ---------------- tiny API layer ---------------- */

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch { /* non-json */ }
  if (!res.ok) {
    throw new Error((data && data.error) || `${res.status} ${res.statusText}`);
  }
  return data;
}

function fileUrl(jobId, path) {
  return `/api/jobs/${encodeURIComponent(jobId)}/file?path=${encodeURIComponent(path)}`;
}

/* ---------------- formatting ---------------- */

function fmtTime(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
function fmtClock(ts) {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
function fmtDur(a, b) {
  if (!a || b == null) return "—";
  const s = Math.max(0, Math.round((b - a) * 10) / 10);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s - m * 60)}s`;
}
function esc(s) {
  return String(s).replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;",
              "'": "&#39;" }[c]));
}

const STATUS_ICON = { running: "clock", done: "check", failed: "x", interrupted: "zap" };

function statusPill(status, run) {
  const cls = status || "none";
  let label = { running: "Running…", done: "Done", failed: "Failed",
                interrupted: "Interrupted" }[cls] || cls;
  if (run && (status === "done" || status === "failed") && run.rc != null)
    label += ` · rc ${run.rc}`;
  return `<span class="pill ${cls}">${icon(STATUS_ICON[cls] || "file", 11)} ${esc(label)}</span>`;
}
function runPill(job) {
  const lr = job.last_run;
  if (!lr) return '<span class="pill">no runs yet</span>';
  const cls = job.running ? "running" : lr.status;
  const label = job.running ? `${lr.command}…` : lr.command;
  return `<span class="pill ${cls}">${icon(STATUS_ICON[cls] || "file", 11)} ${esc(label)}</span>`;
}
function icoFor(path) {
  const e = path.split(".").pop().toLowerCase();
  if (["png", "jpg", "jpeg", "gif", "svg"].includes(e)) return "image";
  if (["csv", "tsv"].includes(e)) return "table";
  if (e === "json") return "braces";
  if (["html", "htm"].includes(e)) return "text";
  if (["log", "txt"].includes(e)) return "terminal";
  if (e === "zip") return "archive";
  if (["docx", "doc"].includes(e)) return "text";
  if (e === "dta") return "flask";
  return "file";
}
function humanSize(n) {
  for (const u of ["B", "KB", "MB", "GB", "TB"]) {
    if (n < 1024 || u === "TB")
      return u === "B" ? `${n | 0} B` : `${(n / 1024).toFixed(1)} ${u}`;
    n /= 1024;
  }
}

/* ---------------- toast ---------------- */

let toastTimer = null;
function toast(msg, isError = false) {
  const el = $("toast");
  el.textContent = msg;
  el.className = `toast${isError ? " error" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 4200);
}

/* ---------------- modal ---------------- */

function openModal(title, node) {
  $("modalTitle").textContent = title;
  const body = $("modalBody");
  body.innerHTML = "";
  body.appendChild(node);
  $("modal").classList.remove("hidden");
}
function closeModal() { $("modal").classList.add("hidden"); }
document.querySelectorAll("#modal [data-close]")
  .forEach((el) => el.addEventListener("click", closeModal));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeModal();
});

async function previewFile(jobId, f) {
  const ext = f.path.split(".").pop().toLowerCase();
  const url = fileUrl(jobId, f.path);
  if (["png", "jpg", "jpeg", "gif", "svg"].includes(ext)) {
    const img = document.createElement("img");
    img.src = url;
    img.alt = f.path;
    img.onerror = () => toast("image failed to load", true);
    openModal(f.path, img);
    return;
  }
  if (["html", "htm"].includes(ext)) {
    window.open(url, "_blank");
    return;
  }
  if (["csv", "json", "log", "txt", "md", "tsv"].includes(ext)) {
    const pre = document.createElement("pre");
    pre.textContent = "loading…";
    openModal(f.path, pre);
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(res.statusText);
      let text = await res.text();
      if (text.length > 400000) {
        text = text.slice(0, 400000) +
          `\n… (truncated — ${text.length} chars total, download for full)`;
      }
      pre.textContent = text;
    } catch (e) {
      pre.textContent = `failed to load: ${e.message}`;
    }
    return;
  }
  window.location.href = url; // download
}

/* ================= staging / upload ================= */

function collectFolder(fileList) {
  let files = [];
  let size = 0;
  const tops = new Set();
  for (const file of fileList) {
    const rel = file.webkitRelativePath || file.name;
    if (!rel) continue;
    files.push({ file, rel });
    size += file.size;
    tops.add(rel.split("/")[0]);
  }
  if (!files.length) return null;
  const name = tops.size === 1 ? [...tops][0] : "upload";
  // The folder picker keeps the picked folder's name in every path
  // ("114-EOL/磺酸根覆盖度/…"); strip a single shared top-level folder so
  // the case contents land at the task root (same as drag-and-drop).
  if (tops.size === 1 && files.every((f) => f.rel.includes("/"))) {
    const prefix = name + "/";
    files = files.map((f) => ({ file: f.file, rel: f.rel.slice(prefix.length) }))
      .filter((f) => f.rel && !f.rel.split("/").pop().startsWith("._")
              && f.rel.split("/").pop() !== ".DS_Store");
  }
  return { mode: "folder", name, files, size };
}

function collectZip(file) {
  if (!file) return null;
  return { mode: "zip", name: file.name.replace(/\.zip$/i, ""), files: [], size: file.size };
}

function renderStage() {
  const st = state.staging;
  const stage = $("stage");
  if (!st) { stage.classList.add("hidden"); return; }
  stage.classList.remove("hidden");
  $("stageName").textContent = st.name;
  $("stageStats").textContent =
    st.mode === "zip"
      ? `1 zip file · ${humanSize(st.size)} — will be extracted into the new task`
      : `${st.files.length} files · ${humanSize(st.size)}`;
  const list = $("stageList");
  if (st.mode === "zip") {
    list.textContent = st.name + ".zip";
  } else {
    const shown = st.files.slice(0, 60).map((f) => f.rel).join("\n");
    const extra = st.files.length - 60;
    list.textContent = shown + (extra > 0 ? `\n… +${extra} more` : "");
  }
}

async function sha256Hex(buf) {
  const d = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function taskHash(st) {
  // Manifest hash: per-file sha256 (files ≤ 64 MB) + size for bigger ones.
  const lines = [];
  if (st.mode === "zip") {
    const f = st.fileObj;
    if (f.size <= 512 * 1024 * 1024) {
      lines.push(`zip\0${f.size}\0${await sha256Hex(await f.arrayBuffer())}`);
    } else {
      lines.push(`zip\0${f.size}\0-`);
    }
  } else {
    await Promise.all(st.files.map(async (e) => {
      let h = "-";
      if (e.file.size <= 64 * 1024 * 1024) {
        try { h = await sha256Hex(await e.file.arrayBuffer()); } catch { h = "-"; }
      }
      lines.push(`${e.rel}\0${e.file.size}\0${h}`);
    }));
    lines.sort();
  }
  return sha256Hex(new TextEncoder().encode(lines.join("\n")).buffer);
}

async function createAndUpload() {
  const st = state.staging;
  if (!st || state.uploading) return;
  state.uploading = true;
  setButtonsBusy(true);
  try {
    const hash = await taskHash(st);
    const nFiles = st.mode === "zip" ? 1 : st.files.length;
    const created = await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: st.name, hash, files: nFiles, size: st.size }),
    });
    const jobId = created.job.id;
    if (created.reused) {
      toast(`Identical upload already exists — opened existing task`);
      state.staging = null;
      renderStage();
      await refreshJobs();
      await selectJob(jobId);
      return;
    }
    await uploadFiles(jobId, st);
    state.staging = null;
    renderStage();
    toast(`Task created — you can now run the analysis`);
    await refreshJobs();
    await selectJob(jobId);
  } catch (e) {
    toast(`upload failed: ${e.message}`, true);
  } finally {
    state.uploading = false;
    setButtonsBusy(false);
  }
}

async function uploadFiles(jobId, st) {
  const wrap = $("progressWrap"), bar = $("stageProgress"), txt = $("progressText");
  wrap.classList.remove("hidden");
  if (st.mode === "zip") {
    txt.textContent = "uploading zip…";
    bar.value = 0;
    await api(`/api/jobs/${encodeURIComponent(jobId)}/zip`, {
      method: "POST", body: st.fileObj,
    });
    bar.value = bar.max;
    return;
  }
  const total = st.size || 1;
  let done = 0, ok = 0;
  const queue = [...st.files];
  const CONC = 4;
  const workers = Array.from({ length: CONC }, async () => {
    while (queue.length) {
      const e = queue.shift();
      if (!e) break;
      try {
        await api(
          `/api/jobs/${encodeURIComponent(jobId)}/file?path=${encodeURIComponent(e.rel)}`,
          { method: "POST", body: e.file });
        ok += 1;
      } catch (err) {
        toast(`file failed: ${e.rel} — ${err.message}`, true);
        throw err;
      }
      done += e.file.size;
      bar.value = Math.round((done / total) * bar.max);
      txt.textContent = `${ok}/${st.files.length} files · ${humanSize(done)} / ${humanSize(total)}`;
    }
  });
  await Promise.all(workers);
  txt.textContent = `done · ${ok}/${st.files.length} files`;
}

function setButtonsBusy(busy) {
  $("btnCreate").disabled = busy;
  $("btnFolder").disabled = busy;
  $("btnZip").disabled = busy;
  if (busy) {
    $("stageProgress").value = 0;
    $("progressWrap").classList.remove("hidden");
    $("progressText").textContent = "hashing files…";
  }
}

/* ================= jobs list / selection ================= */

async function refreshJobs() {
  const data = await api("/api/jobs");
  state.jobs = data.jobs;
  renderJobList();
  $("topMeta").textContent =
    `${state.jobs.length} task${state.jobs.length === 1 ? "" : "s"} · root ${state.rootPath || "…"}`;
}

function renderJobList() {
  const ul = $("jobList");
  ul.innerHTML = "";
  $("noJobs").classList.toggle("hidden", state.jobs.length > 0);
  const pill = $("jobCount");
  pill.classList.toggle("hidden", state.jobs.length === 0);
  pill.textContent = state.jobs.length || "";
  for (const j of state.jobs) {
    const li = document.createElement("li");
    if (j.id === state.selectedId) li.classList.add("selected");
    const lr = j.last_run;
    const dotCls = j.running ? "running"
      : lr ? (lr.status === "done" ? "done" : lr.status === "failed" ? "failed" : "interrupted") : "none";
    li.innerHTML = `
      <div class="tl-top">
        <span class="dot ${dotCls}"></span>
        <span class="tl-name" title="${esc(j.id)}">${esc(j.name)}</span>
        <button type="button" class="tl-del" title="Delete task" aria-label="Delete task">${icon("trash", 13)}</button>
      </div>
      <div class="tl-sub">
        <span>${fmtTime(j.created)}</span>
        <span>${j.files} files</span>
        <span>${humanSize(j.size)}</span>
      </div>
      <div class="tl-run">${runPill(j)}</div>`;
    li.addEventListener("click", (e) => {
      if (e.target.closest(".tl-del")) return;
      selectJob(j.id);
    });
    li.querySelector(".tl-del").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete task “${j.name}” and all its files?`)) return;
      try {
        await api(`/api/jobs/${encodeURIComponent(j.id)}`, { method: "DELETE" });
        if (state.selectedId === j.id) clearSelection();
        toast("task deleted");
        refreshJobs();
      } catch (err) { toast(err.message, true); }
    });
    ul.appendChild(li);
  }
  // Cap the visible rows at ~3 entries; the rest scroll (like the report
  // viewport) so the sidebar stays compact as history grows.
  const first = ul.querySelector("li");
  if (first) {
    const gap = parseFloat(getComputedStyle(ul).rowGap) || 6;
    ul.style.maxHeight = `${first.offsetHeight * 3 + gap * 2 + 2}px`;
  } else {
    ul.style.maxHeight = "";
  }
}

function clearSelection() {
  state.selectedId = null;
  state.job = null;
  state.files = [];
  state.lastRunSig = null;
  state.fileTab = "results";
  state.logOpen = null;
  state.sulfUiUrl = null;
  state.sulfHidden = false;
  $("jobDetail").classList.add("hidden");
  $("emptyDetail").classList.remove("hidden");
  renderJobList();
}

async function selectJob(id) {
  state.selectedId = id;
  state.lastRunSig = null;
  state.fileTab = "results";
  state.logOpen = null;
  state.sulfUiUrl = null;
  state.sulfHidden = false;
  renderJobList();
  const data = await api(`/api/jobs/${encodeURIComponent(id)}`);
  state.job = data.job;
  state.files = (await api(
    `/api/jobs/${encodeURIComponent(id)}/files`)).files;
  renderJobDetail();
}

/* ================= job detail ================= */

function lastRun(job) { return job.runs && job.runs.length ? job.runs[job.runs.length - 1] : null; }
function isRunning(job) { return lastRun(job) && lastRun(job).status === "running"; }

function renderJobDetail() {
  const job = state.job;
  const el = $("jobDetail");
  if (!job) return;
  // remember whether the live-log <details> was open across re-renders
  const prevDetails = el.querySelector(".logbox");
  if (prevDetails) state.logOpen = prevDetails.open;

  $("emptyDetail").classList.add("hidden");
  el.classList.remove("hidden");

  const hasResults = state.files.some((f) =>
    f.path.startsWith("results/") || f.path === "results.json" || f.path === "results.html");
  const allCmd = state.commands.find((c) => c.id === "all");

  el.innerHTML = `
  <!-- task header -->
  <header class="task-head">
    <div>
      <h1 class="th-name">${esc(job.name)}</h1>
      <div class="th-meta">
        <span>${fmtTime(job.created)}</span><span class="sep">·</span>
        <span>${job.files} files</span><span class="sep">·</span>
        <span>${humanSize(job.size)}</span>
        <span class="th-id" title="task id (timestamp + content hash ${esc(job.hash)})">${esc(job.id)}</span>
      </div>
    </div>
    <div class="th-actions">
      <button type="button" class="btn btn-sm" id="jdFiles" title="Re-scan the task folder">${icon("refresh")} Rescan</button>
      ${hasResults ? `<button type="button" class="btn btn-sm" id="jdZip">${icon("download")} results.zip</button>` : ""}
      <button type="button" class="btn btn-sm btn-danger" id="jdDel">${icon("trash")} Delete</button>
    </div>
  </header>

  <!-- run analysis -->
  <section class="panel">
    <div class="panel-head">
      <h2>Run analysis</h2>
      <span class="panel-hint right">runs in the background — <code>mea &lt;command&gt;</code></span>
    </div>

    <div class="run-primary">
      <button type="button" class="btn btn-primary btn-run" id="btnAll">
        ${icon("play", 13)} Run full pipeline
      </button>
      <label class="check"><input type="checkbox" id="noSulf"${state.noSulf ? " checked" : ""}>
        skip sulf-cvrg step (<code>--no-sulf</code>)</label>
    </div>
    ${allCmd ? `<div class="panel-hint" style="margin-top:6px">${esc(allCmd.description)}</div>` : ""}

    <div class="run-steps">
      <span class="rs-label">Single steps</span>
      <div class="step-chips" id="funcgrid"></div>
    </div>

    <div id="runArea"></div>
  </section>

  <!-- report -->
  <section class="panel">
    <div class="panel-head"><h2>Report</h2></div>
    <div id="reportSection"></div>
  </section>

  <!-- files -->
  <section class="panel">
    <div class="panel-head">
      <h2>Files</h2>
      <div class="right"><div class="ftabs" id="ftabs"></div></div>
    </div>
    <div class="filetable" id="filetree"></div>
  </section>

  <div id="histSection"></div>`;

  // ---- single-step chips (all steps except 'all') ----
  const grid = $("funcgrid");
  const running = isRunning(job);
  for (const c of state.commands) {
    if (c.id === "all") continue;
    const b = document.createElement("button");
    b.type = "button";
    b.className = `step-chip${running && c.id === lastRun(job).command ? " active" : ""}`;
    b.disabled = running || state.uploading;
    b.title = c.description;
    b.textContent = c.id;
    b.addEventListener("click", () => runCommand(job.id, c.id));
    grid.appendChild(b);
  }

  // ---- primary all button ----
  const all = $("btnAll");
  all.disabled = running || state.uploading;
  all.addEventListener("click", () => runCommand(job.id, "all"));
  $("noSulf").addEventListener("change", (e) => { state.noSulf = e.target.checked; });

  // ---- header actions ----
  $("jdFiles").addEventListener("click", async () => {
    state.files = (await api(`/api/jobs/${encodeURIComponent(job.id)}/files`)).files;
    renderReportSection();
    renderFiles();
    renderHistory();
    toast("file list refreshed");
  });
  const zb = $("jdZip");
  if (zb) zb.addEventListener("click", () => {
    window.location.href = `/api/jobs/${encodeURIComponent(job.id)}/zip`;
  });
  $("jdDel").addEventListener("click", async () => {
    if (!confirm(`Delete task “${job.name}” and all its files?`)) return;
    try {
      await api(`/api/jobs/${encodeURIComponent(job.id)}`, { method: "DELETE" });
      toast("task deleted");
      clearSelection();
      refreshJobs();
    } catch (e) { toast(e.message, true); }
  });

  renderRunArea();
  renderReportSection();
  renderFiles();
  renderHistory();
}

/* ---------- live run status ---------- */

function renderRunArea() {
  const el = $("runArea");
  if (!el) return;
  const job = state.job;
  const lr = lastRun(job);
  if (!lr) {
    el.innerHTML = "";
    return;
  }
  const running = lr.status === "running";
  const logOpen = state.logOpen != null ? state.logOpen : running;
  el.innerHTML = `
    <div class="statusbar ${lr.status}">
      ${running ? '<span class="spin"></span>' : icon(STATUS_ICON[lr.status] || "file", 15)}
      <span class="sb-cmd">mea ${esc(lr.command)}</span>
      <span class="sb-meta">run #${lr.id} · started ${fmtClock(lr.started)}
        ${lr.finished ? "· " + fmtDur(lr.started, lr.finished) : `· <span id="sbElapsed">…</span>`}</span>
      ${statusPill(lr.status, lr)}
      <span class="sb-right">
        <button type="button" class="btn btn-sm" id="jrLog">Full log</button>
        ${running ? `<button type="button" class="btn btn-sm btn-danger" id="jrStop">${icon("stop", 12)} Stop</button>` : ""}
      </span>
    </div>
    <details class="logbox" ${logOpen ? "open" : ""}>
      <summary>Live log (last 64 KB)</summary>
      <pre class="logpre" id="logTail">…</pre>
    </details>
    ${running && state.sulfUiUrl && !state.sulfHidden ? `
    <div class="sulf-panel">
      <div class="sulf-head">
        <span class="sulf-title">${icon("flask", 13)} Sulfonate coverage — waiting for boundary selection</span>
        <span class="sb-right">
          <button type="button" class="btn btn-sm" id="sulfPop">${icon("external", 12)} Open in new tab</button>
          <button type="button" class="btn btn-sm btn-ghost" id="sulfHide">Hide</button>
        </span>
      </div>
      <iframe class="sulf-frame" src="${esc(state.sulfUiUrl)}" title="Sulfonate coverage boundary selection"></iframe>
    </div>` : ""}
    ${running && state.sulfUiUrl && state.sulfHidden ? `
    <div class="sulf-panel slim">
      <span class="sulf-title">${icon("flask", 13)} Boundary selection is waiting in the background</span>
      <span class="sb-right">
        <button type="button" class="btn btn-sm" id="sulfShow">Show selector</button>
        <button type="button" class="btn btn-sm" id="sulfPop2">${icon("external", 12)} Open in new tab</button>
      </span>
    </div>` : ""}`;
  updateLogTail();
  const pop1 = $("sulfPop"), pop2 = $("sulfPop2");
  if (pop1) pop1.addEventListener("click", () => window.open(state.sulfUiUrl, "_blank"));
  if (pop2) pop2.addEventListener("click", () => window.open(state.sulfUiUrl, "_blank"));
  const hide = $("sulfHide");
  if (hide) hide.addEventListener("click", () => { state.sulfHidden = true; renderRunArea(); });
  const show = $("sulfShow");
  if (show) show.addEventListener("click", () => { state.sulfHidden = false; renderRunArea(); });
  $("jrLog").addEventListener("click", () => showRunLog(job.id, lr.id));
  const stp = $("jrStop");
  if (stp) stp.addEventListener("click", async () => {
    try {
      await api(`/api/jobs/${encodeURIComponent(job.id)}/stop`, { method: "POST" });
      toast("stopping…");
      await refreshJob();
    } catch (e) { toast(e.message, true); }
  });
}

async function updateLogTail() {
  const job = state.job;
  const lr = lastRun(job);
  const pre = $("logTail");
  if (!job || !lr || !pre || !document.body.contains(pre)) return;
  try {
    const d = await api(
      `/api/jobs/${encodeURIComponent(job.id)}/log?run=${lr.id}`);
    pre.textContent = d.text || "(no output yet)";
    pre.scrollTop = pre.scrollHeight;
  } catch { /* ignore */ }
}

/* ---------- report ---------- */

function renderReportSection() {
  const el = $("reportSection");
  if (!el) return;
  const job = state.job;
  const has = state.files.some((f) => f.path === "results.html");
  if (!has) {
    el.innerHTML = `<div class="report-empty">
      No final report yet — run <b>Run full pipeline</b> (or <code>conclude</code> + <code>render</code>)
      to generate <code>results.html</code>.</div>`;
    return;
  }
  el.innerHTML = `
    <div class="report-card">
      <div class="report-head">
        <span class="rt">Final report</span>
        <span>results.json → rendered by <code>mea render</code></span>
        <span class="right">
          <button type="button" class="btn btn-sm" id="rpOpen">${icon("external", 12)} Open in new tab</button>
        </span>
      </div>
      <div class="report-wrap">
        <iframe src="${fileUrl(job.id, "results.html")}" title="MEA analysis report"></iframe>
      </div>
    </div>`;
  $("rpOpen").addEventListener("click", () =>
    window.open(fileUrl(job.id, "results.html"), "_blank"));
}

/* ---------- files (tabbed) ---------- */

function fileGroups() {
  const files = state.files;
  const results = files.filter((f) =>
    f.path.startsWith("results/") || f.path === "results.json" || f.path === "results.html");
  const logs = files.filter((f) => f.path.startsWith("logs/"));
  const known = new Set([...results, ...logs].map((f) => f.path));
  const data = files.filter((f) => !known.has(f.path));
  return [
    ["results", "Results", results],
    ["logs", "Logs", logs],
    ["data", "Uploaded data", data],
  ];
}

function renderFiles() {
  const tabs = $("ftabs"), tree = $("filetree");
  if (!tabs || !tree) return;
  const groups = fileGroups();
  const active = groups.some((g) => g[0] === state.fileTab) ? state.fileTab : "results";

  tabs.innerHTML = groups.map(([id, label, fs]) => `
    <button type="button" class="ftab${id === active ? " active" : ""}" data-tab="${id}">
      ${esc(label)}<span class="n">${fs.length}</span></button>`).join("");
  tabs.querySelectorAll(".ftab").forEach((b) =>
    b.addEventListener("click", () => {
      state.fileTab = b.dataset.tab;
      renderFiles();
    }));

  const [, , fs] = groups.find((g) => g[0] === active);
  if (!fs.length) {
    tree.innerHTML = `<div class="file-empty">nothing here yet</div>`;
    return;
  }
  tree.innerHTML = fs.map((f) => `
    <div class="file-row" data-path="${esc(f.path)}">
      <span class="file-ico">${icon(icoFor(f.path))}</span>
      <span class="file-path">${esc(f.path)}</span>
      <span class="file-size">${f.size_h}</span>
    </div>`).join("");
  tree.querySelectorAll(".file-row").forEach((row) =>
    row.addEventListener("click", () => {
      const f = fs.find((x) => x.path === row.dataset.path);
      if (f) previewFile(state.job.id, f);
    }));
}

/* ---------- run history ---------- */

function renderHistory() {
  const el = $("histSection");
  if (!el) return;
  const job = state.job;
  if (!job.runs || !job.runs.length) {
    el.innerHTML = "";
    return;
  }
  const rows = job.runs.slice().reverse().map((r) => `
    <tr>
      <td>#${r.id}</td>
      <td><b>${esc(r.command)}</b>${r.args && r.args.length ? ` <span class="th-meta" style="margin:0">${esc(r.args.join(" "))}</span>` : ""}</td>
      <td>${fmtTime(r.started)}</td>
      <td>${fmtDur(r.started, r.finished)}</td>
      <td>${statusPill(r.status, r)}</td>
      <td><span class="link" data-runlog="${r.id}">log</span></td>
    </tr>`).join("");
  el.innerHTML = `
    <section class="panel">
      <div class="panel-head"><h2>Run history</h2></div>
      <table class="runs">
        <thead><tr><th>#</th><th>Command</th><th>Started</th><th>Duration</th><th>Status</th><th></th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </section>`;
  el.querySelectorAll("[data-runlog]").forEach((s) =>
    s.addEventListener("click", () => showRunLog(job.id, +s.dataset.runlog)));
}

async function refreshJob() {
  if (!state.selectedId) return;
  const data = await api(`/api/jobs/${encodeURIComponent(state.selectedId)}`);
  state.job = data.job;
  state.files = (await api(
    `/api/jobs/${encodeURIComponent(state.selectedId)}/files`)).files;
  renderJobDetail();
}

async function showRunLog(jobId, runId) {
  const pre = document.createElement("pre");
  pre.textContent = "loading…";
  openModal(`run #${runId} — full log`, pre);
  try {
    const d = await api(`/api/jobs/${encodeURIComponent(jobId)}/log?run=${runId}`);
    pre.textContent = d.text || "(empty log)";
    pre.scrollTop = pre.scrollHeight;
  } catch (e) { pre.textContent = e.message; }
}

/* ================= run command ================= */

async function runCommand(jobId, command) {
  const args = [];
  if (command === "all" && state.noSulf) args.push("--no-sulf");
  try {
    const d = await api(`/api/jobs/${encodeURIComponent(jobId)}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command, args }),
    });
    toast(`mea ${command} started (run #${d.run.id})`);
    await refreshJob();
  } catch (e) { toast(`run failed: ${e.message}`, true); }
}

/* ================= issues (sidebar notebook) ================= */

async function refreshIssues() {
  try {
    const data = await api("/api/issues");
    state.issues = data.issues;
    renderIssues();
  } catch { /* server unreachable */ }
}

function renderIssues() {
  const ul = $("issueList");
  if (!ul) return;
  ul.innerHTML = "";
  $("noIssues").classList.toggle("hidden", state.issues.length > 0);
  const pill = $("issueCount");
  pill.classList.toggle("hidden", state.issues.length === 0);
  pill.textContent = state.issues.length || "";
  for (const it of state.issues) {
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="il-top">
        <span class="il-ico">${icon("note", 13)}</span>
        <span class="il-title" title="${esc(it.name)}">${esc(it.title)}</span>
        <button type="button" class="tl-del" title="Delete issue" aria-label="Delete issue">${icon("trash", 13)}</button>
      </div>
      <div class="il-sub">${fmtTime(it.created)}</div>
      ${it.preview ? `<div class="il-preview">${esc(it.preview)}</div>` : ""}`;
    li.addEventListener("click", (e) => {
      if (e.target.closest(".tl-del")) return;
      viewIssue(it.name);
    });
    li.querySelector(".tl-del").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete issue “${it.title}”?`)) return;
      try {
        await api(`/api/issues/${encodeURIComponent(it.name)}`, { method: "DELETE" });
        toast("issue deleted");
        refreshIssues();
      } catch (err) { toast(err.message, true); }
    });
    ul.appendChild(li);
  }
}

async function viewIssue(name) {
  const pre = document.createElement("pre");
  pre.textContent = "loading…";
  openModal(name, pre);
  try {
    const d = await api(`/api/issues/${encodeURIComponent(name)}`);
    pre.textContent = d.content;
  } catch (e) { pre.textContent = e.message; }
}

function newIssueForm() {
  const form = document.createElement("div");
  form.className = "issue-form";
  form.innerHTML = `
    <label class="if-label">Title
      <input class="if-title" id="ifTitle" maxlength="120"
             placeholder="short summary, e.g. sulf-cvrg 峰边界默认值偏窄">
    </label>
    <label class="if-label">Description <span class="if-hint">Markdown supported</span>
      <textarea class="if-body" id="ifBody" rows="10"
                placeholder="用自然语言描述问题：现象、复现步骤、期望行为…"></textarea>
    </label>
    <div class="if-actions">
      <button type="button" class="btn btn-ghost" id="ifCancel">Cancel</button>
      <button type="button" class="btn btn-primary" id="ifSave">Save issue</button>
    </div>`;
  openModal("New issue", form);
  const titleEl = form.querySelector("#ifTitle");
  titleEl.focus();
  form.querySelector("#ifCancel").addEventListener("click", closeModal);
  const save = async () => {
    const t = titleEl.value.trim();
    const b = form.querySelector("#ifBody").value.trim();
    if (!t) { toast("title is required", true); titleEl.focus(); return; }
    if (!b) { toast("description is required", true); return; }
    try {
      await api("/api/issues", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: t, content: b }),
      });
      closeModal();
      toast("issue saved");
      refreshIssues();
    } catch (e) { toast(`save failed: ${e.message}`, true); }
  };
  form.querySelector("#ifSave").addEventListener("click", save);
  form.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); save(); }
  });
}

/* ================= polling loop ================= */

async function tick() {
  state.tick += 1;
  try {
    await refreshJobs();
  } catch { /* server restarting */ return; }

  if (!state.selectedId) return;
  // keep selection fresh every tick (cheap); detect run-state transitions
  let job;
  try {
    job = (await api(`/api/jobs/${encodeURIComponent(state.selectedId)}`)).job;
  } catch { return; }
  state.job = job;

  const lr = lastRun(job);
  const sig = lr ? `${lr.id}:${lr.status}:${lr.rc}` : "none";
  if (sig !== state.lastRunSig) {
    state.lastRunSig = sig;
    // state changed (new run, finished, failed) → refresh files + UI
    state.files = (await api(
      `/api/jobs/${encodeURIComponent(state.selectedId)}/files`)).files;
    renderJobDetail();
    return;
  }
  if (lr && lr.status === "running") {
    // in-place updates: elapsed time + log tail
    const e = $("sbElapsed");
    if (e && document.body.contains(e)) {
      e.textContent = fmtDur(lr.started, Date.now() / 1000);
    }
    await updateLogTail();
    // discover the interactive sulf-cvrg page (embedded boundary picker)
    try {
      const s = await api(
        `/api/jobs/${encodeURIComponent(state.selectedId)}/sulf-ui`);
      const url = s.active ? s.url : null;
      if (url !== state.sulfUiUrl) {
        state.sulfUiUrl = url;
        state.sulfHidden = false;
        renderRunArea();
      }
    } catch { /* endpoint unavailable */ }
  }
}

/* ================= drag & drop + inputs ================= */

function wireDropzone() {
  const dz = $("dropzone");
  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", (e) => {
    const items = e.dataTransfer.items;
    // prefer the first directory entry if the browser exposes it
    let handled = false;
    if (items && items.length) {
      for (const it of items) {
        if (it.kind === "file") {
          const entry = it.webkitGetAsEntry && it.webkitGetAsEntry();
          if (entry && entry.isDirectory) {
            readEntry(entry);
            handled = true;
            break;
          }
        }
      }
    }
    if (!handled) {
      const fs = e.dataTransfer.files;
      if (fs.length === 1 && fs[0].name.toLowerCase().endsWith(".zip")) {
        state.staging = collectZip(fs[0]);
        state.staging.fileObj = fs[0];
      } else if (fs.length) {
        state.staging = collectFolder(fs);
      }
    }
    renderStage();
  });
  // click / keyboard anywhere on the dropzone opens the folder picker
  dz.addEventListener("click", (e) => {
    if (e.target.closest("button")) return;
    $("folderInput").click();
  });
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      $("folderInput").click();
    }
  });

  $("btnFolder").addEventListener("click", () => $("folderInput").click());
  $("btnZip").addEventListener("click", () => $("zipInput").click());
  $("folderInput").addEventListener("change", (e) => {
    state.staging = collectFolder(e.target.files);
    e.target.value = "";
    renderStage();
  });
  $("zipInput").addEventListener("change", (e) => {
    const f = e.target.files[0];
    const st = collectZip(f);
    if (st) st.fileObj = f;
    state.staging = st;
    e.target.value = "";
    renderStage();
  });
  $("btnCreate").addEventListener("click", createAndUpload);
  $("btnCancel").addEventListener("click", () => {
    state.staging = null;
    renderStage();
  });
}

/* walk a dropped directory entry (webkit) */
async function readEntry(entry) {
  const files = [];
  async function walk(dir, prefix) {
    const reader = dir.createReader();
    for (;;) {
      const batch = await new Promise((res, rej) =>
        reader.readEntries(res, rej));
      if (!batch.length) break;
      for (const en of batch) {
        if (en.isFile) {
          const file = await new Promise((res, rej) => en.file(res, rej));
          file.__rel = prefix + en.name;
          files.push(file);
        } else if (en.isDirectory) {
          await walk(en, prefix + en.name + "/");
        }
      }
    }
  }
  try {
    await walk(entry, "");
  } catch (e) {
    toast(`folder read failed: ${e.message}`, true);
    return;
  }
  const withRel = files.map((f) => ({ file: f, rel: f.__rel || f.name }));
  const size = withRel.reduce((a, b) => a + b.file.size, 0);
  const tops = new Set(withRel.map((x) => x.rel.split("/")[0]));
  state.staging = {
    mode: "folder",
    name: tops.size === 1 ? [...tops][0] : "upload",
    files: withRel,
    size,
  };
  renderStage();
}

/* ================= init ================= */

async function init() {
  wireDropzone();
  $("btnNewIssue").addEventListener("click", newIssueForm);
  try {
    const h = await api("/api/health");
    state.rootPath = h.root;
    $("rootPath").textContent = h.root;
    const c = await api("/api/commands");
    state.commands = c.commands;
  } catch (e) {
    toast(`cannot reach mea-web server: ${e.message}`, true);
  }
  refreshIssues();
  await refreshJobs();
  if (state.jobs.length) await selectJob(state.jobs[0].id);
  setInterval(tick, 2500);
}

init();
