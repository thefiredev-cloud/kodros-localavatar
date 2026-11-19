"use strict";

const $ = (id) => document.getElementById(id);
let currentJob = null;
let pollTimer = null;

function fmtSecs(s) {
  s = Math.max(0, Math.round(s));
  return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  const ct = r.headers.get("Content-Type") || "";
  const body = ct.includes("json") ? await r.json() : null;
  if (!r.ok) {
    const msg = body && (body.errors ? body.errors.join("\n") : body.error);
    throw new Error(msg || "request failed (" + r.status + ")");
  }
  return body;
}

// ---- voices
async function loadVoices() {
  const sel = $("voice");
  try {
    const data = await api("/api/voices");
    const groups = {};
    for (const v of data.voices) {
      const key = v.group;
      if (!groups[key]) {
        groups[key] = document.createElement("optgroup");
        groups[key].label = key;
        sel.appendChild(groups[key]);
      }
      const o = document.createElement("option");
      o.value = v.id;
      o.textContent = v.label;
      groups[key].appendChild(o);
    }
    sel.value = data.default;
  } catch (e) {
    $("form-error").textContent = e.message;
    $("form-error").hidden = false;
  }
}

let previewAudio = null;
$("preview").addEventListener("click", async () => {
  const btn = $("preview");
  btn.disabled = true;
  btn.textContent = "Loading…";
  try {
    const r = await fetch("/api/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ voice: $("voice").value, speed: parseFloat($("speed").value) }),
    });
    if (!r.ok) throw new Error((await r.json()).error || "preview failed");
    const url = URL.createObjectURL(await r.blob());
    if (previewAudio) previewAudio.pause();
    previewAudio = new Audio(url);
    await previewAudio.play();
  } catch (e) {
    $("form-error").textContent = e.message;
    $("form-error").hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = "Play sample";
  }
});

$("speed").addEventListener("input", () => {
  $("speed-out").textContent = parseFloat($("speed").value).toFixed(2).replace(/0$/, "") + "x";
  updateScriptInfo();
});

// ---- photo + script
$("photo").addEventListener("change", () => {
  const f = $("photo").files[0];
  if (!f) return;
  const t = $("thumb");
  t.src = URL.createObjectURL(f);
  t.hidden = false;
  $("drop-text").textContent = f.name;
});

function updateScriptInfo() {
  const words = ($("script").value.match(/\S+/g) || []).length;
  const speed = parseFloat($("speed").value) || 1;
  const secs = words / (2.5 * speed); // about 150 words per minute at 1.0x
  $("script-info").textContent = words + " words, about " + fmtSecs(secs) + " of video";
}
$("script").addEventListener("input", updateScriptInfo);

// ---- submit + poll
$("make").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const err = $("form-error");
  err.hidden = true;
  $("go").disabled = true;
  try {
    const job = await api("/api/jobs", { method: "POST", body: new FormData($("make")) });
    showJob(job.id);
    refreshJobs();
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
  } finally {
    $("go").disabled = false;
  }
});

function showJob(id) {
  currentJob = id;
  $("result").hidden = false;
  $("video").hidden = true;
  $("video").removeAttribute("src");
  $("actions").hidden = true;
  $("log").hidden = true;
  poll();
}

async function poll() {
  clearTimeout(pollTimer);
  if (!currentJob) return;
  const id = currentJob;
  let job;
  try {
    job = await api("/api/jobs/" + id);
  } catch (e) {
    $("status").textContent = e.message;
    return;
  }
  if (id !== currentJob) return;
  renderJob(job);
  if (job.status === "queued" || job.status === "running") {
    pollTimer = setTimeout(poll, 2000);
  } else {
    refreshJobs();
  }
}

function renderJob(job) {
  const st = $("status");
  const log = $("log");
  if (job.status === "queued") {
    st.textContent = "Queued. Waiting for the GPU.";
  } else if (job.status === "running") {
    const started = Date.parse(job.started);
    st.textContent = "Rendering, " + fmtSecs((Date.now() - started) / 1000) + " elapsed. This takes minutes.";
  } else if (job.status === "done") {
    st.textContent = "Done in " + fmtSecs(job.runtime_sec) + ".";
  } else {
    st.textContent = "Failed: " + (job.error || "unknown error");
  }
  log.textContent = (job.log || []).join("\n");
  log.hidden = !(job.log && job.log.length);
  $("cancel").hidden = job.status !== "queued";
  $("actions").hidden = !(job.status === "done" || job.status === "queued");
  const v = $("video");
  if (job.status === "done" && job.output) {
    const src = "/api/jobs/" + job.id + "/video";
    if (!v.getAttribute("src")) v.src = src;
    v.hidden = false;
    $("download").href = "/api/jobs/" + job.id + "/download";
    $("download").hidden = false;
  } else {
    $("download").hidden = true;
  }
}

$("cancel").addEventListener("click", async () => {
  if (!currentJob) return;
  try {
    await api("/api/jobs/" + currentJob + "/cancel", { method: "POST" });
  } catch (e) {
    $("status").textContent = e.message;
  }
  poll();
});

// ---- recent jobs
async function refreshJobs() {
  const data = await api("/api/jobs").catch(() => ({ jobs: [] }));
  const ul = $("jobs");
  ul.replaceChildren();
  $("jobs-empty").hidden = data.jobs.length > 0;
  for (const j of data.jobs) {
    const li = document.createElement("li");
    const tag = document.createElement("span");
    tag.className = "tag " + j.status;
    tag.textContent = j.status;
    const snip = document.createElement("span");
    snip.className = "snip";
    snip.textContent = j.snippet;
    const meta = document.createElement("span");
    meta.className = "hint";
    meta.textContent = j.aspect;
    li.append(tag, snip, meta);
    li.addEventListener("click", () => showJob(j.id));
    ul.appendChild(li);
  }
}

async function loadHealth() {
  try {
    const h = await api("/api/health");
    const msgs = [];
    if (h.missing.length) msgs.push("SadTalker is not set up: run `localavatar setup`.");
    if (h.gpu_reserved) msgs.push("A GPU-reserved process is running, so GPU renders are refused.");
    if (h.cuda_error) {
      msgs.push("GPU unavailable: " + h.cuda_error);
      $("cpu").checked = true;
    }
    $("gpu-warn").textContent = msgs.join(" ");
    $("gpu-warn").hidden = msgs.length === 0;
  } catch (e) {
    /* health is advisory */
  }
}

loadVoices();
updateScriptInfo();
refreshJobs();
loadHealth();
