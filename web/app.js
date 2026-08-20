/* ------------------------------------------------------------------
   AI Meeting Assistant - front end

   Talks to the FastAPI backend in server.py:
     POST /api/jobs               start processing, returns { id }
     GET  /api/jobs/:id           poll status, stage, progress, result
     POST /api/jobs/:id/chat      ask a question about the transcript
     GET  /api/jobs/:id/export    download txt or pdf

   The pipeline takes minutes, so we start a job and poll rather than
   holding a request open.
   ------------------------------------------------------------------ */

const $ = (id) => document.getElementById(id);

const POLL_MS = 1500;

let currentJob = null;
let pollTimer = null;

/* ---------- theme ---------- */

function initTheme() {
  const saved = localStorage.getItem("theme");
  // No saved choice: follow the OS setting.
  const dark = saved ? saved === "dark"
                     : matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(dark);

  $("themeToggle").addEventListener("click", () => {
    const nowDark = document.documentElement.dataset.theme !== "dark";
    applyTheme(nowDark);
    localStorage.setItem("theme", nowDark ? "dark" : "light");
  });
}

function applyTheme(dark) {
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  $("themeIcon").textContent = dark ? "☾" : "☀";
}

/* ---------- input mode tabs ---------- */

function initTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => {
        t.classList.toggle("active", t === tab);
        t.setAttribute("aria-selected", t === tab ? "true" : "false");
      });
      document.querySelectorAll(".pane").forEach((p) => {
        p.classList.toggle("active", p.dataset.pane === tab.dataset.mode);
      });
      hideError();
    });
  });
}

function activeMode() {
  return document.querySelector(".tab.active").dataset.mode;
}

/* ---------- file drop ---------- */

function initDrop() {
  const zone = $("dropZone");
  const input = $("fileInput");

  input.addEventListener("change", () => showChosenFile(input.files[0]));

  // Without preventDefault on dragover the browser just opens the file.
  ["dragenter", "dragover"].forEach((e) =>
    zone.addEventListener(e, (ev) => {
      ev.preventDefault();
      zone.classList.add("over");
    })
  );
  ["dragleave", "drop"].forEach((e) =>
    zone.addEventListener(e, (ev) => {
      ev.preventDefault();
      zone.classList.remove("over");
    })
  );

  zone.addEventListener("drop", (ev) => {
    const file = ev.dataTransfer.files[0];
    if (file) {
      input.files = ev.dataTransfer.files;
      showChosenFile(file);
    }
  });
}

function showChosenFile(file) {
  if (!file) return;
  const mb = (file.size / 1024 / 1024).toFixed(1);
  $("dropTitle").textContent = `${file.name} — ${mb} MB`;
  hideError();
}

/* ---------- starting a job ---------- */

function showError(msg) {
  const el = $("inputError");
  el.textContent = msg;
  el.hidden = false;
}

function hideError() {
  $("inputError").hidden = true;
}

async function startJob() {
  hideError();

  const form = new FormData();
  form.append("language", $("language").value);

  if (activeMode() === "url") {
    const url = $("sourceUrl").value.trim();
    if (!url) return showError("Paste a link first.");
    form.append("source_url", url);
  } else {
    const file = $("fileInput").files[0];
    if (!file) return showError("Choose a file first.");
    form.append("file", file);
  }

  $("startBtn").disabled = true;
  $("startBtn").textContent = "Starting…";

  try {
    const res = await fetch("/api/jobs", { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);

    const { id } = await res.json();
    currentJob = id;

    $("inputCard").hidden = true;
    $("errorCard").hidden = true;
    $("progressCard").hidden = false;
    poll();
  } catch (err) {
    showError(err.message);
  } finally {
    $("startBtn").disabled = false;
    $("startBtn").textContent = "Start";
  }
}

/* ---------- polling ---------- */

async function poll() {
  if (!currentJob) return;

  try {
    const res = await fetch(`/api/jobs/${currentJob}`);
    if (!res.ok) throw new Error("Lost track of that job.");
    const job = await res.json();

    $("progStage").textContent = job.stage;
    $("progSource").textContent = job.source;
    $("progPct").textContent = `${job.progress}%`;
    $("progBar").style.width = `${job.progress}%`;

    if (job.status === "done") {
      renderResult(job);
      return;
    }
    if (job.status === "error") {
      $("progressCard").hidden = true;
      $("errorCard").hidden = false;
      $("errorText").textContent = job.error;
      return;
    }

    pollTimer = setTimeout(poll, POLL_MS);
  } catch (err) {
    $("progressCard").hidden = true;
    $("errorCard").hidden = false;
    $("errorText").textContent = err.message;
  }
}

/* ---------- results ---------- */

function renderResult(job) {
  const r = job.result;

  $("mTitle").textContent = r.title || "Untitled meeting";
  $("mMeta").textContent =
    `${job.source} · ${r.chunk_count} chunk(s) · ${job.language}`;

  $("mSummary").textContent = r.summary;
  $("mActions").textContent = r.action_items;
  $("mDecisions").textContent = r.key_decisions;
  $("mQuestions").textContent = r.open_questions;
  $("mTranscript").textContent = r.transcript;

  $("exportTxt").href = `/api/jobs/${job.id}/export?format=txt`;
  $("exportPdf").href = `/api/jobs/${job.id}/export?format=pdf`;

  $("progressCard").hidden = true;
  $("results").hidden = false;
}

/* ---------- chat ---------- */

function addMessage(who, text, cls = "") {
  const log = $("chatLog");
  log.querySelector(".empty")?.remove();

  const wrap = document.createElement("div");
  wrap.className = `msg ${who === "You" ? "you" : "bot"} ${cls}`.trim();

  const label = document.createElement("span");
  label.className = "who";
  label.textContent = who;

  const body = document.createElement("div");
  body.className = "text";
  // textContent, not innerHTML - the answer is model output and must never
  // be parsed as markup.
  body.textContent = text;

  wrap.append(label, body);
  log.append(wrap);
  log.scrollTop = log.scrollHeight;
  return wrap;
}

async function askQuestion(ev) {
  ev.preventDefault();

  const input = $("chatInput");
  const question = input.value.trim();
  if (!question || !currentJob) return;

  addMessage("You", question);
  input.value = "";
  $("chatSend").disabled = true;

  const pending = addMessage("Assistant", "Thinking…", "pending");

  try {
    const res = await fetch(`/api/jobs/${currentJob}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);

    const { answer } = await res.json();
    pending.classList.remove("pending");
    pending.querySelector(".text").textContent = answer;
  } catch (err) {
    pending.classList.remove("pending");
    pending.querySelector(".text").textContent = `Failed: ${err.message}`;
  } finally {
    $("chatSend").disabled = false;
    input.focus();
    $("chatLog").scrollTop = $("chatLog").scrollHeight;
  }
}

/* ---------- reset ---------- */

function reset() {
  clearTimeout(pollTimer);
  currentJob = null;

  $("results").hidden = true;
  $("errorCard").hidden = true;
  $("progressCard").hidden = true;
  $("inputCard").hidden = false;

  $("chatLog").innerHTML =
    '<p class="hint empty">Ask anything about this meeting.</p>';
  $("sourceUrl").value = "";
  $("fileInput").value = "";
  $("dropTitle").textContent = "Drop a file here";
}

/* ---------- wire up ---------- */

initTheme();
initTabs();
initDrop();

$("startBtn").addEventListener("click", startJob);
$("sourceUrl").addEventListener("keydown", (e) => {
  if (e.key === "Enter") startJob();
});
$("chatForm").addEventListener("submit", askQuestion);
$("newBtn").addEventListener("click", reset);
$("retryBtn").addEventListener("click", reset);
