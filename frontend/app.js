// Plain vanilla JS - no framework, no build step.
// The access token lives in sessionStorage (cleared when the tab closes).

const $ = (id) => document.getElementById(id);
const state = { token: sessionStorage.getItem("token"), email: sessionStorage.getItem("email"), editingId: null };

// --- API helper -------------------------------------------------------------
// Every call goes through here so the bearer token and 401 handling exist in one place.
async function api(path, { method = "GET", body } = {}) {
  const response = await fetch(`/api${path}`, {
    method,
    headers: {
      ...(body ? { "Content-Type": "application/json" } : {}),
      ...(state.token ? { Authorization: `Bearer ${state.token}` } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  if (response.status === 401) { logout(); throw new Error("Session expired - please log in again."); }
  if (response.status === 204) return null;

  const data = await response.json().catch(() => ({}));
  // FastAPI reports errors as {detail: string} or {detail: [{msg}, ...]} for 422.
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(Array.isArray(detail) ? detail.map((d) => d.msg).join(", ") : detail || response.statusText);
  }
  return data;
}

// --- Views ------------------------------------------------------------------
function showApp() {
  $("auth-view").hidden = true;
  $("app-view").hidden = false;
  $("who").textContent = state.email || "";
  $("entry-date").value = new Date().toISOString().slice(0, 10); // default: today
  loadEntries();
}

function logout() {
  sessionStorage.clear();
  state.token = null;
  state.email = null;
  $("app-view").hidden = true;
  $("auth-view").hidden = false;
  $("messages").innerHTML = "";
}

// --- Auth -------------------------------------------------------------------
$("auth-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  // submitter tells us which of the two buttons was pressed.
  const action = event.submitter?.value === "signup" ? "signup" : "login";
  const message = $("auth-msg");
  message.className = "msg";
  message.textContent = "...";
  try {
    const result = await api(`/auth/${action}`, {
      method: "POST",
      body: { email: $("email").value.trim(), password: $("password").value },
    });
    if (!result.access_token) {
      // Signup with email confirmation enabled: account created, no session yet.
      message.className = "msg ok";
      message.textContent = result.detail;
      return;
    }
    state.token = result.access_token;
    state.email = result.user.email;
    sessionStorage.setItem("token", state.token);
    sessionStorage.setItem("email", state.email || "");
    message.textContent = "";
    showApp();
  } catch (error) {
    message.textContent = error.message;
  }
});

$("logout").addEventListener("click", logout);

// --- Entries ----------------------------------------------------------------
async function loadEntries() {
  const list = $("entries");
  try {
    const entries = await api("/entries");
    list.innerHTML = "";
    if (!entries.length) { list.innerHTML = "<li>No entries yet.</li>"; return; }

    for (const entry of entries) {
      const item = document.createElement("li");
      const date = document.createElement("div");
      date.className = "date";
      date.textContent = entry.entry_date;
      const content = document.createElement("div");
      content.className = "content";
      content.textContent = entry.content; // textContent, not innerHTML: no XSS from entry text

      const row = document.createElement("div");
      row.className = "row";
      const edit = document.createElement("button");
      edit.textContent = "Edit";
      edit.onclick = () => startEdit(entry);
      const remove = document.createElement("button");
      remove.textContent = "Delete";
      remove.onclick = () => deleteEntry(entry.id);
      row.append(edit, remove);

      item.append(date, content, row);
      list.append(item);
    }
  } catch (error) {
    list.innerHTML = `<li>${error.message}</li>`;
  }
}

function startEdit(entry) {
  state.editingId = entry.id;
  $("entry-content").value = entry.content;
  $("entry-date").value = entry.entry_date;
  $("cancel-edit").hidden = false;
  $("entry-content").focus();
}

function cancelEdit() {
  state.editingId = null;
  $("entry-form").reset();
  $("entry-date").value = new Date().toISOString().slice(0, 10);
  $("cancel-edit").hidden = true;
}

$("cancel-edit").addEventListener("click", cancelEdit);

$("entry-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = $("entry-msg");
  message.className = "msg";
  message.textContent = "Saving...";
  const body = { content: $("entry-content").value, entry_date: $("entry-date").value };
  try {
    // PUT re-embeds the entry on the server; POST embeds it for the first time.
    if (state.editingId) await api(`/entries/${state.editingId}`, { method: "PUT", body });
    else await api("/entries", { method: "POST", body });
    message.className = "msg ok";
    message.textContent = "Saved.";
    cancelEdit();
    loadEntries();
  } catch (error) {
    message.textContent = error.message;
  }
});

async function deleteEntry(id) {
  if (!confirm("Delete this entry?")) return;
  try {
    await api(`/entries/${id}`, { method: "DELETE" });
    if (state.editingId === id) cancelEdit();
    loadEntries();
  } catch (error) {
    $("entry-msg").textContent = error.message;
  }
}

// --- Chat (SSE streaming) ---------------------------------------------------
function addTurn(who, text, className) {
  const turn = document.createElement("div");
  turn.className = `turn ${className}`;
  const label = document.createElement("span");
  label.className = "who";
  label.textContent = `${who}: `;
  const bodyEl = document.createElement("span");
  bodyEl.className = "answer";
  bodyEl.textContent = text;
  turn.append(label, bodyEl);
  $("messages").append(turn);
  $("messages").scrollTop = $("messages").scrollHeight;
  return { turn, bodyEl };
}

function renderSources(turn, sources) {
  if (!sources.length) return;
  const list = document.createElement("div");
  list.className = "sources";
  const labels = sources.map((s) => `[Entry: ${s.date}] ${Math.round(s.similarity * 100)}%`);
  list.textContent = `Sources: ${labels.join("  ")}`;
  turn.append(list);
}

$("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  $("question").value = "";
  addTurn("You", question, "you");
  const { turn, bodyEl } = addTurn("Journal", "", "bot");

  try {
    // fetch + ReadableStream instead of EventSource, because EventSource cannot
    // send an Authorization header or use POST.
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${state.token}` },
      body: JSON.stringify({ question, stream: true }),
    });
    if (response.status === 401) { logout(); throw new Error("Session expired - please log in again."); }
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || response.statusText);
    }

    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;

      // SSE frames are separated by a blank line; keep any partial frame in the buffer.
      const frames = buffer.split("\n\n");
      buffer = frames.pop();
      for (const frame of frames) {
        const event = frame.match(/^event: (.*)$/m)?.[1];
        const raw = frame.match(/^data: (.*)$/m)?.[1];
        if (!event || raw === undefined) continue;
        const data = JSON.parse(raw);
        if (event === "sources") renderSources(turn, data);
        else if (event === "token") bodyEl.textContent += data;
        else if (event === "error") bodyEl.textContent += `\n[${data}]`;
        $("messages").scrollTop = $("messages").scrollHeight;
      }
    }
  } catch (error) {
    bodyEl.textContent = `[${error.message}]`;
  }
});

// Resume an existing session on reload.
if (state.token) showApp();
