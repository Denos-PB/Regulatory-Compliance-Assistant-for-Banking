const messagesEl = document.getElementById("messages");
const composer = document.getElementById("composer");
const questionEl = document.getElementById("question");
const sendBtn = document.getElementById("send-btn");
const topKInput = document.getElementById("top-k");
const topKValue = document.getElementById("top-k-value");
const sourceFilter = document.getElementById("source-filter");
const showSources = document.getElementById("show-sources");
const statusDot = document.getElementById("status-dot");
const statusLabel = document.getElementById("status-label");
const clearChatBtn = document.getElementById("clear-chat");
const promptList = document.getElementById("prompt-list");

let busy = false;

function autoResizeTextarea(el) {
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatAnswer(text) {
  const escaped = escapeHtml(text.trim());
  const paragraphs = escaped.split(/\n{2,}/).filter(Boolean);
  return paragraphs.map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
}

function createMessage(role, html, extraClass = "") {
  const article = document.createElement("article");
  article.className = `message message--${role} ${extraClass}`.trim();

  const avatar = document.createElement("div");
  avatar.className = `avatar avatar--${role}`;
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = role === "user" ? "You" : "RC";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = html;

  article.append(avatar, bubble);
  return article;
}

function appendUserMessage(text) {
  const msg = createMessage("user", `<p>${escapeHtml(text)}</p>`);
  messagesEl.appendChild(msg);
  scrollToBottom();
}

function appendTypingIndicator() {
  const msg = createMessage(
    "assistant",
    `<div class="typing" aria-label="Generating answer"><span></span><span></span><span></span></div>`,
    "typing-indicator",
  );
  messagesEl.appendChild(msg);
  scrollToBottom();
  return msg;
}

function removeTypingIndicator(el) {
  el?.remove();
}

function appendAssistantMessage(answer, sources) {
  let html = `<div class="answer-body">${formatAnswer(answer)}</div>`;

  if (showSources.checked && sources?.length) {
    const items = sources.map((s) => `<li>${escapeHtml(s)}</li>`).join("");
    html += `
      <div class="sources">
        <p class="sources-title">Source citations</p>
        <ul class="source-list">${items}</ul>
      </div>`;
  }

  const msg = createMessage("assistant", html);
  messagesEl.appendChild(msg);
  scrollToBottom();
}

function appendErrorMessage(detail) {
  const msg = createMessage(
    "assistant",
    `<p class="error-text">${escapeHtml(detail)}</p>`,
  );
  messagesEl.appendChild(msg);
  scrollToBottom();
}

function setBusy(value) {
  busy = value;
  sendBtn.disabled = value;
  questionEl.disabled = value;
}

async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.ok) {
      statusDot.className = "badge-dot badge-dot--ok";
      statusLabel.textContent = "Service ready";
    } else {
      statusDot.className = "badge-dot badge-dot--warn";
      statusLabel.textContent = "API up — check API keys";
    }
  } catch {
    statusDot.className = "badge-dot badge-dot--err";
    statusLabel.textContent = "Cannot reach API";
  }
}

async function loadSources() {
  try {
    const res = await fetch("/sources");
    if (!res.ok) return;
    const data = await res.json();
    for (const src of data.sources || []) {
      const opt = document.createElement("option");
      opt.value = src;
      opt.textContent = shortenSource(src);
      sourceFilter.appendChild(opt);
    }
  } catch {
  }
}

function shortenSource(path) {
  const parts = path.split(/[/\\]/);
  const name = parts[parts.length - 1] || path;
  if (name.length > 48) return `…${name.slice(-45)}`;
  return name;
}

async function askQuestion(question) {
  const trimmed = question.trim();
  if (!trimmed || busy) return;

  setBusy(true);
  appendUserMessage(trimmed);
  questionEl.value = "";
  autoResizeTextarea(questionEl);

  const typing = appendTypingIndicator();

  const body = {
    question: trimmed,
    top_k: Number(topKInput.value),
    return_chunks: false,
  };
  const filter = sourceFilter.value;
  if (filter) body.source_filter = filter;

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));

    removeTypingIndicator(typing);

    if (!res.ok) {
      const detail =
        typeof data.detail === "string"
          ? data.detail.split("\n")[0]
          : "Request failed. Check server logs.";
      appendErrorMessage(detail);
      return;
    }

    appendAssistantMessage(data.answer || "", data.sources || []);
  } catch (err) {
    removeTypingIndicator(typing);
    appendErrorMessage(
      err.message || "Network error. Is the API running on this host?",
    );
  } finally {
    setBusy(false);
    questionEl.focus();
  }
}

function resetChat() {
  messagesEl.innerHTML = "";
  const welcome = createMessage(
    "assistant",
  `<p class="welcome-greeting">Welcome to the compliance desk.</p>
   <p>Ask a question about your indexed regulatory documents. Each answer includes traceable source citations from the knowledge base.</p>
   <p class="muted">Select a suggested inquiry or type your own question below.</p>`,
    "welcome-card",
  );
  messagesEl.appendChild(welcome);
}

topKInput.addEventListener("input", () => {
  topKValue.textContent = topKInput.value;
});

questionEl.addEventListener("input", () => autoResizeTextarea(questionEl));

questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  askQuestion(questionEl.value);
});

promptList.addEventListener("click", (e) => {
  const btn = e.target.closest(".prompt-chip");
  if (!btn) return;
  const prompt = btn.dataset.prompt;
  if (prompt) askQuestion(prompt);
});

clearChatBtn.addEventListener("click", resetChat);

checkHealth();
loadSources();
questionEl.focus();
