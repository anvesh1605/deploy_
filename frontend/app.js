const savedTheme = localStorage.getItem("aruba-qa-theme");
if (savedTheme) {
  document.documentElement.setAttribute("data-theme", savedTheme);
}

const ACTIVE_CONVERSATION_KEY = "aruba-qa-active-conversation";
const API_BASE_URL = (() => {
  const metaBase = document.querySelector('meta[name="api-base-url"]')?.content || "";
  const globalBase =
    window.__ARUBA_API_BASE_URL__ ||
    window.__API_BASE_URL__ ||
    "";
  return String(globalBase || metaBase || "").trim().replace(/\/+$/, "");
})();

const state = {
  activeConversationId: null,
  conversations: [],
  messageCache: new Map(),
  showDebug: false,
  chatBusy: false,
};

const $ = (id) => document.getElementById(id);

const els = {
  conversationList: $("conversationList"),
  exampleList: $("exampleList"),
  contextBadges: $("contextBadges"),
  activeContext: $("activeContext"),
  chatFeed: $("chatFeed"),
  chatForm: $("chatForm"),
  questionInput: $("questionInput"),
  sendBtn: $("sendBtn"),
  hintText: $("hintText"),
  switchSelect: $("switchSelect"),
  domainSelect: $("domainSelect"),
  debugToggle: $("debugToggle"),
  newConversationBtn: $("newConversationBtn"),
  themeToggle: $("themeToggle"),
  mobileMenuToggle: $("mobileMenuToggle"),
  sidebarOverlay: $("sidebarOverlay"),
  sidebar: $("sidebar"),
};

const EXAMPLES = [
  "For 10000 AOS-CX 10.07, what does the guide say about Management Module Failover Overview?",
  "For 8400 AOS-CX 10.17.1000, what is Step 3: Configuring local proxy ARP in switch A?",
  "For 4100i AOS-CX 10.13.1170, what caveat is documented for IP-SLA?",
  "For 4100i AOS-CX 10.16.1030, what caveat is documented for Central?",
  "For 4100i AOS-CX 10.13.1120, what caveat is documented for REST?",
  "For 4100i AOS-CX 10.13.1150, what caveat is documented for Certificates?",
  "For 4100i AOS-CX 10.17.1010, what caveat is documented for Classifiers?",
];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderInlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br>");
}

function splitIntoReadablePoints(text) {
  const normalized = String(text ?? "").replace(/\r\n/g, "\n").trim();
  if (!normalized) return [];
  const paragraphs = normalized
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean);
  const points = [];
  const longText = normalized.replace(/\s+/g, " ").trim();

  for (const paragraph of paragraphs) {
    const lines = paragraph
      .split(/\n+/)
      .map((line) => line.trim())
      .filter(Boolean);
    const chunks = lines.length > 1 ? lines : [paragraph];

    for (const chunk of chunks) {
      const normalizedChunk = chunk.replace(/\s+/g, " ").trim();
      if (!normalizedChunk) continue;

      const structuredMarkers = normalizedChunk
        .split(/(?=\bPOL\d+:\s*|\bShowing\b|\bSyntax\b|\bDescription\b|\bExamples?\b|\bAttached Access List\b|\bAttached Prefix List\b|\bPreference Range\b|\bApplied on VLAN\b|\bApplied on Port\b)/)
        .map((piece) => piece.trim())
        .filter(Boolean);
      if (structuredMarkers.length > 1) {
        points.push(...structuredMarkers);
        continue;
      }

      const sentenceParts = normalizedChunk
        .split(/(?<=[.!?])\s+(?=[A-Z0-9("])/)
        .map((piece) => piece.trim())
        .filter(Boolean);
      if (sentenceParts.length > 1) {
        points.push(...sentenceParts);
        continue;
      }

      if (normalizedChunk.length > 140) {
        const markerParts = normalizedChunk
          .split(/(?=\bPOL\d+:\b|\bPOL\d+:|\bShowing\b|\bStep\s+\d+:\b|\bStep\s+\d+:|\bAttached Access List\b|\bAttached Prefix List\b|\bPreference Range\b|\bApplied on VLAN\b|\bApplied on Port\b)/)
          .map((piece) => piece.trim())
          .filter(Boolean);
        if (markerParts.length > 1) {
          points.push(...markerParts);
          continue;
        }

        const colonParts = normalizedChunk
          .split(/\s+(?=[A-Z][A-Za-z0-9_-]+:\s)/)
          .map((piece) => piece.trim())
          .filter(Boolean);
        if (colonParts.length > 1) {
          points.push(...colonParts);
          continue;
        }

        const commaParts = normalizedChunk
          .split(/,\s+(?=[A-Z0-9("])/)
          .map((piece) => piece.trim())
          .filter(Boolean);
        if (commaParts.length > 1) {
          points.push(...commaParts);
          continue;
        }
      }

      points.push(normalizedChunk);
    }
  }

  if (points.length === 1 && longText.length > 120) {
    const fallbackParts = longText
      .split(/\s+(?=\b[A-Z]{2,}[A-Za-z0-9_-]*:|\bPOL\d+:|\bShowing\b|\bAttached Access List\b|\bAttached Prefix List\b|\bPreference Range\b|\bApplied on VLAN\b|\bApplied on Port\b)/)
      .map((piece) => piece.trim())
      .filter(Boolean);
    if (fallbackParts.length > 1) {
      return fallbackParts;
    }
  }

  return points.filter(Boolean);
}

function renderBulletedText(text) {
  const points = splitIntoReadablePoints(text);
  if (points.length <= 1) {
    return `<p class="answer-paragraph">${renderInlineMarkdown(String(text ?? ""))}</p>`;
  }
  return `
    <ul class="answer-points">
      ${points.map((point) => `<li>${renderInlineMarkdown(point)}</li>`).join("")}
    </ul>
  `;
}

function renderAssistantMarkdown(text) {
  const source = String(text ?? "");
  const parts = [];
  const blockPattern = /```(?:[a-zA-Z0-9_-]+)?\n([\s\S]*?)```/g;
  let lastIndex = 0;
  let match;

  while ((match = blockPattern.exec(source)) !== null) {
    const before = source.slice(lastIndex, match.index);
    if (before) parts.push(renderBulletedText(before));
    parts.push(`
      <div class="code-block">
        <div class="code-block-body">${escapeHtml(match[1].replace(/\n$/, "")).replace(/\n/g, "<br>")}</div>
      </div>
    `);
    lastIndex = blockPattern.lastIndex;
  }

  const after = source.slice(lastIndex);
  if (after) parts.push(renderBulletedText(after));

  return parts.join("");
}

function currentContext() {
  const switchValue = els.switchSelect.value.trim();
  const domainValue = els.domainSelect.value;
  const versionValue = "";
  const subVersionValue = "";
  return {
    switch: switchValue === "CX10000" ? "10000" : switchValue,
    version: versionValue,
    subVersion: subVersionValue,
    domain: domainValue,
  };
}

function renderContextBadges() {
  const { switch: sw, domain } = currentContext();
  const badges = [];
  if (sw) badges.push(sw);
  if (domain && domain !== "auto") badges.push(domain);
  els.activeContext.innerHTML = badges.length
    ? badges.map((value) => `<span class="badge">${escapeHtml(value)}</span>`).join("")
    : `<span class="badge">No context selected</span>`;
}

function renderSidebarContext() {
  const { switch: sw, domain } = currentContext();
  const badges = [];
  if (sw) badges.push(sw);
  if (domain && domain !== "auto") badges.push(domain);
  els.contextBadges.innerHTML = badges.length
    ? badges.map((value) => `<span class="badge">${escapeHtml(value)}</span>`).join("")
    : `<span class="badge">No saved context</span>`;
}

function updateUiState() {
  const hasQuestion = Boolean(els.questionInput.value.trim());
  els.sendBtn.disabled = state.chatBusy || !hasQuestion;
  if (els.hintText) {
    els.hintText.textContent =
      "Press Enter to send. Shift+Enter adds a new line. Chats are saved automatically.";
  }
  renderContextBadges();
  renderSidebarContext();
}

function setChatBusy(value) {
  state.chatBusy = value;
  updateUiState();
}

function renderExamples() {
  els.exampleList.innerHTML = EXAMPLES.map(
    (example) => `<button type="button" class="example-chip" data-example="${escapeHtml(example)}">${escapeHtml(example)}</button>`
  ).join("");
  els.exampleList.querySelectorAll("[data-example]").forEach((btn) => {
    btn.addEventListener("click", () => {
      els.questionInput.value = btn.getAttribute("data-example") || "";
      els.questionInput.focus();
      updateUiState();
    });
  });
}

function renderEmptyState() {
  els.chatFeed.innerHTML = `
    <div class="empty-state">
      <div class="empty-state-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
      </div>
      <h2 class="empty-state-title">Welcome to HPE Aruba Intelligence</h2>
      <p class="empty-state-subtitle">Your AI-powered assistant for network operations.</p>
      
      <div class="empty-state-features">
        <div class="feature-card">
          <div class="feature-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M12 16v-4"></path><path d="M12 8h.01"></path></svg>
          </div>
          <div class="feature-text">
            <h3>Instant Answers</h3>
            <p>Ask about bugs, workarounds, commands, or product documentation.</p>
          </div>
        </div>

        <div class="feature-card">
          <div class="feature-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"></path><polyline points="17 21 17 13 7 13 7 21"></polyline><polyline points="7 3 7 8 15 8"></polyline></svg>
          </div>
          <div class="feature-text">
            <h3>Auto-saved Context</h3>
            <p>Chats are saved automatically and can be reopened from the sidebar.</p>
          </div>
        </div>

        <div class="feature-card">
          <div class="feature-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
          </div>
          <div class="feature-text">
            <h3>Seamless Switching</h3>
            <p>Switching chats keeps the full conversation context intact.</p>
          </div>
        </div>
      </div>
    </div>
  `;
}

function renderMessages(messages) {
  els.chatFeed.innerHTML = "";
  if (!messages || !messages.length) {
    renderEmptyState();
    return;
  }
  for (const message of messages) {
    addMessage(
      message.role,
      message.content || "",
      message.role === "assistant" && state.showDebug
        ? [
          { label: message.predicted_intent || "intent", kind: "warn" },
          { label: message.answer_source || "source", kind: "good" },
        ]
        : null,
      state.showDebug && message.role === "assistant" ? message.debug || null : null
    );
  }
}

function addMessage(role, text, meta = null, debug = null) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;

  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  if (role === "assistant") {
    bubble.innerHTML = renderAssistantMarkdown(text);
  } else {
    bubble.textContent = text;
  }
  wrap.appendChild(bubble);

  if (meta && meta.length) {
    const metaRow = document.createElement("div");
    metaRow.className = "meta-row";
    for (const item of meta) {
      const pill = document.createElement("span");
      pill.className = `pill ${item.kind || ""}`.trim();
      pill.textContent = item.label;
      metaRow.appendChild(pill);
    }
    bubble.appendChild(metaRow);
  }

  if (role === "assistant" && debug) {
    const details = document.createElement("details");
    details.className = "details";
    const summary = document.createElement("summary");
    summary.textContent = "Debug details";
    const content = document.createElement("div");
    content.className = "details-content";
    content.innerHTML = debug;
    details.appendChild(summary);
    details.appendChild(content);
    wrap.appendChild(details);
  }

  els.chatFeed.appendChild(wrap);
  els.chatFeed.scrollTop = els.chatFeed.scrollHeight;
}

function debugHtml(result) {
  const slots = result.slots || {};
  const availability = result.debug?.availability_check || {};
  return `
    <div><strong>Conversation:</strong> ${escapeHtml(result.conversation_id || "-")}</div>
    <div><strong>Answer type:</strong> ${escapeHtml(result.answer_type || "-")}</div>
    <div><strong>Validation passed:</strong> ${result.validation_passed ? "yes" : "no"}</div>
    <div><strong>Rejection reason:</strong> ${escapeHtml(result.rejection_reason || "-")}</div>
    <div><strong>Predicted intent:</strong> ${escapeHtml(result.predicted_intent || "-")}</div>
    <div><strong>Lookup status:</strong> ${escapeHtml(result.lookup_status || "-")}</div>
    <div><strong>Source type:</strong> ${escapeHtml(result.source_type || "-")}</div>
    <div><strong>Data family:</strong> ${escapeHtml(result.data_family || "-")}</div>
    <div><strong>Qwen used:</strong> ${result.qwen_used ? "yes" : "no"}</div>
    <div><strong>Confidence:</strong> ${escapeHtml(result.confidence ?? "-")}</div>
    <div><strong>Similarity:</strong> ${escapeHtml(result.similarity ?? "-")}</div>
    <div><strong>Slots:</strong> ${escapeHtml(JSON.stringify(slots, null, 2))}</div>
    <div><strong>Lookup key used:</strong> ${escapeHtml(result.lookup_key_used || "-")}</div>
    <div><strong>Availability:</strong> ${escapeHtml(JSON.stringify(availability, null, 2))}</div>
    <div><strong>Bypass Qwen:</strong> ${result.debug?.bypass_qwen ? "yes" : "no"}</div>
    <div><strong>Qwen validation:</strong> ${result.qwen_validation_passed ? "passed" : "skipped / failed"}</div>
  `;
}

function renderConversationList() {
  if (!state.conversations.length) {
    els.conversationList.innerHTML = `<div class="conversation-empty">No saved conversations yet.</div>`;
    return;
  }

  els.conversationList.innerHTML = state.conversations
    .map((conversation) => {
      const active = conversation.id === state.activeConversationId ? "active" : "";
      const preview = conversation.last_message_preview || "No messages yet";
      return `
        <div class="conversation-item ${active}" data-conversation-id="${escapeHtml(conversation.id)}">
          <button type="button" class="conversation-main" data-open-conversation="${escapeHtml(conversation.id)}">
            <div class="conversation-title">${escapeHtml(conversation.title || "New chat")}</div>
            <div class="conversation-preview">${escapeHtml(preview)}</div>
          </button>
          <div class="conversation-actions">
            <button type="button" class="conversation-action" data-rename-conversation="${escapeHtml(conversation.id)}">Rename</button>
            <button type="button" class="conversation-action conversation-action--danger" data-delete-conversation="${escapeHtml(conversation.id)}">Delete</button>
          </div>
        </div>
      `;
    })
    .join("");

  els.conversationList.querySelectorAll("[data-open-conversation]").forEach((button) => {
    button.addEventListener("click", () => {
      openConversation(button.getAttribute("data-open-conversation") || "").catch(console.error);
    });
  });

  els.conversationList.querySelectorAll("[data-rename-conversation]").forEach((button) => {
    button.addEventListener("click", async (event) => {
      event.stopPropagation();
      const conversationId = button.getAttribute("data-rename-conversation") || "";
      const current = state.conversations.find((item) => item.id === conversationId);
      const nextTitle = window.prompt("Rename conversation", current?.title || "New chat");
      if (!nextTitle || !conversationId) return;
      await renameConversation(conversationId, nextTitle);
    });
  });

  els.conversationList.querySelectorAll("[data-delete-conversation]").forEach((button) => {
    button.addEventListener("click", async (event) => {
      event.stopPropagation();
      const conversationId = button.getAttribute("data-delete-conversation") || "";
      if (!conversationId) return;
      const current = state.conversations.find((item) => item.id === conversationId);
      const confirmed = window.confirm(`Delete "${current?.title || "this conversation"}"?`);
      if (!confirmed) return;
      await deleteConversation(conversationId);
    });
  });
}

function applyConversationContext(conversation) {
  if (!conversation) return;
  if (typeof conversation.selected_switch === "string") {
    els.switchSelect.value = conversation.selected_switch || "";
  }
  if (typeof conversation.domain === "string") {
    els.domainSelect.value = conversation.domain || "auto";
  }
  updateUiState();
}

async function apiJson(url, options = {}) {
  const resolvedUrl = API_BASE_URL ? new URL(url, `${API_BASE_URL}/`).toString() : url;
  const response = await fetch(resolvedUrl, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`${resolvedUrl} failed: ${response.status} ${detail}`.trim());
  }
  return response.json();
}

async function refreshConversationList() {
  const data = await apiJson("/api/conversations");
  state.conversations = Array.isArray(data.conversations) ? data.conversations : [];
  renderConversationList();
}

async function createConversation(title = null) {
  const context = currentContext();
  const conversation = await apiJson("/api/conversations", {
    method: "POST",
    body: JSON.stringify({
      title,
      domain: context.domain,
      selected_switch: context.switch,
      selected_version: "",
      selected_sub_version: "",
    }),
  });
  await refreshConversationList();
  return conversation;
}

async function openConversation(conversationId) {
  if (!conversationId) return;
  const conversation = await apiJson(`/api/conversations/${encodeURIComponent(conversationId)}`);
  state.activeConversationId = conversation.id;
  localStorage.setItem(ACTIVE_CONVERSATION_KEY, conversation.id);
  state.messageCache.set(conversation.id, conversation.messages || []);
  applyConversationContext(conversation);
  renderMessages(conversation.messages || []);
  renderConversationList();
}

async function renameConversation(conversationId, title) {
  await apiJson(`/api/conversations/${encodeURIComponent(conversationId)}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
  await refreshConversationList();
  if (state.activeConversationId === conversationId) {
    const conversation = state.conversations.find((item) => item.id === conversationId);
    if (conversation) {
      localStorage.setItem(ACTIVE_CONVERSATION_KEY, conversationId);
    }
  }
}

async function persistConversationContext() {
  if (!state.activeConversationId) return;
  const context = currentContext();
  try {
    await apiJson(`/api/conversations/${encodeURIComponent(state.activeConversationId)}`, {
      method: "PATCH",
      body: JSON.stringify({
        domain: context.domain,
        selected_switch: context.switch,
        selected_version: "",
        selected_sub_version: "",
      }),
    });
  } catch (error) {
    console.error(error);
  }
}

async function deleteConversation(conversationId) {
  await apiJson(`/api/conversations/${encodeURIComponent(conversationId)}`, {
    method: "DELETE",
  });
  state.messageCache.delete(conversationId);
  await refreshConversationList();
  if (state.activeConversationId === conversationId) {
    const nextConversation = state.conversations[0] || null;
    if (nextConversation) {
      await openConversation(nextConversation.id);
    } else {
      state.activeConversationId = null;
      localStorage.removeItem(ACTIVE_CONVERSATION_KEY);
      renderEmptyState();
    }
  }
}

async function ensureConversationForChat() {
  if (state.activeConversationId) {
    return state.activeConversationId;
  }
  const created = await createConversation();
  state.activeConversationId = created.id;
  localStorage.setItem(ACTIVE_CONVERSATION_KEY, created.id);
  await openConversation(created.id);
  return created.id;
}

function updateConversationFromResult(result) {
  if (!result?.conversation_id) return;
  const summary = result.conversation_summary || null;
  const existingIndex = state.conversations.findIndex((item) => item.id === result.conversation_id);
  if (summary) {
    if (existingIndex >= 0) {
      state.conversations[existingIndex] = {
        ...state.conversations[existingIndex],
        ...summary,
      };
    } else {
      state.conversations.unshift(summary);
    }
  }
  const currentMessages = state.messageCache.get(result.conversation_id) || [];
  state.messageCache.set(result.conversation_id, currentMessages);
}

async function sendQuestion(question) {
  const conversationId = await ensureConversationForChat();
  const context = currentContext();
  const userMessage = { role: "user", content: question, created_at: new Date().toISOString() };
  const cachedMessages = state.messageCache.get(conversationId) || [];
  cachedMessages.push(userMessage);
  state.messageCache.set(conversationId, cachedMessages);
  renderMessages(cachedMessages);
  setChatBusy(true);
  try {
    const result = await apiJson("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        question,
        conversation_id: conversationId,
        domain: context.domain,
        selected_switch: context.switch,
        selected_version: "",
        selected_sub_version: "",
        show_debug: state.showDebug,
      }),
    });
    state.activeConversationId = result.conversation_id || conversationId;
    localStorage.setItem(ACTIVE_CONVERSATION_KEY, state.activeConversationId);
    const nextMessages = state.messageCache.get(state.activeConversationId) || [];
    nextMessages.push({
      role: "assistant",
      content: result.final_answer || result.answer || result.message || result.lookup_answer || "",
      created_at: new Date().toISOString(),
      predicted_intent: result.predicted_intent,
      answer_source: result.answer_source,
      debug: state.showDebug ? debugHtml(result) : null,
    });
    state.messageCache.set(state.activeConversationId, nextMessages);
    renderMessages(nextMessages);
    updateConversationFromResult(result);
    await refreshConversationList();
    renderConversationList();
    return result;
  } finally {
    setChatBusy(false);
  }
}

function wireEvents() {
  els.chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = els.questionInput.value.trim();
    if (!question || state.chatBusy) return;
    els.questionInput.value = "";
    updateUiState();
    try {
      await sendQuestion(question);
    } catch (error) {
      console.error(error);
      addMessage("assistant", "The local backend could not answer this request.", [
        { label: "Error", kind: "bad" },
      ]);
    }
  });

  els.questionInput.addEventListener("input", updateUiState);
  els.questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (!state.chatBusy && els.questionInput.value.trim()) {
        els.chatForm.requestSubmit();
      }
    }
  });

  els.debugToggle.addEventListener("change", () => {
    state.showDebug = els.debugToggle.checked;
    if (state.activeConversationId) {
      const messages = state.messageCache.get(state.activeConversationId) || [];
      renderMessages(messages);
    }
  });

  [els.switchSelect, els.domainSelect].forEach((element) => {
    element.addEventListener("input", () => {
      updateUiState();
      persistConversationContext().catch(console.error);
    });
    element.addEventListener("change", () => {
      updateUiState();
      persistConversationContext().catch(console.error);
    });
  });

  if (els.themeToggle) {
    els.themeToggle.addEventListener("click", () => {
      const html = document.documentElement;
      const currentTheme = html.getAttribute("data-theme");
      const newTheme = currentTheme === "dark" ? "light" : "dark";
      html.setAttribute("data-theme", newTheme);
      localStorage.setItem("aruba-qa-theme", newTheme);
    });
  }

  if (els.mobileMenuToggle && els.sidebar && els.sidebarOverlay) {
    els.mobileMenuToggle.addEventListener("click", () => {
      const isOpen = els.sidebar.classList.contains("open");
      if (isOpen) {
        els.sidebar.classList.remove("open");
        els.sidebarOverlay.classList.remove("open");
      } else {
        els.sidebar.classList.add("open");
        els.sidebarOverlay.classList.add("open");
      }
    });

    els.sidebarOverlay.addEventListener("click", () => {
      els.sidebar.classList.remove("open");
      els.sidebarOverlay.classList.remove("open");
    });
  }

  els.newConversationBtn.addEventListener("click", async () => {
    const conversation = await createConversation();
    await openConversation(conversation.id);
    els.questionInput.focus();
  });
}

async function bootConversationState() {
  await refreshConversationList();
  const storedActive = localStorage.getItem(ACTIVE_CONVERSATION_KEY);
  if (storedActive && state.conversations.some((item) => item.id === storedActive)) {
    await openConversation(storedActive);
    return;
  }
  if (state.conversations.length) {
    await openConversation(state.conversations[0].id);
    return;
  }
  const created = await createConversation();
  await openConversation(created.id);
}

async function main() {
  renderExamples();
  els.domainSelect.value = "auto";
  wireEvents();
  updateUiState();
  renderEmptyState();
  await bootConversationState();
  updateUiState();
}

main().catch((error) => {
  console.error(error);
  addMessage("assistant", "The local backend is not ready yet.", [{ label: "Error", kind: "bad" }]);
});
