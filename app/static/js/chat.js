// The chat area: the conversation, the question box, and live answers.
import { api } from "./api.js";
import { h, plural, toast } from "./dom.js";
import { linkCitations, renderMarkdown } from "./markdown.js";
import { readEvents } from "./sse.js";

const NEW_TITLE = "New chat";

function sourceList(numbers) {
  return numbers.length === 1 ? `source ${numbers[0]}` : `sources ${numbers.join(", ")}`;
}

// Why a number was flagged, in plain words.
function flagReason(check) {
  if (check.status === "not_found") return "not in any source the model read";
  return `not in ${sourceList(check.cited)}, but in ${sourceList(check.found_in)}`;
}

// Highlight the first place each flagged number appears in the rendered answer.
function markNumbers(root, flagged) {
  for (const check of flagged) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (node) =>
        node.parentElement?.closest(".cite, code, pre, mark") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT,
    });
    while (walker.nextNode()) {
      const node = walker.currentNode;
      const at = node.nodeValue.indexOf(check.text);
      if (at < 0) continue;
      const rest = node.splitText(at);
      rest.splitText(check.text.length);
      rest.replaceWith(h("mark", { class: "num-flag", title: `Check this number: ${flagReason(check)}.` }, check.text));
      break;
    }
  }
}

function modelName(id) {
  if (!id) return null;
  if (id === "aws:qwen2.5-vl") return "Qwen2.5-VL (AWS)";
  return id.replace(/^ollama:/, "");
}

export class ChatView {
  constructor(els, { modelSelect, onChatChanged, onOpenSource, documents }) {
    Object.assign(this, els);
    this.modelSelect = modelSelect;
    this.onChatChanged = onChatChanged; // the list needs refreshing (new chat, new title, new time)
    this.onOpenSource = onOpenSource;
    this.documents = documents; // () => the current document list
    this.chat = null; // the open chat, or null for a chat not saved yet
    this.pendingDocumentIds = null; // the document choice for a chat not saved yet
    this.controller = null; // set while an answer is streaming

    this.form.addEventListener("submit", (event) => {
      event.preventDefault();
      this.send();
    });
    this.input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        this.send();
      }
    });
    this.input.addEventListener("input", () => this.fitInput());
    this.stopButton.addEventListener("click", () => this.controller?.abort());
    this.filter.addEventListener("toggle", () => {
      if (this.filter.open) this.renderFilter();
    });
    this.showWelcome();
  }

  get busy() {
    return this.controller !== null;
  }

  // ---- Opening chats ----

  newChat() {
    if (this.busy) return;
    this.chat = null;
    this.pendingDocumentIds = null;
    this.title.textContent = NEW_TITLE;
    history.replaceState(null, "", location.pathname);
    this.updateFilterLabel();
    this.showWelcome();
    this.input.focus();
  }

  async open(chatId) {
    if (this.busy) return;
    let chat;
    try {
      chat = await api.chat(chatId);
    } catch (error) {
      toast(error.status === 404 ? "That chat no longer exists." : error.message, "error");
      this.newChat();
      return false;
    }
    this.chat = chat;
    this.title.textContent = chat.title;
    history.replaceState(null, "", `#chat=${chat.id}`);
    if (chat.model && [...this.modelSelect.options].some((o) => o.value === chat.model && !o.disabled)) {
      this.modelSelect.value = chat.model;
      this.modelSelect.dispatchEvent(new Event("change"));
    }
    this.updateFilterLabel();
    this.messages.replaceChildren();
    if (!chat.messages.length) this.showWelcome();
    for (const message of chat.messages) {
      this.messages.append(message.role === "user" ? this.userBubble(message.content) : this.savedAnswer(message));
    }
    this.scrollToEnd(true);
    return true;
  }

  showWelcome() {
    this.messages.replaceChildren(
      h("section", { class: "welcome" },
        h("h2", {}, "Ask questions about your PDFs"),
        h("p", {}, "Answers come only from your uploaded documents, with numbered citations you can click to see the exact page or chart."),
        h("ol", { class: "steps" },
          h("li", {}, h("strong", {}, "Upload a PDF"), " on the left and wait until it says ", h("em", {}, "Ready"), "."),
          h("li", {}, h("strong", {}, "Choose a model"), " at the top."),
          h("li", {}, h("strong", {}, "Ask a question"), " below. Follow-up questions like “and in 2024?” work too."),
        ),
      ),
    );
  }

  // ---- Asking ----

  async send() {
    const question = this.input.value.trim();
    if (!question || this.busy) return;
    const model = this.modelSelect.value;
    if (!model) {
      toast("Pick a model at the top first.", "error");
      return;
    }

    if (!this.messages.querySelector(".msg")) this.messages.replaceChildren();
    const userBubble = this.userBubble(question);
    const answer = this.liveAnswer();
    this.messages.append(userBubble, answer.root);
    this.input.value = "";
    this.fitInput();
    this.scrollToEnd(true);
    this.setBusy(new AbortController());

    let response;
    try {
      if (!this.chat) {
        const created = await api.createChat({ model, document_ids: this.pendingDocumentIds });
        this.chat = { ...created, messages: [] };
        history.replaceState(null, "", `#chat=${created.id}`);
      }
      response = await api.ask(this.chat.id, { question, model }, this.controller.signal);
    } catch (error) {
      // Nothing was saved: put the question back so it can be sent again.
      userBubble.remove();
      answer.root.remove();
      this.input.value = question;
      this.fitInput();
      this.setBusy(null);
      if (error.name !== "AbortError") toast(error.message, "error");
      if (!this.messages.querySelector(".msg")) this.showWelcome();
      this.onChatChanged(this.chat?.id);
      return;
    }

    try {
      for await (const { event, data } of readEvents(response)) answer.handle(event, data);
      answer.finish();
    } catch (error) {
      if (error.name === "AbortError") answer.stopped();
      else answer.failed(`The connection was lost: ${error.message}`);
    } finally {
      this.setBusy(null);
      const refreshed = await api.chat(this.chat.id).catch(() => null);
      if (refreshed) {
        this.chat = refreshed;
        this.title.textContent = refreshed.title;
      }
      this.onChatChanged(this.chat.id);
      this.input.focus();
    }
  }

  setBusy(controller) {
    this.controller = controller;
    this.sendButton.hidden = !!controller;
    this.stopButton.hidden = !controller;
    this.input.placeholder = controller ? "Wait for the answer, or press Stop" : "Ask about your documents…";
  }

  // ---- Messages ----

  userBubble(text) {
    return h("div", { class: "msg msg-user" }, h("div", { class: "bubble" }, text));
  }

  savedAnswer(message) {
    const view = this.answerParts();
    view.text = message.content;
    view.sources = message.sources || [];
    view.render();
    // search_query equals the question for first questions, so this only shows for rewritten follow-ups.
    view.showSearchedFor(message.search_query, this.previousQuestion());
    view.numberCheck(message.number_check);
    view.addSources(message.cited || []);
    if (message.status === "stopped") view.note("Stopped. This is what was written before you pressed Stop.", "warn");
    if (message.status === "error") view.note(`The answer failed: ${message.error}`, "bad");
    view.meta(message.model, message.seconds);
    return view.root;
  }

  previousQuestion() {
    const users = this.messages.querySelectorAll(".msg-user .bubble");
    return users.length ? users[users.length - 1].textContent : "";
  }

  liveAnswer() {
    const view = this.answerParts();
    const question = this.input.value.trim();
    const started = performance.now();
    let model = this.modelSelect.value;
    let statusText = "Searching your documents…";
    const status = h("p", { class: "status-line" }, h("span", { class: "spinner", "aria-hidden": "true" }), h("span", { class: "status-text" }, statusText), h("span", { class: "timer" }, "0 s"));
    view.root.prepend(status);
    const tick = setInterval(() => {
      status.querySelector(".timer").textContent = `${Math.round((performance.now() - started) / 1000)} s`;
    }, 1000);
    const setStatus = (text) => {
      statusText = text;
      status.querySelector(".status-text").textContent = text;
    };
    let frame = 0;
    const scheduleRender = () => {
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        view.render();
        this.scrollToEnd();
      });
    };
    const end = () => {
      clearInterval(tick);
      cancelAnimationFrame(frame);
      status.remove();
    };
    let done = false;

    return {
      root: view.root,
      handle: (event, data) => {
        if (event === "status") setStatus(data.message);
        else if (event === "sources") {
          view.sources = data.sources;
          view.showSearchedFor(data.search_query, question);
          if (data.notice) view.note(data.notice, "warn");
          if (statusText === "Searching your documents…") setStatus(data.sources.length ? "Preparing the answer…" : "Nothing found.");
        } else if (event === "token") {
          view.text += data.text;
          scheduleRender();
        } else if (event === "done") {
          done = true;
          end();
          view.text = data.answer;
          view.render();
          view.numberCheck(data.number_check);
          view.addSources(data.cited);
          view.meta(data.model || model, data.seconds);
        } else if (event === "error") {
          done = true;
          end();
          view.text = data.partial_answer || "";
          view.render();
          view.addSources([]);
          view.note(`The answer failed: ${data.message}`, "bad");
          view.meta(model, (performance.now() - started) / 1000);
        }
      },
      finish: () => {
        if (!done) {
          end();
          view.render();
          view.note("The answer ended unexpectedly.", "bad");
        }
      },
      stopped: () => {
        end();
        view.render();
        view.addSources([]);
        view.note("Stopped. This is what was written before you pressed Stop.", "warn");
        view.meta(model, (performance.now() - started) / 1000);
      },
      failed: (message) => {
        end();
        view.render();
        view.note(message, "bad");
      },
    };
  }

  // The parts of one answer, shared by live and saved answers.
  answerParts() {
    const body = h("div", { class: "answer markdown" });
    const root = h("div", { class: "msg msg-assistant" }, body);
    const onCite = (n) => {
      const source = view.sources.find((s) => s.number === n);
      if (source) this.onOpenSource(source);
    };
    const view = {
      root,
      text: "",
      sources: [],
      render: () => {
        const container = h("div", {}, renderMarkdown(view.text));
        linkCitations(container, view.sources.length, onCite);
        body.replaceChildren(...container.childNodes);
      },
      showSearchedFor: (searchQuery, question) => {
        if (searchQuery && question && searchQuery.trim() !== question.trim()) {
          root.insertBefore(h("p", { class: "searched-for" }, "Searched for: ", h("q", {}, searchQuery)), body);
        }
      },
      note: (text, kind) => root.append(h("p", { class: `msg-note note-${kind}` }, text)),
      numberCheck: (check) => {
        if (!check || !check.summary.checked) return;
        const flagged = check.numbers.filter((n) => n.status !== "ok");
        markNumbers(body, flagged);
        const note = flagged.length
          ? h("div", { class: "check check-warn", role: "note" },
              h("p", {}, `${plural(flagged.length, "number")} couldn't be matched to the cited sources. Check before relying on ${flagged.length === 1 ? "it" : "them"}:`),
              h("ul", {}, flagged.map((n) => h("li", {}, h("strong", {}, n.text), ": ", flagReason(n)))),
            )
          : h("p", { class: "check check-ok" }, `✓ ${check.summary.checked === 1 ? "The number in this answer was" : `All ${check.summary.checked} numbers in this answer were`} found in the cited sources.`);
        body.after(note);
      },
      addSources: (cited) => {
        if (!view.sources.length) return;
        const card = (source) =>
          h("button", { type: "button", class: `source-card${cited.includes(source.number) ? " cited" : ""}`, onclick: () => this.onOpenSource(source) },
            h("span", { class: "cite" }, String(source.number)),
            h("span", { class: "source-card-text" },
              h("span", { class: "source-file" }, source.filename),
              h("span", { class: "muted" }, `page ${source.page_no} · ${source.kind}`),
            ),
          );
        const citedSources = view.sources.filter((s) => cited.includes(s.number));
        const others = view.sources.filter((s) => !cited.includes(s.number));
        root.append(
          h("div", { class: "sources" },
            citedSources.length ? h("div", { class: "source-cards" }, citedSources.map(card)) : null,
            others.length
              ? h("details", { class: "more-sources" },
                  h("summary", {}, citedSources.length ? `${plural(others.length, "more source")} the model read` : `${plural(others.length, "source")} the model read`),
                  h("div", { class: "source-cards" }, others.map(card)),
                )
              : null,
          ),
        );
      },
      meta: (model, seconds) => {
        const copy = h("button", { type: "button", class: "link-button", onclick: () => copyText(view.text) }, "Copy");
        root.append(h("p", { class: "msg-meta" }, [modelName(model), seconds != null ? `${Math.round(seconds)} s` : null].filter(Boolean).join(" · "), " · ", copy));
      },
    };
    return view;
  }

  // ---- Documents this chat uses ----

  chosenDocumentIds() {
    return this.chat ? this.chat.document_ids : this.pendingDocumentIds;
  }

  updateFilterLabel() {
    const ids = this.chosenDocumentIds();
    const summary = this.filter.querySelector("summary");
    if (!ids || !ids.length) summary.textContent = "All documents";
    else if (ids.length === 1) summary.textContent = this.documents().find((d) => d.id === ids[0])?.filename || "1 document";
    else summary.textContent = plural(ids.length, "document");
  }

  renderFilter() {
    const chosen = new Set(this.chosenDocumentIds() || []);
    const ready = this.documents().filter((d) => d.status === "ready");
    const body = this.filter.querySelector(".doc-filter-body");
    if (!ready.length) {
      body.replaceChildren(h("p", { class: "muted" }, "No documents are ready yet."));
      return;
    }
    const boxes = ready.map((doc) =>
      h("label", { class: "doc-filter-option" },
        h("input", { type: "checkbox", value: doc.id, checked: chosen.has(doc.id), onchange: () => this.applyFilter() }),
        h("span", {}, doc.filename),
      ),
    );
    body.replaceChildren(
      h("p", { class: "muted" }, "Answer only from these documents (none ticked = all):"),
      ...boxes,
    );
  }

  async applyFilter() {
    const ids = [...this.filter.querySelectorAll("input[type=checkbox]:checked")].map((box) => box.value);
    const value = ids.length ? ids : null;
    if (this.chat) {
      try {
        const updated = await api.updateChat(this.chat.id, { document_ids: value });
        this.chat = { ...this.chat, document_ids: updated.document_ids };
      } catch (error) {
        toast(error.message, "error");
      }
    } else {
      this.pendingDocumentIds = value;
    }
    this.updateFilterLabel();
  }

  // ---- Small things ----

  fitInput() {
    this.input.style.height = "auto";
    this.input.style.height = `${Math.min(this.input.scrollHeight, 200)}px`;
  }

  scrollToEnd(force = false) {
    const el = this.messages;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    if (force || nearBottom) el.scrollTop = el.scrollHeight;
  }
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast("Answer copied.");
  } catch {
    toast("Couldn't copy: the browser blocked it.", "error");
  }
}
