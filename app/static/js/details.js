// The document details dialog: summary, actions, pages with their extracted text, and figures.
import { api } from "./api.js";
import { IN_PROGRESS, extractorName, progressBar, statusClass, statusLabel } from "./documents.js";
import { formatBytes, h, plural, toast } from "./dom.js";

const POLL_MS = 2500;
// Singular names; plural() adds the "s" where English does.
const CHUNK_NAME = { text: "text passage", table: "table", figure: "figure" };
const UNCOUNTABLE = new Set(["decorative", "other", "unread"]);

function countOf(n, kind) {
  return UNCOUNTABLE.has(kind) ? `${n} ${kind}` : plural(n, kind);
}

export class DetailsDialog {
  constructor(dialog, { extractors, onChanged }) {
    this.dialog = dialog;
    this.extractors = extractors; // a function returning the current extractor list
    this.onChanged = onChanged;
    this.timer = null;
    this.docId = null;
    this.tab = "pages";
    dialog.addEventListener("close", () => {
      clearTimeout(this.timer);
      this.docId = null;
    });
    // Clicking the dark area around the dialog closes it.
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });
  }

  async open(docId) {
    this.docId = docId;
    this.tab = "pages";
    this.dialog.replaceChildren(h("p", { class: "loading" }, "Loading…"));
    if (!this.dialog.open) this.dialog.showModal();
    await this.load();
  }

  async load() {
    clearTimeout(this.timer);
    const docId = this.docId;
    if (!docId) return;
    let doc;
    try {
      doc = await api.document(docId);
    } catch (error) {
      this.dialog.replaceChildren(this.header(null), h("p", { class: "error-text" }, error.message));
      return;
    }
    if (docId !== this.docId) return; // closed or switched while loading
    this.doc = doc;
    this.render();
    if (IN_PROGRESS.has(doc.status)) this.timer = setTimeout(() => this.load(), POLL_MS);
  }

  render() {
    const doc = this.doc;
    this.dialog.replaceChildren(
      this.header(doc),
      this.summary(doc),
      this.actions(doc),
      this.tabs(),
      h("div", { class: "tab-body" }, this.tab === "pages" ? this.pagesGrid(doc) : this.figuresList(doc)),
    );
  }

  header(doc) {
    return h(
      "header",
      { class: "details-head" },
      h("div", {},
        h("h2", { id: "details-title" }, doc ? doc.filename : "Document"),
        doc ? h("span", { class: `doc-status ${statusClass(doc)}` }, statusLabel(doc)) : null,
      ),
      h("button", { class: "icon-button", type: "button", "aria-label": "Close", onclick: () => this.dialog.close() }, "×"),
    );
  }

  summary(doc) {
    const kinds = {};
    for (const figure of doc.figures) kinds[figure.kind || "unread"] = (kinds[figure.kind || "unread"] || 0) + 1;
    const chunkParts = Object.entries(doc.chunk_counts).map(([kind, n]) => plural(n, CHUNK_NAME[kind] || kind));
    const facts = [
      ["Pages", doc.page_count ?? "?"],
      ["Size", formatBytes(doc.size_bytes)],
      ["Charts read by", extractorName(doc.extractor)],
      ["Figures", doc.figures.length ? `${doc.figures.length} (${Object.entries(kinds).map(([k, n]) => countOf(n, k)).join(", ")})` : "none"],
      ["Search pieces", chunkParts.length ? chunkParts.join(", ") : "none yet"],
    ];
    return h(
      "section",
      { class: "details-summary" },
      h("dl", {}, facts.map(([term, value]) => [h("dt", {}, term), h("dd", {}, value)])),
      IN_PROGRESS.has(doc.status) ? progressBar(doc) : null,
      doc.error ? h("p", { class: "doc-error" }, doc.error) : null,
      doc.warning ? h("p", { class: "doc-warning" }, doc.warning) : null,
    );
  }

  actions(doc) {
    const busy = IN_PROGRESS.has(doc.status);
    const failed = doc.status === "failed";
    const choices = this.extractors().filter((e) => e.id !== "none");
    const select = h(
      "select",
      { "aria-label": "Read charts with", disabled: busy || failed },
      choices.map((e) => h("option", { value: e.id, disabled: !e.available, selected: e.id === doc.extractor }, e.available ? e.label : `${e.label} (not available)`)),
    );
    const run = (label, action) => async () => {
      try {
        await action();
        toast(label);
        this.onChanged();
        await this.load();
      } catch (error) {
        toast(error.message, "error");
      }
    };
    return h(
      "section",
      { class: "details-actions" },
      h("div", { class: "action-group" },
        select,
        h("button", { type: "button", disabled: busy || failed || !choices.some((e) => e.available), onclick: run("Reading the charts again.", () => api.describe(doc.id, select.value)) }, "Read charts again"),
      ),
      h("button", { type: "button", disabled: busy || failed, onclick: run("Rebuilding the search index.", () => api.reindex(doc.id)) }, "Rebuild search index"),
      h("button", { type: "button", class: "danger", onclick: () => this.remove(doc) }, "Delete"),
    );
  }

  async remove(doc) {
    if (!window.confirm(`Delete ${doc.filename}? Its pages, figures and search index are removed. Saved chats keep their copies of the sources.`)) return;
    try {
      await api.deleteDocument(doc.id);
      toast(`Deleted ${doc.filename}.`);
      this.dialog.close();
      this.onChanged();
    } catch (error) {
      toast(error.message, "error");
    }
  }

  tabs() {
    const tab = (id, label) =>
      h("button", {
        type: "button",
        role: "tab",
        class: "tab",
        "aria-selected": String(this.tab === id),
        onclick: () => {
          this.tab = id;
          this.render();
        },
      }, label);
    return h("div", { class: "tabs", role: "tablist" }, tab("pages", `Pages (${this.doc.pages.length})`), tab("figures", `Figures (${this.doc.figures.length})`));
  }

  pagesGrid(doc) {
    if (!doc.pages.length) return h("p", { class: "muted" }, "Pages appear here as they are read.");
    return h(
      "div",
      { class: "page-grid" },
      doc.pages.map((page) =>
        h(
          "button",
          { type: "button", class: "page-thumb", onclick: () => this.showPage(page.page_no), "aria-label": `Page ${page.page_no}` },
          h("img", { src: page.image_url, alt: "", loading: "lazy" }),
          h("span", { class: "page-label" }, `Page ${page.page_no}`),
          h("span", { class: "badges" },
            page.table_count ? h("span", { class: "badge" }, plural(page.table_count, "table")) : null,
            page.figure_count ? h("span", { class: "badge" }, plural(page.figure_count, "figure")) : null,
            page.needs_ocr ? h("span", { class: "badge badge-warn" }, "scanned") : null,
          ),
        ),
      ),
    );
  }

  async showPage(pageNo) {
    const body = this.dialog.querySelector(".tab-body");
    body.replaceChildren(h("p", { class: "loading" }, "Loading page…"));
    let page;
    try {
      page = await api.page(this.doc.id, pageNo);
    } catch (error) {
      body.replaceChildren(h("p", { class: "error-text" }, error.message));
      return;
    }
    const source = page.text_source === "vision" ? "Text read by the vision model (scanned page)" : "Text extracted from the PDF";
    body.replaceChildren(
      h("div", { class: "page-view-head" },
        h("button", { type: "button", onclick: () => this.render() }, "← All pages"),
        h("strong", {}, `Page ${pageNo} of ${this.doc.page_count}`),
        h("span", { class: "page-nav" },
          h("button", { type: "button", disabled: pageNo <= 1, onclick: () => this.showPage(pageNo - 1), "aria-label": "Previous page" }, "‹"),
          h("button", { type: "button", disabled: pageNo >= this.doc.page_count, onclick: () => this.showPage(pageNo + 1), "aria-label": "Next page" }, "›"),
        ),
      ),
      h("div", { class: "page-view" },
        h("img", { src: page.image_url, alt: `Page ${pageNo}` }),
        h("div", { class: "page-text" },
          h("p", { class: "muted" }, source),
          page.read_error ? h("p", { class: "doc-error" }, page.read_error) : null,
          h("pre", {}, page.text || "(no text on this page)"),
        ),
      ),
    );
  }

  figuresList(doc) {
    if (!doc.figures.length) return h("p", { class: "muted" }, "No charts or pictures were found in this document.");
    return h(
      "ul",
      { class: "figure-list" },
      doc.figures.map((figure) =>
        h(
          "li",
          { class: "figure" },
          h("a", { href: figure.image_url, target: "_blank", rel: "noopener", class: "figure-image" }, h("img", { src: figure.image_url, alt: figure.caption || `Figure on page ${figure.page_no}`, loading: "lazy" })),
          h("div", { class: "figure-body" },
            h("div", { class: "figure-head" },
              h("strong", {}, `Page ${figure.page_no}`),
              figure.kind ? h("span", { class: "badge" }, figure.kind) : null,
              figure.described_by ? h("span", { class: "muted" }, `read by ${figure.described_by}`) : null,
            ),
            figure.caption ? h("p", { class: "figure-caption" }, figure.caption) : null,
            figure.describe_error ? h("p", { class: "doc-error" }, figure.describe_error) : null,
            figure.description
              ? h("pre", {}, figure.description)
              : h("p", { class: "muted" }, "Not read by a vision model. Search uses the text found inside it."),
          ),
        ),
      ),
    );
  }
}
