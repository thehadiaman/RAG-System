// The Documents panel: who reads charts, uploading, and the list with live progress.
import { api } from "./api.js";
import { formatBytes, h, plural, store, toast } from "./dom.js";

export const IN_PROGRESS = new Set(["queued", "extracting", "describing", "indexing"]);
const POLL_MS = 2000;
const EXTRACTOR_KEY = "pdf-rag:extractor";
const SHORT_EXTRACTOR = { local: "Qwen on this computer", aws: "Nova Lite (AWS)", none: "Text only" };

export function statusLabel(doc) {
  switch (doc.status) {
    case "ready":
      return "Ready";
    case "failed":
      return "Failed";
    case "extracted":
      return "Not searchable yet";
    case "queued":
      return "Waiting to start";
    default:
      return doc.progress.note || doc.status;
  }
}

export function statusClass(doc) {
  if (doc.status === "ready") return doc.warning ? "status-warn" : "status-ok";
  if (doc.status === "failed") return "status-bad";
  if (doc.status === "extracted") return "status-warn";
  return "status-busy";
}

export function progressBar(doc) {
  const { done, total } = doc.progress;
  return total > 0
    ? h("progress", { value: done, max: total, "aria-label": statusLabel(doc) })
    : h("progress", { "aria-label": statusLabel(doc) });
}

export function extractorName(id) {
  return SHORT_EXTRACTOR[id] || id || "Text only";
}

export class DocumentsPanel {
  constructor({ list, empty, count, uploads, dropzone, fileInput, extractorChoice, uploadBox, onOpen }) {
    Object.assign(this, { list, empty, count, uploads, dropzone, fileInput, extractorChoice, uploadBox, onOpen });
    this.firstRender = true;
    this.timer = null;
    this.documents = [];
    this.extractors = [];
  }

  async start() {
    this.wireUploads();
    await Promise.all([this.loadExtractors(), this.refresh()]);
  }

  // ---- Who reads charts ----

  async loadExtractors() {
    let data;
    try {
      data = await api.extractors();
    } catch (error) {
      this.extractorChoice.append(h("p", { class: "error-text" }, error.message));
      return;
    }
    this.extractors = data.extractors;
    const saved = store(EXTRACTOR_KEY);
    const usable = (id) => data.extractors.some((e) => e.id === id && e.available);
    const chosen = usable(saved) ? saved : usable(data.default) ? data.default : data.extractors.find((e) => e.available)?.id;

    const options = data.extractors.map((extractor) => {
      const input = h("input", {
        type: "radio",
        name: "extractor",
        value: extractor.id,
        disabled: !extractor.available,
        checked: extractor.id === chosen,
        onchange: () => store(EXTRACTOR_KEY, extractor.id),
      });
      return h(
        "label",
        { class: `extractor${extractor.available ? "" : " unavailable"}`, title: extractor.problem || "" },
        input,
        h("span", { class: "extractor-text" },
          h("span", { class: "extractor-name" }, extractor.label),
          extractor.available ? null : h("span", { class: "extractor-problem" }, extractor.problem),
        ),
      );
    });
    this.extractorChoice.append(...options);
  }

  chosenExtractor() {
    return this.extractorChoice.querySelector("input[name=extractor]:checked")?.value;
  }

  // ---- Uploading ----

  wireUploads() {
    const { dropzone, fileInput } = this;
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        fileInput.click();
      }
    });
    fileInput.addEventListener("change", () => {
      this.uploadAll([...fileInput.files]);
      fileInput.value = "";
    });
    for (const type of ["dragenter", "dragover"]) {
      dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragging");
      });
    }
    for (const type of ["dragleave", "drop"]) {
      dropzone.addEventListener(type, () => dropzone.classList.remove("dragging"));
    }
    dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      this.uploadAll([...event.dataTransfer.files]);
    });
  }

  async uploadAll(files) {
    for (const file of files) await this.uploadOne(file);
  }

  async uploadOne(file) {
    const isPdf = file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
    if (!isPdf) {
      toast(`${file.name} isn't a PDF, so it wasn't uploaded.`, "error");
      return;
    }
    const bar = h("progress", { value: 0, max: 1, "aria-label": `Uploading ${file.name}` });
    const row = h("li", { class: "upload" }, h("span", { class: "upload-name" }, file.name), bar);
    this.uploads.append(row);
    try {
      const { status, data } = await api.upload(file, this.chosenExtractor(), (share) => {
        bar.value = share;
      });
      row.remove();
      toast(status === 200 && data.duplicate ? `${file.name} was already uploaded.` : `Uploaded ${file.name}. Reading it now.`);
      await this.refresh();
    } catch (error) {
      row.remove();
      toast(`${file.name}: ${error.message}`, "error");
    }
  }

  // ---- The list ----

  async refresh() {
    clearTimeout(this.timer);
    try {
      this.documents = await api.documents();
      this.render();
    } catch (error) {
      toast(error.message, "error");
    }
    if (this.documents.some((doc) => IN_PROGRESS.has(doc.status))) {
      this.timer = setTimeout(() => this.refresh(), POLL_MS);
    }
  }

  render() {
    const docs = this.documents;
    if (this.firstRender && this.uploadBox) {
      // Keep the sidebar short once there are documents; the box opens with one click.
      this.uploadBox.open = docs.length === 0;
      this.firstRender = false;
    }
    this.empty.hidden = docs.length > 0;
    this.count.textContent = docs.length ? String(docs.length) : "";
    this.list.replaceChildren(...docs.map((doc) => this.row(doc)));
  }

  row(doc) {
    const busy = IN_PROGRESS.has(doc.status);
    const meta = [doc.page_count ? plural(doc.page_count, "page") : null, formatBytes(doc.size_bytes), extractorName(doc.extractor)]
      .filter(Boolean)
      .join(" · ");
    return h(
      "li",
      { class: "doc", dataset: { id: doc.id, status: doc.status } },
      h(
        "button",
        { class: "doc-main", type: "button", onclick: () => this.onOpen(doc.id), "aria-label": `${doc.filename}: ${statusLabel(doc)}. Open details.` },
        h("span", { class: "doc-icon", "aria-hidden": "true" }),
        h("span", { class: "doc-body" },
          h("span", { class: "doc-name" }, doc.filename),
          h("span", { class: "doc-meta" }, meta),
          h("span", { class: `doc-status ${statusClass(doc)}` }, statusLabel(doc)),
          busy ? progressBar(doc) : null,
          doc.error ? h("span", { class: "doc-error" }, doc.error) : null,
          doc.warning ? h("span", { class: "doc-warning" }, doc.warning) : null,
        ),
      ),
    );
  }
}
