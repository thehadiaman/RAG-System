// The side panel that shows one source: where it is, its picture, and the exact text the model read.
import { h } from "./dom.js";

export class SourcePanel {
  constructor(panel, { onShowInDocument }) {
    this.panel = panel;
    this.onShowInDocument = onShowInDocument;
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !this.panel.hidden && !document.querySelector("dialog[open]")) this.close();
    });
  }

  open(source) {
    const where = [source.filename, `page ${source.page_no}`, source.kind].join(" · ");
    const image = source.figure_image_url || source.page_image_url;
    const close = h("button", { type: "button", class: "icon-button", "aria-label": "Close source", onclick: () => this.close() }, "×");
    const parts = [
      h("header", { class: "source-head" },
        h("h2", { id: "source-title" }, `Source ${source.number}`),
        close,
      ),
      h("p", { class: "source-where" }, where),
      source.section ? h("p", { class: "muted source-section" }, source.section) : null,
      h("a", { href: image, target: "_blank", rel: "noopener", class: "source-image" },
        h("img", { src: image, alt: source.kind === "figure" ? `Figure on page ${source.page_no}` : `Page ${source.page_no}` }),
      ),
      source.image_sent_to_model ? h("p", { class: "muted" }, "The model also looked at this image.") : null,
      h("h3", {}, "Text the model read"),
      // Shown exactly as the model received it: formatting could hide values (a table
      // row with more cells than its header loses the extra cells when rendered).
      h("pre", { class: "passage" }, source.text),
      h("div", { class: "source-actions" },
        h("a", { href: source.page_image_url, target: "_blank", rel: "noopener" }, "Open page image"),
        h("button", { type: "button", onclick: () => this.onShowInDocument(source.document_id, source.page_no) }, "Show in document"),
      ),
    ];
    this.panel.replaceChildren(...parts.filter(Boolean));
    this.panel.hidden = false;
    close.focus();
  }

  close() {
    this.panel.hidden = true;
  }
}
