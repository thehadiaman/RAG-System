// Markdown from the model, turned into safe HTML, with [n] citations turned into buttons.
import DOMPurify from "../vendor/purify.es.mjs";
import { marked } from "../vendor/marked.esm.js";
import { h } from "./dom.js";

marked.setOptions({ gfm: true, breaks: false });

// Answers come from a model reading PDFs, so anything that could load content,
// run code or take input is removed. Images are blocked too: a remote image in
// an answer would tell an outside server that you read it.
const FORBID_TAGS = ["img", "style", "iframe", "form", "input", "button", "textarea", "select", "object", "embed", "svg", "math", "video", "audio"];

DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName === "A") {
    node.setAttribute("target", "_blank");
    node.setAttribute("rel", "noopener noreferrer");
  }
});

export function renderMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(text ?? ""), {
    FORBID_TAGS,
    FORBID_ATTR: ["style"],
    RETURN_DOM_FRAGMENT: true,
  });
}

const CITATION = /\[(\d+(?:\s*[,;]\s*\d+)*)\]/g;

// Replace [2] or [1, 3] in the text of `root` with buttons, for numbers that are real sources.
export function linkCitations(root, sourceCount, onCite) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => (node.parentElement?.closest("code, pre") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
  });
  const textNodes = [];
  while (walker.nextNode()) textNodes.push(walker.currentNode);

  for (const node of textNodes) {
    const text = node.nodeValue;
    const matches = [...text.matchAll(CITATION)];
    if (!matches.length) continue;
    const parts = [];
    let last = 0;
    for (const match of matches) {
      const numbers = match[1].split(/\s*[,;]\s*/).map(Number);
      if (!numbers.every((n) => n >= 1 && n <= sourceCount)) continue; // e.g. a year like [2025]
      parts.push(text.slice(last, match.index));
      for (const n of numbers) {
        parts.push(h("button", { type: "button", class: "cite", "aria-label": `Source ${n}`, onclick: () => onCite(n) }, String(n)));
      }
      last = match.index + match[0].length;
    }
    if (!parts.length) continue;
    parts.push(text.slice(last));
    node.replaceWith(...parts);
  }
}
