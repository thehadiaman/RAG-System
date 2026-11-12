// Small DOM helpers. Text is always inserted as text, never as HTML.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else if (value === true) el.setAttribute(key, "");
    else el.setAttribute(key, String(value));
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : String(child));
  }
  return el;
}

export function toast(message, kind = "info") {
  const area = document.getElementById("toasts");
  const close = h("button", { class: "toast-close", "aria-label": "Dismiss" }, "×");
  const item = h("div", { class: `toast toast-${kind}` }, h("span", {}, message), close);
  const remove = () => item.remove();
  close.addEventListener("click", remove);
  area.append(item);
  setTimeout(remove, kind === "error" ? 12000 : 6000);
}

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function plural(count, word, many = `${word}s`) {
  return `${count} ${count === 1 ? word : many}`;
}

export function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch {
    // Private windows can refuse storage; the choice is just not remembered.
  }
  return null;
}

// "just now", "5 min ago", "3 h ago", "yesterday", or a date.
export function timeAgo(iso) {
  const then = new Date(iso);
  const seconds = (Date.now() - then.getTime()) / 1000;
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  if (seconds < 172800) return "yesterday";
  return then.toLocaleDateString(undefined, { day: "numeric", month: "short" });
}
