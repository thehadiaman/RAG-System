// The banner shown when part of the app needs fixing.
import { api } from "./api.js";
import { h } from "./dom.js";

const NAMES = {
  database: "Database",
  vectors: "Search index",
  embedder: "Search model",
  ollama: "Ollama",
};

export async function checkHealth(banner, providerProblems = {}) {
  const problems = [];
  try {
    const health = await api.health();
    for (const [name, component] of Object.entries(health.components)) {
      if (!component.ok) problems.push([name, component.detail]);
    }
  } catch (error) {
    problems.push(["server", error.message]);
  }
  for (const [name, message] of Object.entries(providerProblems)) {
    if (!problems.some(([existing]) => existing === name)) problems.push([name, message]);
  }

  if (!problems.length) {
    banner.hidden = true;
    banner.replaceChildren();
    return;
  }
  banner.replaceChildren(
    h("strong", {}, "Something needs attention"),
    h("ul", {}, problems.map(([name, message]) => h("li", {}, h("span", { class: "banner-name" }, `${NAMES[name] || name}: `), message))),
  );
  banner.hidden = false;
}
