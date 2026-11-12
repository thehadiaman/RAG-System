// The model picker in the header.
import { api } from "./api.js";
import { h, store } from "./dom.js";

const STORAGE_KEY = "pdf-rag:model";

function optionText(model) {
  const parts = [model.label];
  if (model.parameter_size) parts.push(model.parameter_size);
  if (model.vision) parts.push("reads images");
  if (model.loaded) parts.push("loaded");
  return parts.join(" · ");
}

function noteText(model) {
  if (!model) return "";
  if (model.provider === "aws") return "Runs on AWS";
  return model.vision ? "Can also look at chart images" : "Text only";
}

// Fills the picker. Returns the provider problems ({ollama: "...", aws: "..."}) for the banner.
export async function loadModels(select, note) {
  const data = await api.models();
  const byId = new Map(data.models.map((m) => [m.id, m]));
  const local = data.models.filter((m) => m.provider === "ollama");
  const aws = data.models.filter((m) => m.provider === "aws");

  const groups = [];
  if (local.length) {
    groups.push(h("optgroup", { label: "On this computer (Ollama)" }, local.map((m) => h("option", { value: m.id }, optionText(m)))));
  }
  const awsStatus = data.providers.aws;
  if (aws.length) {
    groups.push(h("optgroup", { label: "AWS" }, aws.map((m) => h("option", { value: m.id }, optionText(m)))));
  } else if (awsStatus && !awsStatus.available) {
    groups.push(h("optgroup", { label: "AWS" }, h("option", { disabled: true, value: "" }, `Qwen2.5-VL (AWS): not set up (${awsStatus.error})`)));
  }
  if (!data.models.length) {
    groups.unshift(h("option", { disabled: true, value: "" }, "No chat models available"));
  }
  select.replaceChildren(...groups);

  const saved = store(STORAGE_KEY);
  select.value = byId.has(saved) ? saved : (data.models[0]?.id ?? "");
  const update = () => {
    note.textContent = noteText(byId.get(select.value));
  };
  select.onchange = () => {
    store(STORAGE_KEY, select.value);
    update();
  };
  update();

  const problems = {};
  for (const [name, status] of Object.entries(data.providers)) {
    if (!status.available && name !== "aws") problems[name] = status.error;
  }
  return problems;
}
