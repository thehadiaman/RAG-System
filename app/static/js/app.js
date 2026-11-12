// Entry point: wires the header, the Documents panel and the details dialog together.
import { DetailsDialog } from "./details.js";
import { DocumentsPanel } from "./documents.js";
import { toast } from "./dom.js";
import { checkHealth } from "./health.js";
import { loadModels } from "./models.js";

const HEALTH_EVERY_MS = 30000;
const $ = (id) => document.getElementById(id);

async function start() {
  const banner = $("banner");
  let providerProblems = {};

  const refreshModelsAndHealth = async () => {
    try {
      providerProblems = await loadModels($("model-select"), $("model-note"));
    } catch (error) {
      providerProblems = { models: error.message };
    }
    await checkHealth(banner, providerProblems);
  };

  const documents = new DocumentsPanel({
    list: $("document-list"),
    empty: $("documents-empty"),
    count: $("documents-count"),
    uploads: $("uploads"),
    dropzone: $("dropzone"),
    fileInput: $("file-input"),
    extractorChoice: $("extractor-choice"),
    onOpen: (docId) => details.open(docId),
  });
  const details = new DetailsDialog($("details"), {
    extractors: () => documents.extractors,
    onChanged: () => documents.refresh(),
  });

  await Promise.all([refreshModelsAndHealth(), documents.start()]);
  setInterval(() => checkHealth(banner, providerProblems), HEALTH_EVERY_MS);
  document.body.dataset.ready = "true";
}

start().catch((error) => toast(`The page couldn't start: ${error.message}`, "error"));
