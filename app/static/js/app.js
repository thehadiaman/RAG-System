// Entry point: wires the header, the sidebar, the chat area and the dialogs together.
import { ChatView } from "./chat.js";
import { ChatList } from "./chats.js";
import { DetailsDialog } from "./details.js";
import { DocumentsPanel } from "./documents.js";
import { toast } from "./dom.js";
import { checkHealth } from "./health.js";
import { loadModels } from "./models.js";
import { SourcePanel } from "./sources.js";

const HEALTH_EVERY_MS = 30000;
const $ = (id) => document.getElementById(id);

function chatIdFromUrl() {
  const match = location.hash.match(/^#chat=([0-9a-f]{32})$/);
  return match ? match[1] : null;
}

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
    uploadBox: $("upload-box"),
    onOpen: (docId) => details.open(docId),
  });
  const details = new DetailsDialog($("details"), {
    extractors: () => documents.extractors,
    onChanged: () => documents.refresh(),
  });
  const sources = new SourcePanel($("source-panel"), {
    onShowInDocument: (docId, pageNo) => details.open(docId, pageNo),
  });

  const chats = new ChatList({
    list: $("chat-list"),
    empty: $("chats-empty"),
    newButton: $("new-chat"),
    onSelect: async (chatId) => {
      if (chat.busy) {
        toast("Wait for the answer to finish, or press Stop.");
        return;
      }
      if (await chat.open(chatId)) chats.setActive(chatId);
    },
    onNew: () => {
      if (chat.busy) {
        toast("Wait for the answer to finish, or press Stop.");
        return;
      }
      chat.newChat();
      chats.setActive(null);
    },
    onDeleted: (chatId) => {
      if (chat.chat?.id === chatId) {
        chat.newChat();
        chats.setActive(null);
      }
    },
  });
  const chat = new ChatView(
    {
      form: $("composer"),
      input: $("question"),
      sendButton: $("send"),
      stopButton: $("stop"),
      messages: $("messages"),
      title: $("chat-title"),
      filter: $("doc-filter"),
    },
    {
      modelSelect: $("model-select"),
      documents: () => documents.documents,
      onOpenSource: (source) => sources.open(source),
      onChatChanged: async (chatId) => {
        await chats.refresh();
        chats.setActive(chatId ?? null);
      },
    },
  );

  await Promise.all([refreshModelsAndHealth(), documents.start(), chats.refresh()]);

  const chatId = chatIdFromUrl();
  if (chatId && (await chat.open(chatId))) chats.setActive(chatId);

  setInterval(() => checkHealth(banner, providerProblems), HEALTH_EVERY_MS);
  document.body.dataset.ready = "true";
}

start().catch((error) => toast(`The page couldn't start: ${error.message}`, "error"));
