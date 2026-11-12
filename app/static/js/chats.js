// The Chats list in the sidebar.
import { api } from "./api.js";
import { h, timeAgo, toast } from "./dom.js";

export class ChatList {
  constructor({ list, empty, newButton, onSelect, onNew, onDeleted }) {
    Object.assign(this, { list, empty, onSelect, onDeleted });
    this.chats = [];
    this.activeId = null;
    newButton.addEventListener("click", onNew);
  }

  async refresh() {
    try {
      this.chats = await api.chats();
    } catch (error) {
      toast(error.message, "error");
      return;
    }
    this.render();
  }

  setActive(chatId) {
    this.activeId = chatId;
    for (const item of this.list.children) item.classList.toggle("active", item.dataset.id === chatId);
  }

  render() {
    this.empty.hidden = this.chats.length > 0;
    this.list.replaceChildren(...this.chats.map((chat) => this.row(chat)));
  }

  row(chat) {
    return h(
      "li",
      { class: `chat-item${chat.id === this.activeId ? " active" : ""}`, dataset: { id: chat.id } },
      h(
        "button",
        { type: "button", class: "chat-open", onclick: () => this.onSelect(chat.id), "aria-current": chat.id === this.activeId ? "true" : null },
        h("span", { class: "chat-title-text" }, chat.title),
        h("span", { class: "chat-time" }, timeAgo(chat.updated_at)),
      ),
      h("span", { class: "chat-actions" },
        h("button", { type: "button", class: "icon-button small", "aria-label": `Rename ${chat.title}`, title: "Rename", onclick: () => this.rename(chat) }, "✎"),
        h("button", { type: "button", class: "icon-button small", "aria-label": `Delete ${chat.title}`, title: "Delete", onclick: () => this.remove(chat) }, "🗑"),
      ),
    );
  }

  async rename(chat) {
    const title = window.prompt("New name for this chat", chat.title);
    if (!title || !title.trim() || title.trim() === chat.title) return;
    try {
      await api.updateChat(chat.id, { title: title.trim() });
      await this.refresh();
    } catch (error) {
      toast(error.message, "error");
    }
  }

  async remove(chat) {
    if (!window.confirm(`Delete the chat "${chat.title}"? This can't be undone.`)) return;
    try {
      await api.deleteChat(chat.id);
      toast("Chat deleted.");
      await this.refresh();
      this.onDeleted(chat.id);
    } catch (error) {
      toast(error.message, "error");
    }
  }
}
