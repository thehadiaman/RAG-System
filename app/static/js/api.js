// Calls to the app's own API.

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

function detailMessage(data) {
  if (!data || data.detail === undefined) return null;
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) return data.detail.map((d) => d.msg).join("; ");
  return JSON.stringify(data.detail);
}

async function request(method, path, body) {
  let response;
  try {
    response = await fetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new ApiError("The app's server isn't responding. Is it running?", 0);
  }
  if (response.status === 204) return null;
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(detailMessage(data) || `${response.status} ${response.statusText}`, response.status);
  }
  return data;
}

// Upload with progress (fetch can't report upload progress). Resolves to { status, data }.
function upload(file, extractor, onProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    if (extractor) form.append("extractor", extractor);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/documents");
    xhr.responseType = "json";
    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    });
    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve({ status: xhr.status, data: xhr.response });
      else reject(new ApiError(detailMessage(xhr.response) || `Upload failed (${xhr.status})`, xhr.status));
    });
    xhr.addEventListener("error", () => reject(new ApiError("The upload didn't reach the server.", 0)));
    xhr.send(form);
  });
}

// Ask a question in a chat. Returns the streaming response; read it with readEvents().
async function ask(chatId, body, signal) {
  let response;
  try {
    response = await fetch(`/api/chats/${chatId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError("The app's server isn't responding. Is it running?", 0);
  }
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new ApiError(detailMessage(data) || `${response.status} ${response.statusText}`, response.status);
  }
  return response;
}

export const api = {
  health: () => request("GET", "/api/health"),
  models: () => request("GET", "/api/models"),
  extractors: () => request("GET", "/api/extractors"),
  documents: () => request("GET", "/api/documents"),
  document: (id) => request("GET", `/api/documents/${id}`),
  page: (id, pageNo) => request("GET", `/api/documents/${id}/pages/${pageNo}`),
  describe: (id, extractor) =>
    request("POST", `/api/documents/${id}/describe?extractor=${encodeURIComponent(extractor)}`),
  reindex: (id) => request("POST", `/api/documents/${id}/index`),
  deleteDocument: (id) => request("DELETE", `/api/documents/${id}`),
  upload,
  chats: () => request("GET", "/api/chats"),
  chat: (id) => request("GET", `/api/chats/${id}`),
  createChat: (body) => request("POST", "/api/chats", body),
  updateChat: (id, body) => request("PATCH", `/api/chats/${id}`, body),
  deleteChat: (id) => request("DELETE", `/api/chats/${id}`),
  ask,
};
