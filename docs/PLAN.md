# Build plan

The app is built in 13 small phases. Each phase ends with a git commit. After each phase the work is reviewed and the next phase's detailed plan is approved before it starts.

## Requirements

- **Chat model** is chosen from the models installed in Ollama, plus Qwen2.5-VL on AWS.
- **PDFs** can contain anything: text, tables, photos, financial charts, scanned pages. Answers must cover all of it.
- **Reading images and charts** while processing a PDF uses Qwen2.5-VL locally (Ollama) or Amazon Nova Lite on AWS.
- Everything else is open source.
- Runs in Docker. One FastAPI server serves both the API and the web page.
- Qwen2.5-VL is not installed locally by this build; it gets installed later with `ollama pull`.

## Decisions

| Topic | Choice |
|---|---|
| Qwen2.5-VL on AWS | Bedrock Custom Model Import. Called by model ARN with `InvokeModel`, using Qwen's own prompt format with base64 images. |
| Nova Lite | `amazon.nova-lite-v1:0` through the Bedrock Converse API. Nova 2 Lite is a one-line setting change. |
| AWS region | `us-east-1` (configurable) |
| Local vision model | `qwen2.5vl:3b`, which fits a 6 GB GPU. `7b` reads charts better but runs partly on CPU. |
| GPU use | Only one chat or vision model loaded in Ollama at a time; others are unloaded first. |
| Embeddings | `nomic-embed-text` in Ollama, on CPU, always loaded. |
| Storage | SQLite (documents, chats, keyword index) and ChromaDB (vectors), all under `./data`. |
| PDF reading | PyMuPDF / pymupdf4llm |
| Web page | Plain HTML, CSS and JavaScript with no build step; marked and DOMPurify for rendering answers. |
| Ollama access from Docker | Host networking, so Ollama's settings don't need to change. |
| Not used | LangChain or other RAG frameworks; the chunker is written for this project. |

## Phases

1. **Project skeleton and Docker.** Git repo, settings, FastAPI app with a health check and placeholder page, Dockerfile, docker-compose, pytest.
2. **Ollama connection and model list.** List installed models and what they can do; load one model at a time; `GET /api/models`.
3. **AWS Bedrock client.** Nova Lite through Converse; Qwen2.5-VL through InvokeModel with retries while the model starts; `/api/aws/check`.
4. **Storage and embeddings.** SQLite tables, ChromaDB, `nomic-embed-text` on CPU.
5. **PDF upload and extraction.** Upload with size limit and duplicate check; text and tables per page; page previews; cropped figures (photos and vector-drawn charts); scanned-page detection.
6. **Reading charts, images and scans.** Local Qwen2.5-VL or Nova Lite, chosen per upload. Charts become data tables plus a summary; scans are transcribed. Runs in the background with progress.
7. **Chunking and indexing.** Heading-aware chunks; tables split by rows with headers repeated; one chunk per figure linked to its image; vector and keyword indexes.
8. **Search.** Meaning and keyword search combined (Reciprocal Rank Fusion), filtered by selected documents.
9. **Answers.** Streaming chat endpoint. Answers only from the PDFs, with page citations and exact numbers. Vision-capable models also get the chart images.
10. **Chat history and follow-ups.** Saved chats; follow-up questions rewritten into full questions before searching.
11. **Web UI, part 1.** Model picker, drag-and-drop upload with processing choice, progress, document list.
12. **Web UI, part 2.** Streaming chat, formatted tables, clickable sources, history sidebar, stop button.
13. **Polish and full test.** Clear error messages, Docker health check, setup guide, full run on a real financial PDF.
