# Build plan and record

The app was built in 13 small phases. Each phase was planned, approved, built, tested against real PDFs, and committed. This page records the requirements, the decisions, what each phase delivered, and where the build changed from the original plan.

## Requirements

- **Chat model** chosen from the models installed in Ollama, plus Qwen2.5-VL on AWS.
- **PDFs** can contain anything: text, tables, photos, financial charts, scanned pages. Answers must cover all of it.
- **Reading images and charts** while processing a PDF uses Qwen2.5-VL locally (Ollama) or Amazon Nova Lite on AWS.
- Everything else open source.
- Runs in Docker. One FastAPI server serves both the API and the web page.
- Qwen2.5-VL not installed by the build (the user installed it).

## Decisions

| Topic | Choice |
|---|---|
| Qwen2.5-VL on AWS | Bedrock Custom Model Import, called by model ARN with `InvokeModel`, using Qwen's own prompt format with base64 images. |
| Nova Lite | `amazon.nova-lite-v1:0` through the Bedrock Converse API. Nova 2 Lite is a one-line setting change. |
| AWS region | `us-east-1` (configurable). |
| Local vision model | `qwen2.5vl:latest` (7B), the one installed. It runs about half on the CPU with a 6 GB GPU; `qwen2.5vl:3b` fits fully. |
| GPU use | One chat or vision model in the GPU at a time. A shared lock makes chat answers and chart reading take turns. |
| Embeddings | `nomic-embed-text` in Ollama, on the CPU, always loaded. |
| Storage | SQLite (documents, chats, FTS5 keyword index, numbered migrations) and ChromaDB (vectors), all under `./data`. |
| PDF reading | PyMuPDF and pymupdf4llm (AGPL-3.0). |
| Search | Meaning search and keyword search merged with Reciprocal Rank Fusion. |
| Web page | Plain HTML, CSS and JavaScript modules with no build step; marked and DOMPurify stored in the repo. |
| Ollama access from Docker | Host networking, so Ollama's settings don't change. |
| Not used | LangChain or other RAG frameworks; the chunker is written for this project. |

## Phases

| # | Phase | Delivered | Changes from the plan, and findings |
|---|---|---|---|
| 1 | Project skeleton and Docker | FastAPI app, settings, Dockerfile with app and test images, compose. | `httpx2` replaced `httpx` (the maintained successor, required by the FastAPI test client). |
| 2 | Ollama connection | Model list with capabilities; load and unload so one model is in the GPU. | |
| 3 | AWS Bedrock client | Nova Lite (Converse) and imported Qwen2.5-VL (InvokeModel with a hand-written prompt), retries while a model wakes up, `/api/aws/check`. | Tested with simulated AWS only: no AWS account is set up yet. |
| 4 | Storage and embeddings | SQLite with migrations, ChromaDB, CPU embedder. | |
| 5 | PDF upload and extraction | Upload checks, per-page markdown text and tables, page previews, figures found by three signals and cropped, scanned-page detection. | Background processing with progress moved here from Phase 6, because large PDFs need it. Figure rules tuned on three real investor decks. |
| 6 | Reading charts and scans | Local Qwen or Nova Lite per upload, figure types, data tables, scanned-page transcription, resume after restart. | Asking the 7B model for insights produced wrong conclusions, so the prompt asks only for printed numbers. Images are limited by total pixels, not longest side. |
| 7 | Chunking and indexing | Own chunker (headings, overlap, tables split with headers, one chunk per figure), FTS5 keyword index, embeddings, `ready` status. | Section names cleaned of footnote markup after checking real output. |
| 8 | Search | Hybrid search with fusion, document filter, keyword-only fallback. | On ten real questions the right page came first 9 times. |
| 9 | Answers | Streamed answers with numbered sources, chart images for vision models, clear errors. | Added the shared GPU lock. Real answers were often right but sometimes mixed up numbers across sources. |
| 10 | Chats and follow-ups | Saved chats; follow-ups rewritten into complete questions; stopped and failed answers saved. | |
| 11 | Web UI: documents | Model picker, chart-reader choice, upload, live progress, document details. | Playwright and headless Chromium added to the test image for browser tests (+1.1 GB, test image only). |
| 12 | Web UI: chat | Streaming chat, citation chips, source panel, chats list, stop, per-chat documents. | The source panel shows the text exactly as the model read it: rendering it as markdown hid values in lopsided tables. |
| 13 | Polish and full test | Number check, start on boot, Docker health check, clear server errors, startup summary, setup guide. | The number check was added here after Phases 9–10 showed the local model's number mix-ups. A looping chart description led to a cap on description length. |

## Known limitations

- **Local answer quality depends on the model.** The 7B Qwen gets main figures right but sometimes takes a number from the wrong source or wrong period. The number check flags many of these cases; a stronger model (for example Qwen on AWS) reduces them.
- **The number check compares values, not meaning.** It can't tell whether a found number is used for the right thing, and it flags numbers written in a different unit.
- **Chart reading is slow locally:** about 20–60 seconds per chart with the 7B model.
- **AWS paths are untested against real AWS** until an account is set up (`curl localhost:8000/api/aws/check`).
- **No login.** The app is meant for one person on their own machine (`APP_HOST=127.0.0.1`).

## Final test (Phase 13)

A PDF not used during the build, the Dubai Islamic Bank FY 2025 investor presentation (41 pages), was uploaded through the app with local Qwen reading its charts. It found 68 figures and 12 pages with tables. Then 9 questions (plus one follow-up) were asked with `qwen2.5vl:latest`, limited to that document, and checked against the deck.

| Question | Deck says | Answer |
|---|---|---|
| Net profit after tax, FY 2025 | AED 7,808 Mn | ✅ AED 7,808 Mn, cited correctly |
| …follow-up: compared with 2024? | +18% vs normalised 6,637; reported 8,165 | ⚠️ "+18%" is right against normalised profit, but it doesn't say that reported profit was higher |
| Total asset growth | +21% to AED 416 bn | ⚠️ the numbers are there, but it says "grew by 416 billion", mixes in deposits, and cites nothing |
| NPF ratio, end of 2025 | 2.65% | ✅ |
| Proposed dividend | 35 fils | ❌ "couldn't find", though the page was the top search result |
| Cost-to-income, Q4'25 | 28.4% (chart only) | ✅ read from the chart |
| Fees, commission & FX share of non-funded income | 44% | ✅ number right; the number check flagged its citation (the 44% is in the other source) |
| Largest sector in gross financing | Consumer Financing, 29% | ❌ "Government", from another chart; the right table was the 2nd search result |
| Dubai GDP, H1 2025 | AED 241 Bn, +4% | ❌ "couldn't find", though the chart was the 2nd search result |
| Share price target (not in the deck) | – | ✅ correctly refused |

Five of ten were correct, two were partly right, and three were wrong or missed. In every miss the page with the answer was among the top two search results, so retrieval worked and the local 7B model didn't use what it was given. Answers took about 60 seconds each. The number check flagged the one citation slip that involved a number. It can't catch a wrong answer with no number in it ("Government") or a missed answer.

What would raise accuracy most is a stronger answer model, for example Qwen2.5-VL on AWS once it is set up. Fewer sources per answer (`CHAT_MAX_SOURCES`) may also help a small model focus, but that wasn't tested.
