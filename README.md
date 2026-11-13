# PDF RAG

Ask questions about your PDFs, including their tables, charts and scanned pages, and get answers with numbered citations to the exact page or chart.

- **Answers** come from any chat model installed in your local Ollama, or from Qwen2.5-VL on AWS Bedrock.
- **Charts, pictures and scanned pages** are read by a vision model when you upload: Qwen2.5-VL on this computer (Ollama) or Amazon Nova Lite (AWS). You can also choose text only.
- **Every number** in an answer is looked up in the sources it cites, and anything that can't be matched is highlighted.
- **One server:** FastAPI serves both the API and the web page. Everything runs in Docker, and everything except the optional AWS models is open source and runs on your machine.

How it was built, phase by phase: [docs/PLAN.md](docs/PLAN.md).

## Contents

1. [What you need](#what-you-need)
2. [Set up](#set-up)
3. [Use it](#use-it)
4. [Optional: AWS models](#optional-aws-models)
5. [Settings](#settings)
6. [Troubleshooting](#troubleshooting)
7. [Data, backups and privacy](#data-backups-and-privacy)
8. [For developers](#for-developers)

## What you need

- **Linux with Docker** and Docker Compose 2.24 or newer (`docker compose version`).
- **Ollama** installed on the same machine and running (`ollama ps` works). The app reaches it at `127.0.0.1:11434`; you don't need to change Ollama's settings.
- **Search needs nothing extra.** Its model (nomic-embed-text v1.5) is built into the app and runs on the CPU, with or without Ollama.
- **Models in Ollama:**
  - At least one **chat model** for answers. `qwen2.5vl:latest` (7B, 6 GB) answers and can also look at chart images. Smaller models answer faster.
  - **`qwen2.5vl`** if you want charts read on this computer. The 7B version runs about half on the CPU with a 6 GB GPU. `qwen2.5vl:3b` fits fully in 6 GB and is much faster, but reads charts less well.
- **Disk:** about 1.3 GB for the app image, plus your PDFs and their page images (roughly 2–4× the PDF size).
- **GPU:** optional. An NVIDIA GPU makes local models much faster. The app keeps only one chat or vision model in the GPU at a time.

```bash
ollama pull qwen2.5vl          # reads charts and answers; or qwen2.5vl:3b for speed
```

## Set up

```bash
cp .env.example .env            # optional: edit settings, add AWS keys
docker compose up -d --build    # build and start in the background
```

Open **http://localhost:8000**.

The app starts again by itself after a crash or a reboot. To stop it, run `docker compose stop`. To start it again, run `docker compose up -d`. To follow its log, run `docker compose logs -f app`. The first lines after a start say what is ready and what isn't.

After changing `.env`, run `docker compose up -d` again so the app picks up the change.

After an update that changes the search model, your documents show *Indexing* for a short while at the first start, as their search index is rebuilt. Their text and chart descriptions are kept.

## Use it

1. **Choose who reads charts and scanned pages** in the Documents panel: Qwen on this computer, Nova Lite on AWS, or text only. Options that can't be used right now are greyed out with the reason.
2. **Upload PDFs** by dropping them on the panel or clicking *choose files* (PDF only, up to 100 MB each). Each one shows its progress: reading pages, reading charts, indexing, then **Ready**. Reading charts locally takes about 20–60 seconds per chart, so a deck with 60 charts can take 30–60 minutes. You can keep working meanwhile.
3. **Pick a model** at the top.
4. **Ask a question** at the bottom. Follow-up questions like "and in 2024?" work too.

In an answer:

- **Citations** like `1` are clickable. They open the page or chart image and the exact text the model read. *Show in document* opens that page.
- **Sources** under the answer list what the model read; the ones it cited come first.
- **Number check:** numbers the app couldn't find in the cited sources are highlighted, with the reason. The usual causes are a number taken from a different source than the one cited, or a number that isn't in any source. A highlight means *check this*, not *wrong*. A number written in another unit ("9,500 million" instead of "$9.5B") is also flagged.
- **Stop** ends an answer early; what was written so far is kept.

Click a document to see its pages next to the text taken from each one, the charts with what the vision model wrote about them, and buttons to read the charts again with another model, rebuild its search index, or delete it. Chats are saved in the Chats list, and the button in the chat header limits a chat to some documents.

## Optional: AWS models

Two features use Amazon Bedrock: **Nova Lite** for reading charts, and **Qwen2.5-VL on AWS** for answers. Without AWS settings the app works fully locally and these options are greyed out.

1. **Keys.** In IAM, create a user (or role) allowed to call `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream`. Create an access key and add it to `.env`:

   ```bash
   AWS_REGION=us-east-1
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...
   ```

2. **Nova Lite.** In the Bedrock console for that region, open *Model access* and make sure Amazon Nova Lite is enabled. The default model id is `amazon.nova-lite-v1:0`. For Nova 2 Lite, set `BEDROCK_NOVA_LITE_MODEL_ID=us.amazon.nova-2-lite-v1:0`.

3. **Qwen2.5-VL** isn't a ready-made Bedrock model; it is added with *Custom Model Import*:
   1. Download the model files from Hugging Face (for example `Qwen/Qwen2.5-VL-7B-Instruct`) and upload the folder to an S3 bucket in the same region.
   2. In the Bedrock console, open *Imported models*, choose *Import model*, and point it at that S3 folder.
   3. When the import finishes, copy the model's ARN into `.env`:

      ```bash
      BEDROCK_QWEN_VL_MODEL_ARN=arn:aws:bedrock:us-east-1:123456789012:imported-model/abcd1234efgh
      ```

   AWS charges imported models for the time a copy is active, in 5-minute windows. After some idle time the model goes to sleep, so the first question can take a few minutes; the app waits up to `BEDROCK_COLD_START_WAIT` seconds.

4. **Check it.** Run `docker compose up -d`, then:

   ```bash
   curl localhost:8000/api/aws/check
   ```

   Each model is asked what colour a small red square is. Both should answer "Red".

## Settings

All settings live in `.env`; [.env.example](.env.example) lists every one with a comment. The ones you're most likely to change:

| Setting | Default | What it does |
|---|---|---|
| `APP_HOST` | `127.0.0.1` | `0.0.0.0` lets other devices on your network open the app. There is no login, so only do this on a network you trust. |
| `APP_PORT` | `8000` | Port of the web page and API. |
| `DEFAULT_EXTRACTOR` | `local` | Who reads charts when an upload doesn't say: `local`, `aws` or `none`. |
| `VISION_LOCAL_MODEL` | `qwen2.5vl:latest` | The Ollama model that reads charts. |
| `CHAT_MAX_SOURCES` | `8` | How many search results the model reads per answer. Fewer means faster answers but less to go on. |
| `CHAT_MAX_IMAGES` | `2` | Chart images sent along to models that can read images. Set `0` for faster answers. |
| `MAX_UPLOAD_MB` | `100` | Largest PDF accepted. |
| `EMBED_THREADS` | `0` | CPU threads for the search model; `0` means up to 4. |
| `OLLAMA_KEEP_ALIVE` | `10m` | How long a model stays in memory after its last use. |

## Troubleshooting

**Answers are slow.** Most of the wait before the first word is the model reading the sources (and chart images). With the 7B Qwen half on the CPU, that is about 30–60 seconds.
- Use a smaller model, such as `qwen2.5vl:3b` or a small text model.
- Lower `CHAT_MAX_SOURCES`, or set `CHAT_MAX_IMAGES=0`.
- Close other programs that use Ollama: each time something else loads a model, the app has to load its own again.

**"Waiting for the GPU".** A PDF is being read by the local vision model. Your question gets the GPU between two charts.

**A banner says Ollama can't be reached.** Start Ollama (`sudo systemctl start ollama`), then reload the page.

**A banner says the search model isn't working.** Its files are part of the app image, so rebuild it: `docker compose up -d --build`. Documents that were read meanwhile show *Not searchable yet*. They are indexed again when the app starts.

**A document is stuck.** Processing continues after a restart from where it stopped: run `docker compose restart app`. The document's details show any error.

**Charts weren't read.** The document shows a warning with the reason (for example, the vision model isn't installed). Fix it, then choose *Read charts again* in the document's details.

**Something else.** Check the log with `docker compose logs --tail 100 app`. The API is documented at http://localhost:8000/docs, and http://localhost:8000/api/health shows the state of each part.

## Data, backups and privacy

Everything the app stores is in `./data`: the uploaded PDFs, page and chart images, the SQLite database (documents, chats, keyword index) and the ChromaDB vectors. It isn't committed to git.

- **Back up:** run `docker compose stop`, copy the `data` folder, then `docker compose up -d`.
- **Start fresh:** stop the app and delete the `data` folder (keep `data/.gitkeep`).

With local models, nothing leaves your machine. When you choose Nova Lite or Qwen on AWS, the chart images or the question with its sources are sent to AWS Bedrock in your account.

## For developers

```bash
docker compose run --rm --build test                      # all tests (about 3 minutes)
docker compose run --rm test pytest tests/test_search.py  # one file
```

The test image includes headless Chromium for the browser tests of the web UI (the app image doesn't). Set `SCREENSHOT_DIR` and mount a folder to also save screenshots:

```bash
docker compose run --rm -v "$PWD/shots:/shots" -e SCREENSHOT_DIR=/shots test pytest tests/test_ui.py tests/test_ui_chat.py
```

Layout:

- `app/ingest/`: reading PDFs (`pdf.py`), the vision step (`vision.py`, `describe.py`), chunking and indexing (`chunker.py`, `index.py`) and the background worker (`jobs.py`).
- `app/llm/`: clients for Ollama and Bedrock, and the built-in search model (`embedder.py`).
- `scripts/fetch_search_model.py`: downloads the search model into the image at build time, pinned and checksummed.
- `app/search.py`, `app/answer.py`, `app/checks.py`: hybrid search, answering and the number check.
- `app/store/`: SQLite (with numbered migrations) and ChromaDB.
- `app/api/`: the HTTP endpoints.
- `app/static/`: the web page, plain HTML, CSS and JavaScript modules with no build step.
- `tests/`: fakes for Ollama and AWS, PDF builders, and the tests.

**Licenses:** the app's own code has no license file yet. Its dependencies are open source. PyMuPDF and pymupdf4llm are **AGPL-3.0**: fine for your own use, but if you offer the app to other people over a network, you must make its source code available to them or buy a commercial PyMuPDF license. marked (MIT) and DOMPurify (Apache-2.0/MPL-2.0) are included in `app/static/vendor/` with their licenses. The search model, nomic-embed-text v1.5, is Apache-2.0 and runs on onnxruntime (MIT).
