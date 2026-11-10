# PDF RAG

Ask questions about your PDFs, including their images and financial charts, and get answers with page citations.

- **Chat models:** any model installed in your local Ollama, or Qwen2.5-VL on AWS Bedrock.
- **Reading charts and images in PDFs:** Qwen2.5-VL running locally in Ollama, or Amazon Nova Lite on AWS Bedrock.
- **One server:** FastAPI serves both the API and the web page. Everything runs in Docker.

The build is split into small phases; see [docs/PLAN.md](docs/PLAN.md). Current status: **Phase 7 of 13** (chunking and indexing).

## Quick start

Needs Docker with Compose, and Ollama running on this machine with the embedding model installed:

```bash
ollama pull nomic-embed-text
```

```bash
cp .env.example .env          # then edit if needed
docker compose up --build     # starts the app
```

Open http://localhost:8000. The health check is at http://localhost:8000/api/health.

## Run the tests

```bash
docker compose run --rm --build test
```

## Where data lives

Everything the app stores (uploaded PDFs, page images, databases) goes in `./data`, which is not committed to git.
