import pytest

from app.ingest.jobs import IngestQueue, document_dir
from app.store import chunks as chunk_store
from app.store import documents
from app.store.db import Database
from app.store.vectors import VectorStore
from tests.fake_embedder import FakeEmbedder, fake_vector
from tests.pdf_factory import build_sample_pdf

pytestmark = pytest.mark.anyio


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "app.db")
    database.migrate()
    return database


@pytest.fixture
def vectors(tmp_path):
    return VectorStore(tmp_path / "chroma")


def add_sample(db, data_dir, status=documents.QUEUED):
    doc_id = documents.new_id()
    folder = document_dir(data_dir, doc_id)
    folder.mkdir(parents=True)
    build_sample_pdf(folder / "original.pdf")
    documents.create(db, doc_id, "annual-report.pdf", doc_id, 1000, 5, "none")
    if status != documents.QUEUED:
        documents.set_status(db, doc_id, status)
    return doc_id


def queue(db, data_dir, vectors, embedder: FakeEmbedder | None = None) -> IngestQueue:
    return IngestQueue(db, data_dir, embedder=embedder or FakeEmbedder(), vectors=vectors)


async def test_document_becomes_ready_with_chunks_and_vectors(db, vectors, tmp_path):
    embedder = FakeEmbedder()
    doc_id = add_sample(db, tmp_path)

    await queue(db, tmp_path, vectors, embedder).process(doc_id)

    doc = documents.get(db, doc_id)
    assert doc["status"] == "ready"
    assert doc["embedded_with"] == "fake-search-model"
    counts = chunk_store.counts(db, doc_id)
    assert counts["table"] == 1 and counts["figure"] == 2 and counts["text"] >= 3
    assert vectors.count() == sum(counts.values())

    # Chunks are embedded with a header saying where they come from.
    first_input = embedder.inputs[0][0]
    assert first_input.startswith("search_document: annual-report › Annual Report 2025\n")


async def test_meaning_search_and_keyword_search_both_find_the_table(db, vectors, tmp_path):
    doc_id = add_sample(db, tmp_path)
    await queue(db, tmp_path, vectors).process(doc_id)
    table = chunk_store.for_document(db, doc_id, kind="table")[0]

    hits = vectors.query(fake_vector("Retail Wholesale Online segment results"), limit=3)
    assert table["id"] in [h.chunk_id for h in hits]
    assert hits[0].metadata["document_id"] == doc_id

    with db.session() as conn:
        rows = conn.execute(
            "SELECT c.id FROM chunks_fts f JOIN chunks c ON c.seq = f.rowid WHERE chunks_fts MATCH 'wholesale'"
        ).fetchall()
    assert [row["id"] for row in rows] == [table["id"]]


async def test_indexing_twice_replaces_the_chunks(db, vectors, tmp_path):
    doc_id = add_sample(db, tmp_path)
    work = queue(db, tmp_path, vectors)
    await work.process(doc_id)
    first = {c["id"] for c in chunk_store.for_document(db, doc_id)}

    documents.set_status(db, doc_id, documents.INDEXING)
    await work.process(doc_id)

    second = {c["id"] for c in chunk_store.for_document(db, doc_id)}
    assert len(second) == len(first)
    assert first.isdisjoint(second)
    assert vectors.count() == len(second)


async def test_document_read_before_indexing_existed_is_indexed_on_start(db, vectors, tmp_path):
    doc_id = add_sample(db, tmp_path)
    await IngestQueue(db, tmp_path).process(doc_id)  # no embedder: stops at "extracted"
    assert documents.get(db, doc_id)["status"] == "extracted"

    work = queue(db, tmp_path, vectors)
    work.start()
    await work.wait_until_idle()
    await work.stop()

    assert documents.get(db, doc_id)["status"] == "ready"
    assert vectors.count() > 0


async def test_a_broken_search_model_keeps_the_text_and_explains(db, vectors, tmp_path):
    doc_id = add_sample(db, tmp_path)

    await queue(db, tmp_path, vectors, FakeEmbedder(problem="The search model's files are missing")).process(doc_id)

    doc = documents.get(db, doc_id)
    assert doc["status"] == "extracted"
    assert doc["warning"] == "Not searchable yet: The search model's files are missing"
    assert len(documents.pages(db, doc_id)) == 5
    assert vectors.count() == 0


async def test_indexing_again_replaces_an_earlier_indexing_problem(db, vectors, tmp_path):
    doc_id = add_sample(db, tmp_path)
    await IngestQueue(db, tmp_path).process(doc_id)  # no embedder: reads the PDF, stops at "extracted"
    documents.set_status(db, doc_id, documents.INDEXING, warning="2 charts couldn't be read. Not searchable yet: down")

    await queue(db, tmp_path, vectors, FakeEmbedder(problem="still down")).process(doc_id)
    assert documents.get(db, doc_id)["warning"] == "2 charts couldn't be read. Not searchable yet: still down"

    documents.set_status(db, doc_id, documents.INDEXING, warning=documents.get(db, doc_id)["warning"])
    await queue(db, tmp_path, vectors).process(doc_id)
    doc = documents.get(db, doc_id)
    assert doc["status"] == "ready"
    assert doc["warning"] == "2 charts couldn't be read."


async def test_documents_indexed_by_another_search_model_are_indexed_again_on_start(db, vectors, tmp_path):
    old = add_sample(db, tmp_path)
    await queue(db, tmp_path, vectors, FakeEmbedder(model="old-model")).process(old)
    current = add_sample(db, tmp_path)
    await queue(db, tmp_path, vectors).process(current)
    before = {c["id"] for c in chunk_store.for_document(db, current)}

    embedder = FakeEmbedder()
    work = queue(db, tmp_path, vectors, embedder)
    work.start()
    assert documents.get(db, old)["status"] == "indexing"
    assert documents.get(db, old)["progress_note"] == "Updating the search index"
    await work.wait_until_idle()
    await work.stop()

    assert documents.get(db, old)["status"] == "ready"
    assert documents.get(db, old)["embedded_with"] == "fake-search-model"
    # The document that was already up to date is left alone.
    assert {c["id"] for c in chunk_store.for_document(db, current)} == before
    assert sum(map(len, embedder.inputs)) == len(chunk_store.for_document(db, old))


async def test_failed_reindex_leaves_the_old_index_in_place(db, vectors, tmp_path):
    embedder = FakeEmbedder()
    doc_id = add_sample(db, tmp_path)
    work = queue(db, tmp_path, vectors, embedder)
    await work.process(doc_id)
    before = vectors.count()

    embedder.broken = "The search model failed"
    documents.set_status(db, doc_id, documents.INDEXING)
    await work.process(doc_id)

    assert documents.get(db, doc_id)["status"] == "extracted"
    assert vectors.count() == before
    assert len(chunk_store.for_document(db, doc_id)) == before
